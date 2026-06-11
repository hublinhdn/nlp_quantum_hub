#!/usr/bin/env python3
"""Interactive demo cho buổi bảo vệ luận văn.

Nhập 1 phrase (2-5 từ) → in ra prediction của 5 model song song:
    1. LR + TF-IDF
    2. LR + GloVe-50d (nếu có)
    3. BiLSTM 1-layer
    4. Quantum Spiders + IQP
    5. Quantum Cups + IQP (nếu có checkpoint)

Cách dùng:
    python scripts/99_demo.py "not one clever line"
    python scripts/99_demo.py "an extraordinary film"
    python scripts/99_demo.py            # interactive mode — gõ phrase, Ctrl-D thoát
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("phrase", nargs="?", default=None, help="Phrase cần predict (để trống = interactive)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quantum-reader", default="spiders", choices=("spiders", "cups"))
    p.add_argument("--quantum-ansatz", default="iqp", choices=("iqp", "sim14"))
    p.add_argument("--quantum-layers", type=int, default=1)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------


class LRTfidfModel:
    name = "LR + TF-IDF"

    def __init__(self, seed=0):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from src.baselines.common import load_qnlp

        print(f"  [{self.name}] training...", end=" ", flush=True)
        splits = load_qnlp()
        self.vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=False, min_df=2)
        X_train = self.vec.fit_transform(splits["train"].texts)
        self.clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed, solver="liblinear")
        self.clf.fit(X_train, splits["train"].labels)
        print("done")

    def predict(self, text: str) -> tuple[int, float]:
        X = self.vec.transform([text.lower()])
        pred = int(self.clf.predict(X)[0])
        prob = float(self.clf.predict_proba(X)[0, 1])
        return pred, prob


class LRGloveModel:
    name = "LR + GloVe-50d"

    def __init__(self, seed=0):
        from sklearn.linear_model import LogisticRegression
        from src.baselines.common import load_qnlp
        from src.baselines.glove import DEFAULT_GLOVE_PATH, encode_texts, load_glove

        if not DEFAULT_GLOVE_PATH.is_file():
            raise FileNotFoundError(f"Không có {DEFAULT_GLOVE_PATH}")
        print(f"  [{self.name}] training...", end=" ", flush=True)
        self.embeddings = load_glove()
        splits = load_qnlp()
        X_train = encode_texts(splits["train"].texts, self.embeddings)
        self.clf = LogisticRegression(C=10.0, max_iter=1000, random_state=seed, solver="liblinear")
        self.clf.fit(X_train, splits["train"].labels)
        print("done")

    def predict(self, text: str) -> tuple[int, float]:
        from src.baselines.glove import encode_texts

        X = encode_texts([text.lower()], self.embeddings)
        pred = int(self.clf.predict(X)[0])
        prob = float(self.clf.predict_proba(X)[0, 1])
        return pred, prob


class BiLSTMDemoModel:
    name = "BiLSTM 1L"

    def __init__(self, seed=0):
        from src.baselines.bilstm import (
            BiLSTMClassifier, build_token_id_map, collate,
            TextDataset, set_seed,
        )
        from src.baselines.common import load_qnlp
        from torch.utils.data import DataLoader

        print(f"  [{self.name}] training...", end=" ", flush=True)
        set_seed(seed)
        self.token_to_id = build_token_id_map(Path("data/processed/qnlp_vocab.tsv"))
        splits = load_qnlp()
        train_ds = TextDataset(splits["train"].texts, splits["train"].labels, self.token_to_id)
        train_loader = DataLoader(
            train_ds, batch_size=32, shuffle=True, collate_fn=collate,
            generator=torch.Generator().manual_seed(seed),
        )
        self.model = BiLSTMClassifier(len(self.token_to_id), emb_dim=32, hidden_dim=64, dropout=0.3)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = torch.nn.BCEWithLogitsLoss()
        for _ in range(10):
            self.model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                logits = self.model(batch["input_ids"].to(self.device), batch["lengths"].to(self.device))
                loss = criterion(logits, batch["labels"].to(self.device))
                loss.backward()
                optimizer.step()
        self.model.eval()
        print("done")
        # Cache collate function for prediction
        from src.baselines.bilstm import encode_text
        self.encode = encode_text

    def predict(self, text: str) -> tuple[int, float]:
        ids = self.encode(text.lower(), self.token_to_id)
        if not ids:
            return 0, 0.5
        input_ids = torch.tensor([ids], dtype=torch.long).to(self.device)
        lengths = torch.tensor([len(ids)], dtype=torch.long)
        with torch.no_grad():
            logit = self.model(input_ids, lengths).item()
        prob = torch.sigmoid(torch.tensor(logit)).item()
        pred = int(prob >= 0.5)
        return pred, prob


class QuantumDemoModel:
    def __init__(self, reader="spiders", ansatz="iqp", n_layers=1, seed=0):
        self.name = f"Quantum-{reader}-{ansatz}-L{n_layers}"
        self.reader_name = reader
        self.ansatz_name = ansatz
        self.n_layers = n_layers

        print(f"  [{self.name}] loading checkpoint...", end=" ", flush=True)

        from lambeq import PennyLaneModel
        from src.discocat.parse import make_parser
        from src.quantum.ansatz import make_ansatz
        from src.quantum.data import load_split

        ckpt_path = self._find_best_checkpoint(reader, ansatz, n_layers)
        if ckpt_path is None:
            raise FileNotFoundError(
                f"Không tìm thấy checkpoint cho quantum {reader}/{ansatz}/L{n_layers}"
            )

        self.ckpt_path = ckpt_path
        self.state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        # Cache base diagrams + circuits (train + dev + test) — same as training
        train_data = load_split(reader, "train")
        dev_data = load_split(reader, "dev")
        test_data = load_split(reader, "test")
        self.base_diagrams = (
            [r["diagram"] for r in train_data]
            + [r["diagram"] for r in dev_data]
            + [r["diagram"] for r in test_data]
        )

        self.ansatz = make_ansatz(ansatz, n_layers)
        self.base_circuits = [self.ansatz(d) for d in self.base_diagrams]

        # Pre-compute training symbol set (for OOV warning)
        tmp = PennyLaneModel.from_diagrams(self.base_circuits)
        self.train_symbols = set(tmp.symbols)
        del tmp

        # Parser cho phrase mới
        self.parser = make_parser(reader_type=reader)
        print("done")

    @staticmethod
    def _find_best_checkpoint(reader, ansatz, n_layers) -> Path | None:
        import json
        base = Path(f"results/quantum/{reader}/{ansatz}/n_layers_{n_layers}")
        if not base.is_dir():
            return None
        best_acc = -1.0
        best_path = None
        for seed_dir in base.glob("seed_*"):
            mp = seed_dir / "metrics.json"
            cp = seed_dir / "checkpoint.pt"
            if not (mp.is_file() and cp.is_file()):
                continue
            with mp.open("r", encoding="utf-8") as f:
                m = json.load(f)
            acc = m.get("best_dev_acc", 0.0)
            if acc > best_acc:
                best_acc = acc
                best_path = cp
        return best_path

    def predict(self, text: str) -> tuple[int, float]:
        from lambeq import PennyLaneModel

        # Parse input phrase
        diagram = self.parser.sentence2diagram(text.lower())
        circuit = self.ansatz(diagram)

        # Check OOV symbols
        new_symbols = set(circuit.free_symbols)
        oov = new_symbols - self.train_symbols
        if oov:
            print(f"\n  [warn] {self.name}: {len(oov)} OOV symbol(s) — "
                  f"prediction sẽ kém chính xác", end="")

        # Build model với input circuit included → forward được trên input mới.
        # initialise_weights() + load_state_dict(strict=False) để chắc chắn load đúng.
        model = PennyLaneModel.from_diagrams(self.base_circuits + [circuit])
        model.initialise_weights()
        model.load_state_dict(self.state_dict, strict=False)
        model.eval()

        with torch.no_grad():
            outputs = model([circuit])
            if not isinstance(outputs, torch.Tensor):
                outputs = torch.tensor(np.asarray(outputs))
            probs = outputs.cpu().numpy()[0]
        pred = int(probs.argmax())
        prob_pos = float(probs[1])
        return pred, prob_pos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def load_all_models(args) -> list:
    models = []

    print("\n[init] Loading classical models...")
    try:
        models.append(LRTfidfModel(seed=args.seed))
    except Exception as e:
        print(f"  [warn] LR+TF-IDF failed: {e}")

    try:
        models.append(LRGloveModel(seed=args.seed))
    except Exception as e:
        print(f"  [warn] LR+GloVe failed: {e}")

    try:
        models.append(BiLSTMDemoModel(seed=args.seed))
    except Exception as e:
        print(f"  [warn] BiLSTM failed: {e}")

    print("\n[init] Loading quantum models...")
    try:
        models.append(
            QuantumDemoModel(args.quantum_reader, args.quantum_ansatz, args.quantum_layers, args.seed)
        )
    except Exception as e:
        print(f"  [warn] Quantum {args.quantum_reader}-{args.quantum_ansatz}-L{args.quantum_layers}: {e}")

    # Bonus: cups quantum nếu có
    try:
        models.append(QuantumDemoModel("cups", args.quantum_ansatz, args.quantum_layers, args.seed))
    except Exception as e:
        pass  # silent

    return models


def predict_and_print(models: list, text: str) -> None:
    text_clean = text.strip()
    n_words = len(text_clean.split())
    print()
    print("─" * 60)
    print(f"  Input: \"{text_clean}\"  ({n_words} từ)")
    print("─" * 60)
    for m in models:
        try:
            pred, prob = m.predict(text_clean)
            label_str = "POS" if pred == 1 else "NEG"
            color = "✓" if prob > 0.7 or prob < 0.3 else "·"
            print(f"  {color} {m.name:<28} → {label_str}  (conf={prob:.3f})")
        except Exception as e:
            print(f"  ✗ {m.name:<28} → ERROR: {e}")
    print("─" * 60)


def main() -> None:
    args = parse_args()

    print("=" * 60)
    print("  Quantum NLP — Interactive Demo")
    print("=" * 60)

    models = load_all_models(args)
    if not models:
        sys.exit("[lỗi] Không load được model nào.")

    if args.phrase:
        predict_and_print(models, args.phrase)
        return

    # Interactive mode
    print("\n[interactive] Gõ phrase 2-5 từ. Ctrl-D hoặc 'quit' để thoát.\n")
    while True:
        try:
            line = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[bye]")
            break
        if not line or line.lower() in ("quit", "exit", "q"):
            print("[bye]")
            break
        predict_and_print(models, line)


if __name__ == "__main__":
    main()
