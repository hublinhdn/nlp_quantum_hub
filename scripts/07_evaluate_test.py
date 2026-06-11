#!/usr/bin/env python3
"""Phase 6 — Evaluate trên TEST SET (lần đầu chạm test).

Quy trình:
    1. Load tất cả best checkpoint từ results/quantum/{reader}/{ansatz}/n_layers_{N}/seed_{S}/
    2. Tái tạo model + load weights
    3. Eval trên qnlp_test.csv (KHÔNG dùng dev/train)
    4. Save predictions + metrics per config + aggregated

Cũng evaluate classical baselines trên test (LR + TF-IDF, LR + GloVe, BiLSTM).

Output:
    results/final/
        test_metrics_quantum.json
        test_metrics_classical.json
        predictions/
            {model_name}/seed_{S}/predictions.csv  (text, true_label, pred, prob)
        summary_test.txt
        summary_test.json

Cách dùng:
    python scripts/07_evaluate_test.py
    python scripts/07_evaluate_test.py --quantum-only
    python scripts/07_evaluate_test.py --classical-only
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch

from src.baselines.common import compute_metrics, load_qnlp  # noqa: E402


DEFAULT_QUANTUM_DIR = Path("results/quantum")
DEFAULT_BASELINE_DIR = Path("results/baseline")
DEFAULT_OUT = Path("results/final")

ALL_READERS = ("spiders", "cups")
ALL_ANSATZES = ("iqp", "sim14")
ALL_LAYERS = (1, 2, 3)
ALL_SEEDS = (0, 1, 2, 3, 4)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--quantum-dir", type=Path, default=DEFAULT_QUANTUM_DIR)
    p.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    p.add_argument("--quantum-only", action="store_true")
    p.add_argument("--classical-only", action="store_true")
    p.add_argument(
        "--readers", nargs="+", choices=ALL_READERS, default=list(ALL_READERS),
    )
    p.add_argument(
        "--ansatzes", nargs="+", choices=ALL_ANSATZES, default=list(ALL_ANSATZES),
    )
    p.add_argument("--n-layers", nargs="+", type=int, default=list(ALL_LAYERS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(ALL_SEEDS))
    return p.parse_args()


# ---------------------------------------------------------------------------
# Quantum eval
# ---------------------------------------------------------------------------


def load_quantum_checkpoint(run_dir: Path) -> tuple[dict, dict] | None:
    """Load metrics.json + checkpoint.pt cho 1 run.

    Trả về (metrics, state_dict) hoặc None nếu thiếu file.
    """
    mp = run_dir / "metrics.json"
    cp = run_dir / "checkpoint.pt"
    if not mp.is_file() or not cp.is_file():
        return None
    with mp.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    state_dict = torch.load(cp, map_location="cpu", weights_only=True)
    return metrics, state_dict


def evaluate_quantum_test(args) -> list[dict]:
    """Iterate tất cả quantum checkpoints, eval trên test set.

    Cần construct PennyLaneModel ĐÚNG y hệt như lúc training:
        - from_diagrams(train + dev) — KHÔNG include test
        - initialise_weights() — bắt buộc để tạo ParameterList structure
        - load_state_dict(state) — keys match nhờ initialise_weights()
        - forward(test_circuits) — chỉ cần test symbols ⊆ train+dev symbols (vocab filter)
    """
    from src.quantum.ansatz import make_ansatz
    from src.quantum.data import load_split
    from lambeq import PennyLaneModel

    print("\n" + "=" * 70)
    print("  Quantum models — TEST evaluation")
    print("=" * 70)

    # Caches per reader: train/dev/test diagrams + texts/labels
    train_dev_diagrams_cache: dict[str, dict] = {}
    test_data_cache: dict[str, list[dict]] = {}
    # Circuits cache per (reader, ansatz, n_layers) để tránh re-convert 5 seeds
    circuits_cache: dict[tuple, dict] = {}

    results: list[dict] = []

    configs = list(product(args.readers, args.ansatzes, args.n_layers, args.seeds))
    print(f"\nTotal configs: {len(configs)}")

    for reader, ansatz_name, n_layers, seed in configs:
        run_dir = (
            args.quantum_dir / reader / ansatz_name
            / f"n_layers_{n_layers}" / f"seed_{seed}"
        )
        loaded = load_quantum_checkpoint(run_dir)
        if loaded is None:
            print(f"  [skip] {reader}/{ansatz_name}/L{n_layers}/seed{seed} — chưa có checkpoint")
            continue
        metrics, state_dict = loaded

        # Load diagrams (cache theo reader)
        if reader not in train_dev_diagrams_cache:
            print(f"  [load diagrams] reader={reader}")
            train_data = load_split(reader, "train")
            dev_data = load_split(reader, "dev")
            test_data = load_split(reader, "test")
            train_dev_diagrams_cache[reader] = {
                "train": [r["diagram"] for r in train_data],
                "dev": [r["diagram"] for r in dev_data],
                "test": [r["diagram"] for r in test_data],
            }
            test_data_cache[reader] = test_data

        test_records = test_data_cache[reader]
        test_diagrams = train_dev_diagrams_cache[reader]["test"]
        test_labels = np.array([r["label"] for r in test_records])
        test_texts = [r["text"] for r in test_records]

        # Convert circuits (cache per ansatz config)
        ckey = (reader, ansatz_name, n_layers)
        if ckey not in circuits_cache:
            print(f"  [ansatz] {ansatz_name} L{n_layers} cho {reader}")
            ansatz = make_ansatz(ansatz_name, n_layers)
            circuits_cache[ckey] = {
                "train": [ansatz(d) for d in train_dev_diagrams_cache[reader]["train"]],
                "dev": [ansatz(d) for d in train_dev_diagrams_cache[reader]["dev"]],
                "test": [ansatz(d) for d in test_diagrams],
            }
        cc = circuits_cache[ckey]

        # Build model với train + dev + test.
        # Sau khi rebuild qnlp dataset (02b với word coverage constraint),
        # test symbols ⊆ train+dev symbols → sorted symbol set không đổi → load OK.
        # Safety net: vẫn dùng strict=False + report nếu bất ngờ mismatch.
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = PennyLaneModel.from_diagrams(cc["train"] + cc["dev"] + cc["test"])
        model.initialise_weights()
        load_result = model.load_state_dict(state_dict, strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            print(
                f"  ! {reader}/{ansatz_name}/L{n_layers}/seed{seed} load warning: "
                f"missing={len(load_result.missing_keys)} "
                f"unexpected={len(load_result.unexpected_keys)} "
                f"(KHÔNG NÊN xảy ra sau rebuild dataset — debug 02b)"
            )
        model.eval()

        try:
            with torch.no_grad():
                outputs = model(cc["test"])
                if not isinstance(outputs, torch.Tensor):
                    outputs = torch.tensor(np.asarray(outputs))
                probs = outputs.cpu().numpy()
                preds = probs.argmax(axis=1)
                scores = probs[:, 1]
        except Exception as e:
            print(f"  ✗ {reader}/{ansatz_name}/L{n_layers}/seed{seed}  FORWARD FAIL: "
                  f"{type(e).__name__}: {str(e)[:100]}")
            continue

        test_metrics = compute_metrics(test_labels, preds, scores)

        # Save predictions
        pred_path = (
            args.out_dir / "predictions"
            / f"quantum_{reader}_{ansatz_name}_L{n_layers}_seed{seed}"
            / "predictions.csv"
        )
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "text": test_texts,
                "true_label": test_labels,
                "pred": preds,
                "prob_pos": scores,
            }
        ).to_csv(pred_path, index=False)

        entry = {
            "model": f"quantum_{reader}_{ansatz_name}_L{n_layers}_seed{seed}",
            "reader": reader,
            "ansatz": ansatz_name,
            "n_layers": n_layers,
            "seed": seed,
            "n_params": metrics.get("n_params", 0),
            "test_metrics": test_metrics.to_dict(),
        }
        results.append(entry)
        print(
            f"  ✓ {reader}/{ansatz_name}/L{n_layers}/seed{seed}  "
            f"test_acc={test_metrics.accuracy:.4f}  "
            f"params={metrics.get('n_params', 0):,}"
        )

    return results


# ---------------------------------------------------------------------------
# Classical eval
# ---------------------------------------------------------------------------


def evaluate_classical_test(args) -> list[dict]:
    """Re-train classical baselines trên train+dev, eval trên test."""
    print("\n" + "=" * 70)
    print("  Classical baselines — TEST evaluation")
    print("=" * 70)

    splits = load_qnlp(args.processed_dir)
    train = splits["train"]
    dev = splits["dev"]
    test = splits["test"]
    print(f"\ntrain={len(train)}  dev={len(dev)}  test={len(test)}")

    results: list[dict] = []

    # --- LR + TF-IDF ---
    print("\n[LR + TF-IDF] grid search on dev, refit, eval test")
    from src.baselines.lr import train_lr_tfidf
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    for seed in (0, 1, 2):
        r = train_lr_tfidf(train, dev, seed=seed)
        # Refit pipeline với best_C để eval test
        vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=False, min_df=2)
        X_train = vec.fit_transform(train.texts)
        X_test = vec.transform(test.texts)
        clf = LogisticRegression(C=r.best_C, max_iter=1000, random_state=seed, solver="liblinear")
        clf.fit(X_train, train.labels)
        preds = clf.predict(X_test)
        scores = clf.predict_proba(X_test)[:, 1]
        tm = compute_metrics(test.labels, preds, scores)
        results.append(
            {
                "model": f"lr_tfidf_seed{seed}",
                "model_family": "LR + TF-IDF",
                "seed": seed,
                "n_params": X_train.shape[1],
                "test_metrics": tm.to_dict(),
            }
        )
        pred_path = args.out_dir / "predictions" / f"lr_tfidf_seed{seed}" / "predictions.csv"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"text": test.texts, "true_label": test.labels, "pred": preds, "prob_pos": scores}
        ).to_csv(pred_path, index=False)
        print(f"  seed={seed} → test_acc={tm.accuracy:.4f}")

    # --- LR + GloVe ---
    glove_path = Path("data/raw/glove/glove.6B.50d.txt")
    if glove_path.is_file():
        print("\n[LR + GloVe-50d] eval test")
        from src.baselines.glove import encode_texts, load_glove
        from src.baselines.lr_glove import train_lr_glove

        embeddings = load_glove(glove_path)
        for seed in (0, 1, 2):
            r = train_lr_glove(train, dev, embeddings=embeddings, seed=seed)
            X_train = encode_texts(train.texts, embeddings)
            X_test = encode_texts(test.texts, embeddings)
            clf = LogisticRegression(
                C=r.best_C, max_iter=1000, random_state=seed, solver="liblinear"
            )
            clf.fit(X_train, train.labels)
            preds = clf.predict(X_test)
            scores = clf.predict_proba(X_test)[:, 1]
            tm = compute_metrics(test.labels, preds, scores)
            results.append(
                {
                    "model": f"lr_glove_seed{seed}",
                    "model_family": "LR + GloVe-50d",
                    "seed": seed,
                    "n_params": 50 + 1,
                    "test_metrics": tm.to_dict(),
                }
            )
            pred_path = args.out_dir / "predictions" / f"lr_glove_seed{seed}" / "predictions.csv"
            pred_path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {"text": test.texts, "true_label": test.labels, "pred": preds, "prob_pos": scores}
            ).to_csv(pred_path, index=False)
            print(f"  seed={seed} → test_acc={tm.accuracy:.4f}")
    else:
        print(f"\n[skip] LR + GloVe (không có {glove_path})")

    # --- BiLSTM ---
    print("\n[BiLSTM] re-train + eval test")
    from src.baselines.bilstm import (
        BiLSTMClassifier, _eval, build_token_id_map, collate,
        TextDataset, set_seed,
    )
    from torch.utils.data import DataLoader

    vocab_path = Path("data/processed/qnlp_vocab.tsv")
    token_to_id = build_token_id_map(vocab_path)
    vocab_size = len(token_to_id)

    for seed in (0, 1, 2):
        set_seed(seed)
        # Train trên train + dev gộp lại để evaluation final
        # Hoặc: train trên train, eval test (simpler — không cần dev cho final)
        train_ds = TextDataset(train.texts, train.labels, token_to_id)
        dev_ds = TextDataset(dev.texts, dev.labels, token_to_id)
        test_ds = TextDataset(test.texts, test.labels, token_to_id)

        train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate,
                                   generator=torch.Generator().manual_seed(seed))
        dev_loader = DataLoader(dev_ds, batch_size=32, shuffle=False, collate_fn=collate)
        test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, collate_fn=collate)

        model = BiLSTMClassifier(vocab_size, emb_dim=32, hidden_dim=64, dropout=0.3)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = torch.nn.BCEWithLogitsLoss()

        # Train với early stop trên dev (10 epoch tối đa cho test eval)
        best_dev_loss = float("inf")
        best_state = None
        no_improve = 0
        for epoch in range(1, 31):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                logits = model(batch["input_ids"].to(device), batch["lengths"].to(device))
                loss = criterion(logits, batch["labels"].to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            dev_loss, _ = _eval(model, dev_loader, criterion, device)
            if dev_loss < best_dev_loss - 1e-6:
                best_dev_loss = dev_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= 5:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        # Eval test
        _, test_metrics = _eval(model, test_loader, criterion, device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        results.append(
            {
                "model": f"bilstm_seed{seed}",
                "model_family": "BiLSTM 1L",
                "seed": seed,
                "n_params": n_params,
                "test_metrics": test_metrics.to_dict(),
            }
        )
        # Predictions
        all_preds = []
        all_scores = []
        all_texts = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                logits = model(batch["input_ids"].to(device), batch["lengths"].to(device))
                p = torch.sigmoid(logits).cpu().numpy()
                all_scores.extend(p.tolist())
                all_preds.extend((p >= 0.5).astype(int).tolist())
                all_labels.extend(batch["labels"].numpy().astype(int).tolist())
        all_texts = test.texts
        pred_path = args.out_dir / "predictions" / f"bilstm_seed{seed}" / "predictions.csv"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"text": all_texts, "true_label": all_labels, "pred": all_preds, "prob_pos": all_scores}
        ).to_csv(pred_path, index=False)
        print(f"  seed={seed} → test_acc={test_metrics.accuracy:.4f}")

    return results


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def aggregate_by_group(results: list[dict], group_keys: list[str]) -> list[dict]:
    """Aggregate metrics qua các seed thuộc cùng (reader, ansatz, layers) hoặc (model_family).

    Trả về list các dict với mean/std accuracy.
    """
    groups: dict[tuple, list[dict]] = {}
    for r in results:
        key = tuple(r.get(k, None) for k in group_keys)
        groups.setdefault(key, []).append(r)
    agg = []
    for key, members in groups.items():
        accs = [m["test_metrics"]["accuracy"] for m in members]
        f1s = [m["test_metrics"]["f1_macro"] for m in members]
        params = [m.get("n_params", 0) for m in members]
        entry = dict(zip(group_keys, key))
        entry["n_seeds"] = len(members)
        entry["acc_mean"] = float(np.mean(accs))
        entry["acc_std"] = float(np.std(accs, ddof=0))
        entry["f1m_mean"] = float(np.mean(f1s))
        entry["f1m_std"] = float(np.std(f1s, ddof=0))
        entry["n_params"] = int(np.mean(params))
        agg.append(entry)
    return agg


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  Phase 6 — Final TEST set evaluation")
    print("=" * 70)
    print("⚠️  Đây là LẦN ĐẦU chạm test set. Không tinh chỉnh hyperparam dựa vào kết quả này.")

    classical_results: list[dict] = []
    quantum_results: list[dict] = []

    if not args.quantum_only:
        classical_results = evaluate_classical_test(args)
        with (args.out_dir / "test_metrics_classical.json").open("w", encoding="utf-8") as f:
            json.dump(classical_results, f, indent=2, ensure_ascii=False)

    if not args.classical_only:
        quantum_results = evaluate_quantum_test(args)
        with (args.out_dir / "test_metrics_quantum.json").open("w", encoding="utf-8") as f:
            json.dump(quantum_results, f, indent=2, ensure_ascii=False)

    # Aggregate
    print("\n" + "=" * 70)
    print("  FINAL TEST RESULTS (mean ± std qua seeds)")
    print("=" * 70)
    print(f"  Majority baseline: 0.5000  (balanced 50:50)")
    print("-" * 70)

    summary_rows: list[dict] = []

    # Classical: aggregate by model_family
    if classical_results:
        for entry in aggregate_by_group(classical_results, ["model_family"]):
            line = (
                f"  {entry['model_family']:<22} "
                f"acc={entry['acc_mean']:.4f}±{entry['acc_std']:.4f}  "
                f"f1m={entry['f1m_mean']:.4f}±{entry['f1m_std']:.4f}  "
                f"params={entry['n_params']:,}  seeds={entry['n_seeds']}"
            )
            print(line)
            summary_rows.append({"group": entry["model_family"], **entry})

    # Quantum: aggregate by (reader, ansatz, n_layers)
    if quantum_results:
        for entry in aggregate_by_group(
            quantum_results, ["reader", "ansatz", "n_layers"]
        ):
            label = f"Q-{entry['reader']}-{entry['ansatz']}-L{entry['n_layers']}"
            line = (
                f"  {label:<22} "
                f"acc={entry['acc_mean']:.4f}±{entry['acc_std']:.4f}  "
                f"f1m={entry['f1m_mean']:.4f}±{entry['f1m_std']:.4f}  "
                f"params={entry['n_params']:,}  seeds={entry['n_seeds']}"
            )
            print(line)
            summary_rows.append({"group": label, **entry})

    print("=" * 70)

    # Save summary
    with (args.out_dir / "summary_test.json").open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)
    with (args.out_dir / "summary_test.txt").open("w", encoding="utf-8") as f:
        f.write("Phase 6 — TEST set final results\n")
        f.write("=" * 70 + "\n")
        f.write("Majority baseline: 0.5000  (balanced 50:50)\n")
        f.write("-" * 70 + "\n")
        for entry in summary_rows:
            f.write(
                f"{entry['group']:<25} acc={entry['acc_mean']:.4f}±{entry['acc_std']:.4f}  "
                f"f1m={entry['f1m_mean']:.4f}±{entry['f1m_std']:.4f}  "
                f"params={entry['n_params']:,}  seeds={entry['n_seeds']}\n"
            )
    print(f"\n[save] {args.out_dir}/summary_test.txt")
    print(f"[save] {args.out_dir}/summary_test.json")
    print(f"[save] {args.out_dir}/predictions/")
    print(f"\n[next] python scripts/08_final_analysis.py")


if __name__ == "__main__":
    main()
