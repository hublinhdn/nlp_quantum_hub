#!/usr/bin/env python3
"""Smoke test cho quantum training pipeline.

Verify end-to-end: load diagrams → ansatz → circuits → PennyLaneModel → train.
Chạy nhỏ (100 train, 20 dev, 5 epoch) ~5-15 phút CPU. Mục đích DUY NHẤT:
phát hiện sớm bug trước khi đốt 30h compute cho full grid.

Cách dùng:
    python scripts/06a_quantum_smoke.py
    python scripts/06a_quantum_smoke.py --reader cups
    python scripts/06a_quantum_smoke.py --ansatz sim14 --n-layers 2
    python scripts/06a_quantum_smoke.py --n-train 50 --n-epochs 3   # nhanh hơn
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--reader", default="spiders", choices=["spiders", "cups"])
    p.add_argument("--ansatz", default="iqp", choices=["iqp", "sim14"])
    p.add_argument("--n-train", type=int, default=100, help="(default 100)")
    p.add_argument("--n-dev", type=int, default=20, help="(default 20)")
    p.add_argument("--n-layers", type=int, default=1, help="ansatz layers (default 1)")
    p.add_argument("--n-epochs", type=int, default=5, help="(default 5)")
    p.add_argument("--lr", type=float, default=0.05, help="learning rate (default 0.05)")
    p.add_argument("--seed", type=int, default=0, help="(default 0)")
    p.add_argument("--n-qubits-n", type=int, default=1, help="qubits cho AtomicType NOUN")
    p.add_argument("--n-qubits-s", type=int, default=1, help="qubits cho AtomicType SENTENCE")
    return p.parse_args()


def load_split(reader: str, split: str) -> list[dict]:
    path = Path(f"data/processed/diagrams_{reader}_{split}.pkl")
    if not path.is_file():
        sys.exit(
            f"[lỗi] Không tìm thấy {path}.\n"
            f"       Chạy trước: python scripts/05_parse_diagrams.py --reader {reader}"
        )
    with path.open("rb") as f:
        return pickle.load(f)


def to_onehot(labels: np.ndarray, n_classes: int = 2) -> np.ndarray:
    arr = np.zeros((len(labels), n_classes), dtype=np.float32)
    arr[np.arange(len(labels)), labels] = 1.0
    return arr


def make_ansatz(ansatz_name: str, n_layers: int, n_qubits_n: int, n_qubits_s: int):
    """Tạo ansatz theo tên. Try defensive imports."""
    from lambeq import AtomicType

    dims = {AtomicType.NOUN: n_qubits_n, AtomicType.SENTENCE: n_qubits_s}

    if ansatz_name == "iqp":
        from lambeq import IQPAnsatz

        return IQPAnsatz(dims, n_layers=n_layers)
    elif ansatz_name == "sim14":
        # lambeq 0.5: có thể là Sim14Ansatz hoặc Sim15Ansatz
        try:
            from lambeq import Sim14Ansatz

            return Sim14Ansatz(dims, n_layers=n_layers)
        except ImportError:
            # Fallback: thử các tên khác
            import lambeq

            for name in ("Sim14Ansatz", "Sim15Ansatz", "SimAnsatz"):
                cls = getattr(lambeq, name, None)
                if cls is not None and isinstance(cls, type):
                    return cls(dims, n_layers=n_layers)
            raise ImportError(
                "Không tìm được Sim14/15Ansatz. "
                "Diagnostic: python -c 'import lambeq; print([x for x in dir(lambeq) if \"nsatz\" in x])'"
            )
    else:
        raise ValueError(f"Unknown ansatz: {ansatz_name}")


def main() -> None:
    args = parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 70)
    print(f"  Quantum smoke test")
    print(f"  reader={args.reader}  ansatz={args.ansatz}  "
          f"n_layers={args.n_layers}  seed={args.seed}")
    print(f"  n_train={args.n_train}  n_dev={args.n_dev}  n_epochs={args.n_epochs}")
    print("=" * 70)

    # === Step 1: Load diagrams ===
    print(f"\n[1/5] Load diagrams")
    train_data = load_split(args.reader, "train")[: args.n_train]
    dev_data = load_split(args.reader, "dev")[: args.n_dev]
    print(f"  train: {len(train_data)}  dev: {len(dev_data)}")

    train_diagrams = [r["diagram"] for r in train_data]
    train_labels = np.array([r["label"] for r in train_data])
    dev_diagrams = [r["diagram"] for r in dev_data]
    dev_labels = np.array([r["label"] for r in dev_data])

    # === Step 2: Create ansatz ===
    print(f"\n[2/5] Create ansatz '{args.ansatz}' (n_layers={args.n_layers})")
    ansatz = make_ansatz(
        args.ansatz, args.n_layers, args.n_qubits_n, args.n_qubits_s
    )
    print(f"  ✓ {type(ansatz).__name__}")

    # === Step 3: Convert diagrams → circuits ===
    print(f"\n[3/5] Convert diagrams → quantum circuits")
    t0 = time.time()
    train_circuits = [ansatz(d) for d in train_diagrams]
    dev_circuits = [ansatz(d) for d in dev_diagrams]
    elapsed = time.time() - t0
    print(f"  ✓ {len(train_circuits)} train + {len(dev_circuits)} dev "
          f"circuits trong {elapsed:.1f}s")

    # === Step 4: Create PennyLaneModel ===
    print(f"\n[4/5] Create PennyLaneModel")
    from lambeq import PennyLaneModel

    model = PennyLaneModel.from_diagrams(train_circuits + dev_circuits)
    model.initialise_weights()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ {n_params:,} trainable params")

    # === Step 5: Training loop ===
    print(f"\n[5/5] Train {args.n_epochs} epochs (lr={args.lr})")

    train_y = torch.tensor(to_onehot(train_labels))
    dev_y_np = dev_labels  # for accuracy comparison

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.n_epochs + 1):
        t0 = time.time()

        # Forward (full batch — smoke test data nhỏ)
        outputs = model(train_circuits)
        # Đảm bảo là tensor
        if not isinstance(outputs, torch.Tensor):
            outputs = torch.tensor(np.asarray(outputs))

        # Cross-entropy với softmax
        # outputs shape (N, 2) chứa probabilities (sum to 1)
        eps = 1e-9
        loss = -((train_y * torch.log(outputs + eps)).sum(dim=1)).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Eval
        with torch.no_grad():
            dev_outputs = model(dev_circuits)
            if not isinstance(dev_outputs, torch.Tensor):
                dev_outputs = torch.tensor(np.asarray(dev_outputs))
            dev_preds = dev_outputs.argmax(dim=1).cpu().numpy()
            dev_acc = float((dev_preds == dev_y_np).mean())

            train_preds = outputs.argmax(dim=1).cpu().numpy()
            train_acc = float((train_preds == train_labels).mean())

        elapsed = time.time() - t0
        print(
            f"  epoch {epoch:>3d}  loss={loss.item():.4f}  "
            f"train_acc={train_acc:.4f}  dev_acc={dev_acc:.4f}  ({elapsed:.1f}s)"
        )

    print(f"\n[done] Smoke test PASS. Pipeline lambeq→PennyLane→PyTorch OK.")
    print(f"       Sẵn sàng chạy full grid với scripts/06_train_quantum.py")


if __name__ == "__main__":
    main()
