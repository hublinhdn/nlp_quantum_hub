#!/usr/bin/env python3
"""Phase 5 — Train quantum models trên full grid configurations.

Grid mặc định:
    reader   : spiders, cups          (2)
    ansatz   : iqp, sim14             (2)
    n_layers : 1, 2, 3                (3)
    seed     : 0, 1, 2, 3, 4          (5)
                                     ---
    Total:                           60 runs

Output:
    results/quantum/
        {reader}/{ansatz}/n_layers_{N}/seed_{S}/
            metrics.json
            curve.csv
            checkpoint.pt
        {reader}/{ansatz}/n_layers_{N}/aggregated.json   # mean ± std qua seeds
        summary.json                                       # toàn bộ grid
        summary.txt                                        # bảng tổng kết

Cách dùng:
    python scripts/06_train_quantum.py                    # full grid 60 runs
    python scripts/06_train_quantum.py --readers spiders  # chỉ spiders (30 runs)
    python scripts/06_train_quantum.py --readers spiders --ansatzes iqp --n-layers 1 --seeds 0
    python scripts/06_train_quantum.py --max-epochs 30 --max-train 1000   # debug nhanh
    python scripts/06_train_quantum.py --skip-existing                    # bỏ qua run đã xong
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.quantum.ansatz import ansatz_display_name, make_ansatz  # noqa: E402
from src.quantum.data import load_split  # noqa: E402
from src.quantum.train import train_one_config  # noqa: E402


DEFAULT_OUT = Path("results/quantum")

ALL_READERS = ("spiders", "cups")
ALL_ANSATZES = ("iqp", "sim14")
ALL_LAYERS = (1, 2, 3)
ALL_SEEDS = (0, 1, 2, 3, 4)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--readers", nargs="+", choices=ALL_READERS, default=list(ALL_READERS))
    p.add_argument("--ansatzes", nargs="+", choices=ALL_ANSATZES, default=list(ALL_ANSATZES))
    p.add_argument("--n-layers", nargs="+", type=int, default=list(ALL_LAYERS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(ALL_SEEDS))
    p.add_argument("--n-qubits-n", type=int, default=1)
    p.add_argument("--n-qubits-s", type=int, default=1)
    p.add_argument("--lr", type=float, default=0.01,
                   help="Learning rate. 0.01 (default) ổn định hơn 0.05.")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-epochs", type=int, default=200,
                   help="Max epochs. 200 (default) cho phép converge chậm với lr=0.01.")
    p.add_argument("--patience", type=int, default=20,
                   help="Early stop patience (default 20). Tăng lên để không bỏ cuộc sớm.")
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="Gradient clipping max_norm. None hoặc 0 = disabled.")
    p.add_argument("--no-scheduler", action="store_true",
                   help="Tắt ReduceLROnPlateau scheduler (default: bật)")
    p.add_argument(
        "--max-train", type=int, default=None,
        help="Giới hạn số phrase train (debug). None = full.",
    )
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--skip-existing", action="store_true",
        help="Bỏ qua run đã có metrics.json (resume)",
    )
    p.add_argument("--quiet", action="store_true", help="Bỏ in epoch log")
    return p.parse_args()


def run_dir_for(out_dir: Path, reader: str, ansatz: str, n_layers: int, seed: int) -> Path:
    return out_dir / reader / ansatz / f"n_layers_{n_layers}" / f"seed_{seed}"


def save_run(result, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    # metrics.json
    metrics = {
        "config": result.config,
        "best_epoch": result.best_epoch,
        "best_dev_loss": result.best_dev_loss,
        "best_dev_acc": result.best_dev_acc,
        "n_params": result.n_params,
        "train_time_s": result.train_time,
    }
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # curve.csv
    if result.train_curve:
        keys = list(result.train_curve[0].keys())
        with (run_dir / "curve.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(result.train_curve)

    # checkpoint
    if result.best_state_dict is not None:
        torch.save(result.best_state_dict, run_dir / "checkpoint.pt")


def aggregate_seeds(metric_list: list[dict]) -> dict:
    accs = np.array([m["best_dev_acc"] for m in metric_list])
    losses = np.array([m["best_dev_loss"] for m in metric_list])
    return {
        "n_seeds": len(metric_list),
        "best_dev_acc_mean": float(accs.mean()),
        "best_dev_acc_std": float(accs.std(ddof=0)),
        "best_dev_loss_mean": float(losses.mean()),
        "best_dev_loss_std": float(losses.std(ddof=0)),
        "n_params": metric_list[0].get("n_params", 0),
        "per_seed": metric_list,
    }


def load_diagrams_for_reader(reader: str, max_train: int | None) -> tuple:
    """Load train + dev diagrams cho 1 reader. KHÔNG load test (để dành Phase 6)."""
    train_data = load_split(reader, "train")
    dev_data = load_split(reader, "dev")
    if max_train is not None:
        train_data = train_data[:max_train]
    train_diagrams = [r["diagram"] for r in train_data]
    train_labels = np.array([r["label"] for r in train_data])
    dev_diagrams = [r["diagram"] for r in dev_data]
    dev_labels = np.array([r["label"] for r in dev_data])
    return train_diagrams, train_labels, dev_diagrams, dev_labels


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Tính tổng số run
    configs = list(product(args.readers, args.ansatzes, args.n_layers, args.seeds))
    print("=" * 70)
    print(f"  Phase 5 — Quantum training")
    print(f"  readers  : {args.readers}")
    print(f"  ansatzes : {args.ansatzes}")
    print(f"  n_layers : {args.n_layers}")
    print(f"  seeds    : {args.seeds}")
    print(f"  TOTAL    : {len(configs)} runs")
    print(f"  max_epochs={args.max_epochs}, patience={args.patience}, "
          f"batch_size={args.batch_size}, lr={args.lr}")
    print("=" * 70)

    # Cache: load diagrams + ansatz + model 1 lần cho mỗi (reader, ansatz, layers)
    # Nhưng vì model tạo từ tất cả circuits của cả train+dev, nên cần reset cho mỗi seed.
    # Để đơn giản: cache diagrams theo reader; ansatz + circuits theo (reader, ansatz, layers).

    diagrams_cache: dict[str, tuple] = {}
    circuits_cache: dict[tuple, tuple] = {}

    t_grid_start = time.time()
    run_idx = 0
    skipped = 0

    for reader, ansatz_name, n_layers, seed in configs:
        run_idx += 1
        run_dir = run_dir_for(args.out_dir, reader, ansatz_name, n_layers, seed)
        config = {
            "reader": reader,
            "ansatz": ansatz_name,
            "n_layers": n_layers,
            "seed": seed,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "n_qubits_n": args.n_qubits_n,
            "n_qubits_s": args.n_qubits_s,
        }

        # Skip nếu đã có
        if args.skip_existing and (run_dir / "metrics.json").is_file():
            print(f"\n[{run_idx}/{len(configs)}] SKIP (đã có) "
                  f"{reader}/{ansatz_name}/L{n_layers}/seed{seed}")
            skipped += 1
            continue

        print(f"\n[{run_idx}/{len(configs)}] "
              f"reader={reader}  ansatz={ansatz_name}  "
              f"n_layers={n_layers}  seed={seed}")

        # Load diagrams (cache theo reader)
        if reader not in diagrams_cache:
            print(f"  [load] diagrams cho reader={reader}")
            diagrams_cache[reader] = load_diagrams_for_reader(reader, args.max_train)
        train_diagrams, train_labels, dev_diagrams, dev_labels = diagrams_cache[reader]

        # Convert circuits (cache theo (reader, ansatz, n_layers))
        cache_key = (reader, ansatz_name, n_layers)
        if cache_key not in circuits_cache:
            print(f"  [ansatz] tạo {ansatz_display_name(ansatz_name)}Ansatz "
                  f"(n_layers={n_layers})")
            ansatz = make_ansatz(
                ansatz_name, n_layers,
                n_qubits_n=args.n_qubits_n, n_qubits_s=args.n_qubits_s,
            )
            t0 = time.time()
            train_circuits = [ansatz(d) for d in train_diagrams]
            dev_circuits = [ansatz(d) for d in dev_diagrams]
            print(f"  [circuit] {len(train_circuits)} train + {len(dev_circuits)} dev "
                  f"trong {time.time() - t0:.1f}s")
            circuits_cache[cache_key] = (train_circuits, dev_circuits)
        train_circuits, dev_circuits = circuits_cache[cache_key]

        # Tạo model — phải tạo lại cho mỗi seed (weight khởi tạo khác nhau)
        from lambeq import PennyLaneModel

        np.random.seed(seed)
        torch.manual_seed(seed)
        model = PennyLaneModel.from_diagrams(train_circuits + dev_circuits)
        model.initialise_weights()

        # Train
        result = train_one_config(
            train_circuits=train_circuits,
            train_labels=train_labels,
            dev_circuits=dev_circuits,
            dev_labels=dev_labels,
            model=model,
            config=config,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            seed=seed,
            grad_clip=args.grad_clip if args.grad_clip and args.grad_clip > 0 else None,
            use_scheduler=not args.no_scheduler,
            verbose=not args.quiet,
        )

        save_run(result, run_dir)
        print(f"  ✓ best_epoch={result.best_epoch}  "
              f"dev_acc={result.best_dev_acc:.4f}  "
              f"({result.train_time:.1f}s, {result.n_params:,} params)")
        print(f"  → {run_dir}")

    total_time = time.time() - t_grid_start
    print(f"\n[grid done] {len(configs)} runs total, {skipped} skipped, "
          f"{total_time / 60:.1f} phút.")

    # Aggregate qua seeds + summary
    print(f"\n[aggregate] tính mean ± std qua seeds...")
    summary_rows: list[dict] = []
    for reader, ansatz_name, n_layers in product(
        args.readers, args.ansatzes, args.n_layers
    ):
        per_seed_metrics: list[dict] = []
        for seed in args.seeds:
            run_dir = run_dir_for(args.out_dir, reader, ansatz_name, n_layers, seed)
            mp = run_dir / "metrics.json"
            if not mp.is_file():
                continue
            with mp.open("r", encoding="utf-8") as f:
                per_seed_metrics.append(json.load(f))
        if not per_seed_metrics:
            continue
        agg = aggregate_seeds(per_seed_metrics)
        agg_path = args.out_dir / reader / ansatz_name / f"n_layers_{n_layers}" / "aggregated.json"
        agg_path.parent.mkdir(parents=True, exist_ok=True)
        with agg_path.open("w", encoding="utf-8") as f:
            json.dump(agg, f, indent=2, ensure_ascii=False)
        summary_rows.append(
            {
                "reader": reader,
                "ansatz": ansatz_name,
                "n_layers": n_layers,
                "n_seeds": agg["n_seeds"],
                "dev_acc_mean": agg["best_dev_acc_mean"],
                "dev_acc_std": agg["best_dev_acc_std"],
                "n_params": agg["n_params"],
            }
        )

    # Print summary
    print(f"\n{'=' * 80}")
    print(f"  Phase 5 — Quantum training summary (DEV set)")
    print(f"{'=' * 80}")
    print(f"  {'reader':<8} {'ansatz':<8} {'layers':>6} {'seeds':>6} "
          f"{'dev_acc (mean±std)':>22} {'params':>10}")
    print("-" * 80)
    for row in summary_rows:
        print(
            f"  {row['reader']:<8} {row['ansatz']:<8} {row['n_layers']:>6} "
            f"{row['n_seeds']:>6}  "
            f"{row['dev_acc_mean']:.4f} ± {row['dev_acc_std']:.4f}  "
            f"{row['n_params']:>10,}"
        )
    print("=" * 80)

    summary_json_path = args.out_dir / "summary.json"
    with summary_json_path.open("w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)
    summary_txt_path = args.out_dir / "summary.txt"
    with summary_txt_path.open("w", encoding="utf-8") as f:
        f.write("Phase 5 — Quantum training summary (DEV set)\n")
        f.write("=" * 80 + "\n")
        for row in summary_rows:
            f.write(
                f"{row['reader']:<8} {row['ansatz']:<8} L{row['n_layers']}  "
                f"acc={row['dev_acc_mean']:.4f}±{row['dev_acc_std']:.4f}  "
                f"params={row['n_params']:,}  seeds={row['n_seeds']}\n"
            )
    print(f"\n[save] {summary_json_path}")
    print(f"[save] {summary_txt_path}")


if __name__ == "__main__":
    main()
