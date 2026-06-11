#!/usr/bin/env python3
"""Train classical baselines cho Phase 3.

Chạy mặc định 3 model (LR+TF-IDF, BiLSTM, LR+GloVe nếu có file) × 3 seeds,
mỗi run save metrics JSON + 1 bảng tổng kết text-only ra console + summary.txt.

KHÔNG đụng tới qnlp_test.csv — chỉ eval trên dev. Test eval để dành Phase 6.

Cách dùng:
    python scripts/04_train_baselines.py
    python scripts/04_train_baselines.py --models tfidf bilstm
    python scripts/04_train_baselines.py --models tfidf --seeds 0 1 2 3 4
    python scripts/04_train_baselines.py --models glove   # cần data/raw/glove/...

Output:
    results/baseline/
        lr_tfidf/seed_{n}/result.json
        lr_tfidf/aggregated.json
        bilstm/seed_{n}/result.json
        bilstm/aggregated.json
        lr_glove/...  (nếu chạy)
        summary.txt
        summary.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.bilstm import result_to_dict as bilstm_to_dict  # noqa: E402
from src.baselines.bilstm import train_bilstm  # noqa: E402
from src.baselines.common import (  # noqa: E402
    AggregatedMetrics,
    Metrics,
    aggregate,
    format_metrics_row,
    load_qnlp,
)
from src.baselines.lr import result_to_dict as lr_to_dict  # noqa: E402
from src.baselines.lr import train_lr_tfidf  # noqa: E402
from src.baselines.lr_glove import result_to_dict as glove_to_dict  # noqa: E402
from src.baselines.lr_glove import train_lr_glove  # noqa: E402
from src.baselines.glove import DEFAULT_GLOVE_PATH, load_glove  # noqa: E402


DEFAULT_PROCESSED = Path("data/processed")
DEFAULT_OUT = Path("results/baseline")
DEFAULT_VOCAB = Path("data/processed/qnlp_vocab.tsv")
DEFAULT_SEEDS = (0, 1, 2)
DEFAULT_MODELS = ("tfidf", "bilstm")
ALL_MODELS = ("tfidf", "bilstm", "glove")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--models", nargs="+", choices=ALL_MODELS, default=list(DEFAULT_MODELS),
        help=f"Model nào để train (default: {' '.join(DEFAULT_MODELS)})",
    )
    p.add_argument(
        "--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS),
        help=f"Seeds (default: {' '.join(str(s) for s in DEFAULT_SEEDS)})",
    )
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--vocab-path", type=Path, default=DEFAULT_VOCAB)
    p.add_argument("--glove-path", type=Path, default=DEFAULT_GLOVE_PATH)
    p.add_argument("--bilstm-max-epochs", type=int, default=50)
    p.add_argument("--bilstm-patience", type=int, default=5)
    p.add_argument("--bilstm-batch-size", type=int, default=32)
    p.add_argument("--bilstm-quiet", action="store_true", help="Bỏ in epoch log của BiLSTM")
    return p.parse_args()


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def run_tfidf(splits, seeds, out_dir) -> tuple[AggregatedMetrics, list[Metrics]]:
    print(f"\n{'=' * 70}\n  LR + TF-IDF\n{'=' * 70}")
    metrics_list: list[Metrics] = []
    for seed in seeds:
        print(f"\n  [seed={seed}]")
        t0 = time.time()
        result = train_lr_tfidf(splits["train"], splits["dev"], seed=seed)
        elapsed = time.time() - t0
        print(f"    best_C = {result.best_C}  features = {result.n_features:,}  "
              f"dev_acc = {result.dev_metrics.accuracy:.4f}  ({elapsed:.1f}s)")
        save_json(lr_to_dict(result), out_dir / "lr_tfidf" / f"seed_{seed}" / "result.json")
        metrics_list.append(result.dev_metrics)
    agg = aggregate(metrics_list)
    save_json(agg.to_dict(), out_dir / "lr_tfidf" / "aggregated.json")
    return agg, metrics_list


def run_bilstm(splits, seeds, out_dir, vocab_path, args) -> tuple[AggregatedMetrics, list[Metrics]]:
    print(f"\n{'=' * 70}\n  BiLSTM\n{'=' * 70}")
    metrics_list: list[Metrics] = []
    for seed in seeds:
        print(f"\n  [seed={seed}]")
        t0 = time.time()
        result = train_bilstm(
            splits["train"], splits["dev"],
            vocab_path=vocab_path,
            seed=seed,
            max_epochs=args.bilstm_max_epochs,
            patience=args.bilstm_patience,
            batch_size=args.bilstm_batch_size,
            verbose=not args.bilstm_quiet,
        )
        elapsed = time.time() - t0
        print(f"    best_epoch = {result.best_epoch}  "
              f"dev_loss = {result.best_dev_loss:.4f}  "
              f"dev_acc = {result.dev_metrics.accuracy:.4f}  ({elapsed:.1f}s)")
        save_json(bilstm_to_dict(result), out_dir / "bilstm" / f"seed_{seed}" / "result.json")
        metrics_list.append(result.dev_metrics)
    agg = aggregate(metrics_list)
    save_json(agg.to_dict(), out_dir / "bilstm" / "aggregated.json")
    return agg, metrics_list


def run_glove(splits, seeds, out_dir, glove_path) -> tuple[AggregatedMetrics, list[Metrics]] | None:
    if not glove_path.is_file():
        print(f"\n[skip] LR + GloVe — không tìm thấy {glove_path}")
        print(f"       Chạy:  python scripts/00_download_glove.py")
        return None
    print(f"\n{'=' * 70}\n  LR + GloVe-50d\n{'=' * 70}")
    print(f"  [load] {glove_path}")
    t0 = time.time()
    embeddings = load_glove(glove_path)
    print(f"  loaded {len(embeddings):,} vectors trong {time.time() - t0:.1f}s")

    metrics_list: list[Metrics] = []
    for seed in seeds:
        print(f"\n  [seed={seed}]")
        t0 = time.time()
        result = train_lr_glove(
            splits["train"], splits["dev"], embeddings=embeddings, seed=seed
        )
        elapsed = time.time() - t0
        print(f"    best_C = {result.best_C}  dim = {result.embedding_dim}  "
              f"dev_acc = {result.dev_metrics.accuracy:.4f}  ({elapsed:.1f}s)")
        save_json(glove_to_dict(result), out_dir / "lr_glove" / f"seed_{seed}" / "result.json")
        metrics_list.append(result.dev_metrics)
    agg = aggregate(metrics_list)
    save_json(agg.to_dict(), out_dir / "lr_glove" / "aggregated.json")
    return agg, metrics_list


def main() -> None:
    args = parse_args()

    print(f"[load] qnlp dataset từ {args.processed_dir}/")
    splits = load_qnlp(args.processed_dir)
    print(f"  train: {len(splits['train']):,}  dev: {len(splits['dev']):,}  "
          f"test: {len(splits['test']):,} (test KHÔNG dùng ở Phase 3)")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, AggregatedMetrics] = {}

    if "tfidf" in args.models:
        agg, _ = run_tfidf(splits, args.seeds, args.out_dir)
        results["LR + TF-IDF"] = agg
    if "bilstm" in args.models:
        agg, _ = run_bilstm(splits, args.seeds, args.out_dir, args.vocab_path, args)
        results["BiLSTM 1L"] = agg
    if "glove" in args.models:
        out = run_glove(splits, args.seeds, args.out_dir, args.glove_path)
        if out is not None:
            results["LR + GloVe-50d"] = out[0]

    # Summary
    print(f"\n{'=' * 70}")
    print("  Phase 3 — Classical baselines (DEV set only)")
    print("=" * 70)
    print(f"  Majority-class baseline = 0.5000  (balanced 50:50)")
    print("-" * 70)
    lines = []
    for name, agg in results.items():
        line = format_metrics_row(name, agg)
        print(f"  {line}")
        lines.append(line)
    print("=" * 70)

    summary = {
        "majority_class_baseline": 0.5,
        "models": {name: agg.to_dict() for name, agg in results.items()},
    }
    save_json(summary, args.out_dir / "summary.json")
    (args.out_dir / "summary.txt").write_text(
        "\n".join([
            "Phase 3 — Classical baselines (DEV set only)",
            "=" * 70,
            "Majority-class baseline = 0.5000  (balanced 50:50)",
            "-" * 70,
            *lines,
            "=" * 70,
        ]) + "\n",
        encoding="utf-8",
    )
    print(f"\n[save] {args.out_dir}/summary.json")
    print(f"[save] {args.out_dir}/summary.txt")


if __name__ == "__main__":
    main()
