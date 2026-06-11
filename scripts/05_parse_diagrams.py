#!/usr/bin/env python3
"""Parse toàn bộ qnlp_{train,dev,test}.csv → DisCoCat string diagram.

Output (tách theo reader để không ghi đè nhau):
    data/processed/
        diagrams_{reader}_train.pkl    # list[{source_id, text, label, diagram}]
        diagrams_{reader}_dev.pkl
        diagrams_{reader}_test.pkl
        parse_failures_{reader}.csv    # ghi nhận phrase fail
    results/diagrams/{reader}/
        train/*.png                    # ~ 20 sample
        dev/*.png                      # ~ 10 sample
        test/*.png                     # ~ 5 sample

Pickle format:
    [
        {"source_id": int, "text": str, "label": int, "diagram": Diagram},
        ...
    ]

Lưu ý:
    - Parser load model ~ 500 MB lần đầu (mất 1-2 phút)
    - Parse 3,800 phrase ~ 10-30 phút tuỳ máy
    - Failures là chuyện bình thường (BobcatParser không 100% chính xác).
      Báo cáo % success ở cuối.

Cách dùng:
    python scripts/05_parse_diagrams.py                          # mặc định bobcat
    python scripts/05_parse_diagrams.py --reader spiders         # không cần download
    python scripts/05_parse_diagrams.py --no-rewriter
    python scripts/05_parse_diagrams.py --splits train           # chỉ parse train
    python scripts/05_parse_diagrams.py --max-per-split 100      # debug nhanh
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.discocat.parse import (  # noqa: E402
    DEFAULT_REWRITE_RULES,
    make_parser,
    make_rewriter,
    parse_batch,
)
from src.discocat.visualize import save_sample_grid  # noqa: E402


DEFAULT_PROCESSED = Path("data/processed")
DEFAULT_DIAGRAMS_DIR = Path("results/diagrams")

ALL_SPLITS = ("train", "dev", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--processed-dir", type=Path, default=DEFAULT_PROCESSED)
    p.add_argument("--diagrams-dir", type=Path, default=DEFAULT_DIAGRAMS_DIR)
    p.add_argument(
        "--splits", nargs="+", choices=ALL_SPLITS, default=list(ALL_SPLITS),
        help=f"Splits cần parse (default: {' '.join(ALL_SPLITS)})",
    )
    p.add_argument(
        "--max-per-split", type=int, default=None,
        help="Giới hạn số phrase mỗi split (debug). None = không giới hạn.",
    )
    p.add_argument(
        "--reader",
        choices=["bobcat", "spiders", "cups", "linear"],
        default="bobcat",
        help="Loại reader (default: bobcat)",
    )
    p.add_argument("--model-path", default=None, help="Path local cho bobcat model")
    p.add_argument("--no-rewriter", action="store_true",
                   help="Bỏ qua rewriter (tự động cho spiders/cups/linear)")
    p.add_argument(
        "--samples-train", type=int, default=20,
        help="Số PNG sample lưu từ split train (default: 20)",
    )
    p.add_argument(
        "--samples-dev", type=int, default=10,
        help="Số PNG sample lưu từ split dev (default: 10)",
    )
    return p.parse_args()


def load_split(processed_dir: Path, split: str, max_n: int | None) -> pd.DataFrame:
    path = processed_dir / f"qnlp_{split}.csv"
    if not path.is_file():
        print(f"[lỗi] không có {path}. Chạy scripts/02b_subsample_qnlp.py trước.")
        sys.exit(1)
    df = pd.read_csv(path)
    if max_n is not None:
        df = df.head(max_n)
    return df


def save_pickle(result, path: Path) -> None:
    """Lưu list of dict (chỉ phrase parse thành công)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for diag, text, label, source_id in zip(
        result.diagrams, result.texts, result.labels, result.source_ids
    ):
        if diag is None:
            continue
        records.append(
            {
                "source_id": source_id,
                "text": text,
                "label": label,
                "diagram": diag,
            }
        )
    with path.open("wb") as f:
        pickle.dump(records, f)


def main() -> None:
    args = parse_args()

    print(f"[load] reader='{args.reader}'")
    if args.reader == "bobcat" and args.model_path is None:
        print(f"  (lần đầu sẽ tải ~ 500 MB)")
    elif args.reader == "bobcat":
        print(f"  (dùng model local: {args.model_path})")
    else:
        print(f"  (không cần download)")
    t0 = time.time()
    parser = make_parser(reader_type=args.reader, model_path=args.model_path)
    # Rewriter chỉ hữu ích cho bobcat
    use_rewriter = (args.reader == "bobcat") and not args.no_rewriter
    rewriter = make_rewriter(DEFAULT_REWRITE_RULES) if use_rewriter else None
    print(f"  ✓ parser sẵn sàng sau {time.time() - t0:.1f}s")
    if rewriter is not None:
        print(f"  rewriter rules: {', '.join(DEFAULT_REWRITE_RULES)}")
    else:
        reason = "không áp với " + args.reader if args.reader != "bobcat" else "--no-rewriter"
        print(f"  rewriter: DISABLED ({reason})")

    args.diagrams_dir.mkdir(parents=True, exist_ok=True)

    all_failures: list[dict] = []
    summary_rows: list[tuple[str, int, int, float]] = []  # (split, total, success, time_s)

    for split in args.splits:
        df = load_split(args.processed_dir, split, args.max_per_split)
        print(f"\n[parse:{split}] n = {len(df):,} phrase")

        t0 = time.time()
        result = parse_batch(
            texts=df["text"].astype(str).tolist(),
            labels=df["label"].astype(int).tolist(),
            source_ids=df.get("source_id", pd.Series([-1] * len(df))).astype(int).tolist(),
            parser=parser,
            rewriter=rewriter,
            apply_rewriter=rewriter is not None,
            reader_type=args.reader,
            desc=f"parse {split}",
        )
        elapsed = time.time() - t0
        print(f"  success: {result.n_success}/{result.n_total} "
              f"({100.0 * result.n_success / max(result.n_total, 1):.1f}%), "
              f"fail: {result.n_failed}, time: {elapsed:.1f}s")
        summary_rows.append((split, result.n_total, result.n_success, elapsed))

        # Annotate failures với split
        for f in result.failures:
            f["split"] = split
        all_failures.extend(result.failures)

        # Save pickle — đặt theo reader để không ghi đè giữa các reader
        out_pkl = args.processed_dir / f"diagrams_{args.reader}_{split}.pkl"
        save_pickle(result, out_pkl)
        print(f"  ✓ saved {out_pkl}")

        # Save sample PNG — tách dir theo reader
        n_samples = {"train": args.samples_train, "dev": args.samples_dev, "test": 5}.get(split, 5)
        sample_dir = args.diagrams_dir / args.reader / split
        print(f"  vẽ {n_samples} PNG sample vào {sample_dir}/")
        save_sample_grid(
            diagrams=result.diagrams,
            texts=result.texts,
            labels=result.labels,
            out_dir=sample_dir,
            n_samples=n_samples,
            prefix=f"{args.reader}_{split}",
        )

    # Save failures (tách theo reader)
    if all_failures:
        fail_df = pd.DataFrame(all_failures)
        fail_path = args.processed_dir / f"parse_failures_{args.reader}.csv"
        fail_df.to_csv(fail_path, index=False)
        print(f"\n[save] failures → {fail_path}  ({len(fail_df)} rows)")

    # Final summary
    print(f"\n{'=' * 60}")
    print("  Phase 4 — Parsing summary")
    print("=" * 60)
    print(f"  {'split':<8} {'total':>8} {'success':>10} {'rate':>8} {'time':>10}")
    print("-" * 60)
    for split, total, success, t in summary_rows:
        rate = 100.0 * success / max(total, 1)
        print(f"  {split:<8} {total:>8,} {success:>10,} {rate:>7.1f}% {t:>9.1f}s")
    print("=" * 60)
    print(f"\n[done] Tiếp theo: Phase 5 — train quantum model trên diagrams.")


if __name__ == "__main__":
    main()
