#!/usr/bin/env python3
"""Lọc Stanford SST thành SST-2 phrase ngắn và lưu thành CSV.

Pipeline:
    1. Đọc data/raw/sst/stanfordSentimentTreebank/
    2. Chọn source = sentences hoặc phrases
    3. Filter theo độ dài (min_words ≤ #words ≤ max_words)
    4. Binarize sentiment (drop neutral band)
    5. Split train/dev/test
    6. Lưu data/processed/sst2_short_{train,dev,test}.csv

Cách dùng:
    python scripts/02_prepare_data.py
    python scripts/02_prepare_data.py --source sentences --max-len 5
    python scripts/02_prepare_data.py --source phrases --max-len 5 --neutral 0.4 0.6
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Cho phép import src.* khi chạy từ project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.sst import prepare_short_sst2  # noqa: E402


DEFAULT_RAW = Path("data/raw/sst/stanfordSentimentTreebank")
DEFAULT_OUT = Path("data/processed")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--raw-dir",
        type=Path,
        default=DEFAULT_RAW,
        help=f"Thư mục Stanford SST raw (default: {DEFAULT_RAW})",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Thư mục lưu CSV (default: {DEFAULT_OUT})",
    )
    p.add_argument(
        "--source",
        choices=["sentences", "phrases"],
        default="phrases",
        help="'sentences' = chỉ câu gốc (~11.8k); 'phrases' = tất cả phrase (~239k). "
        "Khuyến nghị 'phrases' cho QNLP vì cần dữ liệu ngắn nhiều.",
    )
    p.add_argument("--max-len", type=int, default=5, help="Số từ tối đa (default: 5)")
    p.add_argument("--min-len", type=int, default=1, help="Số từ tối thiểu (default: 1)")
    p.add_argument(
        "--neutral",
        nargs=2,
        type=float,
        metavar=("LOW", "HIGH"),
        default=[0.4, 0.6],
        help="Khoảng score bị coi là neutral và drop (default: 0.4 0.6)",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Seed cho stratified split (chỉ áp dụng cho phrases)"
    )
    return p.parse_args()


def fmt_int(n: int) -> str:
    return f"{n:,}"


def summarize(splits: dict, source: str, max_len: int) -> None:
    """In bảng tổng kết ra console — dễ đọc trên SSH."""
    print()
    print("=" * 70)
    print(f"  Kết quả lọc SST → SST-2 (source={source}, max_len={max_len})")
    print("=" * 70)
    print(f"{'split':<8} {'#total':>10} {'#pos':>10} {'#neg':>10} {'pos_ratio':>10}")
    print("-" * 50)
    grand_total = 0
    for name in ("train", "dev", "test"):
        df = splits[name]
        n = len(df)
        n_pos = int((df["label"] == 1).sum())
        n_neg = int((df["label"] == 0).sum())
        ratio = n_pos / n if n > 0 else 0.0
        print(f"{name:<8} {fmt_int(n):>10} {fmt_int(n_pos):>10} {fmt_int(n_neg):>10} {ratio:>10.3f}")
        grand_total += n
    print("-" * 50)
    print(f"{'TOTAL':<8} {fmt_int(grand_total):>10}")
    print("=" * 70)


def sample_rows(splits: dict, n: int = 5) -> None:
    """In vài ví dụ mẫu cho mỗi split."""
    print(f"\n[sample] {n} ví dụ đầu mỗi split:\n")
    for name in ("train", "dev", "test"):
        df = splits[name]
        if df.empty:
            print(f"  [{name}] (rỗng)")
            continue
        print(f"  ─── {name} ─────────────────────────────")
        for _, row in df.head(n).iterrows():
            label_str = "POS" if row["label"] == 1 else "NEG"
            print(f"    [{label_str}] (score={row['score']:.3f})  {row['text']}")
        print()


def save(splits: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in splits.items():
        path = out_dir / f"sst2_short_{name}.csv"
        df.to_csv(path, index=False)
        print(f"  ✓ {path}  ({len(df):,} rows)")


def main() -> None:
    args = parse_args()

    if not args.raw_dir.is_dir():
        print(f"[lỗi] Không tìm thấy {args.raw_dir}.")
        print("       Chạy trước:  python scripts/01_download_sst.py")
        sys.exit(1)

    print(f"[load] đọc Stanford SST từ {args.raw_dir}")
    print(f"[config] source={args.source}, min_len={args.min_len}, max_len={args.max_len}, "
          f"neutral={tuple(args.neutral)}, seed={args.seed}")

    splits = prepare_short_sst2(
        sst_dir=args.raw_dir,
        source=args.source,
        max_words=args.max_len,
        min_words=args.min_len,
        neutral_band=tuple(args.neutral),
        seed=args.seed,
    )

    summarize(splits, args.source, args.max_len)
    sample_rows(splits, n=5)

    print(f"\n[save] ghi CSV vào {args.out_dir}/")
    save(splits, args.out_dir)
    print("\n[done] Tiếp theo: python scripts/03_eda.py")


if __name__ == "__main__":
    main()
