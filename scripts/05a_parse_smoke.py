#!/usr/bin/env python3
"""Smoke test cho lambeq reader/parser + Rewriter.

Mục đích: trước khi chạy parse 3,800 phrase, verify nhanh rằng:
    1. lambeq cài đúng
    2. Reader hoạt động (bobcat cần tải, spiders/cups/linear không cần)
    3. Rewriter chạy được (chỉ bobcat)
    4. Diagram vẽ được PNG headless

Test trên 10 phrase mẫu (5 từ qnlp_train.csv + 5 hardcoded).

Cách dùng:
    python scripts/05a_parse_smoke.py                       # mặc định bobcat
    python scripts/05a_parse_smoke.py --reader spiders     # KHÔNG cần download
    python scripts/05a_parse_smoke.py --reader cups
    python scripts/05a_parse_smoke.py --no-rewriter
    python scripts/05a_parse_smoke.py --extra "the food was terrible"
    python scripts/05a_parse_smoke.py --reader bobcat --model-path /path/to/local
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.discocat.parse import parse_batch  # noqa: E402
from src.discocat.visualize import save_sample_grid  # noqa: E402


DEFAULT_TRAIN_CSV = Path("data/processed/qnlp_train.csv")
DEFAULT_OUT_DIR = Path("results/diagrams/smoke")

HARDCODED_EXAMPLES = [
    ("good movie", 1),
    ("a banal script", 0),
    ("not one clever line", 0),
    ("an extraordinary film", 1),
    ("makes no sense", 0),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--n-from-train", type=int, default=5, help="Lấy n phrase đầu từ train CSV")
    p.add_argument(
        "--reader",
        choices=["bobcat", "spiders", "cups", "linear"],
        default="bobcat",
        help="Loại reader. bobcat = chuẩn CCG (cần model); "
             "spiders/cups/linear = không cần download.",
    )
    p.add_argument(
        "--model-path", default=None,
        help="Path local cho bobcat model (nếu URL online dead)",
    )
    p.add_argument("--no-rewriter", action="store_true",
                   help="Bỏ qua rewriter (tự động cho spiders/cups/linear)")
    p.add_argument(
        "--extra", action="append", default=[],
        help="Thêm phrase custom để test. Có thể lặp lại flag này.",
    )
    return p.parse_args()


def collect_phrases(args) -> tuple[list[str], list[int], list[int]]:
    texts: list[str] = []
    labels: list[int] = []
    source_ids: list[int] = []

    if args.train_csv.is_file() and args.n_from_train > 0:
        df = pd.read_csv(args.train_csv).head(args.n_from_train)
        for i, row in df.iterrows():
            texts.append(str(row["text"]))
            labels.append(int(row["label"]))
            source_ids.append(int(row.get("source_id", -1)))

    for text, label in HARDCODED_EXAMPLES:
        texts.append(text)
        labels.append(label)
        source_ids.append(-1)

    for text in args.extra:
        texts.append(text)
        labels.append(-1)  # unknown
        source_ids.append(-1)

    return texts, labels, source_ids


def main() -> None:
    args = parse_args()
    texts, labels, source_ids = collect_phrases(args)

    if not texts:
        print("[lỗi] không có phrase để test.")
        sys.exit(1)

    print(f"[smoke] sẽ parse {len(texts)} phrase:")
    for i, (t, l) in enumerate(zip(texts, labels)):
        label_str = {1: "POS", 0: "NEG", -1: "???"}[l]
        print(f"   {i:2d}. [{label_str}] {t}")

    print(f"\n[smoke] load reader='{args.reader}'")
    if args.reader == "bobcat" and args.model_path is None:
        print(f"        (lần đầu sẽ tải ~ 500 MB từ URL của lambeq)")
    elif args.reader == "bobcat":
        print(f"        (dùng model local: {args.model_path})")
    else:
        print(f"        (không cần download)")
    t0 = time.time()
    result = parse_batch(
        texts=texts,
        labels=labels,
        source_ids=source_ids,
        reader_type=args.reader,
        model_path=args.model_path,
        apply_rewriter=not args.no_rewriter,
        verbose=True,
        desc="smoke",
    )
    elapsed = time.time() - t0
    print(f"\n[smoke] xong trong {elapsed:.1f}s. "
          f"success {result.n_success}/{result.n_total}, fail {result.n_failed}.")

    if result.failures:
        print("\n[smoke] Failures:")
        for fail in result.failures:
            print(f"   - {fail['text']!r}  ({fail['error_type']}: {fail['error_msg']})")

    sample_dir = args.out_dir / args.reader
    print(f"\n[smoke] vẽ PNG vào {sample_dir}/")
    saved = save_sample_grid(
        diagrams=result.diagrams,
        texts=result.texts,
        labels=result.labels,
        out_dir=sample_dir,
        n_samples=len(texts),
        prefix=f"smoke_{args.reader}",
    )
    print(f"  ✓ lưu {len(saved)} file PNG")
    for p in saved:
        print(f"     - {p}")

    print("\n[done] Nếu PNG đẹp + ít/không failure, sang scripts/05_parse_diagrams.py")


if __name__ == "__main__":
    main()
