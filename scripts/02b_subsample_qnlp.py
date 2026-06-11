#!/usr/bin/env python3
"""Tạo subset `qnlp` cho thí nghiệm lượng tử.

Pipeline:
    1. Load data/processed/sst2_short_{train,dev,test}.csv (sản phẩm của 02_prepare_data.py)
    2. (mặc định) Lowercase text — đồng nhất với vocab tokenizer
    3. Filter min_len ≥ 2 (drop phrase 1-từ — không có composition)
    4. Build vocab từ TRAIN, top-K theo tần suất (default 1000)
    5. Drop phrase ở mọi split có chứa OOV token
    6. (mặc định) Balanced subsample 50:50 → 1500 pos + 1500 neg = 3000 train, ...
       Hoặc stratified theo pos_ratio thực tế nếu --no-balanced.
    7. Lưu data/processed/qnlp_{train,dev,test}.csv + qnlp_vocab.tsv

Cách dùng:
    python scripts/02b_subsample_qnlp.py
    python scripts/02b_subsample_qnlp.py --no-balanced            # giữ tỉ lệ thực tế
    python scripts/02b_subsample_qnlp.py --vocab-size 500         # vocab chặt hơn
    python scripts/02b_subsample_qnlp.py --no-lowercase           # giữ nguyên case
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.vocab import (  # noqa: E402
    build_vocab,
    filter_phrases_in_vocab,
    save_vocab,
)


DEFAULT_IN = Path("data/processed")
DEFAULT_OUT = Path("data/processed")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--in-dir", type=Path, default=DEFAULT_IN, help=f"(default: {DEFAULT_IN})")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help=f"(default: {DEFAULT_OUT})")
    p.add_argument("--min-len", type=int, default=2, help="Bỏ phrase < min-len từ (default: 2)")
    p.add_argument("--max-len", type=int, default=5, help="Bỏ phrase > max-len từ (default: 5)")
    p.add_argument(
        "--vocab-size", type=int, default=1000, help="Top-K token để giữ (default: 1000)"
    )
    p.add_argument("--min-freq", type=int, default=1, help="Bỏ token tần suất < min-freq")
    p.add_argument("--n-train", type=int, default=5000,
                   help="(default 5000) — tăng từ 3000 để vocab coverage tốt hơn")
    p.add_argument("--n-dev", type=int, default=400, help="(default: 400)")
    p.add_argument("--n-test", type=int, default=400, help="(default: 400)")
    p.add_argument("--seed", type=int, default=42, help="(default: 42)")
    p.add_argument(
        "--prefix",
        default="qnlp",
        help="Prefix tên file output, ví dụ 'qnlp' → qnlp_train.csv (default: qnlp)",
    )
    # Balanced sampling (default ON)
    p.add_argument(
        "--balanced",
        dest="balanced",
        action="store_true",
        default=True,
        help="Sample 50:50 pos/neg (default ON). Đảo bằng --no-balanced.",
    )
    p.add_argument(
        "--no-balanced", dest="balanced", action="store_false",
        help="Giữ tỉ lệ pos/neg theo dữ liệu thực (stratified theo ratio).",
    )
    # Lowercase (default ON)
    p.add_argument(
        "--lowercase",
        dest="lowercase",
        action="store_true",
        default=True,
        help="Lowercase text trước khi build vocab + save CSV (default ON). Đảo bằng --no-lowercase.",
    )
    p.add_argument(
        "--no-lowercase", dest="lowercase", action="store_false",
        help="Giữ nguyên case (không khuyến nghị cho DisCoCat/lambeq).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_splits(in_dir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for name in ("train", "dev", "test"):
        path = in_dir / f"sst2_short_{name}.csv"
        if not path.is_file():
            print(f"[lỗi] Không tìm thấy {path}.")
            print("       Chạy trước:  python scripts/02_prepare_data.py")
            sys.exit(1)
        out[name] = pd.read_csv(path)
    return out


def filter_length(df: pd.DataFrame, min_len: int, max_len: int) -> pd.DataFrame:
    n_words = df["text"].astype(str).str.split().str.len()
    return df.loc[(n_words >= min_len) & (n_words <= max_len)].reset_index(drop=True)


def stratified_subsample(
    df: pd.DataFrame, n: int, label_col: str = "label", seed: int = 42
) -> pd.DataFrame:
    """Lấy n row, giữ tỉ lệ label gần với df gốc."""
    if len(df) <= n:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    pos_ratio = (df[label_col] == 1).mean()
    n_pos = round(n * pos_ratio)
    n_neg = n - n_pos

    pos_df = df[df[label_col] == 1]
    neg_df = df[df[label_col] == 0]

    n_pos = min(n_pos, len(pos_df))
    n_neg = min(n_neg, len(neg_df))

    sampled_pos = pos_df.sample(n=n_pos, random_state=seed)
    sampled_neg = neg_df.sample(n=n_neg, random_state=seed)
    out = pd.concat([sampled_pos, sampled_neg], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def balanced_subsample(
    df: pd.DataFrame, n: int, label_col: str = "label", seed: int = 42
) -> pd.DataFrame:
    """Lấy n row với 50:50 pos/neg.

    Nếu n/2 vượt quá số phrase của class thiểu số → cap về min(pos, neg).
    """
    pos_df = df[df[label_col] == 1]
    neg_df = df[df[label_col] == 0]

    per_class = n // 2
    available = min(len(pos_df), len(neg_df))
    if per_class > available:
        per_class = available  # Cap về minority class

    sampled_pos = pos_df.sample(n=per_class, random_state=seed)
    sampled_neg = neg_df.sample(n=per_class, random_state=seed)
    out = pd.concat([sampled_pos, sampled_neg], ignore_index=True)
    return out.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def print_split_stats(name: str, df: pd.DataFrame) -> None:
    n = len(df)
    n_pos = int((df["label"] == 1).sum()) if n else 0
    n_neg = int((df["label"] == 0).sum()) if n else 0
    ratio = n_pos / n if n else 0.0
    mean_len = df["text"].astype(str).str.split().str.len().mean() if n else 0.0
    print(
        f"  {name:<7} n={n:>5,}  pos={n_pos:>5,}  neg={n_neg:>5,}  "
        f"pos_ratio={ratio:.3f}  mean_len={mean_len:.2f}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    print(f"[load] đọc full SST-2 short từ {args.in_dir}/")
    splits = load_splits(args.in_dir)

    print("\n[step 1/6] Stats ban đầu:")
    for name, df in splits.items():
        print_split_stats(name, df)

    # Lowercase (nếu bật)
    if args.lowercase:
        print("\n[step 2/6] Lowercase text (đồng nhất với vocab tokenizer)")
        for name, df in splits.items():
            df["text"] = df["text"].astype(str).str.lower()
    else:
        print("\n[step 2/6] (SKIP lowercase — --no-lowercase)")

    # Filter length
    print(f"\n[step 3/6] Lọc {args.min_len} ≤ len ≤ {args.max_len}")
    splits = {name: filter_length(df, args.min_len, args.max_len) for name, df in splits.items()}
    for name, df in splits.items():
        print_split_stats(name, df)

    # Build vocab từ train
    print(f"\n[step 4/6] Build vocab top-{args.vocab_size} (min_freq={args.min_freq}) từ train")
    vocab = build_vocab(
        splits["train"]["text"].tolist(),
        top_k=args.vocab_size,
        min_freq=args.min_freq,
    )
    print(f"  vocab size = {len(vocab):,} token")
    most_5 = list(vocab.items())[:5]
    least_5 = list(vocab.items())[-5:]
    print(f"  top 5      : {most_5}")
    print(f"  bottom 5   : {least_5}")

    # Drop phrase có OOV
    print(f"\n[step 5/6] Drop phrase chứa OOV (apply vocab cho TẤT CẢ split)")
    splits = {name: filter_phrases_in_vocab(df, vocab) for name, df in splits.items()}
    for name, df in splits.items():
        print_split_stats(name, df)

    # Sanity check
    targets = {"train": args.n_train, "dev": args.n_dev, "test": args.n_test}
    for name, target in targets.items():
        df = splits[name]
        n_avail = len(df)
        if args.balanced:
            min_class = min(int((df["label"] == 1).sum()), int((df["label"] == 0).sum()))
            max_balanced = min_class * 2
            if max_balanced < target:
                print(f"\n[cảnh báo] {name}: balanced sample max = {max_balanced} "
                      f"(minority class = {min_class}) < target {target}")
        elif n_avail < target:
            print(f"\n[cảnh báo] {name}: chỉ còn {n_avail} phrase < target {target}.")

    sampler = balanced_subsample if args.balanced else stratified_subsample
    mode = "balanced 50:50" if args.balanced else "stratified theo ratio"

    # === FIX QUAN TRỌNG: sample train TRƯỚC, sau đó constrain dev/test ⊆ train_words ===
    # Lý do: trong run trước, dev/test được sample độc lập với train. Có khả năng
    # dev/test chứa từ trong vocab top-1000 NHƯNG không xuất hiện trong qnlp_train sample.
    # Quantum model chỉ tạo param cho từ trong qnlp_train+dev → eval test fail vì
    # state_dict load vào wrong indices (lambeq sort symbols alphabetically).
    # Fix: enforce train_words ⊇ dev_words ⊇ test_words by construction.

    print(f"\n[step 6a/7] Subsample TRAIN trước ({mode}) → target {args.n_train}")
    train_out = sampler(splits["train"], args.n_train, "label", args.seed)
    print_split_stats("train", train_out)

    # Tính train_words = unique words trong qnlp_train sample
    train_words: set[str] = set()
    for text in train_out["text"]:
        train_words.update(str(text).split())
    print(f"\n[step 6b/7] qnlp_train có {len(train_words):,} từ unique "
          f"(trên vocab {len(vocab):,})")
    coverage = 100.0 * len(train_words) / len(vocab)
    print(f"            Vocab coverage = {coverage:.1f}%")
    if coverage < 60:
        print(f"            [WARNING] coverage thấp — tăng --n-train hoặc giảm --vocab-size")

    # Filter dev/test pools: chỉ giữ phrase có TẤT CẢ words ⊆ train_words
    print(f"\n[step 6c/7] Filter dev/test pools → chỉ phrase có words ⊆ train_words")

    def phrase_words_in_set(text: str, word_set: set[str]) -> bool:
        return all(w in word_set for w in str(text).split())

    for name in ("dev", "test"):
        pool = splits[name]
        mask = pool["text"].apply(lambda t: phrase_words_in_set(t, train_words))
        filtered = pool[mask].reset_index(drop=True)
        n_pos = int((filtered["label"] == 1).sum())
        n_neg = int((filtered["label"] == 0).sum())
        print(f"  {name}: pool {len(filtered):>5,}/{len(pool):>5,} "
              f"(pos={n_pos}, neg={n_neg})")
        splits[name] = filtered

        target = targets[name]
        if args.balanced:
            min_class = min(n_pos, n_neg)
            if min_class * 2 < target:
                print(f"  [WARN] {name} pool balanced max = {min_class * 2} < target {target}")
        elif len(filtered) < target:
            print(f"  [WARN] {name} pool = {len(filtered)} < target {target}")

    # Subsample dev/test từ filtered pools
    print(f"\n[step 7/7] Subsample dev/test từ filtered pools ({mode})")
    dev_out = sampler(splits["dev"], args.n_dev, "label", args.seed + 1)
    test_out = sampler(splits["test"], args.n_test, "label", args.seed + 2)

    out_splits = {"train": train_out, "dev": dev_out, "test": test_out}
    for name, df in out_splits.items():
        print_split_stats(name, df)

    # Final verification: chắc chắn dev/test words ⊆ train words
    dev_words: set[str] = set()
    for text in dev_out["text"]:
        dev_words.update(str(text).split())
    test_words: set[str] = set()
    for text in test_out["text"]:
        test_words.update(str(text).split())
    oot_dev = dev_words - train_words
    oot_test = test_words - train_words
    print(f"\n[verify] dev words không có trong train: {len(oot_dev)} (should be 0)")
    print(f"[verify] test words không có trong train: {len(oot_test)} (should be 0)")
    if oot_dev or oot_test:
        print(f"[WARNING] Vẫn còn out-of-train words — kiểm tra lại logic filter")

    # 5. Save
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[save] ghi vào {args.out_dir}/")
    for name, df in out_splits.items():
        path = args.out_dir / f"{args.prefix}_{name}.csv"
        df.to_csv(path, index=False)
        print(f"  ✓ {path}  ({len(df):,} rows)")

    vocab_path = args.out_dir / f"{args.prefix}_vocab.tsv"
    save_vocab(vocab, vocab_path)
    print(f"  ✓ {vocab_path}  ({len(vocab):,} token)")

    # Sample preview
    print("\n[sample] 3 ví dụ đầu mỗi split:\n")
    for name, df in out_splits.items():
        print(f"  ─── {name} ──────────────────")
        for _, row in df.head(3).iterrows():
            label_str = "POS" if row["label"] == 1 else "NEG"
            print(f"    [{label_str}] {row['text']}")
        print()

    print("[done] qnlp subset đã sẵn sàng cho Phase 3+.")
    print(f"       Tiếp theo: scripts/03_eda.py --processed-dir {args.out_dir} "
          f"(với prefix qnlp) — hoặc chạy Phase 3 baseline.")


if __name__ == "__main__":
    main()
