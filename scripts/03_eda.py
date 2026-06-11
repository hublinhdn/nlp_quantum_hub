#!/usr/bin/env python3
"""EDA cho tập SST-2 đã lọc.

Đầu vào:
    data/processed/sst2_short_{train,dev,test}.csv

Đầu ra (lưu vào results/eda/):
    - 01_label_distribution.png       — bar chart pos/neg theo split
    - 02_length_distribution.png      — histogram độ dài câu
    - 03_vocab_freq_top30.png         — top 30 từ phổ biến nhất
    - 04_length_vs_label.png          — boxplot độ dài theo nhãn
    - summary.txt                     — bảng thống kê text-only (đọc trên SSH)

Cách dùng:
    python scripts/03_eda.py
    python scripts/03_eda.py --processed-dir data/processed --out-dir results/eda
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # bắt buộc cho SSH: không có display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DEFAULT_IN = Path("data/processed")
DEFAULT_OUT = Path("results/eda")

SPLITS = ("train", "dev", "test")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--processed-dir",
        type=Path,
        default=DEFAULT_IN,
        help=f"Thư mục chứa CSV (default: {DEFAULT_IN})",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Thư mục lưu plot (default: {DEFAULT_OUT})",
    )
    p.add_argument("--top-k-vocab", type=int, default=30, help="Top-K từ phổ biến nhất (default: 30)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_processed(in_dir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for name in SPLITS:
        path = in_dir / f"sst2_short_{name}.csv"
        if not path.is_file():
            print(f"[lỗi] Không tìm thấy {path}.")
            print("       Chạy trước:  python scripts/02_prepare_data.py")
            sys.exit(1)
        df = pd.read_csv(path)
        df["n_words"] = df["text"].astype(str).str.split().str.len()
        out[name] = df
    return out


# ---------------------------------------------------------------------------
# Text summary (in console + ghi summary.txt)
# ---------------------------------------------------------------------------


def write_summary(splits: dict[str, pd.DataFrame], out_path: Path) -> str:
    """Tạo bảng thống kê text-only và ghi ra file. Trả về nội dung để in tiếp ra stdout."""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("  SST-2 short — EDA summary")
    lines.append("=" * 78)
    lines.append("")
    lines.append(
        f"{'split':<6} {'#total':>8} {'#pos':>8} {'#neg':>8} {'pos_ratio':>10} "
        f"{'mean_len':>10} {'min_len':>8} {'max_len':>8}"
    )
    lines.append("-" * 78)
    for name in SPLITS:
        df = splits[name]
        n = len(df)
        n_pos = int((df["label"] == 1).sum())
        n_neg = int((df["label"] == 0).sum())
        ratio = n_pos / n if n > 0 else 0.0
        mean_len = df["n_words"].mean() if n > 0 else 0.0
        min_len = int(df["n_words"].min()) if n > 0 else 0
        max_len = int(df["n_words"].max()) if n > 0 else 0
        lines.append(
            f"{name:<6} {n:>8,} {n_pos:>8,} {n_neg:>8,} {ratio:>10.3f} "
            f"{mean_len:>10.2f} {min_len:>8d} {max_len:>8d}"
        )

    total = sum(len(df) for df in splits.values())
    lines.append("-" * 78)
    lines.append(f"{'TOTAL':<6} {total:>8,}")
    lines.append("=" * 78)

    # Vocab stats
    lines.append("")
    lines.append("Vocab (tính trên train):")
    train_tokens = [tok for text in splits["train"]["text"] for tok in str(text).split()]
    vocab = Counter(train_tokens)
    lines.append(f"  - Tổng số token         : {len(train_tokens):,}")
    lines.append(f"  - Vocab size (unique)   : {len(vocab):,}")
    lines.append(f"  - Token xuất hiện ≥ 5   : {sum(1 for v in vocab.values() if v >= 5):,}")
    lines.append(f"  - Token chỉ 1 lần       : {sum(1 for v in vocab.values() if v == 1):,}")

    # Length distribution train
    lines.append("")
    lines.append("Phân phối độ dài câu (train):")
    counts = splits["train"]["n_words"].value_counts().sort_index()
    for k, v in counts.items():
        bar = "█" * int(40 * v / counts.max())
        lines.append(f"  {int(k):>2}d  {bar:<40} {v:>6,}")

    content = "\n".join(lines) + "\n"
    out_path.write_text(content, encoding="utf-8")
    return content


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_label_distribution(splits: dict[str, pd.DataFrame], out_path: Path) -> None:
    rows = []
    for name, df in splits.items():
        for label in (0, 1):
            count = int((df["label"] == label).sum())
            rows.append({"split": name, "label": "neg" if label == 0 else "pos", "count": count})
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.barplot(
        data=plot_df, x="split", y="count", hue="label", order=SPLITS,
        palette={"neg": "#d9534f", "pos": "#5cb85c"}, ax=ax,
    )
    ax.set_title("Phân phối nhãn theo split (SST-2 short)")
    ax.set_xlabel("split")
    ax.set_ylabel("số phrase")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", fontsize=9, padding=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_length_distribution(splits: dict[str, pd.DataFrame], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name in SPLITS:
        df = splits[name]
        if df.empty:
            continue
        ax.hist(df["n_words"], bins=range(1, int(df["n_words"].max()) + 2),
                alpha=0.55, label=name, edgecolor="black")
    ax.set_title("Phân phối độ dài câu (số từ)")
    ax.set_xlabel("số từ")
    ax.set_ylabel("số phrase")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_vocab_top(splits: dict[str, pd.DataFrame], out_path: Path, top_k: int = 30) -> None:
    tokens = [tok.lower() for text in splits["train"]["text"] for tok in str(text).split()]
    most = Counter(tokens).most_common(top_k)
    if not most:
        return
    words, counts = zip(*most)

    fig, ax = plt.subplots(figsize=(8, max(5, top_k * 0.25)))
    ax.barh(range(len(words)), counts, color="#3a7ca5")
    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_title(f"Top {top_k} từ phổ biến nhất (train, lowercase)")
    ax.set_xlabel("tần suất")
    for i, c in enumerate(counts):
        ax.text(c, i, f" {c}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_length_vs_label(splits: dict[str, pd.DataFrame], out_path: Path) -> None:
    df = pd.concat(
        [splits[name].assign(_split=name) for name in SPLITS], ignore_index=True
    )
    df["label_str"] = df["label"].map({0: "neg", 1: "pos"})

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sns.boxplot(
        data=df, x="_split", y="n_words", hue="label_str",
        order=SPLITS, palette={"neg": "#d9534f", "pos": "#5cb85c"}, ax=ax,
    )
    ax.set_title("Độ dài câu theo nhãn × split")
    ax.set_xlabel("split")
    ax.set_ylabel("số từ")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    print(f"[load] đọc CSV từ {args.processed_dir}/")
    splits = load_processed(args.processed_dir)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary_path = args.out_dir / "summary.txt"
    summary_text = write_summary(splits, summary_path)
    print()
    print(summary_text)
    print(f"[save] {summary_path}")

    print("\n[plot] sinh các biểu đồ...")
    plots = [
        ("01_label_distribution.png", plot_label_distribution),
        ("02_length_distribution.png", plot_length_distribution),
        ("04_length_vs_label.png", plot_length_vs_label),
    ]
    for filename, fn in plots:
        path = args.out_dir / filename
        fn(splits, path)
        print(f"  ✓ {path}")

    vocab_path = args.out_dir / "03_vocab_freq_top30.png"
    plot_vocab_top(splits, vocab_path, top_k=args.top_k_vocab)
    print(f"  ✓ {vocab_path}")

    print("\n[done] EDA hoàn tất.")
    print(f"       Plots ở: {args.out_dir}/")
    print(f"       Summary: {summary_path}")
    print("       Để xem trên máy local:")
    print(f"         scp -r user@remote:~/quantum_project/{args.out_dir} ./results/")


if __name__ == "__main__":
    main()
