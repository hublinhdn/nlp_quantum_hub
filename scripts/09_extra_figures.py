#!/usr/bin/env python3
"""Sinh extra figures cho appendix báo cáo.

Đầu vào:
    results/final/predictions/{model}/predictions.csv
    results/final/test_metrics_quantum.json
    results/final/test_metrics_classical.json

Đầu ra (lưu vào results/final/plots/):
    07_per_length_heatmap.png      — accuracy × phrase length × model
    08_seed_consistency.png        — boxplot quantum acc per config qua 5 seeds
    09_negation_handling.png       — bar chart error rate trên câu có negation
    10_confidence_distribution.png — histogram confidence score per model

Cách dùng:
    python scripts/09_extra_figures.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DEFAULT_FINAL = Path("results/final")
NEGATION_TOKENS = {"not", "n't", "no", "never", "nor", "none", "nothing", "nobody"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_FINAL / "plots")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Discover all model prediction CSVs
# ---------------------------------------------------------------------------


def discover_models(final_dir: Path) -> dict[str, Path]:
    """Trả về dict {display_name: predictions.csv path} cho mỗi unique config.

    Aggregate qua seeds → dùng seed 0 representative cho per-length analysis.
    """
    pred_root = final_dir / "predictions"
    if not pred_root.is_dir():
        raise SystemExit(f"Không có {pred_root}")

    models: dict[str, Path] = {}
    # Classical (3 families, dùng seed 0)
    for fam, prefix in [("LR + TF-IDF", "lr_tfidf"), ("LR + GloVe", "lr_glove"),
                        ("BiLSTM", "bilstm")]:
        cand = pred_root / f"{prefix}_seed0"
        if (cand / "predictions.csv").is_file():
            models[fam] = cand / "predictions.csv"

    # Quantum: 12 unique configs (2 readers × 2 ansatzes × 3 layers)
    pattern = re.compile(r"quantum_(?P<reader>\w+)_(?P<ansatz>\w+)_L(?P<L>\d+)_seed0$")
    for d in sorted(pred_root.iterdir()):
        m = pattern.match(d.name)
        if not m or not (d / "predictions.csv").is_file():
            continue
        label = f"Q-{m['reader']}-{m['ansatz']}-L{m['L']}"
        models[label] = d / "predictions.csv"
    return models


# ---------------------------------------------------------------------------
# Figure 7: Per-length accuracy heatmap (model × phrase length)
# ---------------------------------------------------------------------------


def plot_per_length_heatmap(models: dict[str, Path], out_path: Path) -> None:
    rows = []
    for name, path in models.items():
        df = pd.read_csv(path)
        df["n_words"] = df["text"].astype(str).str.split().str.len()
        for n in sorted(df["n_words"].unique()):
            sub = df[df["n_words"] == n]
            if len(sub) == 0:
                continue
            acc = (sub["pred"] == sub["true_label"]).mean()
            rows.append({"model": name, "n_words": int(n), "acc": acc, "n": len(sub)})
    df_plot = pd.DataFrame(rows)
    if df_plot.empty:
        print("[skip] no predictions for per-length heatmap")
        return

    pivot = df_plot.pivot(index="model", columns="n_words", values="acc")
    fig, ax = plt.subplots(figsize=(8, max(5, 0.35 * len(pivot))))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", cmap="RdYlGn", vmin=0.4, vmax=1.0,
        cbar_kws={"label": "Accuracy"}, ax=ax, linewidths=0.4, linecolor="white",
    )
    ax.set_title("Test accuracy by phrase length (n_words)", fontsize=12)
    ax.set_xlabel("Phrase length (number of tokens)")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 8: Seed consistency boxplot for quantum configs
# ---------------------------------------------------------------------------


def plot_seed_consistency(final_dir: Path, out_path: Path) -> None:
    """Mean accuracy ± std with individual seed dots; clearer than boxplot
    when some configs have zero variance (Spiders saturation)."""
    with (final_dir / "test_metrics_quantum.json").open("r", encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for entry in data:
        rows.append({
            "config": f"Q-{entry['reader'][:3]}-{entry['ansatz'][:3]}-L{entry['n_layers']}",
            "reader": entry["reader"],
            "acc": entry["test_metrics"]["accuracy"],
            "seed": entry["seed"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return

    # Compute mean and std per config, sort by mean descending
    stats = (df.groupby("config")
               .agg(mean=("acc", "mean"), std=("acc", "std"),
                    reader=("reader", "first"))
               .reset_index()
               .sort_values("mean", ascending=False)
               .reset_index(drop=True))

    fig, ax = plt.subplots(figsize=(12, 6))

    # Bar with error bars, colour-coded by reader
    colors = {"spiders": "#5cb85c", "cups": "#f0ad4e"}
    bar_colors = [colors.get(r, "#888") for r in stats["reader"]]
    positions = np.arange(len(stats))

    ax.bar(positions, stats["mean"], yerr=stats["std"], capsize=5,
           color=bar_colors, alpha=0.65, edgecolor="black", linewidth=0.6,
           zorder=2)

    # Individual seed dots overlaid
    for i, cfg in enumerate(stats["config"]):
        seeds_acc = df[df["config"] == cfg]["acc"].values
        # Add small horizontal jitter
        jitter = np.linspace(-0.15, 0.15, len(seeds_acc))
        ax.scatter(np.full_like(seeds_acc, i, dtype=float) + jitter, seeds_acc,
                   color="black", s=22, alpha=0.85, zorder=3,
                   edgecolor="white", linewidth=0.4)

    # Numeric labels for mean above bars
    for i, m in enumerate(stats["mean"]):
        ax.text(i, m + max(stats["std"].max() * 1.2, 0.015), f"{m:.3f}",
                ha="center", va="bottom", fontsize=8.5)

    # Annotations
    ax.set_xticks(positions)
    ax.set_xticklabels(stats["config"], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Test accuracy")
    ax.set_xlabel("")
    ax.set_title("Per-configuration test accuracy across 5 seeds (bar = mean, error = std, dots = individual seeds)")
    ax.set_ylim(0.45, 1.0)
    ax.grid(axis="y", alpha=0.3, zorder=1)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.6, linewidth=0.8)
    ax.text(len(stats) - 0.5, 0.508, "majority baseline (0.5)",
            ha="right", fontsize=8, color="gray")

    # Legend
    from matplotlib.patches import Patch
    legend_elems = [
        Patch(color=colors["spiders"], alpha=0.65, label="Spiders reader"),
        Patch(color=colors["cups"], alpha=0.65, label="Cups reader"),
    ]
    ax.legend(handles=legend_elems, loc="lower left", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 9: Negation handling — error rate on negation phrases per model
# ---------------------------------------------------------------------------


def plot_negation_handling(models: dict[str, Path], out_path: Path) -> None:
    rows = []
    for name, path in models.items():
        df = pd.read_csv(path)
        df["has_negation"] = df["text"].astype(str).apply(
            lambda t: any(tok in NEGATION_TOKENS for tok in t.lower().split())
        )
        for has_neg, label in [(True, "Negation"), (False, "No negation")]:
            sub = df[df["has_negation"] == has_neg]
            if len(sub) == 0:
                continue
            err = (sub["pred"] != sub["true_label"]).mean()
            rows.append({"model": name, "category": label, "error_rate": err, "n": len(sub)})
    df_plot = pd.DataFrame(rows)
    if df_plot.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    pivot = df_plot.pivot(index="model", columns="category", values="error_rate")
    pivot = pivot.reindex(columns=["No negation", "Negation"])
    pivot.plot(kind="bar", ax=ax, width=0.75,
               color=["#5bc0de", "#d9534f"])
    ax.set_ylabel("Error rate (lower is better)")
    ax.set_xlabel("")
    ax.set_title("Error rate on phrases containing negation tokens vs.\\ no negation")
    ax.legend(title="Category", loc="upper right")
    plt.xticks(rotation=35, ha="right", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 10: Confidence distribution per model
# ---------------------------------------------------------------------------


def plot_confidence_distribution(models: dict[str, Path], out_path: Path) -> None:
    # Pick top 6 models for readability
    pick = [
        "LR + TF-IDF", "BiLSTM", "LR + GloVe",
        "Q-spiders-iqp-L1", "Q-cups-iqp-L1", "Q-cups-sim14-L1",
    ]
    pick = [m for m in pick if m in models]
    if not pick:
        return

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, sharey=True)
    for ax, name in zip(axes.flat, pick):
        df = pd.read_csv(models[name])
        # Confidence = max(prob_pos, 1 - prob_pos) → khoảng cách từ 0.5
        df["confidence"] = df["prob_pos"].apply(lambda p: max(p, 1 - p))
        df["correct"] = df["pred"] == df["true_label"]
        ax.hist([df[df["correct"]]["confidence"], df[~df["correct"]]["confidence"]],
                bins=20, stacked=True, color=["#5cb85c", "#d9534f"],
                label=["Correct", "Wrong"])
        ax.set_title(name, fontsize=10)
        ax.set_xlim(0.5, 1.0)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.supxlabel("Prediction confidence (max(p, 1-p))")
    fig.supylabel("Number of test phrases")
    fig.suptitle("Confidence distribution: correct (green) vs. wrong (red) predictions",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] discovering predictions in {args.final_dir}/predictions/")
    models = discover_models(args.final_dir)
    print(f"  Found {len(models)} model configs:")
    for name in models:
        print(f"    - {name}")

    print(f"\n[plot] 07_per_length_heatmap")
    plot_per_length_heatmap(models, args.out_dir / "07_per_length_heatmap.png")

    print(f"[plot] 08_seed_consistency")
    plot_seed_consistency(args.final_dir, args.out_dir / "08_seed_consistency.png")

    print(f"[plot] 09_negation_handling")
    plot_negation_handling(models, args.out_dir / "09_negation_handling.png")

    print(f"[plot] 10_confidence_distribution")
    plot_confidence_distribution(models, args.out_dir / "10_confidence_distribution.png")

    print(f"\n[done] Extra figures in {args.out_dir}/")


if __name__ == "__main__":
    main()
