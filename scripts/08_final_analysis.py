#!/usr/bin/env python3
"""Phase 6 — Phân tích & sinh plot cho luận văn.

Đầu vào:
    results/final/test_metrics_quantum.json
    results/final/test_metrics_classical.json
    results/final/predictions/{model}/predictions.csv

Đầu ra trong results/final/plots/:
    01_accuracy_comparison.png    — bar chart all models
    02_per_length_accuracy.png    — line chart acc theo độ dài câu (2,3,4,5)
    03_param_efficiency.png       — scatter #params vs acc
    04_confusion_matrices.png     — 2x3 grid heatmaps cho top-6 model
    05_error_categories.png       — bar chart % lỗi theo loại
    06_quantum_layers_ablation.png— acc vs n_layers cho mỗi (reader, ansatz)
    mcnemar_results.txt           — McNemar test pairs

Cách dùng:
    python scripts/08_final_analysis.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


DEFAULT_FINAL = Path("results/final")
NEGATION_TOKENS = {"not", "n't", "no", "never", "nor", "none", "nothing", "nobody"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--final-dir", type=Path, default=DEFAULT_FINAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_FINAL / "plots")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def load_metrics(final_dir: Path) -> tuple[list[dict], list[dict]]:
    classical = []
    quantum = []
    cp = final_dir / "test_metrics_classical.json"
    qp = final_dir / "test_metrics_quantum.json"
    if cp.is_file():
        with cp.open("r", encoding="utf-8") as f:
            classical = json.load(f)
    if qp.is_file():
        with qp.open("r", encoding="utf-8") as f:
            quantum = json.load(f)
    return classical, quantum


def load_predictions(model_dir: Path) -> pd.DataFrame | None:
    path = model_dir / "predictions.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_accuracy_comparison(
    classical: list[dict], quantum: list[dict], out_path: Path
) -> None:
    """Bar chart so sánh test accuracy of all models (mean ± std qua seeds)."""
    rows: list[dict] = []
    # Classical aggregate
    fams: dict[str, list[float]] = {}
    for r in classical:
        fams.setdefault(r["model_family"], []).append(r["test_metrics"]["accuracy"])
    for fam, accs in fams.items():
        rows.append({"model": fam, "kind": "Classical", "mean": np.mean(accs), "std": np.std(accs)})

    # Quantum aggregate by (reader, ansatz, layers)
    qgroups: dict[str, list[float]] = {}
    for r in quantum:
        key = f"Q-{r['reader']}-{r['ansatz']}-L{r['n_layers']}"
        qgroups.setdefault(key, []).append(r["test_metrics"]["accuracy"])
    for k, accs in qgroups.items():
        rows.append({"model": k, "kind": "Quantum", "mean": np.mean(accs), "std": np.std(accs)})

    # Add majority baseline
    rows.append({"model": "Majority", "kind": "Baseline", "mean": 0.5, "std": 0.0})

    df = pd.DataFrame(rows).sort_values("mean", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(5, len(df) * 0.45)))
    colors = {"Classical": "#3a7ca5", "Quantum": "#d9534f", "Baseline": "#888"}
    bars = ax.barh(
        df["model"], df["mean"],
        xerr=df["std"],
        color=[colors[k] for k in df["kind"]],
        alpha=0.85, capsize=3, edgecolor="white", linewidth=0.5,
    )
    # Value labels INSIDE the bars (right-aligned, white text) to avoid overlap
    for bar, mean in zip(bars, df["mean"]):
        # Place label inside bar if bar is long enough, otherwise outside
        if mean >= 0.6:
            ax.text(mean - 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{mean:.3f}", va="center", ha="right",
                    fontsize=8.5, color="white", fontweight="bold")
        else:
            ax.text(mean + 0.005, bar.get_y() + bar.get_height() / 2,
                    f"{mean:.3f}", va="center", ha="left", fontsize=8.5)
    ax.set_xlim(0.45, 1.0)
    ax.set_xlabel("Test accuracy")
    ax.set_title("Test accuracy comparison across all models (mean $\\pm$ std)")
    ax.axvline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.tick_params(axis="y", labelsize=9)
    from matplotlib.patches import Patch
    legend_elems = [Patch(color=v, label=k) for k, v in colors.items()]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_per_length_accuracy(
    final_dir: Path, model_dirs: dict[str, Path], out_path: Path
) -> None:
    """Per-length accuracy: acc theo số từ trong câu."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for model_name, mdir in model_dirs.items():
        df = load_predictions(mdir)
        if df is None:
            continue
        df["n_words"] = df["text"].astype(str).str.split().str.len()
        accs = df.groupby("n_words").apply(
            lambda g: (g["pred"] == g["true_label"]).mean()
        )
        ax.plot(accs.index, accs.values, marker="o", label=model_name, linewidth=2)
    ax.set_xlabel("Phrase length (number of tokens)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Test accuracy as a function of phrase length")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_param_efficiency(
    classical: list[dict], quantum: list[dict], out_path: Path
) -> None:
    """Scatter #params vs accuracy. Aggregates by config + annotates carefully."""
    # Aggregate classical by model_family
    fam_classical: dict[str, list] = {}
    for r in classical:
        fam_classical.setdefault(r["model_family"], []).append(r)
    classical_points = []
    for fam, members in fam_classical.items():
        mean_p = float(np.mean([m["n_params"] for m in members]))
        mean_a = float(np.mean([m["test_metrics"]["accuracy"] for m in members]))
        classical_points.append((fam, mean_p, mean_a))

    # Aggregate quantum by (reader, ansatz, layers)
    fam_quantum: dict[str, list] = {}
    for r in quantum:
        key = f"Q-{r['reader'][:3]}-{r['ansatz'][:3]}-L{r['n_layers']}"
        fam_quantum.setdefault(key, []).append(r)
    quantum_points = []
    for key, members in fam_quantum.items():
        mean_p = float(np.mean([m["n_params"] for m in members]))
        mean_a = float(np.mean([m["test_metrics"]["accuracy"] for m in members]))
        quantum_points.append((key, mean_p, mean_a))

    fig, ax = plt.subplots(figsize=(10, 6.5))

    # Plot aggregated points only (one per config)
    for fam, p, a in classical_points:
        ax.scatter(p, a, marker="o", s=130, c="#3a7ca5", alpha=0.85,
                   edgecolor="white", linewidth=1, zorder=3)
        ax.annotate(fam, (p, a), fontsize=9.5, xytext=(10, 7),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="#3a7ca5", alpha=0.85))

    # Quantum points - dedupe Spiders (all 6 identical) into one cluster
    spider_points = [(k, p, a) for k, p, a in quantum_points if "spi" in k.lower()]
    cup_points = [(k, p, a) for k, p, a in quantum_points if "cup" in k.lower()]

    # Spiders: single annotated point (all 6 configs collapse to same value)
    if spider_points:
        p, a = spider_points[0][1], spider_points[0][2]
        ax.scatter(p, a, marker="^", s=180, c="#d9534f", alpha=0.85,
                   edgecolor="white", linewidth=1, zorder=3)
        ax.annotate("Q-spiders (all 6)", (p, a), fontsize=9.5,
                    xytext=(10, -14), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="#d9534f", alpha=0.85))

    # Cups: individual annotations, staggered
    cup_points_sorted = sorted(cup_points, key=lambda x: x[1])  # sort by params
    offsets = [(10, 10), (10, -18), (10, 10), (10, -18), (10, 10), (10, -18)]
    for i, (key, p, a) in enumerate(cup_points_sorted):
        ax.scatter(p, a, marker="^", s=130, c="#d9534f", alpha=0.75,
                   edgecolor="white", linewidth=1, zorder=3)
        ax.annotate(key, (p, a), fontsize=8.5,
                    xytext=offsets[i % len(offsets)],
                    textcoords="offset points", alpha=0.9,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white",
                              ec="#d9534f", alpha=0.7))

    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters (log scale)")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Parameter efficiency: trainable parameters vs.\\ accuracy")
    ax.grid(alpha=0.3, which="both")
    ax.set_ylim(0.55, 1.0)

    from matplotlib.lines import Line2D
    legend_elems = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3a7ca5",
               markersize=11, label="Classical baseline"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#d9534f",
               markersize=11, label="Quantum (DisCoCat)"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrices(
    model_dirs: dict[str, Path], out_path: Path, max_models: int = 6
) -> None:
    """Grid 2x3 confusion matrices cho top models."""
    items = list(model_dirs.items())[:max_models]
    n = len(items)
    rows = 2 if n > 3 else 1
    cols = min(3, n)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx, (name, mdir) in enumerate(items):
        r, c = idx // cols, idx % cols
        ax = axes[r][c]
        df = load_predictions(mdir)
        if df is None:
            ax.set_visible(False)
            continue
        cm = pd.crosstab(df["true_label"], df["pred"]).reindex(
            index=[0, 1], columns=[0, 1], fill_value=0
        )
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
            xticklabels=["NEG", "POS"], yticklabels=["NEG", "POS"],
        )
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

    # Hide unused axes
    for idx in range(len(items), rows * cols):
        r, c = idx // cols, idx % cols
        axes[r][c].set_visible(False)

    fig.suptitle("Confusion matrices for representative top models", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_error_categories(
    model_dirs: dict[str, Path], out_path: Path
) -> None:
    """% error theo category: negation / no negation / short / long."""
    rows = []
    for model_name, mdir in model_dirs.items():
        df = load_predictions(mdir)
        if df is None:
            continue
        df["n_words"] = df["text"].astype(str).str.split().str.len()
        df["has_negation"] = df["text"].astype(str).apply(
            lambda t: any(tok in NEGATION_TOKENS for tok in t.lower().split())
        )
        df["wrong"] = df["pred"] != df["true_label"]
        for label, mask in [
            ("All", pd.Series([True] * len(df), index=df.index)),
            ("Negation", df["has_negation"]),
            ("No negation", ~df["has_negation"]),
            ("Short (≤3 tokens)", df["n_words"] <= 3),
            ("Long (4-5 tokens)", df["n_words"] >= 4),
        ]:
            sub = df[mask]
            if len(sub) == 0:
                continue
            rows.append({"model": model_name, "category": label,
                         "error_rate": sub["wrong"].mean(), "n": len(sub)})

    df = pd.DataFrame(rows)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    pivot = df.pivot(index="category", columns="model", values="error_rate")
    pivot.plot(kind="bar", ax=ax, width=0.8)
    ax.set_ylabel("Error rate")
    ax.set_xlabel("Phrase category")
    ax.set_title("Per-category error rate (lower is better)")
    ax.legend(loc="upper right", fontsize=9)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_quantum_layers_ablation(quantum: list[dict], out_path: Path) -> None:
    """Acc vs n_layers cho mỗi (reader, ansatz)."""
    rows = []
    for r in quantum:
        rows.append({
            "reader": r["reader"],
            "ansatz": r["ansatz"],
            "n_layers": r["n_layers"],
            "acc": r["test_metrics"]["accuracy"],
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for (reader, ansatz), grp in df.groupby(["reader", "ansatz"]):
        means = grp.groupby("n_layers")["acc"].mean()
        stds = grp.groupby("n_layers")["acc"].std()
        label = f"{reader}-{ansatz}"
        ax.errorbar(means.index, means.values, yerr=stds.values,
                    marker="o", label=label, linewidth=2, capsize=4)
    ax.set_xlabel("Ansatz layer depth $L$")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Layer-depth ablation grouped by (reader, ansatz)")
    ax.set_xticks([1, 2, 3])
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# McNemar test
# ---------------------------------------------------------------------------


def mcnemar_test(preds_a: np.ndarray, preds_b: np.ndarray, truth: np.ndarray) -> dict:
    """McNemar test giữa 2 model predictions.

    Returns dict với: b (a sai b đúng), c (a đúng b sai), p_value.
    """
    a_correct = preds_a == truth
    b_correct = preds_b == truth
    b = int(((~a_correct) & b_correct).sum())  # a sai, b đúng
    c = int((a_correct & (~b_correct)).sum())  # a đúng, b sai
    n = b + c
    if n == 0:
        return {"b": b, "c": c, "p_value": 1.0, "stat": 0.0, "n_discordant": 0}
    # Continuity-corrected McNemar chi-square
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    # Approx p-value from chi2(df=1)
    try:
        from scipy.stats import chi2 as chi2_dist

        p = 1.0 - chi2_dist.cdf(chi2, df=1)
    except ImportError:
        # Crude approximation
        p = np.exp(-chi2 / 2)
    return {"b": b, "c": c, "p_value": float(p), "stat": float(chi2), "n_discordant": n}


def run_mcnemar_pairs(
    model_dirs: dict[str, Path], out_path: Path
) -> None:
    """So sánh từng cặp model trên cùng test set."""
    # Load predictions cho mỗi model
    preds_per_model: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name, mdir in model_dirs.items():
        df = load_predictions(mdir)
        if df is None:
            continue
        preds_per_model[name] = (
            df["pred"].to_numpy(),
            df["true_label"].to_numpy(),
        )

    models = list(preds_per_model.keys())
    lines = ["McNemar test (paired) — pairs có p<0.05 → khác biệt có ý nghĩa thống kê", ""]
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            pa, ta = preds_per_model[a]
            pb, tb = preds_per_model[b]
            # Đảm bảo same truth
            if len(pa) != len(pb) or not (ta == tb).all():
                lines.append(f"{a:<35} vs {b:<35}  — SKIP (different test sets)")
                continue
            res = mcnemar_test(pa, pb, ta)
            sig = "***" if res["p_value"] < 0.001 else "**" if res["p_value"] < 0.01 else "*" if res["p_value"] < 0.05 else ""
            lines.append(
                f"{a:<35} vs {b:<35}  "
                f"b={res['b']:>3d}  c={res['c']:>3d}  "
                f"chi2={res['stat']:>6.2f}  p={res['p_value']:.4g}  {sig}"
            )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[save] {out_path}")


# ---------------------------------------------------------------------------
# Pick top models for detailed plots
# ---------------------------------------------------------------------------


def pick_top_models(final_dir: Path, top_k: int = 6) -> dict[str, Path]:
    """Chọn 6 model tiêu biểu cho confusion matrix + per-length.

    Strategy: 1 classical đại diện mỗi family + best quantum config.
    """
    pred_root = final_dir / "predictions"
    if not pred_root.is_dir():
        return {}

    # Lấy seed 0 cho mỗi model family
    out: dict[str, Path] = {}

    # Classical: tfidf, glove, bilstm — seed 0 only
    for fam, prefix in [("LR + TF-IDF", "lr_tfidf"), ("LR + GloVe", "lr_glove"), ("BiLSTM", "bilstm")]:
        cand = pred_root / f"{prefix}_seed0"
        if cand.is_dir():
            out[fam] = cand

    # Quantum: pick all unique (reader, ansatz, L) — seed 0 each
    pattern = re.compile(r"quantum_(?P<reader>\w+)_(?P<ansatz>\w+)_L(?P<L>\d+)_seed0$")
    quantum_dirs = [d for d in pred_root.iterdir() if d.is_dir() and pattern.match(d.name)]
    # Sort + take some
    for d in sorted(quantum_dirs):
        m = pattern.match(d.name)
        if m:
            label = f"Q-{m['reader']}-{m['ansatz']}-L{m['L']}"
            out[label] = d
        if len(out) >= top_k:
            break

    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    classical, quantum = load_metrics(args.final_dir)
    print(f"[load] classical: {len(classical)} entries, quantum: {len(quantum)} entries")

    if not classical and not quantum:
        sys.exit(f"[lỗi] Không có dữ liệu. Chạy: python scripts/07_evaluate_test.py")

    print("\n[plot] 01_accuracy_comparison")
    plot_accuracy_comparison(classical, quantum, args.out_dir / "01_accuracy_comparison.png")

    print("[plot] 03_param_efficiency")
    plot_param_efficiency(classical, quantum, args.out_dir / "03_param_efficiency.png")

    print("[plot] 06_quantum_layers_ablation")
    plot_quantum_layers_ablation(quantum, args.out_dir / "06_quantum_layers_ablation.png")

    # Predictions-based plots
    top_models = pick_top_models(args.final_dir, top_k=6)
    if top_models:
        print(f"\n[top models] {list(top_models.keys())}")
        print("[plot] 02_per_length_accuracy")
        plot_per_length_accuracy(args.final_dir, top_models, args.out_dir / "02_per_length_accuracy.png")
        print("[plot] 04_confusion_matrices")
        plot_confusion_matrices(top_models, args.out_dir / "04_confusion_matrices.png")
        print("[plot] 05_error_categories")
        plot_error_categories(top_models, args.out_dir / "05_error_categories.png")
        print("[mcnemar]")
        run_mcnemar_pairs(top_models, args.final_dir / "mcnemar_results.txt")

    print(f"\n[done] Plots ở: {args.out_dir}/")
    print(f"        Copy về local:")
    print(f"          scp -r user@remote:~/quantum_project/results/final ./results/")


if __name__ == "__main__":
    main()
