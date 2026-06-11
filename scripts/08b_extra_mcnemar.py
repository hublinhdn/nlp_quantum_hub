"""Compute the McNemar pairs missing from Table 4 (review feedback).

Adds:
  - Q-spiders vs BiLSTM         (backs Abstract "indistinguishable" claim)
  - Q-spiders vs LR+TF-IDF      (backs "trailing 2.2 pts" claim)
  - Q-cups-Sim14-L3 vs BiLSTM   (best "true quantum" config vs neural baseline)
  - Q-cups-Sim14-L3 vs LR+TF-IDF
  - Q-cups-Sim14-L3 vs Q-spiders

Uses seed 0 per config (same convention as existing 08_final_analysis.py).
"""
from __future__ import annotations
import math
from pathlib import Path
import pandas as pd

PRED = Path("results/final/predictions")
SEED = 0


def load(config: str):
    """Load (preds, truth) for one config-seed."""
    df = pd.read_csv(PRED / f"{config}_seed{SEED}" / "predictions.csv")
    return df["pred"].to_numpy(), df["true_label"].to_numpy()


def chi2_p(stat: float) -> float:
    """1 - F_{chi2(df=1)}(stat).  Survival function via complementary error fn.

    F(x;1) = erf(sqrt(x/2)).  So 1-F = erfc(sqrt(x/2)).
    """
    return math.erfc(math.sqrt(stat / 2.0))


def mcnemar(a_preds, b_preds, truth):
    a_ok = a_preds == truth
    b_ok = b_preds == truth
    b_count = int(((~a_ok) & b_ok).sum())  # A wrong, B right
    c_count = int((a_ok & (~b_ok)).sum())  # A right, B wrong
    n = b_count + c_count
    if n == 0:
        return b_count, c_count, 0.0, 1.0
    stat = (abs(b_count - c_count) - 1) ** 2 / n
    return b_count, c_count, stat, chi2_p(stat)


def fmt_p(p: float) -> str:
    if p < 1e-15:
        return r"$<10^{-15}$"
    if p < 0.0001:
        return f"{p:.2e}"
    return f"{p:.4f}".rstrip("0").rstrip(".")


def sig(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


pairs = [
    # Backs Abstract "indistinguishable" claim (Spiders 92.05 vs BiLSTM 91.83)
    ("Q-spiders-iqp-L1", "BiLSTM", "quantum_spiders_iqp_L1", "bilstm"),
    # Backs Abstract "trailing 2.2 pts" claim (Spiders 92.05 vs LR+TF-IDF 94.25)
    ("Q-spiders-iqp-L1", "LR + TF-IDF", "quantum_spiders_iqp_L1", "lr_tfidf"),
    # Best "true quantum" config (Sim14 has entanglement) vs neural baseline
    ("Q-cups-sim14-L3", "BiLSTM", "quantum_cups_sim14_L3", "bilstm"),
    ("Q-cups-sim14-L3", "LR + TF-IDF", "quantum_cups_sim14_L3", "lr_tfidf"),
    # Compare best-Q-cups (Sim14, true quantum) vs Spiders (bag-of-words proxy)
    ("Q-cups-sim14-L3", "Q-spiders-iqp-L1", "quantum_cups_sim14_L3", "quantum_spiders_iqp_L1"),
]

print(f"# 5 McNemar pairs (seed {SEED})\n")
print(f"{'Model A':<20} {'Model B':<20} {'b':>4} {'c':>4} {'chi2':>8} {'p':>14}  Sig")
print("-" * 80)
rows_latex = []
for label_a, label_b, cfg_a, cfg_b in pairs:
    pa, ta = load(cfg_a)
    pb, tb = load(cfg_b)
    assert (ta == tb).all(), f"truth mismatch {cfg_a} vs {cfg_b}"
    b, c, stat, p = mcnemar(pa, pb, ta)
    s = sig(p)
    print(f"{label_a:<20} {label_b:<20} {b:>4d} {c:>4d} {stat:>8.2f} {p:>14.4g}  {s}")
    # LaTeX row
    rows_latex.append(
        f"{label_a:<20} & {label_b:<20} & {b:>3d} & {c:>3d} & {stat:>6.2f} & {fmt_p(p):<14} & {s:<3} \\\\"
    )

print("\n# LaTeX rows (paste into tab:mcnemar)\n")
for r in rows_latex:
    print(r)
