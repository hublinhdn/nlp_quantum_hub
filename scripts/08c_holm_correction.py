"""Apply Holm-Bonferroni multiple-comparisons correction to the McNemar
p-values reported in Table 4.

The Holm-Bonferroni procedure controls the family-wise error rate (FWER)
without the conservatism of straight Bonferroni when many tests are run:
- Sort raw p-values ascending: p_{(1)} <= p_{(2)} <= ... <= p_{(m)}.
- Adjusted p_{(k)} = min(1, max over j<=k of  p_{(j)} * (m - j + 1)).
- Reject H_0 for test k iff adjusted p_{(k)} <= alpha.

For p ~ 10^{-15} we report the adjusted as <10^{-14} (still significant
under any reasonable FWER target).
"""
from __future__ import annotations
import math


# (label, raw p) — same order as Table 4 of the paper
TESTS: list[tuple[str, float]] = [
    # Classical
    ("LR+TF-IDF       vs BiLSTM",            0.20),
    # Spiders block (3)
    ("Q-spiders       vs BiLSTM",            0.5959),
    ("Q-spiders       vs LR+TF-IDF",         0.04461),
    # Cups+Sim14-L3 block (3)
    ("Q-cups-Sim14-L3 vs BiLSTM",            0.6434),
    ("Q-cups-Sim14-L3 vs LR+TF-IDF",         0.06675),
    ("Q-cups-Sim14-L3 vs Q-spiders",         0.8597),
    # Cups+IQP block (6)
    ("LR+TF-IDF       vs Q-cups-IQP-L1",     0.0036),
    ("LR+TF-IDF       vs Q-cups-IQP-L2",     1e-15),
    ("LR+TF-IDF       vs Q-cups-IQP-L3",     0.00025),
    ("LR+GloVe        vs Q-cups-IQP-L1",     0.043),
    ("BiLSTM          vs Q-cups-IQP-L1",     0.066),
    ("BiLSTM          vs Q-cups-IQP-L2",     1e-15),
    ("BiLSTM          vs Q-cups-IQP-L3",     0.020),
    # Within Cups+IQP (3)
    ("L1              vs L2 (Cups+IQP)",     1e-15),
    ("L1              vs L3 (Cups+IQP)",     0.79),
    ("L2              vs L3 (Cups+IQP)",     1e-15),
]

m = len(TESTS)
print(f"# Holm-Bonferroni correction over m = {m} McNemar comparisons\n")

# Sort by raw p ascending, remember original index
indexed = list(enumerate(TESTS))
indexed.sort(key=lambda kv: kv[1][1])

# Compute Holm-adjusted p-values
adjusted = [None] * m  # one slot per original index
running_max = 0.0
for k, (orig_idx, (label, raw_p)) in enumerate(indexed):
    multiplier = m - k  # for k=0 multiplier=m, for k=m-1 multiplier=1
    cand = min(1.0, raw_p * multiplier)
    running_max = max(running_max, cand)
    adjusted[orig_idx] = running_max

# Print Table 4 with both columns
print(f"{'Comparison':<40} {'raw p':>10} {'Holm-adj':>12}  Sig(0.05)")
print("-" * 80)
for orig_idx, (label, raw_p) in enumerate(TESTS):
    adj = adjusted[orig_idx]
    if raw_p < 1e-14:
        raw_str = "<10^-15"
        adj_str = "<10^-14"
    else:
        raw_str = f"{raw_p:.4g}"
        adj_str = f"{adj:.4g}" if adj > 1e-14 else "<10^-14"
    sig = "yes" if adj < 0.05 else "no"
    print(f"{label:<40} {raw_str:>10} {adj_str:>12}  {sig}")

print()
print("# Headline impact:")
print("# - Spiders vs LR+TF-IDF raw p = 0.045 → after Holm correction loses significance.")
print("# - All p < 10^-15 results (Cups+IQP L2 collapse) survive any FWER target.")
print("# - LR+TF-IDF vs Q-cups-IQP-{L1,L3} survive (p < 0.01 raw, adj < 0.05).")
