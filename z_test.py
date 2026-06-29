import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import norm, chi2_contingency

# Data
years = [2023, 2024, 2025]

# Denominators per year (the base N for the proportions you're testing)
denoms = {
    2023: 10021,
    2024: 7975,
    2025: 6509
}

# Numerators
counts = {
    "Pass p1":        {2023: 551, 2024: 759, 2025: 680},
    "Not started p2": {2023: 274, 2024: 420, 2025: 348},
    "Not passed p2":  {2023: 72,  2024: 125, 2025: 109},
    "Passed p2":      {2023: 205, 2024: 214, 2025: 223},
}




def two_prop_ztest(x1, n1, x2, n2):
    """
    Two-sided two-proportion z-test using pooled standard error.
    Returns: p1, p2, diff (p2-p1), z, pvalue
    """
    p1 = x1 / n1
    p2 = x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
    if se == 0:
        z = np.nan
        pval = np.nan
    else:
        z = (p2 - p1) / se
        pval = 2 * (1 - norm.cdf(abs(z)))
    return p1, p2, (p2 - p1), z, pval


def overall_chi_square(counts_by_year, denoms_by_year, years):
    """
    For one category, test whether the proportion differs across years.
    3x2 table: Year x {in category, not in category}
    Returns: chi2, dof, pvalue
    """
    table = []
    for y in years:
        x = counts_by_year[y]
        n = denoms_by_year[y]
        table.append([x, n - x])
    chi2, p, dof, expected = chi2_contingency(table, correction=False)
    return chi2, dof, p


pairwise = list(combinations(years, 2))

rows = []
for metric, cts in counts.items():
    # overall chi-square across 3 years
    chi2, dof, p_overall = overall_chi_square(cts, denoms, years)

    for (y1, y2) in pairwise:
        x1, n1 = cts[y1], denoms[y1]
        x2, n2 = cts[y2], denoms[y2]
        p1, p2, diff, z, pval = two_prop_ztest(x1, n1, x2, n2)

        rows.append({
            "metric": metric,
            "comparison": f"{y1} vs {y2}",
            "x1": x1, "n1": n1, "p1": p1,
            "x2": x2, "n2": n2, "p2": p2,
            "diff_p2_minus_p1": diff,
            "z": z,
            "p": pval,
            "overall_chi2_p": p_overall
        })

df = pd.DataFrame(rows)

# ----------------------------
# 4) Multiple-comparisons adjustments
# ----------------------------

# (A) Bonferroni across ALL pairwise tests (metrics * 3 comparisons)
m_all = len(df)
df["p_bonf_all"] = np.minimum(df["p"] * m_all, 1.0)

# (B) Bonferroni WITHIN each metric (3 comparisons per metric)
df["p_bonf_within_metric"] = df.groupby("metric")["p"].transform(
    lambda s: np.minimum(s * len(s), 1.0)
)

# Output

out = df.copy()
out["p1_pct"] = (out["p1"] * 100).round(2)
out["p2_pct"] = (out["p2"] * 100).round(2)
out["diff_pp"] = (out["diff_p2_minus_p1"] * 100).round(2)  # percentage points
out["z"] = out["z"].round(3)
out["p"] = out["p"].map(lambda x: f"{x:.4g}")
out["p_bonf_all"] = out["p_bonf_all"].map(lambda x: f"{x:.4g}")
out["p_bonf_within_metric"] = out["p_bonf_within_metric"].map(lambda x: f"{x:.4g}")
out["overall_chi2_p"] = out["overall_chi2_p"].map(lambda x: f"{x:.4g}")

cols = [
    "metric", "comparison",
    "x1", "n1", "p1_pct",
    "x2", "n2", "p2_pct",
    "diff_pp", "z", "p",
    "p_bonf_within_metric", "p_bonf_all",
    "overall_chi2_p"
]

print("\nPairwise two-proportion z-tests (two-sided):\n")
print(out[cols].to_string(index=False))

