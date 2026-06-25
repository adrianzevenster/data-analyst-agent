"""Statistical hypothesis testing suite.

No new dependencies — scipy.stats and numpy only. Supports:
- Two-sample t-test (Welch's, unequal variance)
- One-sample t-test
- Paired t-test
- Mann-Whitney U (non-parametric two-sample)
- Chi-squared independence test
- One-way ANOVA
- Pearson / Spearman correlation test
- Power analysis for two-sample comparisons (Cohen's formula)
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    if na + nb <= 2:
        return 0.0
    pooled = np.sqrt(((na - 1) * a.std(ddof=1) ** 2 + (nb - 1) * b.std(ddof=1) ** 2) / (na + nb - 2))
    return float((a.mean() - b.mean()) / max(pooled, 1e-12))


def _cramers_v(chi2: float, n: int, r: int, c: int) -> float:
    return float(np.sqrt(chi2 / (n * (min(r, c) - 1)))) if n > 0 and min(r, c) > 1 else 0.0


def _eta_squared(groups: list[np.ndarray]) -> float:
    all_vals = np.concatenate(groups)
    grand_mean = all_vals.mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = sum((v - grand_mean) ** 2 for v in all_vals)
    return float(ss_between / ss_total) if ss_total > 0 else 0.0


def _power_two_sample(d: float, n_per_group: int, alpha: float = 0.05) -> float:
    """Achieved power for a two-sided two-sample t-test."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    ncp = abs(d) * np.sqrt(n_per_group / 2)
    return float(1 - stats.norm.cdf(z_alpha - ncp) + stats.norm.cdf(-z_alpha - ncp))


def _required_n(d: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Required n per group for a two-sided two-sample t-test (Cohen's formula)."""
    z_alpha = stats.norm.ppf(1 - alpha / 2)
    z_beta = stats.norm.ppf(power)
    return int(np.ceil(2 * ((z_alpha + z_beta) / max(abs(d), 1e-9)) ** 2))


def _extract_two_groups(
    df: pd.DataFrame,
    col_a: str,
    group_col: str | None,
    group_a: str | None,
    group_b: str | None,
    col_b: str | None,
) -> tuple[np.ndarray, np.ndarray, str, str]:
    """Return (arr_a, arr_b, label_a, label_b) for a two-group comparison."""
    if group_col and group_col in df.columns and col_a and col_a in df.columns:
        uniq = df[group_col].dropna().unique()
        if group_a and group_b:
            ga, gb = str(group_a), str(group_b)
        elif len(uniq) >= 2:
            ga, gb = str(uniq[0]), str(uniq[1])
        else:
            raise ValueError(f"'{group_col}' has fewer than 2 unique values.")
        arr_a = pd.to_numeric(df.loc[df[group_col].astype(str) == ga, col_a], errors="coerce").dropna().values
        arr_b = pd.to_numeric(df.loc[df[group_col].astype(str) == gb, col_a], errors="coerce").dropna().values
        return arr_a, arr_b, f"{col_a}[{ga}]", f"{col_a}[{gb}]"
    elif col_a and col_b and col_a in df.columns and col_b in df.columns:
        arr_a = pd.to_numeric(df[col_a], errors="coerce").dropna().values
        arr_b = pd.to_numeric(df[col_b], errors="coerce").dropna().values
        return arr_a, arr_b, col_a, col_b
    else:
        raise ValueError("Provide either (col_a + group_col) or (col_a + col_b) for group comparison.")


def hypothesis_test(
    df: pd.DataFrame,
    test_type: Literal[
        "two_sample_t", "one_sample_t", "paired_t",
        "mannwhitney", "chi_squared", "anova",
        "correlation", "power_analysis",
    ] = "two_sample_t",
    col_a: str | None = None,
    col_b: str | None = None,
    group_col: str | None = None,
    group_a: str | None = None,
    group_b: str | None = None,
    alpha: float = 0.05,
    popmean: float = 0.0,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
    effect_size: float | None = None,
    n_obs: int | None = None,
    target_power: float = 0.8,
) -> dict:
    """Run a statistical hypothesis test or power analysis.

    Parameters
    ----------
    test_type : which test to run
    col_a     : primary numeric (or categorical for chi_squared) column
    col_b     : second column — numeric for paired/correlation, categorical for chi_squared
    group_col : grouping column (for two_sample_t, mannwhitney, anova)
    group_a/b : specific group values to compare (optional; defaults to first two unique values)
    alpha     : significance level (default 0.05)
    popmean   : population mean for one_sample_t
    alternative: "two-sided", "less", or "greater"
    effect_size: Cohen's d for power_analysis (required if not computing from data)
    n_obs     : n per group for power_analysis (if given, computes achieved power)
    target_power: desired power for power_analysis (used to compute required n)
    """
    result: dict = {
        "test_type": test_type,
        "alpha": alpha,
        "alternative": alternative,
    }

    # ── Power analysis (doesn't need df) ──────────────────────────────────
    if test_type == "power_analysis":
        d = effect_size
        if d is None:
            # Try to compute from data
            if col_a and col_b and col_a in df.columns and col_b in df.columns:
                arr_a = pd.to_numeric(df[col_a], errors="coerce").dropna().values
                arr_b = pd.to_numeric(df[col_b], errors="coerce").dropna().values
                d = _cohens_d(arr_a, arr_b)
            elif group_col and col_a:
                try:
                    arr_a, arr_b, _, _ = _extract_two_groups(df, col_a, group_col, group_a, group_b, None)
                    d = _cohens_d(arr_a, arr_b)
                except ValueError:
                    pass
        if d is None:
            return {"error": "Provide effect_size (Cohen's d) or col_a+col_b to compute it from data."}

        d_abs = abs(d)
        if n_obs:
            achieved = _power_two_sample(d_abs, n_obs, alpha)
            req_n = _required_n(d_abs, alpha, target_power)
            readout = (
                f"With n={n_obs} per group and d={d_abs:.3f}: achieved power = {achieved:.1%}. "
                f"To reach {target_power:.0%} power, you need {req_n} observations per group."
            )
            result.update({
                "cohens_d": round(d_abs, 4),
                "n_per_group": n_obs,
                "achieved_power": round(achieved, 4),
                "required_n_for_target_power": req_n,
                "target_power": target_power,
                "engineering_readout": readout,
            })
        else:
            req_n = _required_n(d_abs, alpha, target_power)
            achieved = _power_two_sample(d_abs, req_n, alpha)
            # Power curve at common sample sizes
            curve = {str(n): round(_power_two_sample(d_abs, n, alpha), 3) for n in [10, 20, 50, 100, 200, 500]}
            readout = (
                f"Effect size d={d_abs:.3f}: need {req_n} observations per group "
                f"({req_n * 2} total) for {target_power:.0%} power at α={alpha}."
            )
            result.update({
                "cohens_d": round(d_abs, 4),
                "required_n_per_group": req_n,
                "required_n_total": req_n * 2,
                "target_power": target_power,
                "power_curve": curve,
                "engineering_readout": readout,
            })
        return result

    # ── Two-sample t-test (Welch's) ───────────────────────────────────────
    if test_type == "two_sample_t":
        try:
            arr_a, arr_b, label_a, label_b = _extract_two_groups(df, col_a or "", group_col, group_a, group_b, col_b)
        except ValueError as e:
            return {"error": str(e)}
        if len(arr_a) < 2 or len(arr_b) < 2:
            return {"error": "Each group needs at least 2 observations."}
        stat, p = stats.ttest_ind(arr_a, arr_b, equal_var=False, alternative=alternative)
        d = _cohens_d(arr_a, arr_b)
        ci = stats.t.interval(1 - alpha, df=len(arr_a) + len(arr_b) - 2,
                               loc=arr_a.mean() - arr_b.mean(),
                               scale=np.sqrt(arr_a.var(ddof=1) / len(arr_a) + arr_b.var(ddof=1) / len(arr_b)))
        sig = p < alpha
        readout = (
            f"Welch's t-test: {label_a} (mean={arr_a.mean():.4f}, n={len(arr_a)}) vs "
            f"{label_b} (mean={arr_b.mean():.4f}, n={len(arr_b)}). "
            f"t={stat:.4f}, p={p:.4f} ({'significant' if sig else 'not significant'} at α={alpha}). "
            f"Cohen's d={d:.3f}, {alpha:.0%} CI for mean diff: [{ci[0]:.4f}, {ci[1]:.4f}]."
        )
        result.update({
            "group_a": label_a, "group_b": label_b,
            "mean_a": round(float(arr_a.mean()), 4), "mean_b": round(float(arr_b.mean()), 4),
            "n_a": len(arr_a), "n_b": len(arr_b),
            "t_statistic": round(float(stat), 4), "p_value": round(float(p), 6),
            "significant": sig, "cohens_d": round(d, 4),
            "ci_lower": round(float(ci[0]), 4), "ci_upper": round(float(ci[1]), 4),
            "engineering_readout": readout,
        })
        return result

    # ── One-sample t-test ─────────────────────────────────────────────────
    if test_type == "one_sample_t":
        if not col_a or col_a not in df.columns:
            return {"error": f"Column '{col_a}' not found."}
        arr = pd.to_numeric(df[col_a], errors="coerce").dropna().values
        if len(arr) < 2:
            return {"error": "Need at least 2 observations."}
        stat, p = stats.ttest_1samp(arr, popmean=popmean, alternative=alternative)
        d = (arr.mean() - popmean) / max(arr.std(ddof=1), 1e-12)
        sig = p < alpha
        readout = (
            f"One-sample t-test: '{col_a}' mean={arr.mean():.4f} vs μ₀={popmean}. "
            f"t={stat:.4f}, p={p:.4f} ({'significant' if sig else 'not significant'} at α={alpha}). "
            f"Cohen's d={d:.3f}."
        )
        result.update({
            "col": col_a, "sample_mean": round(float(arr.mean()), 4),
            "population_mean": popmean, "n": len(arr),
            "t_statistic": round(float(stat), 4), "p_value": round(float(p), 6),
            "significant": sig, "cohens_d": round(d, 4),
            "engineering_readout": readout,
        })
        return result

    # ── Paired t-test ─────────────────────────────────────────────────────
    if test_type == "paired_t":
        if not col_a or not col_b or col_a not in df.columns or col_b not in df.columns:
            return {"error": "Provide col_a and col_b for a paired t-test."}
        paired = df[[col_a, col_b]].dropna()
        if len(paired) < 2:
            return {"error": "Need at least 2 paired observations."}
        a = pd.to_numeric(paired[col_a], errors="coerce").values
        b = pd.to_numeric(paired[col_b], errors="coerce").values
        stat, p = stats.ttest_rel(a, b, alternative=alternative)
        diff = a - b
        d = diff.mean() / max(diff.std(ddof=1), 1e-12)
        sig = p < alpha
        readout = (
            f"Paired t-test: '{col_a}' vs '{col_b}' (n={len(paired)} pairs). "
            f"Mean difference={diff.mean():.4f}. "
            f"t={stat:.4f}, p={p:.4f} ({'significant' if sig else 'not significant'} at α={alpha}). "
            f"Cohen's d={d:.3f}."
        )
        result.update({
            "col_a": col_a, "col_b": col_b, "n_pairs": len(paired),
            "mean_diff": round(float(diff.mean()), 4),
            "t_statistic": round(float(stat), 4), "p_value": round(float(p), 6),
            "significant": sig, "cohens_d": round(d, 4),
            "engineering_readout": readout,
        })
        return result

    # ── Mann-Whitney U ────────────────────────────────────────────────────
    if test_type == "mannwhitney":
        try:
            arr_a, arr_b, label_a, label_b = _extract_two_groups(df, col_a or "", group_col, group_a, group_b, col_b)
        except ValueError as e:
            return {"error": str(e)}
        if len(arr_a) < 2 or len(arr_b) < 2:
            return {"error": "Each group needs at least 2 observations."}
        stat, p = stats.mannwhitneyu(arr_a, arr_b, alternative=alternative)
        r = 1 - (2 * stat) / (len(arr_a) * len(arr_b))  # rank-biserial correlation
        sig = p < alpha
        readout = (
            f"Mann-Whitney U: {label_a} (n={len(arr_a)}, median={np.median(arr_a):.4f}) vs "
            f"{label_b} (n={len(arr_b)}, median={np.median(arr_b):.4f}). "
            f"U={stat:.1f}, p={p:.4f} ({'significant' if sig else 'not significant'} at α={alpha}). "
            f"Rank-biserial r={r:.3f}."
        )
        result.update({
            "group_a": label_a, "group_b": label_b,
            "median_a": round(float(np.median(arr_a)), 4), "median_b": round(float(np.median(arr_b)), 4),
            "n_a": len(arr_a), "n_b": len(arr_b),
            "u_statistic": round(float(stat), 2), "p_value": round(float(p), 6),
            "significant": sig, "rank_biserial_r": round(float(r), 4),
            "engineering_readout": readout,
        })
        return result

    # ── Chi-squared independence test ─────────────────────────────────────
    if test_type == "chi_squared":
        if not col_a or not col_b or col_a not in df.columns or col_b not in df.columns:
            return {"error": "Provide col_a and col_b (both categorical) for chi-squared test."}
        ct = pd.crosstab(df[col_a], df[col_b])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            return {"error": "Need at least a 2×2 contingency table."}
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        n = int(ct.values.sum())
        v = _cramers_v(chi2, n, ct.shape[0], ct.shape[1])
        sig = p < alpha
        mag = "strong" if v > 0.5 else "moderate" if v > 0.3 else "weak" if v > 0.1 else "negligible"
        readout = (
            f"Chi-squared test of independence between '{col_a}' and '{col_b}' "
            f"({ct.shape[0]}×{ct.shape[1]} table, n={n}). "
            f"χ²={chi2:.4f}, df={dof}, p={p:.4f} ({'significant' if sig else 'not significant'} at α={alpha}). "
            f"Cramér's V={v:.3f} ({mag} association)."
        )
        result.update({
            "col_a": col_a, "col_b": col_b, "n": n,
            "table_shape": list(ct.shape),
            "chi2_statistic": round(float(chi2), 4), "dof": dof, "p_value": round(float(p), 6),
            "significant": sig, "cramers_v": round(v, 4), "association_strength": mag,
            "engineering_readout": readout,
        })
        return result

    # ── One-way ANOVA ─────────────────────────────────────────────────────
    if test_type == "anova":
        if not col_a or not group_col or col_a not in df.columns or group_col not in df.columns:
            return {"error": "Provide col_a (numeric) and group_col (categorical) for ANOVA."}
        groups = [
            pd.to_numeric(df.loc[df[group_col] == g, col_a], errors="coerce").dropna().values
            for g in df[group_col].dropna().unique()
        ]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            return {"error": "Need at least 2 groups with ≥2 observations each for ANOVA."}
        stat, p = stats.f_oneway(*groups)
        eta2 = _eta_squared(groups)
        sig = p < alpha
        group_labels = [str(g) for g in df[group_col].dropna().unique()]
        group_means = {str(g): round(float(arr.mean()), 4) for g, arr in zip(group_labels, groups)}
        mag = "large" if eta2 > 0.14 else "medium" if eta2 > 0.06 else "small"
        readout = (
            f"One-way ANOVA: '{col_a}' across {len(groups)} groups in '{group_col}'. "
            f"F={stat:.4f}, p={p:.4f} ({'significant' if sig else 'not significant'} at α={alpha}). "
            f"η²={eta2:.3f} ({mag} effect). "
            f"Group means: {', '.join(f'{k}={v}' for k, v in list(group_means.items())[:5])}."
        )
        result.update({
            "col": col_a, "group_col": group_col, "n_groups": len(groups),
            "group_means": group_means,
            "f_statistic": round(float(stat), 4), "p_value": round(float(p), 6),
            "significant": sig, "eta_squared": round(eta2, 4), "effect_magnitude": mag,
            "engineering_readout": readout,
        })
        return result

    # ── Correlation test ──────────────────────────────────────────────────
    if test_type == "correlation":
        if not col_a or not col_b or col_a not in df.columns or col_b not in df.columns:
            return {"error": "Provide col_a and col_b for a correlation test."}
        paired = df[[col_a, col_b]].dropna()
        if len(paired) < 3:
            return {"error": "Need at least 3 paired observations."}
        a = pd.to_numeric(paired[col_a], errors="coerce")
        b = pd.to_numeric(paired[col_b], errors="coerce")
        mask = a.notna() & b.notna()
        a, b = a[mask].values, b[mask].values
        pearson_r, pearson_p = stats.pearsonr(a, b)
        spearman_r, spearman_p = stats.spearmanr(a, b)
        sig_p = pearson_p < alpha
        sig_s = spearman_p < alpha
        readout = (
            f"Correlation test between '{col_a}' and '{col_b}' (n={len(a)}). "
            f"Pearson r={pearson_r:.4f} (p={pearson_p:.4f}, {'sig' if sig_p else 'n.s.'}). "
            f"Spearman ρ={spearman_r:.4f} (p={spearman_p:.4f}, {'sig' if sig_s else 'n.s.'})."
        )
        result.update({
            "col_a": col_a, "col_b": col_b, "n": len(a),
            "pearson_r": round(float(pearson_r), 4), "pearson_p": round(float(pearson_p), 6),
            "spearman_r": round(float(spearman_r), 4), "spearman_p": round(float(spearman_p), 6),
            "pearson_significant": sig_p, "spearman_significant": sig_s,
            "engineering_readout": readout,
        })
        return result

    return {"error": f"Unknown test_type: '{test_type}'."}
