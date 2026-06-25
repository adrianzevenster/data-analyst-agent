"""Lightweight causal effect estimation using OLS regression.

No new dependencies: uses scipy.stats and numpy. Supports:
- Average Treatment Effect (ATE) via OLS with optional confounders
- E-value sensitivity analysis (how strong must an unmeasured confounder be to
  explain away the effect? — VanderWeele & Ding, 2017)
- Basic mediation decomposition (direct + indirect effect via intermediate)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from scipy import stats as _scipy_stats
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False


def _e_value(rr: float) -> float:
    """E-value for a risk-ratio-like effect size: minimum confounder strength to explain it away."""
    if rr <= 0 or abs(rr - 1.0) < 1e-9:
        return 1.0
    rr_abs = max(rr, 1.0 / rr)  # always > 1
    return rr_abs + (rr_abs * (rr_abs - 1.0)) ** 0.5


def estimate_causal_effect(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    control_cols: list[str] | None = None,
    mediation_col: str | None = None,
) -> dict:
    """Estimate the effect of treatment_col on outcome_col, controlling for confounders.

    Uses OLS regression with HC3 robust standard errors. Supports binary and
    continuous treatments. Returns ATE, CI, p-value, and E-value.

    For binary treatment: also computes Cohen's d and standardised effect.
    For mediation_col: decomposes effect into direct + indirect paths.
    """
    if not _SCIPY_AVAILABLE:
        return {"error": "scipy is not installed."}

    required = [treatment_col, outcome_col] + (control_cols or []) + ([mediation_col] if mediation_col else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        return {"error": f"Columns not found in dataset: {missing}"}

    # Drop rows with any missing values in the analysis columns
    cols = [treatment_col, outcome_col] + (control_cols or []) + ([mediation_col] if mediation_col else [])
    d = df[cols].dropna()
    if len(d) < 10:
        return {"error": "Fewer than 10 complete rows — insufficient data for causal analysis."}

    # Detect binary treatment
    treatment = d[treatment_col]
    is_binary = treatment.nunique() == 2
    outcome = d[outcome_col]

    # Build design matrix: [treatment, controls]
    X_cols = [treatment_col] + (control_cols or [])
    X_data = d[X_cols].copy()

    # Encode non-numeric controls with label encoding (best-effort)
    for col in X_data.columns:
        if not pd.api.types.is_numeric_dtype(X_data[col]):
            codes = X_data[col].astype("category").cat.codes
            X_data[col] = codes

    # Add intercept
    X = np.column_stack([np.ones(len(X_data))] + [X_data[c].values for c in X_data.columns])
    y = outcome.values.astype(float)

    # OLS via numpy (for speed; statsmodels not required)
    try:
        beta, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)
    except Exception as exc:
        return {"error": f"OLS fit failed: {exc}"}

    n, k = X.shape
    if n <= k:
        return {"error": "More predictors than observations — reduce controls."}

    y_hat = X @ beta
    resid = y - y_hat
    sse = float((resid ** 2).sum())
    mse = sse / (n - k)

    # HC3 robust covariance: Cov(beta) = (X'X)^-1 X' diag(e_i^2 / (1-h_ii)^2) X (X'X)^-1
    try:
        XtX_inv = np.linalg.inv(X.T @ X)
        H = X @ XtX_inv @ X.T
        h = np.diag(H)
        e2 = resid ** 2 / (1 - np.clip(h, 0, 0.9999)) ** 2
        S = np.diag(e2)
        cov_hc3 = XtX_inv @ (X.T @ S @ X) @ XtX_inv
    except Exception:
        # Fallback: classical OLS covariance
        cov_hc3 = mse * np.linalg.inv(X.T @ X)

    # Treatment coefficient is at index 1 (index 0 is intercept)
    ate = float(beta[1])
    se = float(np.sqrt(cov_hc3[1, 1]))
    t_stat = ate / max(se, 1e-12)
    df_resid = n - k
    p_value = float(2 * _scipy_stats.t.sf(abs(t_stat), df_resid))
    ci_lower = float(ate - 1.96 * se)
    ci_upper = float(ate + 1.96 * se)

    # R² and partial R²
    sst = float(((y - y.mean()) ** 2).sum()) or 1.0
    r2 = float(1 - sse / sst)

    # Effect size: standardised (Cohen's d for binary, Beta for continuous)
    outcome_std = float(y.std()) or 1.0
    if is_binary:
        groups = [d.loc[d[treatment_col] == v, outcome_col].values for v in sorted(d[treatment_col].unique())]
        pooled_std = float(np.sqrt(((len(groups[0]) - 1) * groups[0].std() ** 2 + (len(groups[1]) - 1) * groups[1].std() ** 2) / (len(groups[0]) + len(groups[1]) - 2))) or 1.0
        effect_size = ate / pooled_std
        effect_metric = "cohen_d"
    else:
        effect_size = ate * float(treatment.std()) / outcome_std
        effect_metric = "standardised_beta"

    # E-value: map effect size to approx RR via exp(0.91 * |d|) — Chinn (2000)
    rr_proxy = float(np.exp(0.91 * abs(effect_size)))
    e_value = float(_e_value(rr_proxy))

    # Mediation analysis (Baron & Kenny 3-equation approach)
    mediation_result: dict | None = None
    if mediation_col:
        try:
            m = d[mediation_col].values.astype(float)
            # Path a: treatment → mediator
            Xm = np.column_stack([np.ones(n), X_data[treatment_col].values])
            alpha, _, _, _ = np.linalg.lstsq(Xm, m, rcond=None)
            path_a = float(alpha[1])

            # Path b: mediator → outcome (controlling for treatment)
            Xb = np.column_stack([np.ones(n), X_data[treatment_col].values, m])
            beta_b, _, _, _ = np.linalg.lstsq(Xb, y, rcond=None)
            path_b = float(beta_b[2])
            direct_effect = float(beta_b[1])

            indirect_effect = path_a * path_b
            total_effect = direct_effect + indirect_effect
            mediation_pct = abs(indirect_effect / total_effect * 100) if abs(total_effect) > 1e-9 else 0.0

            mediation_result = {
                "mediator": mediation_col,
                "path_a_treatment_to_mediator": round(path_a, 4),
                "path_b_mediator_to_outcome": round(path_b, 4),
                "indirect_effect": round(indirect_effect, 4),
                "direct_effect": round(direct_effect, 4),
                "total_effect": round(total_effect, 4),
                "mediation_pct": round(mediation_pct, 1),
            }
        except Exception:
            mediation_result = {"error": "Mediation decomposition failed."}

    # Descriptive context
    treat_mean = float(treatment.mean())
    outcome_mean = float(outcome.mean())

    significance = "significant" if p_value < 0.05 else "not significant"
    direction = "positive" if ate > 0 else "negative"
    magnitude = "large" if abs(effect_size) > 0.8 else "medium" if abs(effect_size) > 0.5 else "small" if abs(effect_size) > 0.2 else "negligible"

    controls_desc = f" (controlling for {', '.join(control_cols)})" if control_cols else ""
    readout = (
        f"Causal effect of '{treatment_col}' on '{outcome_col}'{controls_desc}: "
        f"ATE = {ate:.4f} (95% CI [{ci_lower:.4f}, {ci_upper:.4f}]), "
        f"p = {p_value:.4f} ({significance}). "
        f"Effect size ({effect_metric}) = {effect_size:.2f} ({magnitude}, {direction}). "
        f"E-value = {e_value:.2f} — an unmeasured confounder would need risk-ratio ≥{e_value:.2f} "
        f"to fully explain away this effect. R² = {r2:.3f}."
    )

    result = {
        "treatment_col": treatment_col,
        "outcome_col": outcome_col,
        "control_cols": control_cols or [],
        "n_observations": n,
        "treatment_is_binary": is_binary,
        "ate": round(ate, 6),
        "std_error_hc3": round(se, 6),
        "ci_lower_95": round(ci_lower, 6),
        "ci_upper_95": round(ci_upper, 6),
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "significant_at_05": p_value < 0.05,
        "effect_size": round(effect_size, 4),
        "effect_metric": effect_metric,
        "effect_magnitude": magnitude,
        "effect_direction": direction,
        "r_squared": round(r2, 4),
        "e_value": round(e_value, 3),
        "treatment_mean": round(treat_mean, 4),
        "outcome_mean": round(outcome_mean, 4),
        "mediation": mediation_result,
        "engineering_readout": readout,
    }
    return result
