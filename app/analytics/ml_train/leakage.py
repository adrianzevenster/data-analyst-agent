"""Pre-training data leakage detection."""
from __future__ import annotations

import pandas as pd


def detect_leakage(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    corr_high: float = 0.95,
    corr_medium: float = 0.85,
) -> list[dict]:
    """
    Scan numeric features for suspiciously high correlation with the target.
    Also flags features whose names contain the target column name.

    Returns a list of warning dicts sorted by severity, each with:
      feature, risk ("high"/"medium"), correlation (float or None), reason
    """
    warnings: list[dict] = []
    flagged: set[str] = set()

    target_series = pd.to_numeric(df[target_col], errors="coerce")
    # For classification targets, encode as integer codes for correlation check
    if target_series.isna().mean() > 0.5:
        target_series = pd.Categorical(df[target_col]).codes.astype(float)
        target_series = pd.Series(target_series, index=df.index)

    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if df[col].nunique(dropna=True) < 2:
            continue
        try:
            corr = float(df[col].corr(target_series))
            if not (corr == corr):  # NaN check
                continue
            abs_corr = abs(corr)
            if abs_corr >= corr_high:
                warnings.append({
                    "feature": col,
                    "risk": "high",
                    "correlation": round(abs_corr, 3),
                    "reason": "correlation",
                })
                flagged.add(col)
            elif abs_corr >= corr_medium:
                warnings.append({
                    "feature": col,
                    "risk": "medium",
                    "correlation": round(abs_corr, 3),
                    "reason": "correlation",
                })
                flagged.add(col)
        except Exception:
            continue

    # Name-based heuristic: feature name is a substring of target (or vice versa)
    target_tok = target_col.lower().replace("_", "").replace(" ", "")
    for col in feature_cols:
        if col in flagged:
            continue
        col_tok = col.lower().replace("_", "").replace(" ", "")
        if (
            (len(target_tok) >= 4 and target_tok in col_tok)
            or (len(col_tok) >= 4 and col_tok in target_tok)
        ):
            warnings.append({
                "feature": col,
                "risk": "medium",
                "correlation": None,
                "reason": "name_similarity",
            })

    # Sort: high first, then by correlation descending
    warnings.sort(key=lambda w: (0 if w["risk"] == "high" else 1, -(w.get("correlation") or 0)))
    return warnings
