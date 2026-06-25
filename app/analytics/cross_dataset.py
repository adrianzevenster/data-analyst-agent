"""Cross-dataset analysis: auto-detect join keys and compute cross-dataset correlations.

No new dependencies. Works on whatever DatasetManager has loaded in the session.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd


_MAX_COLS_PER_DS = 50
_SAMPLE_ROWS = 5000
_HIGH_CORR = 0.7


def _name_similarity(a: str, b: str) -> float:
    """Jaccard similarity on lowercase alphanumeric tokens."""
    ta = set(re.findall(r"[a-z0-9]+", a.lower()))
    tb = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _is_potential_key(series: pd.Series) -> bool:
    """A column looks like a join key when it's an int/string id or has high cardinality."""
    if pd.api.types.is_bool_dtype(series):
        return False
    nuniq = series.nunique(dropna=True)
    n = len(series.dropna())
    if n == 0:
        return False
    uniqueness = nuniq / n
    # High-cardinality integer or string columns
    return uniqueness > 0.5 and nuniq > 5


def _candidate_join_keys(df_a: pd.DataFrame, df_b: pd.DataFrame) -> list[dict]:
    """Find column pairs that could be join keys between two datasets.

    Criteria (any of):
    1. Exact name match.
    2. High name similarity (Jaccard > 0.5) + compatible dtypes.
    3. Overlapping value sets (intersection > 30% of the smaller set).
    """
    candidates = []
    cols_a = list(df_a.columns[:_MAX_COLS_PER_DS])
    cols_b = list(df_b.columns[:_MAX_COLS_PER_DS])

    for ca in cols_a:
        for cb in cols_b:
            score = _name_similarity(ca, cb)
            exact = ca.lower() == cb.lower()
            if not exact and score < 0.5:
                continue

            sa = df_a[ca].dropna().astype(str)
            sb = df_b[cb].dropna().astype(str)
            if sa.empty or sb.empty:
                continue

            set_a = set(sa.unique())
            set_b = set(sb.unique())
            overlap = len(set_a & set_b) / max(min(len(set_a), len(set_b)), 1)

            if overlap < 0.1 and not exact:
                continue

            candidates.append({
                "col_a": ca,
                "col_b": cb,
                "name_similarity": round(score, 3),
                "exact_name_match": exact,
                "value_overlap": round(overlap, 3),
                "recommended": exact or (score > 0.6 and overlap > 0.3),
            })

    # Sort: exact matches first, then by overlap
    candidates.sort(key=lambda x: (-int(x["exact_name_match"]), -x["value_overlap"]))
    return candidates[:10]


def _cross_numeric_correlations(
    df_a: pd.DataFrame, df_b: pd.DataFrame, join_key_a: str, join_key_b: str, top_n: int = 10
) -> list[dict]:
    """Merge on discovered key and compute cross-dataset numeric correlations."""
    try:
        merged = df_a.merge(
            df_b,
            left_on=join_key_a,
            right_on=join_key_b,
            how="inner",
            suffixes=("_A", "_B"),
        )
    except Exception:
        return []

    if len(merged) < 10:
        return []

    num_a = [c for c in df_a.columns if pd.api.types.is_numeric_dtype(df_a[c]) and c != join_key_a]
    num_b = [c for c in df_b.columns if pd.api.types.is_numeric_dtype(df_b[c]) and c != join_key_b]

    results = []
    for ca in num_a[:15]:
        col_a_merged = ca + "_A" if (ca + "_A") in merged.columns else ca
        if col_a_merged not in merged.columns:
            continue
        for cb in num_b[:15]:
            col_b_merged = cb + "_B" if (cb + "_B") in merged.columns else cb
            if col_b_merged not in merged.columns or col_a_merged == col_b_merged:
                continue
            s_a = pd.to_numeric(merged[col_a_merged], errors="coerce")
            s_b = pd.to_numeric(merged[col_b_merged], errors="coerce")
            mask = s_a.notna() & s_b.notna()
            if mask.sum() < 5:
                continue
            corr = float(s_a[mask].corr(s_b[mask]))
            if abs(corr) >= _HIGH_CORR:
                results.append({
                    "col_a": ca,
                    "col_b": cb,
                    "pearson_r": round(corr, 4),
                    "abs_r": round(abs(corr), 4),
                    "n_matched": int(mask.sum()),
                })

    results.sort(key=lambda x: -x["abs_r"])
    return results[:top_n]


def cross_dataset_profile(
    df: pd.DataFrame,
    dataset_id_a: str | None = None,
    dataset_id_b: str | None = None,
) -> dict:
    """Profile the current dataset against other loaded datasets.

    Discovers join key candidates and cross-dataset numeric correlations.
    The `df` argument is the currently active dataset (dataset_id_a).
    Other datasets are loaded via DatasetManager for dataset_id_b.
    """
    from app.analytics.dataset_manager import DatasetManager

    dm = DatasetManager()
    all_meta = dm.list_datasets()

    # If dataset_id_b specified, compare against that one; else all others
    targets = [
        m for m in all_meta
        if m.dataset_id != dataset_id_a
        and (dataset_id_b is None or m.dataset_id == dataset_id_b)
    ]

    if not targets:
        return {
            "error": "No other datasets are loaded. Upload a second dataset to enable cross-dataset analysis.",
            "n_loaded_datasets": len(all_meta),
        }

    comparison_results = []
    for meta in targets[:5]:
        try:
            other_df = dm.load_df(meta.dataset_id, limit=_SAMPLE_ROWS)
        except Exception:
            continue

        candidates = _candidate_join_keys(df, other_df)
        best_key = next((c for c in candidates if c["recommended"]), None)

        cross_corrs: list[dict] = []
        if best_key:
            cross_corrs = _cross_numeric_correlations(
                df, other_df, best_key["col_a"], best_key["col_b"]
            )

        comparison_results.append({
            "dataset_id": meta.dataset_id,
            "filename": meta.filename,
            "n_rows": meta.n_rows,
            "n_cols": len(other_df.columns),
            "join_key_candidates": candidates,
            "best_join_key": best_key,
            "cross_correlations": cross_corrs,
            "n_high_correlations": len(cross_corrs),
        })

    # Build readout
    readout_lines = [f"Cross-dataset profile: {len(comparison_results)} dataset(s) compared."]
    for r in comparison_results:
        fn = r["filename"]
        if r["best_join_key"]:
            bk = r["best_join_key"]
            readout_lines.append(
                f"  '{fn}': best join key '{bk['col_a']}' ↔ '{bk['col_b']}' "
                f"(overlap={bk['value_overlap']:.0%}), "
                f"{r['n_high_correlations']} high cross-correlations found."
            )
        else:
            readout_lines.append(f"  '{fn}': no clear join key detected.")

    return {
        "n_datasets_compared": len(comparison_results),
        "comparisons": comparison_results,
        "engineering_readout": " ".join(readout_lines),
    }
