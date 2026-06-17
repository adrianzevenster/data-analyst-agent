from __future__ import annotations

import pandas as pd

from app.analytics.viz_specs import scatter_spec, simple_bar_spec

MAX_NUMERIC_COLS = 30
MAX_CATEGORICAL_COLS = 10
MAX_CATEGORICAL_CARDINALITY = 20
STRONG_CORRELATION_THRESHOLD = 0.7
STRONG_ASSOCIATION_THRESHOLD = 0.5
MIN_PAIRWISE_OVERLAP_RATIO = 0.5


def _default_numeric_cols(df: pd.DataFrame) -> list[str]:
    return [
        str(c) for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
    ][:MAX_NUMERIC_COLS]


def _default_categorical_cols(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_datetime64_any_dtype(df[c]):
            continue
        nunique = df[c].nunique(dropna=True)
        if 2 <= nunique <= MAX_CATEGORICAL_CARDINALITY:
            cols.append(str(c))
        if len(cols) >= MAX_CATEGORICAL_COLS:
            break
    return cols


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float | None:
    d = pd.DataFrame({"cat": categories, "val": pd.to_numeric(values, errors="coerce")}).dropna()
    if d["cat"].nunique() < 2 or len(d) < 3:
        return None

    grand_mean = d["val"].mean()
    ss_between = sum(len(g) * (g["val"].mean() - grand_mean) ** 2 for _, g in d.groupby("cat"))
    ss_total = ((d["val"] - grand_mean) ** 2).sum()

    if ss_total == 0:
        return None

    return float((ss_between / ss_total) ** 0.5)


def correlation_analysis(
    df: pd.DataFrame,
    numeric_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
    top_n: int = 20,
) -> dict:
    numeric_cols = [c for c in (numeric_cols or _default_numeric_cols(df)) if c in df.columns]
    categorical_cols = [c for c in (categorical_cols or _default_categorical_cols(df)) if c in df.columns]

    n_rows = len(df)
    min_overlap = n_rows * MIN_PAIRWISE_OVERLAP_RATIO

    numeric_pairs: list[dict] = []
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr(numeric_only=True)
        seen: set[tuple[str, str]] = set()
        for col_a in numeric_cols:
            for col_b in numeric_cols:
                if col_a == col_b:
                    continue
                pair: tuple[str, str] = (min(col_a, col_b), max(col_a, col_b))
                if pair in seen:
                    continue
                seen.add(pair)
                value = corr_matrix.loc[col_a, col_b]
                if pd.notna(value):
                    n_pairs = int(df[[col_a, col_b]].dropna().shape[0])
                    numeric_pairs.append(
                        {
                            "column_a": pair[0],
                            "column_b": pair[1],
                            "correlation": round(float(value), 4),
                            "abs_correlation": round(abs(float(value)), 4),
                            "n_pairs": n_pairs,
                            "low_overlap": n_pairs < min_overlap,
                        }
                    )
        numeric_pairs.sort(key=lambda item: item["abs_correlation"], reverse=True)
        numeric_pairs = numeric_pairs[:top_n]

    categorical_associations: list[dict] = []
    for cat_col in categorical_cols:
        for num_col in numeric_cols:
            ratio = correlation_ratio(df[cat_col], df[num_col])
            if ratio is not None:
                categorical_associations.append(
                    {
                        "categorical_col": cat_col,
                        "numeric_col": num_col,
                        "correlation_ratio": round(ratio, 4),
                    }
                )
    categorical_associations.sort(key=lambda item: item["correlation_ratio"], reverse=True)
    categorical_associations = categorical_associations[:top_n]

    findings: list[str] = []
    for corr_pair in numeric_pairs:
        # Pairwise correlation drops rows where either column is null; if
        # that leaves only a small, possibly unrepresentative slice of the
        # data, don't let it surface as a headline finding.
        if corr_pair["abs_correlation"] >= STRONG_CORRELATION_THRESHOLD and not corr_pair["low_overlap"]:
            direction = "positively" if corr_pair["correlation"] > 0 else "negatively"
            findings.append(
                f"'{corr_pair['column_a']}' and '{corr_pair['column_b']}' are strongly {direction} correlated (r={corr_pair['correlation']:.2f})."
            )
    for assoc in categorical_associations:
        if assoc["correlation_ratio"] >= STRONG_ASSOCIATION_THRESHOLD:
            findings.append(
                f"'{assoc['categorical_col']}' is strongly associated with '{assoc['numeric_col']}' "
                f"(correlation ratio={assoc['correlation_ratio']:.2f})."
            )

    charts = []
    if numeric_pairs:
        top_pair = next((p for p in numeric_pairs if not p["low_overlap"]), numeric_pairs[0])
        charts.append(
            scatter_spec(
                df, x=top_pair["column_a"], y=top_pair["column_b"],
                title=f"{top_pair['column_b']} vs {top_pair['column_a']} (r={top_pair['correlation']:.2f})",
            )
        )
    if categorical_associations:
        top_assoc = categorical_associations[0]
        group_means = df.groupby(top_assoc["categorical_col"])[top_assoc["numeric_col"]].mean().reset_index()
        charts.append(
            simple_bar_spec(
                group_means, x=top_assoc["categorical_col"], y=top_assoc["numeric_col"],
                title=f"Mean {top_assoc['numeric_col']} by {top_assoc['categorical_col']}",
            )
        )

    if findings:
        readout = f"Found {len(findings)} notable relationship(s). " + " ".join(findings[:3])
    elif numeric_pairs or categorical_associations:
        readout = "No strong relationships found; the strongest signals are weak-to-moderate."
    else:
        readout = "Not enough numeric or categorical columns to analyze relationships."

    return {
        "numeric_correlations": numeric_pairs,
        "categorical_associations": categorical_associations,
        "findings": findings,
        "charts": charts,
        "engineering_readout": readout,
    }
