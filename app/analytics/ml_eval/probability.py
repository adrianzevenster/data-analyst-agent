from __future__ import annotations

import pandas as pd


def evaluate_prediction_scores(
        df: pd.DataFrame,
        probability_col: str,
        prediction_col: str | None = None,
        id_col: str | None = None,
        top_n: int = 25,
) -> dict:
    d = df.copy()
    d[probability_col] = pd.to_numeric(d[probability_col], errors="coerce")
    scored = d.dropna(subset=[probability_col])

    if scored.empty:
        return {
            "task_type": "scored_predictions",
            "error": f"No numeric scores found in probability column: {probability_col}",
        }

    score = scored[probability_col]

    high_confidence = scored[score >= 0.80]
    medium_confidence = scored[(score >= 0.50) & (score < 0.80)]
    low_confidence = scored[score < 0.50]

    display_cols = []
    for c in [id_col, prediction_col, probability_col]:
        if c and c in scored.columns and c not in display_cols:
            display_cols.append(c)

    if not display_cols:
        display_cols = [probability_col]

    top_predictions = (
        scored.sort_values(probability_col, ascending=False)
        .head(top_n)[display_cols]
        .to_dict(orient="records")
    )

    return {
        "task_type": "scored_predictions",
        "n_rows_scored": int(len(scored)),
        "score_column": probability_col,
        "prediction_column": prediction_col,
        "id_column": id_col,
        "score_summary": {
            "mean": float(score.mean()),
            "min": float(score.min()),
            "p10": float(score.quantile(0.10)),
            "p50": float(score.quantile(0.50)),
            "p90": float(score.quantile(0.90)),
            "p95": float(score.quantile(0.95)),
            "max": float(score.max()),
        },
        "confidence_bands": {
            "high_confidence_0_80_plus": int(len(high_confidence)),
            "medium_confidence_0_50_to_0_80": int(len(medium_confidence)),
            "low_confidence_below_0_50": int(len(low_confidence)),
        },
        "top_predictions": top_predictions,
    }