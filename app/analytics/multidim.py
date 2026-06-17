from __future__ import annotations

import pandas as pd


def multidim_pivot(
        df: pd.DataFrame,
        index: list[str],
        values: str,
        columns: list[str] | None = None,
        agg: str = "sum",
        fillna: float | int | None = 0,
        top_n: int = 50,
) -> pd.DataFrame:
    if not index:
        raise ValueError("index dimensions required")
    if values not in df.columns:
        raise ValueError(f"values column not in df: {values}")

    pt = pd.pivot_table(
        df,
        index=index,
        columns=columns if columns else None,
        values=values,
        aggfunc=agg,
        fill_value=fillna,
        dropna=False,
    )

    pt = pt.reset_index()

    # Reduce huge outputs
    if len(pt) > top_n:
        pt = pt.sort_values(by=pt.columns[-1], ascending=False).head(top_n)

    return pt
