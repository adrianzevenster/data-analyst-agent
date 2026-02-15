from __future__ import annotations

import pandas as pd


def simple_bar_spec(df: pd.DataFrame, x: str, y: str, title: str = "") -> dict:
    data = df[[x, y]].head(200).to_dict(orient="records")
    return {
        "type": "bar",
        "title": title or f"{y} by {x}",
        "x": x,
        "y": y,
        "data": data,
    }
