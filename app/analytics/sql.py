from __future__ import annotations

import duckdb
import pandas as pd


def duckdb_query(df: pd.DataFrame, query: str) -> pd.DataFrame:
    con = duckdb.connect(database=":memory:")
    con.register("t", df)
    out = con.execute(query).df()
    con.close()
    return out
