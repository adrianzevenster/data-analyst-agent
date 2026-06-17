from __future__ import annotations

import re

import duckdb
import pandas as pd


def _safe_table_name(filename: str) -> str:
    name = filename.rsplit(".", 1)[0]
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return (name[:32] if name else "ds")


def duckdb_query(
    df: pd.DataFrame,
    query: str,
    extra_tables: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    # enable_external_access=False blocks file/network reads, COPY TO, and
    # ATTACH so arbitrary SQL from a user/LLM can't escape the loaded dataframe.
    con = duckdb.connect(database=":memory:", config={"enable_external_access": False})
    con.register("t", df)
    if extra_tables:
        for name, tdf in extra_tables.items():
            con.register(name, tdf)
    out = con.execute(query).df()
    con.close()
    return out
