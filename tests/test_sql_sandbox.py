from __future__ import annotations

import duckdb
import pandas as pd
import pytest

from app.analytics.sql import duckdb_query


def test_duckdb_query_runs_normal_query_against_loaded_df():
    df = pd.DataFrame({"a": [1, 2, 3]})

    out = duckdb_query(df, "select sum(a) as total from t")

    assert out["total"].iloc[0] == 6


def test_duckdb_query_blocks_filesystem_read():
    df = pd.DataFrame({"a": [1]})

    with pytest.raises(duckdb.Error):
        duckdb_query(df, "select * from read_csv('/etc/passwd')")


def test_duckdb_query_blocks_filesystem_write(tmp_path):
    df = pd.DataFrame({"a": [1]})
    target = tmp_path / "leak.csv"

    with pytest.raises(duckdb.Error):
        duckdb_query(df, f"COPY t TO '{target}'")

    assert not target.exists()


def test_duckdb_query_blocks_attach():
    df = pd.DataFrame({"a": [1]})

    with pytest.raises(duckdb.Error):
        duckdb_query(df, "ATTACH '/tmp/evil.db' AS evil")
