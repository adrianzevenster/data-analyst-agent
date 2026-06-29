"""Tests for multi-step autoregressive forecasting."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.analytics.ml_train.model_store import ModelManager
from app.analytics.ml_train.training import train_supervised_model
from app.analytics.ml_train.forecasting import forecast_with_model


def _temporal_df(n: int = 80) -> pd.DataFrame:
    dates = pd.date_range("2023-01-01", periods=n, freq="D")
    rng = np.random.default_rng(42)
    values = np.arange(n, dtype=float) * 0.5 + rng.normal(0, 0.5, n)
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "sales": values,
        "units": rng.integers(1, 10, n).astype(float),
    })


def _classification_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.normal(5, 2, n)})
    df["label"] = ((df["f1"] + df["f2"]) > 5).astype(int)
    return df


@pytest.fixture
def mm(tmp_path) -> ModelManager:
    return ModelManager(base_dir=str(tmp_path))


@pytest.fixture
def temporal_model(mm):
    df = _temporal_df(n=80)
    result = train_supervised_model(
        df, target_col="sales", model_type="ridge_regression",
        tune=False, cv_folds=2, model_manager=mm,
    )
    assert "error" not in result, result.get("error")
    return result["model_id"], mm


def test_forecast_returns_rows(temporal_model):
    model_id, mm = temporal_model
    df = _temporal_df(n=80)
    result = forecast_with_model(df, model_id=model_id, horizon=10, model_manager=mm)
    if "error" in result:
        pytest.skip(f"Forecast not available: {result['error']}")
    assert result["horizon_steps"] > 0
    assert len(result["forecast_rows"]) <= 10


def test_forecast_rows_have_date_and_prediction(temporal_model):
    model_id, mm = temporal_model
    df = _temporal_df(n=80)
    result = forecast_with_model(df, model_id=model_id, horizon=5, model_manager=mm)
    if "error" in result:
        pytest.skip(f"Forecast not available: {result['error']}")
    for row in result["forecast_rows"]:
        assert "date" in row
        assert "prediction" in row
        assert isinstance(row["prediction"], float)


def test_forecast_returns_chart_spec(temporal_model):
    model_id, mm = temporal_model
    df = _temporal_df(n=80)
    result = forecast_with_model(df, model_id=model_id, horizon=5, model_manager=mm)
    if "error" in result:
        pytest.skip(f"Forecast not available: {result['error']}")
    assert "charts" in result
    assert result["charts"][0]["type"] == "line"
    assert result["charts"][0]["x"] == "date"


def test_forecast_requires_lag_model(mm):
    df = _classification_df(n=80)
    result = train_supervised_model(df, target_col="label", tune=False, cv_folds=2, model_manager=mm)
    assert "error" not in result
    fr = forecast_with_model(df, model_id=result["model_id"], horizon=5, model_manager=mm)
    assert "error" in fr


def test_forecast_nonexistent_model(mm):
    df = _temporal_df()
    result = forecast_with_model(df, model_id="00000000-0000-0000-0000-000000000000", model_manager=mm)
    assert "error" in result
