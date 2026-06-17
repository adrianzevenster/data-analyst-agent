from __future__ import annotations

import numpy as np
import pandas as pd

from app.agent.executor import Executor
from app.agent.llm import LATEST_TRAINED_MODEL_SENTINEL
from app.analytics.dataset_manager import DatasetManager
from app.analytics.ml_train.model_store import ModelManager
from app.analytics.tooling import get_registry
from app.core.models import ToolCall


def test_registry_exposes_argument_schemas():
    tools = {tool["name"]: tool for tool in get_registry().list()}

    assert "evaluate_ml_predictions" in tools
    schema = tools["evaluate_ml_predictions"]["args_schema"]
    assert "properties" in schema
    assert "task_hint" in schema["properties"]


def test_executor_rejects_unknown_tool_arguments(tmp_path):
    manager = DatasetManager(base_dir=str(tmp_path))
    meta = manager.register_df(pd.DataFrame({"actual": [1, 0], "prediction": [1, 1]}), "predictions.csv")

    executor = Executor()
    executor.dm = manager

    results, tables, charts = executor.run(
        meta.dataset_id,
        [ToolCall(name="evaluate_ml_predictions", arguments={"task_hint": "classification", "made_up": True})],
    )

    assert not results[0].ok
    assert "Invalid arguments" in (results[0].error or "")
    assert tables == []
    assert charts == []


def test_executor_rejects_unknown_column_arguments(tmp_path):
    manager = DatasetManager(base_dir=str(tmp_path))
    meta = manager.register_df(pd.DataFrame({"actual": [1, 0], "prediction": [1, 1]}), "predictions.csv")

    executor = Executor()
    executor.dm = manager

    results, _, _ = executor.run(
        meta.dataset_id,
        [
            ToolCall(
                name="evaluate_ml_predictions",
                arguments={"task_hint": "classification", "actual_col": "missing", "prediction_col": "prediction"},
            )
        ],
    )

    assert not results[0].ok
    assert "actual_col not in dataset" in (results[0].error or "")


def test_multidim_pivot_works_with_explicit_none_columns(tmp_path):
    # Regression test: "columns" had no default in multidim_pivot's signature,
    # but Pydantic args validation strips explicit None values, which used to
    # raise a TypeError for the exact arguments the rule planner sends.
    manager = DatasetManager(base_dir=str(tmp_path))
    meta = manager.register_df(
        pd.DataFrame({"region": ["a", "b", "a"], "revenue": [10, 20, 30]}), "d.csv"
    )

    executor = Executor()
    executor.dm = manager

    results, tables, charts = executor.run(
        meta.dataset_id,
        [
            ToolCall(
                name="multidim_pivot",
                arguments={"index": ["region"], "columns": None, "values": "revenue", "agg": "sum", "top_n": 50},
            )
        ],
    )

    assert results[0].ok, results[0].error
    assert tables
    assert charts


def test_dict_result_with_embedded_charts_key_surfaces_them_directly(tmp_path):
    manager = DatasetManager(base_dir=str(tmp_path))
    meta = manager.register_df(pd.DataFrame({"region": ["a", "a", "b"]}), "d.csv")

    executor = Executor()
    executor.dm = manager

    results, _, charts = executor.run(
        meta.dataset_id,
        [ToolCall(name="overrepresented_categories", arguments={"col": "region", "threshold": 0.5})],
    )

    assert results[0].ok, results[0].error
    assert charts
    assert charts[0]["type"] == "bar"


def test_standalone_chart_tool_result_surfaces_as_chart_not_metric_table(tmp_path):
    manager = DatasetManager(base_dir=str(tmp_path))
    meta = manager.register_df(pd.DataFrame({"revenue": list(range(50))}), "d.csv")

    executor = Executor()
    executor.dm = manager

    results, tables, charts = executor.run(
        meta.dataset_id,
        [ToolCall(name="histogram_spec", arguments={"column": "revenue", "bins": 5})],
    )

    assert results[0].ok, results[0].error
    assert charts and charts[0]["type"] == "histogram"
    # The chart-spec result itself shouldn't also be dumped as a metric table.
    assert tables == []


def test_correlation_analysis_runs_through_executor(tmp_path):
    manager = DatasetManager(base_dir=str(tmp_path))
    meta = manager.register_df(
        pd.DataFrame({"revenue": [1, 2, 3, 4, 5], "cost": [2, 4, 6, 8, 10]}), "d.csv"
    )

    executor = Executor()
    executor.dm = manager

    results, tables, charts = executor.run(
        meta.dataset_id, [ToolCall(name="correlation_analysis", arguments={})]
    )

    assert results[0].ok, results[0].error
    assert charts


def test_correlation_analysis_rejects_unknown_column(tmp_path):
    manager = DatasetManager(base_dir=str(tmp_path))
    meta = manager.register_df(pd.DataFrame({"a": [1, 2, 3]}), "d.csv")

    executor = Executor()
    executor.dm = manager

    results, _, _ = executor.run(
        meta.dataset_id,
        [ToolCall(name="correlation_analysis", arguments={"numeric_cols": ["missing_col"]})],
    )

    assert not results[0].ok
    assert "numeric_cols contains unknown columns" in (results[0].error or "")


def test_trend_analysis_runs_through_executor(tmp_path):
    manager = DatasetManager(base_dir=str(tmp_path))
    meta = manager.register_df(
        pd.DataFrame({"date": pd.date_range("2023-01-01", periods=400), "revenue": range(400)}), "d.csv"
    )

    executor = Executor()
    executor.dm = manager

    results, tables, charts = executor.run(meta.dataset_id, [ToolCall(name="trend_analysis", arguments={})])

    assert results[0].ok, results[0].error
    assert charts


def test_trend_analysis_rejects_unknown_column(tmp_path):
    manager = DatasetManager(base_dir=str(tmp_path))
    meta = manager.register_df(pd.DataFrame({"a": [1, 2, 3]}), "d.csv")

    executor = Executor()
    executor.dm = manager

    results, _, _ = executor.run(
        meta.dataset_id, [ToolCall(name="trend_analysis", arguments={"date_col": "missing_col"})]
    )

    assert not results[0].ok
    assert "date_col not in dataset" in (results[0].error or "")


def test_auto_insights_runs_through_executor(tmp_path):
    manager = DatasetManager(base_dir=str(tmp_path))
    meta = manager.register_df(
        pd.DataFrame({"revenue": [1, 2, 3, 4, 5], "cost": [2, 4, 6, 8, 10]}), "d.csv"
    )

    executor = Executor()
    executor.dm = manager

    results, tables, charts = executor.run(meta.dataset_id, [ToolCall(name="auto_insights", arguments={})])

    assert results[0].ok, results[0].error
    assert tables


def _classification_df(n: int = 50, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"feature1": rng.normal(0, 1, n), "feature2": rng.normal(5, 2, n)})
    df["churn"] = (df["feature1"] > 0).astype(int)
    return df


def test_score_with_model_resolves_sentinel_to_just_trained_model(tmp_path):
    manager = DatasetManager(base_dir=str(tmp_path / "datasets"))
    meta = manager.register_df(_classification_df(), "d.csv")

    executor = Executor()
    executor.dm = manager
    executor.model_manager = ModelManager(base_dir=str(tmp_path / "models"))

    results, _, _ = executor.run(
        meta.dataset_id,
        [
            ToolCall(name="train_supervised_model", arguments={"target_col": "churn"}),
            ToolCall(
                name="score_with_model",
                arguments={"model_id": LATEST_TRAINED_MODEL_SENTINEL},
            ),
        ],
    )

    train_result, score_result = results
    assert train_result.ok, train_result.error
    assert score_result.ok, score_result.error
    assert score_result.result["model_id"] == train_result.result["model_id"]


def test_score_with_model_sentinel_without_prior_training_fails_clearly(tmp_path):
    manager = DatasetManager(base_dir=str(tmp_path / "datasets"))
    meta = manager.register_df(_classification_df(), "d.csv")

    executor = Executor()
    executor.dm = manager
    executor.model_manager = ModelManager(base_dir=str(tmp_path / "models"))

    results, _, _ = executor.run(
        meta.dataset_id,
        [ToolCall(name="score_with_model", arguments={"model_id": LATEST_TRAINED_MODEL_SENTINEL})],
    )

    assert not results[0].ok
    assert "No model was trained earlier in this request" in results[0].error
