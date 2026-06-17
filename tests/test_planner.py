from __future__ import annotations

import pandas as pd

from app.agent.planner import Planner
from app.analytics.dataset_manager import DatasetManager


def test_rule_planner_selects_ml_evaluation_for_prediction_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("ENABLE_RAG", "0")

    manager = DatasetManager(base_dir=str(tmp_path))
    meta = manager.register_df(
        pd.DataFrame(
            {
                "actual": [1, 0, 1],
                "prediction": [1, 1, 1],
                "probability": [0.9, 0.7, 0.6],
            }
        ),
        "predictions.csv",
    )

    planner = Planner()
    planner.dm = manager

    calls, citations, source, llm_error, llm_notes = planner.plan("evaluate model performance", meta.dataset_id)

    assert citations == []
    assert [call.name for call in calls] == ["evaluate_ml_predictions"]
    assert calls[0].arguments["task_hint"] == "classification"
