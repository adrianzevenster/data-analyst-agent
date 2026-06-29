from __future__ import annotations

from app.analytics.ml_train.training import train_supervised_model
from app.analytics.ml_train.scoring import score_with_model
from app.analytics.ml_train.explainability import explain_model, shap_explain_prediction
from app.analytics.ml_train.evaluation import evaluate_trained_model
from app.analytics.ml_train.forecasting import forecast_with_model
from app.analytics.ml_train.pdp import compute_pdp, compute_ice
from app.analytics.ml_train.whatif import what_if_predict
from app.analytics.ml_train.segment_eval import evaluate_by_segment

__all__ = [
    "train_supervised_model",
    "score_with_model",
    "explain_model",
    "shap_explain_prediction",
    "evaluate_trained_model",
    "forecast_with_model",
    "compute_pdp",
    "compute_ice",
    "what_if_predict",
    "evaluate_by_segment",
]
