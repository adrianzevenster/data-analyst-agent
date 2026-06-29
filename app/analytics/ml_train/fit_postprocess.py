from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

from app.analytics.ml_eval.classification import _sample_curve

_FEATURE_IMPORTANCE_TOP_N = 15


def _find_optimal_threshold(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> float | None:
    """Threshold maximising binary F1 on the test split.

    Sweeps 91 thresholds (0.05–0.95). Returns None for multiclass or when
    predict_proba is unavailable.
    """
    if not hasattr(pipeline, "predict_proba"):
        return None
    try:
        _cls = getattr(pipeline, "classes_", None)
        classes: list = list(_cls) if _cls is not None else []
        if not classes and hasattr(pipeline, "named_steps"):
            classes = list(getattr(pipeline.named_steps.get("model"), "classes_", []))
        if len(classes) != 2:
            return None
        pos_label = classes[-1]
        probs = pipeline.predict_proba(X_test)[:, -1]
        best_thresh, best_f1 = 0.5, 0.0
        for t in np.linspace(0.05, 0.95, 91):
            t = float(t)
            preds = pd.Series(
                [pos_label if p >= t else classes[0] for p in probs],
                index=y_test.index,
            )
            score = f1_score(y_test, preds, pos_label=pos_label, average="binary", zero_division=0)
            if score > best_f1:
                best_f1, best_thresh = score, t
        return round(best_thresh, 2)
    except Exception:
        return None


def _extract_feature_importance(pipeline, top_n: int = _FEATURE_IMPORTANCE_TOP_N) -> list[dict]:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocess"]
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        return []
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        coef = model.coef_
        importances = abs(coef[0]) if getattr(coef, "ndim", 1) > 1 else abs(coef)
    else:
        return []
    if len(importances) != len(feature_names):
        return []
    pairs = sorted(zip(feature_names, importances), key=lambda item: -abs(item[1]))[:top_n]
    return [{"feature": str(name), "importance": float(value)} for name, value in pairs]


def compute_ovr_roc(pipeline, X_test: pd.DataFrame, y_test: pd.Series, pipeline_classes: list) -> dict:
    """One-vs-rest ROC curves for multiclass classifiers.

    Returns a dict suitable for merging into the evaluation result.
    """
    result: dict = {}
    try:
        probs_matrix = pipeline.predict_proba(X_test)
        y_bin = label_binarize(y_test.to_numpy(), classes=pipeline_classes)
        ovr_charts: list[dict] = []
        ovr_auc_map: dict[str, float] = {}
        for i, cls in enumerate(pipeline_classes):
            try:
                auc_val = float(roc_auc_score(y_bin[:, i], probs_matrix[:, i]))
                fpr_arr, tpr_arr, _ = roc_curve(y_bin[:, i], probs_matrix[:, i])
                fpr_s, tpr_s = _sample_curve(fpr_arr, tpr_arr)
                ovr_charts.append({
                    "type": "line",
                    "title": f"{cls}  (AUC {auc_val:.3f})",
                    "x": "fpr",
                    "y": "tpr",
                    "data": [{"fpr": round(f, 4), "tpr": round(t, 4)} for f, t in zip(fpr_s, tpr_s)],
                })
                ovr_auc_map[str(cls)] = round(auc_val, 3)
            except Exception:
                pass
        if ovr_charts:
            result["roc_curves_ovr"] = ovr_charts
            result["roc_auc_ovr"] = ovr_auc_map
        macro_auc = float(roc_auc_score(y_bin, probs_matrix, average="macro", multi_class="ovr"))
        result["roc_auc"] = round(macro_auc, 4)
    except Exception:
        pass
    return result


def compute_conformal_regression(y_test_eval: np.ndarray, y_pred: np.ndarray) -> float | None:
    """90% split-conformal prediction interval halfwidth (Papadopoulos et al.).

    Requires at least 10 test samples; returns None otherwise.
    """
    if len(y_test_eval) < 10:
        return None
    try:
        nonconformity = np.abs(y_test_eval - y_pred)
        n_cal = len(nonconformity)
        q_level = min(1.0, np.ceil((n_cal + 1) * 0.90) / n_cal)
        return float(np.quantile(nonconformity, q_level))
    except Exception:
        return None


def compute_conformal_classification(pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> float | None:
    """90% split-conformal prediction set threshold for classifiers.

    Nonconformity score = 1 - p(true class). The quantile threshold gives
    ≥90% marginal coverage on new data under exchangeability.
    """
    if not hasattr(pipeline, "predict_proba") or len(y_test) < 10:
        return None
    try:
        _cls = getattr(pipeline, "classes_", None)
        classes: list = list(_cls) if _cls is not None else []
        if not classes and hasattr(pipeline, "named_steps"):
            classes = list(getattr(pipeline.named_steps.get("model"), "classes_", []))
        if len(classes) < 2:
            return None
        class_to_idx = {c: i for i, c in enumerate(classes)}
        probs = pipeline.predict_proba(X_test)
        ncs = np.array([
            1.0 - probs[i, class_to_idx.get(y_test.iloc[i], 0)]
            for i in range(len(y_test))
        ])
        n_cal = len(ncs)
        q_level = min(1.0, np.ceil((n_cal + 1) * 0.90) / n_cal)
        return float(np.quantile(ncs, q_level))
    except Exception:
        return None
