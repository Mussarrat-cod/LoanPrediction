"""
model_layer.py - Model building, training, calibration, evaluation, and persistence.

Provides:
  - build_preprocessor(): ColumnTransformer (StandardScaler + OneHotEncoder)
  - build_pipelines(): three Pipeline objects (LR, RF, DT)
  - evaluate_model(): metrics for a single model
  - get_feature_names(): map pipeline output back to original names
  - get_global_importance(): intrinsic importance mapped to original names
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from backend.config import (
    ALL_NUMERIC_COLS,
    CATEGORICAL_COLS,
    DECISION_THRESHOLD,
    DT_PARAMS,
    LR_PARAMS,
    PROB_CLIP,
    RANDOM_STATE,
    RF_PARAMS,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------
# Preprocessor
# ----------------------------------------------

def build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """Build a ColumnTransformer from the actual columns present in X.

    Derives numeric and categorical column lists programmatically.

    Returns:
        preprocessor: fitted-ready ColumnTransformer
        numeric_cols: list of numeric column names used
        categorical_cols: list of categorical column names used
    """
    numeric_cols = [c for c in X.columns if c in ALL_NUMERIC_COLS]
    categorical_cols = [c for c in X.columns if c in CATEGORICAL_COLS]

    logger.info("Numeric features (%d): %s", len(numeric_cols), numeric_cols)
    logger.info("Categorical features (%d): %s", len(categorical_cols), categorical_cols)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        remainder="drop",
    )
    return preprocessor, numeric_cols, categorical_cols


# ----------------------------------------------
# Pipelines
# ----------------------------------------------

def build_pipelines(
    preprocessor: ColumnTransformer,
) -> Dict[str, Pipeline]:
    """Build three Pipeline objects: LR, RF, DT.

    Returns a dict keyed by model name.
    """
    pipelines = {
        "Logistic Regression": Pipeline([
            ("preprocess", preprocessor),
            ("classifier", LogisticRegression(**LR_PARAMS)),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", preprocessor),
            ("classifier", RandomForestClassifier(**RF_PARAMS)),
        ]),
        "Decision Tree": Pipeline([
            ("preprocess", preprocessor),
            ("classifier", DecisionTreeClassifier(**DT_PARAMS)),
        ]),
    }
    return pipelines


def calibrate_pipeline(
    pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, model_name: str
) -> Any:
    """Wrap LR and DT in CalibratedClassifierCV(isotonic, cv=5).

    RF is returned as-is (native probability estimation is good enough).
    The calibrated object wraps the *full* pipeline so inference is consistent.
    """
    if model_name == "Random Forest":
        logger.info("RF: skipping calibration (native probabilities).")
        pipeline.fit(X_train, y_train)
        return pipeline

    logger.info("Calibrating %s with isotonic regression (cv=5)...", model_name)
    calibrated = CalibratedClassifierCV(
        pipeline, method="isotonic", cv=5
    )
    calibrated.fit(X_train, y_train)
    return calibrated


# ----------------------------------------------
# Prediction helpers
# ----------------------------------------------

def predict_with_threshold(
    model: Any,
    X: pd.DataFrame,
    threshold: float = DECISION_THRESHOLD,
) -> Tuple[np.ndarray, np.ndarray]:
    """Get clipped probabilities and threshold-based predictions.

    Returns:
        preds: binary predictions (0/1)
        probs: clipped P(Approved)
    """
    raw_probs = model.predict_proba(X)[:, 1]
    probs = np.clip(raw_probs, PROB_CLIP[0], PROB_CLIP[1])
    preds = (probs >= threshold).astype(int)
    return preds, probs


# ----------------------------------------------
# Evaluation
# ----------------------------------------------

def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    threshold: float = DECISION_THRESHOLD,
) -> Dict[str, Any]:
    """Compute all metrics for one model on the test set.

    Returns a dict with accuracy, precision, recall, f1, roc_auc,
    confusion_matrix, and cross-validation results.
    """
    preds, probs = predict_with_threshold(model, X_test, threshold)

    metrics: Dict[str, Any] = {
        "model_name": model_name,
        "threshold": threshold,
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision": float(precision_score(y_test, preds, zero_division=0)),
        "recall": float(recall_score(y_test, preds, zero_division=0)),
        "f1": float(f1_score(y_test, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probs)),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }

    logger.info(
        "%s  -- Acc: %.4f | Prec: %.4f | Rec: %.4f | F1: %.4f | AUC: %.4f",
        model_name, metrics["accuracy"], metrics["precision"],
        metrics["recall"], metrics["f1"], metrics["roc_auc"],
    )
    return metrics


def cross_validate_model(
    model: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_name: str,
) -> Dict[str, Any]:
    """5-fold stratified cross-validation returning mean +/- std of accuracy and F1."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = cross_validate(
        model, X_train, y_train,
        cv=cv,
        scoring=["accuracy", "f1"],
        return_train_score=False,
    )
    results = {
        "cv_accuracy_mean": float(np.mean(cv_results["test_accuracy"])),
        "cv_accuracy_std": float(np.std(cv_results["test_accuracy"])),
        "cv_f1_mean": float(np.mean(cv_results["test_f1"])),
        "cv_f1_std": float(np.std(cv_results["test_f1"])),
    }
    logger.info(
        "%s CV  -- Accuracy: %.4f +/- %.4f | F1: %.4f +/- %.4f",
        model_name,
        results["cv_accuracy_mean"], results["cv_accuracy_std"],
        results["cv_f1_mean"], results["cv_f1_std"],
    )
    return results


def threshold_sweep(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    thresholds: List[float],
) -> List[Dict[str, float]]:
    """Sweep thresholds and compute precision/recall/F1 for each."""
    raw_probs = model.predict_proba(X_test)[:, 1]
    probs = np.clip(raw_probs, PROB_CLIP[0], PROB_CLIP[1])

    results = []
    for t in thresholds:
        preds = (probs >= t).astype(int)
        results.append({
            "threshold": t,
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
        })
    return results


# ----------------------------------------------
# Feature names and importance
# ----------------------------------------------

def get_feature_names(
    preprocessor: ColumnTransformer,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> List[str]:
    """Get output feature names from a fitted preprocessor."""
    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError:
        # Fallback: construct manually
        ohe = preprocessor.named_transformers_["cat"]
        cat_names = list(ohe.get_feature_names_out(categorical_cols))
        return numeric_cols + cat_names


def get_global_importance(
    model: Any,
    feature_names: List[str],
    numeric_cols: List[str],
    categorical_cols: List[str],
    model_name: str,
    X_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
) -> List[Dict[str, Any]]:
    """Extract global feature importance mapped to ORIGINAL feature names.

    For tree-based models: feature_importances_
    For LR: |coef_|
    Aggregates one-hot columns back to original categorical names.

    Returns list of {feature, importance} sorted descending.
    """
    # Get the raw classifier out of Pipeline / CalibratedClassifierCV
    clf = _extract_classifier(model)

    importances: np.ndarray | None = None
    use_raw_columns = False  # flag: permutation importance uses X columns directly

    if hasattr(clf, "feature_importances_"):
        importances = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        importances = np.abs(clf.coef_).flatten()
    else:
        # Fallback: permutation importance (operates on raw X_test columns)
        if X_test is not None and y_test is not None:
            from sklearn.inspection import permutation_importance
            perm = permutation_importance(
                model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE
            )
            importances = perm.importances_mean
            use_raw_columns = True
            logger.info("Used permutation importance for %s.", model_name)

    if importances is None:
        logger.warning("Could not extract importance for %s.", model_name)
        return []

    # Choose the right name list
    if use_raw_columns and X_test is not None:
        names = list(X_test.columns)
    else:
        names = feature_names

    if len(importances) != len(names):
        logger.warning(
            "Importance length (%d) != feature names (%d). Skipping aggregation.",
            len(importances), len(names),
        )
        return []

    imp_dict: Dict[str, float] = {}
    for name, imp in zip(names, importances):
        # Aggregate OHE columns: "cat__education_Graduate" -> "education"
        original = _map_to_original(name, numeric_cols, categorical_cols)
        imp_dict[original] = imp_dict.get(original, 0.0) + float(imp)

    # Sort descending
    sorted_imp = sorted(imp_dict.items(), key=lambda x: x[1], reverse=True)
    return [{"feature": f, "importance": v} for f, v in sorted_imp]


def _extract_classifier(model: Any) -> Any:
    """Unwrap the raw classifier from Pipeline or CalibratedClassifierCV."""
    obj = model

    # CalibratedClassifierCV wraps around the pipeline
    if hasattr(obj, "calibrated_classifiers_"):
        # Get the base estimator's classifier
        base = obj.estimator
        if isinstance(base, Pipeline):
            return base.named_steps["classifier"]
        return base

    # Plain Pipeline
    if isinstance(obj, Pipeline):
        return obj.named_steps["classifier"]

    return obj


def _map_to_original(
    transformed_name: str,
    numeric_cols: List[str],
    categorical_cols: List[str],
) -> str:
    """Map a transformed feature name back to the original column name.

    Examples:
      'num__cibil_score' -> 'cibil_score'
      'cat__education_Graduate' -> 'education'
    """
    # Strip prefix
    if "__" in transformed_name:
        _, bare = transformed_name.split("__", 1)
    else:
        bare = transformed_name

    # Check if it's a known numeric
    if bare in numeric_cols:
        return bare

    # Check if it matches a categorical OHE pattern
    for cat in categorical_cols:
        if bare.startswith(cat + "_") or bare == cat:
            return cat

    return bare
