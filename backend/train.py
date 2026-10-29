"""
train.py - End-to-end training script.

Loads data, engineers features, builds/calibrates/evaluates three models,
selects best by F1, saves artifacts, and prints a comparison table.

Usage:
    python -m backend.train
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from backend.config import (
    ALL_NUMERIC_COLS,
    ARTIFACTS_DIR,
    CATEGORICAL_COLS,
    DECISION_THRESHOLD,
    FIGURES_DIR,
    RANDOM_STATE,
    SHAP_BACKGROUND_SIZE,
    TEST_SIZE,
    THRESHOLD_SWEEP,
)
from backend.data_layer import prepare_dataset
from backend.model_layer import (
    build_pipelines,
    build_preprocessor,
    calibrate_pipeline,
    cross_validate_model,
    evaluate_model,
    get_feature_names,
    get_global_importance,
    predict_with_threshold,
    threshold_sweep,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main training entry point."""
    t_start = time.time()

    # -- 1. Prepare data --------------------------------------
    logger.info("=" * 60)
    logger.info("PHASE 1: Loading and preparing data")
    logger.info("=" * 60)
    df_full, X, y = prepare_dataset()

    # -- 2. Train/test split ----------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Train: %d rows | Test: %d rows", len(X_train), len(X_test))

    # -- 3. Build preprocessor & pipelines --------------------
    logger.info("=" * 60)
    logger.info("PHASE 2: Building models")
    logger.info("=" * 60)
    preprocessor, numeric_cols, categorical_cols = build_preprocessor(X_train)
    pipelines = build_pipelines(preprocessor)

    # -- 4. Train, calibrate, evaluate ------------------------
    all_metrics: dict = {}
    trained_models: dict = {}
    cv_results_all: dict = {}

    for name, pipeline in pipelines.items():
        logger.info("-" * 40)
        logger.info("Training: %s", name)
        logger.info("-" * 40)

        # Cross-validate on the uncalibrated pipeline (needs to be refitted)
        cv_res = cross_validate_model(pipeline, X_train, y_train, name)
        cv_results_all[name] = cv_res

        # Calibrate (fits the model internally)
        calibrated = calibrate_pipeline(pipeline, X_train, y_train, name)
        trained_models[name] = calibrated

        # Evaluate on test set
        metrics = evaluate_model(calibrated, X_test, y_test, name)
        metrics.update(cv_res)
        all_metrics[name] = metrics

    # -- 5. Feature importance --------------------------------
    logger.info("=" * 60)
    logger.info("Computing feature importance")
    logger.info("=" * 60)

    # Get feature names from a fitted preprocessor
    # For RF (Pipeline), we can directly get preprocessor; for calibrated, use
    # a separate fit
    preprocessor_fitted, _, _ = build_preprocessor(X_train)
    preprocessor_fitted.fit(X_train)
    feature_names = get_feature_names(preprocessor_fitted, numeric_cols, categorical_cols)

    for name, model in trained_models.items():
        importance = get_global_importance(
            model, feature_names, numeric_cols, categorical_cols, name,
            X_test, y_test,
        )
        top5 = importance[:5]
        all_metrics[name]["top5_importance"] = top5
        logger.info("%s top-5: %s", name, [(d["feature"], f'{d["importance"]:.4f}') for d in top5])

    # -- 6. Leakage check ------------------------------------
    logger.info("=" * 60)
    logger.info("Leakage check")
    logger.info("=" * 60)
    for name, m in all_metrics.items():
        if m["accuracy"] > 0.99:
            logger.warning(
                "WARNING: %s accuracy > 99%% (%.4f). Possible data leakage!",
                name, m["accuracy"],
            )
            assert "loan_status" not in X_train.columns, "Target column found in X!"

    # -- 7. Ablation: RF without cibil_score ------------------
    logger.info("-" * 40)
    logger.info("Ablation: Random Forest without cibil_score")
    logger.info("-" * 40)
    ablation_cols = [c for c in X_train.columns if c != "cibil_score"]
    X_train_abl = X_train[ablation_cols]
    X_test_abl = X_test[ablation_cols]

    preprocessor_abl, num_abl, cat_abl = build_preprocessor(X_train_abl)
    from sklearn.ensemble import RandomForestClassifier as RFC
    from backend.config import RF_PARAMS as _RF
    pipe_abl = __import__("sklearn.pipeline", fromlist=["Pipeline"]).Pipeline([
        ("preprocess", preprocessor_abl),
        ("classifier", RFC(**_RF)),
    ])
    pipe_abl.fit(X_train_abl, y_train)
    abl_metrics = evaluate_model(pipe_abl, X_test_abl, y_test, "RF (no cibil_score)")
    all_metrics["RF (no cibil_score) ablation"] = abl_metrics

    # -- 8. Best model by F1 ----------------------------------
    main_models = {k: v for k, v in all_metrics.items() if "ablation" not in k}
    best_name = max(main_models, key=lambda k: main_models[k]["f1"])
    best_model = trained_models[best_name]
    logger.info(">> Best model by F1: %s (F1=%.4f)", best_name, all_metrics[best_name]["f1"])

    # -- 9. Threshold sweep -----------------------------------
    logger.info("=" * 60)
    logger.info("Threshold sweep for best model")
    logger.info("=" * 60)
    sweep = threshold_sweep(best_model, X_test, y_test, THRESHOLD_SWEEP)
    all_metrics["threshold_sweep"] = sweep
    for s in sweep:
        logger.info(
            "  t=%.1f  Prec=%.4f  Rec=%.4f  F1=%.4f",
            s["threshold"], s["precision"], s["recall"], s["f1"],
        )

    # -- 10. Save artifacts -----------------------------------
    logger.info("=" * 60)
    logger.info("Saving artifacts")
    logger.info("=" * 60)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Save all three models + best
    for name, model in trained_models.items():
        safe_name = name.lower().replace(" ", "_")
        path = ARTIFACTS_DIR / f"{safe_name}.joblib"
        joblib.dump(model, path)
        logger.info("Saved %s -> %s", name, path)

    joblib.dump(best_model, ARTIFACTS_DIR / "best_model.joblib")
    logger.info("Saved best model -> best_model.joblib")

    # Background set for SHAP (stratified sample)
    bg_idx = y_train.reset_index(drop=True).groupby(y_train.reset_index(drop=True)).apply(
        lambda g: g.sample(
            min(len(g), SHAP_BACKGROUND_SIZE // 2 + 1),
            random_state=RANDOM_STATE,
        )
    ).index.get_level_values(1)
    background = X_train.iloc[bg_idx[:SHAP_BACKGROUND_SIZE]].reset_index(drop=True)
    joblib.dump(background, ARTIFACTS_DIR / "background.joblib")
    logger.info("Saved SHAP background set (%d rows)", len(background))

    # Save test set for figure generation
    joblib.dump((X_test, y_test), ARTIFACTS_DIR / "test_set.joblib")

    # Save train set for figure generation
    joblib.dump((X_train, y_train), ARTIFACTS_DIR / "train_set.joblib")

    # Metrics JSON
    all_metrics["best_model"] = best_name

    # Measure latency
    latencies = []
    sample = X_test.iloc[:20]
    for _ in range(5):
        t0 = time.time()
        best_model.predict_proba(sample)
        latencies.append((time.time() - t0) / len(sample) * 1000)
    avg_latency = float(np.mean(latencies))
    all_metrics["avg_latency_ms_per_sample"] = round(avg_latency, 2)
    logger.info("Avg latency: %.2f ms/sample", avg_latency)

    metrics_path = ARTIFACTS_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=_json_default)
    logger.info("Saved metrics -> %s", metrics_path)

    # Save preprocessor info for XAI engine
    joblib.dump(
        {"numeric_cols": numeric_cols, "categorical_cols": categorical_cols,
         "feature_names": feature_names},
        ARTIFACTS_DIR / "preprocessor_info.joblib",
    )

    # -- 11. Print comparison table ---------------------------
    _print_comparison(all_metrics)

    elapsed = time.time() - t_start
    logger.info("Training completed in %.1f seconds.", elapsed)


def _print_comparison(metrics: dict) -> None:
    """Print a formatted comparison table to stdout."""
    print("\n" + "=" * 90)
    print("MODEL COMPARISON")
    print("=" * 90)
    header = f"{'Model':<30} {'Acc':>7} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7} {'CV-F1':>12}"
    print(header)
    print("-" * 90)

    for name, m in metrics.items():
        if name in ("threshold_sweep", "best_model", "avg_latency_ms_per_sample"):
            continue
        cv_f1 = ""
        if "cv_f1_mean" in m:
            cv_f1 = f"{m['cv_f1_mean']:.4f}+/-{m['cv_f1_std']:.4f}"
        print(
            f"{name:<30} {m['accuracy']:>7.4f} {m['precision']:>7.4f} "
            f"{m['recall']:>7.4f} {m['f1']:>7.4f} {m['roc_auc']:>7.4f} {cv_f1:>12}"
        )

    print("-" * 90)
    print(f">> Best model: {metrics.get('best_model', 'N/A')}")
    print(f"   Avg latency: {metrics.get('avg_latency_ms_per_sample', 'N/A')} ms/sample")
    print("=" * 90 + "\n")


def _json_default(obj):
    """JSON serialization helper for numpy types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


if __name__ == "__main__":
    main()
