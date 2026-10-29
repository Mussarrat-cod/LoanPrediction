"""
app.py -- Flask REST API for loan approval prediction with XAI.

App factory pattern, flask-cors, strict input validation,
artifacts loaded once at startup.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

from backend.config import ARTIFACTS_DIR, DECISION_THRESHOLD
from backend.xai_engine import XAIEngine

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------
# Input validation schema
# ---------------------------------------------------------------
REQUIRED_FIELDS: Dict[str, Dict[str, Any]] = {
    "no_of_dependents":          {"type": (int, float), "min": 0,   "max": 20},
    "education":                 {"type": str, "allowed": ["Graduate", "Not Graduate"]},
    "self_employed":             {"type": str, "allowed": ["Yes", "No"]},
    "income_annum":              {"type": (int, float), "min": 0,   "max": 1e12},
    "loan_amount":               {"type": (int, float), "min": 1,   "max": 1e12},
    "loan_term":                 {"type": (int, float), "min": 1,   "max": 50},
    "cibil_score":               {"type": (int, float), "min": 300, "max": 900},
    "residential_assets_value":  {"type": (int, float), "min": 0,   "max": 1e12},
    "commercial_assets_value":   {"type": (int, float), "min": 0,   "max": 1e12},
    "luxury_assets_value":       {"type": (int, float), "min": 0,   "max": 1e12},
    "bank_asset_value":          {"type": (int, float), "min": 0,   "max": 1e12},
}


def validate_input(data: Optional[Dict]) -> tuple[Dict, list]:
    """Validate request body against the schema.

    Returns (cleaned_data, errors).
    """
    if data is None:
        return {}, ["Request body must be valid JSON."]

    errors = []
    cleaned: Dict[str, Any] = {}

    for field, rules in REQUIRED_FIELDS.items():
        if field not in data:
            errors.append(f"Missing required field: '{field}'.")
            continue

        value = data[field]

        # Type check
        expected = rules["type"]
        if isinstance(expected, tuple):
            if not isinstance(value, expected):
                errors.append(
                    f"'{field}' must be a number, got {type(value).__name__}."
                )
                continue
        elif expected is str:
            if not isinstance(value, str):
                errors.append(
                    f"'{field}' must be a string, got {type(value).__name__}."
                )
                continue

        # Allowed values (categorical)
        if "allowed" in rules:
            if value not in rules["allowed"]:
                errors.append(
                    f"'{field}' must be one of {rules['allowed']}, got '{value}'."
                )
                continue

        # Range checks (numeric)
        if "min" in rules and isinstance(value, (int, float)):
            if value < rules["min"]:
                errors.append(
                    f"'{field}' must be >= {rules['min']}, got {value}."
                )
                continue
        if "max" in rules and isinstance(value, (int, float)):
            if value > rules["max"]:
                errors.append(
                    f"'{field}' must be <= {rules['max']}, got {value}."
                )
                continue

        cleaned[field] = value

    return cleaned, errors


def create_app() -> Flask:
    """Application factory."""
    app = Flask(__name__)
    CORS(app)

    # ---- Load artifacts once at startup ----
    model_path = ARTIFACTS_DIR / "best_model.joblib"
    background_path = ARTIFACTS_DIR / "background.joblib"
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    preproc_path = ARTIFACTS_DIR / "preprocessor_info.joblib"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Run `python -m backend.train` first."
        )

    model = joblib.load(model_path)
    background = joblib.load(background_path)
    preproc_info = joblib.load(preproc_path)

    with open(metrics_path, "r") as f:
        metrics_data = json.load(f)

    # Build XAI engine
    feature_names = list(background.columns)
    engine = XAIEngine(model, background, feature_names)
    model_name = metrics_data.get("best_model", "Unknown")

    logger.info("Loaded model '%s' and XAI engine.", model_name)

    # ---- Routes ----

    @app.route("/api/health", methods=["GET"])
    def health():
        return jsonify({"status": "healthy", "model": model_name})

    @app.route("/api/metrics", methods=["GET"])
    def metrics():
        return jsonify(metrics_data)

    @app.route("/api/predict", methods=["POST"])
    def predict():
        data = request.get_json(silent=True)
        cleaned, errors = validate_input(data)
        if errors:
            return jsonify({"error": "Validation failed", "details": errors}), 400

        result = engine.predict(cleaned)
        result["model_name"] = model_name
        return jsonify(result)

    @app.route("/api/v1/loan/explain", methods=["POST"])
    def explain():
        data = request.get_json(silent=True)
        cleaned, errors = validate_input(data)
        if errors:
            return jsonify({"error": "Validation failed", "details": errors}), 400

        # Prediction
        result = engine.predict(cleaned)
        result["model_name"] = model_name

        # Local SHAP explanation
        local = engine.explain_local(cleaned)
        result["local_contributions"] = local

        # Natural-language explanation
        result["natural_language_explanation"] = engine.generate_nl_explanation(
            local, result["decision"], result["risk_tier"], result["probability"]
        )

        # Recommendations
        result["recommendations"] = engine.generate_recommendations(local)

        # Global importance
        intrinsic = metrics_data.get(model_name, {}).get("top5_importance", [])
        global_imp = engine.explain_global(intrinsic_importance=intrinsic)
        result["global_importance"] = global_imp

        return jsonify(result)

    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_app()
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
