"""
config.py - Single source of truth for all project configuration.

This module centralises every tuneable parameter so that train.py,
data_layer.py, model_layer.py, xai_engine.py and the Flask app all
import from one place.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

# ----------------------------------------------
# Paths
# ----------------------------------------------
BASE_DIR: Path = Path(__file__).resolve().parent
DATA_DIR: Path = BASE_DIR / "data"
ARTIFACTS_DIR: Path = BASE_DIR / "artifacts"
FIGURES_DIR: Path = ARTIFACTS_DIR / "figures"
DATASET_PATH: Path = DATA_DIR / "loan_approval_dataset.csv"

# ----------------------------------------------
# Reproducibility
# ----------------------------------------------
RANDOM_STATE: int = 42

# ----------------------------------------------
# Train / test split
# ----------------------------------------------
TEST_SIZE: float = 0.2  # stratified split

# ----------------------------------------------
# Decision / calibration
# ----------------------------------------------
DECISION_THRESHOLD: float = 0.5
PROB_CLIP: Tuple[float, float] = (0.05, 0.95)

# Threshold sweep range (inclusive endpoints, step 0.1)
THRESHOLD_SWEEP: List[float] = [round(x * 0.1, 1) for x in range(1, 10)]

# ----------------------------------------------
# Domain assumptions
# ----------------------------------------------
ASSUMED_ANNUAL_INTEREST_RATE: float = 0.10
LOAN_TERM_UNIT: str = os.getenv("LOAN_TERM_UNIT", "years")  # "years" | "months"

# ----------------------------------------------
# Risk tiers  (probability of approval)
# ----------------------------------------------
RISK_TIERS: Dict[str, Tuple[float, float]] = {
    "Prime":      (0.80, 1.01),   # p >= 0.80
    "Near Prime": (0.60, 0.80),   # 0.60 <= p < 0.80
    "Subprime":   (0.40, 0.60),   # 0.40 <= p < 0.60
    "High Risk":  (0.00, 0.40),   # p < 0.40
}


def get_risk_tier(prob: float) -> str:
    """Return the risk tier label for a given approval probability."""
    for tier, (lo, hi) in RISK_TIERS.items():
        if lo <= prob < hi:
            return tier
    return "High Risk"


# ----------------------------------------------
# SHAP
# ----------------------------------------------
SHAP_BACKGROUND_SIZE: int = 25

# Significance bucket cutoffs  (fraction of max |shap| value)
SHAP_SIGNIFICANCE_CUTOFFS: Dict[str, float] = {
    "Very High": 0.75,
    "High":      0.50,
    "Moderate":  0.25,
    "Low":       0.00,
}

# ----------------------------------------------
# Model hyperparameters
# ----------------------------------------------
LR_PARAMS: Dict = {
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

RF_PARAMS: Dict = {
    "n_estimators": 200,
    "max_depth": 10,
    "min_samples_split": 10,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

DT_PARAMS: Dict = {
    "max_depth": 8,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

# ----------------------------------------------
# Target encoding
# ----------------------------------------------
TARGET_COL: str = "loan_status"
TARGET_MAP: Dict[str, int] = {"Approved": 1, "Rejected": 0}

# ----------------------------------------------
# Columns
# ----------------------------------------------
DROP_COLS: List[str] = ["loan_id"]

CATEGORICAL_COLS: List[str] = ["education", "self_employed"]

RAW_NUMERIC_COLS: List[str] = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

ENGINEERED_COLS: List[str] = [
    "loan_to_income",
    "emi",
    "debt_income_ratio",
    "income_per_dep",
    "total_assets",
    "asset_coverage",
]

ASSET_COLS: List[str] = [
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]

# All numeric features seen by the preprocessor
ALL_NUMERIC_COLS: List[str] = RAW_NUMERIC_COLS + ENGINEERED_COLS

# ----------------------------------------------
# Feature-name mapping for human-readable output
# ----------------------------------------------
FEATURE_DISPLAY_NAMES: Dict[str, str] = {
    "cibil_score": "credit (CIBIL) score",
    "income_annum": "annual income",
    "loan_amount": "loan amount",
    "loan_term": "loan term",
    "no_of_dependents": "number of dependents",
    "residential_assets_value": "residential asset value",
    "commercial_assets_value": "commercial asset value",
    "luxury_assets_value": "luxury asset value",
    "bank_asset_value": "bank asset value",
    "loan_to_income": "loan-to-income ratio",
    "emi": "estimated monthly instalment (EMI)",
    "debt_income_ratio": "debt-to-income ratio",
    "income_per_dep": "income per dependent",
    "total_assets": "total asset value",
    "asset_coverage": "asset coverage ratio",
    "education": "education level",
    "self_employed": "self-employment status",
}
