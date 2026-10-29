"""
data_layer.py - Data loading, cleaning, validation, and feature engineering.

All data-related operations live here so that the same logic is used during
training (full DataFrame) and at inference time (single applicant dict).
"""
from __future__ import annotations

import logging
from typing import Dict, Union

import numpy as np
import pandas as pd

from backend.config import (
    ASSET_COLS,
    ASSUMED_ANNUAL_INTEREST_RATE,
    CATEGORICAL_COLS,
    DATASET_PATH,
    DROP_COLS,
    LOAN_TERM_UNIT,
    TARGET_COL,
    TARGET_MAP,
)

logger = logging.getLogger(__name__)


# ----------------------------------------------
# Loading and cleaning
# ----------------------------------------------

def load_dataset(path: str | None = None) -> pd.DataFrame:
    """Load the CSV, strip whitespace, encode target, drop identifiers,
    clean negatives, validate ranges, and return a ready-to-use DataFrame.

    Raises:
        FileNotFoundError: if the dataset file is missing.
    """
    csv_path = path or str(DATASET_PATH)
    if not DATASET_PATH.exists() and path is None:
        raise FileNotFoundError(
            f"Dataset not found at {DATASET_PATH}.\n"
            "Download it from Kaggle and place it at backend/data/loan_approval_dataset.csv.\n"
            "See scripts/download_data.md for instructions."
        )

    df = pd.read_csv(csv_path)
    initial_rows = len(df)
    logger.info("Loaded %d rows from %s", initial_rows, csv_path)

    # --- Strip whitespace from column names and string columns ---
    df.columns = df.columns.str.strip()
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()
    logger.info("Stripped whitespace from column names and string values.")

    # --- Drop identifier columns ---
    existing_drop = [c for c in DROP_COLS if c in df.columns]
    if existing_drop:
        df = df.drop(columns=existing_drop)
        logger.info("Dropped columns: %s", existing_drop)

    # --- Drop duplicates ---
    n_dup = df.duplicated().sum()
    if n_dup > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        logger.info("Removed %d duplicate rows.", n_dup)
    else:
        logger.info("No duplicate rows found.")

    # --- Impute missing values ---
    df = _impute_missing(df)

    # --- Handle negative asset values ---
    df = _clip_negative_assets(df)

    # --- Validate ranges ---
    df = _validate_ranges(df)

    # --- Encode target ---
    if TARGET_COL in df.columns:
        unmapped = set(df[TARGET_COL].unique()) - set(TARGET_MAP.keys())
        if unmapped:
            logger.warning("Unknown target values will be dropped: %s", unmapped)
            df = df[df[TARGET_COL].isin(TARGET_MAP.keys())].reset_index(drop=True)
        df[TARGET_COL] = df[TARGET_COL].map(TARGET_MAP)
        logger.info(
            "Encoded target '%s': %s.  Distribution: %s",
            TARGET_COL,
            TARGET_MAP,
            df[TARGET_COL].value_counts().to_dict(),
        )

    # --- Verify loan_term unit ---
    _check_loan_term_unit(df)

    final_rows = len(df)
    logger.info(
        "Data loading complete. %d -> %d rows (%d removed).",
        initial_rows, final_rows, initial_rows - final_rows,
    )
    return df


def _impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute missing values: median for numeric, mode for categorical."""
    n_missing = df.isnull().sum()
    total_missing = n_missing.sum()
    if total_missing == 0:
        logger.info("No missing values found.")
        return df

    logger.info("Missing values per column:\n%s", n_missing[n_missing > 0])

    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if df[col].dtype in ("float64", "int64", "float32", "int32"):
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            logger.info("Imputed '%s' with median = %s", col, median_val)
        else:
            mode_val = df[col].mode()[0]
            df[col] = df[col].fillna(mode_val)
            logger.info("Imputed '%s' with mode = '%s'", col, mode_val)

    return df


def _clip_negative_assets(df: pd.DataFrame) -> pd.DataFrame:
    """Clip negative asset values to 0 and log the count."""
    for col in ASSET_COLS:
        if col not in df.columns:
            continue
        neg_mask = df[col] < 0
        n_neg = neg_mask.sum()
        if n_neg > 0:
            logger.warning(
                "'%s': %d negative values (min=%s) clipped to 0.",
                col, n_neg, df.loc[neg_mask, col].min(),
            )
            df[col] = df[col].clip(lower=0)
    return df


def _validate_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and fix value ranges. Log rows affected by each rule."""
    rules_applied = 0

    # income > 0
    if "income_annum" in df.columns:
        bad = df["income_annum"] <= 0
        n = bad.sum()
        if n > 0:
            logger.warning("income_annum <= 0: %d rows. Setting to 1.", n)
            df.loc[bad, "income_annum"] = 1
            rules_applied += n

    # loan_amount > 0
    if "loan_amount" in df.columns:
        bad = df["loan_amount"] <= 0
        n = bad.sum()
        if n > 0:
            logger.warning("loan_amount <= 0: %d rows. Setting to 1.", n)
            df.loc[bad, "loan_amount"] = 1
            rules_applied += n

    # cibil_score in [300, 900]
    if "cibil_score" in df.columns:
        bad = (df["cibil_score"] < 300) | (df["cibil_score"] > 900)
        n = bad.sum()
        if n > 0:
            logger.warning("cibil_score out of [300, 900]: %d rows. Clipping.", n)
            df["cibil_score"] = df["cibil_score"].clip(300, 900)
            rules_applied += n

    # dependents >= 0
    if "no_of_dependents" in df.columns:
        bad = df["no_of_dependents"] < 0
        n = bad.sum()
        if n > 0:
            logger.warning("no_of_dependents < 0: %d rows. Setting to 0.", n)
            df.loc[bad, "no_of_dependents"] = 0
            rules_applied += n

    if rules_applied == 0:
        logger.info("All range validations passed.")
    return df


def _check_loan_term_unit(df: pd.DataFrame) -> None:
    """Log a sanity check about loan_term unit."""
    if "loan_term" not in df.columns:
        return
    min_t, max_t = df["loan_term"].min(), df["loan_term"].max()
    logger.info(
        "loan_term range: %s - %s.  Configured unit: %s.  "
        "Range 2-20 is consistent with YEARS.",
        min_t, max_t, LOAN_TERM_UNIT,
    )


# ----------------------------------------------
# Feature engineering
# ----------------------------------------------

def engineer_features(data: Union[pd.DataFrame, Dict]) -> pd.DataFrame:
    """Create the six engineered features.

    Accepts either:
      - a full DataFrame (training/evaluation)
      - a single applicant dict  (inference)

    Always returns a DataFrame (single row or many rows).
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = data.copy()

    # Ensure numeric types for computation
    numeric_cols = [
        "income_annum", "loan_amount", "loan_term", "no_of_dependents",
        "residential_assets_value", "commercial_assets_value",
        "luxury_assets_value", "bank_asset_value",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # 1. loan_to_income
    df["loan_to_income"] = df["loan_amount"] / (df["income_annum"] + 1)

    # 2. EMI  =  P * r * (1+r)^n  /  ((1+r)^n - 1)
    r_monthly = ASSUMED_ANNUAL_INTEREST_RATE / 12
    # Convert loan_term to months
    if LOAN_TERM_UNIT == "years":
        n_months = df["loan_term"] * 12
    else:
        n_months = df["loan_term"]

    if r_monthly == 0:
        df["emi"] = df["loan_amount"] / n_months.replace(0, np.nan)
    else:
        compound = (1 + r_monthly) ** n_months
        df["emi"] = (df["loan_amount"] * r_monthly * compound) / (compound - 1)

    # Handle edge cases (NaN / inf)
    df["emi"] = df["emi"].replace([np.inf, -np.inf], np.nan).fillna(0)

    # 3. debt_income_ratio  =  (EMI * 12) / (income_annum + 1)
    df["debt_income_ratio"] = (df["emi"] * 12) / (df["income_annum"] + 1)

    # 4. income_per_dep  =  income_annum / (no_of_dependents + 1)
    df["income_per_dep"] = df["income_annum"] / (df["no_of_dependents"] + 1)

    # 5. total_assets
    df["total_assets"] = (
        df["residential_assets_value"]
        + df["commercial_assets_value"]
        + df["luxury_assets_value"]
        + df["bank_asset_value"]
    )

    # 6. asset_coverage  =  total_assets / (loan_amount + 1)
    df["asset_coverage"] = df["total_assets"] / (df["loan_amount"] + 1)

    return df


def prepare_dataset(path: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Full pipeline: load -> clean -> engineer -> return (df_full, X, y).

    Returns:
        df_full: DataFrame with all columns including engineered features and target.
        X: Feature DataFrame (no target).
        y: Target Series.
    """
    df = load_dataset(path)
    df = engineer_features(df)

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    logger.info("Final feature matrix shape: %s, target shape: %s", X.shape, y.shape)
    return df, X, y
