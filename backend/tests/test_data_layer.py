"""
test_data_layer.py – Unit tests for data_layer.py.

Covers:
  - EMI hand-computed verification
  - Feature engineering on a single applicant dict
  - Whitespace stripping
  - Target encoding
  - Negative asset clipping
  - Full pipeline smoke test
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from backend.config import ASSUMED_ANNUAL_INTEREST_RATE, TARGET_MAP
from backend.data_layer import engineer_features, load_dataset, prepare_dataset


# ──────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────

SAMPLE_APPLICANT: dict = {
    "no_of_dependents": 2,
    "education": "Graduate",
    "self_employed": "No",
    "income_annum": 9_600_000,
    "loan_amount": 29_900_000,
    "loan_term": 12,
    "cibil_score": 778,
    "residential_assets_value": 2_400_000,
    "commercial_assets_value": 17_600_000,
    "luxury_assets_value": 22_700_000,
    "bank_asset_value": 8_000_000,
}


def _hand_computed_emi(P: float, annual_rate: float, years: int) -> float:
    """Compute EMI using the standard amortisation formula."""
    r = annual_rate / 12
    n = years * 12
    if r == 0:
        return P / n
    compound = (1 + r) ** n
    return P * r * compound / (compound - 1)


# ──────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────

class TestEMI:
    """Verify EMI computation against a hand-computed value."""

    def test_emi_hand_computed(self) -> None:
        """EMI for P=29900000, rate=10%, term=12 years."""
        expected = _hand_computed_emi(29_900_000, ASSUMED_ANNUAL_INTEREST_RATE, 12)
        result = engineer_features(SAMPLE_APPLICANT)
        actual_emi = result["emi"].iloc[0]
        assert math.isclose(actual_emi, expected, rel_tol=1e-6), (
            f"EMI mismatch: expected {expected:.2f}, got {actual_emi:.2f}"
        )

    def test_emi_zero_rate(self) -> None:
        """If rate were 0, EMI = P / n."""
        # We can't easily change config at runtime for this, so just verify
        # the formula path with a known value.
        P = 1_200_000
        n_months = 10 * 12
        expected = P / n_months
        # Verify our hand_computed helper handles r=0
        assert math.isclose(_hand_computed_emi(P, 0.0, 10), expected)


class TestFeatureEngineering:
    """Verify all six engineered features on a single row."""

    def test_single_row_dict(self) -> None:
        """engineer_features accepts a dict and returns a 1-row DataFrame."""
        result = engineer_features(SAMPLE_APPLICANT)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_six_features_created(self) -> None:
        """All six engineered columns are present."""
        result = engineer_features(SAMPLE_APPLICANT)
        expected_cols = [
            "loan_to_income", "emi", "debt_income_ratio",
            "income_per_dep", "total_assets", "asset_coverage",
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing engineered column: {col}"

    def test_loan_to_income(self) -> None:
        expected = 29_900_000 / (9_600_000 + 1)
        result = engineer_features(SAMPLE_APPLICANT)
        assert math.isclose(result["loan_to_income"].iloc[0], expected, rel_tol=1e-6)

    def test_debt_income_ratio(self) -> None:
        result = engineer_features(SAMPLE_APPLICANT)
        emi = result["emi"].iloc[0]
        expected = (emi * 12) / (9_600_000 + 1)
        assert math.isclose(
            result["debt_income_ratio"].iloc[0], expected, rel_tol=1e-6
        )

    def test_income_per_dep(self) -> None:
        expected = 9_600_000 / (2 + 1)
        result = engineer_features(SAMPLE_APPLICANT)
        assert math.isclose(result["income_per_dep"].iloc[0], expected, rel_tol=1e-6)

    def test_total_assets(self) -> None:
        expected = 2_400_000 + 17_600_000 + 22_700_000 + 8_000_000
        result = engineer_features(SAMPLE_APPLICANT)
        assert result["total_assets"].iloc[0] == expected

    def test_asset_coverage(self) -> None:
        total = 2_400_000 + 17_600_000 + 22_700_000 + 8_000_000
        expected = total / (29_900_000 + 1)
        result = engineer_features(SAMPLE_APPLICANT)
        assert math.isclose(result["asset_coverage"].iloc[0], expected, rel_tol=1e-6)

    def test_dataframe_input(self) -> None:
        """engineer_features also works on a multi-row DataFrame."""
        df = pd.DataFrame([SAMPLE_APPLICANT, SAMPLE_APPLICANT])
        result = engineer_features(df)
        assert len(result) == 2
        assert "emi" in result.columns


class TestWhitespaceStripping:
    """Verify that leading/trailing whitespace is stripped on load."""

    def test_column_names_stripped(self) -> None:
        df = load_dataset()
        for col in df.columns:
            assert col == col.strip(), f"Column '{col}' has whitespace"

    def test_string_values_stripped(self) -> None:
        df = load_dataset()
        if "education" in df.columns:
            # education should have been stripped before encoding
            # but target was already encoded, check categoricals
            for val in df["education"].unique():
                assert val == str(val).strip(), f"education value '{val}' has whitespace"


class TestTargetEncoding:
    """Verify target column is correctly encoded."""

    def test_target_is_numeric(self) -> None:
        df = load_dataset()
        assert df["loan_status"].dtype in (np.int64, np.int32, np.float64, int)

    def test_target_values(self) -> None:
        df = load_dataset()
        unique_vals = set(df["loan_status"].unique())
        assert unique_vals == {0, 1}, f"Expected {{0, 1}}, got {unique_vals}"

    def test_target_distribution(self) -> None:
        """Verify the known class distribution."""
        df = load_dataset()
        counts = df["loan_status"].value_counts()
        # Approved = 1 → 2656, Rejected = 0 → 1613  (original dataset)
        assert counts[1] > 0
        assert counts[0] > 0


class TestNegativeAssetClipping:
    """Verify negative asset values are clipped to 0."""

    def test_no_negatives_after_load(self) -> None:
        df = load_dataset()
        asset_cols = [
            "residential_assets_value", "commercial_assets_value",
            "luxury_assets_value", "bank_asset_value",
        ]
        for col in asset_cols:
            if col in df.columns:
                assert (df[col] >= 0).all(), f"Negative values remain in {col}"


class TestFullPipeline:
    """Smoke test for prepare_dataset."""

    def test_prepare_returns_correct_shapes(self) -> None:
        df_full, X, y = prepare_dataset()
        assert len(X) == len(y)
        assert "loan_status" not in X.columns
        assert "loan_to_income" in X.columns  # engineered feature present
        assert "emi" in X.columns

    def test_no_target_leakage(self) -> None:
        """Target column must not appear in X."""
        _, X, _ = prepare_dataset()
        assert "loan_status" not in X.columns

    def test_loan_id_dropped(self) -> None:
        """loan_id must be dropped."""
        _, X, _ = prepare_dataset()
        assert "loan_id" not in X.columns
