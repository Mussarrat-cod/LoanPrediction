"""
test_xai_engine.py -- Tests for XAI engine: SHAP accuracy, local/global
explanations, NL explanation, and recommendations.
"""
from __future__ import annotations

import joblib
import pytest

from backend.config import ARTIFACTS_DIR
from backend.xai_engine import XAIEngine


# ---- Fixtures ----

SAMPLE_APPROVED = {
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

SAMPLE_REJECTED = {
    "no_of_dependents": 0,
    "education": "Not Graduate",
    "self_employed": "Yes",
    "income_annum": 4_100_000,
    "loan_amount": 12_200_000,
    "loan_term": 8,
    "cibil_score": 417,
    "residential_assets_value": 2_700_000,
    "commercial_assets_value": 2_200_000,
    "luxury_assets_value": 8_800_000,
    "bank_asset_value": 3_300_000,
}


@pytest.fixture(scope="module")
def engine():
    """Load artifacts and create XAI engine once for all tests."""
    model = joblib.load(ARTIFACTS_DIR / "best_model.joblib")
    background = joblib.load(ARTIFACTS_DIR / "background.joblib")
    return XAIEngine(model, background, list(background.columns))


# ---- Tests ----

class TestPrediction:
    def test_approved_prediction(self, engine):
        result = engine.predict(SAMPLE_APPROVED)
        assert result["decision"] == "Approved"
        assert 0 <= result["probability"] <= 1
        assert result["risk_tier"] in ("Prime", "Near Prime", "Subprime", "High Risk")

    def test_rejected_prediction(self, engine):
        result = engine.predict(SAMPLE_REJECTED)
        assert result["decision"] == "Rejected"
        assert 0 <= result["probability"] <= 1

    def test_engineered_features_in_result(self, engine):
        result = engine.predict(SAMPLE_APPROVED)
        eng = result["engineered_features"]
        assert "emi" in eng
        assert "loan_to_income" in eng
        assert "total_assets" in eng


class TestLocalExplanation:
    def test_local_returns_list(self, engine):
        local = engine.explain_local(SAMPLE_APPROVED)
        assert isinstance(local, list)
        assert len(local) > 0

    def test_local_has_required_keys(self, engine):
        local = engine.explain_local(SAMPLE_APPROVED)
        keys = {"feature", "display_name", "shap_value", "feature_value",
                "direction", "significance"}
        for c in local:
            assert keys.issubset(c.keys())

    def test_local_sorted_by_abs_shap(self, engine):
        local = engine.explain_local(SAMPLE_APPROVED)
        abs_vals = [abs(c["shap_value"]) for c in local]
        assert abs_vals == sorted(abs_vals, reverse=True)


class TestSHAPAccuracy:
    def test_shap_local_accuracy_approved(self, engine):
        result = engine.check_shap_accuracy(SAMPLE_APPROVED, tol=1e-2)
        assert result["within_tolerance"], (
            f"SHAP accuracy failed: diff={result['difference']}"
        )

    def test_shap_local_accuracy_rejected(self, engine):
        result = engine.check_shap_accuracy(SAMPLE_REJECTED, tol=1e-2)
        assert result["within_tolerance"], (
            f"SHAP accuracy failed: diff={result['difference']}"
        )


class TestNLExplanation:
    def test_nl_contains_decision(self, engine):
        local = engine.explain_local(SAMPLE_REJECTED)
        pred = engine.predict(SAMPLE_REJECTED)
        text = engine.generate_nl_explanation(
            local, pred["decision"], pred["risk_tier"], pred["probability"]
        )
        assert "REJECTED" in text or "APPROVED" in text

    def test_nl_contains_factors(self, engine):
        local = engine.explain_local(SAMPLE_REJECTED)
        pred = engine.predict(SAMPLE_REJECTED)
        text = engine.generate_nl_explanation(
            local, pred["decision"], pred["risk_tier"], pred["probability"]
        )
        # Should mention factors
        assert "factors" in text.lower() or "impact" in text.lower()


class TestRecommendations:
    def test_recommendations_for_rejected(self, engine):
        local = engine.explain_local(SAMPLE_REJECTED)
        recs = engine.generate_recommendations(local)
        assert isinstance(recs, list)
        # A rejected applicant with low CIBIL should get recommendations
        assert len(recs) > 0

    def test_recommendation_structure(self, engine):
        local = engine.explain_local(SAMPLE_REJECTED)
        recs = engine.generate_recommendations(local)
        if recs:
            rec = recs[0]
            assert "title" in rec
            assert "detail" in rec
            assert "feature" in rec
