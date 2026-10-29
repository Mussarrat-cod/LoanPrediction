"""
test_api.py -- Flask API integration tests with the test client.

Covers valid, invalid, and edge-case requests for /api/predict,
/api/v1/loan/explain, /api/health, and /api/metrics.
"""
from __future__ import annotations

import json

import pytest

from backend.app import create_app


@pytest.fixture(scope="module")
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


VALID_PAYLOAD = {
    "no_of_dependents": 2,
    "education": "Graduate",
    "self_employed": "No",
    "income_annum": 9600000,
    "loan_amount": 29900000,
    "loan_term": 12,
    "cibil_score": 778,
    "residential_assets_value": 2400000,
    "commercial_assets_value": 17600000,
    "luxury_assets_value": 22700000,
    "bank_asset_value": 8000000,
}


class TestHealth:
    def test_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["status"] == "healthy"


class TestMetrics:
    def test_metrics(self, client):
        resp = client.get("/api/metrics")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "best_model" in data


class TestPredict:
    def test_valid_predict(self, client):
        resp = client.post(
            "/api/predict",
            data=json.dumps(VALID_PAYLOAD),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["decision"] in ("Approved", "Rejected")
        assert 0 <= data["probability"] <= 1
        assert "risk_tier" in data
        assert "model_name" in data

    def test_missing_field(self, client):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "cibil_score"}
        resp = client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert "cibil_score" in str(data)

    def test_invalid_education(self, client):
        payload = {**VALID_PAYLOAD, "education": "PhD"}
        resp = client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_cibil_out_of_range(self, client):
        payload = {**VALID_PAYLOAD, "cibil_score": 100}
        resp = client.post(
            "/api/predict",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_empty_body(self, client):
        resp = client.post(
            "/api/predict",
            data="",
            content_type="application/json",
        )
        assert resp.status_code == 400


class TestExplain:
    def test_valid_explain(self, client):
        resp = client.post(
            "/api/v1/loan/explain",
            data=json.dumps(VALID_PAYLOAD),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert "local_contributions" in data
        assert "natural_language_explanation" in data
        assert "recommendations" in data
        assert "global_importance" in data

    def test_explain_missing_field(self, client):
        payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "loan_amount"}
        resp = client.post(
            "/api/v1/loan/explain",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 400
