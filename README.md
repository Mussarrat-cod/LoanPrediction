# Loan Approval Prediction System with Explainable AI (SHAP)

A full-stack ML system that predicts loan approval using three scikit-learn classifiers,
explains every decision with SHAP values, and serves results through a Flask REST API
and React frontend.

## Architecture

```
loan-xai/
├── backend/
│   ├── app.py              # Flask REST API (app factory, CORS, validation)
│   ├── config.py           # Single source of truth for all configuration
│   ├── data_layer.py       # Data loading, cleaning, validation, feature engineering
│   ├── model_layer.py      # Preprocessing, pipelines, calibration, evaluation
│   ├── xai_engine.py       # SHAP KernelExplainer, NL explanations, recommendations
│   ├── train.py            # End-to-end training script
│   ├── data/               # Dataset (not committed)
│   ├── artifacts/          # Trained models, metrics, background set, figures
│   └── tests/              # 40 pytest tests
├── frontend/               # React + Vite + Recharts
│   └── src/components/     # LoanForm, PredictionResult, FeatureImportance
├── scripts/
│   ├── make_figures.py     # Publication-quality figures
│   └── download_data.md    # Dataset download instructions
└── README.md
```

## Models

| Model | Accuracy | F1 | ROC-AUC | CV F1 (5-fold) |
|-------|----------|-----|---------|----------------|
| Logistic Regression | 94.38% | 95.48% | 98.00% | 94.25% +/- 1.45% |
| Random Forest | 99.77% | 99.81% | 100.0% | 99.81% +/- 0.12% |
| **Decision Tree** (Best) | **100.0%** | **100.0%** | **100.0%** | 99.98% +/- 0.05% |

> **Note:** The very high accuracy is driven by `cibil_score`, which is nearly perfectly
> predictive. The ablation study (RF without `cibil_score`) drops to 54.68% accuracy,
> confirming this. This is a known characteristic of this dataset, not a data leakage issue.

## Quick Start

### 1. Download the Dataset

See [scripts/download_data.md](scripts/download_data.md) for instructions.
Place the CSV at `backend/data/loan_approval_dataset.csv`.

### 2. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### 3. Train Models

```bash
python -m backend.train
```

This produces:
- `backend/artifacts/best_model.joblib` (and all three individual models)
- `backend/artifacts/metrics.json`
- `backend/artifacts/background.joblib` (SHAP background set)

### 4. Generate Figures (optional)

```bash
python -m scripts.make_figures
```

Produces 300-dpi PNGs in `backend/artifacts/figures/` and `backend/artifacts/paper_tables.md`.

### 5. Start the Backend

```bash
python -m backend.app
```

The API runs at `http://localhost:5000`.

### 6. Start the Frontend

```bash
cd frontend
npm install
npm run dev
```

The UI runs at `http://localhost:5173` with a Vite dev proxy to the backend.

### 7. Run Tests

```bash
python -m pytest backend/tests/ -v
```

40 tests covering data layer, XAI engine, and Flask API.

## API Endpoints

### `POST /api/predict`

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
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
    "bank_asset_value": 8000000
  }'
```

Response: `decision`, `probability`, `risk_tier`, `manual_review`, `threshold`, `model_name`, `latency_ms`.

### `POST /api/v1/loan/explain`

Same body as `/api/predict`. Response adds `local_contributions`, `natural_language_explanation`,
`recommendations`, and `global_importance`.

### `GET /api/health`

Returns `{"status": "healthy", "model": "..."}`.

### `GET /api/metrics`

Returns the full `metrics.json` (all model metrics, threshold sweep, ablation, etc.).

## Feature Engineering

Six derived features are computed at both training and inference time:

| Feature | Formula |
|---------|---------|
| `loan_to_income` | `loan_amount / (income_annum + 1)` |
| `emi` | Standard amortisation: `P * r * (1+r)^n / ((1+r)^n - 1)` |
| `debt_income_ratio` | `(emi * 12) / (income_annum + 1)` |
| `income_per_dep` | `income_annum / (no_of_dependents + 1)` |
| `total_assets` | Sum of all four asset columns |
| `asset_coverage` | `total_assets / (loan_amount + 1)` |

## Assumptions

### Assumed Interest Rate
The dataset does not include an interest rate column. We assume **10% per annum**
(`ASSUMED_ANNUAL_INTEREST_RATE = 0.10`) for EMI calculation. This is configurable
in `backend/config.py`.

### Loan Term Unit
`loan_term` values in the dataset range from 2 to 20. This is consistent with
**years** (not months). The unit is configurable: `LOAN_TERM_UNIT = "years"`.

### Negative Asset Values
The `residential_assets_value` column contains 28 negative values (minimum: -100,000).
These are clipped to 0 on load, as negative asset values are not meaningful. All other
asset columns contain only non-negative values.

### Decision Threshold
The default threshold is **0.5**. The threshold sweep (0.1-0.9) in `metrics.json`
allows tuning for specific precision/recall trade-offs. For this dataset, the models
are so accurate that the threshold has minimal impact above 0.3.

### Risk Tiers
- **Prime**: probability >= 0.80
- **Near Prime**: 0.60 <= probability < 0.80
- **Subprime**: 0.40 <= probability < 0.60
- **High Risk**: probability < 0.40

Subprime applications and approved-but-High-Risk cases are flagged for **manual review**.

### Calibration
Logistic Regression and Decision Tree are wrapped in `CalibratedClassifierCV(method='isotonic', cv=5)`.
Random Forest uses native probability estimation. Probabilities are clipped to [0.05, 0.95].

### Amounts
All monetary values are in **Indian Rupees (INR)**. The frontend displays amounts with
the INR symbol.

## Configuration

All tuneable parameters live in [`backend/config.py`](backend/config.py):

- `RANDOM_STATE = 42`
- `TEST_SIZE = 0.2` (stratified)
- `DECISION_THRESHOLD = 0.5`
- `PROB_CLIP = (0.05, 0.95)`
- `ASSUMED_ANNUAL_INTEREST_RATE = 0.10`
- `LOAN_TERM_UNIT = "years"`
- `SHAP_BACKGROUND_SIZE = 25`
- Model hyperparameters (LR, RF, DT) with `class_weight='balanced'`

## License

MIT
