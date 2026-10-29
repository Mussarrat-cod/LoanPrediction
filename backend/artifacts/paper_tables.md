# Paper Tables (Auto-generated)

## Table 1: Dataset Schema

| Column | Type | Range |
|--------|------|-------|
| no_of_dependents | int64 | 0 - 5 |
| education | object | Graduate, Not Graduate |
| self_employed | object | No, Yes |
| income_annum | int64 | 200000 - 9900000 |
| loan_amount | int64 | 300000 - 39500000 |
| loan_term | int64 | 2 - 20 |
| cibil_score | int64 | 300 - 900 |
| residential_assets_value | int64 | 0 - 29100000 |
| commercial_assets_value | int64 | 0 - 19400000 |
| luxury_assets_value | int64 | 300000 - 39200000 |
| bank_asset_value | int64 | 0 - 14700000 |
| loan_status | int64 | 0 - 1 |

## Table 2: Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| Rejected (0) | 1613 | 37.8% |
| Approved (1) | 2656 | 62.2% |
| **Total** | **4269** | 100% |

## Table 3: Hyperparameters

| Parameter | Value |
|-----------|-------|
| Random State | 42 |
| Test Size | 0.2 |
| Interest Rate | 0.1 |
| Loan Term Unit | years |
| SHAP Background | 25 |
| LR | {'max_iter': 1000, 'class_weight': 'balanced', 'random_state': 42} |
| RF | {'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 10, 'class_weight': 'balanced', 'random_state': 42} |
| DT | {'max_depth': 8, 'class_weight': 'balanced', 'random_state': 42} |

## Table 4: Model Comparison

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV Accuracy | CV F1 |
|-------|----------|-----------|--------|-----|---------|-------------|-------|
| Logistic Regression | 94.38% | 95.48% | 95.48% | 95.48% | 98.00% | 93.00 +/- 1.73 | 94.25 +/- 1.45 |
| Random Forest | 99.77% | 99.62% | 100.00% | 99.81% | 100.00% | 99.77 +/- 0.15 | 99.81 +/- 0.12 |
| Decision Tree **(Best)** | 100.00% | 100.00% | 100.00% | 100.00% | 100.00% | 99.97 +/- 0.06 | 99.98 +/- 0.05 |

## Table 5: Top-5 Feature Importance per Model

### Logistic Regression

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | cibil_score | 0.3919 |
| 2 | debt_income_ratio | 0.0457 |
| 3 | income_annum | 0.0111 |
| 4 | asset_coverage | 0.0090 |
| 5 | loan_term | 0.0082 |

### Random Forest

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | cibil_score | 0.8224 |
| 2 | debt_income_ratio | 0.0457 |
| 3 | loan_to_income | 0.0320 |
| 4 | loan_term | 0.0242 |
| 5 | asset_coverage | 0.0164 |

### Decision Tree

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | cibil_score | 0.4295 |
| 2 | debt_income_ratio | 0.0500 |
| 3 | loan_to_income | 0.0210 |
| 4 | asset_coverage | 0.0082 |
| 5 | no_of_dependents | 0.0000 |

## Table 6: Ablation Study (RF without cibil_score)

| Metric | With cibil_score | Without cibil_score |
|--------|-----------------|---------------------|
| accuracy | 99.77% | 54.68% |
| precision | 99.62% | 68.65% |
| recall | 100.00% | 49.91% |
| f1 | 99.81% | 57.80% |
| roc_auc | 100.00% | 60.78% |

## Measured Latency

Average inference latency: **1.05 ms/sample**
