# How to Download the Loan Approval Dataset

## Source
**Kaggle:** [Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/architsharma01/loan-approval-prediction-dataset)

## Manual Download Steps

1. **Create a Kaggle account** (free) at <https://www.kaggle.com/account/login>.
2. Navigate to the dataset page linked above.
3. Click **Download** (the zip file is ~60 KB).
4. Extract the CSV from the zip.
5. Rename the file to `loan_approval_dataset.csv` (if it isn't already).
6. Place it at:
   ```
   backend/data/loan_approval_dataset.csv
   ```

## Using the Kaggle CLI (optional)

```bash
pip install kaggle
# Ensure ~/.kaggle/kaggle.json contains your API token
kaggle datasets download -d architsharma01/loan-approval-prediction-dataset -p backend/data/ --unzip
```

> **Note:** Never commit your Kaggle API token (`kaggle.json`) to version control.

## Expected File

- **File name:** `loan_approval_dataset.csv`
- **Rows:** ~4,269 (header + data)
- **Columns:** `loan_id`, `no_of_dependents`, `education`, `self_employed`,
  `income_annum`, `loan_amount`, `loan_term`, `cibil_score`,
  `residential_assets_value`, `commercial_assets_value`,
  `luxury_assets_value`, `bank_asset_value`, `loan_status`

## Known Quirks

Column names and string values contain **leading whitespace** (e.g., `" education"`, `" Graduate"`).
This is handled automatically by `data_layer.py` — no manual preprocessing needed.
