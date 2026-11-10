# Loan Default Prediction Using Machine Learning Frameworks

ML BD Elective project, PES University.

## 1. Project Overview

This project predicts the probability that a loan defaults, using machine learning and deep learning frameworks on structured loan applicant data.

The problem is a binary classification task. The target variable is `default`:

- `1`: loan defaulted (Charged Off / Default)
- `0`: loan fully repaid (Fully Paid)

Five frameworks are implemented and compared on predictive performance, training time, implementation complexity, and transparency for tabular loan default prediction.

## 2. Objectives

- Prepare and preprocess loan application data for machine learning.
- Perform exploratory data analysis and a leakage audit.
- Build loan default models using five frameworks.
- Evaluate all models with the same classification and probability-error metrics.
- Compare predictive performance, training time, simplicity, and transparency.
- Identify the trade-offs of each framework.

## 3. Dataset

**File:** `LoansData_sample.csv.gz`
**Source:** [MLBD Projects Dataset Repository](https://github.com/tiwvis/MLBD-Projects/blob/main/data_sets/LoansData_sample.csv.gz)

The raw sample has 100,000 loans and 151 columns, all issued between August and December 2014. Only loans with a final outcome (Fully Paid, Charged Off, Default) are used.

After cleaning and preprocessing:

| Item | Value |
|---|---|
| Usable records | 86,141 |
| Default rate | about 18.8% |
| Features | 68 |
| Numerical features | 59 |
| Categorical features | 9 |
| Target | `default` |

### Data splitting

A stratified random split with seed 42 is used. A time-based split was not possible because the sample covers only five months.

| Split | Records |
|---|---:|
| Train | 60,298 |
| Validation | 12,921 |
| Test | 12,922 |
| Total | 86,141 |

Training data is used for fitting, validation data for tuning and model selection, and test data for the final evaluation only.

### Data preparation

All cleaning is done by `src/prepare_data.py`, so the dataset can be regenerated with one command. The main steps:

- Define `default` from `loan_status` and keep only loans with a final outcome.
- Remove columns that would leak the outcome or were not known at issuance (payment and recovery columns, hardship and settlement fields, identifiers, free text). The full list with reasons is in `reports/leakage_audit.md`.
- Convert dates and text fields to numeric features (`credit_hist_years`, `term_months`, `emp_length_yrs`).
- Drop columns with more than 50% missing values, judged on the training split only.
- Neural models additionally get median imputation, standardization, and integer-coded categoricals, all fitted on the training split only (`make_nn_data`).

## 4. Frameworks

| Framework | Implementation |
|---|---|
| XGBoost | Gradient boosting (tree-based baseline) |
| Keras | Neural network with categorical embeddings |
| fastai | Tabular deep learning with categorical processing, missing-value handling, and normalization |
| PyTorch TabNet | Attention-based tabular learning with built-in feature selection |
| Lightning Flash | Tabular classification interface with a TabNet backbone |

## 5. Evaluation Metrics

All frameworks are scored with `src.metrics.evaluate(y_true, p)`.

| Metric | Purpose |
|---|---|
| RMSE | Error between predicted probabilities and actual labels |
| MAE | Average absolute prediction error |
| R² | Proportion of variance explained |
| ROC-AUC | Ability to separate defaulters from non-defaulters |
| PR-AUC | Precision-recall performance, useful for imbalanced data |
| KS statistic | Separation between the score distributions of the two classes |
| Training time | Training efficiency in seconds |

## 6. Results (test set)

| Framework | RMSE | MAE | R² | ROC-AUC | PR-AUC | KS |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.3676 | 0.2710 | 0.1131 | 0.7337 | 0.3890 | 0.3476 |
| Keras | 0.3686 | 0.2719 | 0.1087 | 0.7330 | 0.3898 | 0.3460 |
| fastai | 0.3769 | 0.2674 | 0.0680 | 0.7058 | 0.3612 | 0.3023 |
| TabNet | 0.3693 | 0.2828 | 0.1051 | 0.7296 | 0.3829 | 0.3437 |
| Lightning Flash | 0.3739 | 0.2785 | 0.0828 | 0.7038 | 0.3477 | 0.3009 |

### Training time

| Framework | Seconds |
|---|---:|
| XGBoost | 16.60 |
| Keras | 24.68 |
| fastai | 28.17 |
| TabNet | 129.31 |
| Lightning Flash | 221.00 |

### Key findings

- XGBoost has the highest test ROC-AUC (0.7337), the lowest RMSE (0.3676), and the shortest training time (16.6 s).
- Keras has the highest PR-AUC (0.3898) and is very close to XGBoost on ROC-AUC.
- TabNet is competitive (ROC-AUC 0.7296) but takes about 8 times longer to train than XGBoost.
- fastai has the lowest MAE (0.2674) but lower ROC-AUC.
- Lightning Flash is the slowest (221 s) and has the lowest ROC-AUC (0.7038).
- R² is low for every model because the target is a binary label scored with probabilities, so it should be read alongside ROC-AUC and PR-AUC.

Overall, XGBoost gives the best balance of accuracy and efficiency here, with Keras and TabNet close behind.

## 7. Repository Structure

```text
loan-default-mlbd/
├── data/
│   ├── raw/                  (gitignored)
│   └── processed/            (gitignored; loans_clean.parquet is generated here)
├── notebooks/
│   ├── 02_eda.ipynb
│   ├── 03_xgboost.ipynb
│   ├── 06_tabnet.ipynb
│   ├── 07_flash.ipynb
│   ├── 08_comparison.ipynb
│   └── (Keras and fastai notebooks)
├── reports/
│   ├── figures/
│   ├── leakage_audit.md
│   ├── results_xgboost.json
│   ├── results_keras.json
│   ├── results_fastai.json
│   ├── results_tabnet.json
│   ├── results_flash.json
│   ├── framework_comparison.csv
│   └── comparison_notes.md
├── src/
│   ├── config.py
│   ├── data.py
│   ├── metrics.py
│   └── prepare_data.py
├── requirements.txt
└── README.md
```

The structure highlights the main components. Other files are omitted for readability.

## 8. Installation and Execution

The data files are not stored in the repository. They are generated by a script.

**1. Clone the repository**

```bash
git clone https://github.com/kratipatel/loan-default-mlbd.git
cd loan-default-mlbd
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Generate the dataset**

```bash
python -m src.prepare_data
```

This downloads the sample, cleans it, creates the stratified split, and writes `data/processed/loans_clean.parquet`. Expected output: shape `(86141, 71)` with split sizes 60,298 / 12,921 / 12,922.

**5. Load the data in your own notebook**

```python
from src.data import load_splits, for_trees, make_nn_data
from src.metrics import evaluate

train, val, test = load_splits("data/processed/loans_clean.parquet")

# Tree models (XGBoost): NaNs allowed, categoricals as pandas category
trees, feats = for_trees(train, val, test)

# Neural models (Keras, fastai, TabNet, Flash): 59 numeric + 9 categorical columns
data, num_cols, cat_cols, cardinalities = make_nn_data(train, val, test)
```

`make_nn_data` returns `(X_num, X_cat, y)` for each split. TabNet and Lightning Flash should use these numeric and categorical arrays.

**6. Run the notebooks**

Run the notebooks in `notebooks/` in order. The Flash implementation may need a compatible Python and PyTorch environment. The comparison notebook reads the saved `results_*.json` files and builds the tables and plots.

### Shared rules for all frameworks

- Never use `issue_dt`, `split`, or `default` as input features.
- Tune on the validation split. Report test metrics once, at the end.
- Score every model with `src.metrics.evaluate` and save `reports/results_<framework>.json`.

## 9. Reports

The `reports/` directory contains:

- Per-framework evaluation results in JSON.
- A consolidated comparison table in CSV.
- Written comparison notes in Markdown.
- EDA and SHAP figures in `reports/figures/`.
- The leakage audit (`leakage_audit.md`).

## 10. Team Contributions

| Member | SRN | Responsibility |
|---|---|---|
| Krati Patel | PES2UG24AM077 | Data cleaning, EDA, leakage audit, shared data code, XGBoost |
| Chaya S | PES2UG24AM045 | Keras, fastai |
| Anwesha Manik | PES2UG24AM023 | TabNet, Lightning Flash, metrics and comparison table |

## 11. Limitations

- The sample covers a single five-month window (Aug-Dec 2014), so a time-based split and any claim about behaviour across economic periods are not possible.
- Most labeled loans are 36-month loans, since many 60-month loans from 2014 were still "Current" and were excluded.
- Grade, sub-grade, and interest rate are assigned by Lending Club's own risk model, so they are legitimate inputs but carry much of the predictive signal.
- Results come from a single seed (42) and a single split.

## 12. Conclusion

This project compares five frameworks for loan default prediction using one dataset, one split, and one evaluation method. XGBoost achieved the highest ROC-AUC and the shortest training time, Keras achieved the highest PR-AUC, TabNet was competitive but slower, and fastai and Lightning Flash had lower ROC-AUC in this experiment. The best framework depends on the balance needed between accuracy, training cost, and implementation effort.

## 13. References

1. [MLBD Projects – Loan Dataset](https://github.com/tiwvis/MLBD-Projects/blob/main/data_sets/LoansData_sample.csv.gz)
2. [XGBoost Documentation](https://xgboost.readthedocs.io/)
3. [Keras Documentation](https://keras.io/)
4. [fastai Documentation](https://docs.fast.ai/)
5. [PyTorch TabNet](https://github.com/dreamquark-ai/tabnet)
6. [Lightning Flash Documentation](https://lightning-flash.readthedocs.io/)