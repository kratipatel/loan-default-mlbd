# Loan Default Prediction Using Machine Learning Frameworks

## 1. Project Overview

This project focuses on predicting the probability of loan default using machine learning and deep learning frameworks.

The problem is formulated as a binary classification task using structured loan applicant data. The target variable is `default`:

- `1` – Loan defaulted (Charged Off / Default)
- `0` – Loan fully repaid (Fully Paid)

The project evaluates and compares five different machine learning frameworks to understand their predictive performance, training efficiency, implementation complexity, and usability for tabular loan default prediction.

## 2. Objectives

- Prepare and preprocess loan application data for machine learning.
- Perform exploratory data analysis and feature engineering.
- Develop loan default prediction models using five frameworks.
- Evaluate models using common classification and regression-based metrics.
- Compare predictive performance, training time, simplicity, and transparency.
- Identify the trade-offs associated with each framework.

## 3. Dataset Description

**Dataset:** LoansData_sample.csv.gz

**Source:** [MLBD Projects Dataset Repository](https://github.com/tiwvis/MLBD-Projects/blob/main/data_sets/LoansData_sample.csv.gz)

The dataset contains loan application information, including borrower characteristics, loan attributes, and repayment information.

After data cleaning and preprocessing:

- Total usable records: 86,141
- Features: 68
- Numerical features: 59
- Categorical features: 9
- Target variable: `default`

### Data Splitting

The dataset was divided using stratified sampling with a random seed of 42.

| Dataset | Records |
|---|---:|
| Training | 60,298 |
| Validation | 12,921 |
| Testing | 12,922 |
| Total | 86,141 |

The training dataset was used for model fitting, validation data for model selection and tuning, and test data for final evaluation.

## 4. Machine Learning Frameworks

Five frameworks were implemented and evaluated:

| Framework | Implementation |
|---|---|
| XGBoost | Gradient Boosting |
| Keras | Neural Network with categorical embeddings |
| fastai | Tabular Deep Learning |
| PyTorch TabNet | Attention-based Tabular Learning |
| Lightning Flash | Tabular Classification using Flash |

### Framework Implementations

**XGBoost**

Uses gradient boosting for structured tabular data and serves as a tree-based baseline.

**Keras**

Uses a neural network with categorical embeddings and numerical features.

**fastai**

Uses the fastai tabular API with categorical processing, missing-value handling, and normalization.

**PyTorch TabNet**

Uses an attention-based architecture for tabular learning, allowing the model to learn feature selection during training.

**Lightning Flash**

Uses the Lightning Flash tabular classification interface with a TabNet backbone.

## 5. Evaluation Metrics

All five frameworks were evaluated using the following metrics:

| Metric | Purpose |
|---|---|
| RMSE | Measures the error between predicted probabilities and actual labels |
| MAE | Measures the average absolute prediction error |
| R² | Measures the proportion of variance explained |
| ROC-AUC | Measures the ability to distinguish between default and non-default cases |
| PR-AUC | Measures precision-recall performance, particularly useful for imbalanced data |
| KS Statistic | Measures the separation between the default and non-default score distributions |
| Training Time | Measures model training efficiency in seconds |

## 6. Model Comparison Results

The following table presents the final test-set results.

| Framework | RMSE | MAE | R² | ROC-AUC | PR-AUC | KS |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.3676 | 0.2710 | 0.1131 | 0.7337 | 0.3890 | 0.3476 |
| Keras | 0.3686 | 0.2719 | 0.1087 | 0.7330 | 0.3898 | 0.3460 |
| fastai | 0.3769 | 0.2674 | 0.0680 | 0.7058 | 0.3612 | 0.3023 |
| TabNet | 0.3693 | 0.2828 | 0.1051 | 0.7296 | 0.3829 | 0.3437 |
| Lightning Flash | 0.3739 | 0.2785 | 0.0828 | 0.7038 | 0.3477 | 0.3009 |

### Training Time Comparison

| Framework | Training Time (seconds) |
|---|---:|
| XGBoost | 16.60 |
| Keras | 24.68 |
| fastai | 28.17 |
| TabNet | 129.31 |
| Lightning Flash | 221.00 |

### Key Findings

- XGBoost achieved the highest test ROC-AUC of 0.7337 and the lowest RMSE of 0.3676.
- Keras achieved the highest test PR-AUC of 0.3898.
- TabNet achieved competitive performance with a test ROC-AUC of 0.7296.
- fastai achieved the lowest test MAE of 0.2674.
- XGBoost had the shortest training time of 16.60 seconds.
- Lightning Flash required the longest training time of 221 seconds.

Overall, XGBoost provided the strongest balance of predictive performance and training efficiency in this experiment. Keras and TabNet also demonstrated competitive performance.

## 7. Repository Structure

```text
loan-default-mlbd/
│
├── data/
│   └── processed/
│
├── notebooks/
│   ├── 06_tabnet.ipynb
│   ├── 07_flash.ipynb
│   └── 08_comparison.ipynb
│
├── reports/
│   ├── results_xgboost.json
│   ├── results_keras.json
│   ├── results_fastai.json
│   ├── results_tabnet.json
│   ├── results_flash.json
│   ├── framework_comparison.csv
│   └── comparison_notes.md
│
├── src/
│   ├── config.py
│   ├── data.py
│   ├── metrics.py
│   └── prepare_data.py
│
├── requirements.txt
└── README.md
```

*Note: The structure above highlights the main project components. Other existing files and directories are omitted for readability.*

## 8. Installation and Execution

### Step 1: Clone the repository

```bash
git clone https://github.com/kratipatel/loan-default-mlbd.git
cd loan-default-mlbd
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Prepare the dataset

Place the original dataset in the expected input location and run:

```bash
python -m src.prepare_data
```

This generates the processed dataset used by the model implementations.

### Step 5: Run the notebooks

The framework implementations and comparison are available in the `notebooks/` directory.

The notebooks can be opened using Jupyter Notebook or Google Colab. The Flash implementation may require a compatible Python and PyTorch environment.

The comparison notebook loads the saved result JSON files and generates performance tables and visualizations.

## 9. Results and Reports

The `reports/` directory contains:

- Individual framework evaluation results in JSON format.
- Consolidated framework comparison in CSV format.
- Written comparison findings in Markdown format.

These files support reproducibility and make it easier to inspect and compare model performance.

## 10. Team Contributions

| Member | SRN | Responsibility |
|---|---|---|
| Krati Patel | PES2UG24AM077 | Data cleaning, EDA, XGBoost |
| Chaya S | PES2UG24AM045 | Keras, fastai |
| Anwesha Manik | PES2UG24AM023 | TabNet, Lightning Flash, metrics and comparison table |

## 11. Conclusion

This project compares five machine learning frameworks for loan default prediction using a common dataset and evaluation methodology.

The experimental results highlight the trade-offs between predictive performance, training time, and implementation complexity.

XGBoost achieved the highest test ROC-AUC and the shortest training time, while Keras achieved the highest PR-AUC. TabNet provided competitive results, whereas fastai and Lightning Flash showed comparatively lower ROC-AUC in this experiment.

The study demonstrates that framework selection depends on the desired balance between model performance, computational efficiency, and implementation requirements.

## 12. References

1. [MLBD Projects – Loan Dataset](https://github.com/tiwvis/MLBD-Projects/blob/main/data_sets/LoansData_sample.csv.gz)
2. [XGBoost Documentation](https://xgboost.readthedocs.io/)
3. [Keras Documentation](https://keras.io/)
4. [fastai Documentation](https://docs.fast.ai/)
5. [PyTorch TabNet](https://github.com/dreamquark-ai/tabnet)
6. [Lightning Flash Documentation](https://lightning-flash.readthedocs.io/)
