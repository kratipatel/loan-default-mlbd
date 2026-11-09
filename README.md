\# Loan Default Probability Prediction (ML BD Elective)



\[2-3 sentences in your own words: what problem this solves and why default prediction matters.]



\## Task



Predict the probability that a Lending Club loan is charged off (1) or fully paid (0), and compare five frameworks on simplicity, transparency, and efficacy:

XGBoost, Keras, PyTorch + fastai, PyTorch + TabNet, PyTorch + Lightning Flash.

Metrics: RMSE, R2, MAE on predicted probabilities, plus ROC-AUC, PR-AUC, KS.



\## Team



| Member | SRN | Responsibility |

|---|---|---|

| Krati Patel | \[SRN] | Data cleaning, EDA, XGBoost |

| \[Name] | \[SRN] | Keras, fastai |

| \[Name] | \[SRN] | TabNet, Lightning Flash, metrics and comparison table |



\## Dataset



\- Source: sample provided with the assignment, \[LoansData\_sample.csv.gz](https://github.com/tiwvis/MLBD-Projects/blob/main/data\_sets/LoansData\_sample.csv.gz)

\- 100,000 loans x 151 columns, issue dates August-December 2014

\- Kept loans with a final outcome (Fully Paid, Charged Off, Default): 86,141 loans, 18.8% defaults

\- Split: stratified random 70/15/15, seed 42 (train 60,298 / val 12,921 / test 12,922). A time-based split was not possible because the sample covers only five months.

\- After cleaning: 68 model features (59 numeric, 9 categorical)



\## Data preparation



\[Summarize in your own words: how the target was defined, which columns were removed as leakage and why (see reports/leakage\_audit.md), how dates and numeric columns were handled, why columns with more than 50% missing values were dropped.]



\## Repository structure



```

notebooks/   01\_data\_cleaning, 02\_eda, 03\_xgboost, then one notebook per framework

src/         config.py, data.py (loaders and preprocessing), metrics.py (evaluation)

reports/     figures, leakage\_audit.md, results\_\*.json, final report

data/        gitignored; put loans\_clean.parquet in data/processed/

```



\## How to reproduce



1\. `git clone https://github.com/kratipatel/loan-default-mlbd.git` and `cd loan-default-mlbd`

2\. `pip install -r requirements.txt` (Python version used: \[fill in])

3\. Run `notebooks/01\_data\_cleaning.ipynb`. Set its output path to `data/processed/loans\_clean.parquet`.

4\. Run the notebooks in order.



\## Shared code rules



\- Trees: `for\_trees` in `src/data.py`. Neural models: `make\_nn\_data`. Both fit all statistics on the training split only.

\- Never use `issue\_dt`, `split`, or `default` as input features.

\- Tune on the validation split. Report final metrics on the test split once.

\- Evaluate with `src.metrics.evaluate(y\_true, p)`.



\## Results (test set)



| Framework | RMSE | R2 | MAE | ROC-AUC | PR-AUC | KS | Train time (s) | Lines of code |

|---|---|---|---|---|---|---|---|---|

| XGBoost | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

| Keras | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

| fastai | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

| TabNet | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

| Lightning Flash | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |



\## Comparison



\[Simplicity, transparency, efficacy: fill in after the results are in.]



\## Limitations



\[Your own notes: single five-month sample, class imbalance, dependence on Lending Club's own grade and interest rate, etc.]

