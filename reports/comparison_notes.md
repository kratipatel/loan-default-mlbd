
# Machine Learning Framework Comparison

## 1. Overview

Five machine learning frameworks were evaluated for predicting loan default:
XGBoost, Keras, fastai, PyTorch TabNet, and Lightning Flash.

All frameworks were evaluated using the same train, validation, and test
splits. Model selection and tuning were performed using validation data.
Final performance was assessed using the test set.

## 2. Test Performance Comparison

| Framework | RMSE | MAE | R2 | ROC-AUC | PR-AUC | KS |
|---|---:|---:|---:|---:|---:|---:|
| XGBoost | 0.3676 | 0.2710 | 0.1131 | 0.7337 | 0.3890 | 0.3476 |
| Keras | 0.3686 | 0.2719 | 0.1087 | 0.7330 | 0.3898 | 0.3460 |
| fastai | 0.3769 | 0.2674 | 0.0680 | 0.7058 | 0.3612 | 0.3023 |
| TabNet | 0.3693 | 0.2828 | 0.1051 | 0.7296 | 0.3829 | 0.3437 |
| Lightning Flash | 0.3739 | 0.2785 | 0.0828 | 0.7038 | 0.3477 | 0.3009 |

## 3. Training Time Comparison

| Framework | Training Time (seconds) |
|---|---:|
| XGBoost | 16.60 |
| Keras | 24.68 |
| fastai | 28.17 |
| TabNet | 129.31 |
| Lightning Flash | 221.00 |

XGBoost had the shortest training time, while Lightning Flash required
the longest training time among the five implementations.

## 4. Performance Analysis

- XGBoost achieved the highest test ROC-AUC of 0.7337 and the lowest
  test RMSE of 0.3676.
- Keras achieved the highest test PR-AUC of 0.3898.
- TabNet achieved a test ROC-AUC of 0.7296, close to XGBoost and Keras.
- fastai achieved the lowest test MAE of 0.2674, but had lower ROC-AUC
  and R2 than the leading models.
- Lightning Flash had the lowest test ROC-AUC and PR-AUC among the
  evaluated frameworks.

## 5. Simplicity and Transparency

- XGBoost provides a relatively direct workflow for tabular
  classification and supports feature importance analysis.
- Keras provides a high-level API for defining and training neural
  networks.
- fastai provides higher-level abstractions that simplify model training.
- TabNet uses attention-based feature selection, but requires additional
  configuration for categorical features and model parameters.
- Lightning Flash provides a high-level interface, although its
  integration with the tabular dataset and prediction workflow required
  additional environment and API configuration.

## 6. Implementation Experience

- XGBoost, Keras, and fastai completed their implementations successfully.
- TabNet required experimentation with model dimensions and parameters.
- Lightning Flash required additional environment setup and compatibility
  adjustments. The installed Flash version was 0.8.2.
- Flash was also challenging because of its maintenance status and
  compatibility with the surrounding PyTorch ecosystem.

## 7. Overall Conclusion

XGBoost provided the strongest overall balance of test ROC-AUC, RMSE,
and training efficiency in this experiment.

Keras performed very closely to XGBoost and achieved the highest PR-AUC.
TabNet also demonstrated competitive predictive performance, although
its training time was considerably higher.

fastai and Lightning Flash had lower ROC-AUC scores in this experiment.
Lightning Flash also required the longest training time and additional
implementation effort.

The results indicate that framework choice involves trade-offs between
predictive performance, training efficiency, implementation complexity,
and ease of use. No single framework is necessarily best for every
criterion.
