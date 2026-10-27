import numpy as np
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             roc_auc_score, average_precision_score, roc_curve)


def evaluate(y_true, p):
    """Evaluate predicted default probabilities p against binary labels y_true."""
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float).ravel()
    fpr, tpr, _ = roc_curve(y_true, p)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, p))),
        "mae": float(mean_absolute_error(y_true, p)),
        "r2": float(r2_score(y_true, p)),
        "roc_auc": float(roc_auc_score(y_true, p)),
        "pr_auc": float(average_precision_score(y_true, p)),
        "ks": float(np.max(tpr - fpr)),
    }
