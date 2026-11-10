"""
Model Evaluation Module for CART Regression.

Computes MAE, MSE, RMSE, and R-squared strictly on actual vs predicted arrays.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_regression_model(
    y_true: Any,
    y_pred: Any,
    y_train_true: Any = None,
    y_train_pred: Any = None,
) -> Dict[str, float]:
    """Calculate standard regression metrics on untouched test predictions."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    metrics: Dict[str, float] = {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
    }

    if y_train_true is not None and y_train_pred is not None:
        metrics["train_r2"] = float(r2_score(y_train_true, y_train_pred))
        metrics["train_rmse"] = float(np.sqrt(mean_squared_error(y_train_true, y_train_pred)))

    return metrics


def get_metric_explanations() -> Dict[str, str]:
    """Plain-language explanations for academic presentation."""
    return {
        "MAE": "Mean Absolute Error measures the average absolute rupee gap between predicted and actual car prices.",
        "MSE": "Mean Squared Error averages squared prediction errors, penalizing large valuation misses heavily.",
        "RMSE": "Root Mean Squared Error returns the MSE back into Indian Rupees for intuitive error magnitude.",
        "R2": (
            "R-squared represents the fraction of price variance explained by the CART tree. "
            "Note: A high R2 summarizes aggregate performance and does not guarantee exact predictions for every individual car."
        ),
    }
