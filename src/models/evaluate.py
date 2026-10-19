"""Evaluation metrics used across notebooks 06-09."""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))


def bias(y_true, y_pred) -> float:
    """Mean(actual - predicted), as defined in notebook 08.

    Negative = over-forecasting on average. (The `error` column built later in the same
    notebook uses the opposite sign: predicted - actual.)
    """
    return float((np.asarray(y_true) - np.asarray(y_pred)).mean())


def evaluate(y_true, y_pred) -> dict:
    return {"RMSE": rmse(y_true, y_pred), "MAE": mae(y_true, y_pred)}


def naive_lag7_predictions(X_valid):
    """Naive baseline of notebook 09: yesterday-week sales (lag_7), clipped at 0."""
    return np.clip(X_valid["lag_7"].values, a_min=0, a_max=None)
