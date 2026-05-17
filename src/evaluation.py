"""
Model evaluation metrics for time series forecasting.

Metrics:
- MAPE  (Mean Absolute Percentage Error)
- RMSE  (Root Mean Squared Error)
- MAE   (Mean Absolute Error)
- Directional Accuracy (% of correct up/down predictions)
"""

import numpy as np
import pandas as pd
from typing import Dict


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%)."""
    actual, predicted = np.array(actual), np.array(predicted)
    mask = actual != 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    actual, predicted = np.array(actual), np.array(predicted)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    actual, predicted = np.array(actual), np.array(predicted)
    return float(np.mean(np.abs(actual - predicted)))


def directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """
    Percentage of correct directional predictions.
    Compares sign of change from one step to the next.
    """
    actual, predicted = np.array(actual), np.array(predicted)
    if len(actual) < 2:
        return 0.0
    actual_dir = np.sign(np.diff(actual))
    pred_dir = np.sign(np.diff(predicted))
    return float(np.mean(actual_dir == pred_dir) * 100)


def evaluate_forecast(actual: np.ndarray, predicted: np.ndarray) -> Dict[str, float]:
    """Compute all evaluation metrics."""
    return {
        'MAPE (%)': round(mape(actual, predicted), 2),
        'RMSE': round(rmse(actual, predicted), 4),
        'MAE': round(mae(actual, predicted), 4),
        'Directional Accuracy (%)': round(directional_accuracy(actual, predicted), 2),
    }


def create_comparison_table(
    results: Dict[str, Dict[str, Dict[str, float]]],
) -> pd.DataFrame:
    """
    Create a model comparison table from nested results dict.
    
    Args:
        results: {model_name: {stock_ticker: {metric: value}}}
    
    Returns:
        DataFrame with MultiIndex (model, stock) and metric columns.
    """
    rows = []
    for model_name, stocks in results.items():
        for stock, metrics in stocks.items():
            row = {'Model': model_name, 'Stock': stock}
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)


def rank_models(comparison_df: pd.DataFrame, metric: str = 'MAPE (%)') -> pd.DataFrame:
    """Rank models by average metric across all stocks."""
    avg = comparison_df.groupby('Model')[metric].mean().sort_values()
    return avg.reset_index().rename(columns={metric: f'Avg {metric}'})
