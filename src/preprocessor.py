"""
Data preprocessing module for time series analysis.

Handles:
- Missing value imputation (forward/backward fill)
- Stationarity testing (ADF test) and differencing
- Log returns computation
- Train/test split
- Scaling (MinMaxScaler) and sequence creation for LSTM
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, Any, Optional


# ──────────────────────────────────────────────────────────────
# Missing Values
# ──────────────────────────────────────────────────────────────

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values: forward fill → backward fill → drop residual."""
    before = df.isna().sum().sum()
    df = df.ffill().bfill().dropna()
    after = df.isna().sum().sum()
    filled = before - after
    if filled > 0:
        print(f"    Filled {filled} missing values")
    return df


# ──────────────────────────────────────────────────────────────
# Stationarity
# ──────────────────────────────────────────────────────────────

def test_stationarity(series: pd.Series, significance: float = 0.05) -> Dict[str, Any]:
    """
    Augmented Dickey-Fuller test for stationarity.
    Returns dict with test statistic, p-value, critical values, and verdict.
    """
    clean = series.dropna()
    if len(clean) < 20:
        return {'is_stationary': False, 'p_value': 1.0, 'error': 'Too few observations'}

    result = adfuller(clean, autolag='AIC')
    return {
        'test_statistic': result[0],
        'p_value': result[1],
        'lags_used': result[2],
        'n_observations': result[3],
        'critical_values': result[4],
        'is_stationary': result[1] < significance,
    }


def make_stationary(series: pd.Series, max_diffs: int = 2) -> Tuple[pd.Series, int]:
    """Apply differencing until the series is stationary (or max_diffs reached)."""
    d = 0
    current = series.copy()
    while d < max_diffs:
        result = test_stationarity(current)
        if result['is_stationary']:
            break
        current = current.diff().dropna()
        d += 1
    return current, d


# ──────────────────────────────────────────────────────────────
# Returns & Volatility Helpers
# ──────────────────────────────────────────────────────────────

def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns: ln(P_t / P_{t-1})."""
    return np.log(prices / prices.shift(1)).dropna()


def compute_rolling_volatility(prices: pd.Series, window: int = 21) -> pd.Series:
    """Rolling standard deviation of log returns (annualised would multiply by √252)."""
    log_ret = compute_log_returns(prices)
    return log_ret.rolling(window=window).std().dropna()


def compute_stl_decomposition(prices: pd.Series, period: int = 63):
    """
    STL decomposition (trend + seasonal + residual).
    Default period=63 ≈ quarterly in trading days.
    """
    stl = STL(prices.dropna(), period=period, robust=True)
    return stl.fit()


# ──────────────────────────────────────────────────────────────
# Train/Test Split
# ──────────────────────────────────────────────────────────────

def create_train_test_split(
    df: pd.DataFrame,
    split_date: str = '2025-07-01',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split data into train (before split_date) and test (on or after)."""
    train = df[df.index < split_date].copy()
    test = df[df.index >= split_date].copy()
    return train, test


# ──────────────────────────────────────────────────────────────
# Scaling & Sequences (for LSTM)
# ──────────────────────────────────────────────────────────────

class DataScaler:
    """MinMaxScaler wrapper for time series data."""

    def __init__(self, feature_range: Tuple[float, float] = (0, 1)):
        self.scaler = MinMaxScaler(feature_range=feature_range)

    def fit_transform(self, data):
        arr = self._to_2d(data)
        return self.scaler.fit_transform(arr)

    def transform(self, data):
        arr = self._to_2d(data)
        return self.scaler.transform(arr)

    def inverse_transform(self, data):
        arr = self._to_2d(data)
        return self.scaler.inverse_transform(arr)

    @staticmethod
    def _to_2d(data):
        if isinstance(data, pd.Series):
            return data.values.reshape(-1, 1)
        if isinstance(data, np.ndarray) and data.ndim == 1:
            return data.reshape(-1, 1)
        return data


def create_sequences(data: np.ndarray, seq_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding-window sequences for LSTM.
    Returns (X, y) where X[i] = data[i:i+seq_length], y[i] = data[i+seq_length].
    """
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


# ──────────────────────────────────────────────────────────────
# Full Preprocessing Pipeline
# ──────────────────────────────────────────────────────────────

def preprocess_stock(
    df: pd.DataFrame,
    split_date: str = '2025-07-01',
) -> Dict[str, Any]:
    """
    Full preprocessing pipeline for a single stock.
    Returns dict with cleaned data, train/test splits, stationarity results, log returns.
    """
    df = handle_missing_values(df)
    close = df['Close']

    # Stationarity
    adf = test_stationarity(close)
    _, d_order = make_stationary(close)

    # Log returns
    log_ret = compute_log_returns(close)

    # Train/test split
    train, test = create_train_test_split(df, split_date)

    return {
        'df': df,
        'close': close,
        'train': train,
        'test': test,
        'adf_result': adf,
        'd_order': d_order,
        'log_returns': log_ret,
    }
