"""
Exponential Smoothing (Holt-Winters) forecasting model.

Supports:
- Simple Exponential Smoothing (SES)
- Holt's linear trend
- Holt-Winters (additive and multiplicative seasonality)
"""

import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from typing import Tuple

warnings.filterwarnings('ignore')


class ExpSmoothingForecaster:
    """Holt-Winters Exponential Smoothing model."""

    def __init__(
        self,
        trend: str = 'add',
        seasonal: str = 'add',
        seasonal_periods: int = 63,  # ~quarterly in trading days
    ):
        """
        Args:
            trend: 'add', 'mul', or None.
            seasonal: 'add', 'mul', or None.
            seasonal_periods: Number of periods in a season.
        """
        self.trend = trend
        self.seasonal = seasonal
        self.seasonal_periods = seasonal_periods
        self.model = None
        self.result = None

    def fit(self, train_series: pd.Series) -> 'ExpSmoothingForecaster':
        """Fit Holt-Winters model on training data."""
        try:
            self.model = ExponentialSmoothing(
                train_series,
                trend=self.trend,
                seasonal=self.seasonal,
                seasonal_periods=self.seasonal_periods,
                initialization_method='estimated',
            )
            self.result = self.model.fit(optimized=True)
        except Exception:
            # Fallback: no seasonality if seasonal fit fails
            self.model = ExponentialSmoothing(
                train_series,
                trend=self.trend,
                seasonal=None,
                initialization_method='estimated',
            )
            self.result = self.model.fit(optimized=True)
            self.seasonal = None

        return self

    def predict(self, steps: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forecast `steps` ahead.
        Returns (forecast, lower, upper).
        Confidence intervals approximated from residual std.
        """
        fc = self.result.forecast(steps)
        resid_std = np.std(self.result.resid.dropna())
        margin = 1.96 * resid_std  # 95% CI
        return fc.values, fc.values - margin, fc.values + margin

    def predict_in_sample(self, test_series: pd.Series) -> np.ndarray:
        """Forecast over the test period length."""
        fc = self.result.forecast(len(test_series))
        return fc.values

    def get_params(self):
        """Return fitted model parameters."""
        return {
            'trend': self.trend,
            'seasonal': self.seasonal,
            'seasonal_periods': self.seasonal_periods,
            'aic': self.result.aic if hasattr(self.result, 'aic') else None,
            'bic': self.result.bic if hasattr(self.result, 'bic') else None,
            'smoothing_level': self.result.params.get('smoothing_level', None),
            'smoothing_trend': self.result.params.get('smoothing_trend', None),
            'smoothing_seasonal': self.result.params.get('smoothing_seasonal', None),
        }
