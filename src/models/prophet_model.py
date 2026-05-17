"""
Facebook Prophet forecasting model.

Handles the ds/y column format, Indian market holidays,
and provides forecast with uncertainty intervals.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, Any, Tuple

warnings.filterwarnings('ignore')


# Major Indian market holidays (NSE closures) — representative set
INDIAN_HOLIDAYS = pd.DataFrame({
    'holiday': 'indian_market',
    'ds': pd.to_datetime([
        # 2024
        '2024-01-26', '2024-03-25', '2024-03-29', '2024-04-11',
        '2024-04-14', '2024-04-17', '2024-04-21', '2024-05-01',
        '2024-05-23', '2024-06-17', '2024-07-17', '2024-08-15',
        '2024-09-16', '2024-10-02', '2024-10-12', '2024-10-31',
        '2024-11-01', '2024-11-15', '2024-12-25',
        # 2025
        '2025-01-26', '2025-02-26', '2025-03-14', '2025-03-31',
        '2025-04-06', '2025-04-10', '2025-04-14', '2025-04-18',
        '2025-05-01', '2025-05-12', '2025-06-07', '2025-07-06',
        '2025-08-15', '2025-08-16', '2025-08-27', '2025-10-02',
        '2025-10-21', '2025-10-22', '2025-11-05', '2025-11-26',
        '2025-12-25',
    ]),
    'lower_window': 0,
    'upper_window': 0,
})


class ProphetForecaster:
    """Facebook Prophet model wrapper."""

    def __init__(self, yearly_seasonality: bool = True, weekly_seasonality: bool = True):
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.model = None
        self._train_df = None

    def fit(self, train_series: pd.Series) -> 'ProphetForecaster':
        """
        Fit Prophet on the training series.
        Input series must have a DatetimeIndex.
        """
        from prophet import Prophet

        df = pd.DataFrame({
            'ds': train_series.index,
            'y': train_series.values,
        })
        self._train_df = df

        self.model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=False,
            holidays=INDIAN_HOLIDAYS,
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            interval_width=0.95,
        )
        self.model.fit(df)
        return self

    def predict(self, steps: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forecast `steps` business days ahead.
        Returns (forecast, lower, upper).
        """
        future = self.model.make_future_dataframe(periods=steps, freq='B')
        fc = self.model.predict(future)
        forecast_rows = fc.tail(steps)
        return (
            forecast_rows['yhat'].values,
            forecast_rows['yhat_lower'].values,
            forecast_rows['yhat_upper'].values,
        )

    def predict_on_dates(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Predict on specific dates (for test set evaluation)."""
        future = pd.DataFrame({'ds': dates})
        return self.model.predict(future)

    def predict_in_sample(self, test_series: pd.Series) -> np.ndarray:
        """Predict on test dates for evaluation."""
        fc = self.predict_on_dates(test_series.index)
        return fc['yhat'].values

    def get_components(self) -> pd.DataFrame:
        """Return trend and seasonality components."""
        future = self.model.make_future_dataframe(periods=0)
        return self.model.predict(future)
