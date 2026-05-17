"""
ARIMA / SARIMA forecasting model.

Uses pmdarima's auto_arima for automatic order selection via AIC/BIC,
with residual diagnostics and confidence intervals.
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, Any, Optional, Tuple

warnings.filterwarnings('ignore')


class ARIMAForecaster:
    """ARIMA/SARIMA model wrapper using pmdarima."""

    def __init__(self, seasonal: bool = True, m: int = 5):
        """
        Args:
            seasonal: Whether to fit SARIMA (True) or ARIMA (False).
            m: Seasonal period (5 = weekly for trading days).
        """
        self.seasonal = seasonal
        self.m = m
        self.model = None
        self.order = None
        self.seasonal_order = None
        self.fitted_values = None

    def fit(self, train_series: pd.Series) -> 'ARIMAForecaster':
        """
        Fit auto-ARIMA on the training series.
        Automatically selects (p,d,q) and optionally (P,D,Q,m).
        """
        import pmdarima as pm

        self.model = pm.auto_arima(
            train_series.values,
            start_p=0, max_p=5,
            start_q=0, max_q=5,
            d=None,           # auto-select differencing
            seasonal=self.seasonal,
            m=self.m if self.seasonal else 1,
            start_P=0, max_P=2,
            start_Q=0, max_Q=2,
            D=None if self.seasonal else 0,
            trace=False,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
            information_criterion='aic',
        )

        self.order = self.model.order
        self.seasonal_order = self.model.seasonal_order
        self.fitted_values = self.model.fittedvalues()
        return self

    def predict(self, steps: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Forecast `steps` days ahead.
        Returns (forecast, lower_ci, upper_ci).
        """
        fc, ci = self.model.predict(n_periods=steps, return_conf_int=True, alpha=0.05)
        return fc, ci[:, 0], ci[:, 1]

    def predict_in_sample(self, test_series: pd.Series) -> np.ndarray:
        """One-step-ahead rolling predictions on the test set."""
        preds = []
        for val in test_series.values:
            fc, _ = self.model.predict(n_periods=1, return_conf_int=True)
            preds.append(fc[0])
            self.model.update([val])
        return np.array(preds)

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return model diagnostics."""
        from statsmodels.stats.diagnostic import acorr_ljungbox
        resid = self.model.resid()
        lb = acorr_ljungbox(resid, lags=[10], return_df=True)
        return {
            'order': self.order,
            'seasonal_order': self.seasonal_order,
            'aic': self.model.aic(),
            'bic': self.model.bic(),
            'ljung_box_p': lb['lb_pvalue'].values[0],
            'residual_mean': float(np.mean(resid)),
            'residual_std': float(np.std(resid)),
        }

    def summary(self) -> str:
        return str(self.model.summary())
