"""
GARCH(1,1) volatility model.

Models time-varying conditional volatility of log returns.
Used for:
- Volatility estimation (Task 4)
- Volatility-aware portfolio sizing (Strategy B in Task 5)
"""

import pandas as pd
import numpy as np
import warnings
from typing import Dict, Any, Tuple

warnings.filterwarnings('ignore')


class GARCHForecaster:
    """GARCH(1,1) conditional volatility model using the `arch` library."""

    def __init__(self, p: int = 1, q: int = 1, dist: str = 'normal'):
        """
        Args:
            p: GARCH order (number of lagged variances).
            q: ARCH order (number of lagged squared residuals).
            dist: Error distribution — 'normal', 't', or 'skewt'.
        """
        self.p = p
        self.q = q
        self.dist = dist
        self.model = None
        self.result = None
        self._returns = None

    def fit(self, returns: pd.Series) -> 'GARCHForecaster':
        """
        Fit GARCH(p,q) on log returns.
        Input should be log returns (not prices).
        """
        from arch import arch_model

        # Scale returns to percentage for numerical stability
        self._returns = returns.dropna() * 100

        self.model = arch_model(
            self._returns,
            vol='GARCH',
            p=self.p,
            q=self.q,
            dist=self.dist,
            mean='Constant',
        )
        self.result = self.model.fit(disp='off', show_warning=False)
        return self

    def predict_volatility(self, steps: int = 2) -> np.ndarray:
        """
        Forecast conditional volatility for `steps` ahead.
        Returns annualized volatility (multiply daily σ by √252).
        """
        fc = self.result.forecast(horizon=steps, reindex=False)
        # variance forecast → std dev, then un-scale (÷100) and annualise
        daily_vol = np.sqrt(fc.variance.values[-1]) / 100
        return daily_vol  # daily volatility for each step

    def get_conditional_volatility(self) -> pd.Series:
        """Return in-sample conditional volatility series."""
        return self.result.conditional_volatility / 100  # un-scale

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return model diagnostics and parameters."""
        params = self.result.params
        return {
            'omega': float(params.get('omega', 0)),
            'alpha': float(params.get('alpha[1]', 0)),
            'beta': float(params.get('beta[1]', 0)),
            'persistence': float(params.get('alpha[1]', 0) + params.get('beta[1]', 0)),
            'log_likelihood': float(self.result.loglikelihood),
            'aic': float(self.result.aic),
            'bic': float(self.result.bic),
        }

    def summary(self) -> str:
        return str(self.result.summary())
