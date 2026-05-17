"""
Portfolio construction and capital allocation strategies.

Implements four strategies as required by the capstone:
  A — Forecast-Guided Allocation
  B — Volatility-Aware Sizing
  C — Correlation-Based Diversification
  D — Sector Momentum Rotation

And a combiner to blend strategies into a final allocation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

TOTAL_CAPITAL = 10_00_000  # ₹10,00,000


# ──────────────────────────────────────────────────────────────
# Strategy A — Forecast-Guided Allocation
# ──────────────────────────────────────────────────────────────

def strategy_forecast_guided(
    forecasts: Dict[str, np.ndarray],
    current_prices: Dict[str, float],
) -> Dict[str, float]:
    """
    Allocate more capital to stocks with higher predicted returns.
    
    Args:
        forecasts: {ticker: array of forecasted prices (2 days)}
        current_prices: {ticker: last known close price}
    
    Returns:
        {ticker: weight} (sums to 1.0)
    """
    predicted_returns = {}
    for ticker in forecasts:
        fc = forecasts[ticker]
        curr = current_prices[ticker]
        # Expected return = (predicted end price - current) / current
        pred_return = (fc[-1] - curr) / curr
        predicted_returns[ticker] = pred_return

    # Only allocate to positive-return stocks; clip negatives to small weight
    weights = {}
    for ticker, ret in predicted_returns.items():
        weights[ticker] = max(ret, 0.001)  # floor at 0.1%

    # Normalize
    total = sum(weights.values())
    return {t: w / total for t, w in weights.items()}


# ──────────────────────────────────────────────────────────────
# Strategy B — Volatility-Aware Sizing
# ──────────────────────────────────────────────────────────────

def strategy_volatility_aware(
    volatilities: Dict[str, float],
) -> Dict[str, float]:
    """
    Inverse-volatility weighting: wᵢ = (1/σ̂ᵢ) / Σ(1/σ̂ⱼ).
    Less volatile stocks get higher allocation.
    
    Args:
        volatilities: {ticker: estimated volatility (std dev of returns)}
    
    Returns:
        {ticker: weight} (sums to 1.0)
    """
    inv_vol = {t: 1.0 / max(v, 1e-8) for t, v in volatilities.items()}
    total = sum(inv_vol.values())
    return {t: w / total for t, w in inv_vol.items()}


# ──────────────────────────────────────────────────────────────
# Strategy C — Correlation-Based Diversification
# ──────────────────────────────────────────────────────────────

def strategy_correlation_diversified(
    returns_df: pd.DataFrame,
    base_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Penalise highly-correlated stocks to reduce portfolio risk.
    
    If base_weights is provided, adjust them by average pairwise correlation.
    Otherwise, start with equal weights.
    
    Args:
        returns_df: DataFrame of log returns (columns = tickers)
        base_weights: optional starting weights to adjust
    
    Returns:
        {ticker: weight} (sums to 1.0)
    """
    corr_matrix = returns_df.corr()
    tickers = list(returns_df.columns)

    if base_weights is None:
        base_weights = {t: 1.0 / len(tickers) for t in tickers}

    # Average absolute correlation for each stock with others
    avg_corr = {}
    for t in tickers:
        others = [abs(corr_matrix.loc[t, o]) for o in tickers if o != t]
        avg_corr[t] = np.mean(others) if others else 0

    # Penalty: reduce weight proportional to average correlation
    adjusted = {}
    for t in tickers:
        penalty = 1.0 - avg_corr[t]  # lower correlation = higher multiplier
        adjusted[t] = base_weights.get(t, 0) * max(penalty, 0.1)

    total = sum(adjusted.values())
    return {t: w / total for t, w in adjusted.items()}


# ──────────────────────────────────────────────────────────────
# Strategy D — Sector Momentum Rotation
# ──────────────────────────────────────────────────────────────

def strategy_sector_momentum(
    returns_df: pd.DataFrame,
    ticker_sectors: Dict[str, str],
    lookback: int = 126,  # ~6 months of trading days
) -> Dict[str, float]:
    """
    Overweight high-momentum sectors, equal-weight within each sector.
    
    Args:
        returns_df: DataFrame of log returns (columns = tickers)
        ticker_sectors: {ticker: sector_name}
        lookback: number of days for momentum calculation
    
    Returns:
        {ticker: weight} (sums to 1.0)
    """
    recent = returns_df.tail(lookback)

    # Compute sector-level cumulative return
    sector_returns = {}
    sector_tickers = {}
    for ticker in returns_df.columns:
        sector = ticker_sectors.get(ticker, 'Unknown')
        if sector not in sector_returns:
            sector_returns[sector] = []
            sector_tickers[sector] = []
        sector_returns[sector].append(recent[ticker].sum())
        sector_tickers[sector].append(ticker)

    sector_momentum = {s: np.mean(rets) for s, rets in sector_returns.items()}

    # Weight sectors by momentum (shift to positive)
    min_mom = min(sector_momentum.values())
    shifted = {s: m - min_mom + 0.01 for s, m in sector_momentum.items()}
    total_mom = sum(shifted.values())
    sector_weights = {s: m / total_mom for s, m in shifted.items()}

    # Distribute sector weight equally among its stocks
    weights = {}
    for sector, sw in sector_weights.items():
        tickers_in_sector = sector_tickers[sector]
        per_stock = sw / len(tickers_in_sector)
        for t in tickers_in_sector:
            weights[t] = per_stock

    return weights


# ──────────────────────────────────────────────────────────────
# Combine Strategies
# ──────────────────────────────────────────────────────────────

def combine_strategies(
    weight_dicts: List[Dict[str, float]],
    strategy_weights: Optional[List[float]] = None,
) -> Dict[str, float]:
    """
    Blend multiple strategy weight dicts.
    
    Args:
        weight_dicts: List of {ticker: weight} dicts.
        strategy_weights: How much to weight each strategy (default: equal).
    
    Returns:
        {ticker: weight} (sums to 1.0)
    """
    n = len(weight_dicts)
    if strategy_weights is None:
        strategy_weights = [1.0 / n] * n

    all_tickers = set()
    for d in weight_dicts:
        all_tickers.update(d.keys())

    combined = {}
    for t in all_tickers:
        combined[t] = sum(
            sw * wd.get(t, 0) for sw, wd in zip(strategy_weights, weight_dicts)
        )

    total = sum(combined.values())
    return {t: w / total for t, w in combined.items()}


# ──────────────────────────────────────────────────────────────
# Capital Allocation
# ──────────────────────────────────────────────────────────────

def allocate_capital(
    weights: Dict[str, float],
    total_capital: float = TOTAL_CAPITAL,
) -> pd.DataFrame:
    """
    Convert normalised weights to rupee allocations.
    
    Returns:
        DataFrame with columns: Stock, Weight (%), Amount (₹)
    """
    rows = []
    for ticker, weight in sorted(weights.items(), key=lambda x: -x[1]):
        rows.append({
            'Stock': ticker,
            'Weight (%)': round(weight * 100, 2),
            'Amount (₹)': round(weight * total_capital, 2),
        })
    return pd.DataFrame(rows)
