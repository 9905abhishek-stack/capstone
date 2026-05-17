"""
Data acquisition module — fetches NSE stock data via Yahoo Finance (yfinance).

Usage:
    from src.data_fetcher import fetch_all_stocks, SECTOR_MAP
    data = fetch_all_stocks()  # returns Dict[str, pd.DataFrame]
"""

import yfinance as yf
import pandas as pd
import os
from pathlib import Path
from typing import Dict, List, Optional

# ──────────────────────────────────────────────────────────────
# Default Stock Universe — 11 large-cap NSE stocks, 5 sectors
# ──────────────────────────────────────────────────────────────
SECTOR_MAP = {
    'Banking':  ['HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS'],
    'IT':       ['TCS.NS', 'INFY.NS'],
    'Pharma':   ['SUNPHARMA.NS', 'DRREDDY.NS'],
    'FMCG':     ['ITC.NS', 'HINDUNILVR.NS'],
    'Auto':     ['M&M.NS', 'MARUTI.NS'],
}

DEFAULT_START = '2021-01-01'
DEFAULT_END   = '2025-12-31'


def get_all_tickers(sector_map: Optional[Dict] = None) -> List[str]:
    """Flatten sector map to list of tickers."""
    sm = sector_map or SECTOR_MAP
    return [t for tickers in sm.values() for t in tickers]


def get_ticker_sector(sector_map: Optional[Dict] = None) -> Dict[str, str]:
    """Return ticker -> sector mapping."""
    sm = sector_map or SECTOR_MAP
    return {t: sector for sector, tickers in sm.items() for t in tickers}


def fetch_stock_data(
    ticker: str,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV data for a single NSE stock from Yahoo Finance.
    Optionally caches to CSV for reproducibility.
    """
    if cache_dir:
        cache_file = Path(cache_dir) / f"{ticker.replace('.', '_')}.csv"
        if cache_file.exists():
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if not df.empty:
                return df

    df = yf.download(ticker, start=start, end=end, interval='1d', progress=False)

    # yfinance sometimes returns MultiIndex columns — flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if cache_dir and not df.empty:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        cache_file = Path(cache_dir) / f"{ticker.replace('.', '_')}.csv"
        df.to_csv(cache_file)

    return df


def fetch_all_stocks(
    tickers: Optional[List[str]] = None,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    cache_dir: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV data for all stocks in the universe.
    Returns dict mapping ticker -> DataFrame.
    """
    if tickers is None:
        tickers = get_all_tickers()

    data: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        print(f"  ↳ Fetching {ticker}...", end=" ")
        try:
            df = fetch_stock_data(ticker, start, end, cache_dir)
            if df.empty:
                print("⚠ EMPTY")
            else:
                data[ticker] = df
                print(f"✓ {len(df)} rows")
        except Exception as e:
            print(f"✗ ERROR: {e}")

    print(f"\n  Fetched {len(data)}/{len(tickers)} stocks successfully.\n")
    return data


if __name__ == "__main__":
    data = fetch_all_stocks(cache_dir="data/raw")
    for ticker, df in data.items():
        print(f"{ticker}: {df.shape}, {df.index[0].date()} → {df.index[-1].date()}")
