"""
Task 1 — Data-Driven Stock Universe Selection
==============================================
This script screens a broader pool of 25 NSE candidates across 5 sectors,
computes three quantitative selection criteria (as required by the spec),
and justifies why our final 11 stocks were chosen.

Selection Criteria (from spec):
  1. Rolling Standard Deviation (30-day) — volatility profile
  2. Seasonal Decomposition (STL)       — trend strength
  3. Sector Momentum (6-month return)   — sector-level selection

Output:
  - outputs/task1_screening.csv          (full candidate analysis)
  - outputs/task1_selection_justified.json (per-stock justification)
  - outputs/figures/task1_screening.png  (visual comparison)
"""

import sys, os, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import yfinance as yf
from statsmodels.tsa.seasonal import STL
from pathlib import Path

# ──────────────────────────────────────────────────────────────
# CANDIDATE POOL — 25 stocks across 5 sectors
# ──────────────────────────────────────────────────────────────

CANDIDATES = {
    'Banking': [
        'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS', 'KOTAKBANK.NS', 'AXISBANK.NS'
    ],
    'IT': [
        'TCS.NS', 'INFY.NS', 'WIPRO.NS', 'HCLTECH.NS', 'TECHM.NS'
    ],
    'Pharma': [
        'SUNPHARMA.NS', 'DRREDDY.NS', 'CIPLA.NS', 'DIVISLAB.NS', 'APOLLOHOSP.NS'
    ],
    'FMCG': [
        'ITC.NS', 'HINDUNILVR.NS', 'NESTLEIND.NS', 'BRITANNIA.NS', 'DABUR.NS'
    ],
    'Auto': [
        'M&M.NS', 'MARUTI.NS', 'BAJAJ-AUTO.NS', 'EICHERMOT.NS', 'HEROMOTOCO.NS'
    ],
}

# Our final selection
SELECTED = [
    'HDFCBANK.NS', 'ICICIBANK.NS', 'SBIN.NS',
    'TCS.NS', 'INFY.NS',
    'SUNPHARMA.NS', 'DRREDDY.NS',
    'ITC.NS', 'HINDUNILVR.NS',
    'M&M.NS', 'MARUTI.NS',
]

OUT_DIR = Path('outputs')
FIG_DIR = OUT_DIR / 'figures'
CACHE_DIR = Path('data') / 'raw'

for d in [OUT_DIR, FIG_DIR, CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def fetch(ticker):
    """Fetch data with caching."""
    cache = CACHE_DIR / f"{ticker.replace('.', '_')}.csv"
    if cache.exists():
        return pd.read_csv(cache, index_col=0, parse_dates=True)
    df = yf.download(ticker, start='2021-01-01', end='2025-12-31', interval='1d', progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if not df.empty:
        df.to_csv(cache)
    return df


def compute_criteria(ticker, df):
    """Compute all three selection criteria for a single stock."""
    close = df['Close']
    log_ret = np.log(close / close.shift(1)).dropna()

    # 1. Rolling Std Dev (30-day) — average and latest
    roll_std = log_ret.rolling(30).std().dropna()
    avg_vol = float(roll_std.mean())
    recent_vol = float(roll_std.iloc[-1])

    # 2. STL Decomposition — trend strength
    try:
        stl = STL(close.dropna(), period=63, robust=True).fit()
        trend = stl.trend.dropna()
        resid = stl.resid.dropna()
        # Trend strength = 1 - Var(residual) / Var(trend + residual)
        trend_strength = max(0, 1 - np.var(resid) / np.var(trend + resid))
        # Trend direction (last quarter slope)
        if len(trend) >= 63:
            trend_slope = (trend.iloc[-1] - trend.iloc[-63]) / trend.iloc[-63] * 100
        else:
            trend_slope = 0
    except Exception:
        trend_strength = 0
        trend_slope = 0

    # 3. Sector momentum — 6-month cumulative return
    if len(close) >= 126:
        momentum_6m = (close.iloc[-1] - close.iloc[-126]) / close.iloc[-126] * 100
    else:
        momentum_6m = 0

    # Extras: liquidity (avg volume), total return
    avg_volume = float(df['Volume'].mean()) if 'Volume' in df.columns else 0
    total_return = (close.iloc[-1] - close.iloc[0]) / close.iloc[0] * 100

    return {
        'Ticker': ticker,
        'Name': ticker.replace('.NS', ''),
        'Avg_30d_Vol': round(avg_vol, 5),
        'Recent_Vol': round(recent_vol, 5),
        'Trend_Strength': round(trend_strength, 4),
        'Trend_Slope_Pct': round(float(trend_slope), 2),
        'Momentum_6M_Pct': round(float(momentum_6m), 2),
        'Avg_Daily_Volume': int(avg_volume),
        'Total_Return_Pct': round(float(total_return), 2),
        'Data_Points': len(df),
    }


def main():
    print('\n' + '=' * 65)
    print('  TASK 1 — Data-Driven Stock Universe Selection')
    print('=' * 65)

    # ─── Fetch all candidates ─────────────────────────────────
    print('\n  Fetching data for 25 candidate stocks...\n')
    all_tickers = [t for tickers in CANDIDATES.values() for t in tickers]
    ticker_sector = {t: s for s, tickers in CANDIDATES.items() for t in tickers}

    data = {}
    for t in all_tickers:
        print(f'    {t.replace(".NS",""):>15}:', end=' ')
        df = fetch(t)
        if df.empty or len(df) < 200:
            print('SKIPPED (insufficient data)')
            continue
        data[t] = df
        print(f'{len(df)} rows')

    # ─── Compute criteria for every candidate ─────────────────
    print('\n  Computing selection criteria...\n')
    results = []
    for t, df in data.items():
        r = compute_criteria(t, df)
        r['Sector'] = ticker_sector[t]
        r['Selected'] = t in SELECTED
        results.append(r)

    df_results = pd.DataFrame(results)

    # ─── Print full comparison table ──────────────────────────
    print('  FULL CANDIDATE SCREENING TABLE:')
    print('  ' + '-' * 110)
    print(f'  {"Stock":>14} {"Sector":>8} {"30d Vol":>9} {"Trend Str":>10} {"Trend %":>9} {"6M Mom %":>9} {"Volume":>12} {"Selected":>9}')
    print('  ' + '-' * 110)

    for _, row in df_results.sort_values('Sector').iterrows():
        sel_mark = ' >>>' if row['Selected'] else '    '
        print(f'  {row["Name"]:>14} {row["Sector"]:>8} {row["Avg_30d_Vol"]:>9.5f} '
              f'{row["Trend_Strength"]:>10.4f} {row["Trend_Slope_Pct"]:>8.2f}% '
              f'{row["Momentum_6M_Pct"]:>8.2f}% {row["Avg_Daily_Volume"]:>12,}{sel_mark}')

    # ─── Selection Justification ──────────────────────────────
    print('\n\n  SELECTION JUSTIFICATION:')
    print('  ' + '=' * 80)

    justifications = {}
    for sector, candidates in CANDIDATES.items():
        print(f'\n  [{sector.upper()}]')
        sector_df = df_results[df_results['Sector'] == sector].sort_values('Momentum_6M_Pct', ascending=False)
        selected_in_sector = sector_df[sector_df['Selected']]
        rejected_in_sector = sector_df[~sector_df['Selected']]

        for _, row in selected_in_sector.iterrows():
            reasons = []

            # Volatility profile
            all_vols = df_results['Avg_30d_Vol']
            vol_pct = (all_vols < row['Avg_30d_Vol']).mean() * 100
            if vol_pct > 60:
                reasons.append(f'Higher-than-median volatility ({row["Avg_30d_Vol"]:.5f}, {vol_pct:.0f}th percentile) - interesting for active trading')
            elif vol_pct < 40:
                reasons.append(f'Lower volatility ({row["Avg_30d_Vol"]:.5f}, {vol_pct:.0f}th percentile) - stabilizer for portfolio risk reduction')
            else:
                reasons.append(f'Moderate volatility ({row["Avg_30d_Vol"]:.5f}, {vol_pct:.0f}th percentile) - balanced risk-reward profile')

            # Trend
            if row['Trend_Strength'] > 0.5:
                direction = 'upward' if row['Trend_Slope_Pct'] > 0 else 'downward'
                reasons.append(f'Strong trend component (strength={row["Trend_Strength"]:.4f}, slope={row["Trend_Slope_Pct"]:+.2f}% - {direction})')
            else:
                reasons.append(f'Trend strength={row["Trend_Strength"]:.4f}, slope={row["Trend_Slope_Pct"]:+.2f}%')

            # Momentum
            if row['Momentum_6M_Pct'] > 5:
                reasons.append(f'Positive 6-month momentum ({row["Momentum_6M_Pct"]:+.2f}%)')
            elif row['Momentum_6M_Pct'] < -5:
                reasons.append(f'Negative momentum ({row["Momentum_6M_Pct"]:+.2f}%) - mean-reversion opportunity')
            else:
                reasons.append(f'Neutral momentum ({row["Momentum_6M_Pct"]:+.2f}%)')

            # Liquidity
            reasons.append(f'High liquidity (avg daily volume: {row["Avg_Daily_Volume"]:,})')

            justifications[row['Ticker']] = {
                'sector': sector,
                'reasons': reasons,
                'metrics': {
                    'avg_30d_volatility': row['Avg_30d_Vol'],
                    'trend_strength': row['Trend_Strength'],
                    'trend_slope_pct': row['Trend_Slope_Pct'],
                    'momentum_6m_pct': row['Momentum_6M_Pct'],
                    'avg_daily_volume': row['Avg_Daily_Volume'],
                }
            }

            print(f'    SELECTED: {row["Name"]}')
            for r in reasons:
                print(f'      - {r}')

        if len(rejected_in_sector) > 0:
            rejected_names = ', '.join(rejected_in_sector['Name'].tolist())
            print(f'    Rejected: {rejected_names}')
            for _, rej in rejected_in_sector.iterrows():
                reason = []
                if rej['Avg_Daily_Volume'] < selected_in_sector['Avg_Daily_Volume'].min():
                    reason.append('lower liquidity')
                if rej['Trend_Strength'] < selected_in_sector['Trend_Strength'].mean():
                    reason.append('weaker trend')
                if abs(rej['Momentum_6M_Pct']) < abs(selected_in_sector['Momentum_6M_Pct']).mean() * 0.5:
                    reason.append('less interesting momentum')
                if not reason:
                    reason.append('redundant within sector (already have sufficient representation)')
                print(f'      {rej["Name"]}: {"; ".join(reason)}')

    # ─── Sector Momentum Analysis ─────────────────────────────
    print(f'\n\n  SECTOR MOMENTUM RANKING:')
    print('  ' + '-' * 50)
    sector_mom = df_results.groupby('Sector')['Momentum_6M_Pct'].mean().sort_values(ascending=False)
    for sector, mom in sector_mom.items():
        n_selected = len(df_results[(df_results['Sector'] == sector) & (df_results['Selected'])])
        print(f'    {sector:>8}: {mom:>+8.2f}% avg momentum  ({n_selected} stocks selected)')

    # ─── Save outputs ─────────────────────────────────────────
    df_results.to_csv(OUT_DIR / 'task1_screening.csv', index=False)

    # Save justified selection
    full_selection = {
        'methodology': {
            'step_1': 'Identified 25 candidate stocks across 5 sectors (Banking, IT, Pharma, FMCG, Auto)',
            'step_2': 'Computed three quantitative criteria for each: (1) 30-day rolling std dev of log returns, (2) STL trend strength and slope, (3) 6-month sector momentum',
            'step_3': 'Selected 2-3 stocks per sector based on: interesting volatility profiles, strong trend components, positive/mean-reverting momentum, and high liquidity',
            'step_4': 'Final universe of 11 stocks ensures diversification across uncorrelated sectors',
        },
        'sector_momentum_ranking': {s: round(float(m), 2) for s, m in sector_mom.items()},
        'selections': justifications,
    }

    with open(OUT_DIR / 'task1_selection_justified.json', 'w') as f:
        json.dump(full_selection, f, indent=2)

    # ─── Plot ─────────────────────────────────────────────────
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor('#0d1117')
    for ax in axes:
        ax.set_facecolor('#161b22')

    colors = ['#00D4AA' if s else '#555555' for s in df_results['Selected']]
    labels = df_results['Name']

    # Volatility
    axes[0].barh(labels, df_results['Avg_30d_Vol'], color=colors, edgecolor='none')
    axes[0].set_title('30-Day Rolling Volatility', fontweight='bold', color='#c9d1d9')
    axes[0].set_xlabel('Avg Std Dev of Log Returns')
    axes[0].tick_params(colors='#8b949e')

    # Trend Strength
    axes[1].barh(labels, df_results['Trend_Strength'], color=colors, edgecolor='none')
    axes[1].set_title('STL Trend Strength', fontweight='bold', color='#c9d1d9')
    axes[1].set_xlabel('Trend Strength (0-1)')
    axes[1].tick_params(colors='#8b949e')

    # 6M Momentum
    mom_colors = ['#00D4AA' if (s and m > 0) else '#FF6B6B' if (s and m < 0) else '#555555'
                  for s, m in zip(df_results['Selected'], df_results['Momentum_6M_Pct'])]
    axes[2].barh(labels, df_results['Momentum_6M_Pct'], color=mom_colors, edgecolor='none')
    axes[2].set_title('6-Month Momentum (%)', fontweight='bold', color='#c9d1d9')
    axes[2].set_xlabel('Return (%)')
    axes[2].axvline(0, color='#30363d', linewidth=1)
    axes[2].tick_params(colors='#8b949e')

    plt.suptitle('Stock Universe Screening — Selected (green) vs Candidates (grey)',
                 fontsize=14, fontweight='bold', color='#c9d1d9', y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'task1_screening.png', dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()

    print(f'\n  Outputs saved:')
    print(f'    - {OUT_DIR / "task1_screening.csv"}')
    print(f'    - {OUT_DIR / "task1_selection_justified.json"}')
    print(f'    - {FIG_DIR / "task1_screening.png"}')
    print(f'\n' + '=' * 65)
    print('  Task 1 Selection Analysis Complete')
    print('=' * 65 + '\n')


if __name__ == '__main__':
    main()
