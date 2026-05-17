"""
Visualization module for the capstone project.

All plots use a consistent dark theme and are saved to outputs/figures/.
"""

import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List

matplotlib.use('Agg')  # non-interactive backend for scripts

# ──────────────────────────────────────────────────────────────
# Global Style
# ──────────────────────────────────────────────────────────────

COLORS = [
    '#00D4AA', '#FF6B6B', '#4ECDC4', '#FFE66D', '#95E1D3',
    '#A8E6CF', '#FF8B94', '#F38181', '#AA96DA', '#FCBAD3',
    '#6C5B7B', '#F67280',
]

def setup_style():
    """Apply consistent dark theme to all matplotlib plots."""
    plt.style.use('dark_background')
    plt.rcParams.update({
        'figure.facecolor': '#0d1117',
        'axes.facecolor': '#161b22',
        'axes.edgecolor': '#30363d',
        'axes.labelcolor': '#c9d1d9',
        'text.color': '#c9d1d9',
        'xtick.color': '#8b949e',
        'ytick.color': '#8b949e',
        'grid.color': '#21262d',
        'grid.alpha': 0.6,
        'font.family': 'sans-serif',
        'font.size': 11,
        'figure.dpi': 120,
        'savefig.bbox': 'tight',
        'savefig.facecolor': '#0d1117',
    })

setup_style()

FIG_DIR = Path('outputs/figures')
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _save(fig, name):
    fig.savefig(FIG_DIR / f"{name}.png", dpi=150)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────
# Stock Price Plots
# ──────────────────────────────────────────────────────────────

def plot_stock_prices(data: Dict[str, pd.DataFrame], save_name='stock_prices'):
    """Plot closing prices for all stocks on one figure."""
    fig, ax = plt.subplots(figsize=(14, 7))
    for i, (ticker, df) in enumerate(data.items()):
        ax.plot(df.index, df['Close'], label=ticker.replace('.NS', ''),
                color=COLORS[i % len(COLORS)], linewidth=1.2)
    ax.set_title('Stock Universe — Daily Close Prices (2021–2025)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (₹)')
    ax.legend(fontsize=8, ncol=3, loc='upper left')
    ax.grid(True, alpha=0.3)
    _save(fig, save_name)


def plot_stock_individual(df: pd.DataFrame, ticker: str, save_name=None):
    """Plot individual stock with volume subplot."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), height_ratios=[3, 1],
                                    sharex=True, gridspec_kw={'hspace': 0.05})
    name = ticker.replace('.NS', '')
    ax1.plot(df.index, df['Close'], color=COLORS[0], linewidth=1.5)
    ax1.fill_between(df.index, df['Close'], alpha=0.1, color=COLORS[0])
    ax1.set_title(f'{name} — Price & Volume', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Close Price (₹)')
    ax1.grid(True, alpha=0.3)

    ax2.bar(df.index, df['Volume'], color=COLORS[2], alpha=0.6, width=1)
    ax2.set_ylabel('Volume')
    ax2.set_xlabel('Date')
    _save(fig, save_name or f'stock_{name}')


# ──────────────────────────────────────────────────────────────
# Forecast Plots
# ──────────────────────────────────────────────────────────────

def plot_forecast_vs_actual(
    actual: pd.Series,
    predicted: np.ndarray,
    ticker: str,
    model_name: str,
    forecast: Optional[np.ndarray] = None,
    forecast_lower: Optional[np.ndarray] = None,
    forecast_upper: Optional[np.ndarray] = None,
    save_name: str = None,
):
    """Plot actual vs predicted prices, with optional future forecast."""
    fig, ax = plt.subplots(figsize=(12, 6))
    name = ticker.replace('.NS', '')

    ax.plot(actual.index, actual.values, label='Actual', color=COLORS[0], linewidth=1.5)
    ax.plot(actual.index[:len(predicted)], predicted, label=f'{model_name} Predicted',
            color=COLORS[1], linewidth=1.5, linestyle='--')

    if forecast is not None:
        last_date = actual.index[-1]
        fc_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=len(forecast))
        ax.plot(fc_dates, forecast, label='Forecast', color=COLORS[3],
                linewidth=2, marker='o', markersize=6)
        if forecast_lower is not None and forecast_upper is not None:
            ax.fill_between(fc_dates, forecast_lower, forecast_upper,
                           alpha=0.2, color=COLORS[3], label='95% CI')

    ax.set_title(f'{name} — {model_name} Forecast', fontsize=13, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (₹)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, save_name or f'forecast_{name}_{model_name}')


# ──────────────────────────────────────────────────────────────
# Portfolio Allocation
# ──────────────────────────────────────────────────────────────

def plot_portfolio_allocation(allocation_df: pd.DataFrame, save_name='portfolio_allocation'):
    """Pie chart and bar chart of portfolio allocation."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    labels = [s.replace('.NS', '') for s in allocation_df['Stock']]
    sizes = allocation_df['Weight (%)']
    colors = COLORS[:len(labels)]

    # Pie chart
    wedges, texts, autotexts = ax1.pie(
        sizes, labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, pctdistance=0.85,
        textprops={'fontsize': 9, 'color': '#c9d1d9'},
    )
    centre_circle = plt.Circle((0, 0), 0.55, fc='#161b22')
    ax1.add_artist(centre_circle)
    ax1.set_title('Portfolio Allocation', fontsize=13, fontweight='bold')

    # Bar chart
    ax2.barh(labels[::-1], allocation_df['Amount (₹)'].values[::-1],
             color=colors[::-1], edgecolor='none', height=0.6)
    ax2.set_xlabel('Amount (₹)')
    ax2.set_title('Capital Deployed per Stock', fontsize=13, fontweight='bold')
    ax2.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    _save(fig, save_name)


# ──────────────────────────────────────────────────────────────
# Correlation Heatmap
# ──────────────────────────────────────────────────────────────

def plot_correlation_heatmap(returns_df: pd.DataFrame, save_name='correlation_heatmap'):
    """Heatmap of pairwise return correlations."""
    corr = returns_df.corr()
    labels = [c.replace('.NS', '') for c in corr.columns]

    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(
        corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
        xticklabels=labels, yticklabels=labels, ax=ax,
        linewidths=0.5, linecolor='#30363d',
        cbar_kws={'shrink': 0.8},
        vmin=-1, vmax=1,
    )
    ax.set_title('Stock Return Correlations', fontsize=13, fontweight='bold')
    plt.tight_layout()
    _save(fig, save_name)


# ──────────────────────────────────────────────────────────────
# Volatility
# ──────────────────────────────────────────────────────────────

def plot_volatility(
    vol_data: Dict[str, pd.Series],
    save_name='volatility',
):
    """Plot rolling volatility for all stocks."""
    fig, ax = plt.subplots(figsize=(14, 6))
    for i, (ticker, vol) in enumerate(vol_data.items()):
        ax.plot(vol.index, vol.values, label=ticker.replace('.NS', ''),
                color=COLORS[i % len(COLORS)], linewidth=1)
    ax.set_title('21-Day Rolling Volatility (Std Dev of Log Returns)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel('Volatility')
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    _save(fig, save_name)


# ──────────────────────────────────────────────────────────────
# STL Decomposition
# ──────────────────────────────────────────────────────────────

def plot_decomposition(decomposition, ticker: str, save_name=None):
    """Plot STL decomposition (trend, seasonal, residual)."""
    name = ticker.replace('.NS', '')
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(decomposition.observed, color=COLORS[0])
    axes[0].set_ylabel('Observed')
    axes[0].set_title(f'{name} — STL Decomposition', fontsize=13, fontweight='bold')

    axes[1].plot(decomposition.trend, color=COLORS[2])
    axes[1].set_ylabel('Trend')

    axes[2].plot(decomposition.seasonal, color=COLORS[3])
    axes[2].set_ylabel('Seasonal')

    axes[3].plot(decomposition.resid, color=COLORS[4], alpha=0.7)
    axes[3].set_ylabel('Residual')
    axes[3].set_xlabel('Date')

    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, save_name or f'decomposition_{name}')


# ──────────────────────────────────────────────────────────────
# Model Comparison
# ──────────────────────────────────────────────────────────────

def plot_model_comparison(comparison_df: pd.DataFrame, save_name='model_comparison'):
    """Bar chart comparing models by MAPE and RMSE."""
    avg_metrics = comparison_df.groupby('Model')[['MAPE (%)', 'RMSE']].mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    models = avg_metrics.index.tolist()
    x = np.arange(len(models))

    ax1.bar(x, avg_metrics['MAPE (%)'], color=COLORS[:len(models)], edgecolor='none')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=30, ha='right')
    ax1.set_ylabel('Avg MAPE (%)')
    ax1.set_title('Model Comparison — MAPE', fontsize=13, fontweight='bold')
    ax1.grid(True, axis='y', alpha=0.3)

    ax2.bar(x, avg_metrics['RMSE'], color=COLORS[:len(models)], edgecolor='none')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=30, ha='right')
    ax2.set_ylabel('Avg RMSE')
    ax2.set_title('Model Comparison — RMSE', fontsize=13, fontweight='bold')
    ax2.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    _save(fig, save_name)


# ──────────────────────────────────────────────────────────────
# Predicted vs Actual (Task 8)
# ──────────────────────────────────────────────────────────────

def plot_predicted_vs_actual_live(
    predicted: Dict[str, List[float]],
    actual: Dict[str, List[float]],
    days: List[str] = ['Day 1', 'Day 2'],
    save_name='predicted_vs_actual_live',
):
    """Bar chart comparing predicted vs actual prices for live trading days."""
    tickers = list(predicted.keys())
    n = len(tickers)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for i, ticker in enumerate(tickers):
        name = ticker.replace('.NS', '')
        x = np.arange(len(days))
        w = 0.35
        axes[i].bar(x - w/2, predicted[ticker], w, label='Predicted', color=COLORS[0])
        axes[i].bar(x + w/2, actual[ticker], w, label='Actual', color=COLORS[1])
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(days)
        axes[i].set_title(name, fontsize=11, fontweight='bold')
        axes[i].legend(fontsize=8)
        axes[i].grid(True, axis='y', alpha=0.3)

    plt.suptitle('Predicted vs Actual — Live Trading Window', fontsize=14, fontweight='bold')
    plt.tight_layout()
    _save(fig, save_name)
