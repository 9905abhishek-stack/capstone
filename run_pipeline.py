"""
╔══════════════════════════════════════════════════════════════════════╗
║   Time Series Analysis 2026 — Capstone Pipeline                     ║
║   IIT Guwahati × StockGro                                           ║
║                                                                      ║
║   This script runs the complete pipeline:                            ║
║     Task 1: Stock Universe Selection                                 ║
║     Task 2: Data Preprocessing                                       ║
║     Task 3: Time Series Forecasting (ARIMA, Prophet, LSTM, ExpSmooth)║
║     Task 4: Volatility & Trend Analysis (GARCH, STL)                ║
║     Task 5: Portfolio Construction & Allocation                      ║
║     Task 6: Model Comparison                                         ║
║                                                                      ║
║   All results are saved to outputs/ for the dashboard and report.    ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import sys
import os
import json
import pickle
import warnings
import traceback
import time
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_fetcher import (
    fetch_all_stocks, get_all_tickers, get_ticker_sector, SECTOR_MAP
)
from src.preprocessor import (
    handle_missing_values, test_stationarity, make_stationary,
    compute_log_returns, compute_rolling_volatility,
    compute_stl_decomposition, create_train_test_split, preprocess_stock
)
from src.evaluation import evaluate_forecast, create_comparison_table, rank_models
from src.portfolio import (
    strategy_forecast_guided, strategy_volatility_aware,
    strategy_correlation_diversified, strategy_sector_momentum,
    combine_strategies, allocate_capital, TOTAL_CAPITAL
)
from src.visualizations import (
    plot_stock_prices, plot_stock_individual, plot_forecast_vs_actual,
    plot_portfolio_allocation, plot_correlation_heatmap,
    plot_volatility, plot_decomposition, plot_model_comparison,
)

# ──────────────────────────────────────────────────────────────
# Output directories
# ──────────────────────────────────────────────────────────────
OUT_DIR = Path('outputs')
FORECAST_DIR = OUT_DIR / 'forecasts'
FIGURES_DIR = OUT_DIR / 'figures'
DATA_DIR = Path('data') / 'raw'

for d in [OUT_DIR, FORECAST_DIR, FIGURES_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)


def save_json(obj, path):
    """Save object as JSON (handles numpy types)."""
    class NpEncoder(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, (np.bool_,)):
                return bool(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
            return super().default(o)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, cls=NpEncoder)


def header(text):
    width = 60
    print(f"\n{'═' * width}")
    print(f"  {text}")
    print(f"{'═' * width}\n")


# ══════════════════════════════════════════════════════════════
# TASK 1 — Stock Universe Selection
# ══════════════════════════════════════════════════════════════

def task1_stock_selection():
    header("TASK 1 — Stock Universe Selection")

    print("  Selected stock universe across 5 sectors:\n")
    for sector, tickers in SECTOR_MAP.items():
        names = ', '.join(t.replace('.NS', '') for t in tickers)
        print(f"    {sector:>10}: {names}")

    tickers = get_all_tickers()
    ticker_sectors = get_ticker_sector()
    print(f"\n  Total stocks: {len(tickers)}\n")

    # Save selection rationale
    selection = {
        'sectors': {s: t for s, t in SECTOR_MAP.items()},
        'ticker_sectors': ticker_sectors,
        'rationale': {
            'Banking': 'High-liquidity large-cap banks representing India\'s financial backbone. HDFC Bank & ICICI for private sector; SBI for PSU exposure.',
            'IT': 'TCS and Infosys — top IT exporters. IT sector shows strong seasonal patterns tied to global tech demand and USD/INR dynamics.',
            'Pharma': 'Sun Pharma and Dr. Reddy\'s — defensive sector with low correlation to cyclicals. Interesting volatility profiles during health events.',
            'FMCG': 'ITC and Hindustan Unilever — stable, low-volatility stocks for portfolio risk reduction. Strong trend components.',
            'Auto': 'Tata Motors and Maruti — cyclical sector with high momentum potential. EV transition creates interesting regime shifts.',
        }
    }
    save_json(selection, OUT_DIR / 'task1_selection.json')
    return tickers, ticker_sectors


# ══════════════════════════════════════════════════════════════
# TASK 2 — Data Preprocessing
# ══════════════════════════════════════════════════════════════

def task2_preprocessing(tickers):
    header("TASK 2 — Data Fetching & Preprocessing")

    # Fetch data
    print("  Downloading historical data from Yahoo Finance...\n")
    raw_data = fetch_all_stocks(tickers, cache_dir=str(DATA_DIR))

    # Preprocess each stock
    processed = {}
    adf_results = {}
    print("\n  Preprocessing stocks:\n")

    for ticker, df in raw_data.items():
        name = ticker.replace('.NS', '')
        print(f"    {name:>15}:", end=" ")
        proc = preprocess_stock(df)
        processed[ticker] = proc
        adf = proc['adf_result']
        status = "Stationary ✓" if adf['is_stationary'] else f"Non-stationary (d={proc['d_order']})"
        print(f"Train={len(proc['train'])}, Test={len(proc['test'])}, ADF p={adf['p_value']:.4f} → {status}")
        adf_results[ticker] = {
            'p_value': adf['p_value'],
            'is_stationary': adf['is_stationary'],
            'd_order': proc['d_order'],
        }

    # Save ADF results
    save_json(adf_results, OUT_DIR / 'task2_stationarity.json')

    # Plot all stock prices
    plot_stock_prices(raw_data)
    print(f"\n  ✓ Stock prices plot saved to {FIGURES_DIR}/stock_prices.png")

    return raw_data, processed


# ══════════════════════════════════════════════════════════════
# TASK 3 — Time Series Forecasting
# ══════════════════════════════════════════════════════════════

def task3_forecasting(raw_data, processed):
    header("TASK 3 — Time Series Forecasting")

    all_forecasts = {}   # {model: {ticker: forecast_array}}
    all_metrics = {}     # {model: {ticker: {metric: value}}}
    forecast_details = {}  # {ticker: {model: {forecast, lower, upper}}}

    tickers = list(processed.keys())

    # ─── ARIMA / SARIMA ───────────────────────────────────────
    print("  [1/4] ARIMA / SARIMA\n")
    from src.models.arima_model import ARIMAForecaster

    arima_forecasts = {}
    arima_metrics = {}

    for ticker in tickers:
        name = ticker.replace('.NS', '')
        print(f"    {name:>15}:", end=" ")
        try:
            train_close = processed[ticker]['train']['Close']
            test_close = processed[ticker]['test']['Close']

            model = ARIMAForecaster(seasonal=True, m=5)
            model.fit(train_close)

            # Evaluate on test (use a fresh model for rolling prediction)
            model_eval = ARIMAForecaster(seasonal=True, m=5)
            model_eval.fit(train_close)
            preds = model_eval.predict_in_sample(test_close)

            metrics = evaluate_forecast(test_close.values, preds)
            arima_metrics[ticker] = metrics

            # Forecast 2 days ahead (use the full series)
            model_full = ARIMAForecaster(seasonal=True, m=5)
            model_full.fit(processed[ticker]['close'])
            fc, lower, upper = model_full.predict(steps=2)
            arima_forecasts[ticker] = fc

            if ticker not in forecast_details:
                forecast_details[ticker] = {}
            forecast_details[ticker]['ARIMA'] = {
                'forecast': fc.tolist(),
                'lower': lower.tolist(),
                'upper': upper.tolist(),
                'test_predictions': preds.tolist(),
            }

            diag = model_full.get_diagnostics()
            print(f"Order={diag['order']}, MAPE={metrics['MAPE (%)']:.2f}%, AIC={diag['aic']:.0f}")

            # Plot
            plot_forecast_vs_actual(test_close, preds, ticker, 'ARIMA', fc, lower, upper)
        except Exception as e:
            print(f"ERROR — {e}")
            arima_metrics[ticker] = {'MAPE (%)': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'Directional Accuracy (%)': np.nan}

    all_forecasts['ARIMA'] = arima_forecasts
    all_metrics['ARIMA'] = arima_metrics

    # ─── Prophet ──────────────────────────────────────────────
    print("\n  [2/4] Facebook Prophet\n")
    from src.models.prophet_model import ProphetForecaster

    prophet_forecasts = {}
    prophet_metrics = {}

    for ticker in tickers:
        name = ticker.replace('.NS', '')
        print(f"    {name:>15}:", end=" ")
        try:
            train_close = processed[ticker]['train']['Close']
            test_close = processed[ticker]['test']['Close']

            model = ProphetForecaster()
            model.fit(train_close)

            preds = model.predict_in_sample(test_close)
            metrics = evaluate_forecast(test_close.values, preds)
            prophet_metrics[ticker] = metrics

            # Forecast 2 days using full series
            model_full = ProphetForecaster()
            model_full.fit(processed[ticker]['close'])
            fc, lower, upper = model_full.predict(steps=2)
            prophet_forecasts[ticker] = fc

            if ticker not in forecast_details:
                forecast_details[ticker] = {}
            forecast_details[ticker]['Prophet'] = {
                'forecast': fc.tolist(),
                'lower': lower.tolist(),
                'upper': upper.tolist(),
                'test_predictions': preds.tolist(),
            }

            print(f"MAPE={metrics['MAPE (%)']:.2f}%")
            plot_forecast_vs_actual(test_close, preds, ticker, 'Prophet', fc, lower, upper)
        except Exception as e:
            print(f"ERROR — {e}")
            prophet_metrics[ticker] = {'MAPE (%)': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'Directional Accuracy (%)': np.nan}

    all_forecasts['Prophet'] = prophet_forecasts
    all_metrics['Prophet'] = prophet_metrics

    # ─── LSTM ─────────────────────────────────────────────────
    print("\n  [3/4] LSTM (PyTorch)\n")
    lstm_forecasts = {}
    lstm_metrics = {}

    try:
        from src.models.lstm_model import LSTMForecaster

        for ticker in tickers:
            name = ticker.replace('.NS', '')
            print(f"    {name:>15}:", end=" ")
            try:
                full_close = processed[ticker]['close']
                train_close = processed[ticker]['train']['Close']
                test_close = processed[ticker]['test']['Close']

                model = LSTMForecaster(seq_length=60, epochs=50, batch_size=32, patience=10)
                model.fit(full_close)

                preds = model.predict_in_sample(test_close, full_close)
                metrics = evaluate_forecast(test_close.values, preds)
                lstm_metrics[ticker] = metrics

                fc, lower, upper = model.predict(steps=2)
                lstm_forecasts[ticker] = fc

                if ticker not in forecast_details:
                    forecast_details[ticker] = {}
                forecast_details[ticker]['LSTM'] = {
                    'forecast': fc.tolist(),
                    'lower': lower.tolist(),
                    'upper': upper.tolist(),
                    'test_predictions': preds.tolist(),
                }

                print(f"MAPE={metrics['MAPE (%)']:.2f}%")
                plot_forecast_vs_actual(test_close, preds, ticker, 'LSTM', fc, lower, upper)
            except Exception as e:
                print(f"ERROR — {e}")
                lstm_metrics[ticker] = {'MAPE (%)': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'Directional Accuracy (%)': np.nan}
    except ImportError:
        print("    ⚠ PyTorch not available — skipping LSTM")

    if lstm_forecasts:
        all_forecasts['LSTM'] = lstm_forecasts
        all_metrics['LSTM'] = lstm_metrics

    # ─── Exponential Smoothing ────────────────────────────────
    print("\n  [4/4] Exponential Smoothing (Holt-Winters)\n")
    from src.models.exp_smoothing_model import ExpSmoothingForecaster

    es_forecasts = {}
    es_metrics = {}

    for ticker in tickers:
        name = ticker.replace('.NS', '')
        print(f"    {name:>15}:", end=" ")
        try:
            train_close = processed[ticker]['train']['Close']
            test_close = processed[ticker]['test']['Close']

            model = ExpSmoothingForecaster(trend='add', seasonal='add', seasonal_periods=63)
            model.fit(train_close)

            preds = model.predict_in_sample(test_close)
            metrics = evaluate_forecast(test_close.values, preds)
            es_metrics[ticker] = metrics

            # Forecast from full series
            model_full = ExpSmoothingForecaster(trend='add', seasonal='add', seasonal_periods=63)
            model_full.fit(processed[ticker]['close'])
            fc, lower, upper = model_full.predict(steps=2)
            es_forecasts[ticker] = fc

            if ticker not in forecast_details:
                forecast_details[ticker] = {}
            forecast_details[ticker]['ExpSmoothing'] = {
                'forecast': fc.tolist(),
                'lower': lower.tolist(),
                'upper': upper.tolist(),
                'test_predictions': preds.tolist(),
            }

            print(f"MAPE={metrics['MAPE (%)']:.2f}%")
            plot_forecast_vs_actual(test_close, preds, ticker, 'ExpSmoothing', fc, lower, upper)
        except Exception as e:
            print(f"ERROR — {e}")
            es_metrics[ticker] = {'MAPE (%)': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'Directional Accuracy (%)': np.nan}

    all_forecasts['ExpSmoothing'] = es_forecasts
    all_metrics['ExpSmoothing'] = es_metrics

    # Save all forecasts and metrics
    save_json(forecast_details, FORECAST_DIR / 'all_forecasts.json')
    save_json(all_metrics, OUT_DIR / 'task3_metrics.json')

    return all_forecasts, all_metrics, forecast_details


# ══════════════════════════════════════════════════════════════
# TASK 4 — Volatility & Trend Analysis
# ══════════════════════════════════════════════════════════════

def task4_volatility_analysis(processed):
    header("TASK 4 — Volatility & Trend Analysis")

    volatility_data = {}
    garch_diagnostics = {}
    trend_assessments = {}

    print("  Computing rolling volatility & GARCH for each stock...\n")

    from src.models.garch_model import GARCHForecaster

    for ticker, proc in processed.items():
        name = ticker.replace('.NS', '')
        print(f"    {name:>15}:", end=" ")

        # Rolling volatility
        roll_vol = compute_rolling_volatility(proc['close'], window=21)
        volatility_data[ticker] = roll_vol

        # GARCH
        try:
            garch = GARCHForecaster(p=1, q=1)
            garch.fit(proc['log_returns'])
            vol_forecast = garch.predict_volatility(steps=2)
            diag = garch.get_diagnostics()
            garch_diagnostics[ticker] = {
                **diag,
                'vol_forecast_2d': vol_forecast.tolist(),
                'avg_daily_vol': float(roll_vol.mean()),
            }
            print(f"Persistence={diag['persistence']:.3f}, Vol forecast={vol_forecast[0]:.4f}")
        except Exception as e:
            print(f"GARCH failed ({e}), using rolling vol")
            garch_diagnostics[ticker] = {
                'avg_daily_vol': float(roll_vol.mean()),
                'vol_forecast_2d': [float(roll_vol.iloc[-1])] * 2,
            }

        # STL Decomposition
        try:
            decomp = compute_stl_decomposition(proc['close'], period=63)
            trend = decomp.trend.dropna()
            trend_slope = (trend.iloc[-1] - trend.iloc[-63]) / trend.iloc[-63] * 100

            if trend_slope > 2:
                direction = 'UPWARD'
            elif trend_slope < -2:
                direction = 'DOWNWARD'
            else:
                direction = 'SIDEWAYS'

            trend_assessments[ticker] = {
                'direction': direction,
                'trend_slope_pct': round(float(trend_slope), 2),
            }

            plot_decomposition(decomp, ticker)
        except Exception as e:
            trend_assessments[ticker] = {'direction': 'UNKNOWN', 'error': str(e)}

    # Plot volatility
    plot_volatility(volatility_data)

    # Save results
    save_json(garch_diagnostics, OUT_DIR / 'task4_garch.json')
    save_json(trend_assessments, OUT_DIR / 'task4_trends.json')

    print(f"\n  Trend summary:")
    for ticker, ta in trend_assessments.items():
        print(f"    {ticker.replace('.NS',''):>15}: {ta['direction']} ({ta.get('trend_slope_pct', '?')}%)")

    return volatility_data, garch_diagnostics, trend_assessments


# ══════════════════════════════════════════════════════════════
# TASK 5 — Portfolio Construction
# ══════════════════════════════════════════════════════════════

def task5_portfolio(all_forecasts, processed, volatility_data, garch_diag, ticker_sectors):
    header("TASK 5 — Portfolio Construction (₹10,00,000)")

    tickers = list(processed.keys())

    # Current (last) prices
    current_prices = {t: float(processed[t]['close'].iloc[-1]) for t in tickers}

    # Build a log-returns DataFrame for correlation analysis
    returns_df = pd.DataFrame({
        t: processed[t]['log_returns'] for t in tickers
    }).dropna()

    # Pick the best-performing model's forecasts (lowest avg MAPE)
    best_model = None
    best_mape = float('inf')
    for model_name, forecasts in all_forecasts.items():
        if len(forecasts) == len(tickers):
            best_model = model_name
            break
    if best_model is None:
        best_model = list(all_forecasts.keys())[0]
    forecasts = all_forecasts[best_model]

    print(f"  Using {best_model} forecasts for Strategy A\n")

    # Strategy A — Forecast-Guided
    weights_a = strategy_forecast_guided(forecasts, current_prices)
    print("  Strategy A (Forecast-Guided):")
    for t, w in sorted(weights_a.items(), key=lambda x: -x[1]):
        print(f"    {t.replace('.NS',''):>15}: {w*100:.1f}%")

    # Strategy B — Volatility-Aware
    volatilities = {}
    for t in tickers:
        if t in garch_diag and 'avg_daily_vol' in garch_diag[t]:
            volatilities[t] = garch_diag[t]['avg_daily_vol']
        else:
            volatilities[t] = float(volatility_data[t].mean())

    weights_b = strategy_volatility_aware(volatilities)
    print("\n  Strategy B (Volatility-Aware):")
    for t, w in sorted(weights_b.items(), key=lambda x: -x[1]):
        print(f"    {t.replace('.NS',''):>15}: {w*100:.1f}%")

    # Strategy C — Correlation Diversification
    weights_c = strategy_correlation_diversified(returns_df)
    print("\n  Strategy C (Correlation-Diversified):")
    for t, w in sorted(weights_c.items(), key=lambda x: -x[1]):
        print(f"    {t.replace('.NS',''):>15}: {w*100:.1f}%")

    # Strategy D — Sector Momentum
    weights_d = strategy_sector_momentum(returns_df, ticker_sectors)
    print("\n  Strategy D (Sector Momentum):")
    for t, w in sorted(weights_d.items(), key=lambda x: -x[1]):
        print(f"    {t.replace('.NS',''):>15}: {w*100:.1f}%")

    # Combine: weighted average (A=40%, B=30%, C=20%, D=10%)
    final_weights = combine_strategies(
        [weights_a, weights_b, weights_c, weights_d],
        strategy_weights=[0.40, 0.30, 0.20, 0.10],
    )
    allocation = allocate_capital(final_weights)

    print(f"\n  {'─' * 50}")
    print(f"  FINAL PORTFOLIO ALLOCATION (Combined Strategy)")
    print(f"  {'─' * 50}")
    for _, row in allocation.iterrows():
        print(f"    {row['Stock'].replace('.NS',''):>15}: ₹{row['Amount (₹)']:>10,.0f}  ({row['Weight (%)']:>5.1f}%)")
    print(f"  {'─' * 50}")
    print(f"    {'TOTAL':>15}: ₹{allocation['Amount (₹)'].sum():>10,.0f}")

    # Plot
    plot_portfolio_allocation(allocation)
    plot_correlation_heatmap(returns_df)

    # Save
    allocation.to_csv(OUT_DIR / 'task5_allocation.csv', index=False)
    save_json({
        'strategy_weights': {'A': 0.40, 'B': 0.30, 'C': 0.20, 'D': 0.10},
        'final_weights': final_weights,
        'individual_strategies': {
            'A_forecast_guided': weights_a,
            'B_volatility_aware': weights_b,
            'C_correlation_diversified': weights_c,
            'D_sector_momentum': weights_d,
        },
    }, OUT_DIR / 'task5_strategies.json')

    return allocation, final_weights


# ══════════════════════════════════════════════════════════════
# TASK 6 — Model Comparison
# ══════════════════════════════════════════════════════════════

def task6_model_comparison(all_metrics):
    header("TASK 6 — Model Comparison")

    comparison = create_comparison_table(all_metrics)
    rankings = rank_models(comparison, 'MAPE (%)')

    print("  Average MAPE by model:\n")
    for _, row in rankings.iterrows():
        print(f"    {row['Model']:>20}: {row['Avg MAPE (%)']:.2f}%")

    print(f"\n  Detailed comparison:\n")
    print(comparison.to_string(index=False))

    # Plot
    plot_model_comparison(comparison)

    # Save
    comparison.to_csv(OUT_DIR / 'task6_comparison.csv', index=False)
    rankings.to_csv(OUT_DIR / 'task6_rankings.csv', index=False)

    return comparison, rankings


# ══════════════════════════════════════════════════════════════
# Save All Results for Dashboard
# ══════════════════════════════════════════════════════════════

def save_dashboard_data(raw_data, processed, forecast_details, allocation, garch_diag, trend_assessments, all_metrics):
    """Save all data needed by the dashboard in a single pickle file."""
    header("Saving Dashboard Data")

    # Build close-price DataFrame
    close_df = pd.DataFrame({
        t: proc['close'] for t, proc in processed.items()
    })

    # Build returns DataFrame
    returns_df = pd.DataFrame({
        t: proc['log_returns'] for t, proc in processed.items()
    }).dropna()

    # Test set data
    test_data = {t: proc['test']['Close'] for t, proc in processed.items()}

    dashboard_data = {
        'close_df': close_df,
        'returns_df': returns_df,
        'test_data': test_data,
        'forecast_details': forecast_details,
        'allocation': allocation,
        'garch_diagnostics': garch_diag,
        'trend_assessments': trend_assessments,
        'all_metrics': all_metrics,
        'sector_map': SECTOR_MAP,
        'ticker_sectors': get_ticker_sector(),
    }

    with open(OUT_DIR / 'dashboard_data.pkl', 'wb') as f:
        pickle.dump(dashboard_data, f)

    print(f"  ✓ Saved to {OUT_DIR / 'dashboard_data.pkl'}")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    start_time = time.time()

    print("\n" + "█" * 60)
    print("  TIME SERIES ANALYSIS 2026 — CAPSTONE PIPELINE")
    print("  IIT Guwahati × StockGro")
    print("█" * 60)

    # Task 1
    tickers, ticker_sectors = task1_stock_selection()

    # Task 2
    raw_data, processed = task2_preprocessing(tickers)

    # Task 3
    all_forecasts, all_metrics, forecast_details = task3_forecasting(raw_data, processed)

    # Task 4
    volatility_data, garch_diag, trend_assessments = task4_volatility_analysis(processed)

    # Task 5
    allocation, final_weights = task5_portfolio(
        all_forecasts, processed, volatility_data, garch_diag, ticker_sectors
    )

    # Task 6
    comparison, rankings = task6_model_comparison(all_metrics)

    # Save for dashboard
    save_dashboard_data(raw_data, processed, forecast_details, allocation, garch_diag, trend_assessments, all_metrics)

    elapsed = time.time() - start_time
    print(f"\n{'═' * 60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Results saved to: {OUT_DIR.resolve()}")
    print(f"  Figures saved to: {FIGURES_DIR.resolve()}")
    print(f"{'═' * 60}\n")

    print("  Next steps:")
    print("    1. Run the dashboard:  python dashboard/app.py")
    print("    2. Execute trades on StockGro (see GUIDE.md)")
    print("    3. After trading, run:  python record_actuals.py")
    print()


if __name__ == '__main__':
    main()
