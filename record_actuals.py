"""
Record actual prices from StockGro after your trading window.

Usage:
    python record_actuals.py

This script will prompt you to enter the actual closing prices
for each stock on Day 1 and Day 2 of your trading window,
then compute prediction errors and portfolio performance.
"""

import sys, os, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.evaluation import evaluate_forecast
from src.visualizations import plot_predicted_vs_actual_live

OUT_DIR = Path('outputs')


def main():
    print("\n" + "═" * 60)
    print("  TASK 8 — Record Actual Prices from StockGro")
    print("═" * 60)

    # Load forecasts
    with open(OUT_DIR / 'forecasts' / 'all_forecasts.json') as f:
        forecast_details = json.load(f)

    # Load allocation
    allocation = pd.read_csv(OUT_DIR / 'task5_allocation.csv')

    tickers = list(forecast_details.keys())

    print("\n  Enter your trading day pair:")
    day1_label = input("    Day 1 date (e.g., 2026-05-12): ").strip()
    day2_label = input("    Day 2 date (e.g., 2026-05-13): ").strip()

    actual_prices = {}
    predicted_prices = {}

    print(f"\n  Enter actual closing prices from StockGro:\n")

    for ticker in tickers:
        name = ticker.replace('.NS', '')
        print(f"  {name}:")
        d1 = float(input(f"    Day 1 ({day1_label}) close: ₹"))
        d2 = float(input(f"    Day 2 ({day2_label}) close: ₹"))
        actual_prices[ticker] = [d1, d2]

        # Get predicted from best model
        models = forecast_details[ticker]
        best_model = list(models.keys())[0]  # pick first available
        predicted_prices[ticker] = models[best_model]['forecast'][:2]

    # ─── Compute metrics ──────────────────────────────────────
    print(f"\n{'═' * 60}")
    print("  PREDICTED vs ACTUAL COMPARISON")
    print(f"{'═' * 60}\n")

    results = []
    for ticker in tickers:
        name = ticker.replace('.NS', '')
        pred = np.array(predicted_prices[ticker])
        actual = np.array(actual_prices[ticker])

        pct_error_d1 = abs(pred[0] - actual[0]) / actual[0] * 100
        pct_error_d2 = abs(pred[1] - actual[1]) / actual[1] * 100

        actual_change = actual[1] - actual[0]
        pred_change = pred[1] - pred[0]
        dir_correct = "✓" if (actual_change > 0) == (pred_change > 0) else "✗"

        results.append({
            'Stock': name,
            'Pred Day1': f"₹{pred[0]:.2f}",
            'Actual Day1': f"₹{actual[0]:.2f}",
            'Error Day1': f"{pct_error_d1:.2f}%",
            'Pred Day2': f"₹{pred[1]:.2f}",
            'Actual Day2': f"₹{actual[1]:.2f}",
            'Error Day2': f"{pct_error_d2:.2f}%",
            'Direction': dir_correct,
        })

        print(f"  {name:>15}: Day1 err={pct_error_d1:.2f}%, Day2 err={pct_error_d2:.2f}%, Dir={dir_correct}")

    # Portfolio return
    print(f"\n{'─' * 60}")
    print("  PORTFOLIO PERFORMANCE")
    print(f"{'─' * 60}\n")

    total_invested = allocation['Amount (₹)'].sum()
    portfolio_return = 0

    for _, row in allocation.iterrows():
        ticker = row['Stock']
        if ticker in actual_prices:
            a = actual_prices[ticker]
            stock_return = (a[1] - a[0]) / a[0]
            weight = row['Weight (%)'] / 100
            portfolio_return += weight * stock_return
            contribution = weight * stock_return * total_invested
            print(f"    {ticker.replace('.NS',''):>15}: Return={stock_return*100:+.2f}%, Contribution=₹{contribution:+,.0f}")

    total_pnl = portfolio_return * total_invested
    print(f"\n    {'TOTAL RETURN':>15}: {portfolio_return*100:+.2f}%")
    print(f"    {'P&L':>15}: ₹{total_pnl:+,.0f}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / 'task8_comparison.csv', index=False)

    summary = {
        'trading_window': {'day1': day1_label, 'day2': day2_label},
        'actual_prices': actual_prices,
        'predicted_prices': predicted_prices,
        'portfolio_return_pct': round(portfolio_return * 100, 4),
        'portfolio_pnl': round(total_pnl, 2),
    }
    with open(OUT_DIR / 'task8_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # Plot
    plot_predicted_vs_actual_live(predicted_prices, actual_prices,
                                  days=[day1_label, day2_label])

    print(f"\n  ✓ Results saved to {OUT_DIR / 'task8_comparison.csv'}")
    print(f"  ✓ Summary saved to {OUT_DIR / 'task8_summary.json'}")
    print(f"  ✓ Plot saved to outputs/figures/predicted_vs_actual_live.png\n")


if __name__ == '__main__':
    main()
