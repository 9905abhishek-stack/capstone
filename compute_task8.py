"""Generate Task 8 comparison and the final submission report."""
import json, os, sys
import numpy as np
sys.path.insert(0, '.')

with open('outputs/forecasts/all_forecasts.json') as f:
    fc = json.load(f)
with open('outputs/task8_actuals.json') as f:
    actuals = json.load(f)

print("PREDICTED (ARIMA) vs ACTUAL COMPARISON")
print("=" * 95)
header = f"{'Stock':>13} {'Pred D1':>10} {'Pred D2':>10} {'Act D1':>10} {'Act D2':>10} {'Err D1':>8} {'Err D2':>8} {'Dir':>4}"
print(header)
print("-" * 95)

errors_d1 = []
errors_d2 = []
dir_correct = 0
total = 0

for ticker, info in actuals['stocks'].items():
    arima_fc = fc[ticker]['ARIMA']['forecast']
    act_d1 = info['day1_ltp']
    act_d2 = info['day2_ltp']
    err_d1 = abs(arima_fc[0] - act_d1) / act_d1 * 100
    err_d2 = abs(arima_fc[1] - act_d2) / act_d2 * 100
    errors_d1.append(err_d1)
    errors_d2.append(err_d2)

    # Directional accuracy: did model predict the right direction from D1 to D2?
    pred_dir = arima_fc[1] - arima_fc[0]
    actual_dir = act_d2 - act_d1
    dc = "Y" if (pred_dir > 0) == (actual_dir > 0) or (pred_dir == 0 and actual_dir == 0) else "N"
    if dc == "Y":
        dir_correct += 1
    total += 1

    name = ticker.replace('.NS', '')
    print(f"{name:>13} {arima_fc[0]:>10.2f} {arima_fc[1]:>10.2f} {act_d1:>10.2f} {act_d2:>10.2f} {err_d1:>7.2f}% {err_d2:>7.2f}% {dc:>4}")

print("-" * 95)
print(f"{'AVG MAPE':>13} {'':>10} {'':>10} {'':>10} {'':>10} {np.mean(errors_d1):>7.2f}% {np.mean(errors_d2):>7.2f}% {dir_correct}/{total}")
print()
print(f"Portfolio: Invested Rs.{actuals['invested_value']:,.2f}")
print(f"  Day 1 value: Rs.{actuals['day1_portfolio_value']:,.2f} ({actuals['day1_return_pct']:+.2f}%)")
print(f"  Day 2 value: Rs.{actuals['day2_portfolio_value']:,.2f} ({actuals['day2_return_pct']:+.2f}%)")
print(f"  Total P&L:   Rs.{actuals['day2_total_loss']:+,.2f}")
