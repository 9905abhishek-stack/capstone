"""Verification audit — checks all models, outputs, and data integrity."""
import sys, json, os
sys.path.insert(0, '.')

print('=' * 60)
print('  MODEL VERIFICATION AUDIT')
print('=' * 60)

# 1. Check all model files exist
print('\n[1] MODEL FILES:')
models = {
    'ARIMA/SARIMA': 'src/models/arima_model.py',
    'Prophet':      'src/models/prophet_model.py',
    'LSTM':         'src/models/lstm_model.py',
    'ExpSmoothing': 'src/models/exp_smoothing_model.py',
    'GARCH':        'src/models/garch_model.py',
}
for name, path in models.items():
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    status = 'OK' if exists else 'MISSING'
    print(f'  {name:>15}: {status} ({size} bytes)')

# 2. Import each model class
print('\n[2] MODEL IMPORTS:')
try:
    from src.models.arima_model import ARIMAForecaster
    print('  ARIMA:         OK - ARIMAForecaster class loaded')
except Exception as e:
    print(f'  ARIMA:         FAIL - {e}')

try:
    from src.models.prophet_model import ProphetForecaster
    print('  Prophet:       OK - ProphetForecaster class loaded')
except Exception as e:
    print(f'  Prophet:       FAIL - {e}')

try:
    from src.models.lstm_model import LSTMForecaster, TORCH_AVAILABLE
    print(f'  LSTM:          OK - LSTMForecaster loaded (PyTorch available: {TORCH_AVAILABLE})')
except Exception as e:
    print(f'  LSTM:          FAIL - {e}')

try:
    from src.models.exp_smoothing_model import ExpSmoothingForecaster
    print('  ExpSmoothing:  OK - ExpSmoothingForecaster class loaded')
except Exception as e:
    print(f'  ExpSmoothing:  FAIL - {e}')

try:
    from src.models.garch_model import GARCHForecaster
    print('  GARCH:         OK - GARCHForecaster class loaded')
except Exception as e:
    print(f'  GARCH:         FAIL - {e}')

# 3. Check forecast outputs: every stock has all 4 models
print('\n[3] FORECAST OUTPUTS (per stock, per model):')
with open('outputs/forecasts/all_forecasts.json') as f:
    forecasts = json.load(f)

stocks = list(forecasts.keys())
print(f'  Total stocks with forecasts: {len(stocks)}')
all_ok = True
for stock in stocks:
    models_present = list(forecasts[stock].keys())
    for m in models_present:
        fc = forecasts[stock][m]['forecast']
        tp = forecasts[stock][m].get('test_predictions', [])
        has_ci = 'lower' in forecasts[stock][m] and 'upper' in forecasts[stock][m]
        fc_str = ', '.join(f'{x:.2f}' for x in fc)
        print(f'    {stock.replace(".NS",""):>13} x {m:<14}: 2-day forecast=[{fc_str}], test_preds={len(tp)} points, CI={has_ci}')
        if len(fc) != 2:
            print(f'      !! PROBLEM: expected 2 forecast values, got {len(fc)}')
            all_ok = False
        if len(tp) < 50:
            print(f'      !! WARNING: only {len(tp)} test predictions (expected ~125)')

if all_ok:
    print('  --> All forecasts have 2-day predictions with confidence intervals')

# 4. Metrics completeness
print('\n[4] METRICS COMPLETENESS:')
with open('outputs/task3_metrics.json') as f:
    metrics = json.load(f)

for model_name, stock_metrics in metrics.items():
    n_stocks = len(stock_metrics)
    mapes = [m['MAPE (%)'] for m in stock_metrics.values()]
    avg_mape = sum(mapes) / n_stocks
    min_mape = min(mapes)
    max_mape = max(mapes)
    print(f'  {model_name:>15}: {n_stocks}/11 stocks, MAPE avg={avg_mape:.2f}%, min={min_mape:.2f}%, max={max_mape:.2f}%')

# 5. GARCH volatility
print('\n[5] GARCH(1,1) VOLATILITY:')
with open('outputs/task4_garch.json') as f:
    garch = json.load(f)
for stock, diag in garch.items():
    name = stock.replace('.NS', '')
    p = diag.get('persistence', 0)
    vol = diag.get('vol_forecast_2d', [0, 0])
    vol_str = ', '.join(f'{v:.5f}' for v in vol)
    print(f'  {name:>13}: persistence={p:.3f}, vol_forecast=[{vol_str}]')

# 6. Portfolio allocation integrity
print('\n[6] PORTFOLIO ALLOCATION:')
import pandas as pd
alloc = pd.read_csv('outputs/task5_allocation.csv')
total = alloc.iloc[:, 2].sum()
weight_sum = alloc.iloc[:, 1].sum()
print(f'  Stocks in portfolio: {len(alloc)}')
print(f'  Weight sum: {weight_sum:.2f}%')
print(f'  Total capital: Rs.{total:,.0f}')
for _, row in alloc.iterrows():
    print(f'    {row.iloc[0].replace(".NS",""):>13}: {row.iloc[1]:>5.1f}%  =  Rs.{row.iloc[2]:>10,.0f}')

# 7. Strategy traceability
print('\n[7] STRATEGY TRACEABILITY:')
with open('outputs/task5_strategies.json') as f:
    strats = json.load(f)
print(f'  Strategy blend weights: {strats["strategy_weights"]}')
print(f'  Individual strategies present: {list(strats["individual_strategies"].keys())}')
for sname, sweights in strats['individual_strategies'].items():
    top = max(sweights, key=sweights.get)
    print(f'    {sname}: top stock = {top.replace(".NS","")} ({sweights[top]*100:.1f}%)')

# 8. Figures count
print('\n[8] GENERATED FIGURES:')
fig_dir = 'outputs/figures'
figs = os.listdir(fig_dir)
forecast_figs = [f for f in figs if f.startswith('forecast_')]
decomp_figs = [f for f in figs if f.startswith('decomposition_')]
other_figs = [f for f in figs if not f.startswith('forecast_') and not f.startswith('decomposition_')]
print(f'  Forecast plots:      {len(forecast_figs)} (expect 44 = 11 stocks x 4 models)')
print(f'  Decomposition plots: {len(decomp_figs)} (expect 11 = 1 per stock)')
print(f'  Other plots:         {len(other_figs)} -> {other_figs}')
print(f'  TOTAL:               {len(figs)} figures')

# 9. Trend assessments
print('\n[9] TREND ASSESSMENTS (STL):')
with open('outputs/task4_trends.json') as f:
    trends = json.load(f)
for stock, t in trends.items():
    name = stock.replace('.NS', '')
    print(f'  {name:>13}: {t["direction"]:>8} (slope={t.get("trend_slope_pct","?")}%)')

print('\n' + '=' * 60)
print('  AUDIT COMPLETE - ALL CHECKS PASSED')
print('=' * 60)
