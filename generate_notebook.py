"""
Generate the final submission Jupyter Notebook (.ipynb) for the capstone project.
Structure follows the provided sample notebook with our 8 tasks integrated.
"""
import json
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
from pathlib import Path


def md(text):
    return new_markdown_cell(text)

def code(text):
    return new_code_cell(text)

def create_notebook(student_name="Abhishek Raghuwanshi", phone_number="8839661736", email="9905abhishek@gmail.com"):
    nb = new_notebook()
    nb.metadata['kernelspec'] = {
        'display_name': 'Python 3', 'language': 'python', 'name': 'python3',
    }
    cells = []

    # ── TITLE & OVERVIEW ──
    cells.append(md(
        f"# **Time Series Analysis 2026 — Capstone Project**\n\n"
        f"**Student Name**: {student_name}\n\n"
        f"**Registered Phone Number**: {phone_number}\n\n"
        f"**Email**: {email}"
    ))
    
    cells.append(md(
        "## Project Overview\n"
        "This project applies time series forecasting to predict NSE stock prices and construct a ₹10,00,000 virtual portfolio on StockGro.\n"
        "The pipeline includes:\n"
        "1. Stock Universe Selection (Task 1)\n"
        "2. Data Preprocessing (Task 2)\n"
        "3. Time Series Forecasting (Task 3)\n"
        "4. Volatility & Trend Analysis (Task 4)\n"
        "5. Portfolio Construction (Task 5)\n"
        "6. Model Comparison (Task 6)\n"
        "7. Virtual Trading on StockGro (Task 7)\n"
        "8. Predicted vs Actual Comparison (Task 8)"
    ))

    # ── LIBRARIES ──
    cells.append(md("## Key Libraries Used\n"
        "- `yfinance`: Downloading historical stock data\n"
        "- `pandas`, `numpy`: Data manipulation\n"
        "- `matplotlib`, `seaborn`, `plotly`: Visualization\n"
        "- `pmdarima`: Auto ARIMA modeling\n"
        "- `prophet`: Facebook Prophet forecasting\n"
        "- `torch`: PyTorch for LSTM sequence modeling\n"
        "- `statsmodels`: Exponential smoothing, STL decomposition, ADF tests\n"
        "- `arch`: GARCH volatility modeling"
    ))
    
    cells.append(md("### **Import Libraries**"))
    cells.append(code(
        "# Import required libraries\n"
        "import sys, os, json, warnings\n"
        "warnings.filterwarnings('ignore')\n\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "import seaborn as sns\n\n"
        "# Custom project modules\n"
        "sys.path.insert(0, '.')\n"
        "from src.data_fetcher import fetch_all_stocks, SECTOR_MAP, get_all_tickers\n"
        "from src.preprocessor import preprocess_stock\n"
        "from src.evaluation import compute_mape, compute_rmse\n\n"
        "%matplotlib inline\n"
        "plt.style.use('dark_background')\n"
        "print('All libraries imported successfully.')"
    ))

    # ── TASK 1 & DATA LOADING ──
    cells.append(md("## **Time Series Data**\n### Loading data & Stock Selection (Task 1)"))
    cells.append(code(
        "# Retrieve historical stock price data (Jan 2021 - Dec 2025)\n"
        "tickers = get_all_tickers()\n"
        "raw_data = fetch_all_stocks(tickers, cache_dir='data/raw')\n\n"
        "print(f'\\nSelected Stock Universe ({len(tickers)} stocks):')\n"
        "for sector, tkrs in SECTOR_MAP.items():\n"
        "    names = ', '.join(t.replace('.NS', '') for t in tkrs)\n"
        "    print(f'  {sector:>10}: {names}')"
    ))

    cells.append(md("### Plotting historical stock data"))
    cells.append(code(
        "fig, ax = plt.subplots(figsize=(14, 7))\n"
        "fig.patch.set_facecolor('#0d1117')\n"
        "ax.set_facecolor('#161b22')\n"
        "colors = ['#00D4AA','#FF6B6B','#4ECDC4','#FFE66D','#95E1D3',\n"
        "          '#A8E6CF','#FF8B94','#F38181','#AA96DA','#FCBAD3','#6C5B7B']\n"
        "for i, (ticker, df) in enumerate(raw_data.items()):\n"
        "    ax.plot(df.index, df['Close'], label=ticker.replace('.NS',''), color=colors[i % len(colors)], linewidth=1.2)\n"
        "ax.set_title('Stock Universe - Daily Close Prices (2021-2025)', color='#c9d1d9')\n"
        "ax.set_ylabel('Price (INR)', color='#8b949e')\n"
        "ax.legend(ncol=3)\n"
        "ax.grid(True, alpha=0.2)\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ))

    # ── TASK 2 PREPROCESSING ──
    cells.append(md(
        "### Creating train and test datasets & Scaling (Task 2)\n"
        "Preprocessing includes handling missing values, ADF stationarity tests, differencing, min-max scaling for LSTM, and splitting."
    ))
    cells.append(code(
        "processed = {}\n"
        "print(f'{\"Stock\":>15} {\"ADF p-value\":>12} {\"Stationary?\":>12} {\"Train\":>7} {\"Test\":>6}')\n"
        "print('-' * 60)\n"
        "for ticker, df in raw_data.items():\n"
        "    processed[ticker] = preprocess_stock(df)\n"
        "    name = ticker.replace('.NS', '')\n"
        "    adf = processed[ticker]['adf_result']\n"
        "    n_train, n_test = len(processed[ticker]['train']), len(processed[ticker]['test'])\n"
        "    stat = 'Yes' if adf['is_stationary'] else 'No'\n"
        "    print(f'{name:>15} {adf[\"p_value\"]:>12.4f} {stat:>12} {n_train:>7} {n_test:>6}')"
    ))

    # ── TASK 3 FORECASTING ──
    cells.append(md("## **Time Series Prediction Models** (Task 3 & 4)\nModels evaluated: ARIMA, Prophet, LSTM, Exponential Smoothing. GARCH used for volatility."))
    cells.append(code(
        "# Load pre-computed forecasts and GARCH volatility results\n"
        "with open('outputs/forecasts/all_forecasts.json') as f:\n"
        "    all_forecasts = json.load(f)\n"
        "with open('outputs/task4_garch.json') as f:\n"
        "    garch_results = json.load(f)\n\n"
        "print(f'{\"Stock\":>13} {\"ARIMA D1\":>10} {\"ARIMA D2\":>10} | {\"GARCH Vol\":>10}')\n"
        "print('-' * 55)\n"
        "for ticker, models in all_forecasts.items():\n"
        "    fc = models['ARIMA']['forecast']\n"
        "    vol = garch_results[ticker]['avg_daily_vol']\n"
        "    name = ticker.replace('.NS', '')\n"
        "    print(f'{name:>13} {fc[0]:>10.2f} {fc[1]:>10.2f} | {vol:>9.4f}%')"
    ))

    cells.append(md("### Time Series Prediction: Actual vs Predicted on Test Set"))
    cells.append(code(
        "example_ticker = 'HDFCBANK.NS'\n"
        "test_actual = processed[example_ticker]['test']['Close'].values\n"
        "test_pred = all_forecasts[example_ticker]['ARIMA']['test_predictions']\n\n"
        "plt.figure(figsize=(14, 5), facecolor='#0d1117')\n"
        "ax = plt.gca()\n"
        "ax.set_facecolor('#161b22')\n"
        "ax.plot(test_actual, label='Actual', color='#00D4AA')\n"
        "ax.plot(test_pred, label='ARIMA Predicted', color='#FF6B6B', linestyle='--')\n"
        "ax.set_title('HDFC Bank - ARIMA Test Set Predictions', color='#c9d1d9')\n"
        "ax.legend()\n"
        "plt.show()"
    ))

    # ── TASK 5 ALLOCATION ──
    cells.append(md("## **Portfolio Construction** (Task 5)\nAllocating ₹10,00,000 using Forecast-Guided, Volatility-Aware, Correlation-Diversified, and Sector Momentum strategies."))
    cells.append(code(
        "allocation = pd.read_csv('outputs/task5_allocation.csv')\n"
        "print(allocation.to_string(index=False))\n\n"
        "fig, ax = plt.subplots(figsize=(8, 8), facecolor='#0d1117')\n"
        "ax.pie(allocation.iloc[:,1].values, labels=[str(s).replace('.NS','') for s in allocation.iloc[:,0]], autopct='%1.1f%%', textprops={'color': '#c9d1d9'})\n"
        "ax.set_title('Final Portfolio Allocation', color='#c9d1d9')\n"
        "plt.show()"
    ))

    # ── TASK 6 COMPARISON ──
    cells.append(md("## **Model Comparison** (Task 6)"))
    cells.append(code(
        "comparison = pd.read_csv('outputs/task6_comparison.csv')\n"
        "print(comparison.to_string(index=False))\n"
        "print('\\nARIMA selected for portfolio allocation due to best (lowest) MAPE.')"
    ))

    # ── TASK 7 & 8 RESULTS ──
    cells.append(md("## **StockGro Virtual Trading & Results** (Task 7 & 8)"))
    cells.append(code(
        "with open('outputs/task8_actuals.json') as f:\n"
        "    actuals = json.load(f)\n"
        "with open('outputs/forecasts/all_forecasts.json') as f:\n"
        "    forecasts = json.load(f)\n\n"
        "rows = []\n"
        "for ticker, info in actuals['stocks'].items():\n"
        "    pred = forecasts[ticker]['ARIMA']['forecast']\n"
        "    act_d1, act_d2 = info['day1_ltp'], info['day2_ltp']\n"
        "    err_d2 = abs(pred[1] - act_d2) / act_d2 * 100\n"
        "    rows.append({'Stock': ticker.replace('.NS',''), 'Pred D2': round(pred[1],2), 'Actual D2': act_d2, 'Error %': round(err_d2,2)})\n\n"
        "print(pd.DataFrame(rows).to_string(index=False))\n"
        "print(f'\\nInvested: Rs.{actuals[\"invested_value\"]:,.0f}')\n"
        "print(f'Day 2 Value: Rs.{actuals[\"day2_portfolio_value\"]:,.0f} ({actuals[\"day2_return_pct\"]:+.2f}%)')\n"
        "print(f'P&L: Rs.{actuals[\"day2_total_loss\"]:,.0f}')"
    ))

    # ── CONCLUSION ──
    cells.append(md(
        "## **Conclusion**\n"
        "- **What Worked**: ARIMA model performed excellently in backtesting (0.77% MAPE). GARCH effectively captured volatility signatures.\n"
        "- **What Didn't Work**: Gap between Dec 2025 training end and May 2026 trading window caused significant live prediction errors (avg 20% MAPE) due to market shifts.\n"
        "- **Improvements**: Retrain models daily on the most recent data, add exogenous macro-economic variables, and implement stop-loss strategies."
    ))

    nb['cells'] = cells
    out_path = Path('notebooks/capstone_main.ipynb')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f'Notebook saved to {out_path}')


if __name__ == '__main__':
    create_notebook()
