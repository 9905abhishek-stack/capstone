# 📘 StockGro Manual Guide — Time Series Capstone 2026

This guide walks you through every manual step required to complete the capstone, from running the pipeline to executing trades on StockGro and submitting your final deliverables.

---

## ⚡ Quick Start (5-minute overview)

```
Step 1: pip install -r requirements.txt
Step 2: python run_pipeline.py
Step 3: python dashboard/app.py → open http://localhost:8050
Step 4: Execute trades on StockGro (see below)
Step 5: python record_actuals.py
Step 6: Convert notebook & submit
```

---

## 📋 Step 1 — Install Dependencies

```powershell
cd d:\capstone
pip install -r requirements.txt
```

> **Note**: Prophet may require additional setup on Windows. If `prophet` fails to install, try:
> ```powershell
> pip install pystan==2.19.1.1
> pip install prophet
> ```
> If Prophet still fails, the pipeline will still run with ARIMA, LSTM, and Exponential Smoothing.

> **Note**: PyTorch installation depends on your GPU. For CPU-only:
> ```powershell
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> ```

---

## 📋 Step 2 — Run the Pipeline

```powershell
python run_pipeline.py
```

This executes Tasks 1–6 automatically:
- Downloads 5 years of NSE data for 11 stocks
- Preprocesses and tests for stationarity
- Trains 4 forecasting models per stock (ARIMA, Prophet, LSTM, Exponential Smoothing)
- Runs GARCH volatility analysis and STL decomposition
- Constructs a blended portfolio allocation
- Compares all models by MAPE, RMSE, and directional accuracy

**Expected runtime**: 5–15 minutes depending on hardware (LSTM training is the bottleneck).

**Outputs created**:
```
outputs/
├── dashboard_data.pkl           # All data for the dashboard
├── task1_selection.json         # Stock selection rationale
├── task2_stationarity.json      # ADF test results
├── task3_metrics.json           # Model evaluation metrics
├── task4_garch.json             # GARCH diagnostics
├── task4_trends.json            # Trend assessments
├── task5_allocation.csv         # Final portfolio allocation
├── task5_strategies.json        # Individual strategy weights
├── task6_comparison.csv         # Model comparison table
├── task6_rankings.csv           # Model rankings
├── forecasts/
│   └── all_forecasts.json       # All forecasted prices
└── figures/
    ├── stock_prices.png
    ├── forecast_*.png           # Per stock, per model
    ├── portfolio_allocation.png
    ├── correlation_heatmap.png
    ├── volatility.png
    └── decomposition_*.png
```

---

## 📋 Step 3 — Launch the Dashboard

```powershell
python dashboard/app.py
```

Open **http://localhost:8050** in your browser.

The dashboard has 5 pages:
1. **Overview** — KPI cards + stock price chart
2. **Forecasts** — Select any stock + model, see predicted vs actual + 2-day forecast
3. **Volatility** — Rolling volatility chart + GARCH trend assessment table
4. **Portfolio** — Allocation pie chart, sector breakdown, correlation heatmap
5. **Models** — Side-by-side model comparison with sortable/filterable metrics table

Take screenshots of the dashboard for your report.

---

## 📋 Step 4 — Register & Trade on StockGro

### 4a. Register for the Capstone

1. Go to: **https://community.stockgro.club/form/088d176b-8854-46c4-ab9d-fd072acffb48**
2. Fill in your details and submit
3. Note your registered mobile number

### 4b. Install/Update StockGro

1. Download from: **https://stockgro.onelink.me/vNON/21jikjek**
2. Install or update to the latest version
3. Log in with your **registered mobile number**

### 4c. Access the Event Portfolio

1. Open the StockGro app
2. Go to **Trackers**
3. Find the event: **"Portfolio - Time Series Analysis 2026"**
4. If you don't see it, update the app and try again

### 4d. Choose Your Trading Window

Pick **any 2 consecutive trading days** from May 11–15:

| Day 1 | Day 2 |
|-------|-------|
| Mon 11 May | Tue 12 May |
| Tue 12 May | Wed 13 May |
| Wed 13 May | Thu 14 May |
| Thu 14 May | Fri 15 May |

**Recommendation**: Choose **Mon 11 → Tue 12** to get it done early.

### 4e. Execute Trades

On your **Day 1**, before 3:25 PM (market close):

1. Open the event portfolio on StockGro
2. Look at your allocation from `outputs/task5_allocation.csv`:

   | Stock | Amount (₹) |
   |-------|-----------|
   | (check your allocation table) | |

3. For each stock, buy shares worth approximately the allocated amount:
   - **Shares to buy** = Amount (₹) ÷ Current Market Price
   - Example: If HDFC Bank is at ₹1,650 and allocation is ₹1,50,000 → Buy ~91 shares
   
4. Execute all trades before 3:25 PM
5. Take a screenshot of your portfolio after trades are placed

### 4f. Record Performance

On your **Day 2**, at market close (3:25 PM):

1. Note the actual closing price of each stock from StockGro
2. Take a screenshot of your portfolio summary showing total return
3. Record these values — you'll need them for Task 8

---

## 📋 Step 5 — Record Actual Results (Task 8)

After market close on Day 2:

```powershell
python record_actuals.py
```

This interactive script will:
1. Ask for your Day 1 and Day 2 dates
2. Prompt you to enter actual closing prices for each stock
3. Compute prediction errors (MAPE per stock)
4. Check directional accuracy
5. Calculate total portfolio return and P&L
6. Save comparison table and summary to `outputs/`
7. Generate a predicted vs actual bar chart

---

## 📋 Step 6 — Create the Jupyter Notebook

The pipeline runs as a Python script, but submission requires `.ipynb`. To convert:

### Option A: Use the generate script
```powershell
python generate_notebook.py
```

### Option B: Manual in VS Code
1. Open `run_pipeline.py` in VS Code
2. Copy sections into a new Jupyter notebook
3. Add markdown cells explaining each task
4. Add the visualizations inline

### What the notebook should contain:
1. **Introduction & Objective**
2. **Task 1**: Stock selection with sector justification
3. **Task 2**: Preprocessing — missing values, ADF tests, log returns
4. **Task 3**: Model training & forecasting (show code + output for each model)
5. **Task 4**: Volatility analysis — GARCH, STL decomposition
6. **Task 5**: Portfolio construction — show all 4 strategies + final allocation
7. **Task 6**: Model comparison table
8. **Task 7**: StockGro execution summary (include screenshots)
9. **Task 8**: Predicted vs actual comparison (include the table from `record_actuals.py`)
10. **Reflections**: What worked, what didn't, improvements

---

## 📋 Step 7 — Write the Report (Max 10 Pages)

Suggested structure:

| Section | Pages | Content |
|---------|-------|---------|
| 1. Methodology | 1 | Models used, data period, approach |
| 2. Stock Selection | 1 | Sector rationale, data-driven justification |
| 3. Preprocessing | 0.5 | ADF results, handling missing data |
| 4. Forecasting Results | 2 | Per-model results, confidence intervals |
| 5. Volatility & Trends | 1 | GARCH results, STL decomposition, trend directions |
| 6. Portfolio Strategy | 1.5 | Four strategies, final allocation table |
| 7. Model Comparison | 1 | Comparison table, strengths/weaknesses |
| 8. StockGro Execution | 0.5 | Trades placed, screenshots |
| 9. Predicted vs Actual | 1 | Comparison table, portfolio return |
| 10. Reflections | 0.5 | Lessons learned, improvements |

---

## 📋 Step 8 — Submit

**Submission link**: https://forms.cloud.microsoft/r/tVvn2jKSug

**Deadline**: 17th May 2026, End of Day

**What to submit**:
- [x] Python Notebook (.ipynb) — full pipeline
- [x] Final Report (PDF, max 10 pages)
- [x] StockGro execution screenshots
- [x] Portfolio allocation summary
- [x] Dashboard screenshots (optional bonus)

---

## 🔧 Troubleshooting

| Issue | Solution |
|-------|----------|
| `yfinance` returns empty data | Check internet connection; some tickers may need `.NS` suffix |
| Prophet installation fails | Use conda: `conda install -c conda-forge prophet` |
| LSTM training is slow | Reduce epochs in `run_pipeline.py` (change `epochs=50` to `epochs=20`) |
| Dashboard won't load | Make sure `outputs/dashboard_data.pkl` exists (run pipeline first) |
| StockGro event not visible | Update the app to latest version |

---

## 📁 Project Structure

```
d:\capstone\
├── run_pipeline.py              ← Run this first (Tasks 1-6)
├── record_actuals.py            ← Run after trading (Task 8)
├── requirements.txt             ← Dependencies
├── GUIDE.md                     ← You are here
├── src/
│   ├── data_fetcher.py          ← Yahoo Finance data download
│   ├── preprocessor.py          ← Cleaning, ADF, scaling
│   ├── evaluation.py            ← MAPE, RMSE, directional accuracy
│   ├── portfolio.py             ← 4 allocation strategies
│   ├── visualizations.py        ← All matplotlib plots
│   └── models/
│       ├── arima_model.py       ← ARIMA / SARIMA
│       ├── prophet_model.py     ← Facebook Prophet
│       ├── lstm_model.py        ← PyTorch LSTM
│       ├── exp_smoothing_model.py ← Holt-Winters
│       └── garch_model.py       ← GARCH(1,1) volatility
├── dashboard/
│   └── app.py                   ← Interactive Plotly Dash dashboard
├── data/raw/                    ← Cached stock CSVs
└── outputs/
    ├── figures/                 ← All generated plots
    ├── forecasts/               ← Model forecast JSONs
    └── *.csv / *.json           ← Task results
```

---

Good luck! 🚀📈
