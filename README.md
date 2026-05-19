# Time Series Analysis 2026 - Capstone Project

This repository contains the end-to-end implementation of the Time Series Analysis Capstone Project for IIT Guwahati x StockGro (2026). The project focuses on predicting NSE stock prices using quantitative modeling and constructing a live ₹10,00,000 virtual portfolio on the StockGro platform.

## 📊 Project Overview
The pipeline implements an 8-task methodology designed to screen, forecast, and allocate capital efficiently:

- **Task 1: Stock Selection** - Screened 25 NSE stocks across 5 sectors to select 11 highly liquid candidates.
- **Task 2: Data Preprocessing** - Time series decomposition, stationarity testing (ADF), differencing, and scaling.
- **Task 3: Time Series Forecasting** - Modeled 11 stocks using ARIMA, Prophet, Exponential Smoothing, and PyTorch LSTMs.
- **Task 4: Volatility & Trend Analysis** - Captured variance and trend signals using GARCH(1,1) and STL Decomposition.
- **Task 5: Portfolio Construction** - Allocated capital using a blended strategy (Forecast, Volatility, Correlation, Momentum).
- **Task 6: Model Comparison** - Benchmarked models by MAPE and Directional Accuracy. ARIMA was selected for allocation.
- **Task 7: Virtual Trading** - Executed the portfolio on StockGro across a 2-day trading window.
- **Task 8: Results Analysis** - Compared Day 2 forecasted predictions against actual market closing prices.

## 📂 Key Deliverables

- **`notebooks/capstone_main.ipynb`**: The primary executable Jupyter Notebook containing the full modeling pipeline, properly commented and structured.
- **`report.pdf`**: A beautifully formatted 10-page academic report outlining the methodology, rationale, strategy reflections, and visual charts.
- **`results/`**: Execution proof via screenshots of the StockGro portfolio on Day 1 and Day 2.
- **`outputs/`**: Generated CSV tables, JSON metrics, and visual forecast/decomposition plots.

## 📈 Bonus Task: Visual Dashboard

As an optional bonus, an interactive dashboard was built using **Plotly Dash** to provide a dynamic view of the project's outputs. 

The dashboard includes:
- **Interactive Forecast Plots**: Visualize ARIMA, Prophet, LSTM, and ExpSmoothing predictions alongside actuals.
- **Volatility Tracking**: 21-day rolling volatility graphs.
- **Portfolio Allocation**: Dynamic pie charts and sector breakdown graphs.
- **Correlation Heatmap**: Visualizing return correlations across the 11 selected stocks.

*Note: The dashboard code is located in the `dashboard/app.py` directory and is configured for simple cloud deployment (e.g., Render) using the provided `requirements.txt` and WSGI configurations.*

## 🚀 How to Run Locally

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the main pipeline (optional, outputs are already cached):
   ```bash
   python run_pipeline.py
   ```
3. Run the interactive dashboard:
   ```bash
   python dashboard/app.py
   ```
   *Then open `http://127.0.0.1:8050` in your browser.*

---
*Author: Abhishek Raghuwanshi*
