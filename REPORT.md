# Time Series Analysis 2026 — Capstone Report
# IIT Guwahati x StockGro
# ==============================================================================

## 1. Methodology and Models Used

### Objective
Apply time series forecasting techniques to select NSE stocks, build predictive
models, allocate Rs.10,00,000 in a virtual portfolio on StockGro, and compare
predictions against actual market outcomes.

### Models Implemented
We implemented **5 forecasting/analysis models**:

| Model | Library | Purpose |
|-------|---------|---------|
| **ARIMA/SARIMA** | pmdarima | Price forecasting with auto-order selection via AIC |
| **Facebook Prophet** | prophet | Trend + seasonality decomposition with Indian market holidays |
| **LSTM** | PyTorch | Deep learning sequence model (2-layer, 60-step lookback) |
| **Exponential Smoothing** | statsmodels | Holt-Winters triple exponential smoothing |
| **GARCH(1,1)** | arch | Conditional volatility estimation |

### Dataset
- **Source**: Yahoo Finance via `yfinance`
- **Period**: January 2021 - December 2025 (1,235 trading days)
- **Split**: Training (Jan 2021 - Jun 2025), Testing (Jul - Dec 2025)

---

## 2. Stock Selection Rationale (Task 1)

We screened **25 candidate stocks** across 5 sectors using three quantitative criteria:

1. **30-day Rolling Volatility** — standard deviation of log returns
2. **STL Trend Strength** — ratio of trend variance to total variance
3. **6-Month Sector Momentum** — cumulative return over trailing 126 days

### Final Selection (11 stocks):

| Sector | Selected | Key Justification |
|--------|----------|-------------------|
| Banking | HDFC Bank, ICICI Bank, SBI | Highest liquidity (15-25M daily vol), strong trends |
| IT | TCS, Infosys | Top-2 by market cap, mean-reversion signals |
| Pharma | Sun Pharma, Dr. Reddy's | Low correlation to cyclicals, defensive |
| FMCG | ITC, Hindustan Unilever | Lowest volatility (4th-8th percentile), portfolio stabilizers |
| Auto | M&M, Maruti | Highest sector momentum (+25.6%), strong upward trends |

**Rejected candidates** (14 stocks): Kotak Bank, Axis Bank, Wipro, HCL Tech, Tech Mahindra, Cipla, Divi's Lab, Apollo Hospitals, Nestle, Britannia, Dabur, Hero MotoCorp, Eicher Motors, Bajaj Auto — due to lower liquidity, weaker trends, or redundancy within sector.

---

## 3. Data Preprocessing Steps (Task 2)

### Missing Values
Forward-fill applied to handle gaps from market holidays and trading halts.

### Stationarity Testing
Augmented Dickey-Fuller (ADF) test applied to each stock's closing price series:

| Stock | ADF p-value | Stationary? | Action |
|-------|-------------|-------------|--------|
| HDFCBANK | 0.5843 | No | Differenced (d=1) |
| ICICIBANK | 0.7105 | No | Differenced (d=1) |
| SBIN | 0.8456 | No | Differenced (d=1) |
| TCS | 0.1793 | No | Differenced (d=1) |
| INFY | 0.1418 | No | Differenced (d=1) |
| SUNPHARMA | 0.8047 | No | Differenced (d=1) |
| DRREDDY | 0.6386 | No | Differenced (d=1) |
| ITC | 0.5971 | No | Differenced (d=1) |
| HINDUNILVR | 0.0475 | Yes | No differencing needed |
| M&M | 0.9845 | No | Differenced (d=1) |
| MARUTI | 0.9845 | No | Differenced (d=1) |

### Scaling
Min-Max normalization applied for LSTM model inputs (sequence length = 60 days).

### Train/Test Split
- **Training**: All data before July 2025 (1,110 observations)
- **Testing**: July - December 2025 (125 observations)

---

## 4. Forecast Results with Confidence Intervals (Task 3)

### Backtest Performance (Jul-Dec 2025 test set)

| Model | Avg MAPE | Best Stock | Worst Stock |
|-------|----------|-----------|-------------|
| **ARIMA** | **0.77%** | HDFCBANK (0.55%) | M&M (1.01%) |
| **LSTM** | **1.39%** | HDFCBANK (0.69%) | M&M (2.87%) |
| ExpSmoothing | 6.50% | HDFCBANK (2.47%) | MARUTI (14.84%) |
| Prophet | 7.64% | DRREDDY (2.73%) | MARUTI (17.33%) |

ARIMA was the best-performing model with sub-1% MAPE across all stocks. All models produced 2-day forecasts with 95% confidence intervals.

### Sample 2-Day Forecasts (ARIMA):

| Stock | Day 1 Forecast | Day 2 Forecast | 95% CI Width |
|-------|---------------|----------------|-------------|
| HDFCBANK | Rs.990.90 | Rs.990.90 | +/-20.0 |
| ICICIBANK | Rs.1,343.85 | Rs.1,346.29 | +/-48.1 |
| TCS | Rs.3,188.83 | Rs.3,188.83 | +/-140.5 |
| MARUTI | Rs.16,663.28 | Rs.16,670.25 | +/-580.0 |

---

## 5. Volatility and Trend Analysis (Task 4)

### GARCH(1,1) Volatility Estimates

| Stock | Persistence (alpha+beta) | Avg Daily Vol | Regime |
|-------|------------------------|-------------|--------|
| DRREDDY | 0.997 | 1.33% | High persistence |
| SUNPHARMA | 0.995 | 1.26% | High persistence |
| ICICIBANK | 0.994 | 1.29% | High persistence |
| MARUTI | 0.990 | 1.38% | High persistence |
| M&M | 0.976 | 1.73% | Moderate |
| TCS | 0.975 | 1.27% | Moderate |
| HDFCBANK | 0.970 | 1.26% | Moderate |
| ITC | 0.849 | 1.20% | Lower persistence |
| SBIN | 0.835 | 1.53% | Lower persistence |
| HINDUNILVR | 0.563 | 1.22% | Low persistence |
| INFY | 0.174 | 1.46% | Very low persistence |

### STL Trend Analysis

| Stock | Direction | Trend Slope (Quarterly) |
|-------|-----------|----------------------|
| SBIN | UPWARD | +15.76% |
| M&M | UPWARD | +9.25% |
| MARUTI | UPWARD | +9.08% |
| INFY | UPWARD | +7.98% |
| SUNPHARMA | UPWARD | +6.48% |
| TCS | UPWARD | +4.05% |
| HDFCBANK | UPWARD | +2.49% |
| ITC | SIDEWAYS | -1.56% |
| DRREDDY | SIDEWAYS | -1.88% |
| ICICIBANK | DOWNWARD | -2.89% |
| HINDUNILVR | DOWNWARD | -4.33% |

---

## 6. Portfolio Composition and Allocation Rationale (Task 5)

### Strategies Used (4 strategies combined):

| Strategy | Weight | Approach |
|----------|--------|----------|
| A: Forecast-Guided | 40% | Higher allocation to stocks with larger predicted returns |
| B: Volatility-Aware | 30% | Inverse volatility weighting (stable stocks get more) |
| C: Correlation-Diversified | 20% | Minimize inter-stock correlation |
| D: Sector Momentum | 10% | Overweight high-momentum sectors (Auto led at +25.6%) |

### Final Allocation:

| Stock | Weight | Amount (Rs.) | Primary Strategy Driver |
|-------|--------|-------------|----------------------|
| ICICI Bank | 13.5% | 1,34,614 | Strategy A (highest predicted return) |
| Maruti | 11.9% | 1,19,072 | Strategy D (auto momentum) |
| M&M | 11.0% | 1,10,239 | Strategy D (auto momentum) |
| HDFC Bank | 8.2% | 81,897 | Strategy B (low volatility) |
| Sun Pharma | 8.2% | 81,869 | Strategy B (low volatility) |
| Dr. Reddy's | 8.1% | 81,172 | Strategy C (low correlation) |
| HUL | 8.1% | 80,646 | Strategy B (lowest volatility) |
| ITC | 8.0% | 80,132 | Strategy B (low volatility) |
| TCS | 7.9% | 78,751 | Strategy C (diversification) |
| SBI | 7.7% | 76,708 | Strategy A (upward trend) |
| Infosys | 7.5% | 74,900 | Strategy C (diversification) |
| **TOTAL** | **100%** | **10,00,000** | |

---

## 7. Model Comparison and Accuracy Evaluation (Task 6)

### Comparative Evaluation

| Model | Avg MAPE | Avg RMSE | Avg Dir. Accuracy | Strengths | Weaknesses |
|-------|----------|---------|-------------------|-----------|------------|
| ARIMA | 0.77% | 33.7 | 50.6% | Best accuracy, interpretable, fast | Assumes linearity |
| LSTM | 1.39% | 62.3 | 47.9% | Captures non-linear patterns | Needs tuning, slow to train |
| ExpSmooth | 6.50% | 362.6 | 48.9% | Simple, handles seasonality | Poor on volatile stocks |
| Prophet | 7.64% | 406.4 | 52.9% | Best directional accuracy | Overestimates trends |

**Final model choice**: ARIMA was used for portfolio allocation decisions (Strategy A) due to its lowest MAPE. GARCH was used for volatility estimation (Strategy B).

---

## 8. StockGro Execution Summary (Task 7)

### Trading Window: May 12-13, 2026

All trades were placed on Day 1 (May 12) before market close and held through Day 2 (May 13).

| Stock | Qty | Avg Buy Price | Amount Invested |
|-------|-----|--------------|----------------|
| SBIN | 79 | Rs.964.41 | Rs.76,188 |
| Dr. Reddy's | 64 | Rs.1,267.02 | Rs.81,089 |
| HDFC Bank | 109 | Rs.745.88 | Rs.81,301 |
| HUL | 35 | Rs.2,246.96 | Rs.78,644 |
| ITC | 264 | Rs.302.94 | Rs.79,976 |
| Sun Pharma | 44 | Rs.1,864.93 | Rs.82,057 |
| Infosys | 64 | Rs.1,148.98 | Rs.73,535 |
| ICICI Bank | 106 | Rs.1,242.92 | Rs.131,750 |
| M&M | 34 | Rs.3,219.41 | Rs.109,460 |
| Maruti | 9 | Rs.13,186.23 | Rs.118,676 |
| TCS | 33 | Rs.2,337.20 | Rs.77,128 |
| **TOTAL** | | | **Rs.9,89,803** |

---

## 9. Predicted vs Actual Outcomes (Task 8)

### Comparison Table (ARIMA predictions vs StockGro actuals):

| Stock | Predicted D1 | Actual D1 | Error D1 | Predicted D2 | Actual D2 | Error D2 | Dir |
|-------|-------------|-----------|----------|-------------|-----------|----------|-----|
| SBIN | 973.52 | 961.70 | 1.23% | 974.10 | 956.52 | 1.84% | Wrong |
| DRREDDY | 1,265.61 | 1,257.58 | 0.64% | 1,265.61 | 1,254.76 | 0.86% | Correct |
| HDFCBANK | 990.90 | 739.79 | 33.94% | 990.90 | 743.37 | 33.30% | Wrong |
| HUL | 2,290.20 | 2,223.38 | 3.01% | 2,290.20 | 2,237.27 | 2.37% | Wrong |
| ITC | 392.56 | 298.29 | 31.60% | 392.75 | 302.30 | 29.92% | Correct |
| SUNPHARMA | 1,710.40 | 1,834.73 | 6.78% | 1,709.43 | 1,810.54 | 5.58% | Correct |
| INFY | 1,621.60 | 1,126.62 | 43.93% | 1,621.60 | 1,115.42 | 45.38% | Correct |
| ICICIBANK | 1,343.85 | 1,227.15 | 9.51% | 1,346.29 | 1,215.45 | 10.76% | Wrong |
| M&M | 3,662.70 | 3,154.30 | 16.12% | 3,665.10 | 3,091.93 | 18.54% | Wrong |
| MARUTI | 16,663.28 | 12,910.20 | 29.07% | 16,670.25 | 12,927.68 | 28.95% | Correct |
| TCS | 3,188.83 | 2,261.28 | 41.02% | 3,188.83 | 2,234.92 | 42.68% | Correct |

**Average live MAPE: 19.71% (Day 1), 20.02% (Day 2)**
**Directional Accuracy: 6/11 (54.5%)**

### Portfolio Performance
- **Invested**: Rs.9,89,803
- **Day 1 value**: Rs.9,73,644 (**-1.62%**)
- **Day 2 value**: Rs.9,69,134 (**-2.07%**)
- **Total P&L**: **-Rs.20,669**

### Analysis
Several stocks (HDFCBANK, ITC, INFY, TCS) show large prediction errors (30-45%). This is because our models were trained on data ending December 2025, but these stocks underwent significant price movements (stock splits, corrections, structural changes) between January-May 2026 that the models could not anticipate. The models predicted continuation of late-2025 price levels, while the market had already moved substantially by the May 2026 trading window.

Stocks with smaller prediction errors (SBIN at 1.23%, DRREDDY at 0.64%) were those whose May 2026 prices remained closest to their December 2025 levels.

---

## 10. Reflections

### What Worked
- **ARIMA** delivered excellent backtest performance (0.77% MAPE on the test set)
- **Multi-strategy portfolio blending** provided a diversified allocation
- **Data-driven stock selection** using quantitative screening criteria
- **Interactive Plotly Dash dashboard** with 5 pages for visual analysis

### What Didn't Work
- **Large gap between backtest and live performance**: Backtest MAPE was 0.77% but live MAPE was ~20%. This is expected — models extrapolated from Dec 2025 data to May 2026, a 5-month gap during which prices moved significantly.
- **No retraining**: Models should have been retrained on data up to May 2026 for live trading
- **All-long strategy**: During a broad market downturn, all stocks fell together (-2.07% portfolio loss)

### What I Would Improve
1. **Retrain models daily** on the latest available data before making predictions
2. **Include exogenous variables** (market indices, FII/DII flows, sector ETFs)
3. **Add regime detection** to distinguish bull/bear markets before allocation
4. **Consider hedging strategies** — use volatility predictions to size stop-losses
5. **Ensemble the models** by weighting predictions based on recent accuracy
