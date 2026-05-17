"""Generate the capstone report as a PDF using fpdf2, embedding plots for a better look and 10-page length."""
import os
from fpdf import FPDF

class Report(FPDF):
    def header(self):
        self.set_font('Helvetica', 'B', 9)
        self.set_text_color(100, 100, 100)
        self.cell(0, 6, 'Time Series Analysis 2026 - Capstone Report', align='R')
        self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font('Helvetica', '', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f'Page {self.page_no()}/{{nb}}', align='C')

    def section(self, title):
        self.set_font('Helvetica', 'B', 15)
        self.set_text_color(0, 0, 0)
        self.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(0, 180, 140)
        self.set_line_width(0.6)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(3)

    def subsection(self, title):
        self.set_font('Helvetica', 'B', 12)
        self.set_text_color(40, 40, 40)
        self.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def body(self, text):
        self.set_font('Helvetica', '', 11)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 6, text)
        self.ln(1)

    def bold_body(self, text):
        self.set_font('Helvetica', 'B', 11)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 6, text)
        self.ln(1)

    def bullet(self, text):
        self.set_font('Helvetica', '', 11)
        self.set_text_color(30, 30, 30)
        self.cell(5, 6, '-', new_x="END")
        self.multi_cell(175, 6, text)

    def table(self, headers, rows, col_widths=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)
        # Header
        self.set_font('Helvetica', 'B', 9)
        self.set_fill_color(45, 55, 72)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align='C')
        self.ln()
        # Rows
        self.set_font('Helvetica', '', 9)
        self.set_text_color(30, 30, 30)
        for j, row in enumerate(rows):
            if j % 2 == 0:
                self.set_fill_color(245, 245, 250)
            else:
                self.set_fill_color(255, 255, 255)
            for i, val in enumerate(row):
                self.cell(col_widths[i], 6.5, str(val), border=1, fill=True, align='C')
            self.ln()
        self.ln(2)

    def add_image(self, img_path, w=160, h=None):
        if os.path.exists(img_path):
            self.ln(2)
            # Center the image
            x = (210 - w) / 2
            if h is None:
                self.image(img_path, x=x, w=w)
            else:
                self.image(img_path, x=x, w=w, h=h)
            self.ln(5)

def build():
    pdf = Report()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    # ========================== PAGE 1 ==========================
    pdf.set_font('Helvetica', 'B', 24)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 15, 'Time Series Analysis 2026', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.set_font('Helvetica', '', 16)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(0, 10, 'Capstone Report  |  IIT Guwahati x StockGro', align='C', new_x="LMARGIN", new_y="NEXT")
    pdf.ln(10)

    pdf.section('1. Methodology and Models Used')
    pdf.body('We applied time series forecasting to 11 NSE stocks, allocated Rs.10,00,000 in a virtual portfolio on StockGro, and compared predictions against actual market outcomes. Our models include ARIMA, Prophet, LSTM, Exponential Smoothing, and GARCH.')
    pdf.body('Dataset: Yahoo Finance (yfinance), Jan 2021 - Dec 2025, daily.')
    pdf.body('Split: Training (< Jul 2025) | Testing (Jul-Dec 2025, 125 observations).')
    
    pdf.table(
        ['Model', 'Library', 'Purpose'],
        [
            ['ARIMA/SARIMA', 'pmdarima', 'Price forecasting (auto order via AIC)'],
            ['Facebook Prophet', 'prophet', 'Trend + seasonality with Indian holidays'],
            ['LSTM (2-layer)', 'PyTorch', 'Non-linear deep learning sequence model'],
            ['Exp. Smoothing', 'statsmodels', 'Holt-Winters triple exponential smoothing'],
            ['GARCH(1,1)', 'arch', 'Conditional volatility estimation'],
        ],
        [40, 30, 120]
    )

    pdf.add_image('outputs/figures/stock_prices.png', w=180)

    # ========================== PAGE 2 ==========================
    pdf.add_page()
    pdf.section('2. Stock Selection Rationale (Task 1)')
    pdf.body('We screened 25 candidate stocks across 5 sectors using three quantitative criteria: (1) 30-day rolling std dev, (2) STL trend strength, (3) 6-month sector momentum. Selected 11 stocks:')
    pdf.table(
        ['Sector', 'Selected Stocks', 'Key Justification'],
        [
            ['Banking', 'HDFC Bank, ICICI Bank, SBI', 'Highest liquidity (15-25M daily vol)'],
            ['IT', 'TCS, Infosys', 'Top-2 by market cap, mean-reversion'],
            ['Pharma', 'Sun Pharma, Dr. Reddy\'s', 'Low correlation, defensive sector'],
            ['FMCG', 'ITC, Hindustan Unilever', 'Lowest volatility (4th-8th pctl)'],
            ['Auto', 'M&M, Maruti', 'Highest sector momentum (+25.6%)'],
        ],
        [25, 65, 100]
    )
    pdf.body('14 candidates were rejected due to lower liquidity, weaker trends, or redundancy within their respective sectors.')
    
    pdf.add_image('outputs/figures/task1_screening.png', w=170)

    # ========================== PAGE 3 ==========================
    pdf.add_page()
    pdf.section('3. Data Preprocessing Steps (Task 2)')
    pdf.bullet('Missing values: Forward-fill for market holidays and trading halts.')
    pdf.bullet('Stationarity: Augmented Dickey-Fuller (ADF) test applied; 10 out of 11 stocks required first-order differencing (d=1).')
    pdf.bullet('Scaling: Min-Max normalization for LSTM inputs (60-day lookback window).')
    pdf.bullet('Log returns: Computed for GARCH volatility modeling.')
    pdf.bullet('STL Decomposition: Period=63 for trend/seasonal/residual separation.')
    pdf.bullet('Split: Training <Jul 2025 (~1110 obs) | Test Jul-Dec 2025 (125 obs).')
    
    pdf.add_image('outputs/figures/decomposition_HDFCBANK.png', w=160)
    pdf.body('Above: STL Decomposition example showing strong trend extraction for HDFC Bank.')

    # ========================== PAGE 4 ==========================
    pdf.add_page()
    pdf.section('4. Forecast Results with Confidence Intervals (Task 3)')
    pdf.body('Four models were trained on each stock, validated on 125 test points, and used to forecast 2 days ahead with 95% confidence intervals:')
    
    pdf.table(
        ['Model', 'Avg MAPE', 'Best Stock', 'Worst Stock', 'Avg Dir. Acc.'],
        [
            ['ARIMA', '0.77%', 'HDFCBANK (0.55%)', 'M&M (1.01%)', '50.6%'],
            ['LSTM', '1.39%', 'HDFCBANK (0.69%)', 'M&M (2.87%)', '47.9%'],
            ['ExpSmoothing', '6.50%', 'HDFCBANK (2.47%)', 'MARUTI (14.84%)', '48.9%'],
            ['Prophet', '7.64%', 'DRREDDY (2.73%)', 'MARUTI (17.33%)', '52.9%'],
        ],
        [30, 25, 45, 50, 30]
    )
    pdf.body('ARIMA achieved the best accuracy with sub-1% average MAPE across all 11 stocks.')

    pdf.add_image('outputs/figures/forecast_HDFCBANK_ARIMA.png', w=150)
    pdf.add_image('outputs/figures/forecast_TCS_ARIMA.png', w=150)

    # ========================== PAGE 5 ==========================
    pdf.add_page()
    pdf.section('5. Volatility and Trend Analysis (Task 4)')
    pdf.subsection('GARCH(1,1) Volatility')
    pdf.body('High-persistence stocks (DRREDDY: 0.997, SUNPHARMA: 0.995) show slowly-decaying volatility clustering. Low-persistence stocks (INFY: 0.174, HINDUNILVR: 0.563) have mean-reverting volatility.')
    
    pdf.add_image('outputs/figures/volatility.png', w=170)

    pdf.subsection('STL Trend Analysis')
    pdf.body('7 stocks upward (SBIN +15.76%, M&M +9.25%, MARUTI +9.08%), 2 sideways (ITC, DRREDDY), 2 downward (ICICIBANK -2.89%, HUL -4.33%).')

    # ========================== PAGE 6 ==========================
    pdf.add_page()
    pdf.section('6. Portfolio Composition and Allocation (Task 5)')
    pdf.body('Blended 4 strategies: Forecast-Guided (40%), Volatility-Aware (30%), Correlation-Diversified (20%), Sector Momentum (10%).')
    
    pdf.table(
        ['Stock', 'Weight', 'Amount (Rs.)', 'Primary Driver'],
        [
            ['ICICI Bank', '13.5%', '1,34,614', 'Forecast-Guided'],
            ['Maruti', '11.9%', '1,19,072', 'Sector Momentum'],
            ['M&M', '11.0%', '1,10,239', 'Sector Momentum'],
            ['HDFC Bank', '8.2%', '81,897', 'Volatility-Aware'],
            ['Sun Pharma', '8.2%', '81,869', 'Volatility-Aware'],
            ['Dr. Reddy\'s', '8.1%', '81,172', 'Correlation-Diversified'],
            ['HUL', '8.1%', '80,646', 'Volatility-Aware'],
            ['ITC', '8.0%', '80,132', 'Volatility-Aware'],
            ['TCS', '7.9%', '78,751', 'Correlation-Diversified'],
            ['SBI', '7.7%', '76,708', 'Forecast-Guided'],
            ['Infosys', '7.5%', '74,900', 'Correlation-Diversified'],
            ['TOTAL', '100%', '10,00,000', ''],
        ],
        [30, 20, 30, 110]
    )

    pdf.add_image('outputs/figures/portfolio_allocation.png', w=120)

    # ========================== PAGE 7 ==========================
    pdf.add_page()
    pdf.body('To ensure a well-diversified portfolio, we evaluated the correlation matrix of daily log returns among the selected stocks (Strategy C).')
    pdf.add_image('outputs/figures/correlation_heatmap.png', w=160)

    pdf.section('7. Model Comparison (Task 6)')
    pdf.table(
        ['Model', 'Avg MAPE', 'Avg RMSE', 'Dir. Acc.', 'Assessment'],
        [
            ['ARIMA', '0.77%', '33.7', '50.6%', 'Best accuracy; interpretable'],
            ['LSTM', '1.39%', '62.3', '47.9%', 'Good non-linear; slow'],
            ['ExpSmooth', '6.50%', '362.6', '48.9%', 'Simple; poor on volatile'],
            ['Prophet', '7.64%', '406.4', '52.9%', 'Best dir. acc.; overestimates'],
        ],
        [25, 22, 22, 20, 101]
    )
    pdf.body('ARIMA was used for portfolio allocation (Strategy A) due to having the lowest MAPE. GARCH was used for volatility estimation (Strategy B).')
    pdf.add_image('outputs/figures/model_comparison.png', w=140)

    # ========================== PAGE 8 ==========================
    pdf.add_page()
    pdf.section('8. StockGro Execution Summary (Task 7)')
    pdf.body('Trading Window: Day 1 = May 12, 2026 | Day 2 = May 13, 2026.')
    pdf.body('All 11 stocks were purchased on Day 1 at market open. The portfolio was held without changes through Day 2. Total invested amount: Rs.9,89,803.')
    
    pdf.add_image('results/Screenshot_12-5-2026_161413_app.stockgro.club.jpeg', w=150)
    pdf.add_image('results/Screenshot_12-5-2026_161433_app.stockgro.club.jpeg', w=150)
    
    # ========================== PAGE 9 ==========================
    pdf.add_page()
    pdf.section('9. Predicted vs Actual Outcomes (Task 8)')
    pdf.table(
        ['Stock', 'Pred D1', 'Act D1', 'Err D1', 'Pred D2', 'Act D2', 'Err D2'],
        [
            ['SBIN', '973.52', '961.70', '1.23%', '974.10', '956.52', '1.84%'],
            ['DRREDDY', '1265.61', '1257.58', '0.64%', '1265.61', '1254.76', '0.86%'],
            ['HDFCBANK', '990.90', '739.79', '33.94%', '990.90', '743.37', '33.30%'],
            ['HUL', '2290.20', '2223.38', '3.01%', '2290.20', '2237.27', '2.37%'],
            ['ITC', '392.56', '298.29', '31.60%', '392.75', '302.30', '29.92%'],
            ['SUNPHARMA', '1710.40', '1834.73', '6.78%', '1709.43', '1810.54', '5.58%'],
            ['INFY', '1621.60', '1126.62', '43.93%', '1621.60', '1115.42', '45.38%'],
            ['ICICIBANK', '1343.85', '1227.15', '9.51%', '1346.29', '1215.45', '10.76%'],
            ['M&M', '3662.70', '3154.30', '16.12%', '3665.10', '3091.93', '18.54%'],
            ['MARUTI', '16663.28', '12910.20', '29.07%', '16670.25', '12927.68', '28.95%'],
            ['TCS', '3188.83', '2261.28', '41.02%', '3188.83', '2234.92', '42.68%'],
            ['AVG MAPE', '', '', '19.71%', '', '', '20.02%'],
        ],
        [24, 24, 24, 22, 24, 24, 22]
    )
    pdf.bold_body('Portfolio: Invested Rs.9,89,803 | Day 2 value Rs.9,69,134 | Return: -2.07% (loss Rs.20,669)')
    pdf.body('Directional Accuracy: 6/11 (54.5%)')
    
    pdf.add_image('results/Screenshot_13-5-2026_174457_app.stockgro.club.jpeg', w=130)

    # ========================== PAGE 10 ==========================
    pdf.add_page()
    pdf.section('10. Reflections & Analysis')
    
    pdf.body('Analysis of Errors: Stocks like HDFCBANK, ITC, INFY, TCS show 30-45% errors due to likely corporate actions (stock splits/bonus issues) between Jan-May 2026. Stocks near Dec 2025 levels (SBIN: 1.23%, DRREDDY: 0.64%) had excellent predictions.')
    pdf.ln(5)

    pdf.subsection('What Worked')
    pdf.bullet('ARIMA delivered excellent backtest accuracy (0.77% avg MAPE)')
    pdf.bullet('Data-driven stock selection using quantitative screening of 25 candidates')
    pdf.bullet('Multi-strategy portfolio blending provided diversified allocation')
    pdf.bullet('GARCH correctly identified high vs low-risk stocks')
    pdf.ln(3)

    pdf.subsection("What Didn't Work")
    pdf.bullet('Backtest vs live gap: 0.77% backtest MAPE vs 20% live MAPE due to the 5-month extrapolation gap.')
    pdf.bullet('Corporate actions in HDFCBANK, ITC, INFY changed price levels drastically.')
    pdf.bullet('All-long strategy: broad market downturn dragged all stocks down.')
    pdf.ln(3)

    pdf.subsection('What I Would Improve')
    pdf.bullet('1. Retrain models daily on latest data before live predictions')
    pdf.bullet('2. Include exogenous variables (Nifty 50, FII/DII flows, USD/INR)')
    pdf.bullet('3. Add regime detection for bull/bear market switching')
    pdf.bullet('4. Ensemble models weighted by recent per-stock accuracy')
    pdf.bullet('5. Stop-loss mechanisms based on GARCH volatility forecasts')

    pdf.output('capstone_report.pdf')
    print('Report saved to capstone_report.pdf')


if __name__ == '__main__':
    build()
