# Stock Screening & Forecasting Dashboard

An interactive stock screening and price forecasting dashboard built in Python for JupyterLab. Screens the full S&P 500 (or a custom ticker list) using technical analysis and valuation metrics, then forecasts prices using a three-model ensemble — SARIMAX, Prophet, and XGBoost — with walk-forward cross-validation ensuring every model is tested on genuinely unseen data before its predictions are trusted.

Built as part of MSc Financial Technology research at Bristol Business School, UWE (2025–2026).

---

## Overview

Most stock forecasting implementations share a fundamental flaw: they fit models to the full dataset and report in-sample accuracy, which tells you nothing about whether the model actually works. This dashboard addresses that directly through rigorous walk-forward cross-validation — the same methodology used in professional quantitative research.

Three core problems this tool solves:

1. **Screening at scale** — manually reviewing 500 stocks for technical signals and valuation is impractical. This tool automates the full pipeline, surfacing only the candidates worth investigating further.
2. **Blind model fitting** — fitting SARIMAX, Prophet, or XGBoost to full price histories inflates apparent accuracy. Walk-forward CV tests each model on genuinely unseen historical windows before selecting hyperparameters or assigning ensemble weights.
3. **Single-model dependency** — no single forecasting model reliably outperforms across all stocks and regimes. The weighted ensemble assigns more influence to models that performed better in cross-validation, dynamically adapting to each stock.

---

## Key Features

### Data Collection
- Downloads historical price data for all 500 S&P 500 constituents or a custom ticker list via yfinance
- Fetches live P/E ratios for valuation screening
- S&P 500 ticker list cached locally to avoid repeated Wikipedia fetches
- Handles missing data, delisted tickers, and mixed universes gracefully
- Automatic dependency installation with graceful fallback per model

### Technical Analysis
- Fast and slow moving averages: SMA or EMA, fully configurable window lengths
- Crossover signal detection: identifies bullish (fast crosses above slow) and bearish (fast crosses below slow) signals
- Current moving average state reported per stock: fast above slow, fast below slow, or neutral

### Valuation Screening
- Absolute P/E threshold screening (e.g. flag stocks with P/E below 15)
- Relative screening: flags stocks trading below a user-defined discount to the median P/E across the full run universe
- Both screens can be used independently or combined
- Undervalued stocks automatically sorted to the top of results

### Forecasting Models

**SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables)**
- Classical statistical time series model fitted on log-transformed prices
- Walk-forward CV over expanding windows to select optimal (p, d, q) and seasonal parameters
- Hyperparameter grid search capped by configurable trial budget
- Captures trend, autocorrelation, and seasonal structure in price history

**Prophet (Meta's trend-decomposition model)**
- Decomposes price series into trend, weekly seasonality, and yearly seasonality components
- Walk-forward CV to tune changepoint prior scale and seasonality prior scale
- Robust to missing data and structural breaks in price history
- Log-transformation applied for stability

**XGBoost (Gradient boosted trees)**
- Machine learning model trained on engineered price features: lagged returns, rolling statistics, RSI(14), momentum indicators
- Recursive multi-step forecasting — each predicted value fed back as a feature for the next step
- Walk-forward CV with time-series-aware splits (no look-ahead contamination)
- Captures non-linear relationships that statistical models miss

### Weighted Ensemble
- Each model's ensemble weight derived from its walk-forward CV RMSE
- Lower CV RMSE → higher ensemble weight automatically
- Final forecast is a weighted combination of all available models
- Gracefully handles missing models — ensemble adapts to however many models are available

### Walk-Forward Cross-Validation
- Expanding-window CV across configurable number of folds
- Each fold trains on all past data and tests on the immediately following unseen window
- Prevents look-ahead bias — the most common source of inflated accuracy in financial forecasting
- CV RMSE and MAPE reported per model per stock in the results table
- Best hyperparameter configuration selected based on mean CV RMSE across all folds

### Interactive Results Dashboard
- Sortable results table: ticker, P/E, undervaluation flag, last close, MA state, crossover signal, best model, CV RMSE, CV MAPE
- Stock selector dropdown to pick any screened stock
- Plot Forecast button: renders price history + moving averages + ensemble forecast (or individual model)
- Model Scores button: displays per-model CV scores, tuned hyperparameters, and ensemble weight allocation
- Filter modes: show all, undervalued only, cross-up only, best RMSE top quartile
- Results export to CSV
- In-session model caching — re-plotting or changing filter does not re-run CV

---

## Dashboard Structure

The UI is built entirely in ipywidgets within JupyterLab.

| Control | Options |
|---|---|
| Universe | Full S&P 500 or custom ticker list |
| Ticker limit | How many tickers to run (0 = no limit) |
| Lookback | Historical data window in days |
| MA type | EMA or SMA |
| MA fast / slow | Configurable window lengths |
| Forecast horizon | Days ahead to forecast |
| Fold test size | Test window per CV fold (days) |
| CV folds | Number of walk-forward folds |
| Max trials | Hyperparameter candidates to try per model |
| Seasonal SARIMAX | Add seasonal (1,0,1,5) candidate to SARIMAX grid |
| P/E max | Absolute undervaluation threshold |
| Relative P/E | Enable relative P/E screen |
| Relative discount | Discount to median P/E for relative screen |
| Filter | All / Undervalued / Cross-up / Best RMSE quartile |
| Plot | Ensemble / SARIMAX / Prophet / XGBoost / All overlay |

---

## Screenshots

### Full Control Panel
The complete widget UI showing universe selection, MA configuration, forecast CV settings, valuation screening, and progress tracking.

![Dashboard Control Panel](screenshots/Screenshot%202026-03-20%20at%2016.12.15.png)

---

### Screening Results Table
Live results across 25 S&P 500 stocks showing P/E ratios, undervaluation flags, MA state, crossover signals, best model selection, CV RMSE, and CV MAPE. Undervalued stocks sorted to the top — AES (P/E 10.79), ALL (P/E 5.42), and MO (P/E 15.66) flagged as undervalued by both absolute and relative P/E screens.

![Screening Results Table](screenshots/Screenshot%202026-03-20%20at%2016.12.19.png)

---

### Ensemble Forecast Chart — ADBE
Price history with EMA(9) and EMA(12) moving averages alongside the 30-day ensemble forecast. Ensemble weights: SARIMAX 50%, Prophet 50%. Chart shows the downtrend in ADBE from its 2024 peak with near-term forecast continuation.

![Ensemble Forecast Chart](screenshots/Screenshot%202026-03-20%20at%2016.13.51.png)

---

### Model Scores & Ensemble Weights — ADBE
Per-model CV RMSE and MAPE scores alongside tuned hyperparameter selections and final ensemble weight allocation. SARIMAX CV RMSE 26.14 (weight 0.496), Prophet CV RMSE 25.69 (weight 0.504). Full tuning picks displayed for each model.

![Model Scores and Ensemble Weights](screenshots/Screenshot%202026-03-20%20at%2016.14.04.png)

---

## Technical Stack

| Component | Library |
|---|---|
| Data download | yfinance |
| Statistical forecasting | statsmodels (SARIMAX) |
| Trend decomposition | prophet |
| Machine learning | xgboost |
| Feature engineering | pandas, numpy |
| Visualisation | matplotlib |
| Interactive UI | ipywidgets |

---

## Installation & Usage

### Requirements
```
numpy>=1.24.0
pandas>=2.0.0
yfinance>=0.2.36
matplotlib>=3.7.0
ipywidgets>=8.0.0
statsmodels>=0.14.0
prophet>=1.1.5
xgboost>=2.0.0
scikit-learn>=1.3.0
jupyterlab>=4.0.0
lxml>=4.9.0
```

Install all dependencies:
```bash
pip install numpy pandas yfinance matplotlib ipywidgets statsmodels prophet xgboost scikit-learn jupyterlab lxml
```

### Running the Dashboard
1. Clone or download this repository
2. Open `stock_screening_forecasting.ipynb` in JupyterLab
3. Run the single cell (Shift+Enter)
4. Configure inputs using the widget UI
5. Click **"Run Analysis"**

> **Note:** Running the full S&P 500 with three models and walk-forward CV is computationally intensive. For faster results during exploration, start with a custom list of 10–20 stocks. Full S&P 500 runs are best left overnight.

> **Prophet installation:** Prophet can be finicky to install on some systems. If it fails, the dashboard will run with SARIMAX and XGBoost only — the ensemble adapts automatically.

---

## Methodology Notes

### Why Walk-Forward Cross-Validation?
Standard train/test splits on time series data are insufficient for financial forecasting because they test the model on only one historical window. Walk-forward CV tests the model across multiple sequential windows — each time training only on past data and predicting the immediately following unseen period. This directly simulates real-world conditions where you train on history and forecast the future. Models that perform well in walk-forward CV are genuinely predictive; models that only perform well in-sample are overfit.

### Why Three Models?
SARIMAX excels at capturing autocorrelation and seasonality in stationary series. Prophet handles trend breaks and missing data well but can overfit to noise. XGBoost captures non-linear feature interactions that statistical models miss. No single model consistently dominates across all stocks and time periods — the ensemble combines their complementary strengths while the CV-derived weighting ensures the best-performing model for each specific stock receives the most influence.

### Why P/E Screening?
Price forecasting in isolation ignores valuation. A stock can be technically bullish and still be expensive relative to earnings. Combining momentum signals with valuation screening surfaces a higher-quality candidate set than either approach alone. The relative P/E screen is particularly useful — it adapts to the current market environment rather than applying a fixed absolute threshold regardless of prevailing valuations.

### Limitations
- **Data source:** yfinance provides adjusted prices suitable for research. Licensed data required for production use.
- **Forecasting horizon:** Short-term price forecasting is inherently noisy. CV scores indicate relative model quality, not absolute predictive certainty.
- **P/E data:** P/E ratios from yfinance may lag or be unavailable for some tickers. Always verify against primary sources.
- **Computational cost:** Full S&P 500 runs with walk-forward CV across three models are slow. Parallelisation is not currently implemented.
- **Recursive XGBoost:** Multi-step recursive forecasting compounds errors over longer horizons. Treat longer-horizon XGBoost forecasts with additional caution.

---

## Academic Context

This tool was developed as part of MSc Financial Technology research at Bristol Business School, University of the West of England (2025–2026), alongside the portfolio optimisation dashboard in the companion repository [`portfolio-optimisation-dashboard`](https://github.com/ThomasOxley/portfolio-optimisation-dashboard).

---

## Disclaimer

This tool is an academic research project. All outputs are for educational and illustrative purposes only. Nothing in this repository constitutes financial advice or a recommendation to buy or sell any security.

---

*Thomas Oxley | MSc Financial Technology | Bristol Business School, UWE*
*linkedin.com/in/thomas-oxley-868047174 | github.com/ThomasOxley*

---
