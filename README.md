# Stock Screening & Forecasting Dashboard

An interactive stock screening and price forecasting dashboard built in Python for JupyterLab. Screens the full S&P 500 (or a custom ticker list) using technical analysis and valuation metrics, then forecasts prices using a three-model ensemble — SARIMAX, Prophet, and XGBoost — with walk-forward cross-validation ensuring every model is tested on genuinely unseen data before its predictions are trusted.

Built as part of MSc Financial Technology research at Bristol Business School, UWE (2024–2025).

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
- Handles mixed universes, missing data, and delisted tickers gracefully

### Technical Analysis
- Fast and slow moving averages: SMA or EMA, fully configurable window lengths
- Crossover signal detection: identifies bullish (fast crosses above slow) and bearish (fast crosses below slow) signals
- Current moving average state reported per stock

### Valuation Screening
- Absolute P/E threshold screening (e.g. flag stocks with P/E below 15)
- Relative screening: flags stocks trading below a user-defined discount to the median P/E across the full run universe
- Combines technical and valuation signals into a unified screening output

### Forecasting Models

**SARIMAX (Seasonal AutoRegressive Integrated Moving Average with eXogenous variables)**
- Classical statistical time series model
- Walk-forward CV over rolling windows to select optimal (p, d, q) and seasonal parameters
- Captures trend, seasonality, and autocorrelation structure in price history

**Prophet (Meta's trend-decomposition model)**
- Decomposes price series into trend, seasonality, and holiday components
- Walk-forward CV to tune changepoint sensitivity and seasonality parameters
- Robust to missing data and structural breaks in price history

**XGBoost (Gradient boosted trees)**
- Machine learning model trained on engineered price features: lagged returns, rolling statistics, momentum indicators
- Walk-forward CV with time-series-aware splits (no look-ahead contamination)
- Captures non-linear relationships that statistical models miss

### Weighted Ensemble
- Each model's ensemble weight is derived from its walk-forward CV accuracy
- Better-performing models automatically receive more influence in the final forecast
- Final ensemble forecast combines all three models proportionally to their validated performance

### Walk-Forward Cross-Validation
- Expanding or rolling window CV across multiple historical folds
- Each fold trains on past data only and tests on the immediately following unseen window
- Prevents look-ahead bias — the most common source of inflated accuracy in financial forecasting
- CV scores reported per model per stock in the results table

### Interactive Results Dashboard
- Sortable results table: P/E ratio, moving average state, crossover signal, CV accuracy scores per model, ensemble weights
- Stock selector: click any screened stock to view its forecast chart
- Individual model forecast plots with confidence intervals
- Ensemble forecast overlay
- Model comparison: per-stock accuracy scores and weight allocation

---

## Dashboard Structure

The UI is built entirely in ipywidgets within JupyterLab.

**Inputs:**
- Ticker source: full S&P 500 or custom list
- Date range for historical data
