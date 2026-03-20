#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ================================================================================
# Stock Screening & Forecasting Dashboard
# ================================================================================
# Author:      Thomas Oxley
# Institution: Bristol Business School, UWE — MSc Financial Technology
# Date:        2024–2025
# Version:     1.0
# 
# Description:
#    An interactive stock screening and price forecasting dashboard built for
#    JupyterLab. Screens the full S&P 500 (or a custom ticker list) using
#    technical analysis and valuation metrics, then forecasts prices using a
#    three-model ensemble — SARIMAX, Prophet, and XGBoost — with walk-forward
#    cross-validation ensuring every model is tested on genuinely unseen data
#    before its predictions are trusted.
#
# Key Features:
#    - Screens full S&P 500 or custom ticker universe via yfinance
#    - Technical analysis: SMA/EMA crossover signal detection
#    - Valuation screening: absolute and relative P/E ratio filtering
#    - SARIMAX: classical time series forecasting with hyperparameter search
#    - Prophet: Meta's trend-decomposition model with CV-tuned parameters
#    - XGBoost: gradient boosted trees with engineered price features
#    - Walk-forward cross-validation: expanding window, multiple folds,
#      no look-ahead bias — tests each model on genuinely unseen data
#    - Weighted ensemble: CV RMSE-derived weights, better models get
#      more influence automatically
#    - Interactive UI: ipywidgets dashboard with results table,
#      forecast charts, and model diagnostics
#    - Results export to CSV
#    - In-session model caching for fast re-runs

# Dependencies:
#    See requirements.txt
# 
# Usage:
#    Run the single cell in JupyterLab. Configure inputs using the widget UI,
#    then click "Run Analysis".
#
# Note:
#    This is an academic research tool. Outputs are for educational and
#    illustrative purposes and do not constitute financial advice.
# ================================================================================


Add stock screening and forecasting dashboard with walk-forward CV ensemble
import sys, subprocess, time, warnings
warnings.filterwarnings("ignore")

def _pip_install(pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkgs)

# --- Ensure deps (graceful fallback) ---
missing = []
try:
    import numpy as np
    import pandas as pd
    import yfinance as yf
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    import urllib.request
except Exception:
    missing += ["numpy", "pandas", "yfinance", "matplotlib", "ipywidgets", "lxml"]

HAVE_STATSMODELS, HAVE_PROPHET, HAVE_XGBOOST = True, True, True

try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:
    HAVE_STATSMODELS = False
    missing += ["statsmodels"]

try:
    from prophet import Prophet
except Exception:
    HAVE_PROPHET = False

try:
    import xgboost as xgb
except Exception:
    HAVE_XGBOOST = False
    missing += ["xgboost"]

if missing:
    _pip_install(missing)
    import numpy as np
    import pandas as pd
    import yfinance as yf
    import matplotlib.pyplot as plt
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    import urllib.request
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        HAVE_STATSMODELS = True
    except Exception:
        HAVE_STATSMODELS = False
    try:
        import xgboost as xgb
        HAVE_XGBOOST = True
    except Exception:
        HAVE_XGBOOST = False

# Prophet is often finicky; try install separately
if not HAVE_PROPHET:
    try:
        _pip_install(["prophet"])
        from prophet import Prophet
        HAVE_PROPHET = True
    except Exception:
        HAVE_PROPHET = False

plt.rcParams["figure.figsize"] = (12, 5)

# -----------------------------
# Universe
# -----------------------------
WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def normalize_tickers(tickers):
    out = []
    for t in tickers:
        if t is None: 
            continue
        s = str(t).strip().upper()
        if not s:
            continue
        out.append(s.replace(".", "-"))
    seen, uniq = set(), []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq

def _fetch_html_with_user_agent(url: str) -> bytes:
    req = urllib.request.Request(
        url,
        headers={"User-Agent":"Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari"}
    )
    with urllib.request.urlopen(req) as resp:
        return resp.read()

def load_sp500_tickers_cached(cache_path="sp500_tickers.csv"):
    import os
    if os.path.exists(cache_path):
        s = pd.read_csv(cache_path, header=None)[0].astype(str).tolist()
        s = [x.strip().upper().replace(".", "-") for x in s if str(x).strip()]
        return normalize_tickers(s)
    html = _fetch_html_with_user_agent(WIKI_SP500_URL)
    tables = pd.read_html(html)
    df = tables[0]
    tickers = df["Symbol"].astype(str).str.strip().tolist()
    tickers = [t.replace(".", "-") for t in tickers]
    tickers = normalize_tickers(tickers)
    pd.Series(tickers).to_csv(cache_path, index=False, header=False)
    return tickers

# -----------------------------
# Data + indicators
# -----------------------------
def batch_download_prices(tickers, lookback_days, interval):
    if not tickers:
        return {}
    raw = yf.download(
        tickers,
        period=f"{int(lookback_days)}d",
        interval=str(interval),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    out = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            if t not in raw.columns.get_level_values(0):
                continue
            df = raw[t].copy()
            df.columns = [c.title() for c in df.columns]
            if "Close" in df.columns and not df["Close"].dropna().empty:
                out[t] = df.dropna(how="all")
        return out
    df = raw.copy()
    df.columns = [c.title() for c in df.columns]
    if "Close" in df.columns and not df["Close"].dropna().empty:
        out[tickers[0]] = df.dropna(how="all")
    return out

def moving_average(series: pd.Series, period: int, use_ema: bool) -> pd.Series:
    if use_ema:
        return series.ewm(span=period, adjust=False).mean()
    return series.rolling(window=period, min_periods=period).mean()

def crossover_signal(fast: pd.Series, slow: pd.Series) -> str:
    df = pd.DataFrame({"fast": fast, "slow": slow}).dropna()
    if df.shape[0] < 3:
        return "no_data"
    prev, last = df.iloc[-2], df.iloc[-1]
    if prev["fast"] <= prev["slow"] and last["fast"] > last["slow"]:
        return "cross_up"
    if prev["fast"] >= prev["slow"] and last["fast"] < last["slow"]:
        return "cross_down"
    return "no_cross"

def get_trailing_pe(ticker: str):
    try:
        info = yf.Ticker(ticker).info
        pe = info.get("trailingPE", None)
        if pe is None:
            return None
        pe = float(pe)
        if not np.isfinite(pe) or pe <= 0:
            return None
        return pe
    except Exception:
        return None

# -----------------------------
# Metrics + Walk-forward CV
# -----------------------------
def _rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def _mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true), 1e-9)
    return float(np.mean(np.abs((y_true - y_pred) / denom)))

def make_walkforward_splits(series: pd.Series, test_size: int, folds: int, min_train: int = 252):
    """
    Expanding-window walk-forward splits.
    Returns list of (train_series, test_series) from oldest to newest.
    """
    s = series.dropna().astype(float)
    if s.shape[0] < min_train + test_size * folds:
        return []
    splits = []
    end = s.shape[0]
    # last fold tests the last test_size points
    for k in range(folds, 0, -1):
        test_end = end - (k - 1) * test_size
        test_start = test_end - test_size
        train = s.iloc[:test_start]
        test = s.iloc[test_start:test_end]
        if train.shape[0] >= min_train and test.shape[0] == test_size:
            splits.append((train, test))
    return splits

# -----------------------------
# Model: SARIMAX (log)
# -----------------------------
def sarimax_fit_predict(train: pd.Series, horizon: int, order, seasonal_order=(0,0,0,0)):
    y = train[train > 0].astype(float)
    ly = np.log(y)
    model = SARIMAX(
        ly,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    res = model.fit(disp=False)
    fc = res.get_forecast(steps=horizon)
    mean = np.exp(fc.predicted_mean.values)
    ci = fc.conf_int()
    lo = np.exp(ci.iloc[:, 0].values)
    hi = np.exp(ci.iloc[:, 1].values)
    return mean, lo, hi

def sarimax_search_cv(close: pd.Series, test_size: int, folds: int, seasonal: bool, max_trials: int):
    if not HAVE_STATSMODELS:
        return {"ok": False, "name":"SARIMAX", "err":"statsmodels unavailable"}

    splits = make_walkforward_splits(close, test_size, folds)
    if not splits:
        return {"ok": False, "name":"SARIMAX", "err":"Not enough data for CV"}

    # Small grid; capped by max_trials
    orders = [(0,1,1), (1,1,0), (1,1,1), (2,1,1)]
    seasonals = [(0,0,0,0)]
    if seasonal:
        # weekly-ish (5 business days)
        seasonals = [(0,0,0,0), (1,0,1,5)]

    candidates = []
    for o in orders:
        for so in seasonals:
            candidates.append((o, so))
    candidates = candidates[:max_trials]

    best = None
    for (o, so) in candidates:
        rmses, mapes = [], []
        ok = True
        for train, test in splits:
            try:
                pred, _, _ = sarimax_fit_predict(train, len(test), o, so)
                rmses.append(_rmse(test.values, pred))
                mapes.append(_mape(test.values, pred))
            except Exception:
                ok = False
                break
        if not ok:
            continue
        score = float(np.mean(rmses))
        rec = {"order": o, "seasonal_order": so, "rmse": score, "mape": float(np.mean(mapes))}
        if best is None or rec["rmse"] < best["rmse"]:
            best = rec

    if best is None:
        return {"ok": False, "name":"SARIMAX", "err":"All SARIMAX candidates failed"}

    return {"ok": True, "name":"SARIMAX", "best": best}

# -----------------------------
# Model: Prophet (log)
# -----------------------------
def prophet_fit_predict(train: pd.Series, horizon: int, cps: float, sps: float):
    df = pd.DataFrame({"ds": train.index.tz_localize(None), "y": np.log(train.values)})
    m = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=cps,
        seasonality_prior_scale=sps
    )
    m.fit(df)
    future = m.make_future_dataframe(periods=horizon, freq="B")
    fc = m.predict(future).tail(horizon)
    mean = np.exp(fc["yhat"].values)
    lo = np.exp(fc["yhat_lower"].values)
    hi = np.exp(fc["yhat_upper"].values)
    return mean, lo, hi

def prophet_search_cv(close: pd.Series, test_size: int, folds: int, max_trials: int):
    if not HAVE_PROPHET:
        return {"ok": False, "name":"Prophet", "err":"prophet unavailable"}

    splits = make_walkforward_splits(close, test_size, folds)
    if not splits:
        return {"ok": False, "name":"Prophet", "err":"Not enough data for CV"}

    cps_grid = [0.01, 0.05, 0.1, 0.3]
    sps_grid = [1.0, 5.0, 10.0]
    candidates = [(cps, sps) for cps in cps_grid for sps in sps_grid]
    candidates = candidates[:max_trials]

    best = None
    for cps, sps in candidates:
        rmses, mapes = [], []
        ok = True
        for train, test in splits:
            try:
                pred, _, _ = prophet_fit_predict(train, len(test), cps, sps)
                rmses.append(_rmse(test.values, pred))
                mapes.append(_mape(test.values, pred))
            except Exception:
                ok = False
                break
        if not ok:
            continue
        rec = {"cps": cps, "sps": sps, "rmse": float(np.mean(rmses)), "mape": float(np.mean(mapes))}
        if best is None or rec["rmse"] < best["rmse"]:
            best = rec

    if best is None:
        return {"ok": False, "name":"Prophet", "err":"All Prophet candidates failed"}
    return {"ok": True, "name":"Prophet", "best": best}

# -----------------------------
# Model: XGBoost (features + recursive multi-step)
# -----------------------------
def _make_xgb_features(close: pd.Series):
    s = close.dropna().astype(float)
    df = pd.DataFrame({"close": s})
    df["ret1"] = df["close"].pct_change()
    df["ret5"] = df["close"].pct_change(5)
    df["ma9"] = df["close"].rolling(9).mean()
    df["ma12"] = df["close"].rolling(12).mean()
    df["vol20"] = df["ret1"].rolling(20).std()
    # RSI 14
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["rsi14"] = 100 - (100 / (1 + rs))
    for k in [1,2,3,5,10]:
        df[f"lag{k}"] = df["close"].shift(k)
    return df.dropna()

def xgb_fit_predict(train_series: pd.Series, horizon: int, params: dict):
    feat = _make_xgb_features(train_series)
    X = feat.drop(columns=["close"])
    y = feat["close"]
    model = xgb.XGBRegressor(**params, random_state=42)
    model.fit(X, y)

    # Recursive forecast
    hist = train_series.dropna().astype(float).copy()
    preds = []
    for _ in range(horizon):
        fdf = _make_xgb_features(hist)
        last_X = fdf.drop(columns=["close"]).iloc[-1:]
        next_y = float(model.predict(last_X)[0])
        preds.append(next_y)
        next_idx = (hist.index[-1] + pd.tseries.offsets.BDay(1))
        hist.loc[next_idx] = next_y
    return np.array(preds, dtype=float), None, None

def xgb_search_cv(close: pd.Series, test_size: int, folds: int, max_trials: int):
    if not HAVE_XGBOOST:
        return {"ok": False, "name":"XGBoost", "err":"xgboost unavailable"}

    splits = make_walkforward_splits(close, test_size, folds)
    if not splits:
        return {"ok": False, "name":"XGBoost", "err":"Not enough data for CV"}

    # Small candidate set; capped by max_trials
    candidates = [
        dict(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0),
        dict(n_estimators=700, learning_rate=0.03, max_depth=5, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0),
        dict(n_estimators=900, learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.5),
        dict(n_estimators=1200, learning_rate=0.02, max_depth=5, subsample=0.7, colsample_bytree=0.9, reg_lambda=1.0),
    ][:max_trials]

    best = None
    for params in candidates:
        rmses, mapes = [], []
        ok = True
        for train, test in splits:
            try:
                pred, _, _ = xgb_fit_predict(train, len(test), params)
                rmses.append(_rmse(test.values, pred))
                mapes.append(_mape(test.values, pred))
            except Exception:
                ok = False
                break
        if not ok:
            continue
        rec = {"params": params, "rmse": float(np.mean(rmses)), "mape": float(np.mean(mapes))}
        if best is None or rec["rmse"] < best["rmse"]:
            best = rec

    if best is None:
        return {"ok": False, "name":"XGBoost", "err":"All XGBoost candidates failed"}
    return {"ok": True, "name":"XGBoost", "best": best}

# -----------------------------
# Fit best model on full data & forecast
# -----------------------------
def forecast_with_best(close: pd.Series, horizon: int, sarimax_best, prophet_best, xgb_best):
    out = []
    # SARIMAX
    if sarimax_best and sarimax_best.get("ok"):
        b = sarimax_best["best"]
        try:
            mean, lo, hi = sarimax_fit_predict(close.dropna().astype(float), horizon, b["order"], b["seasonal_order"])
            out.append({"ok": True, "name":"SARIMAX", "rmse": b["rmse"], "mape": b["mape"], "mean": mean, "lo": lo, "hi": hi, "best_params": b})
        except Exception as e:
            out.append({"ok": False, "name":"SARIMAX", "err": str(e)})
    else:
        out.append({"ok": False, "name":"SARIMAX", "err": sarimax_best.get("err") if sarimax_best else "n/a"})

    # Prophet
    if prophet_best and prophet_best.get("ok"):
        b = prophet_best["best"]
        try:
            mean, lo, hi = prophet_fit_predict(close.dropna().astype(float), horizon, b["cps"], b["sps"])
            out.append({"ok": True, "name":"Prophet", "rmse": b["rmse"], "mape": b["mape"], "mean": mean, "lo": lo, "hi": hi, "best_params": b})
        except Exception as e:
            out.append({"ok": False, "name":"Prophet", "err": str(e)})
    else:
        out.append({"ok": False, "name":"Prophet", "err": prophet_best.get("err") if prophet_best else "n/a"})

    # XGBoost
    if xgb_best and xgb_best.get("ok"):
        b = xgb_best["best"]
        try:
            mean, lo, hi = xgb_fit_predict(close.dropna().astype(float), horizon, b["params"])
            out.append({"ok": True, "name":"XGBoost", "rmse": b["rmse"], "mape": b["mape"], "mean": mean, "lo": lo, "hi": hi, "best_params": b})
        except Exception as e:
            out.append({"ok": False, "name":"XGBoost", "err": str(e)})
    else:
        out.append({"ok": False, "name":"XGBoost", "err": xgb_best.get("err") if xgb_best else "n/a"})

    return out

def ensemble_from_models(models):
    ok = [m for m in models if m.get("ok")]
    if not ok:
        return {"ok": False, "name":"Ensemble", "err":"No models available"}
    rmses = np.array([m["rmse"] for m in ok], dtype=float)
    w = 1.0 / (rmses + 1e-9)
    w = w / np.sum(w)
    means = np.stack([m["mean"] for m in ok], axis=0)
    ens = np.sum(means * w[:, None], axis=0)
    return {"ok": True, "name":"Ensemble", "weights": {ok[i]["name"]: float(w[i]) for i in range(len(ok))}, "mean": ens}

# -----------------------------
# Dashboard UI
# -----------------------------
def _tt(widget, text):
    widget.tooltip = text
    return widget

title = widgets.HTML(
    "<h3>Forecasting Pro Dashboard</h3>"
    "<div style='color:#666'>Multi-fold walk-forward CV + bounded hyperparameter search + ensemble weighting by CV RMSE</div>"
)

use_sp500 = _tt(widgets.ToggleButtons(
    options=[("Use S&P 500", True), ("Use Custom", False)],
    value=True,
    description="Universe:",
), "Choose S&P500 or custom tickers.")

tickers_text = _tt(widgets.Textarea(
    value="AAPL, MSFT, NVDA",
    description="Tickers:",
    layout=widgets.Layout(width="560px", height="70px"),
), "Custom tickers list (comma/space/newline separated).")

cache_path = _tt(widgets.Text(
    value="sp500_tickers.csv",
    description="S&P cache:",
    layout=widgets.Layout(width="360px"),
), "Local cache for S&P500 tickers.")

limit = _tt(widgets.IntText(value=25, description="Limit:"), "How many tickers to run (0=no limit). Start small—CV+tuning is heavy.")
interval = _tt(widgets.Dropdown(options=["1d"], value="1d", description="Interval:"), "Use 1d for forecasting.")
lookback_days = _tt(widgets.IntText(value=1200, description="Lookback:"), "History length (days). More helps CV.")
use_ema = _tt(widgets.ToggleButtons(options=[("EMA", True), ("SMA", False)], value=True, description="MA type:"), "EMA is more responsive.")
ma_fast = _tt(widgets.IntSlider(value=9, min=2, max=50, step=1, description="MA fast:"), "Fast MA period.")
ma_slow = _tt(widgets.IntSlider(value=12, min=2, max=100, step=1, description="MA slow:"), "Slow MA period.")

# Forecast controls
forecast_horizon = _tt(widgets.IntSlider(value=30, min=5, max=252, step=5, description="Horizon:"), "Forecast horizon in trading days.")
test_size = _tt(widgets.IntSlider(value=30, min=10, max=126, step=5, description="Fold test:"), "Test window per CV fold (days).")
cv_folds = _tt(widgets.IntSlider(value=3, min=2, max=6, step=1, description="Folds:"), "Number of walk-forward folds.")
max_trials = _tt(widgets.IntSlider(value=6, min=2, max=12, step=1, description="Trials:"), "Max candidate configs per model to try.")
seasonal_sarimax = _tt(widgets.Checkbox(value=False, description="Seasonal SARIMAX"), "Adds a simple (1,0,1,5) seasonal candidate to SARIMAX grid.")

# P/E
pe_max = _tt(widgets.FloatText(value=15.0, description="P/E max:"), "Absolute undervaluation rule.")
use_relative_pe = _tt(widgets.Checkbox(value=True, description="Relative P/E"), "Also undervalued if <= discount × median P/E.")
rel_discount = _tt(widgets.FloatSlider(value=0.80, min=0.20, max=1.20, step=0.01, description="Rel disc:"), "Relative undervaluation discount.")

filter_mode = _tt(widgets.Dropdown(
    options=[
        ("Show all", "all"),
        ("Undervalued only", "undervalued"),
        ("Cross_up only", "cross_up"),
        ("Best RMSE (top quartile)", "best_rmse_q1"),
    ],
    value="all",
    description="Filter:",
), "Filter results table.")

forecast_view = _tt(widgets.Dropdown(
    options=[("Ensemble", "ensemble"), ("SARIMAX", "sarimax"), ("Prophet", "prophet"), ("XGBoost", "xgboost"), ("All overlay", "all")],
    value="ensemble",
    description="Plot:",
), "Choose which forecast to plot.")

out_csv = _tt(widgets.Text(value="", description="Save CSV:", placeholder="results.csv", layout=widgets.Layout(width="260px")),
              "Optional: save filtered results to CSV.")

run_btn = _tt(widgets.Button(description="Run Analysis", button_style="success", icon="play"), "Run analysis (includes CV+tuning).")
clear_btn = _tt(widgets.Button(description="Clear", icon="trash"), "Clear outputs.")
plot_dropdown = _tt(widgets.Dropdown(options=[], description="Ticker:"), "Select a ticker to plot.")
plot_btn = _tt(widgets.Button(description="Plot Forecast", button_style="info", icon="line-chart"), "Plot selected forecast + intervals.")
scores_btn = _tt(widgets.Button(description="Model Scores", button_style="warning", icon="table"), "Show tuned params + CV scores + ensemble weights.")

progress = widgets.IntProgress(value=0, min=0, max=100, description="Progress:")
status = widgets.HTML("")

results_out = widgets.Output()
charts_out = widgets.Output()
tabs = widgets.Tab(children=[results_out, charts_out])
tabs.set_title(0, "Results")
tabs.set_title(1, "Charts")

controls = widgets.VBox([
    title,
    widgets.HBox([widgets.VBox([use_sp500, tickers_text, widgets.HBox([cache_path, limit]), widgets.HBox([interval, lookback_days])]),
                  widgets.VBox([use_ema, ma_fast, ma_slow,
                                widgets.Label("Forecast CV + tuning"),
                                forecast_horizon, test_size, cv_folds, max_trials, seasonal_sarimax,
                                widgets.Label("Valuation"),
                                pe_max, widgets.HBox([use_relative_pe, rel_discount]),
                                filter_mode, forecast_view
                               ])]),
    widgets.HBox([out_csv, run_btn, clear_btn]),
    progress,
    status,
    widgets.HBox([plot_dropdown, plot_btn, scores_btn]),
    tabs
])

# -----------------------------
# State + caching
# -----------------------------
_PRICE_MAP = {}
_RESULTS = pd.DataFrame()
_MED_PE = float("nan")

# Cache: (ticker, horizon, test_size, folds, trials, seasonal)-> dict(pack)
_MODEL_CACHE = {}

def _parse_custom_tickers(text):
    raw = text.replace("\n", ",").replace(" ", ",")
    parts = [p.strip() for p in raw.split(",")]
    return normalize_tickers([p for p in parts if p])

def _apply_filter(df):
    if df is None or df.empty:
        return df
    mode = filter_mode.value
    if mode == "all":
        return df
    if mode == "undervalued":
        return df[df["undervalued"] == True]
    if mode == "cross_up":
        return df[df["crossover"] == "cross_up"]
    if mode == "best_rmse_q1":
        tmp = df[np.isfinite(df["best_rmse"].fillna(np.nan))].copy()
        if tmp.empty:
            return tmp
        q = tmp["best_rmse"].quantile(0.25)
        return tmp[tmp["best_rmse"] <= q]
    return df

def compute_models_tuned(ticker: str, close: pd.Series):
    key = (ticker, int(forecast_horizon.value), int(test_size.value), int(cv_folds.value), int(max_trials.value), bool(seasonal_sarimax.value))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    horizon = int(forecast_horizon.value)
    ts = int(test_size.value)
    folds = int(cv_folds.value)
    trials = int(max_trials.value)
    seasonal = bool(seasonal_sarimax.value)

    # CV+tuning
    sar_best = sarimax_search_cv(close, ts, folds, seasonal, trials)
    pro_best = prophet_search_cv(close, ts, folds, trials)
    xgb_best = xgb_search_cv(close, ts, folds, trials)

    # Fit best on full and forecast
    models = forecast_with_best(close, horizon, sar_best, pro_best, xgb_best)
    ens = ensemble_from_models(models)

    # Find best model by CV RMSE
    ok = [m for m in models if m.get("ok")]
    if ok:
        best = sorted(ok, key=lambda x: x["rmse"])[0]
        best_name = best["name"]
        best_rmse = float(best["rmse"])
        best_mape = float(best["mape"])
    else:
        best_name, best_rmse, best_mape = "n/a", np.nan, np.nan

    pack = {
        "sarimax": sar_best,
        "prophet": pro_best,
        "xgboost": xgb_best,
        "models": models,
        "ensemble": ens,
        "best_name": best_name,
        "best_rmse": best_rmse,
        "best_mape": best_mape,
    }
    _MODEL_CACHE[key] = pack
    return pack

def run_screener(tickers):
    global _PRICE_MAP, _RESULTS, _MED_PE

    price_map = batch_download_prices(tickers, lookback_days.value, interval.value)
    _PRICE_MAP = price_map

    # PE map for median
    pe_map, pe_vals = {}, []
    for i, t in enumerate(tickers, 1):
        pe = get_trailing_pe(t)
        pe_map[t] = pe
        if pe is not None:
            pe_vals.append(pe)
        progress.value = int((i / max(1, len(tickers))) * 10)
    med_pe = float(np.median(pe_vals)) if pe_vals else float("nan")
    _MED_PE = med_pe

    rows = []
    for i, t in enumerate(tickers, 1):
        df = price_map.get(t)
        if df is None or df.empty or "Close" not in df.columns or df["Close"].dropna().empty:
            rows.append({"ticker": t, "pe": pe_map.get(t), "undervalued": False, "last_close": np.nan,
                         "ma_state":"no_data", "crossover":"no_data",
                         "best_model":"n/a", "best_rmse":np.nan, "best_mape":np.nan,
                         "notes":"no_price_data"})
            progress.value = 10 + int((i / max(1, len(tickers))) * 90)
            continue

        close = df["Close"].dropna().astype(float)
        fast = moving_average(close, int(ma_fast.value), bool(use_ema.value))
        slow = moving_average(close, int(ma_slow.value), bool(use_ema.value))

        mf = float(fast.iloc[-1]) if pd.notna(fast.iloc[-1]) else np.nan
        ms = float(slow.iloc[-1]) if pd.notna(slow.iloc[-1]) else np.nan

        ma_state = "neutral"
        if np.isfinite(mf) and np.isfinite(ms):
            if mf > ms: ma_state = "fast_above_slow"
            elif mf < ms: ma_state = "fast_below_slow"

        cross = crossover_signal(fast, slow)

        # Undervalued
        pe = pe_map.get(t)
        undervalued = False
        if pe is not None:
            abs_ok = pe <= float(pe_max.value)
            rel_ok = False
            if bool(use_relative_pe.value) and np.isfinite(med_pe) and med_pe > 0:
                rel_ok = pe <= (float(rel_discount.value) * med_pe)
            undervalued = abs_ok or (bool(use_relative_pe.value) and rel_ok)

        # Forecast CV+tuning (heavy)
        pack = compute_models_tuned(t, close)

        rows.append({
            "ticker": t,
            "pe": pe,
            "undervalued": bool(undervalued),
            "last_close": float(close.iloc[-1]),
            "ma_state": ma_state,
            "crossover": cross,
            "best_model": pack["best_name"],
            "best_rmse": pack["best_rmse"],
            "best_mape": pack["best_mape"],
            "notes": f"packages: statsmodels={HAVE_STATSMODELS}, prophet={HAVE_PROPHET}, xgboost={HAVE_XGBOOST}"
        })

        progress.value = 10 + int((i / max(1, len(tickers))) * 90)

    out = pd.DataFrame(rows)

    # Sort: undervalued first, then lower RMSE, then lower PE
    out["pe_sort"] = out["pe"].fillna(np.inf)
    out["rmse_sort"] = out["best_rmse"].fillna(np.inf)
    out = out.sort_values(by=["undervalued", "rmse_sort", "pe_sort"], ascending=[False, True, True]).drop(columns=["pe_sort", "rmse_sort"])

    _RESULTS = out
    return out

def render_results(df):
    with results_out:
        clear_output()
        if df is None or df.empty:
            print("No results.")
            return
        display(df.style.format({
            "pe":"{:.2f}",
            "last_close":"{:.2f}",
            "best_rmse":"{:.4f}",
            "best_mape":"{:.2%}",
        }).hide(axis="index"))

def plot_forecast_for_ticker(ticker: str):
    with charts_out:
        clear_output()
        df = _PRICE_MAP.get(ticker)
        if df is None or df.empty:
            print("No price data cached for", ticker)
            return
        close = df["Close"].dropna().astype(float)
        if close.shape[0] < 400:
            print("Not much data; forecasts may be unstable for", ticker)

        pack = compute_models_tuned(ticker, close)
        models = pack["models"]
        ens = pack["ensemble"]

        horizon = int(forecast_horizon.value)
        future_idx = pd.date_range(start=close.index[-1] + pd.tseries.offsets.BDay(1), periods=horizon, freq="B")

        # MAs
        fast = moving_average(close, int(ma_fast.value), bool(use_ema.value))
        slow = moving_average(close, int(ma_slow.value), bool(use_ema.value))

        plt.figure(figsize=(12,5))
        plt.plot(close.index, close.values, label="Close")
        plt.plot(fast.index, fast.values, label=f"{'EMA' if use_ema.value else 'SMA'}({ma_fast.value})")
        plt.plot(slow.index, slow.values, label=f"{'EMA' if use_ema.value else 'SMA'}({ma_slow.value})")

        view = forecast_view.value

        def plot_one(m):
            if not m.get("ok"):
                return
            plt.plot(future_idx, m["mean"], label=f"{m['name']} (CV RMSE {m['rmse']:.3f})")
            if m.get("lo") is not None and m.get("hi") is not None:
                plt.fill_between(future_idx, m["lo"], m["hi"], alpha=0.15)

        if view == "all":
            for m in models:
                plot_one(m)
            if ens.get("ok"):
                plt.plot(future_idx, ens["mean"], linewidth=2.5, label="Ensemble")
        elif view == "ensemble":
            if ens.get("ok"):
                plt.plot(future_idx, ens["mean"], linewidth=2.5, label="Ensemble")
                wtxt = ", ".join([f"{k}:{v:.2f}" for k,v in ens.get("weights",{}).items()])
                plt.title(f"{ticker} — Ensemble weights: {wtxt}")
            else:
                plt.title(f"{ticker} — Ensemble unavailable")
        else:
            name_map = {"sarimax":"SARIMAX", "prophet":"Prophet", "xgboost":"XGBoost"}
            target = name_map.get(view, "")
            mm = next((m for m in models if m.get("name")==target), None)
            if mm and mm.get("ok"):
                plot_one(mm)
                plt.title(f"{ticker} — {target} tuned forecast")
            else:
                plt.title(f"{ticker} — {target} unavailable")

        plt.grid(True)
        plt.legend()
        plt.show()

def show_model_scores(ticker: str):
    with charts_out:
        clear_output()
        df = _PRICE_MAP.get(ticker)
        if df is None or df.empty:
            print("No price data cached for", ticker)
            return
        close = df["Close"].dropna().astype(float)
        pack = compute_models_tuned(ticker, close)

        rows = []
        for m in pack["models"]:
            if m.get("ok"):
                rows.append({"model": m["name"], "cv_rmse": m["rmse"], "cv_mape": m["mape"], "best_params": str(m.get("best_params", ""))[:200]})
            else:
                rows.append({"model": m["name"], "cv_rmse": np.nan, "cv_mape": np.nan, "error": m.get("err","")})
        display(pd.DataFrame(rows))

        ens = pack["ensemble"]
        if ens.get("ok"):
            print("\nEnsemble weights (lower CV RMSE => higher weight):")
            for k,v in ens.get("weights", {}).items():
                print(f"  {k}: {v:.3f}")

        print("\nTuning picks:")
        print("  SARIMAX:", pack["sarimax"].get("best") if pack["sarimax"].get("ok") else pack["sarimax"].get("err"))
        print("  Prophet:", pack["prophet"].get("best") if pack["prophet"].get("ok") else pack["prophet"].get("err"))
        print("  XGBoost:", pack["xgboost"].get("best") if pack["xgboost"].get("ok") else pack["xgboost"].get("err"))

# -----------------------------
# Button callbacks
# -----------------------------
def on_run(_):
    global _MODEL_CACHE
    _MODEL_CACHE = {}  # reset when settings change
    progress.value = 0
    status.value = "<b>Starting...</b>"

    try:
        if use_sp500.value:
            tickers = load_sp500_tickers_cached(cache_path.value.strip() or "sp500_tickers.csv")
        else:
            tickers = _parse_custom_tickers(tickers_text.value)

        tickers = normalize_tickers(tickers)
        lim = int(limit.value)
        if lim > 0:
            tickers = tickers[:lim]
        if not tickers:
            status.value = "<span style='color:red'><b>No tickers selected.</b></span>"
            return

        status.value = f"Running CV+tuning for <b>{len(tickers)}</b> tickers — this is compute-heavy."
        tabs.selected_index = 0

        df = run_screener(tickers)
        df_show = _apply_filter(df)

        plot_dropdown.options = df_show["ticker"].tolist()[:200] if not df_show.empty else []
        render_results(df_show)

        if out_csv.value.strip():
            df_show.to_csv(out_csv.value.strip(), index=False)
            with results_out:
                print(f"\nSaved CSV: {out_csv.value.strip()}")

        status.value = f"Done. Median P/E (available): <b>{_MED_PE:.2f}</b> • packages: statsmodels={HAVE_STATSMODELS}, prophet={HAVE_PROPHET}, xgboost={HAVE_XGBOOST}"

    except Exception as e:
        status.value = f"<span style='color:red'><b>Error:</b> {e}</span>"
        raise
    finally:
        progress.value = 100

def on_clear(_):
    with results_out: clear_output()
    with charts_out: clear_output()
    status.value = ""
    progress.value = 0

def on_plot(_):
    t = plot_dropdown.value
    if not t:
        return
    tabs.selected_index = 1
    plot_forecast_for_ticker(t)

def on_scores(_):
    t = plot_dropdown.value
    if not t:
        return
    tabs.selected_index = 1
    show_model_scores(t)

run_btn.on_click(on_run)
clear_btn.on_click(on_clear)
plot_btn.on_click(on_plot)
scores_btn.on_click(on_scores)

display(controls)


# In[ ]:




