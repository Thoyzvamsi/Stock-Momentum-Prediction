# Stock Momentum Prediction

> A generalized multi-stock momentum prediction system — end-to-end ML pipeline with live signals, backtesting, fundamental scoring, and AI news sentiment.

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35%2B-red)](https://streamlit.io)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3%2B-green)](https://lightgbm.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## What It Does

Trains a single LightGBM classifier on combined OHLCV data from multiple stocks, then deploys it as a live Streamlit dashboard that generates **BUY / SELL / HOLD** signals for any stock symbol in real time.

Key design choice: one model generalizes across stocks rather than one model per ticker. This forces the features to be scale-invariant and stock-relative — which makes the approach more robust and more deployable.

---

## Architecture

```
Stock-Momentum-Prediction/
│
├── training/                # Python pipeline and pinned environment
│   ├── config.py            # Shared data, model, and artifact paths
│   ├── data/                # Price pipeline and combined OHLCV training data
│   ├── features/            # Feature and target engineering
│   ├── engine/              # Indicators and cost-aware backtester
│   ├── fundamentals/        # Fundamental scoring
│   ├── models/              # Saved model artifacts
│   ├── predictor/           # Live inference
│   ├── model_training.py    # LightGBM training, evaluation, save/load
│   ├── run_train.py         # Training entry point
│   ├── requirements.txt     # Exact direct dependency pins
│   └── .venv/               # Local virtual environment (not committed)
│
├── tests/                   # Focused pipeline and sentiment tests
├── dashboard.py             # Streamlit app
├── ticker_utils.py           # Defaults bare ticker symbols to NSE (.NS)
├── groq_sentiment.py         # Groq model selection, JSON validation, and errors
├── news_utils.py             # Bounded timeout retry for Google News RSS
└── .streamlit/config.toml   # Dark theme config
```

In the sidebar, enter an NSE ticker as `HDFCBANK` or `RELIANCE`; `.NS` is added
automatically. Explicit exchange suffixes such as `.BO` are kept as entered.

---

## Pipeline Flow

```
raw_data.csv (multi-stock OHLCV)
        │
        ▼
GeneralizedFeatureEngineer          ← 22 scale-invariant features, calculated per ticker
        │
        ▼
GeneralizedTargetEngineer           ← 3-class target: BUY(+1) / NEUTRAL(0) / SELL(-1)
        │                              forward_return > threshold → 1, < -threshold → -1
        ▼
GeneralizedMLTrainer
  ├── walk_forward_validation()     ← Date-based folds with a forward-horizon embargo
  ├── train_final_model()           ← LightGBM, class_weight='balanced'
  └── evaluate() + save_model()
        │
        ▼
CostAwareBacktester                 ← commission=0.1%, slippage=0.2%, hold_bars=15
        │
        ▼
training/models/generalized_momentum_*.pkl  ← ready for live inference
        │
        ▼
dashboard.py (Streamlit)
  ├── StockPredictor.predict_latest()   ← BUY/SELL/HOLD + confidence
  ├── Fundamentals.compute_score()      ← 0-100 investment score
  └── fetch_and_analyze_news()          ← Groq LLaMA-3.3 sentiment analysis
```

---

## Feature Engineering

All features are designed to be **scale-invariant** — they work on a ₹50 stock and a ₹5000 stock equally because they use ratios, percentiles, and per-stock z-scores instead of raw prices or volumes.

| Tier | Feature | What It Captures |
|------|---------|-----------------|
| 1 | `vol_imbalance` | Buy vs sell pressure over 20 bars |
| 1 | `candle_size_pct` | Is this candle abnormally large? (rolling percentile) |
| 1 | `volume_surge` | Volume vs its own 20-bar average |
| 1 | `tick_momentum` | Net directional pressure over 10 bars |
| 1 | `vwap_dist` | Price distance from VWAP (cumulative per session) |
| 2 | `body_wick_ratio` | Candle conviction (body / total range) |
| 2 | `vol_percentile` | Is volatility high or low? (rolling z-score) |
| 2 | `consecutive_bars` | How many bars in the same direction |
| 2 | `price_structure` | Higher-highs vs lower-lows count |
| 2 | `acceleration` | Change in 5-bar momentum vs 10-bar momentum |
| 3 | `is_first_hour` | Market open effect (09:15–10:15 IST) |
| 3 | `is_lunch` | Lunch lull effect (12:00–14:00) |
| 3 | `is_last_hour` | Power hour effect (15:00+) |
| 3 | `range_position` | Where close sits in the high-low range |
| 3 | `gap_size` | Opening gap from previous close |
| 3 | `mfi` | Money Flow Index (volume-weighted RSI) |
| Market | `relative_volatility` | Vol z-scored within each stock's own history |
| Market | `momentum_rank` | 20-bar return percentile within stock's history |
| Market | `volume_rank` | Normalized volume percentile within stock's history |

**Why per-stock z-scoring instead of cross-sectional ranking:** Cross-sectional ranking requires all stocks to be sampled at the same timestamps. Multi-stock intraday data is uneven (NSE vs NYSE trading hours, halts, etc.) — per-stock rolling z-scores sidestep this entirely while still capturing relative context.

---

## Target Variable

3-class classification on forward returns:

```
forward_return = (close[t + forward_bars] / close[t]) - 1

  BUY  (+1):  forward_return >  threshold  (default 0.5%)
  SELL (-1):  forward_return < -threshold
  NEUTRAL (0): within ±threshold (dropped during training)
```

Neutral samples are removed before training. This keeps the model focused on high-conviction setups rather than noise.

---

## Model

**LightGBM classifier** with:
- `n_estimators=500`, `max_depth=5`, `learning_rate=0.01`, `num_leaves=31`
- `class_weight='balanced'` — handles class imbalance
- `subsample=0.8`, `colsample_bytree=0.7` — regularization
- Walk-forward validation (5 folds) to verify no lookahead bias before training the final model
- StandardScaler normalization before fit

---

## Backtesting

`CostAwareBacktester` simulates realistic execution costs:

| Parameter | Default | Notes |
|-----------|---------|-------|
| Commission | 0.1% | One-way (charged on entry and exit) |
| Slippage | 0.2% | One-way estimate |
| Round-trip cost | 0.6% | Total drag per trade |
| Hold period | 15 bars | Fixed exit after N bars |
| Min confidence | 65% | Below this → no trade |
| Position size | 10 shares | Fixed shares per trade |

Metrics reported: Win Rate, Total Return, Annual Return, Sharpe Ratio, Max Drawdown, Profit Factor, Expected Value, per-ticker breakdown.

---

## Setup

### 1. Clone and set up Python

```powershell
git clone https://github.com/Thoyzvamsi/stock-momentum-prediction.git
Set-Location stock-momentum-prediction
py -3.11 -m venv training\.venv
training\.venv\Scripts\python.exe -m pip install --upgrade pip
training\.venv\Scripts\python.exe -m pip install -r training\requirements.txt
```

The checked-in dependency pins were verified with Python 3.11. Use the virtual
environment's Python for commands below.
VS Code is configured to use `training\.venv\Scripts\python.exe` for this workspace.
The root `requirements.txt` is used by Streamlit Community Cloud; the training
requirements also include pytest for local validation.

### 2. Training data

The combined training CSV is `training/data/raw_data.csv`. Required columns:

```
Open, High, Low, Close, Volume, Ticker
```

The file should contain data for multiple tickers (e.g. TCS.NS, INFY.NS, RELIANCE.NS) and a `Datetime` column. More tickers = better generalization.

### 3. Set API key

For **local development**, create `.env`:

```powershell
Copy-Item .env.example .env
```

Then edit `.env` locally and replace the empty key:

```
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```
`GROQ_MODEL` is optional. The app checks Groq's available models and uses this
model first, then tries its configured fallbacks. If Groq rejects the key,
replace it; sentiment will report as unavailable while the rest of the app
continues to work. Start from `.env.example`.

For **Streamlit Cloud**, add to Settings → Secrets:
```toml
GROQ_API_KEY = "your_key_here"
GROQ_MODEL = "llama-3.3-70b-versatile"
```

Get a free key at [console.groq.com](https://console.groq.com).

### 4. Train the model

First run the bounded smoke test. It uses synthetic data (3 tickers, 125 sessions,
30 training rounds) and verifies feature/label boundaries and model save/load.

```powershell
training\.venv\Scripts\python.exe -m training.run_smoke
```

Only after it passes, start the full training run:

```powershell
training\.venv\Scripts\python.exe -m training.run_train
```

Optional flags:
```powershell
training\.venv\Scripts\python.exe -m training.run_train --forward-bars 8 --threshold 0.005 --test-size 0.2 --commission 0.001 --slippage 0.002 --min-confidence 0.65 --hold-bars 15
```

### 5. Launch dashboard

```powershell
training\.venv\Scripts\streamlit.exe run dashboard.py
```

Run the focused tests:

```powershell
training\.venv\Scripts\python.exe -m pytest tests -q
```

---

## Dashboard

The Streamlit dashboard has four sections:

**Signal Panel** — Latest BUY/SELL/HOLD signal with confidence score, expected return, and action label (ENTER_LONG, CONSIDER_LONG, etc.)

**Candlestick Chart** — 15-minute OHLCV candles for the last 2 trading days with volume overlay. Market hours gaps removed from the x-axis (supports both NSE and NYSE sessions).

**Fundamental Analysis** — Investment score (0–100) computed from Forward P/E, PEG Ratio, Return on Equity, Debt/Equity, and Revenue Growth. Analyst consensus targets when available.

**AI News Sentiment** — Fetches recent headlines via Google News RSS, analyzes with LLaMA-3.3-70b (Groq) to produce: overall sentiment, sentiment score (-100 to +100), bullish/bearish factors, short-term outlook, and risk factors.

---

## Known Limitations

- **Backtest realism**: The backtester uses fixed hold periods and doesn't model partial fills, market impact, or overnight gaps.
- **VWAP calculation**: Currently cumulative (session-level), not reset per trading day. This means VWAP drifts on multi-day data.
- **No live order execution**: This is a research/signal tool. It does not connect to any broker API.
- **15m timeframe only**: The dashboard fetches 15-minute bars. The model was trained on 15-minute data. Don't apply it to daily bars without retraining.
- **NSE bias**: Default ticker is TCS.NS. The model generalizes across stocks but was developed and tested primarily on Indian equities.

---

## Deployment

Deployed on Streamlit Community Cloud:  
**[stock-momentum-prediction.streamlit.app](https://stock-momentum-prediction-psd5apu4lbgtwyxwo2obrk.streamlit.app/)**

To deploy or update the app on Streamlit Community Cloud:

1. Push the repository to GitHub and confirm all three files in
   `training/models/` (model, scaler, and config) are committed. The app does
   not need the training CSV at runtime.
2. Create or open the app in Streamlit Community Cloud using this repository,
   select the deployment branch, and set the main file to `dashboard.py`.
3. Select Python 3.11 in the app's advanced settings. Cloud installs the
   dashboard dependencies from the root `requirements.txt`.
4. Add `GROQ_API_KEY` under App settings → Secrets if AI news sentiment is
   wanted. `GROQ_MODEL` is optional. Do not commit `.env` or Streamlit secrets.
   The dashboard remains usable without a Groq key, but sentiment is disabled.
5. Restart the app after changing secrets or replacing the committed model
   artifacts.

For local launch, use:

```powershell
training\.venv\Scripts\streamlit.exe run dashboard.py
```

Training runs are visible in the terminal and mirrored to
`training/logs/<run-id>.log`. Each completed or failed run also writes a JSON
report under `experiments/runs/` and adds a summary to `experiments/LOG.md`.

---

## Results

Full training completed on 2026-10-25 using 145,046 rows from five tickers
(2019-03-01 through 2023-10-12), with 72,294 directional training samples and
22 features. The dataset is historical and ends in 2023; refresh it before
using this model for current market analysis.

| Metric | Value |
|--------|-------|
| Walk-forward accuracy | 51.2% ± 1.7% |
| Holdout accuracy / F1 | 52.8% / 52.8% |
| Backtest win rate (169 trades) | 50.3% |
| Backtest total return / Sharpe | 252.5% / 1.51 |
| Backtest max drawdown | -2.2% |

These metrics are exploratory, not a production-performance claim. The
walk-forward and holdout scores are only modestly above chance, and the
backtest still needs comparisons against always-HOLD, buy-and-hold, and an EMA
crossover before its reported return should be trusted. This app is for
research purposes only, not financial advice.

---

## Tech Stack

| Layer | Library |
|-------|---------|
| Data | yfinance, pandas, numpy |
| ML | LightGBM, scikit-learn |
| Backtesting | Custom (`CostAwareBacktester`) |
| Dashboard | Streamlit, Plotly |
| Sentiment | Groq API (LLaMA-3.3-70b) |
| Fundamentals | yfinance `.info` |

---

## License

MIT
