# Stock Momentum Prediction

> A generalized multi-stock momentum prediction system — end-to-end ML pipeline with live signals, backtesting, fundamental scoring, and AI news sentiment.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
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
├── data/
│   ├── pipeline.py          # MultiStockDataPipeline — live fetch (yfinance) + validation
│   └── raw_data.csv         # Combined OHLCV training data [Open, High, Low, Close, Volume, Ticker]
│
├── features/
│   └── engineering.py       # GeneralizedFeatureEngineer (19 features) + TargetEngineer
│
├── training/
│   └── model_training.py    # GeneralizedMLTrainer — walk-forward CV, LightGBM, save/load
│
├── engine/
│   └── backtest.py          # CostAwareBacktester — commission + slippage, per-ticker stats
│
├── predictor/
│   └── stock_predictor.py   # StockPredictor + SignalGenerator — live inference
│
├── fundamentals/
│   └── scoring.py           # Fundamentals — forward P/E, PEG, ROE, D/E, revenue growth
│
├── models/                  # Saved artifacts (pkl + json) after training
│
├── .streamlit/
│   └── config.toml          # Dark theme config
│
├── dashboard.py             # Streamlit app — main entry point
├── multi_stock_main.py      # Training pipeline entry point
└── requirements.txt
```

---

## Pipeline Flow

```
raw_data.csv (multi-stock OHLCV)
        │
        ▼
GeneralizedFeatureEngineer          ← 19 scale-invariant features (z-scored per stock)
        │
        ▼
GeneralizedTargetEngineer           ← 3-class target: BUY(+1) / NEUTRAL(0) / SELL(-1)
        │                              forward_return > threshold → 1, < -threshold → -1
        ▼
GeneralizedMLTrainer
  ├── walk_forward_validation()     ← TimeSeriesSplit (5 folds) to detect lookahead bias
  ├── train_final_model()           ← LightGBM, class_weight='balanced'
  └── evaluate() + save_model()
        │
        ▼
CostAwareBacktester                 ← commission=0.1%, slippage=0.2%, hold_bars=15
        │
        ▼
models/generalized_momentum_*.pkl  ← ready for live inference
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

### 1. Clone & install

```bash
git clone https://github.com/Thoyzvamsi/stock-momentum-prediction.git
cd stock-momentum-prediction
pip install -r requirements.txt
```

### 2. Add training data

Place a CSV file named `raw_data.csv` in the `data/` folder. Required columns:

```
Open, High, Low, Close, Volume, Ticker
```

The file should contain data for multiple tickers (e.g. TCS.NS, INFY.NS, RELIANCE.NS). More tickers = better generalization.

### 3. Set API key

For **local development**, create `.env`:
```
GROQ_API_KEY=your_key_here
```

For **Streamlit Cloud**, add to Settings → Secrets:
```toml
GROQ_API_KEY = "your_key_here"
```

Get a free key at [console.groq.com](https://console.groq.com).

### 4. Train the model

```bash
python multi_stock_main.py
```

Optional flags:
```bash
python multi_stock_main.py \
  --forward-bars 8 \
  --threshold 0.005 \
  --test-size 0.2 \
  --commission 0.001 \
  --slippage 0.002 \
  --min-confidence 0.65 \
  --hold-bars 15
```

### 5. Launch dashboard

```bash
streamlit run dashboard.py
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

The `models/` directory (trained artifacts) must be committed to the repo for Streamlit Cloud to load the predictor. Model files are ~5MB and tracked in Git.

---

## Results

*Fill this in after running `multi_stock_main.py` on your data.*

| Metric | Value |
|--------|-------|
| Walk-forward accuracy | — |
| Test accuracy | — |
| Win rate (backtest) | — |
| Sharpe ratio | — |
| Max drawdown | — |

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
