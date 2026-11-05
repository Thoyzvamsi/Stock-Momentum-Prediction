# Stock Momentum Prediction

Stock Momentum is a stock-research web app with a FastAPI backend and a
JavaScript, HTML, and CSS frontend. It provides a Home workspace for market
context and a Stock Analyzer that fits a model to the selected stock's own
recent price history.

> **For experiments only. Not financial advice.** Predictions and expected
> returns are estimates, not trade instructions or guaranteed outcomes. They
> are not independently validated performance claims. Market data may be
> delayed, incomplete, or unavailable.

## What the app includes

- **Home:** editable watchlist, symbol search/filter, latest prices and daily
  changes, a default NIFTY 50 market chart, and chart selections for Brent
  crude, S&P 500, and EURO STOXX 50.
- **Stock Analyzer:** per-stock prediction and expected return, confidence,
  15-minute price chart, and selectable 30-, 45-, or 60-day history.
- **Supporting research:** trend indicators, FII/FPI and DII flows,
  fundamentals, analyst information, peer comparisons, and quarterly results
  when the data sources provide them.
- **AI market read:** related headlines and a sentiment summary when a Groq
  API key is configured.

The watchlist is stored in the current browser's local storage; it is not
shared between browsers or devices. It supports up to 20 symbols and refreshes
quotes about once a minute. It does not run stock predictions. The Analyzer
trains a LightGBM model from the selected symbol's available 15-minute
history, then caches it in application memory for up to six hours. Models are
not persisted across server restarts.

## How a prediction is produced

When you analyze a symbol, the server fetches its adjusted 15-minute OHLCV bars
for the selected 30-, 45-, or 60-day window. It builds features from that
symbol's own history; the model does not use other stocks' training rows. The
watchlist does not train models or generate predictions.

The flow is:

1. **Build features from each bar.** The feature set describes recent price,
   volume, candle shape, volatility, momentum, and time-of-day context. For
   example, it includes volume imbalance and surge, distance from VWAP,
   consecutive up/down bars, higher-high/lower-low structure, price and volume
   percentiles, money flow, ATR, and ADX. Features are calculated from current
   and prior bars, not future prices.
2. **Create historical labels.** For every bar with enough future data, the
   app calculates
   `return = (close 8 bars later / current close) - 1`.
   A return above **+0.5%** is labeled upward (`+1`); below **-0.5%** is
   labeled downward (`-1`); returns between those thresholds are neutral
   (`0`). The neutral rows are excluded from model fitting, so the model learns
   to distinguish upward from downward moves rather than predict the neutral
   class. Eight 15-minute trading bars are about two hours of market trading,
   but overnight closures, weekends, and missing bars can make the elapsed
   clock time longer.
3. **Fit a stock-specific classifier.** The app drops rows with incomplete
   features or labels, then requires at least 250 usable directional examples
   and at least 10 examples of each direction. A `StandardScaler` is fitted on
   those rows, followed by a LightGBM classifier. The current settings are
   150 boosting iterations, maximum depth 5, learning rate 0.01, 31 leaves,
   and balanced class weights.
4. **Score the latest bar.** The newest complete feature row is transformed
   with the fitted scaler. LightGBM returns probabilities for the upward and
   downward classes. The class with the larger probability is the predicted
   direction, and the larger probability is reported as confidence. A
   probability below **65%** changes the displayed action to `HOLD`; at 65%
   or above the action is `BUY` or `SELL` according to the more likely class.
   Strength is `STRONG` at 70% or higher, `MODERATE` from 65% to below 70%,
   and `WEAK` below 65%.
5. **Estimate expected return.** During fitting, the app calculates the
   historical mean forward return for each direction after clipping extreme
   returns to the 1st and 99th percentiles. It weights those two averages by
   the latest upward/downward probabilities and adds them. The result is
   displayed as a percentage. It is an estimate based on this stock's recent
   labeled history—not a promised return or a separately trained return
   forecast. It is possible to see `HOLD` with a non-zero expected return,
   because the action uses the 65% confidence threshold while expected return
   uses both class probabilities.

The first request for a symbol and history window fits the model; subsequent
requests reuse that model for up to six hours. The cache is in memory and is
lost on restart. A different symbol or window gets its own model. Market
trend, FII/DII flows, fundamentals, peers, quarterly results, and the optional
AI news read are supporting context shown alongside the prediction; they are
not inputs to this classifier and do not override its output.

### Important model limitations

The classifier is fitted on all eligible rows in the selected recent window.
The app does **not** currently hold out a validation set, run walk-forward
validation, tune hyperparameters per stock, calibrate the probabilities, or
backtest the live prediction. As a result, confidence is the model's raw
probability estimate, not a proven likelihood of being correct. The expected
return is derived from the same training window. Neither metric should be
treated as evidence of future profitability. This model is for exploration,
not financial advice or automated trading.

## Data and limitations

The app retrieves price, company, analyst, peer, and quarterly data through
Yahoo Finance's `yfinance` library. FII/FPI and DII flows come from NSE's
public market-flow feed when available; the feed may provide only the latest
session. News headlines are fetched from Google News RSS. AI summarization
uses Groq and is optional.

These third-party sources can be delayed, rate-limited, or unavailable. Some
stocks may not have enough recent history or both directional outcomes to fit
a usable model. In that case, analysis may be unavailable. The application
does not connect to a broker, execute trades, or guarantee prediction
accuracy.

## Run locally

Requires Python 3.11 or newer. From the repository root, create a virtual
environment and install the app dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

Start the app:

```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>. FastAPI's interactive API documentation is at
<http://127.0.0.1:8000/docs>.

### Optional Groq configuration

AI news analysis requires a Groq API key. Set `GROQ_API_KEY` in the server
environment, or create a root-level `.env` file:

```dotenv
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

`GROQ_MODEL` is optional. The app checks available models and uses its
configured fallback choices. Without a valid key or available model, the rest
of the application remains usable and the AI read is reported as unavailable.
Keep `.env` private; never commit the key or expose it in the browser.

## API

- `GET /api/health` — application health check.
- `GET /api/market` — market overview instruments and chart history.
- `GET /api/index` — default watchlist quotes and FII/DII flows.
- `GET /api/index?tickers=HDFCBANK,ITC,TCS` — quotes for the supplied symbols
  (maximum 20).
- `GET /api/stock/{ticker}?period=60d` — stock analysis. Supported periods
  are `30d`, `45d`, and `60d`.

Common symbols can be entered without an exchange suffix; the server
normalizes them for data lookup while the interface displays the clean symbol.

## Tests

The development requirements include pytest and Uvicorn. Run the test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

## Project layout

```text
api/                         FastAPI routes, data services, and response schemas
training/config.py           Runtime model and market-data settings
training/engine/             Technical indicators
training/features/           Stock-specific feature and target engineering
training/fundamentals/       Fundamental scoring
public/                      CDN-served HTML, CSS, and JavaScript
tests/                       API, feature, ticker, and Groq helper tests
main.py                      ASGI entrypoint (main:app)
requirements.txt             Runtime dependencies
vercel.json                  SPA route rewrites
```

## Deploy on Vercel

Import this repository into Vercel. It detects the FastAPI app exported as
`app` from the root `main.py` (`main:app`), installs runtime dependencies from
`pyproject.toml` and `requirements.txt`, serves files from `public/` through
its CDN, and uses the rewrites in `vercel.json` for analyzer page URLs. Set
`GROQ_API_KEY` as a Vercel environment variable if AI news analysis is needed;
`GROQ_MODEL` is optional. The model cache is in memory, so a cold function
start can trigger training on the first analysis request for each stock and
history window. The Vercel build hook downloads a pinned, SHA-256-verified
OpenMP runtime required by LightGBM and bundles it with the function; no Docker
image is used.
