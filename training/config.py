DEFAULT_WATCHLIST = (
    "ADANIPORTS.NS",
    "BEL.NS",
    "BHARATFORG.NS",
    "ITC.NS",
    "TCS.NS",
)
MARKET_OVERVIEW_INSTRUMENTS = (
    {
        "key": "nifty",
        "label": "NIFTY 50",
        "ticker": "^NSEI",
        "asset_class": "India index",
        "currency": "INR",
        "unit": "index points",
    },
    {
        "key": "brent",
        "label": "BRENT CRUDE",
        "ticker": "BZ=F",
        "asset_class": "Energy futures",
        "currency": "USD",
        "unit": "USD / barrel",
    },
    {
        "key": "sp500",
        "label": "S&P 500",
        "ticker": "^GSPC",
        "asset_class": "US index",
        "currency": "USD",
        "unit": "index points",
    },
    {
        "key": "eurostoxx",
        "label": "EURO STOXX 50",
        "ticker": "^STOXX50E",
        "asset_class": "Europe index",
        "currency": "EUR",
        "unit": "index points",
    },
)
LIVE_DATA_PERIOD = "60d"
LIVE_DATA_INTERVAL = "15m"
LIVE_DATA_CACHE_SECONDS = 300
WATCHLIST_QUOTE_CACHE_SECONDS = 30
WATCHLIST_QUOTE_INTERVAL = "1m"
STOCK_MODEL_CACHE_SECONDS = 21600
MARKET_DATA_CACHE_SECONDS = 900
FII_DII_CACHE_SECONDS = 900
FUNDAMENTALS_CACHE_SECONDS = 7200
NEWS_CACHE_SECONDS = 1800
SENTIMENT_CACHE_SECONDS = 3600
MIN_CONFIDENCE = 0.65
ADX_TREND_THRESHOLD = 25.0
FORWARD_BARS = 8
TARGET_THRESHOLD = 0.005
STOCK_MODEL_PARAMS = {
    "n_estimators": 150,
    "max_depth": 5,
    "learning_rate": 0.01,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "class_weight": "balanced",
    "verbose": -1,
    "random_state": 42,
    "n_jobs": 1,
}
