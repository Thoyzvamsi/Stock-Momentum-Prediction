from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from html import unescape
import json
import logging
import os
import re
from pathlib import Path
from threading import Lock
from time import monotonic
from urllib.parse import quote, urlparse
import xml.etree.ElementTree as ET

import httpx
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from dotenv import load_dotenv
from groq import Groq

from groq_sentiment import (
    GroqCooldownError,
    GroqInvalidResponseError,
    GroqRequestLimiter,
    groq_error_message,
    rate_limit_retry_after,
    select_available_model,
)
from news_utils import NewsFetchTimeoutError, fetch_news_rss
from ticker_utils import normalize_ticker_symbol
from training.config import (
    ADX_TREND_THRESHOLD,
    DEFAULT_WATCHLIST,
    FII_DII_CACHE_SECONDS,
    FUNDAMENTALS_CACHE_SECONDS,
    LIVE_DATA_CACHE_SECONDS,
    LIVE_DATA_INTERVAL,
    LIVE_DATA_PERIOD,
    MARKET_DATA_CACHE_SECONDS,
    MARKET_OVERVIEW_INSTRUMENTS,
    FORWARD_BARS,
    MIN_CONFIDENCE,
    NEWS_CACHE_SECONDS,
    SENTIMENT_CACHE_SECONDS,
    STOCK_MODEL_PARAMS,
    STOCK_MODEL_CACHE_SECONDS,
    TARGET_THRESHOLD,
    WATCHLIST_QUOTE_INTERVAL,
    WATCHLIST_QUOTE_CACHE_SECONDS,
)
from training.engine.indicators import Indicators
from training.features.engineering import (
    StockFeatureEngineer,
    StockTargetEngineer,
)
from training.fundamentals.scoring import Fundamentals


logger = logging.getLogger(__name__)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPOSITORY_ROOT / ".env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PREFERRED_GROQ_MODEL = (
    os.getenv("GROQ_MODEL", "").strip() or "llama-3.3-70b-versatile"
)


class TTLCache:
    def __init__(self):
        self._values = {}
        self._lock = Lock()

    def get(self, key):
        with self._lock:
            item = self._values.get(key)
            if item is None:
                return None
            expires_at, value = item
            if monotonic() >= expires_at:
                del self._values[key]
                return None
            return value

    def set(self, key, value, ttl_seconds):
        with self._lock:
            self._values[key] = (monotonic() + ttl_seconds, value)


_cache = TTLCache()
_stock_model_lock = Lock()
_groq_model_limiter = GroqRequestLimiter(min_interval_seconds=60)
_groq_sentiment_limiter = GroqRequestLimiter()
NSE_FII_DII_URL = "https://www.nseindia.com/api/fiidiiTradeReact"


def normalize_and_validate_ticker(raw_ticker: str) -> str:
    ticker = normalize_ticker_symbol(raw_ticker)
    if not ticker or len(ticker) > 20 or not re.fullmatch(r"[A-Z0-9^.-]+", ticker):
        raise ValueError("Enter a valid ticker symbol.")
    return ticker


def _fetch_price_data(ticker: str, period: str) -> pd.DataFrame:
    cache_key = ("prices", ticker, period)
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached.copy()

    data = yf.download(
        ticker,
        period=period,
        interval=LIVE_DATA_INTERVAL,
        progress=False,
        auto_adjust=True,
        threads=False,
        timeout=20,
    )
    if data is None or data.empty:
        raise LookupError(f"No {LIVE_DATA_INTERVAL} price data was returned.")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.columns = [str(column).strip().capitalize() for column in data.columns]
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Market data is missing required columns: {', '.join(missing)}")
    data = data[required].dropna()
    data = data[~data.index.duplicated(keep="last")].sort_index()
    data["Ticker"] = ticker
    data.index = pd.to_datetime(data.index)
    if data.empty:
        raise LookupError("No complete market-data bars were returned.")
    _cache.set(cache_key, data.copy(), LIVE_DATA_CACHE_SECONDS)
    return data


def _fetch_watchlist_quote(ticker: str) -> dict:
    cache_key = ("watchlist-quote", ticker)
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        data = yf.download(
            ticker,
            period="5d",
            interval=WATCHLIST_QUOTE_INTERVAL,
            progress=False,
            auto_adjust=True,
            threads=False,
            timeout=20,
        )
        if data is None or data.empty:
            raise LookupError("No current quote data was returned.")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        close_column = next(
            (column for column in data.columns if str(column).lower() == "close"),
            None,
        )
        if close_column is None:
            raise ValueError("Quote data did not contain a close price.")
        closes = pd.to_numeric(data[close_column], errors="coerce").dropna()
        closes = closes[~closes.index.duplicated(keep="last")].sort_index()
        if closes.empty:
            raise LookupError("No complete quote data was returned.")

        session_dates = pd.Index(closes.index.date).unique()
        previous_close = None
        if len(session_dates) > 1:
            previous_session = closes.loc[
                pd.Index(closes.index.date) == session_dates[-2]
            ]
            if not previous_session.empty:
                previous_close = float(previous_session.iloc[-1])
        latest = float(closes.iloc[-1])
        result = {
            "status": "ok",
            "current_price": latest,
            "day_change_pct": (
                (latest / previous_close - 1) * 100
                if previous_close not in (None, 0)
                else None
            ),
            "as_of": _as_utc_timestamp(closes.index[-1]).isoformat(),
            "message": None,
        }
    except Exception as error:
        logger.warning(
            "Watchlist quote lookup failed for %s (%s)",
            ticker,
            type(error).__name__,
        )
        result = {
            "status": "unavailable",
            "current_price": None,
            "day_change_pct": None,
            "as_of": None,
            "message": "Current market prices are temporarily unavailable.",
        }
    _cache.set(cache_key, result, WATCHLIST_QUOTE_CACHE_SECONDS)
    return result


def _fetch_market_instrument(instrument: dict) -> dict:
    ticker = instrument["ticker"]
    public_instrument = {
        key: value for key, value in instrument.items() if key != "ticker"
    }
    cache_key = ("market-overview", ticker)
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        data = yf.download(
            ticker,
            period="6mo",
            interval="1d",
            progress=False,
            auto_adjust=True,
            threads=False,
            timeout=20,
        )
        if data is None or data.empty:
            raise LookupError("No daily price data was returned.")
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        close_column = next(
            (column for column in data.columns if str(column).lower() == "close"),
            None,
        )
        if close_column is None:
            raise ValueError("Daily price data did not contain a close price.")
        closes = pd.to_numeric(data[close_column], errors="coerce").dropna()
        closes = closes[~closes.index.duplicated(keep="last")].sort_index()
        if closes.empty:
            raise LookupError("No complete daily close prices were returned.")
        latest = float(closes.iloc[-1])
        previous = float(closes.iloc[-2]) if len(closes) > 1 else None
        history = [
            {
                "t": pd.Timestamp(timestamp).date().isoformat(),
                "value": float(value),
            }
            for timestamp, value in closes.tail(90).items()
        ]
        result = {
            **public_instrument,
            "status": "available",
            "price": latest,
            "change_pct": (
                (latest / previous - 1) * 100
                if previous not in (None, 0)
                else None
            ),
            "as_of": history[-1]["t"],
            "history": history,
            "message": None,
        }
    except Exception as error:
        logger.warning(
            "Market overview lookup failed for %s (%s)",
            ticker,
            type(error).__name__,
        )
        result = {
            **public_instrument,
            "status": "unavailable",
            "price": None,
            "change_pct": None,
            "as_of": None,
            "history": [],
            "message": "Market prices are temporarily unavailable.",
        }
    _cache.set(cache_key, result, MARKET_DATA_CACHE_SECONDS)
    return result


def build_market_overview() -> dict:
    with ThreadPoolExecutor(max_workers=len(MARKET_OVERVIEW_INSTRUMENTS)) as executor:
        instruments = list(executor.map(
            _fetch_market_instrument,
            MARKET_OVERVIEW_INSTRUMENTS,
        ))
    return {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source": "Market data",
        "instruments": instruments,
    }


def _unavailable_fii_dii(message: str) -> dict:
    return {
        "status": "unavailable",
        "lean": None,
        "as_of": None,
        "stale": True,
        "days": [],
        "message": message,
        "source_url": NSE_FII_DII_URL,
    }


def _fetch_fii_dii_flows() -> dict:
    cache_key = ("fii-dii-flows",)
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        response = httpx.get(
            NSE_FII_DII_URL,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/130.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json, text/plain, */*",
                "Referer": "https://www.nseindia.com/reports/fii-dii",
            },
            timeout=12,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError("NSE returned an unexpected FII/DII response.")

        latest = {}
        for row in payload:
            if not isinstance(row, dict):
                continue
            category = str(row.get("category", "")).strip().upper()
            category_key = "fii" if category.startswith("FII") else (
                "dii" if category == "DII" else None
            )
            if category_key is None:
                continue
            report_date = datetime.strptime(
                str(row["date"]).strip(), "%d-%b-%Y"
            ).date()
            day = latest.setdefault(report_date, {"date": report_date.isoformat()})
            day[f"{category_key}_buy"] = float(row["buyValue"])
            day[f"{category_key}_sell"] = float(row["sellValue"])
            day[f"{category_key}_net"] = float(row["netValue"])

        if not latest:
            raise ValueError("NSE returned no FII/DII records.")
        report_date = max(latest)
        day = latest[report_date]
        age_days = (datetime.now(timezone.utc).date() - report_date).days
        stale = age_days > 4
        message = (
            "Latest NSE cash-market flows in ₹ crore. The public endpoint currently "
            "provides only the latest session."
        )
        if stale:
            message = f"NSE data is {age_days} days old. " + message
        result = {
            "status": "available",
            "lean": None,
            "as_of": report_date.isoformat(),
            "stale": stale,
            "days": [day],
            "message": message,
            "source_url": NSE_FII_DII_URL,
        }
    except (httpx.HTTPError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        logger.warning("NSE FII/DII lookup failed (%s)", type(error).__name__)
        result = _unavailable_fii_dii(
            "NSE FII/DII data is temporarily unavailable. The source is a free, "
            "undocumented public endpoint and may be rate-limited or change."
        )
        _cache.set(cache_key, result, 300)
        return result

    _cache.set(cache_key, result, FII_DII_CACHE_SECONDS)
    return result


def _fetch_fundamental_info(ticker: str) -> tuple[dict, str | None]:
    cache_key = ("fundamentals", ticker)
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    instrument = yf.Ticker(ticker)
    errors = []
    info = {}
    try:
        candidate = instrument.info
        if isinstance(candidate, dict):
            info = {key: value for key, value in candidate.items() if value is not None}
    except Exception as error:
        errors.append(type(error).__name__)
        logger.warning("Fundamental info lookup failed for %s (%s)", ticker, type(error).__name__)

    if len(info) < 4:
        try:
            fast_info = instrument.fast_info
            info.setdefault("regularMarketPrice", getattr(fast_info, "last_price", None))
            info.setdefault("marketCap", getattr(fast_info, "market_cap", None))
            info.setdefault("longName", getattr(fast_info, "long_name", None))
        except Exception as error:
            errors.append(type(error).__name__)
            logger.warning("Fast fundamental lookup failed for %s (%s)", ticker, type(error).__name__)

    message = None
    if not info:
        message = "Fundamental data is unavailable from the market-data provider."
    elif errors or len(info) < 8:
        message = "Some fundamental fields are unavailable; displayed metrics may be incomplete."
    result = (info, message)
    _cache.set(cache_key, result, FUNDAMENTALS_CACHE_SECONDS)
    return result


def _company_name(ticker: str, info: dict) -> str:
    return (
        str(info.get("longName") or info.get("shortName") or "").strip()
        or ticker.split(".")[0].replace("-", " ").title()
    )


def _json_value(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        return _json_value(value.item())
    return str(value)


def _build_fundamentals(ticker: str, info: dict, message: str | None) -> dict:
    metric_names = (
        "Forward P/E",
        "PEG Ratio",
        "Return on Equity",
        "Debt / Equity",
        "Revenue Growth (YoY)",
    )
    score, breakdown = Fundamentals().compute_fundamental_score(info)
    available_count = sum(
        value != "N/A" for value, _, _ in breakdown.values()
    )
    status = "unavailable" if available_count == 0 else "partial" if message else "available"
    lean = (
        "Bullish" if score >= 65 else "Neutral" if score >= 40 else "Bearish"
    ) if available_count else None
    metrics = []
    for name in metric_names:
        value, points, maximum = breakdown[name]
        metrics.append(
            {
                "name": name,
                "value": _json_value(value),
                "score": int(points),
                "max_score": int(maximum),
            }
        )

    target = _number(info.get("targetMeanPrice"))
    current = _number(info.get("regularMarketPrice"))
    analyst = {
        "consensus": (
            str(info.get("recommendationKey", "")).replace("_", " ").upper()
            or None
        ),
        "target": target,
        "upside_pct": (
            (target - current) / current * 100
            if target is not None and current
            else None
        ),
        "count": _integer(info.get("numberOfAnalystOpinions")),
    }
    detail_keys = (
        "sector",
        "industry",
        "marketCap",
        "trailingPE",
        "forwardPE",
        "dividendYield",
        "beta",
        "pegRatio",
        "returnOnEquity",
        "debtToEquity",
        "revenueGrowth",
    )
    details = {key: _json_value(info.get(key)) for key in detail_keys}
    return {
        "status": status,
        "lean": lean,
        "score": int(score) if available_count else None,
        "as_of": datetime.now(timezone.utc).date().isoformat() if available_count else None,
        "metrics": metrics,
        "info": details,
        "analyst": analyst,
        "message": message,
    }


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if np.isfinite(number) else None


def _integer(value):
    number = _number(value)
    return int(number) if number is not None else None


def _fetch_quarterly_results(
    ticker: str,
    currency: str | None,
) -> tuple[list[dict], str | None]:
    cache_key = ("quarterly-results", ticker)
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        statements = yf.Ticker(ticker).quarterly_financials
        if statements is None or statements.empty:
            raise LookupError("Quarterly financial statements are unavailable.")

        row_names = {
            str(index).strip().lower(): index for index in statements.index
        }

        def find_row(*candidates):
            return next(
                (
                    row_names[candidate]
                    for candidate in candidates
                    if candidate in row_names
                ),
                None,
            )

        rows = {
            "revenue": find_row("total revenue", "revenue"),
            "operating_income": find_row("operating income"),
            "net_income": find_row("net income"),
            "diluted_eps": find_row("diluted eps"),
        }
        quarter_columns = sorted(
            statements.columns,
            key=lambda value: pd.Timestamp(value),
            reverse=True,
        )[:4]
        results = []
        for column in quarter_columns:
            quarter = {
                "quarter": pd.Timestamp(column).date().isoformat(),
                "currency": currency,
            }
            for key, row in rows.items():
                quarter[key] = (
                    _number(statements.at[row, column])
                    if row is not None
                    else None
                )
            results.append(quarter)

        if not any(
            any(quarter[key] is not None for key in rows)
            for quarter in results
        ):
            raise LookupError("Quarterly financial statements contain no usable values.")
        result = (results, None)
    except Exception as error:
        logger.warning(
            "Quarterly results lookup failed for %s (%s)",
            ticker,
            type(error).__name__,
        )
        result = (
            [],
            "Quarterly results are not available from the market-data source.",
        )
    _cache.set(cache_key, result, FUNDAMENTALS_CACHE_SECONDS)
    return result


def _fetch_industry_peers(
    ticker: str,
    info: dict,
) -> tuple[list[dict], str | None]:
    industry = str(info.get("industry") or "").strip()
    if not industry:
        return [], "Industry data is unavailable, so comparable companies could not be selected."
    exchange = (
        "NSI"
        if ticker.endswith(".NS")
        else "BSE"
        if ticker.endswith(".BO")
        else str(info.get("exchange") or "").strip()
    )
    if not exchange:
        return [], "Exchange data is unavailable, so comparable companies could not be selected."

    cache_key = ("industry-peers", ticker)
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    try:
        query = yf.EquityQuery(
            "and",
            [
                yf.EquityQuery("eq", ["industry", industry]),
                yf.EquityQuery("eq", ["exchange", exchange]),
            ],
        )
        response = yf.screen(
            query,
            size=12,
            sortField="intradaymarketcap",
            sortAsc=False,
        )
        matches = response.get("quotes", [])
        candidates = []
        for match in matches:
            symbol = str(match.get("symbol") or "").strip().upper()
            if not symbol or symbol == ticker:
                continue
            candidates.append((symbol, match))
        candidates = candidates[:3]

        def build_peer(candidate):
            symbol, match = candidate
            peer_info, _ = _fetch_fundamental_info(symbol)
            peer_industry = str(peer_info.get("industry") or "").strip()
            if peer_industry and peer_industry.casefold() != industry.casefold():
                return None
            return {
                "ticker": symbol,
                "currency": peer_info.get("currency"),
                "company_name": _company_name(
                    symbol,
                    {
                        **peer_info,
                        "longName": (
                            peer_info.get("longName")
                            or match.get("longname")
                            or match.get("shortname")
                        ),
                    },
                ),
                "market_cap": _number(peer_info.get("marketCap")),
                "trailing_pe": _number(peer_info.get("trailingPE")),
                "revenue_growth": _number(peer_info.get("revenueGrowth")),
                "return_on_equity": _number(peer_info.get("returnOnEquity")),
            }

        if candidates:
            with ThreadPoolExecutor(max_workers=min(3, len(candidates))) as executor:
                peers = [
                    peer for peer in executor.map(build_peer, candidates)
                    if peer is not None
                ]
        else:
            peers = []
        result = (
            peers,
            None if peers else "No comparable companies were available for this industry.",
        )
    except Exception as error:
        logger.warning(
            "Industry peer lookup failed for %s (%s)",
            ticker,
            type(error).__name__,
        )
        result = (
            [],
            "Comparable company data is temporarily unavailable.",
        )
    _cache.set(cache_key, result, FUNDAMENTALS_CACHE_SECONDS)
    return result


def _as_utc_timestamp(value) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _parse_headlines(rss_content: str) -> list[dict]:
    try:
        root = ET.fromstring(rss_content)
    except ET.ParseError as error:
        raise ValueError("The news provider returned malformed RSS.") from error

    seen = set()
    headlines = []
    for item in root.findall(".//item"):
        title = (item.findtext("title") or "").strip()
        if not title or title.casefold() in seen:
            continue
        seen.add(title.casefold())
        source = (item.findtext("source") or "").strip()
        link = (item.findtext("link") or "").strip()
        if not source:
            source = urlparse(link).hostname or "Google News"
            source = source.removeprefix("www.")
        published = (item.findtext("pubDate") or "").strip()
        date = None
        if published:
            try:
                date = parsedate_to_datetime(published).isoformat()
            except (TypeError, ValueError, OverflowError):
                date = None
        headlines.append(
            {"title": unescape(title), "source": source, "date": date}
        )
        if len(headlines) == 10:
            break
    return headlines


def _fetch_news(ticker: str, company_name: str) -> list[dict]:
    cache_key = ("news", ticker)
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached
    query = quote(f"{company_name} stock NSE")
    url = (
        "https://news.google.com/rss/search?"
        f"q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    )
    rss = fetch_news_rss(url)
    headlines = _parse_headlines(rss)
    _cache.set(cache_key, headlines, NEWS_CACHE_SECONDS)
    return headlines


def _available_groq_model() -> str:
    cache_key = ("groq_models", "available")
    cached = _cache.get(cache_key)
    if cached is not None:
        model = select_available_model(cached, PREFERRED_GROQ_MODEL)
        if model:
            return model
        raise ValueError("No supported Groq model is available to this key.")

    _groq_model_limiter.acquire()
    try:
        available = [model.id for model in Groq(api_key=GROQ_API_KEY, max_retries=0).models.list().data]
    except Exception as error:
        if getattr(error, "status_code", None) == 429:
            retry_after = _groq_model_limiter.defer_after_rate_limit(
                rate_limit_retry_after(error)
            )
            raise GroqCooldownError(retry_after) from error
        raise
    _cache.set(cache_key, available, 86400)
    model = select_available_model(available, PREFERRED_GROQ_MODEL)
    if model is None:
        raise ValueError("No supported Groq model is available to this key.")
    return model


def _analyze_news(
    ticker: str,
    company_name: str,
    headlines: list[dict],
    prediction: dict,
    trend: dict,
    flows: dict,
    fundamentals: dict,
) -> dict:
    if not GROQ_API_KEY:
        return _unavailable_ai(headlines, "AI news read is unavailable: GROQ_API_KEY is not configured.")
    if not headlines:
        return _unavailable_ai(headlines, "No current news headlines were found.")

    context = {
        "prediction": prediction,
        "trend": trend,
        "flows": flows,
        "fundamentals": {
            key: fundamentals.get(key)
            for key in ("status", "lean", "score", "as_of")
        },
        "headlines": headlines,
    }
    cache_key = ("sentiment", ticker, json.dumps(context, sort_keys=True, default=str))
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached

    model = None
    try:
        model = _available_groq_model()
        _groq_sentiment_limiter.acquire()
        prompt = (
            "Write a short supporting market read for the given stock. You are an "
            "explainer only: never change or recommend changing the model prediction, "
            "trend/flow/fundamental leans, or alignment. Use only supplied numbers. "
            "If evidence is unavailable, say so. Keep summary to at most 3 short "
            "sentences and risks to at most 3 items. Mood must be Bullish, Neutral, "
            "or Bearish. Return JSON only with keys mood, summary, risks.\n\n"
            f"Company: {company_name} ({ticker})\n"
            f"Evidence JSON: {json.dumps(context, ensure_ascii=True, default=str)}"
        )
        response = Groq(api_key=GROQ_API_KEY, max_retries=0).chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            max_tokens=512,
            temperature=0.1,
        )
    except Exception as error:
        if getattr(error, "status_code", None) == 429:
            retry_after = _groq_sentiment_limiter.defer_after_rate_limit(
                rate_limit_retry_after(error)
            )
            error = GroqCooldownError(retry_after)
        body = getattr(error, "body", None)
        provider_error = body.get("error") if isinstance(body, dict) else None
        provider_code = (
            provider_error.get("code")
            if isinstance(provider_error, dict)
            else None
        )
        provider_message = (
            provider_error.get("message")
            if isinstance(provider_error, dict)
            else None
        )
        if isinstance(provider_message, str):
            provider_message = re.sub(
                r"gsk_[A-Za-z0-9_-]+",
                "[REDACTED]",
                provider_message,
            )[:300]
        else:
            provider_message = None
        logger.warning(
            "Groq sentiment request failed (model=%s, status=%s, code=%s, "
            "provider_message=%s, error_type=%s)",
            model or "unselected",
            getattr(error, "status_code", None),
            provider_code,
            provider_message,
            type(error).__name__,
        )
        return _unavailable_ai(headlines, groq_error_message(error))

    if not response.choices:
        return _unavailable_ai(headlines, "AI news read is unavailable: the provider returned no response.")
    try:
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise GroqInvalidResponseError("The AI read was truncated.")
        read = json.loads(choice.message.content)
        if not isinstance(read, dict):
            raise GroqInvalidResponseError("The AI read was not a JSON object.")
        mood = str(read.get("mood", "")).title()
        summary = read.get("summary")
        risks = read.get("risks")
        if (
            mood not in {"Bullish", "Neutral", "Bearish"}
            or not isinstance(summary, str)
            or not isinstance(risks, list)
            or any(not isinstance(item, str) for item in risks)
        ):
            raise GroqInvalidResponseError("The AI read did not match its required schema.")
        result = {
            "status": "ok",
            "mood": mood,
            "summary": summary,
            "risks": risks[:3],
            "headlines": headlines,
            "model_used": model,
        }
    except (GroqInvalidResponseError, json.JSONDecodeError, TypeError) as error:
        result = _unavailable_ai(headlines, groq_error_message(error))
    if result["status"] == "ok":
        _cache.set(cache_key, result, SENTIMENT_CACHE_SECONDS)
    return result


def _unavailable_ai(headlines: list[dict], message: str) -> dict:
    return {
        "status": "unavailable",
        "mood": None,
        "summary": "AI read unavailable.",
        "risks": [],
        "headlines": headlines,
        "model_used": None,
        "message": message,
    }


def _trend_evidence(data: pd.DataFrame) -> dict:
    latest = data.iloc[-1]
    adx = _number(latest.get("adx"))
    di_plus = _number(latest.get("di_plus"))
    di_minus = _number(latest.get("di_minus"))
    if adx is None or di_plus is None or di_minus is None:
        lean = None
        trending = None
        strength = None
    else:
        trending = adx >= ADX_TREND_THRESHOLD
        lean = (
            "Neutral"
            if not trending
            else "Bullish"
            if di_plus > di_minus
            else "Bearish"
        )
        strength = (
            "Very Strong"
            if adx >= 40
            else "Strong"
            if adx >= ADX_TREND_THRESHOLD
            else "Weak"
            if adx >= 20
            else "No Trend"
        )
    series = []
    for timestamp, row in data.tail(100).iterrows():
        series.append(
            {
                "t": _as_utc_timestamp(timestamp).isoformat(),
                "adx": _number(row.get("adx")),
                "di_plus": _number(row.get("di_plus")),
                "di_minus": _number(row.get("di_minus")),
            }
        )
    return {
        "status": "available" if lean is not None else "unavailable",
        "lean": lean,
        "adx": adx,
        "di_plus": di_plus,
        "di_minus": di_minus,
        "trending": trending,
        "strength": strength,
        "as_of": _as_utc_timestamp(data.index[-1]).isoformat(),
        "series": series,
    }


def _alignment(action: str, leans: list[str | None]) -> dict | None:
    if action == "HOLD":
        return None
    direction = "Bullish" if action == "BUY" else "Bearish"
    available = [lean for lean in leans if lean is not None]
    return {
        "agree": sum(lean == direction for lean in available),
        "against": sum(lean in {"Bullish", "Bearish"} and lean != direction for lean in available),
        "neutral": sum(lean == "Neutral" for lean in available),
    }


def _candles(data: pd.DataFrame, ticker: str) -> list[dict]:
    filtered = data
    if ticker.endswith((".NS", ".BO")):
        local = data.index.tz_localize("UTC") if data.index.tz is None else data.index
        local = local.tz_convert("Asia/Kolkata")
        minutes = local.hour * 60 + local.minute
        market_mask = (
            (minutes >= 9 * 60 + 15)
            & (minutes <= 15 * 60 + 30)
            & (local.dayofweek < 5)
        )
        market_data = data.loc[market_mask]
        if len(market_data) > 10:
            filtered = market_data
    return [
        {
            "t": _as_utc_timestamp(timestamp).isoformat(),
            "o": float(row["Open"]),
            "h": float(row["High"]),
            "l": float(row["Low"]),
            "c": float(row["Close"]),
            "v": float(row["Volume"]),
        }
        for timestamp, row in filtered.tail(120).iterrows()
    ]


def _fit_stock_model(ticker: str, data: pd.DataFrame) -> dict:
    features = StockFeatureEngineer(data).create_all_features()
    targets = StockTargetEngineer.create_forward_return_target(
        data,
        forward_bars=FORWARD_BARS,
        threshold=TARGET_THRESHOLD,
    )
    future_returns = (
        data["Close"].shift(-FORWARD_BARS) / data["Close"] - 1
    ).rename("forward_return")
    training = features.replace([np.inf, -np.inf], np.nan).join(targets).join(
        future_returns
    ).dropna()
    training = training[training["target"] != 0]
    if len(training) < 250:
        raise LookupError(
            "There is not enough stock-specific history to train a reliable model."
        )

    class_counts = training["target"].value_counts()
    if len(class_counts) < 2 or class_counts.min() < 10:
        raise LookupError(
            "This stock's history does not contain enough examples for each prediction class."
        )

    feature_names = features.columns.tolist()
    x_train = training[feature_names]
    y_train = training["target"].astype(int)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train)
    model = lgb.LGBMClassifier(**STOCK_MODEL_PARAMS)
    model.fit(x_scaled, y_train)

    lower, upper = training["forward_return"].quantile([0.01, 0.99])
    clipped_returns = training["forward_return"].clip(lower, upper)
    class_returns = {
        int(label): float(clipped_returns[y_train == label].mean())
        for label in model.classes_
    }
    logger.info(
        "Trained stock-specific model for %s on %d labeled bars",
        ticker,
        len(training),
    )
    return {
        "model": model,
        "scaler": scaler,
        "feature_names": feature_names,
        "class_returns": class_returns,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": len(training),
    }


def _load_stock_model(ticker: str, period: str, data: pd.DataFrame) -> dict:
    cache_key = ("stock-model", ticker, period)
    cached = _cache.get(cache_key)
    if cached is not None:
        return cached
    with _stock_model_lock:
        cached = _cache.get(cache_key)
        if cached is not None:
            return cached
        fitted = _fit_stock_model(ticker, data)
        _cache.set(cache_key, fitted, STOCK_MODEL_CACHE_SECONDS)
        return fitted


def _predict_latest_stock_model(model_info: dict, data: pd.DataFrame) -> dict:
    features = StockFeatureEngineer(data).create_all_features()
    latest = features.replace([np.inf, -np.inf], np.nan).dropna().tail(1)
    if latest.empty:
        raise LookupError("There are not enough complete bars to calculate a prediction.")

    model = model_info["model"]
    scaled = model_info["scaler"].transform(
        latest[model_info["feature_names"]]
    )
    probabilities = model.predict_proba(scaled)[0]
    predicted_class = int(model.classes_[int(np.argmax(probabilities))])
    confidence = float(np.max(probabilities))
    effective_class = predicted_class if confidence >= MIN_CONFIDENCE else 0
    signal = "BUY" if effective_class > 0 else "SELL" if effective_class < 0 else "HOLD"
    expected_return = sum(
        float(probability) * model_info["class_returns"][int(label)]
        for label, probability in zip(model.classes_, probabilities)
    )
    return {
        "signal": signal,
        "confidence": confidence,
        "strength": "STRONG" if confidence >= 0.70 else "MODERATE" if confidence >= MIN_CONFIDENCE else "WEAK",
        "expected_return": expected_return,
        "price": float(data["Close"].iloc[-1]),
        "model": {
            "name": "Stock-specific history model",
            "trained_at": model_info["trained_at"],
            "training_samples": model_info["n_samples"],
        },
    }


def build_stock_snapshot(
    raw_ticker: str,
    period: str = LIVE_DATA_PERIOD,
    include_ai: bool = True,
) -> dict:
    ticker = normalize_and_validate_ticker(raw_ticker)
    if period not in {"30d", "45d", "60d"}:
        raise ValueError("Period must be one of 30d, 45d, or 60d.")
    try:
        data = _fetch_price_data(ticker, period)
    except LookupError:
        raise
    except Exception as error:
        logger.exception("Price data request failed for %s (%s)", ticker, type(error).__name__)
        raise RuntimeError("Market data could not be loaded. Try again later.") from error

    try:
        model_info = _load_stock_model(ticker, period, data)
        with_indicators = Indicators.add_all(data)
        latest_prediction = _predict_latest_stock_model(model_info, data)
    except LookupError as error:
        raise RuntimeError(str(error)) from error
    except Exception as error:
        logger.exception("Prediction service failed for %s (%s)", ticker, type(error).__name__)
        raise RuntimeError(
            "The stock-specific prediction model could not be trained. Try again later."
        ) from error

    trend = _trend_evidence(with_indicators)
    try:
        info, fundamentals_message = _fetch_fundamental_info(ticker)
        fundamentals = _build_fundamentals(ticker, info, fundamentals_message)
    except Exception as error:
        logger.exception("Fundamentals processing failed for %s (%s)", ticker, type(error).__name__)
        info = {}
        fundamentals = _build_fundamentals(
            ticker, {}, "Fundamental metrics could not be computed."
        )
    with ThreadPoolExecutor(max_workers=2) as executor:
        quarterly_future = executor.submit(
            _fetch_quarterly_results,
            ticker,
            info.get("financialCurrency") or info.get("currency"),
        )
        peers_future = executor.submit(_fetch_industry_peers, ticker, info)
        quarterly_results, quarterly_results_message = quarterly_future.result()
        peer_comparisons, peer_comparisons_message = peers_future.result()
    company_name = _company_name(ticker, info)
    flows = _fetch_fii_dii_flows()
    action = latest_prediction["signal"]
    confidence = float(latest_prediction["confidence"])
    prediction = {
        "action": action,
        "confidence": confidence,
        "strength": latest_prediction["strength"],
        "expected_return_pct": float(latest_prediction["expected_return"]) * 100,
        "model": latest_prediction["model"],
    }
    alignment = _alignment(
        action,
        [trend["lean"], flows["lean"], fundamentals["lean"]],
    )
    headlines = []
    if include_ai:
        try:
            headlines = _fetch_news(ticker, company_name)
            ai_read = _analyze_news(
                ticker,
                company_name,
                headlines,
                prediction,
                trend,
                flows,
                fundamentals,
            )
        except NewsFetchTimeoutError as error:
            ai_read = _unavailable_ai(
                headlines,
                f"News request timed out after {error.attempts} attempts.",
            )
        except Exception as error:
            logger.exception("News analysis failed for %s (%s)", ticker, type(error).__name__)
            ai_read = _unavailable_ai(
                headlines,
                "News analysis is unavailable. The prediction and evidence are still shown.",
            )
    else:
        ai_read = _unavailable_ai(headlines, "News analysis is not loaded in the watchlist.")

    response = {
        "ticker": ticker,
        "company_name": company_name,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "period": period,
        "price": float(latest_prediction["price"]),
        "prediction": prediction,
        "candles_15m": _candles(data, ticker),
        "evidence": {
            "trend": trend,
            "flows": flows,
            "fundamentals": fundamentals,
        },
        "alignment": alignment,
        "ai_read": ai_read,
        "quarterly_results": quarterly_results,
        "quarterly_results_message": quarterly_results_message,
        "peer_comparisons": peer_comparisons,
        "peer_comparisons_message": peer_comparisons_message,
    }
    return _json_value(response)


def build_watchlist(tickers: list[str] | None = None) -> dict:
    requested_tickers = DEFAULT_WATCHLIST if tickers is None else tickers

    def build_row(ticker: str) -> dict:
        quote = _fetch_watchlist_quote(ticker)
        try:
            info, _ = _fetch_fundamental_info(ticker)
        except Exception as error:
            logger.warning(
                "Watchlist company lookup failed for %s (%s)",
                ticker,
                type(error).__name__,
            )
            info = {}
        return {
            "ticker": ticker,
            "company_name": _company_name(ticker, info),
            "currency": info.get(
                "currency",
                "INR" if ticker.endswith((".NS", ".BO")) else None,
            ),
            **quote,
        }

    if requested_tickers:
        with ThreadPoolExecutor(max_workers=min(5, len(requested_tickers))) as executor:
            rows = list(executor.map(build_row, requested_tickers))
    else:
        rows = []

    return {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
        "flows": _fetch_fii_dii_flows(),
    }
