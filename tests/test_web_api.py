from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from api.main import app
from main import app as root_app


client = TestClient(app)


def test_root_asgi_entrypoint_exports_fastapi_app():
    assert root_app is app
    assert isinstance(root_app, FastAPI)


def test_health_endpoint():
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_market_route_validates_market_overview_contract(monkeypatch):
    monkeypatch.setattr(
        "api.main.build_market_overview",
        lambda: {
            "updated_at": "2026-10-26T10:00:00+00:00",
            "source": "Market data",
            "instruments": [
                {
                    "key": "nifty",
                    "label": "NIFTY 50",
                    "asset_class": "India index",
                    "currency": "INR",
                    "unit": "index points",
                    "status": "available",
                    "price": 25000.0,
                    "change_pct": 0.5,
                    "as_of": "2026-10-26",
                    "history": [{"t": "2026-10-26", "value": 25000.0}],
                }
            ],
        },
    )

    response = client.get("/api/market")

    assert response.status_code == 200
    assert response.json()["instruments"][0]["label"] == "NIFTY 50"
    assert "ticker" not in response.json()["instruments"][0]
    assert response.json()["instruments"][0]["history"] == [
        {"t": "2026-10-26", "value": 25000.0}
    ]


def test_market_instrument_uses_daily_closes_and_calculates_change(monkeypatch):
    import pandas as pd
    from api.services import _fetch_market_instrument

    monkeypatch.setattr("api.services._cache.get", lambda key: None)
    monkeypatch.setattr("api.services._cache.set", lambda key, value, ttl: None)
    monkeypatch.setattr(
        "api.services.yf.download",
        lambda *args, **kwargs: pd.DataFrame(
            {"Close": [100.0, 105.0]},
            index=pd.to_datetime(["2026-10-23", "2026-10-26"]),
        ),
    )

    result = _fetch_market_instrument(
        {
            "key": "nifty",
            "label": "NIFTY 50",
            "ticker": "^NSEI",
            "asset_class": "India index",
            "currency": "INR",
            "unit": "index points",
        }
    )

    assert result["status"] == "available"
    assert "ticker" not in result
    assert result["price"] == 105.0
    assert result["change_pct"] == pytest.approx(5.0)
    assert result["history"][-1] == {"t": "2026-10-26", "value": 105.0}


def test_watchlist_quote_uses_latest_intraday_price_and_previous_close(monkeypatch):
    import pandas as pd
    from api.services import _fetch_watchlist_quote

    captured = {}
    monkeypatch.setattr("api.services._cache.get", lambda key: None)
    monkeypatch.setattr("api.services._cache.set", lambda *args: None)

    def download(*args, **kwargs):
        captured.update(kwargs)
        return pd.DataFrame(
            {"Close": [100.0, 102.0, 103.0]},
            index=pd.to_datetime(
                [
                    "2026-10-26 15:00",
                    "2026-10-27 09:30",
                    "2026-10-27 09:31",
                ]
            ),
        )

    monkeypatch.setattr("api.services.yf.download", download)
    result = _fetch_watchlist_quote("HDFCBANK.NS")

    assert captured["interval"] == "1m"
    assert result["status"] == "ok"
    assert result["current_price"] == 103.0
    assert result["day_change_pct"] == pytest.approx(3.0)
    assert result["as_of"].startswith("2026-10-27T")


def test_watchlist_uses_quotes_and_never_computes_predictions(monkeypatch):
    from api.services import build_watchlist

    monkeypatch.setattr(
        "api.services._fetch_watchlist_quote",
        lambda ticker: {
            "status": "ok",
            "current_price": 123.0,
            "day_change_pct": 1.5,
            "as_of": "2026-10-27T10:00:00+00:00",
            "message": None,
        },
    )
    monkeypatch.setattr(
        "api.services._fetch_fundamental_info",
        lambda ticker: ({"longName": "Test Company", "currency": "INR"}, None),
    )
    monkeypatch.setattr(
        "api.services._fetch_fii_dii_flows",
        lambda: {"status": "unavailable", "message": "Unavailable."},
    )

    result = build_watchlist(["TEST.NS"])

    assert result["rows"] == [
        {
            "ticker": "TEST.NS",
            "company_name": "Test Company",
            "currency": "INR",
            "status": "ok",
            "current_price": 123.0,
            "day_change_pct": 1.5,
            "as_of": "2026-10-27T10:00:00+00:00",
            "message": None,
        }
    ]
    assert "prediction" not in result["rows"][0]
    assert result["flows"]["status"] == "unavailable"


def test_stock_specific_fit_excludes_neutral_and_unlabeled_bars(monkeypatch):
    import numpy as np
    import pandas as pd
    from api.services import _fit_stock_model

    index = pd.date_range("2026-01-01", periods=360, freq="15min")
    data = pd.DataFrame(
        {
            "Open": np.arange(360, dtype=float) + 100,
            "High": np.arange(360, dtype=float) + 102,
            "Low": np.arange(360, dtype=float) + 99,
            "Close": np.arange(360, dtype=float) + 101,
            "Volume": np.arange(360, dtype=float) + 1000,
            "Ticker": "TEST.NS",
        },
        index=index,
    )

    class FakeModel:
        def __init__(self, **kwargs):
            pass

        def fit(self, features, labels):
            self.classes_ = np.sort(labels.unique())
            self.training_rows = len(labels)

    monkeypatch.setattr(
        "api.services.StockFeatureEngineer.create_all_features",
        lambda self: pd.DataFrame(
            {"feature": np.arange(len(self.df), dtype=float)},
            index=self.df.index,
        ),
    )
    targets = pd.Series(1, index=index, dtype=float, name="target")
    targets.iloc[::2] = -1
    targets.iloc[-8:] = np.nan
    monkeypatch.setattr(
        "api.services.StockTargetEngineer.create_forward_return_target",
        staticmethod(lambda *args, **kwargs: targets),
    )
    monkeypatch.setattr("api.services.lgb.LGBMClassifier", FakeModel)

    result = _fit_stock_model("TEST.NS", data)

    assert result["n_samples"] == 352
    assert result["model"].training_rows == 352
    assert set(result["class_returns"]) == {-1, 1}
    assert all(np.isfinite(value) for value in result["class_returns"].values())


def test_stock_model_cache_is_scoped_to_history_period(monkeypatch):
    from api.services import _load_stock_model

    cache = {}
    fits = []
    monkeypatch.setattr("api.services._cache.get", lambda key: cache.get(key))
    monkeypatch.setattr(
        "api.services._cache.set",
        lambda key, value, ttl: cache.__setitem__(key, value),
    )
    monkeypatch.setattr(
        "api.services._fit_stock_model",
        lambda ticker, data: fits.append((ticker, len(data))) or {"fit": len(fits)},
    )

    first = _load_stock_model("TEST.NS", "30d", [1, 2, 3])
    second = _load_stock_model("TEST.NS", "60d", [1, 2, 3])
    cached = _load_stock_model("TEST.NS", "30d", [4, 5])

    assert first == {"fit": 1}
    assert second == {"fit": 2}
    assert cached == first
    assert len(fits) == 2


def test_quarterly_results_returns_latest_four_reports(monkeypatch):
    import pandas as pd
    from api.services import _fetch_quarterly_results

    statements = pd.DataFrame(
        {
            pd.Timestamp("2026-06-30"): [1000, 120, 80, 2.5],
            pd.Timestamp("2026-03-31"): [900, 110, 70, 2.0],
            pd.Timestamp("2025-12-31"): [800, 100, 60, 1.8],
            pd.Timestamp("2025-09-30"): [700, 90, 50, 1.5],
            pd.Timestamp("2025-06-30"): [600, 80, 40, 1.0],
        },
        index=["Total Revenue", "Operating Income", "Net Income", "Diluted EPS"],
    )

    class FakeTicker:
        @property
        def quarterly_financials(self):
            return statements

    monkeypatch.setattr("api.services._cache.get", lambda key: None)
    monkeypatch.setattr("api.services._cache.set", lambda *args: None)
    monkeypatch.setattr("api.services.yf.Ticker", lambda ticker: FakeTicker())

    results, message = _fetch_quarterly_results("TEST.NS", "INR")

    assert message is None
    assert len(results) == 4
    assert results[0] == {
        "quarter": "2026-06-30",
        "currency": "INR",
        "revenue": 1000.0,
        "operating_income": 120.0,
        "net_income": 80.0,
        "diluted_eps": 2.5,
    }


def test_industry_peers_are_screened_on_same_exchange_and_industry(monkeypatch):
    from api.services import _fetch_industry_peers

    captured = {}
    monkeypatch.setattr("api.services._cache.get", lambda key: None)
    monkeypatch.setattr("api.services._cache.set", lambda *args: None)

    def screen(query, **kwargs):
        captured["query"] = query.to_dict()
        captured["options"] = kwargs
        return {
            "quotes": [
                {"symbol": "TEST.NS"},
                {"symbol": "PEER.NS", "shortName": "Peer Company"},
            ]
        }

    monkeypatch.setattr("api.services.yf.screen", screen)
    monkeypatch.setattr(
        "api.services._fetch_fundamental_info",
        lambda ticker: (
            {
                "longName": "Peer Company",
                "industry": "Aerospace & Defense",
                "currency": "INR",
                "marketCap": 500.0,
            },
            None,
        ),
    )

    peers, message = _fetch_industry_peers(
        "TEST.NS",
        {"industry": "Aerospace & Defense"},
    )

    assert message is None
    assert captured["query"]["operator"] == "AND"
    assert captured["query"]["operands"][0]["operands"] == [
        "industry",
        "Aerospace & Defense",
    ]
    assert captured["query"]["operands"][1]["operands"] == ["exchange", "NSI"]
    assert [peer["ticker"] for peer in peers] == ["PEER.NS"]
    assert peers[0]["market_cap"] == 500.0


def test_groq_request_rejection_is_logged_with_redacted_provider_details(
    monkeypatch,
    caplog,
):
    from api.services import _analyze_news

    class RejectedRequest(Exception):
        status_code = 400
        body = {
            "error": {
                "code": "invalid_request",
                "message": "Rejected request for gsk_sensitive-token",
            }
        }

    class Completions:
        def create(self, **kwargs):
            raise RejectedRequest()

    class FakeGroq:
        def __init__(self, **kwargs):
            self.chat = type(
                "Chat",
                (),
                {"completions": Completions()},
            )()

    monkeypatch.setattr("api.services.GROQ_API_KEY", "configured")
    monkeypatch.setattr("api.services._available_groq_model", lambda: "test-model")
    monkeypatch.setattr("api.services._cache.get", lambda key: None)
    monkeypatch.setattr("api.services._groq_sentiment_limiter.acquire", lambda: None)
    monkeypatch.setattr("api.services.Groq", FakeGroq)

    result = _analyze_news(
        "TEST.NS",
        "Test Company",
        [{"title": "Test news", "source": "Test", "date": None}],
        {},
        {},
        {},
        {},
    )

    assert result["status"] == "unavailable"
    assert "Groq rejected the request" in result["message"]
    assert "invalid_request" in caplog.text
    assert "[REDACTED]" in caplog.text
    assert "gsk_sensitive-token" not in caplog.text


def test_fii_dii_parser_uses_latest_nse_cash_market_values(monkeypatch):
    from datetime import datetime, timedelta, timezone
    from api.services import _fetch_fii_dii_flows

    today = datetime.now(timezone.utc).date()
    previous = today - timedelta(days=1)

    class NSEResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return [
                {
                    "category": "FII/FPI",
                    "date": previous.strftime("%d-%b-%Y"),
                    "buyValue": "900.0",
                    "sellValue": "799.75",
                    "netValue": "100.25",
                },
                {
                    "category": "DII",
                    "date": previous.strftime("%d-%b-%Y"),
                    "buyValue": "700.0",
                    "sellValue": "740.10",
                    "netValue": "-40.10",
                },
                {
                    "category": "FII/FPI",
                    "date": today.strftime("%d-%b-%Y"),
                    "buyValue": "1500.0",
                    "sellValue": "1249.25",
                    "netValue": "250.75",
                },
                {
                    "category": "DII",
                    "date": today.strftime("%d-%b-%Y"),
                    "buyValue": "1200.0",
                    "sellValue": "1169.50",
                    "netValue": "30.50",
                },
            ]

    monkeypatch.setattr("api.services._cache.get", lambda key: None)
    monkeypatch.setattr("api.services._cache.set", lambda key, value, ttl: None)
    monkeypatch.setattr("api.services.httpx.get", lambda *args, **kwargs: NSEResponse())

    result = _fetch_fii_dii_flows()

    assert result["status"] == "available"
    assert result["as_of"] == today.isoformat()
    assert result["days"] == [
        {
            "date": today.isoformat(),
            "fii_buy": 1500.0,
            "fii_sell": 1249.25,
            "fii_net": 250.75,
            "dii_buy": 1200.0,
            "dii_sell": 1169.5,
            "dii_net": 30.5,
        }
    ]
    assert result["source_url"] == "https://www.nseindia.com/api/fiidiiTradeReact"
    assert "only the latest session" in result["message"]


def test_static_client_routes():
    for route in ("/", "/analyzer", "/analyzer/HDFCBANK.NS", "/stock/HDFCBANK.NS"):
        response = client.get(route)

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert 'id="app"' in response.text


def test_static_assets_are_available_for_local_development():
    response = client.get("/assets/styles.css")

    assert response.status_code == 200
    assert "text/css" in response.headers["content-type"]
    assert ".app-shell" in response.text


def test_legacy_model_endpoint_is_removed():
    response = client.get("/api/model")

    assert response.status_code == 404


def test_stock_route_rejects_invalid_ticker():
    response = client.get("/api/stock/INVALID!")

    assert response.status_code == 400
    assert response.json()["detail"] == "Enter a valid ticker symbol."


def test_stock_route_rejects_unsupported_period():
    response = client.get("/api/stock/HDFCBANK?period=2d")

    assert response.status_code == 422


def test_stock_route_normalizes_ticker_and_validates_response(monkeypatch):
    snapshot = {
        "ticker": "HDFCBANK.NS",
        "company_name": "HDFC Bank",
        "updated_at": "2026-10-26T10:00:00+00:00",
        "period": "60d",
        "price": 100.0,
        "prediction": {
            "action": "BUY",
            "confidence": 0.7,
            "strength": "STRONG",
            "expected_return_pct": 0.7,
            "model": {"name": "test-model", "trained_at": None},
        },
        "candles_15m": [],
        "evidence": {
            "trend": {
                "status": "available",
                "lean": "Bullish",
                "adx": 30.0,
                "di_plus": 25.0,
                "di_minus": 10.0,
                "trending": True,
                "strength": "Strong",
                "as_of": "2026-10-26T10:00:00+00:00",
                "series": [],
            },
            "flows": {
                "status": "unavailable",
                "message": "No feed configured.",
            },
            "fundamentals": {
                "status": "unavailable",
                "lean": None,
                "score": None,
                "as_of": None,
                "metrics": [],
                "info": {},
                "analyst": {
                    "consensus": None,
                    "target": None,
                    "upside_pct": None,
                    "count": None,
                },
            },
        },
        "alignment": {"agree": 1, "against": 0, "neutral": 0},
        "ai_read": {
            "status": "unavailable",
            "mood": None,
            "summary": "AI read unavailable.",
            "risks": [],
            "headlines": [],
            "model_used": None,
        },
    }
    seen = {}

    def build_snapshot(ticker, period):
        seen["ticker"] = ticker
        seen["period"] = period
        return snapshot

    monkeypatch.setattr("api.main.build_stock_snapshot", build_snapshot)
    response = client.get("/api/stock/HDFCBANK")

    assert response.status_code == 200
    assert response.json()["ticker"] == "HDFCBANK.NS"
    assert seen == {"ticker": "HDFCBANK.NS", "period": "60d"}


def test_index_route_accepts_custom_watchlist(monkeypatch):
    captured = {}

    def build_watchlist(tickers):
        captured["tickers"] = tickers
        return {"updated_at": "2026-10-26T00:00:00+00:00", "rows": []}

    monkeypatch.setattr("api.main.build_watchlist", build_watchlist)

    response = client.get("/api/index?tickers=HDFCBANK,ITC.NS,HDFCBANK.NS")

    assert response.status_code == 200
    assert captured["tickers"] == ["HDFCBANK.NS", "ITC.NS"]


def test_index_route_accepts_empty_custom_watchlist(monkeypatch):
    monkeypatch.setattr(
        "api.main.build_watchlist",
        lambda tickers: {"updated_at": "2026-10-26T00:00:00+00:00", "rows": []},
    )

    response = client.get("/api/index?tickers=")

    assert response.status_code == 200
    assert response.json()["rows"] == []


def test_index_route_rejects_watchlist_over_limit():
    tickers = ",".join(f"STOCK{index}.NS" for index in range(21))

    response = client.get(f"/api/index?tickers={tickers}")

    assert response.status_code == 400
    assert response.json()["detail"] == "A watchlist can contain at most 20 symbols."
