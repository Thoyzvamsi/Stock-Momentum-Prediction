from pathlib import Path
import logging

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

from api.contracts import MarketOverview, StockSnapshot, WatchlistResponse
from api.services import (
    build_market_overview,
    build_stock_snapshot,
    build_watchlist,
    normalize_and_validate_ticker,
)


logging.basicConfig(level=logging.INFO)
PUBLIC_DIR = Path(__file__).resolve().parents[1] / "public"
ASSETS_DIR = PUBLIC_DIR / "assets"
INDEX_FILE = PUBLIC_DIR / "index.html"
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Stock Momentum Prediction API",
    version="1.0.0",
    description="API for stock predictions and supporting market evidence.",
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/market", response_model=MarketOverview)
def market():
    try:
        return build_market_overview()
    except Exception as error:
        logger.exception(
            "Market overview request failed (%s)", type(error).__name__
        )
        raise HTTPException(
            status_code=502,
            detail="The market overview could not be loaded. Try again later.",
        ) from error


@app.get("/api/index", response_model=WatchlistResponse)
def index(tickers: str | None = Query(default=None, max_length=500)):
    try:
        selected_tickers = None
        if tickers is not None:
            selected_tickers = []
            if tickers:
                raw_tickers = [item.strip() for item in tickers.split(",")]
                if any(not item for item in raw_tickers):
                    raise ValueError("Enter valid ticker symbols separated by commas.")
                if len(raw_tickers) > 20:
                    raise ValueError("A watchlist can contain at most 20 symbols.")
                seen_tickers = set()
                for raw_ticker in raw_tickers:
                    normalized = normalize_and_validate_ticker(raw_ticker)
                    if normalized not in seen_tickers:
                        selected_tickers.append(normalized)
                        seen_tickers.add(normalized)
        return build_watchlist(selected_tickers)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        logger.exception(
            "Watchlist request failed (%s)", type(error).__name__
        )
        raise HTTPException(
            status_code=502,
            detail="The watchlist could not be loaded. Try again later.",
        ) from error


@app.get("/api/stock/{raw_ticker}", response_model=StockSnapshot)
def stock(
    raw_ticker: str,
    period: str = Query(default="60d", pattern="^(30d|45d|60d)$"),
):
    try:
        ticker = normalize_and_validate_ticker(raw_ticker)
        return build_stock_snapshot(ticker, period=period)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except LookupError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error


@app.get("/", include_in_schema=False)
@app.get("/analyzer", include_in_schema=False)
@app.get("/analyzer/{raw_ticker}", include_in_schema=False)
@app.get("/stock/{raw_ticker}", include_in_schema=False)
def web_app(raw_ticker: str | None = None):
    if not INDEX_FILE.is_file():
        raise HTTPException(status_code=500, detail="Web client files are missing.")
    return FileResponse(INDEX_FILE)


@app.get("/assets/{asset_path:path}", include_in_schema=False)
def web_asset(asset_path: str):
    assets_root = ASSETS_DIR.resolve()
    asset_file = (assets_root / asset_path).resolve()
    if not asset_file.is_relative_to(assets_root) or not asset_file.is_file():
        raise HTTPException(status_code=404, detail="Web asset not found.")
    return FileResponse(asset_file)
