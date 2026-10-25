def normalize_ticker_symbol(symbol: str) -> str:
    """Default bare ticker symbols to Yahoo Finance's NSE suffix."""
    normalized = symbol.strip().upper()
    if normalized and "." not in normalized and not normalized.startswith("^"):
        return f"{normalized}.NS"
    return normalized
