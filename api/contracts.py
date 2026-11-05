from typing import Literal

from pydantic import BaseModel, Field


Lean = Literal["Bullish", "Neutral", "Bearish"]


class Prediction(BaseModel):
    action: Literal["BUY", "HOLD", "SELL"]
    confidence: float
    strength: str
    expected_return_pct: float
    model: dict


class Candle(BaseModel):
    t: str
    o: float
    h: float
    l: float
    c: float
    v: float


class TrendEvidence(BaseModel):
    status: Literal["available", "unavailable"]
    lean: Lean | None
    adx: float | None
    di_plus: float | None
    di_minus: float | None
    trending: bool | None
    strength: str | None
    as_of: str | None
    series: list[dict]


class FlowsEvidence(BaseModel):
    status: Literal["available", "unavailable"]
    lean: Lean | None = None
    as_of: str | None = None
    stale: bool = False
    days: list[dict] = Field(default_factory=list)
    message: str
    source_url: str | None = None


class FundamentalMetric(BaseModel):
    name: str
    value: str | float | None
    score: int
    max_score: int


class AnalystView(BaseModel):
    consensus: str | None
    target: float | None
    upside_pct: float | None
    count: int | None


class FundamentalsEvidence(BaseModel):
    status: Literal["available", "partial", "unavailable"]
    lean: Lean | None
    score: int | None
    as_of: str | None
    metrics: list[FundamentalMetric]
    info: dict
    analyst: AnalystView
    message: str | None = None


class StockEvidence(BaseModel):
    trend: TrendEvidence
    flows: FlowsEvidence
    fundamentals: FundamentalsEvidence


class Headline(BaseModel):
    title: str
    source: str
    date: str | None


class AIRead(BaseModel):
    status: Literal["ok", "unavailable"]
    mood: Lean | None
    summary: str
    risks: list[str]
    headlines: list[Headline]
    model_used: str | None
    message: str | None = None


class Alignment(BaseModel):
    agree: int
    against: int
    neutral: int


class QuarterlyResult(BaseModel):
    quarter: str
    currency: str | None = None
    revenue: float | None = None
    operating_income: float | None = None
    net_income: float | None = None
    diluted_eps: float | None = None


class PeerComparison(BaseModel):
    ticker: str
    company_name: str
    currency: str | None = None
    market_cap: float | None = None
    trailing_pe: float | None = None
    revenue_growth: float | None = None
    return_on_equity: float | None = None


class StockSnapshot(BaseModel):
    ticker: str
    company_name: str
    updated_at: str
    period: str
    price: float
    prediction: Prediction
    candles_15m: list[Candle]
    evidence: StockEvidence
    alignment: Alignment | None
    ai_read: AIRead
    quarterly_results: list[QuarterlyResult] = Field(default_factory=list)
    quarterly_results_message: str | None = None
    peer_comparisons: list[PeerComparison] = Field(default_factory=list)
    peer_comparisons_message: str | None = None


class WatchlistRow(BaseModel):
    ticker: str
    company_name: str
    status: Literal["ok", "unavailable"]
    current_price: float | None = None
    day_change_pct: float | None = None
    currency: str | None = None
    as_of: str | None = None
    message: str | None = None


class WatchlistResponse(BaseModel):
    updated_at: str
    rows: list[WatchlistRow]
    flows: FlowsEvidence | None = None


class MarketInstrument(BaseModel):
    key: str
    label: str
    asset_class: str
    currency: str
    unit: str
    status: Literal["available", "unavailable"]
    price: float | None = None
    change_pct: float | None = None
    as_of: str | None = None
    history: list[dict] = Field(default_factory=list)
    message: str | None = None


class MarketOverview(BaseModel):
    updated_at: str
    source: str
    instruments: list[MarketInstrument]
