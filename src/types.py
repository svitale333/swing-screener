from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass
class ScreenedCandidate:
    ticker: str
    price: float
    avg_volume: float
    market_cap: float
    sector: str
    industry: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TechnicalSetup:
    ticker: str
    setup_type: str          # "bull_flag", "squeeze_breakout", "mean_reversion", "trend_pullback"
    direction: str           # "long" or "short"
    technical_score: float   # 1-10
    support_level: float
    resistance_level: float
    key_levels: dict = field(default_factory=dict)
    indicators: dict = field(default_factory=dict)
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class SentimentResult:
    ticker: str
    sentiment: str           # "bullish", "neutral", "bearish"
    confidence: float        # 1-10
    catalysts: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    earnings_date: str | None = None
    summary: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TradePlay:
    ticker: str
    setup_type: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float | None
    risk_reward_ratio: float
    position_risk_pct: float
    technical_score: float
    sentiment_score: float
    composite_score: float
    sentiment_summary: str = ""
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
