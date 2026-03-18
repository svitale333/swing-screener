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
    days_to_earnings: int | None = None
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
    risk_reward_ratio_tp2: float | None
    position_risk_pct: float
    technical_score: float
    sentiment_score: float
    composite_score: float
    sentiment_summary: str = ""
    catalysts: list[str] = field(default_factory=list)
    risk_flags: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_report_dict(self) -> dict:
        """Format the play for human-readable output."""
        def fmt_price(p: float | None) -> str:
            return f"${p:.2f}" if p is not None else "N/A"

        return {
            "ticker": self.ticker,
            "setup": self.setup_type,
            "direction": self.direction,
            "entry": fmt_price(self.entry_price),
            "stop": fmt_price(self.stop_loss),
            "tp1": fmt_price(self.take_profit_1),
            "tp2": fmt_price(self.take_profit_2),
            "r_r": f"{self.risk_reward_ratio:.2f}",
            "r_r_tp2": f"{self.risk_reward_ratio_tp2:.2f}" if self.risk_reward_ratio_tp2 else "N/A",
            "risk_pct": f"{self.position_risk_pct:.1f}%",
            "tech_score": f"{self.technical_score:.1f}",
            "sent_score": f"{self.sentiment_score:.1f}",
            "composite": f"{self.composite_score:.1f}",
            "sentiment": self.sentiment_summary,
            "catalysts": ", ".join(self.catalysts) if self.catalysts else "None",
            "risk_flags": ", ".join(self.risk_flags) if self.risk_flags else "None",
            "notes": self.notes,
        }
