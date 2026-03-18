from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass
class ScreenerConfig:
    min_avg_volume: int = 1_000_000
    min_price: float = 10.0
    max_price: float = 500.0
    min_market_cap: float = 500_000_000.0
    min_options_volume: int = 500
    max_candidates: int = 300
    cache_ttl_hours: float = 4.0
    ticker_cache_ttl_hours: float = 24.0
    max_workers: int = 10
    blacklist: list[str] = field(default_factory=lambda: [
        "BRK.A", "BRK.B", "GOOG",  # duplicates or data issues
    ])


@dataclass
class TechnicalConfig:
    lookback_days: int = 120
    bb_squeeze_percentile: float = 20.0
    atr_contraction_window: int = 14
    volume_dryup_threshold: float = 0.5
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    min_consolidation_days: int = 5
    max_consolidation_days: int = 30


@dataclass
class SentimentConfig:
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024
    min_earnings_gap_days: int = 7
    max_tickers_per_batch: int = 5
    api_call_delay_seconds: float = 1.0
    max_retries: int = 3


@dataclass
class ScoringConfig:
    min_risk_reward: float = 2.0
    technical_weight: float = 0.5
    sentiment_weight: float = 0.3
    rr_weight: float = 0.2
    min_confidence_score: float = 6.0
    target_play_count: int = 5
    max_play_count: int = 10


@dataclass
class AgentConfig:
    screener: ScreenerConfig = field(default_factory=ScreenerConfig)
    technical: TechnicalConfig = field(default_factory=TechnicalConfig)
    sentiment: SentimentConfig = field(default_factory=SentimentConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    max_iterations: int = 3
    output_dir: str = "outputs"
