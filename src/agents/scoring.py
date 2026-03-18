from __future__ import annotations

from src.config import ScoringConfig
from src.types import TechnicalSetup, SentimentResult, TradePlay
from src.utils.logging import get_logger

logger = get_logger(__name__)


class ScoringAgent:
    """Calculates R:R ratios and composite scores to produce final trade plays."""

    def __init__(self, config: ScoringConfig) -> None:
        self.config = config

    def run(
        self,
        setups: list[TechnicalSetup],
        sentiments: list[SentimentResult],
    ) -> list[TradePlay]:
        """Score setups with sentiment data and return ranked trade plays."""
        logger.info("ScoringAgent not yet implemented")
        return []
