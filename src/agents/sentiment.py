from __future__ import annotations

from src.config import SentimentConfig
from src.types import TechnicalSetup, SentimentResult
from src.utils.logging import get_logger

logger = get_logger(__name__)


class SentimentAgent:
    """Analyzes sentiment for technically-qualified tickers via Claude API."""

    def __init__(self, config: SentimentConfig) -> None:
        self.config = config

    def run(self, setups: list[TechnicalSetup]) -> list[SentimentResult]:
        """Run sentiment analysis on tickers with technical setups."""
        logger.info("SentimentAgent not yet implemented")
        return []
