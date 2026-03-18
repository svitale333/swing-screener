from __future__ import annotations

from src.config import TechnicalConfig
from src.types import ScreenedCandidate, TechnicalSetup
from src.utils.logging import get_logger

logger = get_logger(__name__)


class TechnicalAgent:
    """Detects technical setups on screened candidates."""

    def __init__(self, config: TechnicalConfig) -> None:
        self.config = config

    def run(self, candidates: list[ScreenedCandidate]) -> list[TechnicalSetup]:
        """Analyze candidates for technical setups."""
        logger.info("TechnicalAgent not yet implemented")
        return []
