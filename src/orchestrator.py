from __future__ import annotations

from src.config import AgentConfig
from src.types import TradePlay
from src.utils.logging import get_logger

logger = get_logger(__name__)


class Orchestrator:
    """Coordinates the agent pipeline with an iterative loop."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config

    def run(self) -> list[TradePlay]:
        """Execute the full screening pipeline."""
        logger.info("Orchestrator not yet implemented")
        return []
