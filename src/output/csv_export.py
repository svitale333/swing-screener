from __future__ import annotations

import csv
import io
import os

from src.types import OrchestratorResult, TradePlay
from src.utils.logging import get_logger

logger = get_logger(__name__)

_COLUMNS = [
    "rank",
    "ticker",
    "direction",
    "setup_type",
    "entry",
    "stop_loss",
    "tp1",
    "tp2",
    "rr_ratio",
    "composite_score",
    "sentiment",
    "sentiment_confidence",
    "risk_flags",
    "notes",
]


class CSVExporter:
    """Exports OrchestratorResult plays to a flat CSV."""

    def __init__(self, output_dir: str = "outputs") -> None:
        self.output_dir = output_dir

    def save(self, result: OrchestratorResult) -> str:
        """Write CSV to outputs/{run_id}.csv. Returns the file path."""
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"{result.run_id}.csv")

        content = self.render(result)
        with open(path, "w", newline="") as f:
            f.write(content)

        logger.info(f"CSV report saved to {path}")
        return path

    def render(self, result: OrchestratorResult) -> str:
        """Build CSV string from plays."""
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=_COLUMNS)
        writer.writeheader()

        for rank, play in enumerate(result.plays, 1):
            writer.writerow(self._play_to_row(rank, play))

        return output.getvalue()

    @staticmethod
    def _play_to_row(rank: int, play: TradePlay) -> dict:
        return {
            "rank": rank,
            "ticker": play.ticker,
            "direction": play.direction,
            "setup_type": play.setup_type,
            "entry": f"{play.entry_price:.2f}",
            "stop_loss": f"{play.stop_loss:.2f}",
            "tp1": f"{play.take_profit_1:.2f}",
            "tp2": f"{play.take_profit_2:.2f}" if play.take_profit_2 else "",
            "rr_ratio": f"{play.risk_reward_ratio:.2f}",
            "composite_score": f"{play.composite_score:.1f}",
            "sentiment": play.sentiment_summary,
            "sentiment_confidence": f"{play.sentiment_score:.1f}",
            "risk_flags": "; ".join(play.risk_flags) if play.risk_flags else "",
            "notes": play.notes,
        }
