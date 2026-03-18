from __future__ import annotations

import os
from datetime import datetime

from src.types import OrchestratorResult, TradePlay
from src.utils.logging import get_logger

logger = get_logger(__name__)

_DISCLAIMER = (
    "> **Disclaimer:** This tool is for informational purposes only. "
    "Not financial advice. Always do your own research and manage risk appropriately."
)


class MarkdownReporter:
    """Generates a markdown report from OrchestratorResult."""

    def __init__(self, output_dir: str = "outputs") -> None:
        self.output_dir = output_dir

    def save(self, result: OrchestratorResult) -> str:
        """Write markdown report to outputs/{run_id}.md. Returns the file path."""
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f"{result.run_id}.md")

        content = self.render(result)
        with open(path, "w") as f:
            f.write(content)

        logger.info(f"Markdown report saved to {path}")
        return path

    def render(self, result: OrchestratorResult) -> str:
        """Build the full markdown string."""
        try:
            date_str = datetime.fromisoformat(result.timestamp).strftime("%B %d, %Y")
        except (ValueError, TypeError):
            date_str = result.timestamp

        plays = result.plays
        avg_rr = (
            sum(p.risk_reward_ratio for p in plays) / len(plays) if plays else 0
        )
        avg_score = (
            sum(p.composite_score for p in plays) / len(plays) if plays else 0
        )

        lines: list[str] = []
        lines.append(f"# Swing Trade Setups \u2014 {date_str}")
        lines.append("")
        lines.append(
            f"**Run ID:** {result.run_id} | "
            f"**Plays:** {len(plays)} | "
            f"**Avg R:R:** {avg_rr:.1f}:1 | "
            f"**Avg Score:** {avg_score:.1f}/10"
        )
        lines.append("")
        lines.append("---")

        if not plays:
            lines.append("")
            lines.append("**No plays found for this run.** Try relaxing filters or running again when market conditions change.")
        else:
            for rank, play in enumerate(plays, 1):
                lines.append("")
                lines.extend(self._render_play(rank, play))
                lines.append("")
                lines.append("---")

        lines.append("")
        lines.append(_DISCLAIMER)
        lines.append("")

        return "\n".join(lines)

    def _render_play(self, rank: int, play: TradePlay) -> list[str]:
        direction_label = "LONG" if play.direction == "long" else "SHORT"
        direction_icon = "\U0001f7e2" if play.direction == "long" else "\U0001f534"

        lines: list[str] = []
        lines.append(
            f"## #{rank} \u2014 {play.ticker} ({play.setup_type}) "
            f"{direction_icon} {direction_label}"
        )
        lines.append(
            f"**Score: {play.composite_score:.1f}/10 | R:R: {play.risk_reward_ratio:.1f}:1**"
        )
        lines.append("")

        # Price table
        lines.append("| Level | Price | % from Entry |")
        lines.append("|-------|-------|-------------|")
        lines.append(f"| Entry | ${play.entry_price:.2f} | \u2014 |")
        lines.append(
            f"| Stop Loss | ${play.stop_loss:.2f} | "
            f"{self._pct_from_entry(play.entry_price, play.stop_loss)} |"
        )
        lines.append(
            f"| Take Profit 1 | ${play.take_profit_1:.2f} | "
            f"{self._pct_from_entry(play.entry_price, play.take_profit_1)} |"
        )
        if play.take_profit_2:
            lines.append(
                f"| Take Profit 2 | ${play.take_profit_2:.2f} | "
                f"{self._pct_from_entry(play.entry_price, play.take_profit_2)} |"
            )

        lines.append("")

        # Sentiment
        if play.sentiment_summary:
            lines.append(
                f"**Sentiment:** {play.sentiment_summary}"
            )
            lines.append("")

        # Risk flags
        if play.risk_flags:
            flags = ", ".join(f"\u26a0\ufe0f {f}" for f in play.risk_flags)
            lines.append(f"**Risk Flags:** {flags}")
            lines.append("")

        # Notes
        if play.notes:
            lines.append(f"**Setup Notes:** {play.notes}")

        return lines

    @staticmethod
    def _pct_from_entry(entry: float, target: float) -> str:
        if entry == 0:
            return "N/A"
        pct = ((target - entry) / entry) * 100
        sign = "+" if pct >= 0 else ""
        return f"{sign}{pct:.1f}%"
