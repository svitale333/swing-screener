from __future__ import annotations

from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.types import OrchestratorResult, TradePlay


class ConsoleFormatter:
    """Formats OrchestratorResult as a rich terminal table."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console or Console()

    def render(self, result: OrchestratorResult) -> None:
        """Print the full result to the terminal."""
        if not result.plays:
            self.console.print(
                Panel(
                    "[bold yellow]No plays found for this run.[/bold yellow]\n"
                    "Try relaxing filters or running again when market conditions change.",
                    title="Swing Screener",
                    border_style="yellow",
                )
            )
            self._render_footer(result)
            return

        table = self._build_table(result)
        self.console.print(table)
        self._render_footer(result)

    def build_table(self, result: OrchestratorResult) -> Table:
        """Build and return the Rich Table (for testing)."""
        return self._build_table(result)

    def _build_table(self, result: OrchestratorResult) -> Table:
        try:
            date_str = datetime.fromisoformat(result.timestamp).strftime("%B %d, %Y")
        except (ValueError, TypeError):
            date_str = result.timestamp

        table = Table(
            title=f"SWING PLAYS \u2014 {date_str}",
            title_style="bold cyan",
            show_lines=True,
            padding=(0, 1),
        )

        table.add_column("Rank", justify="center", style="bold", width=4)
        table.add_column("Ticker", style="bold", width=14)
        table.add_column("Entry", justify="right", width=9)
        table.add_column("Stop", justify="right", width=9)
        table.add_column("TP1", justify="right", width=9)
        table.add_column("TP2", justify="right", width=9)
        table.add_column("R:R", justify="center", width=5)
        table.add_column("Score", justify="center", width=5)

        for rank, play in enumerate(result.plays, 1):
            self._add_play_rows(table, rank, play)

        return table

    def _add_play_rows(self, table: Table, rank: int, play: TradePlay) -> None:
        direction_icon = "\U0001f7e2" if play.direction == "long" else "\U0001f534"
        stop_pct = self._pct_from_entry(play.entry_price, play.stop_loss)
        tp1_pct = self._pct_from_entry(play.entry_price, play.take_profit_1)
        tp2_pct = (
            self._pct_from_entry(play.entry_price, play.take_profit_2)
            if play.take_profit_2
            else "N/A"
        )

        # Main row
        table.add_row(
            str(rank),
            f"{play.ticker} {direction_icon}\n{play.setup_type}",
            f"${play.entry_price:.2f}",
            f"${play.stop_loss:.2f}\n{stop_pct}",
            f"${play.take_profit_1:.2f}\n{tp1_pct}",
            f"${play.take_profit_2:.2f}\n{tp2_pct}" if play.take_profit_2 else "N/A",
            f"{play.risk_reward_ratio:.1f}",
            f"{play.composite_score:.1f}",
        )

        # Sentiment + risk flags as a subtitle row
        parts: list[str] = []
        if play.sentiment_summary:
            parts.append(play.sentiment_summary)
        if play.risk_flags:
            flags = " | ".join(f"\u26a0\ufe0f {f}" for f in play.risk_flags)
            parts.append(flags)
        if parts:
            subtitle = " \u2014 ".join(parts)
            table.add_row("", Text(subtitle, style="dim"), "", "", "", "", "", "")

    def _render_footer(self, result: OrchestratorResult) -> None:
        plays = result.plays
        avg_rr = (
            sum(p.risk_reward_ratio for p in plays) / len(plays) if plays else 0
        )
        avg_score = (
            sum(p.composite_score for p in plays) / len(plays) if plays else 0
        )
        minutes = int(result.execution_time_seconds // 60)
        seconds = int(result.execution_time_seconds % 60)

        footer = (
            f"Run: {result.run_id} | "
            f"Plays: {len(plays)} | "
            f"Avg R:R: {avg_rr:.1f}:1 | "
            f"Avg Score: {avg_score:.1f}/10 | "
            f"Cost: ~${result.total_api_cost_estimate:.2f} | "
            f"Time: {minutes}m {seconds:02d}s"
        )
        self.console.print(f"\n[dim]{footer}[/dim]")

    @staticmethod
    def _pct_from_entry(entry: float, target: float) -> str:
        if entry == 0:
            return "N/A"
        pct = ((target - entry) / entry) * 100
        sign = "+" if pct >= 0 else ""
        return f"{sign}{pct:.1f}%"
