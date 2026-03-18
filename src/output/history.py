from __future__ import annotations

import json
import os
from datetime import datetime

from rich.console import Console
from rich.table import Table

from src.types import OrchestratorResult
from src.utils.logging import get_logger

logger = get_logger(__name__)

_HISTORY_FILE = os.path.join("data", "run_history.json")
_MAX_DISPLAY = 20


class RunHistory:
    """Tracks run metadata in data/run_history.json."""

    def __init__(self, history_file: str = _HISTORY_FILE) -> None:
        self.history_file = history_file

    def record(self, result: OrchestratorResult) -> None:
        """Append a run's metadata to the history file."""
        plays = result.plays
        avg_score = (
            sum(p.composite_score for p in plays) / len(plays) if plays else 0
        )
        avg_rr = (
            sum(p.risk_reward_ratio for p in plays) / len(plays) if plays else 0
        )

        entry = {
            "run_id": result.run_id,
            "timestamp": result.timestamp,
            "play_count": len(plays),
            "avg_score": round(avg_score, 2),
            "avg_rr": round(avg_rr, 2),
            "iterations": result.iterations,
            "api_cost": result.total_api_cost_estimate,
            "execution_time": result.execution_time_seconds,
        }

        history = self._load()
        history.append(entry)
        self._save(history)
        logger.info(f"Run recorded in history: {result.run_id}")

    def get_last(self) -> dict | None:
        """Return the most recent run entry, or None if empty."""
        history = self._load()
        return history[-1] if history else None

    def get_last_run_id(self) -> str | None:
        """Return the run_id of the most recent run."""
        last = self.get_last()
        return last["run_id"] if last else None

    def get_recent(self, count: int = _MAX_DISPLAY) -> list[dict]:
        """Return the most recent N run entries."""
        history = self._load()
        return history[-count:]

    def show_history(self, console: Console | None = None) -> None:
        """Print a table of recent runs."""
        console = console or Console()
        entries = self.get_recent()

        if not entries:
            console.print("[yellow]No run history found.[/yellow]")
            return

        table = Table(title="Recent Runs", show_lines=True)
        table.add_column("Run ID", style="bold")
        table.add_column("Date", width=16)
        table.add_column("Plays", justify="center")
        table.add_column("Avg Score", justify="center")
        table.add_column("Avg R:R", justify="center")
        table.add_column("Iterations", justify="center")
        table.add_column("Cost", justify="right")
        table.add_column("Time", justify="right")

        for entry in reversed(entries):
            try:
                dt = datetime.fromisoformat(entry["timestamp"])
                date_str = dt.strftime("%Y-%m-%d %H:%M")
            except (ValueError, TypeError, KeyError):
                date_str = entry.get("timestamp", "?")

            exec_time = entry.get("execution_time", 0)
            minutes = int(exec_time // 60)
            seconds = int(exec_time % 60)

            table.add_row(
                entry.get("run_id", "?"),
                date_str,
                str(entry.get("play_count", 0)),
                f"{entry.get('avg_score', 0):.1f}",
                f"{entry.get('avg_rr', 0):.1f}:1",
                str(entry.get("iterations", 0)),
                f"${entry.get('api_cost', 0):.2f}",
                f"{minutes}m {seconds:02d}s",
            )

        console.print(table)

    def _load(self) -> list[dict]:
        if not os.path.exists(self.history_file):
            return []
        try:
            with open(self.history_file) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning(f"Could not read history file: {self.history_file}")
            return []

    def _save(self, history: list[dict]) -> None:
        os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
        with open(self.history_file, "w") as f:
            json.dump(history, f, indent=2)
