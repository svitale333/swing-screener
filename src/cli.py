from __future__ import annotations

import argparse
import json
import os
import sys

from rich.console import Console
from rich.panel import Panel

from src.config import AgentConfig
from src.orchestrator import Orchestrator
from src.output.console import ConsoleFormatter
from src.output.csv_export import CSVExporter
from src.output.history import RunHistory
from src.output.json_report import JSONReporter
from src.output.markdown_report import MarkdownReporter
from src.utils.logging import get_logger

logger = get_logger(__name__)
console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="swing-screener",
        description="Swing trade screening pipeline with sub-agent architecture.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose (DEBUG) logging"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Execute the full screening pipeline")
    run_parser.add_argument(
        "--min-rr",
        type=float,
        default=None,
        help="Minimum risk-reward ratio (must be > 1.0, default: 2.0)",
    )
    run_parser.add_argument(
        "--target-plays",
        type=int,
        default=None,
        help="Target number of plays (1-20, default: 5)",
    )
    run_parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Max orchestrator iterations (default: 3)",
    )
    run_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run screener + technical only, skip sentiment API calls",
    )
    run_parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Force refresh the screener cache",
    )

    # --- show-last ---
    subparsers.add_parser("show-last", help="Show results from the last run")

    # --- history ---
    subparsers.add_parser("history", help="List recent runs")

    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments, exit on invalid input."""
    if args.command == "run":
        if args.min_rr is not None and args.min_rr <= 1.0:
            console.print("[red]Error: --min-rr must be greater than 1.0[/red]")
            sys.exit(1)
        if args.target_plays is not None and not (1 <= args.target_plays <= 20):
            console.print("[red]Error: --target-plays must be between 1 and 20[/red]")
            sys.exit(1)
        if args.max_iterations is not None and args.max_iterations < 1:
            console.print("[red]Error: --max-iterations must be at least 1[/red]")
            sys.exit(1)


def apply_config_overrides(config: AgentConfig, args: argparse.Namespace) -> AgentConfig:
    """Apply CLI argument overrides to the config (does not modify files)."""
    if args.min_rr is not None:
        config.scoring.min_risk_reward = args.min_rr
    if args.target_plays is not None:
        config.scoring.target_play_count = args.target_plays
    if args.max_iterations is not None:
        config.max_iterations = args.max_iterations
    if args.force_refresh:
        config.screener.cache_ttl_hours = 0
    return config


def print_banner(config: AgentConfig, dry_run: bool) -> None:
    """Display a startup summary before running."""
    mode = "[yellow]DRY RUN[/yellow] (no API calls)" if dry_run else "[green]FULL RUN[/green]"
    info = (
        f"[bold]Swing Screener[/bold]\n"
        f"Mode: {mode}\n"
        f"Target plays: {config.scoring.target_play_count} | "
        f"Min R:R: {config.scoring.min_risk_reward:.1f} | "
        f"Max iterations: {config.max_iterations}"
    )
    console.print(Panel(info, border_style="cyan"))


def cmd_run(args: argparse.Namespace) -> None:
    """Execute the full pipeline."""
    config = AgentConfig()
    config = apply_config_overrides(config, args)

    print_banner(config, args.dry_run)

    orchestrator = Orchestrator(config)
    result = orchestrator.run(dry_run=args.dry_run)

    # Output to all formats
    output_dir = config.output_dir

    ConsoleFormatter(console).render(result)
    JSONReporter(output_dir).save(result)
    MarkdownReporter(output_dir).save(result)
    CSVExporter(output_dir).save(result)

    # Record in history
    RunHistory().record(result)

    console.print(f"\n[dim]Reports saved to {output_dir}/[/dim]")


def cmd_show_last(args: argparse.Namespace) -> None:
    """Re-render the last run's results."""
    history = RunHistory()
    run_id = history.get_last_run_id()

    if not run_id:
        console.print("[yellow]No previous runs found.[/yellow]")
        return

    json_path = os.path.join("outputs", f"{run_id}.json")
    if not os.path.exists(json_path):
        console.print(f"[red]JSON report not found: {json_path}[/red]")
        return

    with open(json_path) as f:
        data = json.load(f)

    # Reconstruct OrchestratorResult from JSON
    from src.types import OrchestratorResult, TradePlay

    plays = [TradePlay(**p) for p in data.get("plays", [])]
    result = OrchestratorResult(
        plays=plays,
        metadata=data.get("metadata", {}),
        iterations=data.get("iterations", 0),
        run_id=data.get("run_id", "unknown"),
        timestamp=data.get("timestamp", ""),
        total_api_cost_estimate=data.get("total_api_cost_estimate", 0),
        execution_time_seconds=data.get("execution_time_seconds", 0),
    )

    ConsoleFormatter(console).render(result)


def cmd_history(args: argparse.Namespace) -> None:
    """Show recent run history."""
    RunHistory().show_history(console)


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.verbose:
        import logging

        logging.getLogger().setLevel(logging.DEBUG)
        for name in logging.Logger.manager.loggerDict:
            logging.getLogger(name).setLevel(logging.DEBUG)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    validate_args(args)

    commands = {
        "run": cmd_run,
        "show-last": cmd_show_last,
        "history": cmd_history,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
