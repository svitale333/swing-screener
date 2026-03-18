from __future__ import annotations

import csv
import io
import json
import os
import tempfile

import pytest
from rich.console import Console
from rich.table import Table

from src.output.console import ConsoleFormatter
from src.output.csv_export import CSVExporter
from src.output.json_report import JSONReporter
from src.output.markdown_report import MarkdownReporter
from src.output.history import RunHistory
from src.types import OrchestratorResult, TradePlay


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_play(**overrides) -> TradePlay:
    defaults = {
        "ticker": "NVDA",
        "setup_type": "bull_flag",
        "direction": "long",
        "entry_price": 142.50,
        "stop_loss": 137.80,
        "take_profit_1": 150.00,
        "take_profit_2": 155.50,
        "risk_reward_ratio": 2.8,
        "risk_reward_ratio_tp2": 3.8,
        "position_risk_pct": 3.3,
        "technical_score": 7.5,
        "sentiment_score": 8.0,
        "composite_score": 8.2,
        "sentiment_summary": "Bullish — AI chip demand catalyst",
        "catalysts": ["AI chip demand", "Data center growth"],
        "risk_flags": [],
        "notes": "5-day consolidation following 8% impulse move.",
    }
    defaults.update(overrides)
    return TradePlay(**defaults)


def _make_result(plays: list[TradePlay] | None = None) -> OrchestratorResult:
    if plays is None:
        plays = [
            _make_play(),
            _make_play(
                ticker="AMZN",
                setup_type="squeeze_breakout",
                entry_price=198.20,
                stop_loss=192.00,
                take_profit_1=207.50,
                take_profit_2=213.00,
                risk_reward_ratio=2.5,
                risk_reward_ratio_tp2=3.2,
                position_risk_pct=3.1,
                technical_score=7.0,
                sentiment_score=6.0,
                composite_score=7.8,
                sentiment_summary="Neutral — no near-term catalysts",
                catalysts=[],
                risk_flags=["Earnings in 10 days"],
                notes="Squeeze setup with declining volume.",
            ),
        ]
    return OrchestratorResult(
        plays=plays,
        metadata={
            "universe_size": 280,
            "total_setups": 18,
            "sentiment_analyzed": 12,
            "adjustment_history": [],
            "stage_timings": {"screener": 5.2, "technical_iter1": 3.1},
        },
        iterations=2,
        run_id="run_20260318_0830",
        timestamp="2026-03-18T08:30:00",
        total_api_cost_estimate=0.08,
        execution_time_seconds=45.5,
    )


def _make_empty_result() -> OrchestratorResult:
    return _make_result(plays=[])


# ---------------------------------------------------------------------------
# Console Formatter
# ---------------------------------------------------------------------------

class TestConsoleFormatter:
    def test_render_produces_table(self):
        result = _make_result()
        formatter = ConsoleFormatter()
        table = formatter.build_table(result)

        assert isinstance(table, Table)
        assert table.row_count > 0

    def test_render_with_plays(self):
        result = _make_result()
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=True, width=120)
        formatter = ConsoleFormatter(console)
        formatter.render(result)

        output = buf.getvalue()
        assert "NVDA" in output
        assert "AMZN" in output
        assert "run_20260318_0830" in output

    def test_render_zero_plays(self):
        result = _make_empty_result()
        buf = io.StringIO()
        console = Console(file=buf, force_terminal=True, width=120)
        formatter = ConsoleFormatter(console)
        formatter.render(result)

        output = buf.getvalue()
        assert "No plays found" in output

    def test_table_has_correct_columns(self):
        result = _make_result()
        formatter = ConsoleFormatter()
        table = formatter.build_table(result)

        col_names = [c.header for c in table.columns]
        assert "Rank" in col_names
        assert "Ticker" in col_names
        assert "Entry" in col_names
        assert "R:R" in col_names
        assert "Score" in col_names

    def test_risk_flags_displayed(self):
        result = _make_result()
        buf = io.StringIO()
        console = Console(file=buf, no_color=True, width=200)
        formatter = ConsoleFormatter(console)
        formatter.render(result)

        output = buf.getvalue()
        # Strip Rich table borders and collapse whitespace for matching
        import re
        stripped = re.sub(r"[│┃├┤┌┐└┘┬┴┼━─╇╈╋┏┓┗┛┡┩┠┨╌╍]+", " ", output)
        flat = " ".join(stripped.split())
        assert "Earnings in 10 days" in flat

    def test_pct_from_entry(self):
        assert ConsoleFormatter._pct_from_entry(100.0, 110.0) == "+10.0%"
        assert ConsoleFormatter._pct_from_entry(100.0, 90.0) == "-10.0%"
        assert ConsoleFormatter._pct_from_entry(0, 10.0) == "N/A"


# ---------------------------------------------------------------------------
# JSON Reporter
# ---------------------------------------------------------------------------

class TestJSONReporter:
    def test_json_is_valid(self):
        result = _make_result()
        json_str = JSONReporter.to_json_string(result)
        data = json.loads(json_str)

        assert isinstance(data, dict)

    def test_json_contains_required_fields(self):
        result = _make_result()
        json_str = JSONReporter.to_json_string(result)
        data = json.loads(json_str)

        assert "plays" in data
        assert "metadata" in data
        assert "iterations" in data
        assert "run_id" in data
        assert "timestamp" in data
        assert "total_api_cost_estimate" in data
        assert "execution_time_seconds" in data

    def test_json_play_count_matches(self):
        result = _make_result()
        json_str = JSONReporter.to_json_string(result)
        data = json.loads(json_str)

        assert len(data["plays"]) == len(result.plays)

    def test_json_save_to_file(self):
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = JSONReporter(output_dir=tmpdir)
            path = reporter.save(result)

            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert data["run_id"] == "run_20260318_0830"

    def test_json_zero_plays(self):
        result = _make_empty_result()
        json_str = JSONReporter.to_json_string(result)
        data = json.loads(json_str)

        assert data["plays"] == []
        assert data["run_id"] == "run_20260318_0830"


# ---------------------------------------------------------------------------
# Markdown Reporter
# ---------------------------------------------------------------------------

class TestMarkdownReporter:
    def test_markdown_contains_header(self):
        result = _make_result()
        md = MarkdownReporter().render(result)

        assert "# Swing Trade Setups" in md
        assert "March 18, 2026" in md

    def test_markdown_contains_play_headers(self):
        result = _make_result()
        md = MarkdownReporter().render(result)

        assert "## #1" in md
        assert "NVDA" in md
        assert "## #2" in md
        assert "AMZN" in md

    def test_markdown_contains_price_table(self):
        result = _make_result()
        md = MarkdownReporter().render(result)

        assert "| Level | Price | % from Entry |" in md
        assert "| Entry |" in md
        assert "| Stop Loss |" in md
        assert "| Take Profit 1 |" in md

    def test_markdown_contains_disclaimer(self):
        result = _make_result()
        md = MarkdownReporter().render(result)

        assert "Disclaimer" in md
        assert "Not financial advice" in md

    def test_markdown_save_to_file(self):
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            reporter = MarkdownReporter(output_dir=tmpdir)
            path = reporter.save(result)

            assert os.path.exists(path)
            assert path.endswith(".md")

    def test_markdown_zero_plays(self):
        result = _make_empty_result()
        md = MarkdownReporter().render(result)

        assert "No plays found" in md
        assert "Disclaimer" in md

    def test_markdown_risk_flags(self):
        result = _make_result()
        md = MarkdownReporter().render(result)

        assert "Earnings in 10 days" in md


# ---------------------------------------------------------------------------
# CSV Exporter
# ---------------------------------------------------------------------------

class TestCSVExporter:
    def test_csv_has_correct_columns(self):
        result = _make_result()
        csv_str = CSVExporter().render(result)
        reader = csv.DictReader(io.StringIO(csv_str))
        fieldnames = reader.fieldnames

        expected = [
            "rank", "ticker", "direction", "setup_type", "entry",
            "stop_loss", "tp1", "tp2", "rr_ratio", "composite_score",
            "sentiment", "sentiment_confidence", "risk_flags", "notes",
        ]
        assert fieldnames == expected

    def test_csv_row_count_matches_plays(self):
        result = _make_result()
        csv_str = CSVExporter().render(result)
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)

        assert len(rows) == len(result.plays)

    def test_csv_ticker_values(self):
        result = _make_result()
        csv_str = CSVExporter().render(result)
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)

        tickers = [r["ticker"] for r in rows]
        assert "NVDA" in tickers
        assert "AMZN" in tickers

    def test_csv_save_to_file(self):
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            exporter = CSVExporter(output_dir=tmpdir)
            path = exporter.save(result)

            assert os.path.exists(path)
            assert path.endswith(".csv")

    def test_csv_zero_plays(self):
        result = _make_empty_result()
        csv_str = CSVExporter().render(result)
        reader = csv.DictReader(io.StringIO(csv_str))
        rows = list(reader)

        assert len(rows) == 0
        # Header should still be present
        assert reader.fieldnames is not None


# ---------------------------------------------------------------------------
# Run History
# ---------------------------------------------------------------------------

class TestRunHistory:
    def test_record_and_get_last(self):
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            history_file = os.path.join(tmpdir, "history.json")
            history = RunHistory(history_file=history_file)

            history.record(result)
            last = history.get_last()

            assert last is not None
            assert last["run_id"] == "run_20260318_0830"
            assert last["play_count"] == 2

    def test_history_appends_correctly(self):
        r1 = _make_result()
        r2 = _make_result(plays=[_make_play(ticker="AAPL")])
        r2.run_id = "run_20260319_0830"
        r2.timestamp = "2026-03-19T08:30:00"

        with tempfile.TemporaryDirectory() as tmpdir:
            history_file = os.path.join(tmpdir, "history.json")
            history = RunHistory(history_file=history_file)

            history.record(r1)
            history.record(r2)

            entries = history.get_recent()
            assert len(entries) == 2
            assert entries[0]["run_id"] == "run_20260318_0830"
            assert entries[1]["run_id"] == "run_20260319_0830"

    def test_get_last_run_id(self):
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            history_file = os.path.join(tmpdir, "history.json")
            history = RunHistory(history_file=history_file)

            assert history.get_last_run_id() is None

            history.record(result)
            assert history.get_last_run_id() == "run_20260318_0830"

    def test_show_last_returns_correct_entry(self):
        r1 = _make_result()
        r2 = _make_result(plays=[_make_play(ticker="TSLA")])
        r2.run_id = "run_20260319_0830"

        with tempfile.TemporaryDirectory() as tmpdir:
            history_file = os.path.join(tmpdir, "history.json")
            history = RunHistory(history_file=history_file)

            history.record(r1)
            history.record(r2)

            last = history.get_last()
            assert last["run_id"] == "run_20260319_0830"

    def test_empty_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_file = os.path.join(tmpdir, "history.json")
            history = RunHistory(history_file=history_file)

            assert history.get_last() is None
            assert history.get_recent() == []

    def test_show_history_renders(self):
        result = _make_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            history_file = os.path.join(tmpdir, "history.json")
            history = RunHistory(history_file=history_file)
            history.record(result)

            buf = io.StringIO()
            console = Console(file=buf, force_terminal=True, width=120)
            history.show_history(console)

            output = buf.getvalue()
            assert "run_20260318_0830" in output

    def test_show_history_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_file = os.path.join(tmpdir, "history.json")
            history = RunHistory(history_file=history_file)

            buf = io.StringIO()
            console = Console(file=buf, force_terminal=True, width=120)
            history.show_history(console)

            output = buf.getvalue()
            assert "No run history found" in output

    def test_record_zero_plays(self):
        result = _make_empty_result()
        with tempfile.TemporaryDirectory() as tmpdir:
            history_file = os.path.join(tmpdir, "history.json")
            history = RunHistory(history_file=history_file)

            history.record(result)
            last = history.get_last()

            assert last["play_count"] == 0
            assert last["avg_score"] == 0
