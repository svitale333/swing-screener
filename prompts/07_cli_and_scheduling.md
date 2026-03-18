# Prompt 7: CLI Interface, Output Formatting & Scheduling

## Context
The full pipeline is built: Orchestrator → Screener → Technical → Sentiment → Scoring. This final step wraps everything in a clean CLI, produces publication-quality output reports, and makes the system cron-ready for daily pre-market runs.

## What to Build

### 1. CLI: `src/cli.py`

Build a CLI using `argparse` (no external dependency needed). Entry point: `python -m src.cli`

**Commands and flags:**

```bash
# Full pipeline run (default)
python -m src.cli run

# Full run with custom config overrides
python -m src.cli run --min-rr 2.5 --target-plays 8 --max-iterations 4

# Dry run: screener + technical only, no API calls
python -m src.cli run --dry-run

# Force refresh the screener cache
python -m src.cli run --force-refresh

# Show last run's results without re-running
python -m src.cli show-last

# List recent runs
python -m src.cli history

# Run with verbose logging
python -m src.cli run -v
```

**Argument handling:**
- CLI arguments override the defaults in `AgentConfig` but do NOT modify config files
- Validate inputs (e.g., `min-rr` must be > 1.0, `target-plays` must be 1-20)
- Show a startup banner with the run config summary before executing

### 2. Output Formatting: `src/output/`

Create multiple output formatters. All read from `OrchestratorResult`.

#### `src/output/console.py` — Rich Terminal Output
Using `rich`, format the final plays as a table in the terminal:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    🎯 SWING PLAYS — March 17, 2026                        │
├──────┬──────────────┬─────────┬─────────┬─────────┬─────────┬─────┬──────┤
│ Rank │ Ticker       │ Entry   │ Stop    │ TP1     │ TP2     │ R:R │ Score│
├──────┼──────────────┼─────────┼─────────┼─────────┼─────────┼─────┼──────┤
│  1   │ NVDA 🟢      │ $142.50 │ $137.80 │ $150.00 │ $155.50 │ 2.8 │  8.2 │
│      │ Bull Flag    │         │ -3.3%   │ +5.3%   │ +9.1%   │     │      │
│      │ Bullish sentiment — AI chip demand catalyst              │      │
├──────┼──────────────┼─────────┼─────────┼─────────┼─────────┼─────┼──────┤
│  2   │ AMZN 🟢      │ $198.20 │ $192.00 │ $207.50 │ $213.00 │ 2.5 │  7.8 │
│      │ Squeeze      │         │ -3.1%   │ +4.7%   │ +7.5%   │     │      │
│      │ Neutral sentiment — no near-term catalysts               │      │
└──────┴──────────────┴─────────┴─────────┴─────────┴─────────┴─────┴──────┘
```

Each play should show:
- Rank, ticker, setup type
- Entry, stop loss (with % from entry), TP1 (with %), TP2 (with %), R:R, composite score
- One-line sentiment summary
- Direction indicator (🟢 long, 🔴 short)
- Risk flags highlighted if present (e.g., ⚠️ Earnings in 10 days)

#### `src/output/json_report.py` — Structured JSON
Save the full `OrchestratorResult` as JSON to `outputs/{run_id}.json`:
- All plays with full field detail
- Metadata: run config, iteration history, timing, cost
- Pretty-printed with 2-space indent

#### `src/output/markdown_report.py` — Markdown Report
Generate a markdown file at `outputs/{run_id}.md` suitable for sharing or pasting into Slack/Confluence:

```markdown
# Swing Trade Setups — March 17, 2026

**Run ID:** run_20260317_0830 | **Plays:** 6 | **Avg R:R:** 2.7:1 | **Avg Score:** 7.2/10

---

## #1 — NVDA (Bull Flag) 🟢 LONG
**Score: 8.2/10 | R:R: 2.8:1**

| Level | Price | % from Entry |
|-------|-------|-------------|
| Entry | $142.50 | — |
| Stop Loss | $137.80 | -3.3% |
| Take Profit 1 | $150.00 | +5.3% |
| Take Profit 2 | $155.50 | +9.1% |

**Sentiment:** Bullish (8/10) — AI chip demand driving sector momentum. No earnings for 30+ days.

**Setup Notes:** 5-day consolidation following 8% impulse move. Volume declining within flag. BB bandwidth at 15th percentile. RS line at new highs.

---
```

#### `src/output/csv_export.py` — Flat CSV
Export a simplified CSV to `outputs/{run_id}.csv` with columns:
`rank, ticker, direction, setup_type, entry, stop_loss, tp1, tp2, rr_ratio, composite_score, sentiment, sentiment_confidence, risk_flags, notes`

This is for easy import into a spreadsheet or trading journal.

### 3. Run History: `src/output/history.py`
- Maintain a `data/run_history.json` file that logs each run's metadata (run_id, timestamp, play count, avg score, cost)
- `show-last` command reads the most recent entry and re-renders its JSON output through the console formatter
- `history` command shows a table of the last 20 runs with key stats

### 4. Scheduling Setup: `scripts/`

#### `scripts/run_daily.sh`
A shell script for cron / launchd / Task Scheduler:
```bash
#!/bin/bash
cd /path/to/swing-screener
source .venv/bin/activate
python -m src.cli run --force-refresh 2>&1 | tee "logs/$(date +%Y%m%d_%H%M).log"
```

#### Scheduling instructions in README
Add a section to the main `README.md` explaining how to set up:
- **macOS**: `launchd` plist for weekday 7:30 AM ET runs
- **Linux**: `crontab` entry for weekday 7:30 AM ET runs
- **GCP Cloud Run Job**: Instructions for deploying as a scheduled Cloud Run job (since Sam uses GCP), including Dockerfile and `cloud-scheduler` command

### 5. Main Entry Point: `__main__.py`
Create `src/__main__.py` so `python -m src` works as an alias for the CLI.

### 6. Update `README.md`
Replace the scaffold README with a comprehensive one covering:
- What the tool does (one paragraph)
- Quickstart: install, configure `.env`, run
- Architecture overview (the sub-agent diagram)
- Configuration reference (all `AgentConfig` fields with defaults and descriptions)
- Output formats and where files land
- Scheduling for daily runs
- Cost estimates per run (approximate based on Claude API pricing)
- Limitations and disclaimers (not financial advice, paper trade first, etc.)

### Tests: `tests/test_output.py`
Write tests for:
- Console formatter produces valid Rich renderables given mock `OrchestratorResult`
- JSON report is valid JSON and contains all required fields
- Markdown report contains expected headers and table structure
- CSV export has correct columns and row count matching play count
- History file appends correctly and `show-last` returns the correct entry
- Edge case: zero plays — all formatters handle this gracefully (show "no plays found" message)

## Important Notes
- The console output is the primary UX. Make it look great — this is what Sam will see every morning.
- All file outputs go to `outputs/` with the `run_id` prefix for easy organization.
- The markdown report should be copy-pasteable into Slack/Confluence without modification.
- The CSV should be importable into Excel/Google Sheets without any cleanup.
- Include a disclaimer in both the markdown report and README: "This tool is for informational purposes only. Not financial advice. Always do your own research and manage risk appropriately."
- The scheduling setup should default to weekdays only — no point running on weekends when markets are closed.
- The GCP Cloud Run instructions should reference Sam's existing GCP + GitHub Actions CI/CD workflow.
