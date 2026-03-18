# CLAUDE.md

## Project Overview

Swing Screener is a Python-based swing trade screening agent that uses a sub-agent architecture to identify high-probability swing trade setups. The pipeline is: Screener → Technical Analyzer → Sentiment Analyst (Claude API) → Scoring Engine, coordinated by an Orchestrator with an iterative loop.

## Architecture Rules

- **Sub-agent architecture**: Each agent is an independent module in `src/agents/`. Agents communicate ONLY through shared dataclasses defined in `src/types.py`. No agent should import from another agent.
- **Config-driven**: Every numeric threshold, parameter, and tunable lives in `src/config.py`. Zero magic numbers in agent code. If you need a threshold, add it to the config dataclass first, then reference it.
- **The orchestrator is the only coordinator**: Only `src/orchestrator.py` calls agents. Agents never call each other directly.

## Code Standards

- Python 3.11+. Use `from __future__ import annotations` in all files.
- Type hints on all function signatures. Use the dataclasses from `src/types.py` — do not pass raw dicts between agents.
- Use `dataclasses` for data structures, not Pydantic (keep dependencies lean).
- Logging via `src/utils/logging.py` — use `get_logger(__name__)` in every module. No `print()` statements.
- Error handling: agents should catch and log errors for individual tickers, never crash the pipeline. A single ticker failure should not stop the run.
- Tests go in `tests/` with `pytest`. Mock external calls (yfinance, Claude API) — tests must not make real network requests.

## File Organization

```
src/
  config.py          — All configuration dataclasses
  types.py           — Shared data structures (ScreenedCandidate, TechnicalSetup, SentimentResult, TradePlay, OrchestratorResult)
  agents/
    screener.py      — Universe filtering (yfinance, no LLM)
    technical.py     — Pattern detection (pandas-ta, no LLM)
    sentiment.py     — Claude API + web search
    scoring.py       — R:R calculation, composite scoring
  orchestrator.py    — Main iterative loop
  output/            — Formatters (console, json, markdown, csv)
  utils/             — Helpers (logging, technical_helpers, trade_math)
  prompts/           — Prompt templates for Claude API calls
  cli.py             — argparse CLI
```

## Dependencies

- `yfinance` for market data (free, no API key needed)
- `pandas-ta` for technical indicators
- `anthropic` SDK for sentiment agent (requires ANTHROPIC_API_KEY in .env)
- `rich` for terminal output
- `pandas`, `numpy` as core data libraries

## Key Design Decisions

- The screener runs once per orchestrator execution. The market universe doesn't change within a run.
- Sentiment analysis (Claude API) is the expensive step. Only send technically-qualified tickers (typically 15-30, not 300+).
- The orchestrator loops up to `max_iterations` (default 3), relaxing parameters if results are insufficient. It tracks which tickers already have sentiment analysis to avoid redundant API calls.
- A ticker can have multiple technical setups. The scoring agent picks the best one per ticker for the final output.
- All outputs go to `outputs/` with a `run_id` prefix.

## What NOT to Do

- Do not install additional dependencies beyond what's in requirements.txt without asking.
- Do not use async/await — keep the pipeline synchronous for simplicity and debuggability.
- Do not build a web UI or API server. This is a CLI tool.
- Do not use Pydantic, FastAPI, or any web framework.
- Do not hardcode ticker lists inside agent logic — all static data lives in `data/`.
- Do not make real API calls in tests.

## Build Sequence

This project is built incrementally via prompt files in `prompts/`. Each prompt builds on the previous step. The sequence is:
1. Project scaffold (config, types, structure)
2. Screener agent
3. Technical agent
4. Sentiment agent
5. Scoring agent
6. Orchestrator
7. CLI and output formatting

When implementing a prompt, read it fully before writing any code. Implement exactly what it specifies — do not skip sections or add unspecified features.

- Use `git config user.email svitale1997@gmail.com` and `git config user.name svitale333` before making any commits

- Before starting a new working directory, make sure to check out the latest version of main