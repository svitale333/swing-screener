#!/bin/bash
# Daily pre-market swing screener run.
# Schedule this with cron (Linux) or launchd (macOS) for weekday runs.
# See README.md for scheduling instructions.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Activate virtual environment
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
fi

# Create logs directory if needed
mkdir -p logs

# Run the pipeline with forced cache refresh
python -m src.cli run --force-refresh 2>&1 | tee "logs/$(date +%Y%m%d_%H%M).log"
