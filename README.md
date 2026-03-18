# Swing Screener

A Python CLI tool that identifies high-probability swing trade setups using a sub-agent architecture. The pipeline screens a broad market universe, detects technical patterns, analyzes sentiment via the Claude API, and scores/ranks the best trade setups — all coordinated by an iterative orchestrator.

## Quickstart

```bash
# Clone and set up
git clone https://github.com/svitale333/swing-screener.git
cd swing-screener
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY

# Run the full pipeline
python -m src.cli run

# Dry run (no API calls — screener + technical only)
python -m src.cli run --dry-run
```

## Architecture

```
                  ┌──────────────┐
                  │ Orchestrator │  (iterative loop, up to 3 iterations)
                  └──────┬───────┘
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
   ┌──────────┐  ┌───────────┐  ┌───────────┐
   │ Screener │  │ Technical │  │ Sentiment │
   │ (yfinance│  │ (pandas-ta│  │ (Claude   │
   │  filter) │  │  patterns)│  │  API +    │
   └────┬─────┘  └─────┬─────┘  │  search)  │
        │               │        └─────┬─────┘
        │               │              │
        └───────────────┼──────────────┘
                        ▼
                 ┌────────────┐
                 │  Scoring   │  (R:R calc, composite scoring, ranking)
                 │   Agent    │
                 └────────────┘
                        │
                        ▼
                 ┌────────────┐
                 │   Output   │  (console, JSON, markdown, CSV)
                 └────────────┘
```

**Pipeline flow:**
1. **Screener** — Filters the S&P 500 + supplemental universe by volume, price, and market cap (typically ~300 candidates)
2. **Technical** — Detects bull flags, squeeze breakouts, mean reversions, and trend pullbacks using pandas-ta indicators
3. **Sentiment** — Sends technically-qualified tickers (~15-30) to Claude with web search for catalyst/risk analysis
4. **Scoring** — Calculates trade parameters (entry, stop, targets), composite scores, and ranks plays
5. **Orchestrator** — Runs the loop, relaxing parameters if results are insufficient, avoiding redundant API calls

## CLI Usage

```bash
# Full pipeline run (default)
python -m src.cli run

# Custom config overrides
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

## Configuration Reference

All config lives in `src/config.py` as dataclasses. CLI flags override defaults at runtime without modifying files.

### `AgentConfig` (top-level)

| Field | Default | Description |
|-------|---------|-------------|
| `max_iterations` | `3` | Max orchestrator loop iterations |
| `output_dir` | `"outputs"` | Directory for report files |

### `ScreenerConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `min_avg_volume` | `1,000,000` | Minimum average daily volume |
| `min_price` | `10.0` | Minimum stock price |
| `max_price` | `500.0` | Maximum stock price |
| `min_market_cap` | `500,000,000` | Minimum market cap |
| `max_candidates` | `300` | Max tickers to pass downstream |
| `cache_ttl_hours` | `4.0` | Screener cache TTL |

### `TechnicalConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `lookback_days` | `120` | Historical data window |
| `bb_squeeze_percentile` | `20.0` | Bollinger Band squeeze threshold |
| `rsi_oversold` | `30.0` | RSI oversold level |
| `rsi_overbought` | `70.0` | RSI overbought level |
| `min_consolidation_days` | `5` | Min consolidation period |
| `max_consolidation_days` | `30` | Max consolidation period |

### `ScoringConfig`

| Field | Default | Description |
|-------|---------|-------------|
| `min_risk_reward` | `2.0` | Minimum R:R to qualify |
| `technical_weight` | `0.5` | Weight for technical score |
| `sentiment_weight` | `0.3` | Weight for sentiment score |
| `rr_weight` | `0.2` | Weight for R:R in composite |
| `target_play_count` | `5` | Desired number of plays |
| `max_play_count` | `10` | Max plays in output |

## Output Formats

All outputs are saved to `outputs/` with the `run_id` prefix (e.g., `run_20260318_0830`).

| Format | File | Description |
|--------|------|-------------|
| Console | (terminal) | Rich table with colors, icons, and sentiment summaries |
| JSON | `{run_id}.json` | Full structured data with metadata and timing |
| Markdown | `{run_id}.md` | Copy-pasteable report for Slack/Confluence |
| CSV | `{run_id}.csv` | Flat export for spreadsheets/trading journals |

## Scheduling for Daily Runs

A helper script is provided at `scripts/run_daily.sh`. It activates the virtualenv, runs the pipeline with `--force-refresh`, and logs output to `logs/`.

### macOS (launchd)

Create `~/Library/LaunchAgents/com.swingscreener.daily.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.swingscreener.daily</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/swing-screener/scripts/run_daily.sh</string>
    </array>
    <key>StartCalendarInterval</key>
    <array>
        <dict><key>Weekday</key><integer>1</integer><key>Hour</key><integer>7</integer><key>Minute</key><integer>30</integer></dict>
        <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>7</integer><key>Minute</key><integer>30</integer></dict>
        <dict><key>Weekday</key><integer>3</integer><key>Hour</key><integer>7</integer><key>Minute</key><integer>30</integer></dict>
        <dict><key>Weekday</key><integer>4</integer><key>Hour</key><integer>7</integer><key>Minute</key><integer>30</integer></dict>
        <dict><key>Weekday</key><integer>5</integer><key>Hour</key><integer>7</integer><key>Minute</key><integer>30</integer></dict>
    </array>
    <key>StandardOutPath</key>
    <string>/path/to/swing-screener/logs/launchd.log</string>
    <key>StandardErrorPath</key>
    <string>/path/to/swing-screener/logs/launchd_error.log</string>
</dict>
</plist>
```

Load with: `launchctl load ~/Library/LaunchAgents/com.swingscreener.daily.plist`

### Linux (crontab)

```bash
# Run weekdays at 7:30 AM ET
# Edit with: crontab -e
30 7 * * 1-5 /path/to/swing-screener/scripts/run_daily.sh
```

Make sure your system timezone is set to `America/New_York`, or use `TZ=America/New_York` prefix.

### GCP Cloud Run Job

Build and deploy as a scheduled Cloud Run job:

```bash
# Build container
docker build -t gcr.io/YOUR_PROJECT/swing-screener .

# Push to GCR
docker push gcr.io/YOUR_PROJECT/swing-screener

# Create Cloud Run job
gcloud run jobs create swing-screener \
  --image gcr.io/YOUR_PROJECT/swing-screener \
  --region us-east1 \
  --set-env-vars ANTHROPIC_API_KEY=your-key \
  --memory 1Gi \
  --task-timeout 600

# Schedule weekdays at 7:30 AM ET (12:30 UTC during EDT)
gcloud scheduler jobs create http swing-screener-daily \
  --location us-east1 \
  --schedule "30 12 * * 1-5" \
  --time-zone "America/New_York" \
  --uri "https://us-east1-run.googleapis.com/apis/run.googleapis.com/v1/namespaces/YOUR_PROJECT/jobs/swing-screener:run" \
  --http-method POST \
  --oauth-service-account-email YOUR_SA@YOUR_PROJECT.iam.gserviceaccount.com
```

Integrate with your existing GitHub Actions CI/CD by adding a deploy step that builds and pushes the container on merge to `main`.

## Cost Estimates

Each full pipeline run costs approximately **$0.05 - $0.15** in Claude API usage, depending on the number of tickers sent to sentiment analysis (typically 15-30). At one run per trading day (~252 days/year), annual cost is roughly **$12 - $40**.

## Disclaimer

This tool is for informational purposes only. Not financial advice. Always do your own research and manage risk appropriately. Paper trade first before using any signals in live trading.
