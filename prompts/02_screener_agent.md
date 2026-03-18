# Prompt 2: Screener Agent

## Context
I'm building a swing trade screener agent. The project scaffold is already in place (see `src/config.py` for config dataclasses, `src/types.py` for shared types). This step implements the **Screener Agent** — a pure Python module (no LLM calls) that filters the market down to a tradeable candidate universe using `yfinance`.

## What to Build

### Implement `src/agents/screener.py`

The `ScreenerAgent` class should:

#### 1. Build the Starting Universe
Pull tickers from a curated, high-volume list rather than scraping all of NYSE/NASDAQ (which is slow and unreliable with yfinance). Use a tiered approach:

**Tier 1 — Index constituents (reliable, covers most liquid names):**
- S&P 500 tickers: scrape from Wikipedia's S&P 500 table (`https://en.wikipedia.org/wiki/List_of_S%26P_500_companies`) using pandas `read_html`. Cache this list locally in a `data/` directory as `sp500_tickers.csv` with a 24-hour TTL so we're not hitting Wikipedia on every run.
- Add a hardcoded supplemental list of ~30-50 popular/liquid tickers NOT in the S&P 500 but commonly traded for swings (e.g., RIVN, PLTR, MARA, COIN, SOFI, HOOD, RBLX, SNAP, etc.). Store this in a `data/supplemental_tickers.py` file so it's easy to update.

**Tier 2 — Validation via yfinance:**
For each ticker in the combined universe, pull summary data and apply filters.

#### 2. Screening Filters
Use `yfinance.Ticker(symbol).info` and `yfinance.download()` to apply these filters (all thresholds from `ScreenerConfig`):

- **Average Volume**: `averageVolume` >= `min_avg_volume` (default 1M)
- **Price Range**: Current price between `min_price` and `max_price`
- **Market Cap**: Exclude micro-caps (< $500M market cap) — add this to config
- **Options Availability**: Use `ticker.options` to check that options chains exist (if the call raises or returns empty, skip the ticker)
- **Not in Blacklist**: Maintain a small blacklist in config for tickers that are technically liquid but poor swing candidates (e.g., BRK.A, or anything with known data issues)

#### 3. Data Enrichment
For each ticker that passes filters, populate a `ScreenedCandidate` dataclass with:
- `ticker`, `price` (current close), `avg_volume`, `market_cap`
- `sector` and `industry` from `ticker.info`

#### 4. Performance Considerations
- Use `yfinance.download(tickers, group_by='ticker')` for batch downloading price data (much faster than individual calls)
- Use threading or `concurrent.futures.ThreadPoolExecutor` for `.info` lookups (yfinance is I/O-bound), with a max of 10 workers
- Add a progress bar using `rich.progress` so the user can see screening progress
- The entire screening step should complete in under 2 minutes for ~500 tickers
- Implement graceful error handling — if a single ticker fails, log a warning and skip it, don't crash the pipeline

#### 5. Output
- Return `list[ScreenedCandidate]` sorted by average volume descending
- Cap the list at `max_candidates` from config
- Log summary stats: total universe size, tickers passing each filter stage, final count, sectors represented

#### 6. Caching
- Cache the screened results to `data/screener_cache.json` with a configurable TTL (default 4 hours). If a fresh cache exists, return it instead of re-screening. Add a `force_refresh: bool` parameter to bypass cache.

### Tests: `tests/test_screener.py`
Write tests for:
- Filter logic works correctly given mock ticker data (mock `yfinance` responses, don't make real API calls in tests)
- Blacklist exclusion works
- Cache read/write/TTL logic
- Edge cases: ticker with missing `.info` fields, ticker with no options chains
- Output is sorted by volume and capped at `max_candidates`

## Important Notes
- This agent makes ZERO LLM calls. It's pure data fetching and filtering.
- All filter thresholds come from `ScreenerConfig` — no hardcoded numbers in the agent logic.
- The supplemental ticker list should be clearly separated so it's easy to add/remove names.
- Handle yfinance's known quirks: some tickers return `None` for info fields, `.info` can throw exceptions for delisted tickers, rate limiting can occur with aggressive parallel requests (add a small sleep/backoff).
