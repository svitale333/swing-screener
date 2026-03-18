# Prompt 3: Technical Setup Detection Agent

## Context
The Screener Agent (Step 2) outputs a `list[ScreenedCandidate]` of ~100-300 liquid, tradeable tickers. This step implements the **Technical Setup Agent** — a Python module using `pandas-ta` and custom logic to detect high-probability swing trade setups across the screened universe. No LLM calls; this is quantitative pattern detection.

## What to Build

### Implement `src/agents/technical.py`

The `TechnicalAgent` class should:

#### 1. Data Retrieval
- Accept `list[ScreenedCandidate]` as input
- Download OHLCV data for all candidates using `yfinance.download()` with `period` derived from `TechnicalConfig.lookback_days` (default 120 days)
- Use batch download for efficiency: `yfinance.download([tickers], group_by='ticker')`

#### 2. Indicator Computation
For each ticker's OHLCV DataFrame, compute the following using `pandas-ta`:
- **Bollinger Bands** (20, 2): `bbands`. Extract bandwidth (`BBB`) for squeeze detection.
- **RSI** (14): `rsi`
- **ATR** (14): `atr`. Also compute a normalized ATR (ATR / close) for cross-ticker comparison.
- **VWAP**: `vwap` (intraday proxy — for daily bars, use anchored VWAP from the swing low/high)
- **Moving Averages**: EMA 9, EMA 21, SMA 50, SMA 200
- **Volume SMA** (20): for volume comparison
- **MACD** (12, 26, 9): `macd`
- **ADX** (14): `adx` for trend strength

Store all indicators in a dict keyed by ticker for downstream setup detection.

#### 3. Setup Detection
Implement a detector for each setup type. Each detector receives the ticker's OHLCV + indicators DataFrame and returns `TechnicalSetup | None`. A ticker can match MULTIPLE setups — return all matches.

**Setup 1: Squeeze / Breakout (`squeeze_breakout`)**
- Bollinger Band Bandwidth is in the bottom `bb_squeeze_percentile` (config, default 20th percentile) of its own N-day lookback
- ATR has contracted (current ATR < ATR SMA over `atr_contraction_window`)
- Volume in the last 5 sessions is below the 20-day volume SMA by at least `volume_dryup_threshold` (50%)
- Price is above the 50 SMA (bias: look for bullish breakouts in uptrends)
- **Support**: Lower Bollinger Band or recent swing low
- **Resistance**: Upper Bollinger Band or recent swing high
- **Score factors**: How tight the squeeze is (tighter = higher score), trend alignment (above 200 SMA = bonus), ADX < 20 (confirming low trend = good for breakout)

**Setup 2: Bull Flag / Flat Base (`bull_flag`)**
- Identify a strong impulse move: price gained >= 5% within 5 trading days at some point in the last 30 days
- After the impulse, price has consolidated: the range over the last `min_consolidation_days` to `max_consolidation_days` sessions is <= 50% of the impulse range
- Volume during consolidation is declining (average volume of consolidation period < 70% of impulse period volume)
- Price is holding above the 21 EMA during consolidation
- **Support**: Bottom of the consolidation range or 21 EMA
- **Resistance**: Top of the consolidation range (breakout trigger)
- **Score factors**: Flag tightness (tighter = better), volume decline steepness, distance above 50 SMA

**Setup 3: Mean Reversion / Oversold Bounce (`mean_reversion`)**
- RSI <= `rsi_oversold` (default 30) at some point in the last 5 sessions
- Price is at or near a support level: within 2% of the SMA 50 or SMA 200, or at a prior swing low
- Bullish RSI divergence: price made a lower low but RSI made a higher low (compare last two swing lows in a 20-day window)
- Volume spike on the most recent session (> 1.5x the 20-day average) suggesting capitulation/reversal
- **Support**: The swing low or MA level being tested
- **Resistance**: EMA 21 (first target), then prior swing high
- **Score factors**: Divergence strength, proximity to known support, volume capitulation signal, MACD histogram turning positive

**Setup 4: Trend Pullback / Continuation (`trend_pullback`)**
- Strong existing uptrend: price above rising 50 SMA, 50 SMA above 200 SMA, ADX > 25
- Current pullback: price has pulled back to the 21 EMA (within 1.5% of the 21 EMA) from a recent high
- The pullback is orderly: each pullback candle has lower volume than the prior impulse candles
- RSI between 40-55 (not overbought, not oversold — "reset")
- **Support**: 21 EMA or 50 SMA
- **Resistance**: Recent swing high
- **Score factors**: Trend strength (ADX value), how cleanly price touches the 21 EMA, relative strength vs SPY (if available)

#### 4. Scoring
Each setup's `technical_score` (1-10) should be computed as a weighted sum of the relevant score factors for that setup type. Normalize each factor to 0-1 before weighting. Document the weights in comments.

#### 5. Output
- Return `list[TechnicalSetup]` containing all detected setups across all tickers
- Sort by `technical_score` descending
- Log: total setups found, breakdown by setup type, top 10 tickers by score

#### 6. Helper Functions
Create utility functions in `src/utils/technical_helpers.py`:
- `find_swing_highs(series, window)` → list of (index, price) tuples
- `find_swing_lows(series, window)` → list of (index, price) tuples
- `compute_impulse_moves(df, min_pct, window)` → list of (start_idx, end_idx, pct_change)
- `detect_divergence(price_series, indicator_series, window)` → bool
- `consolidation_range(df, start_idx, end_idx)` → (low, high, range_pct)

### Tests: `tests/test_technical.py`
Write tests for:
- Each setup detector with synthetic OHLCV data that clearly matches (or clearly doesn't match) the pattern
- Scoring normalization produces values in 1-10 range
- Helper functions: swing detection, divergence detection, impulse move detection
- Edge cases: ticker with insufficient data (< lookback_days), flat price action, extreme volatility

## Important Notes
- All thresholds from `TechnicalConfig`. No hardcoded numbers in detection logic.
- A ticker CAN produce multiple setups. That's fine — the scoring agent will pick the best one later.
- Use `pandas-ta` for indicator computation, but write the setup detection logic from scratch (pandas-ta doesn't have pattern detectors for these).
- The impulse move detection for bull flags should look backward from the current consolidation — don't require the impulse to be the most recent move.
- When computing support/resistance levels, prefer clearly-defined levels (swing lows/highs, MAs) over fuzzy zones. These levels will directly feed into stop loss and take profit calculations.
