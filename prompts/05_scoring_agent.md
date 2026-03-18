# Prompt 5: Scoring & Risk/Reward Agent

## Context
At this point we have `list[TechnicalSetup]` (with support/resistance levels and technical scores) and `list[SentimentResult]` (with sentiment, catalysts, risk flags). This step implements the **Scoring Agent** — the module that joins technical and sentiment data, computes concrete entry/stop/target levels, calculates risk-reward ratios, and produces the final ranked list of `TradePlay` objects.

## What to Build

### Implement `src/agents/scoring.py`

The `ScoringAgent` class should:

#### 1. Data Joining
- Accept `list[TechnicalSetup]` and `list[SentimentResult]` as inputs
- Join on ticker. If a ticker has multiple technical setups, evaluate each independently (same sentiment, different entries/stops).
- If a ticker has a technical setup but no sentiment result (e.g., sentiment agent failed for that ticker), assign default neutral sentiment with confidence 5 and a note "no_sentiment_data"

#### 2. Trade Parameter Calculation
For each (TechnicalSetup, SentimentResult) pair, compute:

**Entry Price:**
- `squeeze_breakout`: Entry = resistance level + (ATR * 0.1) as breakout confirmation buffer
- `bull_flag`: Entry = top of consolidation range + (ATR * 0.1)
- `mean_reversion`: Entry = current price (or a limit at the support level + ATR * 0.2 for a pullback entry)
- `trend_pullback`: Entry = current price when within 1.5% of 21 EMA, or the 21 EMA itself as a limit

**Stop Loss:**
- `squeeze_breakout`: Stop = support level - (ATR * 0.5). This is below the consolidation low with ATR cushion.
- `bull_flag`: Stop = bottom of consolidation range - (ATR * 0.5). Below the flag low.
- `mean_reversion`: Stop = swing low - (ATR * 0.75). Wider stop for mean reversion since these can retest.
- `trend_pullback`: Stop = 50 SMA - (ATR * 0.3). Below the next major MA support.

**Take Profit 1 (conservative, ~1.5-2R):**
- Measured from entry at 1.5x the risk distance (entry - stop), or the next resistance level — whichever is closer
- This is the "take half off" level

**Take Profit 2 (aggressive, ~2.5-3R+):**
- Measured move target: for breakouts, the height of the consolidation range added to the breakout point
- For flags: the impulse move length projected from the breakout
- For mean reversion: the prior swing high
- For pullbacks: the prior swing high or a 2.5R target

**Risk/Reward Ratio:**
- `rr_ratio = (take_profit_1 - entry) / (entry - stop_loss)` for the conservative target
- Also compute for TP2 and store both

**Position Risk %:**
- `position_risk_pct = (entry - stop_loss) / entry * 100`
- If this exceeds 8%, flag it as wide stop — the setup may not be suitable for standard position sizing

#### 3. Composite Scoring
Compute a `composite_score` (0-10) using weights from `ScoringConfig`:

```python
composite = (
    technical_score * config.technical_weight +
    sentiment_modifier * config.sentiment_weight +
    rr_score * config.rr_weight
) * 10 / max_possible
```

Where:
- `technical_score`: Raw score from the technical agent (already 1-10)
- `sentiment_modifier`: Map sentiment to a multiplier:
  - bullish + high confidence: sentiment_result.confidence (use directly, 1-10)
  - neutral: 5.0
  - bearish: 10 - sentiment_result.confidence (inverts — high-confidence bearish = low score)
- `rr_score`: Normalize R:R to 1-10 scale. R:R of 2.0 = 5, R:R of 3.0 = 7.5, R:R of 4.0+ = 10. Below 2.0 = linear scale from 0-5.

#### 4. Filtering
Apply hard filters to remove plays that don't meet minimum criteria:
- `risk_reward_ratio` < `ScoringConfig.min_risk_reward` (default 2.0) → DROP
- `composite_score` < `ScoringConfig.min_confidence_score` (default 6.0) → DROP
- Sentiment is "bearish" with confidence >= 7 → DROP (strong headwind)
- `position_risk_pct` > 10% → DROP (stop is too wide for responsible sizing)
- Has risk flag "earnings_proximity" with `days_to_earnings` < `min_earnings_gap_days` → DROP

#### 5. Ranking and Selection
- Sort remaining plays by `composite_score` descending
- If multiple setups exist for the same ticker, keep only the highest-scoring one
- Cap output at `ScoringConfig.max_play_count` (default 10)
- Return the list along with a `metadata` dict containing: total candidates evaluated, dropped counts by filter reason, score distribution

#### 6. Output
- Return `list[TradePlay]` (the final plays) and `dict` (metadata)
- Each `TradePlay` should have all fields fully populated — this is the final output format
- Log: final play count, average R:R, average composite score, setup type distribution

### Create `src/utils/trade_math.py`
Utility functions for trade calculations:
- `calculate_rr_ratio(entry, stop, target)` → float
- `calculate_position_size(account_size, risk_pct, entry, stop)` → shares (for optional position sizing output)
- `atr_buffer(atr_value, multiplier)` → float
- `normalize_score(value, min_val, max_val, target_min=1, target_max=10)` → float

### Tests: `tests/test_scoring.py`
Write tests for:
- Entry/stop/target calculation for each setup type (use concrete price examples)
- R:R calculation correctness
- Composite score with known inputs produces expected output
- Hard filters: verify each filter drops the correct plays
- Deduplication: same ticker with two setups keeps only the best
- Edge cases: all plays filtered out (should return empty list with metadata explaining why)
- Position risk % calculation

## Important Notes
- All ATR multipliers for stop/entry buffers should be in config, not hardcoded. Add them to `ScoringConfig` or a new `TradeParamsConfig` dataclass.
- The R:R ratio should use TP1 (conservative target) as the primary metric. TP2 is the upside optionality.
- When a play is dropped by a filter, the metadata dict should track WHY. This feeds back to the orchestrator so it can decide whether to relax filters on the next iteration.
- The composite score formula should be documented clearly in comments — this is the most important tunable in the system and will be iterated frequently.
- Consider adding a `TradePlay.to_report_dict()` method that formats the play for human-readable output (prices formatted to 2 decimal places, percentages formatted, etc.).
