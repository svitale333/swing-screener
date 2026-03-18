# Prompt 6: Orchestrator Agent

## Context
All four sub-agents are now implemented: Screener → Technical → Sentiment → Scoring. This step builds the **Orchestrator** — the main controller that wires the sub-agents together, runs the iterative loop, and manages the "don't stop until you have high-conviction plays" logic.

## What to Build

### Implement `src/orchestrator.py`

The `Orchestrator` class should:

#### 1. Initialization
- Accept a top-level `AgentConfig` (which contains all sub-configs)
- Instantiate all four sub-agents with their respective configs
- Set up logging with a run ID (timestamp-based, e.g., `run_20260317_0830`) for traceability

#### 2. Main Pipeline Method: `run() -> OrchestratorResult`

Create a new dataclass in `src/types.py`:
```python
@dataclass
class OrchestratorResult:
    plays: list[TradePlay]
    metadata: dict                # Aggregated metadata from all stages
    iterations: int               # How many loops it took
    run_id: str
    timestamp: str
    total_api_cost_estimate: float
    execution_time_seconds: float
```

#### 3. The Iterative Loop
This is the core logic. The orchestrator runs the pipeline and evaluates whether the output meets quality thresholds. If not, it adjusts parameters and re-runs.

```python
def run(self) -> OrchestratorResult:
    iteration = 0
    best_plays = []
    adjustment_history = []

    while iteration < self.config.max_iterations:
        iteration += 1
        log(f"=== Iteration {iteration}/{self.config.max_iterations} ===")

        # Stage 1: Screen (only on first iteration — universe doesn't change)
        if iteration == 1:
            candidates = self.screener.run()
        
        # Stage 2: Technical detection
        setups = self.technical.run(candidates)
        
        if not setups:
            log("No technical setups found. Adjusting parameters...")
            self._relax_technical_filters()
            adjustment_history.append({"iteration": iteration, "action": "relaxed_technical"})
            continue
        
        # Stage 3: Sentiment (only for top setups)
        top_setups = self._select_for_sentiment(setups)
        sentiment_results = self.sentiment.run(top_setups)
        
        # Stage 4: Score and rank
        plays, scoring_metadata = self.scoring.run(setups, sentiment_results)
        
        # Evaluate: are we satisfied?
        if self._is_satisfied(plays):
            best_plays = plays
            break
        
        # Not satisfied — keep best so far, adjust, and retry
        best_plays = self._merge_best(best_plays, plays)
        adjustments = self._compute_adjustments(plays, scoring_metadata)
        self._apply_adjustments(adjustments)
        adjustment_history.append({"iteration": iteration, "adjustments": adjustments})
    
    # If we exhausted iterations, return whatever we have
    return OrchestratorResult(
        plays=best_plays[:self.config.scoring.max_play_count],
        metadata={...},
        iterations=iteration,
        ...
    )
```

#### 4. Satisfaction Criteria: `_is_satisfied(plays) -> bool`
The orchestrator is "satisfied" when:
- `len(plays) >= target_play_count` (default 5)
- Average `composite_score` across plays >= `min_confidence_score` (default 6.0)
- At least 2 different setup types are represented (diversity)
- At least 2 different sectors are represented (concentration risk check)

All thresholds should be configurable. If ANY criterion fails, the loop continues (unless max iterations reached).

#### 5. Parameter Adjustment Logic: `_compute_adjustments()`
When the orchestrator isn't satisfied, it diagnoses WHY and makes targeted adjustments:

**Not enough plays (count below target):**
- Relax `min_risk_reward` by 0.25 (floor at 1.5)
- Relax `min_confidence_score` by 0.5 (floor at 4.0)
- Widen technical detection parameters (e.g., increase `max_consolidation_days`, relax `bb_squeeze_percentile`)

**Plays exist but scores are too low:**
- Don't relax scoring thresholds — instead, expand the technical detection to find more candidates, then re-run sentiment on the new additions only (avoid re-analyzing already-scored tickers)
- Increase the number of setups sent to sentiment analysis

**All plays are the same sector:**
- Apply a sector diversity penalty in the composite score: if > 3 plays from one sector, subtract 0.5 from each play in that sector beyond the 3rd

**All plays are the same setup type:**
- Similar diversity penalty: if > 3 plays of one setup type, subtract 0.3 from each beyond the 3rd

#### 6. Avoiding Redundant Work
- Track which tickers have already been analyzed for sentiment — don't re-call Claude for them on subsequent iterations
- Cache technical analysis results — only re-run technical detection if parameters changed
- The screener only runs once per orchestrator run (the universe doesn't change within a single execution)

#### 7. Merging Across Iterations: `_merge_best()`
- If iteration 2 finds new plays, merge them with iteration 1's plays
- Deduplicate by ticker (keep the highest-scoring version)
- Re-sort by composite score

#### 8. Timing and Cost Tracking
- Time each stage and the total run
- Aggregate API cost estimates from the sentiment agent
- Include all timing and cost data in `OrchestratorResult.metadata`

#### 9. Run Summary Logging
At the end of the run, log a rich summary table:
```
╔══════════════════════════════════════════════════╗
║          Swing Screener — Run Summary            ║
╠══════════════════════════════════════════════════╣
║ Run ID:       run_20260317_0830                  ║
║ Iterations:   2 / 3                              ║
║ Universe:     287 tickers screened                ║
║ Setups Found: 34 technical setups                ║
║ Sentiment:    18 tickers analyzed                ║
║ Final Plays:  6 (target: 5)                      ║
║ Avg R:R:      2.7:1                              ║
║ Avg Score:    7.2 / 10                           ║
║ API Cost:     ~$0.42                             ║
║ Total Time:   3m 24s                             ║
╚══════════════════════════════════════════════════╝
```

### Tests: `tests/test_orchestrator.py`
Write tests for:
- Happy path: mock all sub-agents to return good data, verify single-iteration completion
- Iteration trigger: mock sub-agents to return insufficient plays on first pass, verify loop runs again with relaxed params
- Max iteration cap: verify the loop stops after `max_iterations` even if not satisfied
- Merge logic: two iterations with overlapping tickers, verify deduplication
- Satisfaction criteria: test each criterion independently
- Adjustment logic: verify correct parameters are relaxed for each failure mode
- Sentiment cache: verify tickers aren't re-analyzed on subsequent iterations

## Important Notes
- The orchestrator should NEVER modify the original config objects. Clone/copy them before adjusting. This ensures each run starts from the base config.
- The satisfaction criteria should err on the side of "good enough" — the agent should not loop forever chasing perfection. If iteration 1 produces 4 solid plays and the target is 5, that's close enough after 2 iterations.
- The diversity checks (sector, setup type) are soft preferences, not hard filters. They influence the loop decision but shouldn't cause plays to be dropped from the final output.
- If the screener returns zero candidates (market holiday, data issue), the orchestrator should abort gracefully with a clear error message, not loop.
- Consider adding a `--dry-run` mode that runs screener + technical only (no API calls) for testing the pipeline.
