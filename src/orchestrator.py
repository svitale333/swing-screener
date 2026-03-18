from __future__ import annotations

import copy
import time
from collections import Counter
from datetime import datetime

from src.agents.screener import ScreenerAgent
from src.agents.scoring import ScoringAgent
from src.agents.sentiment import SentimentAgent
from src.agents.technical import TechnicalAgent
from src.config import AgentConfig
from src.types import (
    OrchestratorResult,
    ScreenedCandidate,
    SentimentResult,
    TechnicalSetup,
    TradePlay,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Adjustment floors — never relax below these
_MIN_RISK_REWARD_FLOOR = 1.5
_MIN_CONFIDENCE_FLOOR = 4.0

# Diversity penalty thresholds
_SECTOR_DIVERSITY_THRESHOLD = 3
_SECTOR_DIVERSITY_PENALTY = 0.5
_SETUP_DIVERSITY_THRESHOLD = 3
_SETUP_DIVERSITY_PENALTY = 0.3


class Orchestrator:
    """Coordinates the agent pipeline with an iterative loop."""

    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        # Work on copies so we never mutate the original config
        self._scoring_config = copy.deepcopy(config.scoring)
        self._technical_config = copy.deepcopy(config.technical)

        self.screener = ScreenerAgent(config.screener)
        self.technical = TechnicalAgent(self._technical_config)
        self.scoring = ScoringAgent(self._scoring_config)
        self.sentiment = SentimentAgent(config.sentiment)

        self.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M')}"

    # ------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------

    def run(self, dry_run: bool = False) -> OrchestratorResult:
        """Execute the full screening pipeline with iterative refinement.

        Args:
            dry_run: If True, skip sentiment analysis (no API calls).
        """
        start_time = time.time()
        iteration = 0
        best_plays: list[TradePlay] = []
        adjustment_history: list[dict] = []
        stage_timings: dict[str, float] = {}

        # Caches to avoid redundant work
        sentiment_cache: dict[str, SentimentResult] = {}
        candidates: list[ScreenedCandidate] = []
        ticker_sector_map: dict[str, str] = {}
        prev_technical_setups: list[TechnicalSetup] = []
        technical_params_changed = True

        while iteration < self.config.max_iterations:
            iteration += 1
            logger.info(f"=== Iteration {iteration}/{self.config.max_iterations} ===")

            # Stage 1: Screen (only on first iteration)
            if iteration == 1:
                t0 = time.time()
                candidates = self.screener.run()
                stage_timings["screener"] = time.time() - t0
                logger.info(f"Screener: {len(candidates)} candidates ({stage_timings['screener']:.1f}s)")

                if not candidates:
                    logger.warning("Screener returned zero candidates — aborting run")
                    break

                ticker_sector_map = {c.ticker: c.sector for c in candidates}

            # Stage 2: Technical detection (re-run only if params changed)
            if technical_params_changed:
                t0 = time.time()
                setups = self.technical.run(candidates)
                elapsed = time.time() - t0
                stage_timings[f"technical_iter{iteration}"] = elapsed
                logger.info(f"Technical: {len(setups)} setups ({elapsed:.1f}s)")
                prev_technical_setups = setups
                technical_params_changed = False
            else:
                setups = prev_technical_setups

            if not setups:
                logger.info("No technical setups found. Adjusting parameters...")
                self._relax_technical_filters()
                technical_params_changed = True
                adjustment_history.append(
                    {"iteration": iteration, "action": "relaxed_technical"}
                )
                continue

            # Stage 3: Sentiment (skip in dry_run, only for new tickers)
            if dry_run:
                sentiment_results = list(sentiment_cache.values())
            else:
                top_setups = self._select_for_sentiment(setups, sentiment_cache)
                if top_setups:
                    t0 = time.time()
                    new_results = self.sentiment.run(top_setups)
                    elapsed = time.time() - t0
                    stage_timings[f"sentiment_iter{iteration}"] = elapsed
                    logger.info(f"Sentiment: {len(new_results)} new results ({elapsed:.1f}s)")
                    for r in new_results:
                        sentiment_cache[r.ticker] = r
                sentiment_results = list(sentiment_cache.values())

            # Stage 4: Score and rank
            t0 = time.time()
            # Rebuild scoring agent with potentially adjusted config
            self.scoring = ScoringAgent(self._scoring_config)
            plays, scoring_metadata = self.scoring.run(setups, sentiment_results)
            elapsed = time.time() - t0
            stage_timings[f"scoring_iter{iteration}"] = elapsed

            # Apply diversity penalties
            plays = self._apply_diversity_penalties(plays, ticker_sector_map)

            # Re-sort after penalties
            plays.sort(key=lambda p: p.composite_score, reverse=True)

            # Evaluate satisfaction
            if self._is_satisfied(plays, ticker_sector_map):
                best_plays = plays
                logger.info(f"Satisfied after iteration {iteration}")
                break

            # Not satisfied — merge with previous best, adjust, retry
            best_plays = self._merge_best(best_plays, plays)
            adjustments = self._compute_adjustments(
                plays, scoring_metadata, ticker_sector_map
            )
            self._apply_adjustments(adjustments)
            if adjustments.get("relaxed_technical"):
                technical_params_changed = True
            adjustment_history.append(
                {"iteration": iteration, "adjustments": adjustments}
            )

        # Final output: take whatever we have
        final_plays = best_plays[: self._scoring_config.max_play_count]

        # Estimate API cost
        total_cost = self._estimate_api_cost()

        total_time = time.time() - start_time

        result = OrchestratorResult(
            plays=final_plays,
            metadata={
                "universe_size": len(candidates),
                "total_setups": len(prev_technical_setups),
                "sentiment_analyzed": len(sentiment_cache),
                "adjustment_history": adjustment_history,
                "stage_timings": stage_timings,
            },
            iterations=iteration,
            run_id=self.run_id,
            timestamp=datetime.now().isoformat(),
            total_api_cost_estimate=total_cost,
            execution_time_seconds=round(total_time, 2),
        )

        self._log_run_summary(result)
        return result

    # ------------------------------------------------------------------
    # Satisfaction criteria
    # ------------------------------------------------------------------

    def _is_satisfied(
        self,
        plays: list[TradePlay],
        ticker_sector_map: dict[str, str],
    ) -> bool:
        """Check if current plays meet quality thresholds."""
        target = self._scoring_config.target_play_count
        min_score = self._scoring_config.min_confidence_score

        # Enough plays?
        if len(plays) < target:
            logger.info(f"Not satisfied: {len(plays)} plays < target {target}")
            return False

        # Average composite score high enough?
        avg_score = sum(p.composite_score for p in plays) / len(plays) if plays else 0
        if avg_score < min_score:
            logger.info(f"Not satisfied: avg score {avg_score:.2f} < {min_score}")
            return False

        # Setup type diversity: at least 2 different types
        setup_types = {p.setup_type for p in plays}
        if len(setup_types) < 2:
            logger.info(f"Not satisfied: only {len(setup_types)} setup type(s)")
            return False

        # Sector diversity: at least 2 different sectors
        sectors = {ticker_sector_map.get(p.ticker, "Unknown") for p in plays}
        if len(sectors) < 2:
            logger.info(f"Not satisfied: only {len(sectors)} sector(s)")
            return False

        return True

    # ------------------------------------------------------------------
    # Parameter adjustment logic
    # ------------------------------------------------------------------

    def _compute_adjustments(
        self,
        plays: list[TradePlay],
        scoring_metadata: dict,
        ticker_sector_map: dict[str, str],
    ) -> dict:
        """Diagnose why we're unsatisfied and compute targeted adjustments."""
        adjustments: dict = {}
        target = self._scoring_config.target_play_count

        # Not enough plays
        if len(plays) < target:
            if self._scoring_config.min_risk_reward > _MIN_RISK_REWARD_FLOOR:
                adjustments["relax_min_rr"] = max(
                    self._scoring_config.min_risk_reward - 0.25,
                    _MIN_RISK_REWARD_FLOOR,
                )
            if self._scoring_config.min_confidence_score > _MIN_CONFIDENCE_FLOOR:
                adjustments["relax_min_confidence"] = max(
                    self._scoring_config.min_confidence_score - 0.5,
                    _MIN_CONFIDENCE_FLOOR,
                )
            adjustments["relaxed_technical"] = True

        # Plays exist but scores too low
        avg_score = (
            sum(p.composite_score for p in plays) / len(plays) if plays else 0
        )
        if plays and avg_score < self._scoring_config.min_confidence_score:
            adjustments["relaxed_technical"] = True
            adjustments["expand_sentiment_pool"] = True

        # Sector concentration
        sector_counts = Counter(
            ticker_sector_map.get(p.ticker, "Unknown") for p in plays
        )
        for sector, count in sector_counts.items():
            if count > _SECTOR_DIVERSITY_THRESHOLD:
                adjustments["sector_concentration"] = sector

        # Setup type concentration
        type_counts = Counter(p.setup_type for p in plays)
        for stype, count in type_counts.items():
            if count > _SETUP_DIVERSITY_THRESHOLD:
                adjustments["setup_concentration"] = stype

        return adjustments

    def _apply_adjustments(self, adjustments: dict) -> None:
        """Apply computed adjustments to working config copies."""
        if "relax_min_rr" in adjustments:
            self._scoring_config.min_risk_reward = adjustments["relax_min_rr"]
            logger.info(
                f"Relaxed min_risk_reward to {self._scoring_config.min_risk_reward}"
            )

        if "relax_min_confidence" in adjustments:
            self._scoring_config.min_confidence_score = adjustments[
                "relax_min_confidence"
            ]
            logger.info(
                f"Relaxed min_confidence_score to "
                f"{self._scoring_config.min_confidence_score}"
            )

        if adjustments.get("relaxed_technical"):
            self._relax_technical_filters()

    def _relax_technical_filters(self) -> None:
        """Widen technical detection parameters to find more setups."""
        tc = self._technical_config
        tc.max_consolidation_days = min(tc.max_consolidation_days + 10, 60)
        tc.bb_squeeze_percentile = min(tc.bb_squeeze_percentile + 5.0, 40.0)
        tc.rsi_oversold = min(tc.rsi_oversold + 5.0, 40.0)
        tc.volume_dryup_threshold = min(tc.volume_dryup_threshold + 0.1, 0.8)
        # Rebuild the technical agent with updated config
        self.technical = TechnicalAgent(self._technical_config)
        logger.info(
            f"Relaxed technical filters: max_consol={tc.max_consolidation_days}, "
            f"bb_squeeze={tc.bb_squeeze_percentile}, rsi_oversold={tc.rsi_oversold}"
        )

    # ------------------------------------------------------------------
    # Diversity penalties
    # ------------------------------------------------------------------

    def _apply_diversity_penalties(
        self,
        plays: list[TradePlay],
        ticker_sector_map: dict[str, str],
    ) -> list[TradePlay]:
        """Apply soft diversity penalties to composite scores."""
        if not plays:
            return plays

        # Sector penalties: if > 3 plays from one sector, penalize extras
        sector_counts: Counter[str] = Counter()
        for play in plays:
            sector = ticker_sector_map.get(play.ticker, "Unknown")
            sector_counts[sector] += 1
            if sector_counts[sector] > _SECTOR_DIVERSITY_THRESHOLD:
                play.composite_score = max(
                    0, play.composite_score - _SECTOR_DIVERSITY_PENALTY
                )

        # Setup type penalties
        type_counts: Counter[str] = Counter()
        for play in plays:
            type_counts[play.setup_type] += 1
            if type_counts[play.setup_type] > _SETUP_DIVERSITY_THRESHOLD:
                play.composite_score = max(
                    0, play.composite_score - _SETUP_DIVERSITY_PENALTY
                )

        return plays

    # ------------------------------------------------------------------
    # Sentiment selection & caching
    # ------------------------------------------------------------------

    def _select_for_sentiment(
        self,
        setups: list[TechnicalSetup],
        sentiment_cache: dict[str, SentimentResult],
    ) -> list[TechnicalSetup]:
        """Select top setups that haven't been sentiment-analyzed yet."""
        new_setups = [s for s in setups if s.ticker not in sentiment_cache]
        # Deduplicate by ticker, keeping highest technical score
        best_per_ticker: dict[str, TechnicalSetup] = {}
        for s in sorted(new_setups, key=lambda x: x.technical_score, reverse=True):
            if s.ticker not in best_per_ticker:
                best_per_ticker[s.ticker] = s
        return list(best_per_ticker.values())

    # ------------------------------------------------------------------
    # Merge across iterations
    # ------------------------------------------------------------------

    def _merge_best(
        self,
        existing: list[TradePlay],
        new: list[TradePlay],
    ) -> list[TradePlay]:
        """Merge plays from multiple iterations, dedup by ticker (keep best)."""
        best: dict[str, TradePlay] = {}
        for play in existing + new:
            current = best.get(play.ticker)
            if current is None or play.composite_score > current.composite_score:
                best[play.ticker] = play

        merged = sorted(best.values(), key=lambda p: p.composite_score, reverse=True)
        return merged

    # ------------------------------------------------------------------
    # Cost estimation
    # ------------------------------------------------------------------

    def _estimate_api_cost(self) -> float:
        """Estimate total API cost from the sentiment agent's usage."""
        input_tokens = self.sentiment._total_input_tokens
        output_tokens = self.sentiment._total_output_tokens

        from src.agents.sentiment import _MODEL_PRICING, _DEFAULT_PRICING

        pricing = _MODEL_PRICING.get(self.config.sentiment.model, _DEFAULT_PRICING)
        cost = (
            input_tokens / 1_000_000 * pricing["input"]
            + output_tokens / 1_000_000 * pricing["output"]
        )
        return round(cost, 4)

    # ------------------------------------------------------------------
    # Run summary
    # ------------------------------------------------------------------

    def _log_run_summary(self, result: OrchestratorResult) -> None:
        """Log a rich summary table at the end of the run."""
        plays = result.plays
        avg_rr = (
            sum(p.risk_reward_ratio for p in plays) / len(plays) if plays else 0
        )
        avg_score = (
            sum(p.composite_score for p in plays) / len(plays) if plays else 0
        )
        total_time = result.execution_time_seconds
        minutes = int(total_time // 60)
        seconds = int(total_time % 60)

        universe = result.metadata.get("universe_size", 0)
        total_setups = result.metadata.get("total_setups", 0)
        sentiment_count = result.metadata.get("sentiment_analyzed", 0)
        target = self._scoring_config.target_play_count

        summary = (
            f"\n"
            f"{'=' * 52}\n"
            f"          Swing Screener - Run Summary\n"
            f"{'=' * 52}\n"
            f" Run ID:       {result.run_id}\n"
            f" Iterations:   {result.iterations} / {self.config.max_iterations}\n"
            f" Universe:     {universe} tickers screened\n"
            f" Setups Found: {total_setups} technical setups\n"
            f" Sentiment:    {sentiment_count} tickers analyzed\n"
            f" Final Plays:  {len(plays)} (target: {target})\n"
            f" Avg R:R:      {avg_rr:.1f}:1\n"
            f" Avg Score:    {avg_score:.1f} / 10\n"
            f" API Cost:     ~${result.total_api_cost_estimate:.2f}\n"
            f" Total Time:   {minutes}m {seconds:02d}s\n"
            f"{'=' * 52}"
        )
        logger.info(summary)
