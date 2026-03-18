from __future__ import annotations

from collections import Counter

from src.config import ScoringConfig
from src.types import TechnicalSetup, SentimentResult, TradePlay
from src.utils.logging import get_logger
from src.utils.trade_math import calculate_rr_ratio, atr_buffer, normalize_score

logger = get_logger(__name__)


class ScoringAgent:
    """Joins technical and sentiment data, computes trade parameters,
    scores, filters, and returns the final ranked list of TradePlay objects."""

    def __init__(self, config: ScoringConfig) -> None:
        self.config = config
        self.tp = config.trade_params  # shorthand for trade-param config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        setups: list[TechnicalSetup],
        sentiments: list[SentimentResult],
    ) -> tuple[list[TradePlay], dict]:
        """Score, filter, rank and return final trade plays with metadata."""
        logger.info(
            "Scoring agent starting — %d setups, %d sentiment results",
            len(setups),
            len(sentiments),
        )

        sentiment_map = {s.ticker: s for s in sentiments}
        plays: list[TradePlay] = []
        errors = 0

        for setup in setups:
            sentiment = sentiment_map.get(setup.ticker)
            if sentiment is None:
                sentiment = self._default_sentiment(setup.ticker)
            try:
                play = self._build_trade_play(setup, sentiment)
                if play is not None:
                    plays.append(play)
            except Exception:
                logger.exception("Error building play for %s", setup.ticker)
                errors += 1

        logger.info("Built %d candidate plays (%d errors)", len(plays), errors)

        # --- Filtering ---
        plays, drop_counts = self._apply_filters(plays, sentiment_map)

        # --- Dedup: keep best per ticker ---
        plays = self._deduplicate(plays)

        # --- Sort & cap ---
        plays.sort(key=lambda p: p.composite_score, reverse=True)
        plays = plays[: self.config.max_play_count]

        metadata = self._build_metadata(
            total_candidates=len(setups),
            final_count=len(plays),
            drop_counts=drop_counts,
            plays=plays,
            errors=errors,
        )

        self._log_summary(plays, metadata)
        return plays, metadata

    # ------------------------------------------------------------------
    # Trade parameter calculation
    # ------------------------------------------------------------------

    def _build_trade_play(
        self, setup: TechnicalSetup, sentiment: SentimentResult
    ) -> TradePlay | None:
        atr = setup.indicators.get("atr_14", 0.0)
        if atr <= 0:
            logger.debug("Skipping %s — ATR is zero or missing", setup.ticker)
            return None

        entry = self._calc_entry(setup, atr)
        stop = self._calc_stop(setup, atr)
        if entry <= stop:
            logger.debug(
                "Skipping %s — entry (%.2f) <= stop (%.2f)", setup.ticker, entry, stop
            )
            return None

        risk = entry - stop
        tp1 = self._calc_tp1(setup, entry, risk)
        tp2 = self._calc_tp2(setup, entry, risk)

        rr1 = calculate_rr_ratio(entry, stop, tp1)
        rr2 = calculate_rr_ratio(entry, stop, tp2) if tp2 is not None else None

        position_risk_pct = (risk / entry) * 100.0
        sentiment_score = self._sentiment_to_score(sentiment)

        # --- Composite scoring ---
        # rr_score: R:R of 2.0 → 5, 3.0 → 7.5, 4.0+ → 10, <2.0 → linear 0-5
        rr_score = self._rr_to_score(rr1)

        # composite = weighted sum normalised to 0-10
        # technical_score is already 1-10, sentiment_score 1-10, rr_score 0-10
        max_possible = (
            10.0 * self.config.technical_weight
            + 10.0 * self.config.sentiment_weight
            + 10.0 * self.config.rr_weight
        )
        composite = (
            setup.technical_score * self.config.technical_weight
            + sentiment_score * self.config.sentiment_weight
            + rr_score * self.config.rr_weight
        ) * 10.0 / max_possible

        notes_parts: list[str] = []
        if position_risk_pct > self.tp.wide_stop_warning_pct:
            notes_parts.append("wide_stop")
        if setup.notes:
            notes_parts.append(setup.notes)

        return TradePlay(
            ticker=setup.ticker,
            setup_type=setup.setup_type,
            direction=setup.direction,
            entry_price=round(entry, 2),
            stop_loss=round(stop, 2),
            take_profit_1=round(tp1, 2),
            take_profit_2=round(tp2, 2) if tp2 is not None else None,
            risk_reward_ratio=round(rr1, 2),
            risk_reward_ratio_tp2=round(rr2, 2) if rr2 is not None else None,
            position_risk_pct=round(position_risk_pct, 2),
            technical_score=round(setup.technical_score, 2),
            sentiment_score=round(sentiment_score, 2),
            composite_score=round(composite, 2),
            sentiment_summary=sentiment.summary,
            catalysts=list(sentiment.catalysts),
            risk_flags=list(sentiment.risk_flags),
            notes="; ".join(notes_parts),
        )

    # --- Entry calculation per setup type ---

    def _calc_entry(self, setup: TechnicalSetup, atr: float) -> float:
        st = setup.setup_type
        if st == "squeeze_breakout":
            return setup.resistance_level + atr_buffer(atr, self.tp.breakout_entry_buffer)
        if st == "bull_flag":
            return setup.resistance_level + atr_buffer(atr, self.tp.flag_entry_buffer)
        if st == "mean_reversion":
            return setup.support_level + atr_buffer(atr, self.tp.mean_reversion_entry_buffer)
        if st == "trend_pullback":
            ema_21 = setup.indicators.get("ema_21", setup.indicators.get("close", 0.0))
            current = setup.indicators.get("close", ema_21)
            proximity = abs(current - ema_21) / ema_21 * 100.0 if ema_21 else 100.0
            if proximity <= self.tp.pullback_ema_proximity_pct:
                return current
            return ema_21
        # fallback
        return setup.resistance_level

    # --- Stop calculation per setup type ---

    def _calc_stop(self, setup: TechnicalSetup, atr: float) -> float:
        st = setup.setup_type
        if st == "squeeze_breakout":
            return setup.support_level - atr_buffer(atr, self.tp.breakout_stop_atr)
        if st == "bull_flag":
            return setup.support_level - atr_buffer(atr, self.tp.flag_stop_atr)
        if st == "mean_reversion":
            swing_low = setup.key_levels.get("swing_low", setup.support_level)
            return swing_low - atr_buffer(atr, self.tp.mean_reversion_stop_atr)
        if st == "trend_pullback":
            sma_50 = setup.indicators.get("sma_50", setup.support_level)
            return sma_50 - atr_buffer(atr, self.tp.pullback_stop_atr)
        return setup.support_level - atr_buffer(atr, 0.5)

    # --- Take-profit calculation ---

    def _calc_tp1(
        self, setup: TechnicalSetup, entry: float, risk: float
    ) -> float:
        """Conservative TP: 1.5R or next resistance, whichever is closer."""
        tp_from_risk = entry + risk * self.tp.tp1_risk_multiple
        next_resistance = setup.key_levels.get("next_resistance")
        if next_resistance is not None and next_resistance > entry:
            return min(tp_from_risk, next_resistance)
        return tp_from_risk

    def _calc_tp2(
        self, setup: TechnicalSetup, entry: float, risk: float
    ) -> float | None:
        """Aggressive TP based on measured-move or prior swing high."""
        st = setup.setup_type
        if st in ("squeeze_breakout", "bull_flag"):
            # Measured move: consolidation range projected from breakout
            consolidation_range = setup.resistance_level - setup.support_level
            if consolidation_range > 0:
                return entry + consolidation_range
        if st == "mean_reversion":
            swing_high = setup.key_levels.get("swing_high")
            if swing_high is not None and swing_high > entry:
                return swing_high
        if st == "trend_pullback":
            swing_high = setup.key_levels.get("swing_high")
            if swing_high is not None and swing_high > entry:
                return swing_high
        # Fallback: 2.5R
        return entry + risk * self.tp.tp2_default_risk_multiple

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _sentiment_to_score(sentiment: SentimentResult) -> float:
        """Map sentiment to a 1-10 score."""
        if sentiment.sentiment == "bullish":
            return sentiment.confidence
        if sentiment.sentiment == "neutral":
            return 5.0
        # bearish: high-confidence bearish → low score
        return 10.0 - sentiment.confidence

    @staticmethod
    def _rr_to_score(rr: float) -> float:
        """Normalise R:R to a 0-10 scale.

        R:R 2.0 → 5, 3.0 → 7.5, 4.0+ → 10. Below 2.0 → linear 0-5.
        """
        if rr >= 4.0:
            return 10.0
        if rr >= 2.0:
            return normalize_score(rr, 2.0, 4.0, target_min=5.0, target_max=10.0)
        return normalize_score(rr, 0.0, 2.0, target_min=0.0, target_max=5.0)

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def _apply_filters(
        self,
        plays: list[TradePlay],
        sentiment_map: dict[str, SentimentResult],
    ) -> tuple[list[TradePlay], dict[str, int]]:
        """Apply hard filters. Returns surviving plays and drop counts."""
        drop_counts: dict[str, int] = Counter()
        filtered: list[TradePlay] = []

        for play in plays:
            reason = self._should_drop(play, sentiment_map)
            if reason:
                drop_counts[reason] += 1
                logger.debug("Dropped %s — %s", play.ticker, reason)
            else:
                filtered.append(play)

        logger.info(
            "Filtering: %d → %d plays (dropped: %s)",
            len(plays),
            len(filtered),
            dict(drop_counts) or "none",
        )
        return filtered, dict(drop_counts)

    def _should_drop(
        self,
        play: TradePlay,
        sentiment_map: dict[str, SentimentResult],
    ) -> str | None:
        if play.risk_reward_ratio < self.config.min_risk_reward:
            return "low_rr"
        if play.composite_score < self.config.min_confidence_score:
            return "low_score"
        if play.position_risk_pct > self.tp.max_position_risk_pct:
            return "wide_stop"

        sentiment = sentiment_map.get(play.ticker)
        if sentiment is not None:
            if sentiment.sentiment == "bearish" and sentiment.confidence >= 7:
                return "bearish_headwind"
            if (
                "earnings_proximity" in sentiment.risk_flags
                and sentiment.days_to_earnings is not None
                and sentiment.days_to_earnings < self.config.min_earnings_gap_days
            ):
                return "earnings_proximity"

        return None

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(plays: list[TradePlay]) -> list[TradePlay]:
        """Keep only the highest-scoring play per ticker."""
        best: dict[str, TradePlay] = {}
        for play in plays:
            existing = best.get(play.ticker)
            if existing is None or play.composite_score > existing.composite_score:
                best[play.ticker] = play
        return list(best.values())

    # ------------------------------------------------------------------
    # Default sentiment
    # ------------------------------------------------------------------

    @staticmethod
    def _default_sentiment(ticker: str) -> SentimentResult:
        return SentimentResult(
            ticker=ticker,
            sentiment="neutral",
            confidence=5,
            summary="no_sentiment_data",
        )

    # ------------------------------------------------------------------
    # Metadata & logging
    # ------------------------------------------------------------------

    def _build_metadata(
        self,
        total_candidates: int,
        final_count: int,
        drop_counts: dict[str, int],
        plays: list[TradePlay],
        errors: int,
    ) -> dict:
        scores = [p.composite_score for p in plays]
        return {
            "total_candidates": total_candidates,
            "final_count": final_count,
            "dropped": drop_counts,
            "errors": errors,
            "score_distribution": {
                "min": round(min(scores), 2) if scores else 0,
                "max": round(max(scores), 2) if scores else 0,
                "avg": round(sum(scores) / len(scores), 2) if scores else 0,
            },
        }

    @staticmethod
    def _log_summary(plays: list[TradePlay], metadata: dict) -> None:
        if not plays:
            logger.info("No plays survived filtering. Metadata: %s", metadata)
            return

        avg_rr = sum(p.risk_reward_ratio for p in plays) / len(plays)
        avg_score = sum(p.composite_score for p in plays) / len(plays)
        setup_dist = Counter(p.setup_type for p in plays)

        logger.info(
            "Final plays: %d | avg R:R %.2f | avg score %.2f | setups: %s",
            len(plays),
            avg_rr,
            avg_score,
            dict(setup_dist),
        )
