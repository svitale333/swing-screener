from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.config import AgentConfig, ScoringConfig, TechnicalConfig
from src.orchestrator import Orchestrator
from src.types import (
    OrchestratorResult,
    ScreenedCandidate,
    SentimentResult,
    TechnicalSetup,
    TradePlay,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_candidate(
    ticker: str = "AAPL",
    sector: str = "Technology",
    price: float = 180.0,
) -> ScreenedCandidate:
    return ScreenedCandidate(
        ticker=ticker,
        price=price,
        avg_volume=5_000_000.0,
        market_cap=2_000_000_000.0,
        sector=sector,
        industry="Software",
    )


def _make_setup(
    ticker: str = "AAPL",
    setup_type: str = "squeeze_breakout",
    technical_score: float = 8.0,
) -> TechnicalSetup:
    return TechnicalSetup(
        ticker=ticker,
        setup_type=setup_type,
        direction="long",
        technical_score=technical_score,
        support_level=170.0,
        resistance_level=185.0,
        key_levels={},
        indicators={"atr_14": 3.0, "close": 182.0, "ema_21": 180.0, "sma_50": 175.0},
        notes="test setup",
    )


def _make_sentiment(
    ticker: str = "AAPL",
    sentiment: str = "bullish",
    confidence: float = 8.0,
) -> SentimentResult:
    return SentimentResult(
        ticker=ticker,
        sentiment=sentiment,
        confidence=confidence,
        catalysts=["catalyst"],
        risk_flags=[],
        summary="bullish outlook",
    )


def _make_play(
    ticker: str = "AAPL",
    setup_type: str = "squeeze_breakout",
    composite_score: float = 7.5,
    risk_reward: float = 2.5,
) -> TradePlay:
    return TradePlay(
        ticker=ticker,
        setup_type=setup_type,
        direction="long",
        entry_price=185.30,
        stop_loss=168.50,
        take_profit_1=210.50,
        take_profit_2=200.30,
        risk_reward_ratio=risk_reward,
        risk_reward_ratio_tp2=1.89,
        position_risk_pct=5.0,
        technical_score=8.0,
        sentiment_score=8.0,
        composite_score=composite_score,
        sentiment_summary="bullish",
        catalysts=["catalyst"],
        risk_flags=[],
    )


def _diverse_plays(count: int = 6) -> list[TradePlay]:
    """Generate plays with diverse sectors and setup types."""
    tickers = ["AAPL", "MSFT", "XOM", "JPM", "JNJ", "AMZN", "GOOGL", "NVDA"]
    sectors = [
        "Technology", "Technology", "Energy", "Financials",
        "Healthcare", "Consumer", "Technology", "Technology",
    ]
    types = [
        "squeeze_breakout", "bull_flag", "mean_reversion", "trend_pullback",
        "squeeze_breakout", "bull_flag", "mean_reversion", "trend_pullback",
    ]
    return [
        _make_play(
            ticker=tickers[i],
            setup_type=types[i],
            composite_score=8.0 - i * 0.1,
        )
        for i in range(count)
    ]


def _diverse_candidates(count: int = 6) -> list[ScreenedCandidate]:
    """Generate candidates with diverse sectors."""
    tickers = ["AAPL", "MSFT", "XOM", "JPM", "JNJ", "AMZN", "GOOGL", "NVDA"]
    sectors = [
        "Technology", "Technology", "Energy", "Financials",
        "Healthcare", "Consumer", "Technology", "Technology",
    ]
    return [_make_candidate(tickers[i], sectors[i]) for i in range(count)]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> AgentConfig:
    """Relaxed config for tests."""
    return AgentConfig(
        scoring=ScoringConfig(
            min_risk_reward=0.0,
            min_confidence_score=0.0,
            target_play_count=5,
            max_play_count=10,
        ),
        max_iterations=3,
    )


@pytest.fixture
def orchestrator(config: AgentConfig) -> Orchestrator:
    return Orchestrator(config)


# ---------------------------------------------------------------------------
# Happy path: single iteration completion
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_single_iteration_satisfied(self, config: AgentConfig):
        """Mock all agents to return good data, verify single-iteration completion."""
        orch = Orchestrator(config)

        candidates = _diverse_candidates(6)
        setups = [_make_setup(c.ticker, types[i]) for i, c in enumerate(candidates)
                  for types in [["squeeze_breakout", "bull_flag", "mean_reversion",
                                 "trend_pullback", "squeeze_breakout", "bull_flag"]]]
        sentiments = [_make_sentiment(c.ticker) for c in candidates]

        # Create diverse plays that satisfy all criteria
        good_plays = _diverse_plays(6)
        scoring_metadata = {"total_candidates": 6, "final_count": 6, "dropped": {}, "errors": 0,
                            "score_distribution": {"min": 7.5, "max": 8.0, "avg": 7.75}}

        with patch.object(orch.screener, "run", return_value=candidates), \
             patch.object(orch.technical, "run", return_value=setups), \
             patch.object(orch.sentiment, "run", return_value=sentiments), \
             patch.object(orch.scoring, "run", return_value=(good_plays, scoring_metadata)):

            result = orch.run()

        assert isinstance(result, OrchestratorResult)
        assert result.iterations == 1
        assert len(result.plays) > 0
        assert result.run_id.startswith("run_")

    def test_dry_run_skips_sentiment(self, config: AgentConfig):
        """Dry run should not call sentiment agent."""
        orch = Orchestrator(config)

        candidates = _diverse_candidates(6)
        setups = [_make_setup(c.ticker) for c in candidates]
        good_plays = _diverse_plays(6)
        scoring_metadata = {"total_candidates": 6, "final_count": 6, "dropped": {},
                            "errors": 0, "score_distribution": {"min": 7.5, "max": 8.0, "avg": 7.75}}

        with patch.object(orch.screener, "run", return_value=candidates), \
             patch.object(orch.technical, "run", return_value=setups), \
             patch.object(orch.sentiment, "run") as mock_sentiment, \
             patch.object(orch.scoring, "run", return_value=(good_plays, scoring_metadata)):

            result = orch.run(dry_run=True)

        mock_sentiment.assert_not_called()
        assert isinstance(result, OrchestratorResult)


# ---------------------------------------------------------------------------
# Iteration trigger: loop runs again with relaxed params
# ---------------------------------------------------------------------------


class TestIterationTrigger:
    def test_loops_when_insufficient_plays(self, config: AgentConfig):
        """When first iteration returns too few plays, orchestrator should loop."""
        config.scoring.target_play_count = 5
        config.scoring.min_confidence_score = 6.0
        orch = Orchestrator(config)

        candidates = _diverse_candidates(6)
        setups = [_make_setup(c.ticker) for c in candidates]

        # First call: only 2 plays (insufficient)
        few_plays = _diverse_plays(2)
        # Second call: 6 plays (satisfied)
        many_plays = _diverse_plays(6)
        scoring_metadata = {"total_candidates": 6, "final_count": 6, "dropped": {},
                            "errors": 0, "score_distribution": {"min": 7.5, "max": 8.0, "avg": 7.75}}

        call_count = {"n": 0}

        def mock_scoring_run(setups_arg, sentiments_arg):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return few_plays, scoring_metadata
            return many_plays, scoring_metadata

        with patch.object(orch.screener, "run", return_value=candidates), \
             patch.object(orch.technical, "run", return_value=setups), \
             patch.object(orch.sentiment, "run", return_value=[_make_sentiment(c.ticker) for c in candidates]):

            # We need to patch every ScoringAgent that gets created
            with patch("src.orchestrator.ScoringAgent") as MockScoring:
                mock_instance = MagicMock()
                mock_instance.run.side_effect = mock_scoring_run
                MockScoring.return_value = mock_instance

                result = orch.run()

        assert result.iterations >= 2


# ---------------------------------------------------------------------------
# Max iteration cap
# ---------------------------------------------------------------------------


class TestMaxIterationCap:
    def test_stops_at_max_iterations(self):
        """Loop should stop after max_iterations even if not satisfied."""
        config = AgentConfig(
            scoring=ScoringConfig(
                min_risk_reward=0.0,
                min_confidence_score=0.0,
                target_play_count=100,  # impossible to satisfy
            ),
            max_iterations=2,
        )
        orch = Orchestrator(config)

        candidates = _diverse_candidates(3)
        setups = [_make_setup(c.ticker) for c in candidates]
        plays = [_make_play(c.ticker) for c in candidates]
        scoring_metadata = {"total_candidates": 3, "final_count": 3, "dropped": {},
                            "errors": 0, "score_distribution": {"min": 7.5, "max": 7.5, "avg": 7.5}}

        with patch.object(orch.screener, "run", return_value=candidates), \
             patch.object(orch.technical, "run", return_value=setups), \
             patch.object(orch.sentiment, "run", return_value=[_make_sentiment(c.ticker) for c in candidates]):

            with patch("src.orchestrator.ScoringAgent") as MockScoring:
                mock_instance = MagicMock()
                mock_instance.run.return_value = (plays, scoring_metadata)
                MockScoring.return_value = mock_instance

                result = orch.run()

        assert result.iterations == 2


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------


class TestMergeLogic:
    def test_merge_deduplicates_by_ticker(self, orchestrator: Orchestrator):
        """Two iterations with overlapping tickers should keep highest score."""
        play1 = _make_play("AAPL", composite_score=7.0)
        play2 = _make_play("AAPL", composite_score=8.0)
        play3 = _make_play("MSFT", composite_score=6.0)

        merged = orchestrator._merge_best([play1, play3], [play2])

        tickers = [p.ticker for p in merged]
        assert tickers.count("AAPL") == 1
        assert tickers.count("MSFT") == 1

        aapl = next(p for p in merged if p.ticker == "AAPL")
        assert aapl.composite_score == 8.0

    def test_merge_sorted_by_score(self, orchestrator: Orchestrator):
        """Merged result should be sorted by composite_score descending."""
        plays1 = [_make_play("AAPL", composite_score=6.0)]
        plays2 = [_make_play("MSFT", composite_score=9.0)]

        merged = orchestrator._merge_best(plays1, plays2)
        assert merged[0].ticker == "MSFT"
        assert merged[1].ticker == "AAPL"


# ---------------------------------------------------------------------------
# Satisfaction criteria
# ---------------------------------------------------------------------------


class TestSatisfactionCriteria:
    def test_satisfied_with_good_plays(self, orchestrator: Orchestrator):
        """Plays meeting all criteria should satisfy."""
        plays = _diverse_plays(6)
        sector_map = {
            "AAPL": "Technology", "MSFT": "Technology", "XOM": "Energy",
            "JPM": "Financials", "JNJ": "Healthcare", "AMZN": "Consumer",
        }
        assert orchestrator._is_satisfied(plays, sector_map)

    def test_not_satisfied_too_few_plays(self, orchestrator: Orchestrator):
        """Too few plays should not satisfy."""
        plays = _diverse_plays(2)
        sector_map = {"AAPL": "Technology", "MSFT": "Technology"}
        assert not orchestrator._is_satisfied(plays, sector_map)

    def test_not_satisfied_low_scores(self):
        """Average score below threshold should not satisfy."""
        config = AgentConfig(
            scoring=ScoringConfig(
                min_confidence_score=9.0,
                target_play_count=2,
            ),
        )
        orch = Orchestrator(config)
        plays = [
            _make_play("AAPL", setup_type="squeeze_breakout", composite_score=5.0),
            _make_play("XOM", setup_type="bull_flag", composite_score=5.0),
        ]
        sector_map = {"AAPL": "Technology", "XOM": "Energy"}
        assert not orch._is_satisfied(plays, sector_map)

    def test_not_satisfied_single_setup_type(self):
        """All same setup type should not satisfy."""
        config = AgentConfig(
            scoring=ScoringConfig(
                min_confidence_score=0.0,
                target_play_count=2,
            ),
        )
        orch = Orchestrator(config)
        plays = [
            _make_play("AAPL", setup_type="squeeze_breakout", composite_score=8.0),
            _make_play("XOM", setup_type="squeeze_breakout", composite_score=8.0),
        ]
        sector_map = {"AAPL": "Technology", "XOM": "Energy"}
        assert not orch._is_satisfied(plays, sector_map)

    def test_not_satisfied_single_sector(self):
        """All same sector should not satisfy."""
        config = AgentConfig(
            scoring=ScoringConfig(
                min_confidence_score=0.0,
                target_play_count=2,
            ),
        )
        orch = Orchestrator(config)
        plays = [
            _make_play("AAPL", setup_type="squeeze_breakout", composite_score=8.0),
            _make_play("MSFT", setup_type="bull_flag", composite_score=8.0),
        ]
        sector_map = {"AAPL": "Technology", "MSFT": "Technology"}
        assert not orch._is_satisfied(plays, sector_map)


# ---------------------------------------------------------------------------
# Adjustment logic
# ---------------------------------------------------------------------------


class TestAdjustmentLogic:
    def test_relaxes_rr_when_too_few_plays(self):
        """Not enough plays should relax min_risk_reward."""
        config = AgentConfig(
            scoring=ScoringConfig(min_risk_reward=2.0, target_play_count=10),
        )
        orch = Orchestrator(config)
        plays = [_make_play("AAPL")]
        sector_map = {"AAPL": "Technology"}
        scoring_metadata = {"total_candidates": 1, "final_count": 1, "dropped": {},
                            "errors": 0, "score_distribution": {"min": 7.5, "max": 7.5, "avg": 7.5}}

        adjustments = orch._compute_adjustments(plays, scoring_metadata, sector_map)
        assert "relax_min_rr" in adjustments
        assert adjustments["relax_min_rr"] == 1.75

    def test_rr_floor(self):
        """min_risk_reward should not go below floor of 1.5."""
        config = AgentConfig(
            scoring=ScoringConfig(min_risk_reward=1.5, target_play_count=10),
        )
        orch = Orchestrator(config)
        plays = [_make_play("AAPL")]
        sector_map = {"AAPL": "Technology"}
        scoring_metadata = {"total_candidates": 1, "final_count": 1, "dropped": {},
                            "errors": 0, "score_distribution": {"min": 7.5, "max": 7.5, "avg": 7.5}}

        adjustments = orch._compute_adjustments(plays, scoring_metadata, sector_map)
        # At floor — should not appear in adjustments
        assert "relax_min_rr" not in adjustments

    def test_relaxes_confidence_when_too_few(self):
        """Not enough plays should relax min_confidence_score."""
        config = AgentConfig(
            scoring=ScoringConfig(min_confidence_score=7.0, target_play_count=10),
        )
        orch = Orchestrator(config)
        plays = [_make_play("AAPL")]
        sector_map = {"AAPL": "Technology"}
        scoring_metadata = {"total_candidates": 1, "final_count": 1, "dropped": {},
                            "errors": 0, "score_distribution": {"min": 7.5, "max": 7.5, "avg": 7.5}}

        adjustments = orch._compute_adjustments(plays, scoring_metadata, sector_map)
        assert "relax_min_confidence" in adjustments
        assert adjustments["relax_min_confidence"] == 6.5

    def test_confidence_floor(self):
        """min_confidence_score should not go below floor of 4.0."""
        config = AgentConfig(
            scoring=ScoringConfig(min_confidence_score=4.0, target_play_count=10),
        )
        orch = Orchestrator(config)
        plays = [_make_play("AAPL")]
        sector_map = {"AAPL": "Technology"}
        scoring_metadata = {"total_candidates": 1, "final_count": 1, "dropped": {},
                            "errors": 0, "score_distribution": {"min": 7.5, "max": 7.5, "avg": 7.5}}

        adjustments = orch._compute_adjustments(plays, scoring_metadata, sector_map)
        assert "relax_min_confidence" not in adjustments

    def test_sector_concentration_detected(self, orchestrator: Orchestrator):
        """Detects when too many plays are in one sector."""
        plays = [_make_play(f"T{i}", composite_score=8.0) for i in range(5)]
        sector_map = {f"T{i}": "Technology" for i in range(5)}
        scoring_metadata = {"total_candidates": 5, "final_count": 5, "dropped": {},
                            "errors": 0, "score_distribution": {"min": 8.0, "max": 8.0, "avg": 8.0}}

        adjustments = orchestrator._compute_adjustments(
            plays, scoring_metadata, sector_map
        )
        assert "sector_concentration" in adjustments

    def test_setup_concentration_detected(self, orchestrator: Orchestrator):
        """Detects when too many plays are the same setup type."""
        plays = [
            _make_play(f"T{i}", setup_type="squeeze_breakout", composite_score=8.0)
            for i in range(5)
        ]
        sector_map = {
            "T0": "Technology", "T1": "Energy", "T2": "Financials",
            "T3": "Healthcare", "T4": "Consumer",
        }
        scoring_metadata = {"total_candidates": 5, "final_count": 5, "dropped": {},
                            "errors": 0, "score_distribution": {"min": 8.0, "max": 8.0, "avg": 8.0}}

        adjustments = orchestrator._compute_adjustments(
            plays, scoring_metadata, sector_map
        )
        assert "setup_concentration" in adjustments


# ---------------------------------------------------------------------------
# Sentiment cache: no re-analysis
# ---------------------------------------------------------------------------


class TestSentimentCache:
    def test_cached_tickers_not_re_analyzed(self, orchestrator: Orchestrator):
        """Tickers already in sentiment cache should be excluded from new analysis."""
        setups = [_make_setup("AAPL"), _make_setup("MSFT"), _make_setup("GOOG")]
        cache = {"AAPL": _make_sentiment("AAPL")}

        selected = orchestrator._select_for_sentiment(setups, cache)
        selected_tickers = {s.ticker for s in selected}

        assert "AAPL" not in selected_tickers
        assert "MSFT" in selected_tickers
        assert "GOOG" in selected_tickers

    def test_sentiment_deduplicated_per_ticker(self, orchestrator: Orchestrator):
        """Multiple setups for same ticker should result in one sentiment call."""
        setups = [
            _make_setup("AAPL", setup_type="squeeze_breakout", technical_score=9.0),
            _make_setup("AAPL", setup_type="bull_flag", technical_score=7.0),
            _make_setup("MSFT", setup_type="trend_pullback", technical_score=8.0),
        ]
        cache: dict[str, SentimentResult] = {}

        selected = orchestrator._select_for_sentiment(setups, cache)
        tickers = [s.ticker for s in selected]

        assert tickers.count("AAPL") == 1
        assert tickers.count("MSFT") == 1


# ---------------------------------------------------------------------------
# Diversity penalties
# ---------------------------------------------------------------------------


class TestDiversityPenalties:
    def test_sector_penalty_applied(self, orchestrator: Orchestrator):
        """Plays beyond 3 in same sector get penalized."""
        # Use different setup types so only the sector penalty fires
        setup_types = ["squeeze_breakout", "bull_flag", "mean_reversion",
                       "trend_pullback", "squeeze_breakout"]
        plays = [
            _make_play(f"T{i}", setup_type=setup_types[i], composite_score=8.0)
            for i in range(5)
        ]
        sector_map = {f"T{i}": "Technology" for i in range(5)}

        result = orchestrator._apply_diversity_penalties(plays, sector_map)

        # First 3 keep 8.0, 4th gets -0.5, 5th gets -0.5
        assert result[0].composite_score == 8.0
        assert result[1].composite_score == 8.0
        assert result[2].composite_score == 8.0
        assert result[3].composite_score == pytest.approx(7.5)
        assert result[4].composite_score == pytest.approx(7.5)

    def test_setup_type_penalty_applied(self, orchestrator: Orchestrator):
        """Plays beyond 3 of same setup type get penalized."""
        plays = [
            _make_play(f"T{i}", setup_type="squeeze_breakout", composite_score=8.0)
            for i in range(5)
        ]
        sector_map = {
            "T0": "Technology", "T1": "Energy", "T2": "Financials",
            "T3": "Healthcare", "T4": "Consumer",
        }

        result = orchestrator._apply_diversity_penalties(plays, sector_map)

        # First 3 keep 8.0, 4th and 5th each get -0.3
        assert result[3].composite_score == pytest.approx(7.7)
        assert result[4].composite_score == pytest.approx(7.7)

    def test_no_penalty_for_diverse_plays(self, orchestrator: Orchestrator):
        """Diverse plays should not get penalized."""
        plays = _diverse_plays(4)
        sector_map = {
            "AAPL": "Technology", "MSFT": "Technology",
            "XOM": "Energy", "JPM": "Financials",
        }

        original_scores = [p.composite_score for p in plays]
        result = orchestrator._apply_diversity_penalties(plays, sector_map)
        result_scores = [p.composite_score for p in result]

        assert original_scores == result_scores


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_zero_candidates_aborts(self, config: AgentConfig):
        """Screener returning empty should abort gracefully."""
        orch = Orchestrator(config)

        with patch.object(orch.screener, "run", return_value=[]):
            result = orch.run()

        assert result.plays == []
        assert result.iterations == 1

    def test_no_setups_triggers_relaxation(self, config: AgentConfig):
        """No technical setups should trigger parameter relaxation."""
        config.max_iterations = 2
        orch = Orchestrator(config)

        candidates = [_make_candidate("AAPL")]
        plays = [_make_play("AAPL", setup_type="squeeze_breakout", composite_score=8.0),
                 _make_play("XOM", setup_type="bull_flag", composite_score=8.0)]
        setups = [_make_setup("AAPL"), _make_setup("XOM", setup_type="bull_flag")]
        scoring_metadata = {"total_candidates": 2, "final_count": 2, "dropped": {},
                            "errors": 0, "score_distribution": {"min": 8.0, "max": 8.0, "avg": 8.0}}

        # First technical run returns empty, second returns setups
        technical_call = {"n": 0}

        def mock_technical_run(cands):
            technical_call["n"] += 1
            if technical_call["n"] == 1:
                return []
            return setups

        with patch.object(orch.screener, "run", return_value=candidates):
            # Patch TechnicalAgent class to handle rebuild
            with patch("src.orchestrator.TechnicalAgent") as MockTech:
                mock_tech_instance = MagicMock()
                mock_tech_instance.run.side_effect = mock_technical_run
                MockTech.return_value = mock_tech_instance

                with patch.object(orch.sentiment, "run", return_value=[_make_sentiment("AAPL")]):
                    with patch("src.orchestrator.ScoringAgent") as MockScoring:
                        mock_scoring = MagicMock()
                        mock_scoring.run.return_value = (plays, scoring_metadata)
                        MockScoring.return_value = mock_scoring

                        result = orch.run()

        assert result.iterations == 2
        assert "relaxed_technical" in str(result.metadata.get("adjustment_history", []))

    def test_original_config_not_modified(self):
        """Orchestrator should never modify the original config."""
        config = AgentConfig(
            scoring=ScoringConfig(min_risk_reward=2.0, min_confidence_score=6.0),
        )
        original_rr = config.scoring.min_risk_reward
        original_conf = config.scoring.min_confidence_score

        orch = Orchestrator(config)

        # Manually apply adjustments
        orch._apply_adjustments({
            "relax_min_rr": 1.5,
            "relax_min_confidence": 4.0,
        })

        # Original config should be unchanged
        assert config.scoring.min_risk_reward == original_rr
        assert config.scoring.min_confidence_score == original_conf

    def test_result_has_all_fields(self, config: AgentConfig):
        """OrchestratorResult should include all required fields."""
        orch = Orchestrator(config)

        with patch.object(orch.screener, "run", return_value=[]):
            result = orch.run()

        assert hasattr(result, "plays")
        assert hasattr(result, "metadata")
        assert hasattr(result, "iterations")
        assert hasattr(result, "run_id")
        assert hasattr(result, "timestamp")
        assert hasattr(result, "total_api_cost_estimate")
        assert hasattr(result, "execution_time_seconds")
        assert result.execution_time_seconds >= 0

    def test_empty_plays_satisfaction(self, orchestrator: Orchestrator):
        """Empty plays should not satisfy."""
        assert not orchestrator._is_satisfied([], {})
