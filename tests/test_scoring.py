from __future__ import annotations

import pytest

from src.agents.scoring import ScoringAgent
from src.config import ScoringConfig, TradeParamsConfig
from src.types import TechnicalSetup, SentimentResult, TradePlay
from src.utils.trade_math import (
    calculate_rr_ratio,
    calculate_position_size,
    atr_buffer,
    normalize_score,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> ScoringConfig:
    """Relaxed config for tests that don't care about filtering."""
    return ScoringConfig(min_risk_reward=0.0, min_confidence_score=0.0)


@pytest.fixture
def agent(config: ScoringConfig) -> ScoringAgent:
    return ScoringAgent(config)


def _make_setup(
    ticker: str = "AAPL",
    setup_type: str = "squeeze_breakout",
    technical_score: float = 8.0,
    support: float = 170.0,
    resistance: float = 185.0,
    atr: float = 3.0,
    close: float = 182.0,
    ema_21: float = 180.0,
    sma_50: float = 175.0,
    key_levels: dict | None = None,
    notes: str = "",
) -> TechnicalSetup:
    indicators = {"atr_14": atr, "close": close, "ema_21": ema_21, "sma_50": sma_50}
    return TechnicalSetup(
        ticker=ticker,
        setup_type=setup_type,
        direction="long",
        technical_score=technical_score,
        support_level=support,
        resistance_level=resistance,
        key_levels=key_levels or {},
        indicators=indicators,
        notes=notes,
    )


def _make_sentiment(
    ticker: str = "AAPL",
    sentiment: str = "bullish",
    confidence: float = 8.0,
    risk_flags: list[str] | None = None,
    days_to_earnings: int | None = None,
) -> SentimentResult:
    return SentimentResult(
        ticker=ticker,
        sentiment=sentiment,
        confidence=confidence,
        catalysts=["catalyst_a"],
        risk_flags=risk_flags or [],
        days_to_earnings=days_to_earnings,
        summary=f"{sentiment} summary",
    )


# ---------------------------------------------------------------------------
# trade_math utility tests
# ---------------------------------------------------------------------------


class TestTradeMath:
    def test_rr_ratio_basic(self):
        # Entry 100, stop 95, target 110 → risk 5, reward 10 → R:R 2.0
        assert calculate_rr_ratio(100.0, 95.0, 110.0) == 2.0

    def test_rr_ratio_zero_risk(self):
        assert calculate_rr_ratio(100.0, 100.0, 110.0) == 0.0

    def test_rr_ratio_negative_reward(self):
        assert calculate_rr_ratio(100.0, 95.0, 90.0) == 0.0

    def test_position_size(self):
        # $100k account, 1% risk, entry $50, stop $48 → $1000 / $2 = 500 shares
        assert calculate_position_size(100_000, 1.0, 50.0, 48.0) == 500

    def test_position_size_zero_risk(self):
        assert calculate_position_size(100_000, 1.0, 50.0, 50.0) == 0

    def test_atr_buffer(self):
        assert atr_buffer(3.0, 0.5) == 1.5

    def test_normalize_score(self):
        assert normalize_score(5.0, 0.0, 10.0, 1.0, 10.0) == 5.5
        assert normalize_score(0.0, 0.0, 10.0, 1.0, 10.0) == 1.0
        assert normalize_score(10.0, 0.0, 10.0, 1.0, 10.0) == 10.0

    def test_normalize_score_clamp(self):
        assert normalize_score(15.0, 0.0, 10.0, 1.0, 10.0) == 10.0
        assert normalize_score(-5.0, 0.0, 10.0, 1.0, 10.0) == 1.0

    def test_normalize_score_same_range(self):
        # min == max → midpoint
        assert normalize_score(5.0, 5.0, 5.0, 1.0, 10.0) == 5.5


# ---------------------------------------------------------------------------
# Entry / Stop / Target calculation tests
# ---------------------------------------------------------------------------


class TestEntryCalculation:
    """Verify entry prices for each setup type with concrete numbers."""

    def test_squeeze_breakout_entry(self, agent: ScoringAgent):
        setup = _make_setup(setup_type="squeeze_breakout", resistance=185.0, atr=3.0)
        # Entry = resistance + ATR * 0.1 = 185 + 0.3 = 185.3
        entry = agent._calc_entry(setup, 3.0)
        assert entry == pytest.approx(185.3)

    def test_bull_flag_entry(self, agent: ScoringAgent):
        setup = _make_setup(setup_type="bull_flag", resistance=185.0, atr=3.0)
        # Entry = resistance + ATR * 0.1 = 185 + 0.3 = 185.3
        entry = agent._calc_entry(setup, 3.0)
        assert entry == pytest.approx(185.3)

    def test_mean_reversion_entry(self, agent: ScoringAgent):
        setup = _make_setup(setup_type="mean_reversion", support=170.0, atr=3.0)
        # Entry = support + ATR * 0.2 = 170 + 0.6 = 170.6
        entry = agent._calc_entry(setup, 3.0)
        assert entry == pytest.approx(170.6)

    def test_trend_pullback_entry_near_ema(self, agent: ScoringAgent):
        # Close is within 1.5% of EMA → use current price
        setup = _make_setup(
            setup_type="trend_pullback", close=180.5, ema_21=180.0, atr=3.0
        )
        entry = agent._calc_entry(setup, 3.0)
        assert entry == pytest.approx(180.5)

    def test_trend_pullback_entry_far_from_ema(self, agent: ScoringAgent):
        # Close is far from EMA → use EMA as limit
        setup = _make_setup(
            setup_type="trend_pullback", close=190.0, ema_21=180.0, atr=3.0
        )
        entry = agent._calc_entry(setup, 3.0)
        assert entry == pytest.approx(180.0)


class TestStopCalculation:
    def test_squeeze_breakout_stop(self, agent: ScoringAgent):
        setup = _make_setup(setup_type="squeeze_breakout", support=170.0, atr=3.0)
        # Stop = support - ATR * 0.5 = 170 - 1.5 = 168.5
        stop = agent._calc_stop(setup, 3.0)
        assert stop == pytest.approx(168.5)

    def test_bull_flag_stop(self, agent: ScoringAgent):
        setup = _make_setup(setup_type="bull_flag", support=170.0, atr=3.0)
        stop = agent._calc_stop(setup, 3.0)
        assert stop == pytest.approx(168.5)

    def test_mean_reversion_stop(self, agent: ScoringAgent):
        setup = _make_setup(
            setup_type="mean_reversion",
            support=170.0,
            atr=3.0,
            key_levels={"swing_low": 168.0},
        )
        # Stop = swing_low - ATR * 0.75 = 168 - 2.25 = 165.75
        stop = agent._calc_stop(setup, 3.0)
        assert stop == pytest.approx(165.75)

    def test_mean_reversion_stop_no_swing_low(self, agent: ScoringAgent):
        # Falls back to support_level
        setup = _make_setup(setup_type="mean_reversion", support=170.0, atr=3.0)
        stop = agent._calc_stop(setup, 3.0)
        assert stop == pytest.approx(170.0 - 3.0 * 0.75)

    def test_trend_pullback_stop(self, agent: ScoringAgent):
        setup = _make_setup(
            setup_type="trend_pullback", support=170.0, sma_50=175.0, atr=3.0
        )
        # Stop = sma_50 - ATR * 0.3 = 175 - 0.9 = 174.1
        stop = agent._calc_stop(setup, 3.0)
        assert stop == pytest.approx(174.1)


class TestTakeProfitCalculation:
    def test_tp1_from_risk(self, agent: ScoringAgent):
        setup = _make_setup()
        # TP1 = entry + risk * 1.5 (no next_resistance)
        tp1 = agent._calc_tp1(setup, entry=185.3, risk=16.8)
        assert tp1 == pytest.approx(185.3 + 16.8 * 1.5)

    def test_tp1_capped_by_next_resistance(self, agent: ScoringAgent):
        setup = _make_setup(key_levels={"next_resistance": 190.0})
        # TP from risk would be higher, but next_resistance is closer
        tp1 = agent._calc_tp1(setup, entry=185.3, risk=16.8)
        assert tp1 == pytest.approx(190.0)

    def test_tp2_breakout_measured_move(self, agent: ScoringAgent):
        setup = _make_setup(
            setup_type="squeeze_breakout", support=170.0, resistance=185.0
        )
        # Measured move: entry + (resistance - support) = 185.3 + 15 = 200.3
        tp2 = agent._calc_tp2(setup, entry=185.3, risk=16.8)
        assert tp2 == pytest.approx(185.3 + 15.0)

    def test_tp2_mean_reversion_swing_high(self, agent: ScoringAgent):
        setup = _make_setup(
            setup_type="mean_reversion", key_levels={"swing_high": 195.0}
        )
        tp2 = agent._calc_tp2(setup, entry=170.6, risk=4.85)
        assert tp2 == pytest.approx(195.0)

    def test_tp2_fallback_to_risk_multiple(self, agent: ScoringAgent):
        # No measured move available → 2.5R
        setup = _make_setup(
            setup_type="mean_reversion", key_levels={}
        )
        tp2 = agent._calc_tp2(setup, entry=170.6, risk=5.0)
        assert tp2 == pytest.approx(170.6 + 5.0 * 2.5)


# ---------------------------------------------------------------------------
# R:R calculation correctness
# ---------------------------------------------------------------------------


class TestRRCalculation:
    def test_rr_known_values(self, agent: ScoringAgent):
        setup = _make_setup(
            setup_type="squeeze_breakout",
            support=170.0,
            resistance=185.0,
            atr=3.0,
            technical_score=8.0,
        )
        sentiment = _make_sentiment(ticker="AAPL", sentiment="bullish", confidence=8.0)
        plays, _ = agent.run([setup], [sentiment])
        assert len(plays) == 1
        play = plays[0]
        # Primary R:R uses TP2 (the actual trade target); TP1 is the partial exit
        expected_rr_tp2 = (play.take_profit_2 - play.entry_price) / (
            play.entry_price - play.stop_loss
        )
        assert play.risk_reward_ratio == pytest.approx(expected_rr_tp2, abs=0.01)
        # TP1 R:R stored in risk_reward_ratio_tp2 for reference
        expected_rr_tp1 = (play.take_profit_1 - play.entry_price) / (
            play.entry_price - play.stop_loss
        )
        assert play.risk_reward_ratio_tp2 == pytest.approx(expected_rr_tp1, abs=0.01)


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------


class TestCompositeScoring:
    def test_composite_with_known_inputs(self, agent: ScoringAgent):
        """Verify composite score formula with known inputs."""
        # technical_score = 8.0, bullish confidence 8.0 → sentiment_score = 8.0
        setup = _make_setup(technical_score=8.0, atr=3.0)
        sentiment = _make_sentiment(sentiment="bullish", confidence=8.0)
        plays, _ = agent.run([setup], [sentiment])
        play = plays[0]

        # Re-compute expected composite
        rr_score = agent._rr_to_score(play.risk_reward_ratio)
        max_possible = 10.0 * 0.5 + 10.0 * 0.3 + 10.0 * 0.2  # = 10.0
        expected = (8.0 * 0.5 + 8.0 * 0.3 + rr_score * 0.2) * 10.0 / max_possible
        assert play.composite_score == pytest.approx(expected, abs=0.01)

    def test_sentiment_score_bullish(self):
        s = _make_sentiment(sentiment="bullish", confidence=9.0)
        assert ScoringAgent._sentiment_to_score(s) == 9.0

    def test_sentiment_score_neutral(self):
        s = _make_sentiment(sentiment="neutral", confidence=7.0)
        assert ScoringAgent._sentiment_to_score(s) == 5.0

    def test_sentiment_score_bearish(self):
        s = _make_sentiment(sentiment="bearish", confidence=8.0)
        assert ScoringAgent._sentiment_to_score(s) == 2.0  # 10 - 8

    def test_rr_to_score_mapping(self):
        assert ScoringAgent._rr_to_score(4.0) == 10.0
        assert ScoringAgent._rr_to_score(5.0) == 10.0  # clamped
        assert ScoringAgent._rr_to_score(2.0) == 5.0
        assert ScoringAgent._rr_to_score(3.0) == pytest.approx(7.5)
        assert ScoringAgent._rr_to_score(0.0) == 0.0
        assert ScoringAgent._rr_to_score(1.0) == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# Hard filters
# ---------------------------------------------------------------------------


class TestFilters:
    def test_filter_low_rr(self):
        """Plays with R:R < 2.0 should be dropped."""
        config = ScoringConfig(min_risk_reward=2.0, min_confidence_score=0.0)
        agent = ScoringAgent(config)
        # squeeze_breakout with tp1_risk_multiple=1.5 → R:R = 1.5 < 2.0
        setup = _make_setup(
            support=170.0,
            resistance=185.0,
            atr=3.0,
            technical_score=9.0,
        )
        sentiment = _make_sentiment(sentiment="bullish", confidence=9.0)
        plays, meta = agent.run([setup], [sentiment])
        assert len(plays) == 0
        assert "low_rr" in meta["dropped"]

    def test_filter_low_composite_score(self):
        """Plays below min_confidence_score should be dropped."""
        config = ScoringConfig(min_confidence_score=9.5, min_risk_reward=0.0)
        agent = ScoringAgent(config)
        setup = _make_setup(technical_score=5.0, atr=3.0)
        sentiment = _make_sentiment(sentiment="neutral", confidence=5.0)
        plays, meta = agent.run([setup], [sentiment])
        assert len(plays) == 0
        assert "low_score" in meta["dropped"]

    def test_filter_wide_stop(self):
        """Plays with position risk > 10% should be dropped."""
        config = ScoringConfig(min_risk_reward=0.0, min_confidence_score=0.0)
        agent = ScoringAgent(config)
        # Huge ATR relative to price → wide stop
        setup = _make_setup(
            support=50.0, resistance=55.0, atr=20.0, technical_score=9.0
        )
        sentiment = _make_sentiment(sentiment="bullish", confidence=9.0)
        plays, meta = agent.run([setup], [sentiment])
        assert len(plays) == 0
        assert "wide_stop" in meta["dropped"]

    def test_filter_bearish_headwind(self):
        """Strong bearish sentiment (confidence >= 7) should be dropped."""
        config = ScoringConfig(min_confidence_score=0.0, min_risk_reward=0.0)
        agent = ScoringAgent(config)
        setup = _make_setup(atr=3.0, technical_score=9.0)
        sentiment = _make_sentiment(
            sentiment="bearish", confidence=8.0
        )
        plays, meta = agent.run([setup], [sentiment])
        assert len(plays) == 0
        assert "bearish_headwind" in meta["dropped"]

    def test_filter_earnings_proximity(self):
        """Tickers with earnings_proximity risk flag and < min days should drop."""
        config = ScoringConfig(
            min_confidence_score=0.0, min_risk_reward=0.0, min_earnings_gap_days=7
        )
        agent = ScoringAgent(config)
        setup = _make_setup(atr=3.0, technical_score=9.0)
        sentiment = _make_sentiment(
            sentiment="bullish",
            confidence=8.0,
            risk_flags=["earnings_proximity"],
            days_to_earnings=3,
        )
        plays, meta = agent.run([setup], [sentiment])
        assert len(plays) == 0
        assert "earnings_proximity" in meta["dropped"]

    def test_no_filter_bearish_low_confidence(self):
        """Bearish with confidence < 7 should NOT be dropped by headwind filter."""
        config = ScoringConfig(min_confidence_score=0.0, min_risk_reward=0.0)
        agent = ScoringAgent(config)
        setup = _make_setup(atr=3.0, technical_score=9.0)
        sentiment = _make_sentiment(sentiment="bearish", confidence=4.0)
        plays, meta = agent.run([setup], [sentiment])
        assert "bearish_headwind" not in meta.get("dropped", {})


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_same_ticker_keeps_best(self, agent: ScoringAgent):
        """Two setups for same ticker → keep highest composite_score."""
        setup_high = _make_setup(
            ticker="AAPL",
            setup_type="squeeze_breakout",
            technical_score=9.0,
            atr=3.0,
        )
        setup_low = _make_setup(
            ticker="AAPL",
            setup_type="bull_flag",
            technical_score=5.0,
            atr=3.0,
        )
        sentiment = _make_sentiment(ticker="AAPL", sentiment="bullish", confidence=8.0)
        config = ScoringConfig(min_confidence_score=0.0, min_risk_reward=0.0)
        agent = ScoringAgent(config)
        plays, _ = agent.run([setup_high, setup_low], [sentiment])
        # Only one play for AAPL
        aapl_plays = [p for p in plays if p.ticker == "AAPL"]
        assert len(aapl_plays) == 1
        # Should be the higher-scoring one
        assert aapl_plays[0].technical_score == 9.0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_all_filtered_returns_empty_with_metadata(self):
        config = ScoringConfig(min_confidence_score=99.0)
        agent = ScoringAgent(config)
        setup = _make_setup(atr=3.0, technical_score=5.0)
        sentiment = _make_sentiment(sentiment="neutral", confidence=5.0)
        plays, meta = agent.run([setup], [sentiment])
        assert plays == []
        assert meta["final_count"] == 0
        assert meta["total_candidates"] == 1
        assert sum(meta["dropped"].values()) > 0

    def test_no_sentiment_uses_default(self, agent: ScoringAgent):
        """Setup with no matching sentiment → default neutral confidence 5."""
        setup = _make_setup(ticker="NFLX", atr=3.0, technical_score=8.0)
        config = ScoringConfig(min_confidence_score=0.0, min_risk_reward=0.0)
        agent = ScoringAgent(config)
        plays, _ = agent.run([setup], [])  # empty sentiments
        assert len(plays) == 1
        assert plays[0].sentiment_summary == "no_sentiment_data"
        assert plays[0].sentiment_score == 5.0

    def test_zero_atr_skipped(self, agent: ScoringAgent):
        setup = _make_setup(atr=0.0)
        sentiment = _make_sentiment()
        plays, _ = agent.run([setup], [sentiment])
        assert len(plays) == 0

    def test_empty_inputs(self, agent: ScoringAgent):
        plays, meta = agent.run([], [])
        assert plays == []
        assert meta["total_candidates"] == 0

    def test_max_play_count_cap(self):
        config = ScoringConfig(
            max_play_count=2, min_confidence_score=0.0, min_risk_reward=0.0
        )
        agent = ScoringAgent(config)
        setups = [
            _make_setup(ticker=f"T{i}", technical_score=7.0 + i * 0.1, atr=3.0)
            for i in range(5)
        ]
        sentiments = [
            _make_sentiment(ticker=f"T{i}", sentiment="bullish", confidence=7.0)
            for i in range(5)
        ]
        plays, _ = agent.run(setups, sentiments)
        assert len(plays) <= 2


# ---------------------------------------------------------------------------
# Position risk %
# ---------------------------------------------------------------------------


class TestPositionRisk:
    def test_position_risk_calculation(self, agent: ScoringAgent):
        setup = _make_setup(
            setup_type="squeeze_breakout",
            support=170.0,
            resistance=185.0,
            atr=3.0,
        )
        sentiment = _make_sentiment()
        config = ScoringConfig(min_confidence_score=0.0, min_risk_reward=0.0)
        agent = ScoringAgent(config)
        plays, _ = agent.run([setup], [sentiment])
        assert len(plays) == 1
        play = plays[0]
        # Position risk = (entry - stop) / entry * 100
        expected = (play.entry_price - play.stop_loss) / play.entry_price * 100
        assert play.position_risk_pct == pytest.approx(expected, abs=0.01)


# ---------------------------------------------------------------------------
# to_report_dict
# ---------------------------------------------------------------------------


class TestToReportDict:
    def test_report_dict_format(self):
        play = TradePlay(
            ticker="AAPL",
            setup_type="squeeze_breakout",
            direction="long",
            entry_price=185.30,
            stop_loss=168.50,
            take_profit_1=210.50,
            take_profit_2=200.30,
            risk_reward_ratio=1.50,
            risk_reward_ratio_tp2=1.89,
            position_risk_pct=9.07,
            technical_score=8.0,
            sentiment_score=8.0,
            composite_score=7.5,
            sentiment_summary="bullish summary",
            catalysts=["earnings_beat", "new_product"],
            risk_flags=["high_short_interest"],
            notes="wide_stop",
        )
        report = play.to_report_dict()
        assert report["ticker"] == "AAPL"
        assert report["entry"] == "$185.30"
        assert report["stop"] == "$168.50"
        assert report["risk_pct"] == "9.1%"
        assert report["catalysts"] == "earnings_beat, new_product"
        assert report["risk_flags"] == "high_short_interest"

    def test_report_dict_none_tp2(self):
        play = TradePlay(
            ticker="AAPL",
            setup_type="squeeze_breakout",
            direction="long",
            entry_price=185.30,
            stop_loss=168.50,
            take_profit_1=210.50,
            take_profit_2=None,
            risk_reward_ratio=1.50,
            risk_reward_ratio_tp2=None,
            position_risk_pct=9.07,
            technical_score=8.0,
            sentiment_score=8.0,
            composite_score=7.5,
        )
        report = play.to_report_dict()
        assert report["tp2"] == "N/A"
        assert report["r_r_tp2"] == "N/A"
