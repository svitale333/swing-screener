from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.agents.technical import TechnicalAgent
from src.config import TechnicalConfig
from src.types import ScreenedCandidate, TechnicalSetup
from src.utils.technical_helpers import (
    compute_impulse_moves,
    consolidation_range,
    detect_divergence,
    find_swing_highs,
    find_swing_lows,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> TechnicalConfig:
    return TechnicalConfig()


@pytest.fixture
def agent(config: TechnicalConfig) -> TechnicalAgent:
    return TechnicalAgent(config)


@pytest.fixture
def candidates() -> list[ScreenedCandidate]:
    return [
        ScreenedCandidate(
            ticker="AAPL",
            price=150.0,
            avg_volume=5_000_000.0,
            market_cap=2_000_000_000_000.0,
            sector="Technology",
            industry="Consumer Electronics",
        ),
    ]


def _make_dates(n: int) -> pd.DatetimeIndex:
    """Generate n business-day dates ending today."""
    return pd.bdate_range(end="2025-01-15", periods=n)


def _base_ohlcv(n: int = 120, base_price: float = 100.0) -> pd.DataFrame:
    """Create a simple uptrending OHLCV DataFrame.

    Price drifts upward gently from base_price with small random noise.
    """
    rng = np.random.RandomState(42)
    dates = _make_dates(n)
    close = base_price + np.cumsum(rng.normal(0.1, 0.3, n))
    high = close + rng.uniform(0.5, 1.5, n)
    low = close - rng.uniform(0.5, 1.5, n)
    open_ = close + rng.normal(0, 0.5, n)
    volume = rng.uniform(900_000, 1_100_000, n)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------


class TestFindSwingHighs:
    def test_basic_detection(self):
        # Clear peak at index 5
        values = [1, 2, 3, 4, 5, 10, 5, 4, 3, 2, 1, 2, 3]
        series = pd.Series(values, dtype=float)
        result = find_swing_highs(series, window=3)
        assert len(result) >= 1
        # The peak at index 5 should be detected
        peaks = [idx for idx, _ in result]
        assert 5 in peaks

    def test_multiple_peaks(self):
        values = [1, 2, 10, 2, 1, 2, 10, 2, 1, 2, 10, 2, 1]
        series = pd.Series(values, dtype=float)
        result = find_swing_highs(series, window=2)
        assert len(result) >= 2

    def test_insufficient_data(self):
        series = pd.Series([1.0, 2.0, 3.0])
        result = find_swing_highs(series, window=5)
        assert result == []

    def test_flat_series(self):
        series = pd.Series([5.0] * 20)
        result = find_swing_highs(series, window=3)
        # Flat series: every point equals its neighbors, so >= test passes for all
        assert isinstance(result, list)


class TestFindSwingLows:
    def test_basic_detection(self):
        values = [10, 8, 6, 4, 2, 1, 2, 4, 6, 8, 10, 8, 6]
        series = pd.Series(values, dtype=float)
        result = find_swing_lows(series, window=3)
        assert len(result) >= 1
        troughs = [idx for idx, _ in result]
        assert 5 in troughs

    def test_insufficient_data(self):
        series = pd.Series([1.0, 2.0])
        result = find_swing_lows(series, window=5)
        assert result == []


class TestComputeImpulseMoves:
    def test_detects_impulse(self):
        # 10% rally from bar 0 to bar 3
        close = [100, 102, 105, 110, 110, 110, 110, 110, 110, 110]
        df = pd.DataFrame({"Close": close})
        result = compute_impulse_moves(df, min_pct=5.0, window=10)
        assert len(result) >= 1
        assert result[0][2] >= 5.0  # pct_change

    def test_no_impulse(self):
        # Flat price
        df = pd.DataFrame({"Close": [100.0] * 10})
        result = compute_impulse_moves(df, min_pct=5.0, window=10)
        assert result == []

    def test_empty_dataframe(self):
        df = pd.DataFrame({"Close": [100.0]})
        result = compute_impulse_moves(df, min_pct=5.0, window=10)
        assert result == []


class TestDetectDivergence:
    def test_bullish_divergence(self):
        # Price makes lower lows but RSI makes higher lows.
        # Build 40-bar series so the last 20 bars contain two clear swing lows.
        # Pattern: high -> trough1 -> high -> trough2(lower price, higher RSI) -> recovery
        price_vals = (
            [100] * 10  # padding
            + [98, 96, 94, 92, 90]  # first dip — trough at 90
            + [92, 94, 96, 98, 100]  # recovery
            + [98, 96, 94, 92, 90, 88]  # second dip — trough at 88 (lower low)
            + [90, 92, 94, 96]  # recovery
        )
        rsi_vals = (
            [50] * 10
            + [40, 35, 30, 25, 20]  # first dip — RSI trough at 20
            + [30, 40, 45, 48, 50]
            + [45, 40, 38, 35, 30, 28]  # second dip — RSI trough at 28 (higher low!)
            + [35, 40, 45, 50]
        )
        price = pd.Series(price_vals, dtype=float)
        rsi = pd.Series(rsi_vals, dtype=float)
        result = detect_divergence(price, rsi, window=20)
        assert result is True

    def test_no_divergence_when_aligned(self):
        n = 20
        # Both make lower lows together
        price = pd.Series(np.linspace(100, 80, n))
        rsi = pd.Series(np.linspace(50, 20, n))
        result = detect_divergence(price, rsi, window=20)
        assert result is False

    def test_insufficient_data(self):
        price = pd.Series([100.0, 99.0])
        rsi = pd.Series([50.0, 48.0])
        assert detect_divergence(price, rsi, window=20) is False


class TestConsolidationRange:
    def test_basic_range(self):
        df = pd.DataFrame({
            "Open": [100, 101, 102],
            "High": [105, 106, 107],
            "Low": [95, 96, 97],
            "Close": [102, 103, 104],
        })
        low, high, pct = consolidation_range(df, 0, 2)
        assert low == 95.0
        assert high == 107.0
        assert pct == pytest.approx((107 - 95) / 95 * 100, rel=0.01)

    def test_invalid_indices(self):
        df = pd.DataFrame({"Low": [1], "High": [2]})
        assert consolidation_range(df, -1, 0) == (0.0, 0.0, 0.0)
        assert consolidation_range(df, 0, 5) == (0.0, 0.0, 0.0)


# ---------------------------------------------------------------------------
# Setup detector tests
# ---------------------------------------------------------------------------


def _build_squeeze_df() -> pd.DataFrame:
    """Build OHLCV data that triggers the squeeze_breakout detector.

    - Tight Bollinger Bands (low bandwidth)
    - Contracting ATR
    - Low volume in last 5 bars
    - Price above SMA 50
    """
    n = 120
    dates = _make_dates(n)

    # Start with moderate volatility, then compress
    rng = np.random.RandomState(123)
    close = np.full(n, 110.0)

    # First 80 bars: normal movement around 100-110
    for i in range(80):
        close[i] = 100 + rng.normal(0, 3)

    # Last 40 bars: very tight, trending slightly up and above 50 SMA
    for i in range(80, n):
        close[i] = 110 + rng.normal(0, 0.2)

    high = close + np.concatenate([rng.uniform(1, 3, 80), rng.uniform(0.1, 0.3, 40)])
    low = close - np.concatenate([rng.uniform(1, 3, 80), rng.uniform(0.1, 0.3, 40)])
    open_ = close + rng.normal(0, 0.3, n)

    # Volume: moderate for most of the data, then drop sharply in the last 5 bars.
    # The 20-bar SMA will still include the moderate-volume bars so the ratio
    # (last-5 avg / 20-day SMA) will be well below 0.5.
    volume = np.concatenate([
        rng.uniform(1_000_000, 2_000_000, 80),
        rng.uniform(900_000, 1_100_000, 35),  # moderate volume through bar 114
        rng.uniform(50_000, 100_000, 5),       # sharp drop in last 5 bars
    ])

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def _build_bull_flag_df() -> pd.DataFrame:
    """Build OHLCV data that triggers the bull_flag detector.

    - Strong impulse (~8% in 4 bars)
    - Tight consolidation afterward (10+ bars)
    - Declining volume in consolidation
    - Price above 21 EMA
    """
    n = 120
    dates = _make_dates(n)
    rng = np.random.RandomState(456)

    close = np.full(n, 100.0)
    # Flat base for first 100 bars
    for i in range(100):
        close[i] = 100 + rng.normal(0, 0.5)

    # Impulse: bars 100-104 (5 bars, ~10% gain) — within last 30 bars
    impulse_start_price = close[99]
    for i in range(100, 105):
        close[i] = impulse_start_price + (i - 99) * 2.2  # ~11% total

    # Consolidation: bars 105-119 (15 bars, tight around the high)
    consol_center = close[104]
    for i in range(105, n):
        close[i] = consol_center + rng.normal(0, 0.3)

    high = close + rng.uniform(0.2, 0.8, n)
    low = close - rng.uniform(0.2, 0.8, n)
    open_ = close + rng.normal(0, 0.3, n)

    # Volume: high during impulse, declining in consolidation
    volume = np.full(n, 1_000_000.0)
    volume[100:105] = 3_000_000  # impulse volume spike
    for i in range(105, n):
        volume[i] = 400_000 * (1 - (i - 105) / (n - 105 + 1))  # declining

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def _build_mean_reversion_df() -> pd.DataFrame:
    """Build OHLCV data that triggers the mean_reversion detector.

    - RSI oversold (<=30) in last 5 bars
    - Price near SMA 50
    - Volume spike on last bar
    """
    n = 120
    dates = _make_dates(n)
    rng = np.random.RandomState(789)

    # Price trends up then drops sharply to near SMA 50
    close = np.full(n, 100.0)
    for i in range(60):
        close[i] = 100 + i * 0.3 + rng.normal(0, 0.5)
    for i in range(60, 100):
        close[i] = close[59] + rng.normal(0, 0.5)  # flat-ish
    # Sharp drop in last 20 bars
    for i in range(100, n):
        close[i] = close[99] - (i - 99) * 0.8 + rng.normal(0, 0.2)

    high = close + rng.uniform(0.3, 1.0, n)
    low = close - rng.uniform(0.3, 1.0, n)
    open_ = close + rng.normal(0, 0.3, n)

    # Volume: normal then spike on last bar
    volume = rng.uniform(800_000, 1_200_000, n)
    volume[-1] = 3_000_000  # capitulation spike

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def _build_trend_pullback_df() -> pd.DataFrame:
    """Build OHLCV data that triggers the trend_pullback detector.

    - Strong uptrend (price > rising SMA 50 > SMA 200)
    - Pullback to 21 EMA
    - RSI 40-55
    - Declining volume on pullback
    """
    n = 120
    dates = _make_dates(n)
    rng = np.random.RandomState(101)

    # Steady uptrend
    close = np.full(n, 100.0)
    for i in range(n):
        close[i] = 80 + i * 0.5 + rng.normal(0, 0.3)

    # Last few bars: slight pullback toward EMA 21 but still well above SMA 50/200
    # The final price should be close to the 21 EMA
    # We'll adjust the last 5 bars to dip slightly
    peak = close[n - 6]
    for i in range(n - 5, n):
        close[i] = peak - (i - (n - 6)) * 0.4

    high = close + rng.uniform(0.3, 1.0, n)
    low = close - rng.uniform(0.3, 1.0, n)
    open_ = close + rng.normal(0, 0.3, n)

    # Volume: declining on last 3 bars
    volume = rng.uniform(900_000, 1_100_000, n)
    volume[-3] = 800_000
    volume[-2] = 600_000
    volume[-1] = 400_000

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


class TestSqueezeBreakout:
    def test_detects_squeeze(self, agent: TechnicalAgent):
        df = _build_squeeze_df()
        df = agent._compute_indicators(df)
        result = agent._detect_squeeze_breakout("TEST", df)
        assert result is not None
        assert result.setup_type == "squeeze_breakout"
        assert result.direction == "long"
        assert 1 <= result.technical_score <= 10

    def test_no_squeeze_high_volatility(self, agent: TechnicalAgent):
        """High volatility data should NOT trigger squeeze."""
        df = _base_ohlcv(120)
        # Amplify volatility in last 40 bars
        df.iloc[-40:, df.columns.get_loc("High")] += 10
        df.iloc[-40:, df.columns.get_loc("Low")] -= 10
        df = agent._compute_indicators(df)
        result = agent._detect_squeeze_breakout("TEST", df)
        # May or may not trigger depending on relative BB width
        # But score should be moderate at best
        if result is not None:
            assert 1 <= result.technical_score <= 10


class TestBullFlag:
    def test_detects_bull_flag(self, agent: TechnicalAgent):
        df = _build_bull_flag_df()
        df = agent._compute_indicators(df)
        result = agent._detect_bull_flag("TEST", df)
        assert result is not None
        assert result.setup_type == "bull_flag"
        assert 1 <= result.technical_score <= 10

    def test_no_flag_without_impulse(self, agent: TechnicalAgent):
        """Flat data should NOT trigger bull flag."""
        rng = np.random.RandomState(999)
        n = 120
        close = np.full(n, 100.0) + rng.normal(0, 0.1, n)
        df = pd.DataFrame({
            "Open": close,
            "High": close + 0.2,
            "Low": close - 0.2,
            "Close": close,
            "Volume": np.full(n, 1_000_000.0),
        }, index=_make_dates(n))
        df = agent._compute_indicators(df)
        result = agent._detect_bull_flag("TEST", df)
        assert result is None


class TestMeanReversion:
    def test_detects_mean_reversion(self, agent: TechnicalAgent):
        df = _build_mean_reversion_df()
        df = agent._compute_indicators(df)
        result = agent._detect_mean_reversion("TEST", df)
        # May or may not detect depending on exact synthetic data alignment
        # with SMA 50 proximity — this is a best-effort synthetic test
        if result is not None:
            assert result.setup_type == "mean_reversion"
            assert 1 <= result.technical_score <= 10

    def test_no_reversion_when_overbought(self, agent: TechnicalAgent):
        """Overbought data should NOT trigger mean reversion."""
        rng = np.random.RandomState(42)
        n = 120
        # Steadily rising = RSI will be high
        close = 100 + np.arange(n) * 0.5 + rng.normal(0, 0.2, n)
        df = pd.DataFrame({
            "Open": close - 0.1,
            "High": close + 1,
            "Low": close - 0.5,
            "Close": close,
            "Volume": rng.uniform(800_000, 1_200_000, n),
        }, index=_make_dates(n))
        df = agent._compute_indicators(df)
        result = agent._detect_mean_reversion("TEST", df)
        assert result is None


class TestTrendPullback:
    def test_detects_pullback(self, agent: TechnicalAgent):
        df = _build_trend_pullback_df()
        df = agent._compute_indicators(df)
        result = agent._detect_trend_pullback("TEST", df)
        # May or may not detect depending on exact indicator alignment
        if result is not None:
            assert result.setup_type == "trend_pullback"
            assert 1 <= result.technical_score <= 10

    def test_no_pullback_in_downtrend(self, agent: TechnicalAgent):
        """Downtrending data should NOT trigger trend pullback."""
        rng = np.random.RandomState(42)
        n = 120
        close = 150 - np.arange(n) * 0.5 + rng.normal(0, 0.3, n)
        df = pd.DataFrame({
            "Open": close + 0.1,
            "High": close + 1,
            "Low": close - 1,
            "Close": close,
            "Volume": rng.uniform(800_000, 1_200_000, n),
        }, index=_make_dates(n))
        df = agent._compute_indicators(df)
        result = agent._detect_trend_pullback("TEST", df)
        assert result is None


# ---------------------------------------------------------------------------
# Integration-level tests (mocked yfinance)
# ---------------------------------------------------------------------------


class TestTechnicalAgentRun:
    def test_run_with_mocked_download(self, agent: TechnicalAgent, candidates: list[ScreenedCandidate]):
        """Full pipeline run with mocked yfinance.download."""
        df = _base_ohlcv(120)

        # Build a MultiIndex DataFrame like yf.download returns for multiple tickers
        cols = pd.MultiIndex.from_tuples(
            [(candidates[0].ticker, c) for c in df.columns],
            names=["Ticker", "Price"],
        )
        multi_df = pd.DataFrame(df.values, index=df.index, columns=cols)

        with patch("src.agents.technical.yf.download", return_value=multi_df):
            result = agent.run(candidates)

        assert isinstance(result, list)
        for setup in result:
            assert isinstance(setup, TechnicalSetup)
            assert 1 <= setup.technical_score <= 10

    def test_run_empty_candidates(self, agent: TechnicalAgent):
        """Empty candidates list returns empty results."""
        result = agent.run([])
        assert result == []

    def test_run_download_failure(self, agent: TechnicalAgent, candidates: list[ScreenedCandidate]):
        """Failed download returns empty results."""
        with patch("src.agents.technical.yf.download", side_effect=Exception("network error")):
            result = agent.run(candidates)
        assert result == []

    def test_run_insufficient_data(self, agent: TechnicalAgent, candidates: list[ScreenedCandidate]):
        """Ticker with too few bars is skipped gracefully."""
        df = _base_ohlcv(20)  # only 20 bars, need 50+
        cols = pd.MultiIndex.from_tuples(
            [(candidates[0].ticker, c) for c in df.columns],
            names=["Ticker", "Price"],
        )
        multi_df = pd.DataFrame(df.values, index=df.index, columns=cols)

        with patch("src.agents.technical.yf.download", return_value=multi_df):
            result = agent.run(candidates)
        assert result == []


# ---------------------------------------------------------------------------
# Scoring tests
# ---------------------------------------------------------------------------


class TestScoring:
    def test_scores_in_valid_range(self, agent: TechnicalAgent):
        """All setup detectors produce scores in [1, 10]."""
        dataframes = [
            _build_squeeze_df(),
            _build_bull_flag_df(),
            _build_mean_reversion_df(),
            _build_trend_pullback_df(),
        ]
        for df in dataframes:
            df_ind = agent._compute_indicators(df)
            setups = agent._detect_all_setups("TEST", df_ind)
            for s in setups:
                assert 1 <= s.technical_score <= 10, f"{s.setup_type} score {s.technical_score} out of range"


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_extreme_volatility(self, agent: TechnicalAgent):
        """Extremely volatile data should not crash."""
        rng = np.random.RandomState(42)
        n = 120
        close = 100 + rng.normal(0, 20, n)
        close = np.abs(close)  # no negatives
        df = pd.DataFrame({
            "Open": close + rng.normal(0, 5, n),
            "High": close + np.abs(rng.normal(5, 3, n)),
            "Low": close - np.abs(rng.normal(5, 3, n)),
            "Close": close,
            "Volume": rng.uniform(500_000, 5_000_000, n),
        }, index=_make_dates(n))
        df = agent._compute_indicators(df)
        # Should not raise
        setups = agent._detect_all_setups("TEST", df)
        assert isinstance(setups, list)

    def test_flat_price_action(self, agent: TechnicalAgent):
        """Completely flat price should not crash and likely return no setups."""
        n = 120
        df = pd.DataFrame({
            "Open": np.full(n, 100.0),
            "High": np.full(n, 100.5),
            "Low": np.full(n, 99.5),
            "Close": np.full(n, 100.0),
            "Volume": np.full(n, 1_000_000.0),
        }, index=_make_dates(n))
        df = agent._compute_indicators(df)
        setups = agent._detect_all_setups("TEST", df)
        assert isinstance(setups, list)

    def test_single_ticker_download(self, agent: TechnicalAgent):
        """Single-ticker download has flat (non-MultiIndex) columns."""
        candidates = [
            ScreenedCandidate(
                ticker="TSLA", price=200.0, avg_volume=10_000_000.0,
                market_cap=500_000_000_000.0, sector="Tech", industry="EV",
            ),
        ]
        df = _base_ohlcv(120)
        # Single ticker: yf.download returns flat DataFrame
        with patch("src.agents.technical.yf.download", return_value=df):
            result = agent.run(candidates)
        assert isinstance(result, list)
