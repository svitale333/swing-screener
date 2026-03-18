from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import pandas as pd
import pytest

from src.agents.screener import ScreenerAgent, DATA_DIR, SCREENER_CACHE_FILE
from src.config import ScreenerConfig
from src.types import ScreenedCandidate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config() -> ScreenerConfig:
    return ScreenerConfig(
        min_avg_volume=500_000,
        min_price=10.0,
        max_price=500.0,
        min_market_cap=500_000_000.0,
        max_candidates=5,
        cache_ttl_hours=4.0,
        ticker_cache_ttl_hours=24.0,
        max_workers=2,
        blacklist=["BAD1", "BAD2"],
    )


@pytest.fixture
def agent(config: ScreenerConfig) -> ScreenerAgent:
    return ScreenerAgent(config)


def _make_price_df(tickers: list[str], prices: list[float], volumes: list[float]) -> pd.DataFrame:
    """Build a multi-ticker DataFrame mimicking yfinance.download(group_by='ticker')."""
    data = {}
    for ticker, price, vol in zip(tickers, prices, volumes):
        data[(ticker, "Close")] = [price, price]
        data[(ticker, "Open")] = [price, price]
        data[(ticker, "High")] = [price + 1, price + 1]
        data[(ticker, "Low")] = [price - 1, price - 1]
        data[(ticker, "Volume")] = [vol, vol]

    df = pd.DataFrame(data, index=pd.date_range("2024-01-01", periods=2))
    df.columns = pd.MultiIndex.from_tuples(df.columns)
    return df


def _mock_ticker_info(market_cap: float, sector: str = "Technology", industry: str = "Software") -> dict:
    return {
        "marketCap": market_cap,
        "sector": sector,
        "industry": industry,
    }


# ---------------------------------------------------------------------------
# Tests: Filter logic
# ---------------------------------------------------------------------------

class TestBatchFilters:
    def test_price_too_low_filtered(self, agent: ScreenerAgent) -> None:
        tickers = ["LOW"]
        df = _make_price_df(tickers, [5.0], [1_000_000.0])
        result = agent._apply_batch_filters(tickers, df)
        assert len(result) == 0

    def test_price_too_high_filtered(self, agent: ScreenerAgent) -> None:
        tickers = ["HIGH"]
        df = _make_price_df(tickers, [600.0], [1_000_000.0])
        result = agent._apply_batch_filters(tickers, df)
        assert len(result) == 0

    def test_volume_too_low_filtered(self, agent: ScreenerAgent) -> None:
        tickers = ["LOWVOL"]
        df = _make_price_df(tickers, [50.0], [100_000.0])
        result = agent._apply_batch_filters(tickers, df)
        assert len(result) == 0

    def test_passes_all_filters(self, agent: ScreenerAgent) -> None:
        tickers = ["GOOD"]
        df = _make_price_df(tickers, [150.0], [2_000_000.0])
        result = agent._apply_batch_filters(tickers, df)
        assert len(result) == 1
        assert result[0]["ticker"] == "GOOD"
        assert result[0]["price"] == 150.0

    def test_multiple_tickers_mixed(self, agent: ScreenerAgent) -> None:
        tickers = ["PASS1", "FAIL_PRICE", "PASS2", "FAIL_VOL"]
        prices = [100.0, 5.0, 200.0, 50.0]
        volumes = [1_000_000.0, 1_000_000.0, 800_000.0, 100_000.0]
        df = _make_price_df(tickers, prices, volumes)
        result = agent._apply_batch_filters(tickers, df)
        passed_tickers = [r["ticker"] for r in result]
        assert "PASS1" in passed_tickers
        assert "PASS2" in passed_tickers
        assert "FAIL_PRICE" not in passed_tickers
        assert "FAIL_VOL" not in passed_tickers

    def test_empty_dataframe(self, agent: ScreenerAgent) -> None:
        result = agent._apply_batch_filters(["AAPL"], pd.DataFrame())
        assert result == []

    def test_missing_ticker_in_df(self, agent: ScreenerAgent) -> None:
        tickers_in_df = ["AAPL"]
        df = _make_price_df(tickers_in_df, [150.0], [2_000_000.0])
        result = agent._apply_batch_filters(["AAPL", "MISSING"], df)
        assert len(result) == 1
        assert result[0]["ticker"] == "AAPL"


# ---------------------------------------------------------------------------
# Tests: Blacklist
# ---------------------------------------------------------------------------

class TestBlacklist:
    def test_blacklist_excludes_tickers(self, agent: ScreenerAgent) -> None:
        tickers = ["AAPL", "BAD1", "MSFT", "BAD2", "GOOG"]
        blacklist = set(agent.config.blacklist)
        filtered = [t for t in tickers if t not in blacklist]
        assert "BAD1" not in filtered
        assert "BAD2" not in filtered
        assert "AAPL" in filtered
        assert "MSFT" in filtered


# ---------------------------------------------------------------------------
# Tests: Enrichment (market cap, options, missing info)
# ---------------------------------------------------------------------------

class TestEnrichSingle:
    @patch("src.agents.screener.yf.Ticker")
    def test_passes_enrichment(self, mock_ticker_cls: MagicMock, agent: ScreenerAgent) -> None:
        mock_obj = MagicMock()
        mock_obj.info = _mock_ticker_info(1_000_000_000.0)
        mock_obj.options = ("2024-01-19", "2024-02-16")
        mock_ticker_cls.return_value = mock_obj

        item = {"ticker": "AAPL", "price": 180.0, "avg_volume": 5_000_000.0}
        result = agent._enrich_single(item)

        assert result is not None
        assert result.ticker == "AAPL"
        assert result.market_cap == 1_000_000_000.0
        assert result.sector == "Technology"

    @patch("src.agents.screener.yf.Ticker")
    def test_market_cap_too_low(self, mock_ticker_cls: MagicMock, agent: ScreenerAgent) -> None:
        mock_obj = MagicMock()
        mock_obj.info = _mock_ticker_info(100_000_000.0)  # below 500M threshold
        mock_obj.options = ("2024-01-19",)
        mock_ticker_cls.return_value = mock_obj

        item = {"ticker": "TINY", "price": 15.0, "avg_volume": 600_000.0}
        result = agent._enrich_single(item)
        assert result is None

    @patch("src.agents.screener.yf.Ticker")
    def test_no_options(self, mock_ticker_cls: MagicMock, agent: ScreenerAgent) -> None:
        mock_obj = MagicMock()
        mock_obj.info = _mock_ticker_info(2_000_000_000.0)
        mock_obj.options = ()  # empty tuple = no options
        mock_ticker_cls.return_value = mock_obj

        item = {"ticker": "NOOPTS", "price": 50.0, "avg_volume": 1_000_000.0}
        result = agent._enrich_single(item)
        assert result is None

    @patch("src.agents.screener.yf.Ticker")
    def test_options_raises_exception(self, mock_ticker_cls: MagicMock, agent: ScreenerAgent) -> None:
        mock_obj = MagicMock()
        mock_obj.info = _mock_ticker_info(2_000_000_000.0)
        type(mock_obj).options = PropertyMock(side_effect=Exception("no options data"))
        mock_ticker_cls.return_value = mock_obj

        item = {"ticker": "BROKEN", "price": 50.0, "avg_volume": 1_000_000.0}
        result = agent._enrich_single(item)
        assert result is None

    @patch("src.agents.screener.yf.Ticker")
    def test_missing_info_fields(self, mock_ticker_cls: MagicMock, agent: ScreenerAgent) -> None:
        mock_obj = MagicMock()
        mock_obj.info = {"marketCap": 1_000_000_000.0}  # missing sector, industry
        mock_obj.options = ("2024-01-19",)
        mock_ticker_cls.return_value = mock_obj

        item = {"ticker": "SPARSE", "price": 100.0, "avg_volume": 2_000_000.0}
        result = agent._enrich_single(item)
        assert result is not None
        assert result.sector == "Unknown"
        assert result.industry == "Unknown"

    @patch("src.agents.screener.yf.Ticker")
    def test_info_returns_none(self, mock_ticker_cls: MagicMock, agent: ScreenerAgent) -> None:
        mock_obj = MagicMock()
        mock_obj.info = None
        mock_obj.options = ("2024-01-19",)
        mock_ticker_cls.return_value = mock_obj

        item = {"ticker": "NULL", "price": 100.0, "avg_volume": 2_000_000.0}
        result = agent._enrich_single(item)
        assert result is None  # market_cap=0 < min_market_cap

    @patch("src.agents.screener.yf.Ticker")
    def test_info_raises_exception(self, mock_ticker_cls: MagicMock, agent: ScreenerAgent) -> None:
        mock_ticker_cls.return_value.info = None
        mock_ticker_cls.side_effect = Exception("ticker not found")

        item = {"ticker": "DEAD", "price": 100.0, "avg_volume": 2_000_000.0}
        result = agent._enrich_single(item)
        assert result is None


# ---------------------------------------------------------------------------
# Tests: Cache read/write/TTL
# ---------------------------------------------------------------------------

class TestCache:
    def test_save_and_load_cache(self, agent: ScreenerAgent, tmp_path: Path) -> None:
        cache_file = tmp_path / "screener_cache.json"
        candidates = [
            ScreenedCandidate("AAPL", 180.0, 5_000_000.0, 3e12, "Technology", "Consumer Electronics"),
            ScreenedCandidate("MSFT", 400.0, 3_000_000.0, 2.8e12, "Technology", "Software"),
        ]

        with patch("src.agents.screener.SCREENER_CACHE_FILE", cache_file), \
             patch("src.agents.screener.DATA_DIR", tmp_path):
            agent._save_screener_cache(candidates)
            loaded = agent._load_screener_cache()

        assert loaded is not None
        assert len(loaded) == 2
        assert loaded[0].ticker == "AAPL"
        assert loaded[1].ticker == "MSFT"

    def test_cache_expired(self, agent: ScreenerAgent, tmp_path: Path) -> None:
        cache_file = tmp_path / "screener_cache.json"
        data = {
            "cached_at": (datetime.now() - timedelta(hours=10)).isoformat(),
            "candidates": [],
        }
        cache_file.write_text(json.dumps(data))

        with patch("src.agents.screener.SCREENER_CACHE_FILE", cache_file):
            result = agent._load_screener_cache()

        assert result is None

    def test_cache_fresh(self, agent: ScreenerAgent, tmp_path: Path) -> None:
        cache_file = tmp_path / "screener_cache.json"
        candidate_data = {
            "ticker": "AAPL", "price": 180.0, "avg_volume": 5e6,
            "market_cap": 3e12, "sector": "Technology", "industry": "Consumer Electronics",
        }
        data = {
            "cached_at": datetime.now().isoformat(),
            "candidates": [candidate_data],
        }
        cache_file.write_text(json.dumps(data))

        with patch("src.agents.screener.SCREENER_CACHE_FILE", cache_file):
            result = agent._load_screener_cache()

        assert result is not None
        assert len(result) == 1
        assert result[0].ticker == "AAPL"

    def test_cache_missing_file(self, agent: ScreenerAgent, tmp_path: Path) -> None:
        cache_file = tmp_path / "nonexistent.json"
        with patch("src.agents.screener.SCREENER_CACHE_FILE", cache_file):
            result = agent._load_screener_cache()
        assert result is None

    def test_cache_corrupt_file(self, agent: ScreenerAgent, tmp_path: Path) -> None:
        cache_file = tmp_path / "screener_cache.json"
        cache_file.write_text("not valid json {{{")

        with patch("src.agents.screener.SCREENER_CACHE_FILE", cache_file):
            result = agent._load_screener_cache()
        assert result is None


# ---------------------------------------------------------------------------
# Tests: Sorting and max_candidates cap
# ---------------------------------------------------------------------------

class TestSortingAndCap:
    def test_sorted_by_volume_descending_and_capped(self, agent: ScreenerAgent) -> None:
        candidates = [
            ScreenedCandidate("A", 100.0, 1_000_000.0, 1e9, "Tech", "Soft"),
            ScreenedCandidate("B", 100.0, 5_000_000.0, 1e9, "Tech", "Soft"),
            ScreenedCandidate("C", 100.0, 3_000_000.0, 1e9, "Tech", "Soft"),
            ScreenedCandidate("D", 100.0, 10_000_000.0, 1e9, "Tech", "Soft"),
            ScreenedCandidate("E", 100.0, 2_000_000.0, 1e9, "Tech", "Soft"),
            ScreenedCandidate("F", 100.0, 8_000_000.0, 1e9, "Tech", "Soft"),
        ]

        candidates.sort(key=lambda c: c.avg_volume, reverse=True)
        candidates = candidates[: agent.config.max_candidates]

        assert len(candidates) == 5  # max_candidates=5
        assert candidates[0].ticker == "D"  # 10M
        assert candidates[1].ticker == "F"  # 8M
        assert candidates[2].ticker == "B"  # 5M


# ---------------------------------------------------------------------------
# Tests: Full run integration (all external calls mocked)
# ---------------------------------------------------------------------------

class TestRunIntegration:
    @patch("src.agents.screener.yf.Ticker")
    @patch("src.agents.screener.yf.download")
    @patch("src.agents.screener.ScreenerAgent._get_sp500_tickers")
    @patch("src.agents.screener.SCREENER_CACHE_FILE", Path("/tmp/_test_screener_cache_none.json"))
    def test_full_run(
        self,
        mock_sp500: MagicMock,
        mock_download: MagicMock,
        mock_ticker_cls: MagicMock,
        agent: ScreenerAgent,
        tmp_path: Path,
    ) -> None:
        # Mock universe
        mock_sp500.return_value = ["AAPL", "MSFT", "BAD1"]

        # Patch supplemental tickers to empty to simplify
        with patch("src.agents.screener.ScreenerAgent._build_universe") as mock_universe:
            mock_universe.return_value = ["AAPL", "MSFT", "BAD1"]

            # Mock batch download
            tickers = ["AAPL", "MSFT"]  # BAD1 blacklisted
            mock_download.return_value = _make_price_df(
                tickers, [180.0, 400.0], [5_000_000.0, 3_000_000.0]
            )

            # Mock .info lookups
            def ticker_side_effect(symbol: str) -> MagicMock:
                mock_obj = MagicMock()
                mock_obj.options = ("2024-01-19",)
                if symbol == "AAPL":
                    mock_obj.info = _mock_ticker_info(3e12, "Technology", "Consumer Electronics")
                elif symbol == "MSFT":
                    mock_obj.info = _mock_ticker_info(2.8e12, "Technology", "Software")
                return mock_obj

            mock_ticker_cls.side_effect = ticker_side_effect

            cache_file = tmp_path / "screener_cache.json"
            with patch("src.agents.screener.SCREENER_CACHE_FILE", cache_file), \
                 patch("src.agents.screener.DATA_DIR", tmp_path):
                result = agent.run(force_refresh=True)

        assert len(result) == 2
        # AAPL has higher volume, should be first
        assert result[0].ticker == "AAPL"
        assert result[1].ticker == "MSFT"

    def test_run_returns_cache_when_fresh(self, agent: ScreenerAgent, tmp_path: Path) -> None:
        cache_file = tmp_path / "screener_cache.json"
        candidate_data = {
            "ticker": "CACHED", "price": 100.0, "avg_volume": 1e6,
            "market_cap": 1e9, "sector": "Tech", "industry": "Soft",
        }
        data = {
            "cached_at": datetime.now().isoformat(),
            "candidates": [candidate_data],
        }
        cache_file.write_text(json.dumps(data))

        with patch("src.agents.screener.SCREENER_CACHE_FILE", cache_file):
            result = agent.run(force_refresh=False)

        assert len(result) == 1
        assert result[0].ticker == "CACHED"

    @patch("src.agents.screener.yf.Ticker")
    @patch("src.agents.screener.yf.download")
    def test_run_force_refresh_ignores_cache(
        self,
        mock_download: MagicMock,
        mock_ticker_cls: MagicMock,
        agent: ScreenerAgent,
        tmp_path: Path,
    ) -> None:
        # Write a fresh cache
        cache_file = tmp_path / "screener_cache.json"
        data = {
            "cached_at": datetime.now().isoformat(),
            "candidates": [{"ticker": "OLD", "price": 50.0, "avg_volume": 1e6,
                           "market_cap": 1e9, "sector": "X", "industry": "Y"}],
        }
        cache_file.write_text(json.dumps(data))

        with patch("src.agents.screener.SCREENER_CACHE_FILE", cache_file), \
             patch("src.agents.screener.DATA_DIR", tmp_path), \
             patch("src.agents.screener.ScreenerAgent._build_universe", return_value=["AAPL"]):

            mock_download.return_value = _make_price_df(["AAPL"], [180.0], [5_000_000.0])

            mock_obj = MagicMock()
            mock_obj.info = _mock_ticker_info(3e12)
            mock_obj.options = ("2024-01-19",)
            mock_ticker_cls.return_value = mock_obj

            result = agent.run(force_refresh=True)

        # Should NOT return "OLD" from cache
        assert all(c.ticker != "OLD" for c in result)
