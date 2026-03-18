from __future__ import annotations

import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch, call

import pytest

from src.agents.sentiment import SentimentAgent
from src.config import SentimentConfig
from src.types import TechnicalSetup, SentimentResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config() -> SentimentConfig:
    return SentimentConfig(
        api_call_delay_seconds=0.0,  # No delay in tests
        max_retries=3,
    )


@pytest.fixture
def agent(config: SentimentConfig) -> SentimentAgent:
    with patch("src.agents.sentiment.anthropic.Anthropic"):
        a = SentimentAgent(config)
    return a


@pytest.fixture
def sample_setups() -> list[TechnicalSetup]:
    return [
        TechnicalSetup(
            ticker="AAPL",
            setup_type="squeeze_breakout",
            direction="long",
            technical_score=8.5,
            support_level=170.0,
            resistance_level=185.0,
        ),
        TechnicalSetup(
            ticker="MSFT",
            setup_type="bull_flag",
            direction="long",
            technical_score=7.2,
            support_level=400.0,
            resistance_level=420.0,
        ),
    ]


def _make_api_response(json_data: dict) -> MagicMock:
    """Build a mock anthropic Message with a single text block."""
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = json.dumps(json_data)

    response = MagicMock()
    response.content = [text_block]
    response.usage = SimpleNamespace(input_tokens=500, output_tokens=200)
    return response


def _make_api_response_from_text(text: str) -> MagicMock:
    """Build a mock anthropic Message with arbitrary text."""
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text

    response = MagicMock()
    response.content = [text_block]
    response.usage = SimpleNamespace(input_tokens=500, output_tokens=200)
    return response


CLEAN_JSON = {
    "ticker": "AAPL",
    "sentiment": "bullish",
    "confidence": 7.5,
    "catalysts": ["Strong iPhone sales", "Services revenue growth"],
    "risk_flags": [],
    "earnings_date": "2025-04-25",
    "days_to_earnings": 30,
    "summary": "Positive sentiment driven by strong product cycle.",
}


# ---------------------------------------------------------------------------
# Response Parsing — Clean JSON
# ---------------------------------------------------------------------------


class TestParseCleanJSON:
    def test_parses_valid_json(self, agent: SentimentAgent) -> None:
        result = agent._parse_json_text("AAPL", json.dumps(CLEAN_JSON))
        assert result.ticker == "AAPL"
        assert result.sentiment == "bullish"
        assert result.confidence == 7.5
        assert len(result.catalysts) == 2
        assert result.risk_flags == []
        assert result.earnings_date == "2025-04-25"
        assert result.days_to_earnings == 30

    def test_clamps_confidence_to_range(self, agent: SentimentAgent) -> None:
        data = {**CLEAN_JSON, "confidence": 15.0}
        result = agent._parse_json_text("AAPL", json.dumps(data))
        assert result.confidence == 10.0

        data = {**CLEAN_JSON, "confidence": -5.0}
        result = agent._parse_json_text("AAPL", json.dumps(data))
        assert result.confidence == 1.0

    def test_invalid_sentiment_defaults_to_neutral(self, agent: SentimentAgent) -> None:
        data = {**CLEAN_JSON, "sentiment": "very_bullish"}
        result = agent._parse_json_text("AAPL", json.dumps(data))
        assert result.sentiment == "neutral"

    def test_missing_fields_use_defaults(self, agent: SentimentAgent) -> None:
        data = {"ticker": "AAPL", "sentiment": "bullish", "confidence": 6.0}
        result = agent._parse_json_text("AAPL", json.dumps(data))
        assert result.catalysts == []
        assert result.risk_flags == []
        assert result.earnings_date is None
        assert result.days_to_earnings is None
        assert result.summary == ""


# ---------------------------------------------------------------------------
# Response Parsing — Markdown-Wrapped JSON
# ---------------------------------------------------------------------------


class TestParseMarkdownJSON:
    def test_parses_json_in_code_fence(self, agent: SentimentAgent) -> None:
        text = "```json\n" + json.dumps(CLEAN_JSON) + "\n```"
        result = agent._parse_json_text("AAPL", text)
        assert result.sentiment == "bullish"
        assert result.ticker == "AAPL"

    def test_parses_json_in_plain_fence(self, agent: SentimentAgent) -> None:
        text = "```\n" + json.dumps(CLEAN_JSON) + "\n```"
        result = agent._parse_json_text("AAPL", text)
        assert result.sentiment == "bullish"


# ---------------------------------------------------------------------------
# Response Parsing — Preamble Text
# ---------------------------------------------------------------------------


class TestParsePreambleJSON:
    def test_parses_json_after_preamble(self, agent: SentimentAgent) -> None:
        text = "Here is my analysis:\n\n" + json.dumps(CLEAN_JSON)
        result = agent._parse_json_text("AAPL", text)
        assert result.sentiment == "bullish"
        assert result.ticker == "AAPL"

    def test_parses_json_with_trailing_text(self, agent: SentimentAgent) -> None:
        text = json.dumps(CLEAN_JSON) + "\n\nI hope this helps!"
        result = agent._parse_json_text("AAPL", text)
        assert result.sentiment == "bullish"


# ---------------------------------------------------------------------------
# Response Parsing — Failed Parse
# ---------------------------------------------------------------------------


class TestParseFailure:
    def test_no_json_returns_default(self, agent: SentimentAgent) -> None:
        result = agent._parse_json_text("AAPL", "No JSON here at all.")
        assert result.sentiment == "neutral"
        assert result.confidence == 3.0
        assert "sentiment_parse_error" in result.risk_flags

    def test_invalid_json_returns_default(self, agent: SentimentAgent) -> None:
        result = agent._parse_json_text("AAPL", "{invalid json, not real}")
        assert result.sentiment == "neutral"
        assert result.confidence == 3.0
        assert "sentiment_parse_error" in result.risk_flags

    def test_empty_response_returns_default(self, agent: SentimentAgent) -> None:
        response = MagicMock()
        response.content = []
        result = agent._parse_response("AAPL", response)
        assert result.sentiment == "neutral"
        assert "sentiment_parse_error" in result.risk_flags


# ---------------------------------------------------------------------------
# Earnings Proximity Downgrade
# ---------------------------------------------------------------------------


class TestEarningsProximity:
    def test_downgrade_when_earnings_within_gap(self, agent: SentimentAgent) -> None:
        result = SentimentResult(
            ticker="AAPL",
            sentiment="bullish",
            confidence=8.0,
            catalysts=[],
            risk_flags=[],
            earnings_date="2025-02-01",
            days_to_earnings=5,
            summary="Test.",
        )
        updated = agent._apply_earnings_check(result)
        assert updated.confidence == 5.0  # 8.0 - 3.0
        assert "earnings_within_7_days" in updated.risk_flags

    def test_no_downgrade_when_earnings_far(self, agent: SentimentAgent) -> None:
        result = SentimentResult(
            ticker="AAPL",
            sentiment="bullish",
            confidence=8.0,
            catalysts=[],
            risk_flags=[],
            earnings_date="2025-04-01",
            days_to_earnings=30,
            summary="Test.",
        )
        updated = agent._apply_earnings_check(result)
        assert updated.confidence == 8.0
        assert "earnings_within_7_days" not in updated.risk_flags

    def test_confidence_floors_at_1(self, agent: SentimentAgent) -> None:
        result = SentimentResult(
            ticker="AAPL",
            sentiment="bullish",
            confidence=2.0,
            catalysts=[],
            risk_flags=[],
            earnings_date="2025-02-01",
            days_to_earnings=3,
            summary="Test.",
        )
        updated = agent._apply_earnings_check(result)
        assert updated.confidence == 1.0

    def test_no_downgrade_when_days_is_none(self, agent: SentimentAgent) -> None:
        result = SentimentResult(
            ticker="AAPL",
            sentiment="bullish",
            confidence=8.0,
            catalysts=[],
            risk_flags=[],
            earnings_date=None,
            days_to_earnings=None,
            summary="Test.",
        )
        updated = agent._apply_earnings_check(result)
        assert updated.confidence == 8.0

    def test_does_not_duplicate_risk_flag(self, agent: SentimentAgent) -> None:
        result = SentimentResult(
            ticker="AAPL",
            sentiment="bullish",
            confidence=8.0,
            catalysts=[],
            risk_flags=["earnings_within_7_days"],
            earnings_date="2025-02-01",
            days_to_earnings=3,
            summary="Test.",
        )
        updated = agent._apply_earnings_check(result)
        assert updated.risk_flags.count("earnings_within_7_days") == 1


# ---------------------------------------------------------------------------
# Retry / Backoff Behavior
# ---------------------------------------------------------------------------


class TestRetryBackoff:
    def test_retries_on_rate_limit(self, agent: SentimentAgent) -> None:
        import anthropic as anthropic_mod

        setup = TechnicalSetup(
            ticker="AAPL",
            setup_type="squeeze_breakout",
            direction="long",
            technical_score=8.0,
            support_level=170.0,
            resistance_level=185.0,
        )

        # Fail twice with rate limit, succeed on third
        mock_response = _make_api_response(CLEAN_JSON)
        agent.client.messages.create = MagicMock(
            side_effect=[
                anthropic_mod.RateLimitError(
                    message="rate limited",
                    response=MagicMock(status_code=429, headers={}),
                    body=None,
                ),
                anthropic_mod.RateLimitError(
                    message="rate limited",
                    response=MagicMock(status_code=429, headers={}),
                    body=None,
                ),
                mock_response,
            ]
        )

        with patch("src.agents.sentiment.time.sleep"):
            result = agent._analyze_ticker("AAPL", setup)

        assert result.sentiment == "bullish"
        assert agent.client.messages.create.call_count == 3

    def test_returns_default_after_exhausted_retries(self, agent: SentimentAgent) -> None:
        import anthropic as anthropic_mod

        setup = TechnicalSetup(
            ticker="AAPL",
            setup_type="squeeze_breakout",
            direction="long",
            technical_score=8.0,
            support_level=170.0,
            resistance_level=185.0,
        )

        agent.client.messages.create = MagicMock(
            side_effect=anthropic_mod.RateLimitError(
                message="rate limited",
                response=MagicMock(status_code=429, headers={}),
                body=None,
            )
        )

        with patch("src.agents.sentiment.time.sleep"):
            result = agent._analyze_ticker("AAPL", setup)

        assert result.sentiment == "neutral"
        assert result.confidence == 3.0
        assert "api_call_failed" in result.risk_flags

    def test_retries_on_api_error(self, agent: SentimentAgent) -> None:
        import anthropic as anthropic_mod

        setup = TechnicalSetup(
            ticker="AAPL",
            setup_type="squeeze_breakout",
            direction="long",
            technical_score=8.0,
            support_level=170.0,
            resistance_level=185.0,
        )

        mock_response = _make_api_response(CLEAN_JSON)
        agent.client.messages.create = MagicMock(
            side_effect=[
                anthropic_mod.APIError(
                    message="server error",
                    request=MagicMock(),
                    body=None,
                ),
                mock_response,
            ]
        )

        with patch("src.agents.sentiment.time.sleep"):
            result = agent._analyze_ticker("AAPL", setup)

        assert result.sentiment == "bullish"
        assert agent.client.messages.create.call_count == 2


# ---------------------------------------------------------------------------
# Batching / Deduplication
# ---------------------------------------------------------------------------


class TestBatchingDeduplication:
    def test_deduplicates_tickers(self, agent: SentimentAgent) -> None:
        setups = [
            TechnicalSetup(
                ticker="AAPL",
                setup_type="squeeze_breakout",
                direction="long",
                technical_score=8.5,
                support_level=170.0,
                resistance_level=185.0,
            ),
            TechnicalSetup(
                ticker="AAPL",
                setup_type="bull_flag",
                direction="long",
                technical_score=7.0,
                support_level=165.0,
                resistance_level=180.0,
            ),
            TechnicalSetup(
                ticker="MSFT",
                setup_type="trend_pullback",
                direction="long",
                technical_score=6.5,
                support_level=400.0,
                resistance_level=420.0,
            ),
        ]

        mock_response = _make_api_response(CLEAN_JSON)
        agent.client.messages.create = MagicMock(return_value=mock_response)

        with patch.object(agent, "_record_usage"):
            results = agent.run(setups)

        # Should only make 2 API calls (one per unique ticker)
        assert agent.client.messages.create.call_count == 2
        assert len(results) == 2

    def test_respects_delay_between_calls(self, agent: SentimentAgent) -> None:
        agent.config.api_call_delay_seconds = 0.5
        setups = [
            TechnicalSetup(
                ticker="AAPL",
                setup_type="squeeze_breakout",
                direction="long",
                technical_score=8.5,
                support_level=170.0,
                resistance_level=185.0,
            ),
            TechnicalSetup(
                ticker="MSFT",
                setup_type="bull_flag",
                direction="long",
                technical_score=7.2,
                support_level=400.0,
                resistance_level=420.0,
            ),
        ]

        mock_response = _make_api_response(CLEAN_JSON)
        agent.client.messages.create = MagicMock(return_value=mock_response)

        with patch("src.agents.sentiment.time.sleep") as mock_sleep, \
             patch.object(agent, "_record_usage"):
            agent.run(setups)

        # Should sleep once between 2 tickers
        mock_sleep.assert_called_once_with(0.5)

    def test_empty_setups_returns_empty(self, agent: SentimentAgent) -> None:
        results = agent.run([])
        assert results == []


# ---------------------------------------------------------------------------
# Cost Tracking
# ---------------------------------------------------------------------------


class TestCostTracking:
    def test_records_usage_to_file(self, agent: SentimentAgent, tmp_path) -> None:
        usage_file = tmp_path / "api_usage.json"

        with patch("src.agents.sentiment._USAGE_FILE", str(usage_file)):
            agent._total_input_tokens = 1000
            agent._total_output_tokens = 500
            agent._api_calls = 2
            agent._record_usage()

        data = json.loads(usage_file.read_text())
        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["api_calls"] == 2
        assert data[0]["input_tokens"] == 1000
        assert data[0]["output_tokens"] == 500
        assert "estimated_cost_usd" in data[0]

    def test_appends_to_existing_usage_file(self, agent: SentimentAgent, tmp_path) -> None:
        usage_file = tmp_path / "api_usage.json"
        usage_file.write_text(json.dumps([{"api_calls": 1}]))

        with patch("src.agents.sentiment._USAGE_FILE", str(usage_file)):
            agent._total_input_tokens = 500
            agent._total_output_tokens = 200
            agent._api_calls = 1
            agent._record_usage()

        data = json.loads(usage_file.read_text())
        assert len(data) == 2


# ---------------------------------------------------------------------------
# Full Run Integration (mocked API)
# ---------------------------------------------------------------------------


class TestFullRun:
    def test_full_run_returns_results(self, agent: SentimentAgent, sample_setups) -> None:
        aapl_json = {**CLEAN_JSON, "ticker": "AAPL"}
        msft_json = {**CLEAN_JSON, "ticker": "MSFT", "sentiment": "neutral"}

        responses = [
            _make_api_response(aapl_json),
            _make_api_response(msft_json),
        ]
        agent.client.messages.create = MagicMock(side_effect=responses)

        with patch.object(agent, "_record_usage"):
            results = agent.run(sample_setups)

        assert len(results) == 2
        tickers = {r.ticker for r in results}
        assert "AAPL" in tickers
        assert "MSFT" in tickers

    def test_single_ticker_failure_does_not_crash_run(
        self, agent: SentimentAgent, sample_setups
    ) -> None:
        import anthropic as anthropic_mod

        # First ticker fails all retries, second succeeds
        msft_json = {**CLEAN_JSON, "ticker": "MSFT"}
        agent.client.messages.create = MagicMock(
            side_effect=[
                anthropic_mod.APIError(
                    message="server error",
                    request=MagicMock(),
                    body=None,
                ),
                anthropic_mod.APIError(
                    message="server error",
                    request=MagicMock(),
                    body=None,
                ),
                anthropic_mod.APIError(
                    message="server error",
                    request=MagicMock(),
                    body=None,
                ),
                _make_api_response(msft_json),
            ]
        )

        with patch("src.agents.sentiment.time.sleep"), \
             patch.object(agent, "_record_usage"):
            results = agent.run(sample_setups)

        assert len(results) == 2
        aapl_result = next(r for r in results if r.ticker == "AAPL")
        assert aapl_result.sentiment == "neutral"
        assert "api_call_failed" in aapl_result.risk_flags
