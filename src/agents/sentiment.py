from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone

import anthropic

from src.config import SentimentConfig
from src.types import TechnicalSetup, SentimentResult
from src.prompts.sentiment_system import SENTIMENT_SYSTEM_PROMPT, build_user_prompt
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Approximate token pricing (per 1M tokens) for cost tracking
_MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-20250514": {"input": 3.0, "output": 15.0},
}
_DEFAULT_PRICING = {"input": 3.0, "output": 15.0}

_USAGE_FILE = os.path.join("data", "api_usage.json")


class SentimentAgent:
    """Analyzes sentiment for technically-qualified tickers via Claude API."""

    def __init__(self, config: SentimentConfig) -> None:
        self.config = config
        self.client = anthropic.Anthropic()
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._api_calls = 0

    def run(self, setups: list[TechnicalSetup]) -> list[SentimentResult]:
        """Run sentiment analysis on tickers with technical setups.

        1. Deduplicate tickers (one API call per unique ticker)
        2. Sort by technical_score desc and process sequentially
        3. Attach results to all setups for each ticker
        4. Apply earnings proximity checks
        5. Log summary and track costs
        """
        if not setups:
            logger.info("No setups to analyze for sentiment")
            return []

        # Deduplicate: group setups by ticker, pick highest-scoring for prompt context
        ticker_best: dict[str, TechnicalSetup] = {}
        for setup in sorted(setups, key=lambda s: s.technical_score, reverse=True):
            if setup.ticker not in ticker_best:
                ticker_best[setup.ticker] = setup

        unique_tickers = list(ticker_best.keys())
        logger.info(
            f"Sentiment analysis for {len(unique_tickers)} unique tickers "
            f"(from {len(setups)} setups)"
        )

        results: list[SentimentResult] = []
        for i, ticker in enumerate(unique_tickers):
            setup = ticker_best[ticker]
            logger.info(f"[{i + 1}/{len(unique_tickers)}] Analyzing {ticker}")

            result = self._analyze_ticker(ticker, setup)
            result = self._apply_earnings_check(result)
            results.append(result)

            # Delay between calls (skip after last)
            if i < len(unique_tickers) - 1:
                time.sleep(self.config.api_call_delay_seconds)

        self._log_summary(results)
        self._record_usage()

        return results

    def _analyze_ticker(self, ticker: str, setup: TechnicalSetup) -> SentimentResult:
        """Make a Claude API call for a single ticker with retry logic."""
        prompt = build_user_prompt(
            ticker=ticker,
            price=setup.support_level,  # Use support as approximate current price
            setup_type=setup.setup_type,
            direction=setup.direction,
            support=setup.support_level,
            resistance=setup.resistance_level,
        )

        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                start_time = time.time()
                response = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=self.config.max_tokens,
                    system=SENTIMENT_SYSTEM_PROMPT,
                    tools=[{"type": "web_search_20250305", "name": "web_search"}],
                    messages=[{"role": "user", "content": prompt}],
                )
                elapsed = time.time() - start_time

                # Track token usage
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                self._total_input_tokens += input_tokens
                self._total_output_tokens += output_tokens
                self._api_calls += 1

                logger.info(
                    f"  {ticker}: {elapsed:.1f}s, "
                    f"{input_tokens} in / {output_tokens} out tokens"
                )

                return self._parse_response(ticker, response)

            except anthropic.RateLimitError as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning(
                    f"  {ticker}: rate limited (attempt {attempt}/{self.config.max_retries}), "
                    f"waiting {wait}s"
                )
                time.sleep(wait)

            except anthropic.APIError as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning(
                    f"  {ticker}: API error (attempt {attempt}/{self.config.max_retries}): {e}, "
                    f"waiting {wait}s"
                )
                time.sleep(wait)

            except Exception as e:
                last_error = e
                logger.exception(f"  {ticker}: unexpected error on attempt {attempt}")
                break

        logger.error(f"  {ticker}: all retries exhausted — {last_error}")
        return self._default_result(ticker, risk_flag="api_call_failed")

    def _parse_response(self, ticker: str, response: anthropic.types.Message) -> SentimentResult:
        """Extract JSON from Claude's response and parse into SentimentResult."""
        # Collect text blocks from the response
        text_parts: list[str] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)

        if not text_parts:
            logger.warning(f"  {ticker}: no text content in response")
            return self._default_result(ticker, risk_flag="sentiment_parse_error")

        raw_text = "\n".join(text_parts)
        return self._parse_json_text(ticker, raw_text)

    def _parse_json_text(self, ticker: str, raw_text: str) -> SentimentResult:
        """Parse JSON from raw text, handling markdown fences and preamble."""
        # Try to extract JSON from markdown code fences first
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
        if fence_match:
            json_str = fence_match.group(1)
        else:
            # Try to find a JSON object in the raw text
            brace_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
            if brace_match:
                json_str = brace_match.group(0)
            else:
                logger.warning(f"  {ticker}: no JSON found in response")
                logger.debug(f"  Raw response: {raw_text[:500]}")
                return self._default_result(ticker, risk_flag="sentiment_parse_error")

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            logger.warning(f"  {ticker}: JSON parse failed")
            logger.debug(f"  JSON string: {json_str[:500]}")
            return self._default_result(ticker, risk_flag="sentiment_parse_error")

        # Validate and extract fields
        sentiment = data.get("sentiment", "neutral")
        if sentiment not in ("bullish", "neutral", "bearish"):
            sentiment = "neutral"

        confidence = data.get("confidence", 5.0)
        try:
            confidence = float(confidence)
            confidence = max(1.0, min(10.0, confidence))
        except (TypeError, ValueError):
            confidence = 5.0

        catalysts = data.get("catalysts", [])
        if not isinstance(catalysts, list):
            catalysts = []

        risk_flags = data.get("risk_flags", [])
        if not isinstance(risk_flags, list):
            risk_flags = []

        earnings_date = data.get("earnings_date")
        if earnings_date is not None and not isinstance(earnings_date, str):
            earnings_date = None

        days_to_earnings = data.get("days_to_earnings")
        if days_to_earnings is not None:
            try:
                days_to_earnings = int(days_to_earnings)
            except (TypeError, ValueError):
                days_to_earnings = None

        summary = data.get("summary", "")
        if not isinstance(summary, str):
            summary = ""

        return SentimentResult(
            ticker=data.get("ticker", ticker),
            sentiment=sentiment,
            confidence=round(confidence, 1),
            catalysts=catalysts,
            risk_flags=risk_flags,
            earnings_date=earnings_date,
            days_to_earnings=days_to_earnings,
            summary=summary,
        )

    def _apply_earnings_check(self, result: SentimentResult) -> SentimentResult:
        """Downgrade confidence if earnings are too close."""
        if (
            result.days_to_earnings is not None
            and result.days_to_earnings < self.config.min_earnings_gap_days
        ):
            logger.info(
                f"  {result.ticker}: earnings in {result.days_to_earnings} days — "
                f"downgrading confidence by 3"
            )
            if "earnings_within_7_days" not in result.risk_flags:
                result.risk_flags.append("earnings_within_7_days")
            result.confidence = max(1.0, result.confidence - 3.0)
        return result

    def _default_result(self, ticker: str, risk_flag: str) -> SentimentResult:
        """Return a neutral default when parsing or API calls fail."""
        return SentimentResult(
            ticker=ticker,
            sentiment="neutral",
            confidence=3.0,
            catalysts=[],
            risk_flags=[risk_flag],
            earnings_date=None,
            days_to_earnings=None,
            summary="Sentiment analysis unavailable.",
        )

    def _log_summary(self, results: list[SentimentResult]) -> None:
        """Log sentiment distribution and flagged tickers."""
        if not results:
            return

        counts = {"bullish": 0, "neutral": 0, "bearish": 0}
        earnings_flagged = 0
        for r in results:
            counts[r.sentiment] = counts.get(r.sentiment, 0) + 1
            if "earnings_within_7_days" in r.risk_flags:
                earnings_flagged += 1

        logger.info(
            f"Sentiment summary: {len(results)} tickers analyzed — "
            f"bullish={counts['bullish']}, neutral={counts['neutral']}, "
            f"bearish={counts['bearish']}"
        )
        logger.info(
            f"API usage: {self._api_calls} calls, "
            f"{self._total_input_tokens} input tokens, "
            f"{self._total_output_tokens} output tokens"
        )
        if earnings_flagged:
            logger.info(f"Earnings proximity flags: {earnings_flagged} tickers")

    def _record_usage(self) -> None:
        """Append usage stats to data/api_usage.json."""
        pricing = _MODEL_PRICING.get(self.config.model, _DEFAULT_PRICING)
        estimated_cost = (
            self._total_input_tokens / 1_000_000 * pricing["input"]
            + self._total_output_tokens / 1_000_000 * pricing["output"]
        )

        entry = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "model": self.config.model,
            "api_calls": self._api_calls,
            "input_tokens": self._total_input_tokens,
            "output_tokens": self._total_output_tokens,
            "estimated_cost_usd": round(estimated_cost, 4),
        }

        # Load existing usage data
        usage_data: list[dict] = []
        if os.path.exists(_USAGE_FILE):
            try:
                with open(_USAGE_FILE) as f:
                    usage_data = json.load(f)
                if not isinstance(usage_data, list):
                    usage_data = []
            except (json.JSONDecodeError, OSError):
                usage_data = []

        usage_data.append(entry)

        try:
            os.makedirs(os.path.dirname(_USAGE_FILE), exist_ok=True)
            with open(_USAGE_FILE, "w") as f:
                json.dump(usage_data, f, indent=2)
            logger.info(f"Estimated cost: ${estimated_cost:.4f}")
        except OSError:
            logger.warning("Failed to write API usage file")
