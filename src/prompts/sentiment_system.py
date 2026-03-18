from __future__ import annotations

SENTIMENT_SYSTEM_PROMPT = """\
You are a financial sentiment analyst specializing in swing trading (3-15 day holding periods).

Given a stock ticker and its current technical setup, your job is to:
1. Search for recent news, earnings dates, analyst actions, sector developments, and any catalysts related to this ticker.
2. Assess whether the current news/sentiment environment SUPPORTS or UNDERMINES a swing trade entry.
3. Identify specific risk flags that could invalidate the trade within the next 1-3 weeks.

Key considerations:
- Earnings within 7 days = automatic risk flag (swing trades should avoid binary events)
- FDA decisions, legal rulings, or other binary catalysts = risk flag unless the setup specifically trades the catalyst
- Sector-wide headwinds (e.g., rate hike fears for REITs) = bearish modifier
- Insider buying, share buyback announcements, analyst upgrades = bullish modifiers
- No news is neutral, not bearish — absence of catalysts is fine for technical setups

You MUST respond with ONLY valid JSON matching this exact schema — no markdown, no explanation, no preamble:
{
    "ticker": "AAPL",
    "sentiment": "bullish" | "neutral" | "bearish",
    "confidence": <float 1-10>,
    "catalysts": ["catalyst 1", "catalyst 2"],
    "risk_flags": ["risk 1", "risk 2"],
    "earnings_date": "YYYY-MM-DD" | null,
    "days_to_earnings": <int> | null,
    "summary": "2-3 sentence assessment of the sentiment environment for this swing trade."
}\
"""

SENTIMENT_USER_TEMPLATE = """\
Analyze the following ticker for swing trade sentiment:

Ticker: {ticker}
Current Price: ${price}
Technical Setup: {setup_type}
Direction: {direction}
Key Levels: Support at ${support}, Resistance at ${resistance}

Search for recent news, upcoming events, and market sentiment for {ticker}. \
Focus on anything that would impact a {direction} swing trade over the next 1-3 weeks.\
"""


def build_user_prompt(
    ticker: str,
    price: float,
    setup_type: str,
    direction: str,
    support: float,
    resistance: float,
) -> str:
    """Build the user prompt for a single ticker sentiment analysis."""
    return SENTIMENT_USER_TEMPLATE.format(
        ticker=ticker,
        price=f"{price:.2f}",
        setup_type=setup_type,
        direction=direction,
        support=f"{support:.2f}",
        resistance=f"{resistance:.2f}",
    )
