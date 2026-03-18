# Prompt 4: Sentiment Agent (Claude API + Web Search)

## Context
The Technical Agent (Step 3) outputs a `list[TechnicalSetup]` of tickers with detected swing setups and scores. This step implements the **Sentiment Agent** — the module that calls the Anthropic Claude API with web search enabled to assess news, catalysts, and sentiment for each technically-qualified ticker. This is where the LLM reasoning layer adds intelligence on top of the quantitative signals.

## What to Build

### Implement `src/agents/sentiment.py`

The `SentimentAgent` class should:

#### 1. Input Filtering
- Accept `list[TechnicalSetup]` as input
- Sort by `technical_score` descending and take only the top N tickers (configurable via `SentimentConfig.max_tickers_per_batch` — but this is per API call; process all qualified setups, just batch them)
- Deduplicate: if a ticker has multiple setups, only run sentiment once and attach the result to all its setups

#### 2. Claude API Call Structure
For each ticker (or small batch of tickers), make a Claude API call using the `anthropic` Python SDK with web search enabled.

**API call configuration:**
```python
import anthropic

client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY from env

response = client.messages.create(
    model=config.model,  # "claude-sonnet-4-20250514"
    max_tokens=config.max_tokens,
    tools=[
        {
            "type": "web_search_20250305",
            "name": "web_search"
        }
    ],
    messages=[
        {"role": "user", "content": prompt}
    ]
)
```

**System prompt for sentiment analysis (store in `src/prompts/sentiment_system.py`):**
```
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
}
```

**User prompt template:**
```
Analyze the following ticker for swing trade sentiment:

Ticker: {ticker}
Current Price: ${price}
Technical Setup: {setup_type}
Direction: {direction}
Key Levels: Support at ${support}, Resistance at ${resistance}

Search for recent news, upcoming events, and market sentiment for {ticker}. Focus on anything that would impact a {direction} swing trade over the next 1-3 weeks.
```

#### 3. Response Parsing
- Extract the JSON from Claude's response. Handle cases where Claude includes markdown code fences or preamble text — strip those before parsing.
- Parse into `SentimentResult` dataclass
- If parsing fails, log the raw response and assign a default neutral sentiment with confidence 3 and a risk flag of "sentiment_parse_error"

#### 4. Batching and Rate Limiting
- Process tickers sequentially (not in parallel) to respect API rate limits
- Add a configurable delay between calls (default 1 second, stored in `SentimentConfig`)
- If a call fails (rate limit, API error), retry up to 3 times with exponential backoff
- Log each API call: ticker, response time, tokens used (from response headers)

#### 5. Earnings Proximity Check
- If the sentiment response includes `days_to_earnings` and it's less than `SentimentConfig.min_earnings_gap_days` (default 7), automatically set a risk flag and downgrade sentiment confidence by 3 points (floor at 1)
- This is a hard filter — the scoring agent will likely drop these, but flag them here

#### 6. Output
- Return `list[SentimentResult]` for all analyzed tickers
- Log: total API calls made, total tokens consumed, sentiment distribution (bullish/neutral/bearish counts), tickers flagged for earnings proximity

#### 7. Cost Tracking
- Track and log the estimated cost of the sentiment analysis run. Use approximate token pricing for the configured model. Store running totals in a simple `data/api_usage.json` file that appends per run.

### Create `src/prompts/sentiment_system.py`
Store the system and user prompt templates as string constants so they're easy to iterate on without touching agent logic.

### Tests: `tests/test_sentiment.py`
Write tests for:
- Response parsing with clean JSON
- Response parsing with markdown-wrapped JSON (```json ... ```)
- Response parsing with preamble text before the JSON
- Failed parse falls back to neutral default
- Earnings proximity downgrade logic
- Retry/backoff behavior (mock the API client)
- Batching respects the configured delay

## Important Notes
- Use `anthropic` SDK, not raw HTTP requests. The SDK handles auth, retries, and streaming.
- The web search tool is critical — without it, Claude can only use training data which may be stale for recent catalysts.
- Process one ticker per API call for now. Batching multiple tickers into one call can confuse the search behavior (Claude might only search for the first ticker). We can optimize later if cost is an issue.
- The system prompt must enforce JSON-only output. Claude is good at this but occasionally adds commentary — the parser must handle that gracefully.
- Store prompt templates separately from agent logic. When tuning the sentiment analysis, you only want to edit the prompts, not the plumbing.
- This is the most expensive step in the pipeline. The orchestrator should only send tickers here that passed technical screening — never the full 300-ticker universe.
