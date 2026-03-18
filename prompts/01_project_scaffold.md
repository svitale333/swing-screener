# Prompt 1: Project Scaffold

## Context
I'm building a swing trade screener agent in this repo (`swing-screener`). The agent uses a sub-agent architecture: Screener → Technical Analyzer → Sentiment Analyst → Scoring Engine, coordinated by an Orchestrator. This first step sets up the project structure, dependencies, configuration, and shared types that all subsequent agents will use.

## What to Build

### 1. Directory Structure
Create the following:

```
swing-screener/
├── src/
│   ├── __init__.py
│   ├── config.py              # All configurable parameters
│   ├── types.py               # Shared dataclasses / TypedDicts
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── screener.py        # (placeholder) Universe screening
│   │   ├── technical.py       # (placeholder) Technical setup detection
│   │   ├── sentiment.py       # (placeholder) Claude API sentiment
│   │   └── scoring.py         # (placeholder) R:R and composite scoring
│   ├── orchestrator.py        # (placeholder) Main agent loop
│   └── utils/
│       ├── __init__.py
│       └── logging.py         # Structured logging setup
├── outputs/                   # Where final reports land
├── tests/
│   └── __init__.py
├── prompts/                   # This prompt kit lives here
├── .env.example               # Template for env vars
├── requirements.txt
├── pyproject.toml
└── README.md
```

### 2. `requirements.txt`
```
yfinance>=0.2.36
pandas>=2.1.0
pandas-ta>=0.3.14b1
numpy>=1.24.0
anthropic>=0.40.0
python-dotenv>=1.0.0
rich>=13.7.0
```

### 3. `src/config.py`
Create a config module using a dataclass. All parameters must be tunable — no magic numbers buried in agent code. Include:

```python
@dataclass
class ScreenerConfig:
    min_avg_volume: int = 1_000_000       # Minimum average daily volume
    min_price: float = 10.0               # Minimum stock price
    max_price: float = 500.0              # Maximum stock price
    min_options_volume: int = 500          # Minimum options open interest (daily avg)
    max_candidates: int = 300             # Cap the screened universe

@dataclass
class TechnicalConfig:
    lookback_days: int = 120              # Historical data window
    bb_squeeze_percentile: float = 20.0   # Bollinger bandwidth percentile for squeeze detection
    atr_contraction_window: int = 14      # ATR period for contraction signals
    volume_dryup_threshold: float = 0.5   # Volume ratio vs 20-day avg to flag dryup
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0
    min_consolidation_days: int = 5       # Minimum days in a consolidation range
    max_consolidation_days: int = 30

@dataclass
class SentimentConfig:
    model: str = "claude-sonnet-4-20250514"
    max_tokens: int = 1024
    min_earnings_gap_days: int = 7        # Skip tickers with earnings within N days
    max_tickers_per_batch: int = 5        # Tickers per Claude API call

@dataclass
class ScoringConfig:
    min_risk_reward: float = 2.0          # Minimum R:R ratio to qualify
    technical_weight: float = 0.5
    sentiment_weight: float = 0.3
    rr_weight: float = 0.2
    min_confidence_score: float = 6.0     # Out of 10
    target_play_count: int = 5            # Desired number of output plays
    max_play_count: int = 10

@dataclass
class AgentConfig:
    screener: ScreenerConfig
    technical: TechnicalConfig
    sentiment: SentimentConfig
    scoring: ScoringConfig
    max_iterations: int = 3               # Max orchestrator loops before accepting best available
    output_dir: str = "outputs"
```

Load `.env` for the Anthropic API key using `python-dotenv`.

### 4. `src/types.py`
Define shared data structures that flow between agents. Use dataclasses with `to_dict()` methods for JSON serialization:

```python
@dataclass
class ScreenedCandidate:
    ticker: str
    price: float
    avg_volume: float
    market_cap: float
    sector: str
    industry: str

@dataclass
class TechnicalSetup:
    ticker: str
    setup_type: str          # "bull_flag", "squeeze_breakout", "mean_reversion", "trend_pullback"
    direction: str           # "long" or "short"
    technical_score: float   # 1-10
    support_level: float
    resistance_level: float
    key_levels: dict         # Flexible dict for setup-specific levels
    indicators: dict         # RSI, BB width, ATR, etc.
    notes: str

@dataclass
class SentimentResult:
    ticker: str
    sentiment: str           # "bullish", "neutral", "bearish"
    confidence: float        # 1-10
    catalysts: list[str]
    risk_flags: list[str]
    earnings_date: str | None
    summary: str

@dataclass
class TradePlay:
    ticker: str
    setup_type: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float | None
    risk_reward_ratio: float
    position_risk_pct: float  # % distance from entry to stop
    technical_score: float
    sentiment_score: float
    composite_score: float
    sentiment_summary: str
    notes: str
```

### 5. `src/utils/logging.py`
Set up structured logging using Python's `logging` module with `rich` for console output. Include a `get_logger(name)` function that returns a logger with consistent formatting. Log levels should be configurable via env var `LOG_LEVEL`.

### 6. `.env.example`
```
ANTHROPIC_API_KEY=your-api-key-here
LOG_LEVEL=INFO
```

### 7. Placeholder agent files
Each file in `src/agents/` should have:
- The class definition with `__init__(self, config)` accepting the relevant config dataclass
- A stubbed `run()` method that logs "Agent not yet implemented" and returns an empty list
- Type hints for input and output matching `src/types.py`

### 8. `pyproject.toml`
Standard Python project config with the package name `swing_screener`, Python 3.11+ requirement.

## Important Notes
- Every numeric threshold should live in `config.py`, never hardcoded in agent logic
- All agent communication happens via the shared types in `types.py`
- The config should be instantiable with all defaults (zero-arg construction) for easy testing
- Use `from __future__ import annotations` in all files for forward reference support
