from __future__ import annotations

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

from src.config import ScreenerConfig
from src.types import ScreenedCandidate
from src.utils.logging import get_logger

logger = get_logger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
SP500_CACHE_FILE = DATA_DIR / "sp500_tickers.csv"
SCREENER_CACHE_FILE = DATA_DIR / "screener_cache.json"
SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


class ScreenerAgent:
    """Filters the market universe down to swing trade candidates."""

    def __init__(self, config: ScreenerConfig) -> None:
        self.config = config

    def run(self, force_refresh: bool = False) -> list[ScreenedCandidate]:
        """Screen the market universe and return qualifying candidates."""
        if not force_refresh:
            cached = self._load_screener_cache()
            if cached is not None:
                logger.info(f"Returning {len(cached)} candidates from cache")
                return cached

        logger.info("Starting screener run")

        # Step 1: Build universe
        tickers = self._build_universe()
        logger.info(f"Universe size: {len(tickers)} tickers")

        # Step 2: Remove blacklisted tickers
        blacklist = set(self.config.blacklist)
        tickers = [t for t in tickers if t not in blacklist]
        logger.info(f"After blacklist removal: {len(tickers)} tickers")

        # Step 3: Batch download price data for volume/price filtering
        price_data = self._batch_download_prices(tickers)

        # Step 4: Apply price/volume filters from batch data
        passed_batch = self._apply_batch_filters(tickers, price_data)
        logger.info(f"After price/volume filters: {len(passed_batch)} tickers")

        # Step 5: Enrich with .info lookups (market cap, sector, options) in parallel
        candidates = self._enrich_candidates(passed_batch, price_data)
        logger.info(f"After enrichment filters: {len(candidates)} candidates")

        # Step 6: Sort by avg volume descending, cap at max_candidates
        candidates.sort(key=lambda c: c.avg_volume, reverse=True)
        candidates = candidates[: self.config.max_candidates]

        # Log summary
        self._log_summary(candidates)

        # Cache results
        self._save_screener_cache(candidates)

        return candidates

    # ------------------------------------------------------------------
    # Universe construction
    # ------------------------------------------------------------------

    def _build_universe(self) -> list[str]:
        """Combine S&P 500 tickers with supplemental list, deduplicated."""
        sp500 = self._get_sp500_tickers()
        from data.supplemental_tickers import SUPPLEMENTAL_TICKERS

        combined = list(dict.fromkeys(sp500 + SUPPLEMENTAL_TICKERS))
        return combined

    def _get_sp500_tickers(self) -> list[str]:
        """Fetch S&P 500 tickers, using a cached CSV if fresh enough."""
        if SP500_CACHE_FILE.exists():
            mod_time = datetime.fromtimestamp(SP500_CACHE_FILE.stat().st_mtime)
            ttl = timedelta(hours=self.config.ticker_cache_ttl_hours)
            if datetime.now() - mod_time < ttl:
                df = pd.read_csv(SP500_CACHE_FILE)
                tickers = df["Symbol"].tolist()
                logger.info(f"Loaded {len(tickers)} S&P 500 tickers from cache")
                return tickers

        logger.info("Fetching S&P 500 tickers from Wikipedia")
        try:
            tables = pd.read_html(SP500_URL)
            df = tables[0]
            # Wikipedia uses '.' in tickers but yfinance uses '-' (e.g. BRK.B -> BRK-B)
            df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
            tickers = df["Symbol"].tolist()

            DATA_DIR.mkdir(parents=True, exist_ok=True)
            df[["Symbol"]].to_csv(SP500_CACHE_FILE, index=False)
            logger.info(f"Cached {len(tickers)} S&P 500 tickers")
            return tickers
        except Exception:
            logger.exception("Failed to fetch S&P 500 list from Wikipedia")
            if SP500_CACHE_FILE.exists():
                df = pd.read_csv(SP500_CACHE_FILE)
                logger.warning("Falling back to stale S&P 500 cache")
                return df["Symbol"].tolist()
            return []

    # ------------------------------------------------------------------
    # Batch price download & filtering
    # ------------------------------------------------------------------

    def _batch_download_prices(self, tickers: list[str]) -> pd.DataFrame:
        """Download recent price data for all tickers in one batch call."""
        logger.info("Batch downloading price data")
        try:
            df = yf.download(
                tickers,
                period="5d",
                group_by="ticker",
                threads=True,
                progress=False,
            )
            return df
        except Exception:
            logger.exception("Batch download failed")
            return pd.DataFrame()

    def _apply_batch_filters(
        self, tickers: list[str], price_data: pd.DataFrame
    ) -> list[dict]:
        """Filter tickers by price and volume using batch-downloaded data.

        Returns a list of dicts with ticker, price, avg_volume for tickers
        that pass the price/volume thresholds.
        """
        passed: list[dict] = []

        for ticker in tickers:
            try:
                if price_data.empty:
                    continue

                # yf.download with a list always returns MultiIndex columns
                if ticker not in price_data.columns.get_level_values(0):
                    continue
                ticker_data = price_data[ticker]

                if ticker_data.empty or ticker_data["Close"].dropna().empty:
                    continue

                close = ticker_data["Close"].dropna().iloc[-1]
                avg_vol = ticker_data["Volume"].dropna().mean()

                if close < self.config.min_price or close > self.config.max_price:
                    continue
                if avg_vol < self.config.min_avg_volume:
                    continue

                passed.append(
                    {"ticker": ticker, "price": float(close), "avg_volume": float(avg_vol)}
                )
            except Exception:
                logger.debug(f"Skipping {ticker} during batch filter: data issue")

        return passed

    # ------------------------------------------------------------------
    # Enrichment (parallel .info lookups)
    # ------------------------------------------------------------------

    def _enrich_candidates(
        self, passed: list[dict], price_data: pd.DataFrame
    ) -> list[ScreenedCandidate]:
        """Enrich passing tickers with market cap, sector, options via parallel .info lookups."""
        candidates: list[ScreenedCandidate] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
        ) as progress:
            task = progress.add_task("Enriching candidates", total=len(passed))

            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = {
                    executor.submit(self._enrich_single, item): item
                    for item in passed
                }

                for future in as_completed(futures):
                    progress.advance(task)
                    result = future.result()
                    if result is not None:
                        candidates.append(result)

        return candidates

    def _enrich_single(self, item: dict) -> ScreenedCandidate | None:
        """Look up .info for a single ticker and return a ScreenedCandidate or None."""
        ticker_str = item["ticker"]
        try:
            ticker_obj = yf.Ticker(ticker_str)
            info = ticker_obj.info or {}

            # Market cap filter
            market_cap = info.get("marketCap") or 0
            if market_cap < self.config.min_market_cap:
                return None

            # Options availability
            try:
                options = ticker_obj.options
                if not options:
                    return None
            except Exception:
                return None

            sector = info.get("sector", "Unknown")
            industry = info.get("industry", "Unknown")

            return ScreenedCandidate(
                ticker=ticker_str,
                price=item["price"],
                avg_volume=item["avg_volume"],
                market_cap=float(market_cap),
                sector=sector,
                industry=industry,
            )
        except Exception:
            logger.debug(f"Skipping {ticker_str}: .info lookup failed")
            return None

    # ------------------------------------------------------------------
    # Caching
    # ------------------------------------------------------------------

    def _load_screener_cache(self) -> list[ScreenedCandidate] | None:
        """Load cached screener results if within TTL."""
        if not SCREENER_CACHE_FILE.exists():
            return None

        try:
            with open(SCREENER_CACHE_FILE, "r") as f:
                data = json.load(f)

            cached_at = datetime.fromisoformat(data["cached_at"])
            ttl = timedelta(hours=self.config.cache_ttl_hours)
            if datetime.now() - cached_at > ttl:
                logger.info("Screener cache expired")
                return None

            candidates = [ScreenedCandidate(**c) for c in data["candidates"]]
            return candidates
        except Exception:
            logger.warning("Failed to load screener cache")
            return None

    def _save_screener_cache(self, candidates: list[ScreenedCandidate]) -> None:
        """Save screener results to cache."""
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            data = {
                "cached_at": datetime.now().isoformat(),
                "candidates": [c.to_dict() for c in candidates],
            }
            with open(SCREENER_CACHE_FILE, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Cached {len(candidates)} candidates")
        except Exception:
            logger.warning("Failed to save screener cache")

    # ------------------------------------------------------------------
    # Logging summary
    # ------------------------------------------------------------------

    def _log_summary(self, candidates: list[ScreenedCandidate]) -> None:
        """Log summary stats about the screening results."""
        if not candidates:
            logger.info("No candidates passed screening")
            return

        sectors: dict[str, int] = {}
        for c in candidates:
            sectors[c.sector] = sectors.get(c.sector, 0) + 1

        logger.info(f"Final candidates: {len(candidates)}")
        logger.info(f"Sectors represented: {len(sectors)}")
        for sector, count in sorted(sectors.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {sector}: {count}")
