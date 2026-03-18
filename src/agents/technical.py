from __future__ import annotations

import numpy as np
import pandas as pd
from src.utils import indicators as ta
import yfinance as yf

from src.config import TechnicalConfig
from src.types import ScreenedCandidate, TechnicalSetup
from src.utils.logging import get_logger
from src.utils.technical_helpers import (
    compute_impulse_moves,
    consolidation_range,
    detect_divergence,
    find_swing_highs,
    find_swing_lows,
)

logger = get_logger(__name__)


class TechnicalAgent:
    """Detects technical setups on screened candidates using pandas-ta indicators."""

    def __init__(self, config: TechnicalConfig) -> None:
        self.config = config

    def run(self, candidates: list[ScreenedCandidate]) -> list[TechnicalSetup]:
        """Analyze candidates for technical setups.

        1. Download OHLCV data in batch
        2. Compute indicators per ticker
        3. Run all four setup detectors per ticker
        4. Return all setups sorted by technical_score descending
        """
        if not candidates:
            logger.info("No candidates to analyze")
            return []

        tickers = [c.ticker for c in candidates]
        logger.info(f"Analyzing {len(tickers)} candidates for technical setups")

        # Step 1: Download OHLCV data
        ohlcv = self._download_data(tickers)
        if ohlcv is None or ohlcv.empty:
            logger.warning("No OHLCV data retrieved — aborting technical analysis")
            return []

        # Step 2-3: Compute indicators and detect setups per ticker
        all_setups: list[TechnicalSetup] = []
        for ticker in tickers:
            try:
                df = self._extract_ticker_df(ohlcv, ticker, len(tickers))
                if df is None or len(df) < 50:
                    logger.debug(f"{ticker}: insufficient data ({0 if df is None else len(df)} bars)")
                    continue

                df = self._compute_indicators(df)
                setups = self._detect_all_setups(ticker, df)
                all_setups.extend(setups)
            except Exception:
                logger.exception(f"Error processing {ticker}")

        # Step 5: Sort by score descending and log summary
        all_setups.sort(key=lambda s: s.technical_score, reverse=True)
        self._log_summary(all_setups)

        return all_setups

    # ------------------------------------------------------------------
    # Data retrieval
    # ------------------------------------------------------------------

    def _download_data(self, tickers: list[str]) -> pd.DataFrame | None:
        """Batch download OHLCV data for all tickers."""
        period = f"{self.config.lookback_days}d"
        logger.info(f"Downloading {period} of data for {len(tickers)} tickers")
        try:
            df = yf.download(
                tickers,
                period=period,
                group_by="ticker",
                threads=True,
                progress=False,
            )
            return df
        except Exception:
            logger.exception("Batch OHLCV download failed")
            return None

    def _extract_ticker_df(
        self, ohlcv: pd.DataFrame, ticker: str, total_tickers: int
    ) -> pd.DataFrame | None:
        """Extract a single ticker's OHLCV DataFrame from the batch download."""
        try:
            if total_tickers == 1:
                # Single-ticker download doesn't have MultiIndex columns
                df = ohlcv.copy()
            else:
                if ticker not in ohlcv.columns.get_level_values(0):
                    return None
                df = ohlcv[ticker].copy()

            df = df.dropna(subset=["Close"])
            if df.empty:
                return None

            # Ensure standard column names
            for col in ["Open", "High", "Low", "Close", "Volume"]:
                if col not in df.columns:
                    return None

            return df
        except Exception:
            logger.debug(f"Failed to extract data for {ticker}")
            return None

    # ------------------------------------------------------------------
    # Indicator computation
    # ------------------------------------------------------------------

    def _compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute all technical indicators and attach them to the DataFrame."""
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        # Bollinger Bands (20, 2)
        bbands = ta.bbands(close, length=20, std=2)
        if bbands is not None:
            df = df.join(bbands)

        # RSI (14)
        rsi = ta.rsi(close, length=14)
        if rsi is not None:
            df["RSI_14"] = rsi

        # ATR (14)
        atr = ta.atr(high, low, close, length=14)
        if atr is not None:
            df["ATR_14"] = atr
            df["ATR_norm"] = df["ATR_14"] / close  # normalized ATR

        # Moving Averages
        df["EMA_9"] = ta.ema(close, length=9)
        df["EMA_21"] = ta.ema(close, length=21)
        df["SMA_50"] = ta.sma(close, length=50)
        df["SMA_200"] = ta.sma(close, length=200)

        # Volume SMA (20)
        df["Vol_SMA_20"] = ta.sma(volume, length=20)

        # MACD (12, 26, 9)
        macd = ta.macd(close, fast=12, slow=26, signal=9)
        if macd is not None:
            df = df.join(macd)

        # ADX (14)
        adx = ta.adx(high, low, close, length=14)
        if adx is not None:
            df = df.join(adx)

        # ATR SMA for contraction detection
        if "ATR_14" in df.columns:
            df["ATR_SMA"] = ta.sma(df["ATR_14"], length=self.config.atr_contraction_window)

        return df

    # ------------------------------------------------------------------
    # Setup detection dispatch
    # ------------------------------------------------------------------

    def _detect_all_setups(self, ticker: str, df: pd.DataFrame) -> list[TechnicalSetup]:
        """Run all four setup detectors and return any matches."""
        setups: list[TechnicalSetup] = []

        detectors = [
            self._detect_squeeze_breakout,
            self._detect_bull_flag,
            self._detect_mean_reversion,
            self._detect_trend_pullback,
        ]

        for detector in detectors:
            try:
                result = detector(ticker, df)
                if result is not None:
                    setups.append(result)
            except Exception:
                logger.debug(f"{ticker}: error in {detector.__name__}")

        return setups

    # ------------------------------------------------------------------
    # Setup 1: Squeeze / Breakout
    # ------------------------------------------------------------------

    def _detect_squeeze_breakout(self, ticker: str, df: pd.DataFrame) -> TechnicalSetup | None:
        """Detect a volatility squeeze / breakout setup.

        Conditions:
        - BB Bandwidth in bottom bb_squeeze_percentile of its own lookback
        - ATR contracted below its SMA
        - Volume in last 5 sessions below 20-day avg by volume_dryup_threshold
        - Price above 50 SMA (bullish bias)
        """
        if "BBB_20_2.0" not in df.columns or "ATR_14" not in df.columns:
            return None
        if "SMA_50" not in df.columns or "ATR_SMA" not in df.columns:
            return None
        if "Vol_SMA_20" not in df.columns:
            return None

        last = df.iloc[-1]
        close = float(last["Close"])

        # BB Bandwidth squeeze check
        bbb = df["BBB_20_2.0"].dropna()
        if len(bbb) < 20:
            return None
        current_bbb = float(bbb.iloc[-1])
        threshold = float(np.percentile(bbb.values, self.config.bb_squeeze_percentile))
        if current_bbb > threshold:
            return None

        # ATR contraction
        if pd.isna(last.get("ATR_SMA")) or float(last["ATR_14"]) >= float(last["ATR_SMA"]):
            return None

        # Volume dry-up: average of last 5 sessions vs 20-day SMA
        vol_recent = df["Volume"].iloc[-5:].mean()
        vol_sma = float(last["Vol_SMA_20"])
        if pd.isna(vol_sma) or vol_sma <= 0:
            return None
        if vol_recent / vol_sma > self.config.volume_dryup_threshold:
            return None

        # Price above 50 SMA
        sma50 = last.get("SMA_50")
        if pd.isna(sma50) or close <= float(sma50):
            return None

        # Support & resistance
        swing_lows = find_swing_lows(df["Low"])
        swing_highs = find_swing_highs(df["High"])

        lower_bb = float(last.get("BBL_20_2.0", close * 0.95))
        upper_bb = float(last.get("BBU_20_2.0", close * 1.05))

        support = float(swing_lows[-1][1]) if swing_lows else lower_bb
        support = min(support, lower_bb)
        resistance = float(swing_highs[-1][1]) if swing_highs else upper_bb
        resistance = max(resistance, upper_bb)

        # Scoring (weights: squeeze_tightness=0.4, trend_alignment=0.3, adx_low=0.3)
        squeeze_tightness = 1.0 - (current_bbb - bbb.min()) / (bbb.max() - bbb.min() + 1e-9)
        squeeze_tightness = float(np.clip(squeeze_tightness, 0, 1))

        sma200 = last.get("SMA_200")
        trend_alignment = 1.0 if (not pd.isna(sma200) and close > float(sma200)) else 0.0

        adx_val = last.get("ADX_14")
        adx_low = 1.0 - min(float(adx_val), 40) / 40 if not pd.isna(adx_val) else 0.5

        raw_score = squeeze_tightness * 0.4 + trend_alignment * 0.3 + adx_low * 0.3
        technical_score = float(np.clip(raw_score * 9 + 1, 1, 10))

        return TechnicalSetup(
            ticker=ticker,
            setup_type="squeeze_breakout",
            direction="long",
            technical_score=round(technical_score, 2),
            support_level=round(support, 2),
            resistance_level=round(resistance, 2),
            key_levels={"lower_bb": round(lower_bb, 2), "upper_bb": round(upper_bb, 2), "sma_50": round(float(sma50), 2)},
            indicators={"bbb": round(current_bbb, 4), "atr": round(float(last["ATR_14"]), 4), "rsi": round(float(last.get("RSI_14", 0)), 2)},
            notes="Volatility squeeze with contracting ATR and volume dry-up",
        )

    # ------------------------------------------------------------------
    # Setup 2: Bull Flag / Flat Base
    # ------------------------------------------------------------------

    def _detect_bull_flag(self, ticker: str, df: pd.DataFrame) -> TechnicalSetup | None:
        """Detect a bull flag / flat base continuation pattern.

        Conditions:
        - Impulse move: >= 5% gain within 5 bars, somewhere in last 30 days
        - Consolidation after impulse: range <= 50% of impulse range
        - Volume declining in consolidation vs impulse
        - Price holding above 21 EMA during consolidation
        """
        if "EMA_21" not in df.columns or "Vol_SMA_20" not in df.columns:
            return None

        impulse_moves = compute_impulse_moves(df, min_pct=5.0, window=30)
        if not impulse_moves:
            return None

        n = len(df)
        best_setup = None
        best_score = 0.0

        for imp_start, imp_end, imp_pct in impulse_moves:
            # Consolidation period: from impulse end to current bar
            consol_start = imp_end + 1
            consol_end = n - 1
            consol_bars = consol_end - consol_start + 1

            if consol_bars < self.config.min_consolidation_days:
                continue
            if consol_bars > self.config.max_consolidation_days:
                continue

            # Consolidation range check
            consol_low, consol_high, consol_range_pct = consolidation_range(df, consol_start, consol_end)
            imp_low, imp_high, imp_range_pct = consolidation_range(df, imp_start, imp_end)

            if imp_range_pct <= 0:
                continue

            imp_range_abs = imp_high - imp_low
            consol_range_abs = consol_high - consol_low

            if imp_range_abs <= 0 or consol_range_abs / imp_range_abs > 0.5:
                continue

            # Volume decline: consolidation avg < 70% of impulse avg
            imp_vol = df["Volume"].iloc[imp_start : imp_end + 1].mean()
            consol_vol = df["Volume"].iloc[consol_start : consol_end + 1].mean()
            if imp_vol <= 0 or consol_vol / imp_vol > 0.7:
                continue

            # Price holding above 21 EMA during consolidation
            consol_closes = df["Close"].iloc[consol_start : consol_end + 1]
            ema21_vals = df["EMA_21"].iloc[consol_start : consol_end + 1].dropna()
            if ema21_vals.empty:
                continue
            if (consol_closes < ema21_vals).any():
                continue

            # Scoring (weights: flag_tightness=0.4, volume_decline=0.3, sma50_dist=0.3)
            flag_tightness = 1.0 - (consol_range_abs / imp_range_abs) / 0.5
            flag_tightness = float(np.clip(flag_tightness, 0, 1))

            vol_decline = 1.0 - (consol_vol / imp_vol) / 0.7
            vol_decline = float(np.clip(vol_decline, 0, 1))

            last_close = float(df["Close"].iloc[-1])
            sma50 = df.get("SMA_50")
            if sma50 is not None and not pd.isna(sma50.iloc[-1]):
                sma50_val = float(sma50.iloc[-1])
                sma50_dist = (last_close - sma50_val) / sma50_val if sma50_val > 0 else 0
                sma50_dist = float(np.clip(sma50_dist * 10, 0, 1))  # 10% above = 1.0
            else:
                sma50_dist = 0.5

            raw_score = flag_tightness * 0.4 + vol_decline * 0.3 + sma50_dist * 0.3
            score = float(np.clip(raw_score * 9 + 1, 1, 10))

            if score > best_score:
                best_score = score
                best_setup = TechnicalSetup(
                    ticker=ticker,
                    setup_type="bull_flag",
                    direction="long",
                    technical_score=round(score, 2),
                    support_level=round(consol_low, 2),
                    resistance_level=round(consol_high, 2),
                    key_levels={"ema_21": round(float(ema21_vals.iloc[-1]), 2), "impulse_high": round(imp_high, 2)},
                    indicators={"impulse_pct": round(imp_pct, 2), "consol_range_pct": round(consol_range_pct, 2)},
                    notes=f"Bull flag: {imp_pct:.1f}% impulse, {consol_bars}-bar consolidation",
                )

        return best_setup

    # ------------------------------------------------------------------
    # Setup 3: Mean Reversion / Oversold Bounce
    # ------------------------------------------------------------------

    def _detect_mean_reversion(self, ticker: str, df: pd.DataFrame) -> TechnicalSetup | None:
        """Detect a mean reversion / oversold bounce setup.

        Conditions:
        - RSI <= rsi_oversold at some point in last 5 sessions
        - Price near support (within 2% of SMA 50/200, or at a swing low)
        - Bullish RSI divergence (optional, boosts score)
        - Volume spike on most recent session (> 1.5x 20-day avg)
        """
        if "RSI_14" not in df.columns or "Vol_SMA_20" not in df.columns:
            return None

        last = df.iloc[-1]
        close = float(last["Close"])

        # RSI oversold in last 5 sessions
        rsi_recent = df["RSI_14"].iloc[-5:].dropna()
        if rsi_recent.empty or float(rsi_recent.min()) > self.config.rsi_oversold:
            return None

        # Price near support: within 2% of SMA 50, SMA 200, or recent swing low
        near_support = False
        support_level = close * 0.98  # default fallback

        sma50 = last.get("SMA_50")
        sma200 = last.get("SMA_200")

        if not pd.isna(sma50) and abs(close - float(sma50)) / float(sma50) <= 0.02:
            near_support = True
            support_level = float(sma50)

        if not pd.isna(sma200) and abs(close - float(sma200)) / float(sma200) <= 0.02:
            near_support = True
            support_level = float(sma200)

        swing_lows = find_swing_lows(df["Low"])
        if swing_lows:
            nearest_low = min(swing_lows, key=lambda x: abs(x[1] - close))
            if abs(nearest_low[1] - close) / close <= 0.02:
                near_support = True
                support_level = nearest_low[1]

        if not near_support:
            return None

        # Volume spike on most recent session
        vol_sma = float(last.get("Vol_SMA_20", 0))
        current_vol = float(last["Volume"])
        if vol_sma <= 0 or current_vol / vol_sma < 1.5:
            return None

        # Check for RSI divergence (boosts score, not required)
        has_divergence = detect_divergence(df["Close"], df["RSI_14"], window=20)

        # Resistance: EMA 21 first, then prior swing high
        ema21 = last.get("EMA_21")
        resistance = float(ema21) if not pd.isna(ema21) else close * 1.05
        swing_highs = find_swing_highs(df["High"])
        if swing_highs:
            resistance = max(resistance, float(swing_highs[-1][1]))

        # Scoring (weights: divergence=0.25, proximity=0.25, volume_spike=0.25, macd_turn=0.25)
        divergence_score = 1.0 if has_divergence else 0.0

        # Proximity to support (closer = better)
        proximity = 1.0 - abs(close - support_level) / (close * 0.02 + 1e-9)
        proximity = float(np.clip(proximity, 0, 1))

        # Volume spike magnitude
        vol_spike = min((current_vol / vol_sma - 1.5) / 1.5, 1.0)  # 3x vol = 1.0
        vol_spike = float(np.clip(vol_spike, 0, 1))

        # MACD histogram turning positive
        macd_hist_col = "MACDh_12_26_9"
        macd_turn = 0.0
        if macd_hist_col in df.columns:
            hist = df[macd_hist_col].dropna()
            if len(hist) >= 2:
                if float(hist.iloc[-1]) > float(hist.iloc[-2]):
                    macd_turn = 1.0

        raw_score = divergence_score * 0.25 + proximity * 0.25 + vol_spike * 0.25 + macd_turn * 0.25
        technical_score = float(np.clip(raw_score * 9 + 1, 1, 10))

        return TechnicalSetup(
            ticker=ticker,
            setup_type="mean_reversion",
            direction="long",
            technical_score=round(technical_score, 2),
            support_level=round(support_level, 2),
            resistance_level=round(resistance, 2),
            key_levels={"sma_50": round(float(sma50), 2) if not pd.isna(sma50) else None},
            indicators={"rsi": round(float(rsi_recent.iloc[-1]), 2), "volume_ratio": round(current_vol / vol_sma, 2)},
            notes=f"Oversold bounce near support{' with RSI divergence' if has_divergence else ''}",
        )

    # ------------------------------------------------------------------
    # Setup 4: Trend Pullback / Continuation
    # ------------------------------------------------------------------

    def _detect_trend_pullback(self, ticker: str, df: pd.DataFrame) -> TechnicalSetup | None:
        """Detect a trend pullback / continuation setup.

        Conditions:
        - Strong uptrend: price above rising 50 SMA, 50 SMA above 200 SMA, ADX > 25
        - Pullback to 21 EMA (within 1.5%)
        - Orderly pullback: declining volume on pullback candles
        - RSI between 40-55 (reset zone)
        """
        required = ["SMA_50", "SMA_200", "EMA_21", "RSI_14", "ADX_14"]
        if any(col not in df.columns for col in required):
            return None

        last = df.iloc[-1]
        close = float(last["Close"])

        # Strong uptrend checks
        sma50 = float(last["SMA_50"])
        sma200 = float(last["SMA_200"])
        ema21 = float(last["EMA_21"])

        if pd.isna(sma50) or pd.isna(sma200) or pd.isna(ema21):
            return None

        # Price above 50 SMA
        if close <= sma50:
            return None

        # 50 SMA above 200 SMA
        if sma50 <= sma200:
            return None

        # 50 SMA rising: current > value 10 bars ago
        sma50_series = df["SMA_50"].dropna()
        if len(sma50_series) < 10:
            return None
        if float(sma50_series.iloc[-1]) <= float(sma50_series.iloc[-10]):
            return None

        # ADX > 25
        adx_val = float(last["ADX_14"])
        if pd.isna(adx_val) or adx_val <= 25:
            return None

        # Pullback to 21 EMA (within 1.5%)
        if abs(close - ema21) / ema21 > 0.015:
            return None

        # RSI in reset zone: 40-55
        rsi = float(last["RSI_14"])
        if pd.isna(rsi) or rsi < 40 or rsi > 55:
            return None

        # Orderly pullback: check that volume is declining over last 3+ bars
        if len(df) >= 4:
            recent_vol = df["Volume"].iloc[-3:].values
            if not (recent_vol[0] >= recent_vol[1] >= recent_vol[2]):
                # At least a general declining trend
                pass  # Not a hard filter, just affects score

        # Support & resistance
        support = ema21
        swing_highs = find_swing_highs(df["High"])
        resistance = float(swing_highs[-1][1]) if swing_highs else close * 1.05

        # Scoring (weights: trend_strength=0.4, ema_touch_quality=0.3, orderly_pullback=0.3)
        # Trend strength: ADX value, normalize 25-50 range to 0-1
        trend_strength = (adx_val - 25) / 25
        trend_strength = float(np.clip(trend_strength, 0, 1))

        # EMA touch quality: how close price is to 21 EMA (closer = better)
        ema_touch = 1.0 - abs(close - ema21) / (ema21 * 0.015)
        ema_touch = float(np.clip(ema_touch, 0, 1))

        # Orderly pullback: declining volume in last 3 bars
        orderly = 0.5  # default
        if len(df) >= 4:
            recent_vol = df["Volume"].iloc[-3:].values
            if recent_vol[0] >= recent_vol[1] >= recent_vol[2]:
                orderly = 1.0
            elif recent_vol[1] >= recent_vol[2]:
                orderly = 0.7

        raw_score = trend_strength * 0.4 + ema_touch * 0.3 + orderly * 0.3
        technical_score = float(np.clip(raw_score * 9 + 1, 1, 10))

        return TechnicalSetup(
            ticker=ticker,
            setup_type="trend_pullback",
            direction="long",
            technical_score=round(technical_score, 2),
            support_level=round(support, 2),
            resistance_level=round(resistance, 2),
            key_levels={"ema_21": round(ema21, 2), "sma_50": round(sma50, 2), "sma_200": round(sma200, 2)},
            indicators={"adx": round(adx_val, 2), "rsi": round(rsi, 2)},
            notes=f"Trend pullback to 21 EMA, ADX={adx_val:.0f}",
        )

    # ------------------------------------------------------------------
    # Logging summary
    # ------------------------------------------------------------------

    def _log_summary(self, setups: list[TechnicalSetup]) -> None:
        """Log summary of detected setups."""
        if not setups:
            logger.info("No technical setups detected")
            return

        # Breakdown by setup type
        type_counts: dict[str, int] = {}
        for s in setups:
            type_counts[s.setup_type] = type_counts.get(s.setup_type, 0) + 1

        logger.info(f"Total setups found: {len(setups)}")
        for setup_type, count in sorted(type_counts.items()):
            logger.info(f"  {setup_type}: {count}")

        # Top 10 by score
        top = setups[:10]
        logger.info("Top 10 setups by score:")
        for s in top:
            logger.info(f"  {s.ticker} ({s.setup_type}): {s.technical_score}")
