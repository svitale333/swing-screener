"""Technical indicator computations using pandas/numpy.

Drop-in replacements for pandas-ta functions. All functions operate on
pandas Series and return pandas Series (or DataFrames for multi-column
indicators like Bollinger Bands, MACD, ADX).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def sma(series: pd.Series, length: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=length, min_periods=length).mean()


def ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=length, adjust=False, min_periods=length).mean()


def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    result = 100 - (100 / (1 + rs))
    return result


def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()


def bbands(close: pd.Series, length: int = 20, std: float = 2.0) -> pd.DataFrame:
    """Bollinger Bands.

    Returns DataFrame with columns:
    - BBL_{length}_{std}: Lower Band
    - BBM_{length}_{std}: Middle Band (SMA)
    - BBU_{length}_{std}: Upper Band
    - BBB_{length}_{std}: Bandwidth (%)
    - BBP_{length}_{std}: Percent B
    """
    mid = close.rolling(window=length, min_periods=length).mean()
    std_dev = close.rolling(window=length, min_periods=length).std()

    upper = mid + std * std_dev
    lower = mid - std * std_dev
    bandwidth = ((upper - lower) / mid) * 100
    pct_b = (close - lower) / (upper - lower)

    # Use float-style suffix to match pandas-ta naming (e.g. "20_2.0")
    suffix = f"{length}_{float(std)}"
    return pd.DataFrame({
        f"BBL_{suffix}": lower,
        f"BBM_{suffix}": mid,
        f"BBU_{suffix}": upper,
        f"BBB_{suffix}": bandwidth,
        f"BBP_{suffix}": pct_b,
    }, index=close.index)


def macd(
    close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """MACD (Moving Average Convergence Divergence).

    Returns DataFrame with columns:
    - MACD_{fast}_{slow}_{signal}: MACD line
    - MACDs_{fast}_{slow}_{signal}: Signal line
    - MACDh_{fast}_{slow}_{signal}: Histogram
    """
    fast_ema = ema(close, fast)
    slow_ema = ema(close, slow)
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line

    suffix = f"{fast}_{slow}_{signal}"
    return pd.DataFrame({
        f"MACD_{suffix}": macd_line,
        f"MACDs_{suffix}": signal_line,
        f"MACDh_{suffix}": histogram,
    }, index=close.index)


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14
) -> pd.DataFrame:
    """Average Directional Index.

    Returns DataFrame with columns:
    - ADX_{length}: ADX
    - DMP_{length}: +DI (Directional Indicator Plus)
    - DMN_{length}: -DI (Directional Indicator Minus)
    """
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    # True Range
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - prev_high
    down_move = prev_low - low

    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=high.index)

    # Smoothed TR and DM
    atr_smooth = tr.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()

    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr_smooth.replace(0, np.nan)
    minus_di = 100 * minus_dm_smooth / atr_smooth.replace(0, np.nan)

    # ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(alpha=1 / length, min_periods=length, adjust=False).mean()

    return pd.DataFrame({
        f"ADX_{length}": adx_val,
        f"DMP_{length}": plus_di,
        f"DMN_{length}": minus_di,
    }, index=high.index)
