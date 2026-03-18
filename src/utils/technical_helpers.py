from __future__ import annotations

import numpy as np
import pandas as pd


def find_swing_highs(series: pd.Series, window: int = 5) -> list[tuple[int, float]]:
    """Find local maxima in a price series.

    A swing high is a point where the value is the highest within `window`
    bars on either side.

    Returns list of (integer-location index, price) tuples.
    """
    if len(series) < 2 * window + 1:
        return []

    values = series.values
    results: list[tuple[int, float]] = []

    for i in range(window, len(values) - window):
        left = values[i - window : i]
        right = values[i + 1 : i + window + 1]
        if values[i] >= left.max() and values[i] >= right.max():
            results.append((i, float(values[i])))

    return results


def find_swing_lows(series: pd.Series, window: int = 5) -> list[tuple[int, float]]:
    """Find local minima in a price series.

    A swing low is a point where the value is the lowest within `window`
    bars on either side.

    Returns list of (integer-location index, price) tuples.
    """
    if len(series) < 2 * window + 1:
        return []

    values = series.values
    results: list[tuple[int, float]] = []

    for i in range(window, len(values) - window):
        left = values[i - window : i]
        right = values[i + 1 : i + window + 1]
        if values[i] <= left.min() and values[i] <= right.min():
            results.append((i, float(values[i])))

    return results


def compute_impulse_moves(
    df: pd.DataFrame, min_pct: float = 5.0, window: int = 30
) -> list[tuple[int, int, float]]:
    """Find strong impulse moves (rallies) within the last `window` bars.

    Scans for any stretch of up to 5 consecutive bars where the close-to-close
    gain is >= `min_pct` percent.

    Returns list of (start_iloc, end_iloc, pct_change) tuples, sorted by
    pct_change descending.
    """
    close = df["Close"].values
    n = len(close)
    if n < 2:
        return []

    start_search = max(0, n - window)
    results: list[tuple[int, int, float]] = []

    for start in range(start_search, n - 1):
        for end in range(start + 1, min(start + 6, n)):  # up to 5-bar moves
            pct = (close[end] - close[start]) / close[start] * 100
            if pct >= min_pct:
                results.append((start, end, float(pct)))

    # Remove overlapping moves — keep the strongest
    results.sort(key=lambda x: x[2], reverse=True)
    kept: list[tuple[int, int, float]] = []
    used_ranges: list[tuple[int, int]] = []
    for start, end, pct in results:
        overlap = False
        for us, ue in used_ranges:
            if start <= ue and end >= us:
                overlap = True
                break
        if not overlap:
            kept.append((start, end, pct))
            used_ranges.append((start, end))

    return kept


def detect_divergence(
    price_series: pd.Series, indicator_series: pd.Series, window: int = 20
) -> bool:
    """Detect bullish divergence: price made a lower low but indicator made a higher low.

    Looks at the last `window` bars. Finds the two most recent swing lows in
    both price and indicator and checks for divergence.
    """
    if len(price_series) < window:
        return False

    price_tail = price_series.iloc[-window:]
    ind_tail = indicator_series.iloc[-window:]

    price_lows = find_swing_lows(price_tail, window=3)
    ind_lows = find_swing_lows(ind_tail, window=3)

    if len(price_lows) < 2 or len(ind_lows) < 2:
        return False

    # Last two swing lows
    p_prev_idx, p_prev_val = price_lows[-2]
    p_curr_idx, p_curr_val = price_lows[-1]
    i_prev_idx, i_prev_val = ind_lows[-2]
    i_curr_idx, i_curr_val = ind_lows[-1]

    # Bullish divergence: price lower low, indicator higher low
    return p_curr_val < p_prev_val and i_curr_val > i_prev_val


def consolidation_range(
    df: pd.DataFrame, start_idx: int, end_idx: int
) -> tuple[float, float, float]:
    """Compute the consolidation range over a slice of a DataFrame.

    Args:
        df: OHLCV DataFrame
        start_idx: integer-location start (inclusive)
        end_idx: integer-location end (inclusive)

    Returns:
        (low, high, range_pct) where range_pct = (high - low) / low * 100
    """
    if start_idx < 0 or end_idx >= len(df) or start_idx > end_idx:
        return (0.0, 0.0, 0.0)

    segment = df.iloc[start_idx : end_idx + 1]
    low = float(segment["Low"].min())
    high = float(segment["High"].max())

    if low <= 0:
        return (low, high, 0.0)

    range_pct = (high - low) / low * 100
    return (low, high, float(range_pct))
