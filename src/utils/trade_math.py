from __future__ import annotations


def calculate_rr_ratio(entry: float, stop: float, target: float) -> float:
    """Calculate risk-reward ratio for a long trade.

    Returns 0.0 if risk is zero or negative (invalid setup).
    """
    risk = entry - stop
    if risk <= 0:
        return 0.0
    reward = target - entry
    if reward <= 0:
        return 0.0
    return reward / risk


def calculate_position_size(
    account_size: float,
    risk_pct: float,
    entry: float,
    stop: float,
) -> int:
    """Calculate number of shares for a given risk budget.

    Returns 0 if risk per share is zero or negative.
    """
    risk_per_share = entry - stop
    if risk_per_share <= 0:
        return 0
    dollar_risk = account_size * (risk_pct / 100.0)
    return int(dollar_risk / risk_per_share)


def atr_buffer(atr_value: float, multiplier: float) -> float:
    """Return ATR * multiplier as a price buffer."""
    return atr_value * multiplier


def normalize_score(
    value: float,
    min_val: float,
    max_val: float,
    target_min: float = 1.0,
    target_max: float = 10.0,
) -> float:
    """Linearly map *value* from [min_val, max_val] to [target_min, target_max].

    Clamps the result to [target_min, target_max].
    """
    if max_val == min_val:
        return (target_min + target_max) / 2.0
    ratio = (value - min_val) / (max_val - min_val)
    ratio = max(0.0, min(1.0, ratio))
    return target_min + ratio * (target_max - target_min)
