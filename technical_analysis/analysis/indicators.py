"""Price and volume calculations, independent of requests and presentation."""
from __future__ import annotations

import numpy as np
import pandas as pd


def wilder_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Seed with the first `period` deltas, then apply Wilder recursion.

    A missing or non-finite close resets the calculation. A flat seed (or a
    later flat state) is neutral at 50; gain-only/loss-only states are 100/0.
    No values are emitted until a fresh run contains period + 1 closes.
    """
    if period < 1:
        raise ValueError("period must be positive")
    values = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(values), np.nan)
    previous = None
    gains: list[float] = []
    losses: list[float] = []
    average_gain = average_loss = None
    for index, value in enumerate(values):
        if not np.isfinite(value):
            previous = None
            gains, losses = [], []
            average_gain = average_loss = None
            continue
        if previous is None:
            previous = value
            continue
        delta = value - previous
        previous = value
        gain, loss = max(delta, 0.0), max(-delta, 0.0)
        if average_gain is None:
            gains.append(gain)
            losses.append(loss)
            if len(gains) < period:
                continue
            average_gain = sum(gains) / period
            average_loss = sum(losses) / period
        else:
            average_gain = (average_gain * (period - 1) + gain) / period
            average_loss = (average_loss * (period - 1) + loss) / period
        if average_gain == 0 and average_loss == 0:
            result[index] = 50.0
        elif average_loss == 0:
            result[index] = 100.0
        elif average_gain == 0:
            result[index] = 0.0
        else:
            result[index] = 100.0 - 100.0 / (1.0 + average_gain / average_loss)
    return pd.Series(result, index=close.index, name=f"rsi{period}")


def daily_indicators(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate over the entire warm-up history, before a display crop."""
    result = frame.copy()
    for period in (20, 50):
        result[f"sma{period}"] = result["close"].rolling(period, min_periods=period).mean()
    result["rsi14"] = wilder_rsi(result["close"])
    return result


def session_vwap(frame: pd.DataFrame) -> pd.DataFrame:
    """Estimate RTH VWAP using HLC3 and 5-minute bar volume.

    A session must start at 09:30 New York and contain every five-minute bar
    through its last observation. If that test fails, withhold the entire
    session's VWAP, session open, and cumulative session volume. This avoids
    presenting a partial-window estimate as a full-session measurement.
    """
    result = frame.copy()
    result["vwap"] = np.nan
    result["session_volume"] = np.nan
    result["session_open"] = np.nan
    result["coverage_complete"] = False
    result["vwap_available"] = False
    if result.empty:
        return result
    incomplete_sessions = set(frame.attrs.get("incomplete_sessions", []))
    for session, group in result.groupby("session", sort=False):
        group = group.sort_values("timestamp")
        stamps = pd.DatetimeIndex(group["timestamp"])
        opening = stamps[0].normalize() + pd.Timedelta("9h30min")
        expected = pd.date_range(opening, stamps[-1], freq="5min")
        covered = stamps.equals(expected) and session not in incomplete_sessions
        result.loc[group.index, "coverage_complete"] = covered
        if not covered:
            continue
        volume = pd.to_numeric(group["volume"], errors="coerce")
        typical_price = (group["high"] + group["low"] + group["close"]) / 3.0
        cumulative_volume = volume.cumsum()
        estimated = (typical_price * volume).cumsum() / cumulative_volume.where(cumulative_volume > 0)
        result.loc[group.index, "vwap"] = estimated
        result.loc[group.index, "session_volume"] = cumulative_volume
        result.loc[group.index, "session_open"] = float(group.iloc[0]["open"])
        result.loc[group.index, "vwap_available"] = estimated.notna()
    return result


def visible_daily(frame: pd.DataFrame, range_label: str) -> pd.DataFrame:
    """Crop already-calculated data relative to its latest actual session."""
    offsets = {"3M": pd.DateOffset(months=3), "6M": pd.DateOffset(months=6), "1Y": pd.DateOffset(years=1)}
    if range_label not in offsets:
        raise ValueError("range_label must be 3M, 6M, or 1Y")
    if frame.empty:
        return frame.copy()
    latest = pd.Timestamp(frame["session"].max())
    cutoff = (latest - offsets[range_label]).date()
    return frame.loc[frame["session"] >= cutoff].copy()
