"""Price and volume calculations, independent of requests and presentation."""
from __future__ import annotations

import numpy as np
import pandas as pd

# Prior highs act as overhead resistance; the 60-session one also defines a breakout.
RESISTANCE_LOOKBACKS = (20, 60, 126, 252)
BREAKOUT_LOOKBACK = 60


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


def wilder_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Wilder ATR that restarts after any missing or non-finite bar.

    The first bar of a run uses high - low; the first `period` true ranges
    seed an arithmetic mean, then `(previous * (period - 1) + TR) / period`.
    """
    highs = pd.to_numeric(high, errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(low, errors="coerce").to_numpy(dtype=float)
    closes = pd.to_numeric(close, errors="coerce").to_numpy(dtype=float)
    result = np.full(len(closes), np.nan)
    previous_close = None
    seed: list[float] = []
    average = None
    for index in range(len(closes)):
        h, l, c = highs[index], lows[index], closes[index]
        if not (np.isfinite(h) and np.isfinite(l) and np.isfinite(c)):
            previous_close, seed, average = None, [], None
            continue
        true_range = h - l if previous_close is None else max(h - l, abs(h - previous_close), abs(l - previous_close))
        previous_close = c
        if average is None:
            seed.append(true_range)
            if len(seed) < period:
                continue
            average = sum(seed) / period
        else:
            average = (average * (period - 1) + true_range) / period
        result[index] = average
    return pd.Series(result, index=close.index, name=f"atr{period}")


def _fibonacci_leg(high: pd.Series, low: pd.Series, lookback: int) -> pd.DataFrame:
    """Highest high and lowest low of each trailing window, and which came last.

    `fib_up` is True when the high is more recent than the low, so the leg is
    an advance and retracements are measured down from the high.
    """
    highs = pd.to_numeric(high, errors="coerce").to_numpy(dtype=float)
    lows = pd.to_numeric(low, errors="coerce").to_numpy(dtype=float)
    size = len(highs)
    result = pd.DataFrame({"fib_high": np.nan, "fib_low": np.nan, "fib_up": np.nan}, index=high.index)
    if size < lookback:
        return result
    high_windows = np.lib.stride_tricks.sliding_window_view(highs, lookback)
    low_windows = np.lib.stride_tricks.sliding_window_view(lows, lookback)
    complete = np.isfinite(high_windows).all(axis=1) & np.isfinite(low_windows).all(axis=1)
    # Ties prefer the latest bar, so a fresh high is the current leg end.
    high_position = lookback - 1 - np.argmax(high_windows[:, ::-1], axis=1)
    low_position = lookback - 1 - np.argmin(low_windows[:, ::-1], axis=1)
    rows = np.arange(lookback - 1, size)
    fib_high = np.where(complete, high_windows.max(axis=1, initial=-np.inf, where=np.isfinite(high_windows)), np.nan)
    fib_low = np.where(complete, low_windows.min(axis=1, initial=np.inf, where=np.isfinite(low_windows)), np.nan)
    result.iloc[rows, 0] = fib_high
    result.iloc[rows, 1] = fib_low
    result.iloc[rows, 2] = np.where(complete, (high_position > low_position).astype(float), np.nan)
    return result


def daily_indicators(frame: pd.DataFrame, fib_lookback: int = 126) -> pd.DataFrame:
    """Calculate over the entire warm-up history, before a display crop.

    `close` is required. When `high`, `low` and `volume` are present, ATR,
    volume, pullback and Fibonacci-leg features are added as well. Missing
    rows (NaN) restart every rolling calculation instead of bridging a gap.
    """
    result = frame.copy()
    close = result["close"]
    for period in (20, 50, 200):
        result[f"sma{period}"] = close.rolling(period, min_periods=period).mean()
    result["rsi14"] = wilder_rsi(close)
    for period in (63, 126):
        result[f"ret{period}"] = close / close.shift(period) - 1
    if {"high", "low"}.issubset(result.columns):
        result["atr14"] = wilder_atr(result["high"], result["low"], close)
        result["high22"] = result["high"].rolling(22, min_periods=22).max()
        result["prev_high"] = result["high"].shift(1)
        for period in RESISTANCE_LOOKBACKS:
            result[f"prior_high{period}"] = result["high"].shift(1).rolling(period, min_periods=period).max()
        result["pullback_atr5"] = ((result["low"] - result["sma20"]) / result["atr14"]).rolling(5, min_periods=5).min()
        leg = _fibonacci_leg(result["high"], result["low"], fib_lookback)
        for column in leg.columns:
            result[column] = leg[column]
    if "volume" in result.columns:
        volume = pd.to_numeric(result["volume"], errors="coerce")
        result["vol_avg20"] = volume.shift(1).rolling(20, min_periods=20).mean()
        result["rel_volume"] = volume / result["vol_avg20"].where(result["vol_avg20"] > 0)
        average20 = volume.rolling(20, min_periods=20).mean()
        result["vol_ratio5_20"] = volume.rolling(5, min_periods=5).mean() / average20.where(average20 > 0)
        change = close.diff()
        up = volume.where(change > 0, 0.0).where(change.notna())
        down = volume.where(change < 0, 0.0).where(change.notna())
        up_sum = up.rolling(20, min_periods=20).sum()
        down_sum = down.rolling(20, min_periods=20).sum()
        ratio = up_sum / down_sum.where(down_sum > 0)
        ratio = ratio.where(down_sum > 0, np.where(up_sum > 0, 5.0, np.nan))
        result["updown_volume20"] = ratio.clip(upper=5.0)
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
