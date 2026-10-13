"""Defensive daily-bar preparation and gap-aware Wilder ATR.

The data service supplies FMP's split-adjusted OHLCV. This module neither
guesses split ratios nor adjusts prices again. Calendar gaps remain gaps.
"""
from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

from data.normalize import NEW_YORK, OHLCV, session_schedule

from .models import PreparedData, SRConfig


_REQUIRED = {"session", "timestamp", "bar_end", "complete", *OHLCV}


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "session": pd.Series(dtype=object),
        "timestamp": pd.Series(dtype="datetime64[ns, America/New_York]"),
        "bar_end": pd.Series(dtype="datetime64[ns, America/New_York]"),
        "complete": pd.Series(dtype=bool),
        **{column: pd.Series(dtype=float) for column in OHLCV},
        "atr": pd.Series(dtype=float),
        "segment_id": pd.Series(dtype=int),
        "missing_before": pd.Series(dtype=int),
        "gap_before": pd.Series(dtype=bool),
    })


def _local_timestamp(value: object) -> pd.Timestamp | None:
    if not isinstance(value, (str, date, datetime, pd.Timestamp)):
        return None
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp) or not 1900 <= stamp.year <= 2100:
            return None
        if stamp.tzinfo is None:
            stamp = stamp.tz_localize(NEW_YORK, ambiguous="NaT", nonexistent="NaT")
        else:
            stamp = stamp.tz_convert(NEW_YORK)
        return None if pd.isna(stamp) else stamp
    except (TypeError, ValueError, OverflowError):
        return None


def _prices(row: dict) -> dict[str, float] | None:
    try:
        if any(isinstance(row[column], (bool, np.bool_)) for column in OHLCV):
            return None
        result = {column: float(row[column]) for column in OHLCV}
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not all(np.isfinite(value) for value in result.values()):
        return None
    if any(result[column] <= 0 for column in OHLCV[:-1]) or result["volume"] < 0:
        return None
    if (result["low"] > min(result["open"], result["close"]) or
            result["high"] < max(result["open"], result["close"]) or
            result["high"] < result["low"]):
        return None
    return result


def _wilder_atr(frame: pd.DataFrame, period: int) -> np.ndarray:
    """Seed with `period` true ranges; reset at each missing NYSE session."""
    values = np.full(len(frame), np.nan)
    previous_close = None
    previous_segment = None
    seed: list[float] = []
    average = None
    for index, row in enumerate(frame.itertuples(index=False)):
        if row.segment_id != previous_segment:
            previous_close = None
            seed = []
            average = None
            previous_segment = row.segment_id
        true_range = row.high - row.low
        if previous_close is not None:
            true_range = max(true_range, abs(row.high - previous_close), abs(row.low - previous_close))
        previous_close = row.close
        if average is None:
            seed.append(true_range)
            if len(seed) < period:
                continue
            average = sum(seed) / period
        else:
            average = (average * (period - 1) + true_range) / period
        values[index] = average
    return values


def prepare_daily(daily: pd.DataFrame, config: SRConfig) -> PreparedData:
    """Validate full history, calculate ATR, then crop to the requested window.

    The returned zero-based frame includes `atr`, `segment_id`, `missing_before`
    and `gap_before`. Consecutive rows in a segment are consecutive NYSE dates;
    consumers must not use windows or cumulative anchors across segments.
    Only literal boolean completion flags establish completed observations.
    """
    if not isinstance(daily, pd.DataFrame) or not _REQUIRED.issubset(daily.columns):
        return PreparedData(_empty_frame(), reason="日线缺少必要的日期、完成状态或 OHLCV 字段，支撑阻力分析不可用。")
    if daily.empty:
        return PreparedData(_empty_frame(), reason="没有可用的已完成日线，支撑阻力分析不可用。")

    warnings: list[str] = []
    counts = {name: 0 for name in ("unfinished", "invalid", "time", "calendar")}
    rows = []
    for row in daily.to_dict("records"):
        if not isinstance(row["complete"], (bool, np.bool_)) or not row["complete"]:
            counts["unfinished"] += 1
            continue
        session_stamp = _local_timestamp(row["session"])
        timestamp = _local_timestamp(row["timestamp"])
        bar_end = _local_timestamp(row["bar_end"])
        if (session_stamp is None or timestamp is None or bar_end is None or
                timestamp.date() != session_stamp.date() or bar_end.date() != session_stamp.date()):
            counts["time"] += 1
            continue
        prices = _prices(row)
        if prices is None:
            counts["invalid"] += 1
            continue
        rows.append({**row, **prices, "session": session_stamp.date(),
                     "timestamp": timestamp, "bar_end": bar_end, "complete": True})

    if rows:
        schedule = session_schedule(min(row["session"] for row in rows), max(row["session"] for row in rows))
        closings = schedule["market_close"].to_dict()
        accepted = []
        for row in rows:
            session = row["session"]
            if session not in closings:
                counts["calendar"] += 1
                continue
            closing = closings[session]
            if row["timestamp"] != closing or row["bar_end"] != closing:
                counts["time"] += 1
                continue
            accepted.append(row)
        rows = accepted

    messages = {
        "unfinished": "已排除 {count} 条未完成或完成状态无效的日线；这些记录不参与支撑阻力确认。",
        "invalid": "已排除 {count} 条缺失、非有限、非正价格或 OHLCV 不一致的日线。",
        "time": "已排除 {count} 条日期或收盘时间不一致的日线。",
        "calendar": "已排除 {count} 条非 NYSE 交易日日线。",
    }
    warnings.extend(messages[name].format(count=count) for name, count in counts.items() if count)
    if not rows:
        return PreparedData(_empty_frame(), tuple(warnings), reason="没有通过校验的已完成日线，支撑阻力分析不可用。")

    frame = pd.DataFrame(rows).sort_values("session").reset_index(drop=True)
    exact = frame.duplicated(subset=["session", *OHLCV], keep="first")
    if exact.any():
        warnings.append(f"已合并 {int(exact.sum())} 条 OHLCV 完全相同的重复日线。")
        frame = frame.loc[~exact].copy()
    conflicts = frame.duplicated("session", keep=False)
    if conflicts.any():
        warnings.append(f"已剔除 {int(conflicts.sum())} 条同一交易日 OHLCV 冲突的日线。")
        frame = frame.loc[~conflicts].copy()
    frame = frame.reset_index(drop=True)
    if frame.empty:
        return PreparedData(_empty_frame(), tuple(warnings), reason="重复日线存在冲突，没有可确认的历史数据。")

    schedule = session_schedule(frame.iloc[0]["session"], frame.iloc[-1]["session"])
    session_positions = {session: index for index, session in enumerate(schedule.index)}
    positions = np.array([session_positions[session] for session in frame["session"]], dtype=int)
    missing_before = np.concatenate(([0], np.maximum(np.diff(positions) - 1, 0)))
    frame["missing_before"] = missing_before
    frame["gap_before"] = missing_before > 0
    frame["segment_id"] = np.cumsum(frame["gap_before"].to_numpy(dtype=int))
    missing = int(missing_before.sum())
    if missing:
        warnings.append(f"已完成历史日线缺失 {missing} 个 NYSE 交易日；不填补价格，ATR、摆动窗口与锚定 VWAP 不跨缺口延续。")
    frame["atr"] = _wilder_atr(frame, config.atr_period)
    changes = frame["close"].pct_change(fill_method=None).abs()
    jumps = frame.loc[changes >= 0.40, "session"].tolist()
    if jumps:
        dates = "、".join(str(session) for session in jumps[:5])
        warnings.append(f"{dates} 出现至少 40% 的相邻有效收盘价变动，可能涉及价格调整或数据不连续，也可能是真实跳空；沿用来源 OHLCV，不推测拆股比例或再次调整。")

    frame = frame.tail(config.lookback_sessions).reset_index(drop=True)
    if len(frame) < config.lookback_sessions:
        warnings.append(f"请求 {config.lookback_sessions} 个交易日，实际可用 {len(frame)} 个已完成交易日；使用部分历史窗口。")
    last_atr = float(frame.iloc[-1]["atr"])
    latest_atr = last_atr if np.isfinite(last_atr) and last_atr > 0 else None
    if len(frame) < config.min_history_sessions:
        reason = f"历史数据不足：需要至少 {config.min_history_sessions} 个有效已完成交易日，当前仅有 {len(frame)} 个。"
    elif latest_atr is None:
        reason = f"最近连续有效日线不足以形成正值 ATR{config.atr_period}，或价格完全无波动；不使用固定金额替代。"
    else:
        reason = None
    return PreparedData(frame, tuple(warnings), latest_atr, reason is None, reason, missing)
