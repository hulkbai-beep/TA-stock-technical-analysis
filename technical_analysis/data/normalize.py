"""Validate OHLCV records and map them to NYSE regular trading sessions.

FMP's FAQ describes timestamps by exchange region. Naive US timestamps are
therefore interpreted in America/New_York, including DST. Intraday timestamps
are assumed to mark the interval start; a five-minute bar ends five minutes later.
Daily observations have no source time, so their timestamps use scheduled close.
"""
from datetime import date, datetime, timezone
from functools import lru_cache
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from models import NormalizedBars

NEW_YORK = ZoneInfo("America/New_York")
OHLCV = ["open", "high", "low", "close", "volume"]
COLUMNS = ["timestamp", "session", "bar_end", "complete", *OHLCV]


def _now_timestamp(now: datetime | None) -> pd.Timestamp:
    value = pd.Timestamp(now or datetime.now(timezone.utc))
    return value.tz_localize(NEW_YORK) if value.tzinfo is None else value.tz_convert(NEW_YORK)


@lru_cache(maxsize=64)
def _schedule_cached(start: str, end: str) -> pd.DataFrame:
    schedule = mcal.get_calendar("NYSE").schedule(start_date=start, end_date=end)
    schedule = schedule[["market_open", "market_close"]].copy()
    for column in schedule:
        schedule[column] = schedule[column].dt.tz_convert(NEW_YORK)
    schedule.index = pd.Index(schedule.index.date, name="session")
    return schedule


def session_schedule(start: date | str, end: date | str) -> pd.DataFrame:
    """Return exchange-local open/close timestamps indexed by Python dates."""
    return _schedule_cached(str(start), str(end)).copy()


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.Series(dtype="datetime64[ns, America/New_York]"),
        "session": pd.Series(dtype=object),
        "bar_end": pd.Series(dtype="datetime64[ns, America/New_York]"),
        "complete": pd.Series(dtype=bool),
        **{column: pd.Series(dtype=float) for column in OHLCV},
    })[COLUMNS]


def _validated_prices(record: dict) -> dict[str, float] | None:
    try:
        if any(isinstance(record.get(column), (bool, np.bool_)) for column in OHLCV):
            return None
        values = {column: float(record[column]) for column in OHLCV}
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    if not all(np.isfinite(value) for value in values.values()):
        return None
    if any(values[column] <= 0 for column in OHLCV[:-1]) or values["volume"] < 0:
        return None
    if (values["low"] > min(values["open"], values["close"]) or
            values["high"] < max(values["open"], values["close"]) or
            values["high"] < values["low"]):
        return None
    return values


def normalize_daily(records: list[dict], now: datetime | None = None) -> NormalizedBars:
    return _normalize(records, now, intraday=False)


def normalize_intraday(records: list[dict], now: datetime | None = None) -> NormalizedBars:
    return _normalize(records, now, intraday=True)


def _normalize(records: list[dict], now: datetime | None, intraday: bool) -> NormalizedBars:
    local_now = _now_timestamp(now)
    issues = []
    counts = {name: 0 for name in ("invalid_price", "invalid_time", "outside", "future", "alignment")}
    parsed = []
    for record in records:
        if not isinstance(record, dict) or (prices := _validated_prices(record)) is None:
            counts["invalid_price"] += 1
            continue
        try:
            if not isinstance(record["date"], (str, datetime, date, pd.Timestamp)):
                raise ValueError
            source = pd.Timestamp(record["date"])
            if pd.isna(source) or source.year < 1900:
                raise ValueError
            if source.tzinfo is None:
                source = source.tz_localize(NEW_YORK, ambiguous="NaT", nonexistent="NaT")
            else:
                source = source.tz_convert(NEW_YORK)
            if pd.isna(source):
                raise ValueError
            session = source.date()
        except (KeyError, TypeError, ValueError, OverflowError):
            counts["invalid_time"] += 1
            continue
        if session > local_now.date() or (intraday and source > local_now):
            counts["future"] += 1
            continue
        parsed.append({"source": source, "session": session, **prices})
    if not parsed:
        return NormalizedBars(_empty_frame(), _validation_issues(counts))

    schedule = session_schedule(min(row["session"] for row in parsed), local_now.date())
    rows = []
    for row in parsed:
        session = row["session"]
        if session not in schedule.index:
            counts["outside"] += 1
            continue
        opening, closing = schedule.loc[session, ["market_open", "market_close"]]
        if local_now < opening:
            counts["future"] += 1
            continue
        source = row.pop("source")
        if intraday:
            if not opening <= source < closing:
                counts["outside"] += 1
                continue
            if (source - opening) % pd.Timedelta("5min") != pd.Timedelta("0ns"):
                counts["alignment"] += 1
                continue
            timestamp = source
            bar_end = source + pd.Timedelta("5min")
        else:
            timestamp = bar_end = closing
        rows.append({"timestamp": timestamp, "bar_end": bar_end,
                     "complete": bool(bar_end <= local_now), **row})
    issues.extend(_validation_issues(counts))
    if not rows:
        return NormalizedBars(_empty_frame(), issues)

    frame = pd.DataFrame(rows)[COLUMNS].sort_values("timestamp").reset_index(drop=True)
    exact = frame.duplicated(subset=["timestamp", *OHLCV], keep="first")
    if exact.any():
        issues.append(f"已合并 {int(exact.sum())} 条完全重复的记录。")
        frame = frame.loc[~exact].copy()
    conflicts = frame.duplicated("timestamp", keep=False)
    if conflicts.any():
        issues.append(f"已剔除 {int(conflicts.sum())} 条同一时间但 OHLCV 冲突的记录。")
        frame = frame.loc[~conflicts].copy()
    frame = frame.reset_index(drop=True)
    if frame.empty:
        return NormalizedBars(_empty_frame(), issues)
    frame["complete"] = frame["complete"].astype(bool)
    for column in OHLCV:
        frame[column] = frame[column].astype(float)
    if intraday:
        incomplete_sessions = []
        for session, bars in frame.groupby("session", sort=True):
            opening, closing = schedule.loc[session, ["market_open", "market_close"]]
            expected = pd.date_range(opening, closing, freq="5min", inclusive="left")
            expected = expected[expected + pd.Timedelta("5min") <= local_now]
            actual = pd.DatetimeIndex(bars.loc[bars["complete"], "timestamp"])
            if not expected.difference(actual).empty:
                incomplete_sessions.append(session)
                issues.append(f"{session.isoformat()} 的正常时段 5 分钟数据不完整或有缺口，session VWAP 暂不可用。")
        frame.attrs["incomplete_sessions"] = incomplete_sessions
    return NormalizedBars(frame, issues)


def _validation_issues(counts: dict[str, int]) -> list[str]:
    templates = {
        "invalid_price": "已剔除 {count} 条缺失字段或 OHLCV 无效的记录。",
        "invalid_time": "已剔除 {count} 条时间无效的记录。",
        "outside": "已剔除 {count} 条非正常交易时段或非交易日记录。",
        "future": "已剔除 {count} 条时间尚未到达的记录。",
        "alignment": "已剔除 {count} 条未对齐 5 分钟区间的记录。",
    }
    return [templates[name].format(count=count) for name, count in counts.items() if count]
