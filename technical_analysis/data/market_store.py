"""Bounded validated OHLCV snapshots, scoped to one provider client.

Actual observed sessions identify snapshots. Calendar-derived epochs only decide
when to ask the provider again; they never manufacture a candle or confirm one.
"""
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from hashlib import sha256
from threading import RLock
import time

import pandas as pd

from data.normalize import COLUMNS, session_schedule

ADJUSTMENT_POLICY = "split_adjusted"
MAX_MARKET_ENTRIES = 128


@dataclass
class MarketDataset:
    frame: pd.DataFrame = field(repr=False)
    issues: list[str]
    retrieved_at: datetime
    endpoint: str
    latest_session: date | None
    version: str
    cache_hit: bool = False
    api_seconds: float = 0.0
    validation_seconds: float = 0.0


def content_version(frame: pd.DataFrame, timeframe: str,
                    adjustment_policy: str = ADJUSTMENT_POLICY) -> str:
    """Hash inputs rather than retrieval time, so unchanged revisions reuse work."""
    digest = sha256(f"{timeframe}|{adjustment_policy}".encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame[COLUMNS], index=False).values.tobytes())
    return digest.hexdigest()


def freshness_epoch(timeframe: str, now: datetime) -> pd.Timestamp | None:
    """Latest scheduled completed close/bar end, including holidays/early closes."""
    schedule = session_schedule(now.date() - timedelta(days=32), now.date())
    if schedule.empty:
        return None
    instant = pd.Timestamp(now)
    completed = schedule.loc[schedule.market_close <= instant]
    previous = None if completed.empty else completed.iloc[-1].market_close
    if timeframe == "daily":
        return previous
    today = schedule.loc[schedule.index == now.date()]
    if today.empty:
        return previous
    opening, closing = today.iloc[0][["market_open", "market_close"]]
    if instant < opening + pd.Timedelta(minutes=5):
        return previous
    elapsed = min(instant, closing) - opening
    return opening + (elapsed // pd.Timedelta(minutes=5)) * pd.Timedelta(minutes=5)


@dataclass
class _Entry:
    dataset: MarketDataset
    epoch: pd.Timestamp | None
    expires: float
    normalized_at: datetime


class MarketStore:
    """One client owns one store; credentials and provider scope stay external."""

    def __init__(self, max_entries: int = MAX_MARKET_ENTRIES):
        self._entries: OrderedDict[tuple[str, str, str, date | None], _Entry] = OrderedDict()
        self._lock = RLock()
        self._max_entries = max_entries

    def get(self, symbol: str, timeframe: str, policy: str,
            epoch: pd.Timestamp | None, now: datetime) -> MarketDataset | None:
        prefix = (symbol, timeframe, policy)
        with self._lock:
            expired = [key for key, entry in self._entries.items()
                       if entry.expires <= time.monotonic()]
            for key in expired:
                del self._entries[key]
            for key in reversed(self._entries):
                entry = self._entries[key]
                if (key[:3] == prefix and entry.epoch == epoch and
                        entry.normalized_at <= now):
                    self._entries.move_to_end(key)
                    dataset = entry.dataset
                    break
            else:
                return None
        return deepcopy(replace(dataset, cache_hit=True, api_seconds=0.0,
                                validation_seconds=0.0))

    def put(self, symbol: str, timeframe: str, policy: str,
            dataset: MarketDataset, epoch: pd.Timestamp | None,
            normalized_at: datetime, ttl_seconds: float) -> None:
        key = (symbol, timeframe, policy, dataset.latest_session)
        entry = _Entry(deepcopy(dataset), epoch, time.monotonic() + ttl_seconds,
                       normalized_at)
        with self._lock:
            self._entries[key] = entry
            self._entries.move_to_end(key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)

    def invalidate(self, symbol: str, timeframe: str) -> None:
        with self._lock:
            for key in [key for key in self._entries if key[:2] == (symbol, timeframe)]:
                del self._entries[key]
