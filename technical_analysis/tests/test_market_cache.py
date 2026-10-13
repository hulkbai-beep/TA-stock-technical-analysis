"""Validated snapshot reuse, exchange-boundary invalidation and singleflight."""
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from threading import Barrier, Event, Lock, get_ident
import time
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from config import Settings
from data import fmp_client
from data.fmp_client import FMPClient
from data.market_store import (MarketDataset, MarketStore, content_version,
                               freshness_epoch)
from data.normalize import normalize_daily
from models import DataError

NY = ZoneInfo("America/New_York")


def local(value):
    return datetime.fromisoformat(value).replace(tzinfo=NY)


def bar(value, close=101):
    return {"date": value, "open": 100, "high": 104, "low": 98,
            "close": close, "volume": 1000}


class Response:
    status_code = 200

    def __init__(self, records):
        self.records = records

    def json(self):
        return self.records


class Session:
    def __init__(self, records=None, delay=0):
        self.records = records or [bar("2025-06-02")]
        self.calls = []
        self.delay = delay
        self.lock = Lock()

    def get(self, url, **kwargs):
        with self.lock:
            self.calls.append((url, kwargs))
        if self.delay:
            time.sleep(self.delay)
        return Response(self.records.copy())


def test_validated_cache_hits_skip_api_validation_and_protect_copy(monkeypatch):
    session = Session()
    client = FMPClient(Settings(api_key="test-key"), session)
    now = local("2025-06-03T10:00:00")
    first = client.fetch_validated_daily("nvda", now)
    assert not first.cache_hit and first.api_seconds > 0 and first.validation_seconds > 0
    first.frame.loc[0, "close"] = -100
    first.issues.append("mutated")
    monkeypatch.setattr(fmp_client, "normalize_daily", lambda *args: pytest.fail("renormalized"))
    second = client.fetch_validated_daily("NVDA", now)
    assert second.cache_hit and second.api_seconds == second.validation_seconds == 0
    assert second.frame.iloc[0].close == 101 and "mutated" not in second.issues
    assert first.version == second.version and second.latest_session == date(2025, 6, 2)
    assert len(session.calls) == 1
    assert client.peek_validated_daily("NVDA", now).version == second.version


def test_actual_session_keys_policy_and_timeframe_are_separate():
    now = local("2025-06-03T10:00:00")
    frame = normalize_daily([bar("2025-06-02")], now).frame
    store = MarketStore()
    version = content_version(frame, "daily")
    dataset = MarketDataset(frame, [], now, "daily", date(2025, 6, 2), version)
    epoch = freshness_epoch("daily", now)
    store.put("NVDA", "daily", "split_adjusted", dataset, epoch, now, 300)
    assert ("NVDA", "daily", "split_adjusted", date(2025, 6, 2)) in store._entries
    assert store.get("AAPL", "daily", "split_adjusted", epoch, now) is None
    assert store.get("NVDA", "5min", "split_adjusted", epoch, now) is None
    assert store.get("NVDA", "daily", "raw", epoch, now) is None
    assert content_version(frame, "5min") != version
    assert content_version(frame, "daily", "raw") != version


@pytest.mark.parametrize("before,after", [
    ("2025-06-03T15:59:59", "2025-06-03T16:00:00"),
    ("2025-11-28T12:59:59", "2025-11-28T13:00:00"),
])
def test_close_boundary_bypasses_both_caches_without_ttl_expiry(before, after):
    session_day = before[:10]
    previous_day = "2025-06-02" if session_day.endswith("06-03") else "2025-11-26"
    session = Session([bar(previous_day), bar(session_day)])
    client = FMPClient(Settings(api_key="test-key"), session)
    forming = client.fetch_validated_daily("NVDA", local(before))
    confirmed = client.fetch_validated_daily("NVDA", local(after))
    assert not forming.frame.iloc[-1].complete
    assert confirmed.frame.iloc[-1].complete
    assert forming.version != confirmed.version
    assert len(session.calls) == 2


def test_holiday_and_midnight_do_not_advance_completed_daily_epoch():
    before = local("2025-11-26T23:59:00")
    holiday = local("2025-11-27T00:01:00")
    assert freshness_epoch("daily", before) == freshness_epoch("daily", holiday)
    assert freshness_epoch("daily", local("2025-11-28T12:59:00")).date() == date(2025, 11, 26)
    assert freshness_epoch("daily", local("2025-11-28T13:00:00")).date() == date(2025, 11, 28)
    session = Session([bar("2025-11-26")])
    client = FMPClient(Settings(api_key="test-key"), session)
    client.fetch_validated_daily("NVDA", before)
    client.fetch_validated_daily("NVDA", holiday)
    assert len(session.calls) == 1


def test_late_daily_candle_keeps_actual_session_and_retries_shortly(monkeypatch):
    tick = [100.0]
    monkeypatch.setattr(fmp_client.time, "monotonic", lambda: tick[0])
    session = Session([bar("2025-06-02")])
    client = FMPClient(Settings(api_key="test-key", cache_ttl_seconds=300), session)
    now = local("2025-06-03T16:01:00")
    late = client.fetch_validated_daily("NVDA", now)
    assert late.latest_session == date(2025, 6, 2)
    assert all(key[-1] == date(2025, 6, 2) for key in client._market_store._entries)
    assert client.fetch_validated_daily("NVDA", now).cache_hit
    tick[0] += 31
    session.records.append(bar("2025-06-03"))
    refreshed = client.fetch_validated_daily("NVDA", now)
    assert refreshed.latest_session == date(2025, 6, 3)
    assert len(session.calls) == 2


def test_cached_forming_bars_stay_forming_and_intraday_boundary_refetches():
    records = [bar("2025-06-03 09:30:00"), bar("2025-06-03 09:35:00")]
    session = Session(records)
    client = FMPClient(Settings(api_key="test-key"), session)
    first = client.fetch_validated_intraday("NVDA", local("2025-06-03T09:37:00"))
    within_epoch = client.fetch_validated_intraday("NVDA", local("2025-06-03T09:39:00"))
    assert first.frame.complete.tolist() == within_epoch.frame.complete.tolist() == [True, False]
    assert first.retrieved_at == within_epoch.retrieved_at
    assert len(session.calls) == 1
    changed = client.fetch_validated_intraday("NVDA", local("2025-06-03T09:40:00"))
    assert changed.frame.complete.tolist() == [True, True]
    assert changed.version != first.version and len(session.calls) == 2


def test_manual_refresh_revisions_version_but_unchanged_response_reuses_version():
    session = Session()
    client = FMPClient(Settings(api_key="test-key"), session)
    now = local("2025-06-03T10:00:00")
    first = client.fetch_validated_daily("NVDA", now)
    unchanged = client.fetch_validated_daily("NVDA", now, refresh=True)
    assert first.version == unchanged.version and not unchanged.cache_hit
    session.records[0] = bar("2025-06-02", close=102)
    revised = client.fetch_validated_daily("NVDA", now, refresh=True)
    assert revised.version != unchanged.version and len(session.calls) == 3


def test_clients_do_not_share_validated_or_raw_cache():
    first, second = Session(), Session([bar("2025-06-02", close=102)])
    a = FMPClient(Settings(api_key="a", base_url="https://a.example"), first)
    b = FMPClient(Settings(api_key="b", base_url="https://b.example"), second)
    now = local("2025-06-03T10:00:00")
    assert a.fetch_validated_daily("NVDA", now).frame.iloc[-1].close == 101
    assert b.fetch_validated_daily("NVDA", now).frame.iloc[-1].close == 102
    assert len(first.calls) == len(second.calls) == 1


def test_identical_concurrent_requests_coalesce_and_have_per_call_api_time():
    session = Session(delay=0.08)
    client = FMPClient(Settings(api_key="test-key"), session)
    now = local("2025-06-03T10:00:00")
    # Prewarm the exchange calendar before simultaneous requests.
    freshness_epoch("daily", now)
    start = Barrier(4)

    def fetch():
        start.wait(timeout=2)
        return client.fetch_validated_daily("NVDA", now)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: fetch(), range(4)))
    assert len(session.calls) == 1
    assert sum(not result.cache_hit for result in results) == 1
    assert sum(result.api_seconds >= 0.075 for result in results) == 1
    assert len({result.version for result in results}) == 1


def test_independent_endpoints_and_tickers_are_not_serialized():
    both_started = Barrier(3)

    class ConcurrentSession(Session):
        def get(self, url, **kwargs):
            both_started.wait(timeout=2)
            return super().get(url, **kwargs)

    now = local("2025-06-03T10:00:00")
    freshness_epoch("daily", now)
    freshness_epoch("5min", now)
    session = ConcurrentSession()
    client = FMPClient(Settings(api_key="test-key"), session)
    with ThreadPoolExecutor(max_workers=2) as pool:
        daily = pool.submit(client.fetch_daily, "NVDA", now)
        intraday = pool.submit(client.fetch_intraday, "AAPL", now)
        both_started.wait(timeout=2)
        assert daily.result(timeout=2).endpoint != intraday.result(timeout=2).endpoint
    assert len(session.calls) == 2


def test_refresh_generation_cannot_be_overwritten_by_older_inflight_response():
    started, release = Event(), Event()

    class RevisionSession(Session):
        def get(self, url, **kwargs):
            with self.lock:
                self.calls.append((url, kwargs))
                first = len(self.calls) == 1
            if first:
                started.set()
                assert release.wait(timeout=3)
            return Response([bar("2025-06-02", close=101 if first else 102)])

    now = local("2025-06-03T10:00:00")
    session = RevisionSession()
    client = FMPClient(Settings(api_key="test-key"), session)
    with ThreadPoolExecutor(max_workers=2) as pool:
        old = pool.submit(client.fetch_validated_daily, "NVDA", now)
        assert started.wait(timeout=2)
        refreshed = client.fetch_validated_daily("NVDA", now, refresh=True)
        release.set()
        assert old.result(timeout=2).frame.iloc[-1].close == 101
    assert refreshed.frame.iloc[-1].close == 102
    assert client.fetch_validated_daily("NVDA", now).frame.iloc[-1].close == 102


def test_default_http_sessions_are_thread_local(monkeypatch):
    created = []

    class ThreadSession(Session):
        def __init__(self):
            super().__init__()
            self.owner = get_ident()
            created.append(self)

        def get(self, url, **kwargs):
            assert self.owner == get_ident()
            return super().get(url, **kwargs)

    monkeypatch.setattr(fmp_client.requests, "Session", ThreadSession)
    client = FMPClient(Settings(api_key="test-key"))
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(client.fetch_daily, "NVDA", local("2025-06-03T10:00:00")).result()
    assert len(created) == 2
    assert all(session.trust_env is False for session in created)


def test_invalid_symbol_and_missing_key_are_sanitized_for_validated_api():
    session = Session()
    client = FMPClient(Settings(api_key=""), session)
    with pytest.raises(DataError) as invalid:
        client.fetch_validated_daily("AAPL?apikey=secret")
    assert invalid.value.code == "invalid_symbol"
    with pytest.raises(DataError) as missing:
        client.fetch_validated_daily("NVDA", local("2025-06-03T10:00:00"))
    assert missing.value.code == "missing_key"
    assert not session.calls


def test_validated_store_is_bounded():
    now = local("2025-06-03T10:00:00")
    frame = normalize_daily([bar("2025-06-02")], now).frame
    dataset = MarketDataset(frame, [], now, "daily", date(2025, 6, 2), content_version(frame, "daily"))
    store = MarketStore(max_entries=2)
    epoch = freshness_epoch("daily", now)
    for symbol in ("NVDA", "AAPL", "MSFT"):
        store.put(symbol, "daily", "split_adjusted", dataset, epoch, now, 300)
    assert len(store._entries) == 2
    assert store.get("NVDA", "daily", "split_adjusted", epoch, now) is None


@pytest.mark.parametrize("status,code,ttl", [
    (403, "permission_denied", 300), (429, "rate_limited", 30),
    (500, "request_failed", 30),
])
def test_sanitized_failure_cache_measures_only_real_requests(monkeypatch, status, code, ttl):
    tick = [100.0]
    monkeypatch.setattr(fmp_client.time, "monotonic", lambda: tick[0])

    class FailingSession(Session):
        def get(self, url, **kwargs):
            response = super().get(url, **kwargs)
            response.status_code = status
            return response

    session = FailingSession(delay=0.01)
    client = FMPClient(Settings(api_key="private-key"), session)
    now = local("2025-06-03T09:37:00")
    with pytest.raises(DataError) as first:
        client.fetch_validated_intraday("NVDA", now)
    assert first.value.code == code and first.value.api_seconds >= 0.009
    assert not first.value.cache_hit
    with pytest.raises(DataError) as cached:
        client.fetch_validated_intraday("NVDA", now)
    assert cached.value.code == code and cached.value.api_seconds == 0
    assert cached.value.cache_hit and len(session.calls) == 1
    assert "private-key" not in repr(client._errors)
    tick[0] += ttl + 1
    with pytest.raises(DataError):
        client.fetch_validated_intraday("NVDA", now)
    assert len(session.calls) == 2
    with pytest.raises(DataError):
        client.fetch_validated_intraday("NVDA", now, refresh=True)
    assert len(session.calls) == 3


def test_no_data_and_network_unavailability_are_cached_without_raw_exceptions():
    class EmptySession(Session):
        def get(self, url, **kwargs):
            super().get(url, **kwargs)
            return Response([])

    class NetworkSession(Session):
        def get(self, url, **kwargs):
            super().get(url, **kwargs)
            raise RuntimeError("https://provider.example?apikey=must-never-appear")

    now = local("2025-06-03T09:37:00")
    for session, code in ((EmptySession(), "no_data"), (NetworkSession(), "network_error")):
        client = FMPClient(Settings(api_key="must-never-appear"), session)
        for index in range(2):
            with pytest.raises(DataError) as unavailable:
                client.fetch_validated_intraday("NVDA", now)
            assert unavailable.value.code == code
            assert unavailable.value.cache_hit == bool(index)
        assert len(session.calls) == 1
        assert "must-never-appear" not in repr(client._errors)


def test_failure_cache_is_scoped_by_endpoint_symbol_and_epoch():
    class EmptySession(Session):
        def get(self, url, **kwargs):
            super().get(url, **kwargs)
            return Response([])

    session = EmptySession()
    client = FMPClient(Settings(api_key="test-key"), session)
    for fetcher, ticker, now in (
        (client.fetch_intraday, "NVDA", "2025-06-03T09:37:00"),
        (client.fetch_intraday, "AAPL", "2025-06-03T09:37:00"),
        (client.fetch_daily, "NVDA", "2025-06-03T09:37:00"),
        (client.fetch_intraday, "NVDA", "2025-06-03T09:40:00"),
    ):
        with pytest.raises(DataError):
            fetcher(ticker, local(now))
    assert len(session.calls) == 4
