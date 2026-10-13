"""Service-level content reuse, isolated ticker results and parallel data paths."""
from dataclasses import replace
from datetime import date, datetime, timedelta
from threading import Event, Lock
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from analysis.support_resistance.models import SRConfig
from data.market_store import MarketDataset, content_version
from data.normalize import normalize_intraday, session_schedule
from models import DataError
from services import analysis_service as service

NY = ZoneInfo("America/New_York")
NOW = datetime(2025, 6, 3, 17, tzinfo=NY)


@pytest.fixture
def daily_frame():
    schedule = session_schedule("2025-01-02", "2025-06-03")
    close = 100 + 8 * np.cos(np.arange(len(schedule)) * np.pi / 10)
    return pd.DataFrame({
        "session": list(schedule.index), "timestamp": list(schedule.market_close),
        "bar_end": list(schedule.market_close), "complete": True,
        "open": close - 0.2, "high": close + 1, "low": close - 1,
        "close": close, "volume": np.full(len(close), 1000.0),
    })


@pytest.fixture
def intraday_frame():
    stamps = pd.date_range("2025-06-03 09:30", "2025-06-03 15:55", freq="5min", tz=NY)
    records = [{"date": str(stamp), "open": 100.0, "high": 102.0,
                "low": 99.0, "close": 101.0, "volume": 100.0} for stamp in stamps]
    return normalize_intraday(records, NOW).frame


class ValidatedClient:
    def __init__(self, daily, intraday, *, now=NOW):
        self.daily = daily.copy(deep=True)
        self.intraday = intraday.copy(deep=True)
        self.retrieved_at = now
        self.calls = []
        self.lock = Lock()
        self.intraday_failure = None

    def _fetch(self, symbol, timeframe, refresh):
        with self.lock:
            self.calls.append((symbol, timeframe, refresh))
        if timeframe == "intraday" and self.intraday_failure is not None:
            raise self.intraday_failure
        frame = (self.daily if timeframe == "daily" else self.intraday).copy(deep=True)
        return MarketDataset(frame, [], self.retrieved_at, f"test-{timeframe}",
                             None if frame.empty else frame.iloc[-1].session,
                             content_version(frame, timeframe), cache_hit=len(self.calls) > 2)

    def fetch_validated_daily(self, symbol, now=None, refresh=False):
        return self._fetch(symbol, "daily", refresh)

    def fetch_validated_intraday(self, symbol, now=None, refresh=False):
        return self._fetch(symbol, "intraday", refresh)


@pytest.fixture
def counted_sr(monkeypatch):
    original = service.analyze_support_resistance
    calls = []

    def counted(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(service, "analyze_support_resistance", counted)
    return calls


def test_daily_and_intraday_loading_overlap_using_events(daily_frame, intraday_frame):
    daily_started, intraday_started = Event(), Event()

    class ConcurrentClient(ValidatedClient):
        def fetch_validated_daily(self, symbol, now=None, refresh=False):
            daily_started.set()
            assert intraday_started.wait(timeout=2)
            return super().fetch_validated_daily(symbol, now, refresh)

        def fetch_validated_intraday(self, symbol, now=None, refresh=False):
            intraday_started.set()
            assert daily_started.wait(timeout=2)
            return super().fetch_validated_intraday(symbol, now, refresh)

    client = ConcurrentClient(daily_frame, intraday_frame)
    result = service.analyze("NVDA", client, now=NOW)
    assert result.symbol == "NVDA" and len(client.calls) == 2
    assert daily_started.is_set() and intraday_started.is_set()


def test_unchanged_content_reuses_all_three_sr_methods(daily_frame, intraday_frame, counted_sr):
    client = ValidatedClient(daily_frame, intraday_frame)
    first = service.analyze("NVDA", client, now=NOW)
    assert first.support_resistance.available
    assert first.support_resistance.pivots
    assert first.support_resistance.profile.available
    assert first.support_resistance.anchors
    client.retrieved_at = NOW + timedelta(minutes=1)
    again = service.analyze("nvda", client, now=client.retrieved_at)
    assert again.support_resistance is first.support_resistance
    assert again.timings.derived_cache_hit
    assert again.timings.support_resistance_ms == 0
    assert len(counted_sr) == 1
    assert again.daily_retrieved_at == client.retrieved_at
    assert again.assessment.evaluated_at == client.retrieved_at
    assert again.support_resistance.pivots == first.support_resistance.pivots
    assert again.support_resistance.profile == first.support_resistance.profile
    assert again.support_resistance.anchors == first.support_resistance.anchors


def test_ticker_switching_keeps_independent_derived_entries(daily_frame, intraday_frame, counted_sr):
    client = ValidatedClient(daily_frame, intraday_frame)
    nvda = service.analyze("NVDA", client, now=NOW)
    aapl = service.analyze("AAPL", client, now=NOW)
    nvda_again = service.analyze("NVDA", client, now=NOW)
    assert nvda.symbol == nvda_again.symbol == "NVDA" and aapl.symbol == "AAPL"
    assert nvda.support_resistance is nvda_again.support_resistance
    assert aapl.support_resistance is not nvda.support_resistance
    assert nvda_again.timings.derived_cache_hit and not aapl.timings.derived_cache_hit
    assert len(counted_sr) == 2


def test_ohlcv_revision_and_explicit_refresh_recompute(daily_frame, intraday_frame, counted_sr):
    client = ValidatedClient(daily_frame, intraday_frame)
    first = service.analyze("NVDA", client, now=NOW)
    client.daily.loc[client.daily.index[-1], "close"] += 0.25
    revised = service.analyze("NVDA", client, now=NOW)
    assert revised.support_resistance is not first.support_resistance
    assert not revised.timings.derived_cache_hit
    assert revised.daily.iloc[-1].close == first.daily.iloc[-1].close + 0.25
    forced = service.analyze("NVDA", client, now=NOW, refresh=True)
    assert forced.support_resistance is not revised.support_resistance
    assert not forced.timings.derived_cache_hit and len(counted_sr) == 3
    assert set(client.calls[-2:]) == {("NVDA", "daily", True), ("NVDA", "intraday", True)}


def test_new_completed_daily_session_recomputes(daily_frame, intraday_frame, counted_sr):
    client = ValidatedClient(daily_frame, intraday_frame)
    first = service.analyze("NVDA", client, now=NOW)
    next_now = NOW + timedelta(days=1)
    close = session_schedule("2025-06-04", "2025-06-04").iloc[0].market_close
    row = client.daily.iloc[-1].copy()
    row["timestamp"] = row["bar_end"] = close
    row["session"] = date(2025, 6, 4)
    row["close"] += 0.5
    client.daily = pd.concat([client.daily, row.to_frame().T], ignore_index=True)
    for column in ("open", "high", "low", "close", "volume"):
        client.daily[column] = client.daily[column].astype(float)
    client.daily["complete"] = client.daily["complete"].astype(bool)
    client.retrieved_at = next_now
    updated = service.analyze("NVDA", client, now=next_now)
    assert first.confirmed_daily_session == date(2025, 6, 3)
    assert updated.confirmed_daily_session == date(2025, 6, 4)
    assert updated.support_resistance.analysis_session == date(2025, 6, 4)
    assert not updated.timings.derived_cache_hit and len(counted_sr) == 2


def test_completion_flag_revision_recomputes_without_promoting_cached_bar(daily_frame, intraday_frame, counted_sr):
    client = ValidatedClient(daily_frame, intraday_frame.iloc[0:0])
    client.daily.loc[client.daily.index[-1], "complete"] = False
    client.retrieved_at = NOW.replace(hour=15, minute=59)
    forming = service.analyze("NVDA", client, now=client.retrieved_at)
    later = service.analyze("NVDA", client, now=NOW)
    assert later.timings.derived_cache_hit
    assert not later.daily.iloc[-1].complete
    assert later.confirmed_daily_session == date(2025, 6, 2)
    assert later.assessment.entry.code == "insufficient_data"
    client.daily.loc[client.daily.index[-1], "complete"] = True
    client.retrieved_at = NOW
    completed = service.analyze("NVDA", client, now=NOW)
    assert not completed.timings.derived_cache_hit
    assert completed.confirmed_daily_session == date(2025, 6, 3)
    assert completed.support_resistance is not forming.support_resistance
    assert len(counted_sr) == 2


def test_provider_client_identity_isolates_identical_content(daily_frame, intraday_frame, counted_sr):
    first_client = ValidatedClient(daily_frame, intraday_frame)
    other_client = ValidatedClient(daily_frame, intraday_frame)
    first = service.analyze("NVDA", first_client, now=NOW)
    other = service.analyze("NVDA", other_client, now=NOW)
    assert first.support_resistance is not other.support_resistance
    assert not other.timings.derived_cache_hit and len(counted_sr) == 2


def test_sr_control_variants_are_memoized_without_data_requests(daily_frame, intraday_frame, counted_sr):
    client = ValidatedClient(daily_frame, intraday_frame)
    result = service.analyze("NVDA", client, now=NOW)
    before = result.daily.copy(deep=True)
    config = replace(SRConfig(), lookback_sessions=126, max_zones_per_side=2)
    variant = service.support_resistance_for(result, config)
    assert service.support_resistance_for(result, config) is variant
    assert service.support_resistance_for(result) is result.support_resistance
    assert len(client.calls) == 2 and len(counted_sr) == 2
    pd.testing.assert_frame_equal(before, result.daily)


def test_mutating_returned_frames_and_lists_does_not_modify_derived_cache(daily_frame, intraday_frame):
    client = ValidatedClient(daily_frame, intraday_frame)
    first = service.analyze("NVDA", client, now=NOW)
    expected_daily, expected_intraday = first.daily.copy(deep=True), first.intraday.copy(deep=True)
    expected_summaries, expected_warnings = list(first.summaries), list(first.warnings)
    first.daily.loc[:, "close"] = -1.0
    first.intraday.loc[:, "close"] = -1.0
    first.summaries.clear()
    first.warnings.append("caller mutation")
    cached = service.analyze("NVDA", client, now=NOW)
    pd.testing.assert_frame_equal(cached.daily, expected_daily)
    pd.testing.assert_frame_equal(cached.intraday, expected_intraday)
    assert cached.summaries == expected_summaries and cached.warnings == expected_warnings
    cached.daily.loc[:, "close"] = -2.0
    again = service.analyze("NVDA", client, now=NOW)
    pd.testing.assert_frame_equal(again.daily, expected_daily)


def test_cached_daily_only_result_keeps_intraday_unavailable(daily_frame, intraday_frame, counted_sr):
    client = ValidatedClient(daily_frame, intraday_frame)
    client.intraday_failure = DataError("测试数据权限不足。", "permission_denied", api_seconds=0.02)
    first = service.analyze("NVDA", client, now=NOW)
    client.intraday_failure = DataError("测试数据权限不足。", "permission_denied", cache_hit=True)
    cached = service.analyze("NVDA", client, now=NOW)
    assert first.intraday.empty and cached.intraday.empty
    assert cached.assessment.intraday.code == "unavailable"
    assert cached.support_resistance is first.support_resistance
    assert first.timings.intraday_api_ms == 20
    assert cached.timings.intraday_api_ms == 0 and cached.timings.intraday_cache_hit
    assert len(counted_sr) == 1


def test_derived_cache_is_bounded_per_client(monkeypatch, daily_frame, intraday_frame):
    client = ValidatedClient(daily_frame, intraday_frame)
    monkeypatch.setattr(service, "_MAX_DERIVED_PER_CLIENT", 2)
    first = service.analyze("NVDA", client, now=NOW)
    service.analyze("AAPL", client, now=NOW)
    service.analyze("MSFT", client, now=NOW)
    evicted = service.analyze("NVDA", client, now=NOW)
    assert evicted.support_resistance is not first.support_resistance
    assert not evicted.timings.derived_cache_hit
    assert len(service._derived_cache(client)) == 2
