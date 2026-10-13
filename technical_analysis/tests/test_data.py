"""Focused checks for session boundaries, dirty data, and credential safety."""
from datetime import date, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
import requests

from config import Settings
from data import fmp_client
from data.fmp_client import FMPClient, validate_symbol
from data.normalize import COLUMNS, normalize_daily, normalize_intraday, session_schedule
from models import DataError

NY = ZoneInfo("America/New_York")
SECRET = "secret-that-must-never-appear"


def local(value):
    return datetime.fromisoformat(value).replace(tzinfo=NY)


def bar(timestamp, close=101, volume=100):
    return {"date": timestamp, "open": 100, "high": 103, "low": 99,
            "close": close, "volume": volume}


class Response:
    def __init__(self, payload=None, status=200, bad_json=False):
        self.status_code = status
        self.payload = payload
        self.bad_json = bad_json

    def json(self):
        if self.bad_json:
            raise ValueError(SECRET)
        return self.payload


class Session:
    def __init__(self, response=None, failure=None):
        self.response = response or Response([bar("2025-11-26")])
        self.failure = failure
        self.calls = []
        self.trust_env = True

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        if self.failure:
            raise self.failure
        return self.response


def test_daily_completed_and_forming_early_close_with_holiday_filter():
    records = [bar("2025-11-28"), bar("2025-11-27"), bar("2025-11-26"), bar("2025-11-29")]
    result = normalize_daily(records, local("2025-11-28T12:59:00"))
    assert list(result.frame.session) == [date(2025, 11, 26), date(2025, 11, 28)]
    assert list(result.frame.complete) == [True, False]
    assert result.frame.iloc[-1].bar_end.hour == 13
    assert result.frame.iloc[-1].timestamp == result.frame.iloc[-1].bar_end
    assert any("非交易日" in issue for issue in result.issues)
    assert normalize_daily(records, local("2025-11-28T13:00:00")).frame.iloc[-1].complete
    before_open = normalize_daily(records, local("2025-11-28T09:00:00"))
    assert list(before_open.frame.session) == [date(2025, 11, 26)]


def test_intraday_unfinished_bar_retained_future_excluded():
    records = [bar("2025-06-03 09:40:00"), bar("2025-06-03 09:35:00"), bar("2025-06-03 09:30:00")]
    result = normalize_intraday(records, local("2025-06-03T09:37:00"))
    assert result.frame.timestamp.dt.strftime("%H:%M").tolist() == ["09:30", "09:35"]
    assert result.frame.complete.tolist() == [True, False]
    assert result.frame.bar_end.dt.strftime("%H:%M").tolist() == ["09:35", "09:40"]
    assert result.frame.attrs["incomplete_sessions"] == []


def test_timezone_dst_and_early_close_regular_hours():
    schedule = session_schedule("2025-03-07", "2025-03-10")
    assert schedule.loc[date(2025, 3, 7), "market_open"].utcoffset().total_seconds() == -5 * 3600
    assert schedule.loc[date(2025, 3, 10), "market_open"].utcoffset().total_seconds() == -4 * 3600
    records = [bar("2025-11-28 12:55:00"), bar("2025-11-28 13:00:00"), bar("2025-11-28 08:00:00")]
    result = normalize_intraday(records, local("2025-11-28T14:00:00"))
    assert result.frame.timestamp.dt.strftime("%H:%M").tolist() == ["12:55"]
    assert result.frame.iloc[0].bar_end.hour == 13
    assert result.frame.iloc[0].complete
    assert str(result.frame.timestamp.dt.tz) == "America/New_York"


@pytest.mark.parametrize("timestamp", ["2025-03-09 02:30:00", "2025-11-02 01:30:00", None, 12345])
def test_invalid_or_ambiguous_source_times_do_not_crash(timestamp):
    result = normalize_intraday([bar(timestamp)], local("2025-11-03T10:00:00"))
    assert result.frame.empty
    assert any("时间无效" in issue for issue in result.issues)


def test_aware_utc_and_exchange_local_timestamps_refer_to_same_bar():
    records = [bar("2025-06-03T13:30:00+00:00"), bar("2025-06-03 09:30:00")]
    result = normalize_intraday(records, local("2025-06-03T09:35:00"))
    assert len(result.frame) == 1
    assert result.frame.iloc[0].timestamp.hour == 9
    assert result.frame.iloc[0].timestamp.utcoffset().total_seconds() == -4 * 3600
    assert result.frame.iloc[0].complete


def test_dirty_records_conflicting_duplicates_and_alignment_are_rejected():
    records = [bar("2025-06-03 09:30:00"), bar("2025-06-03 09:30:00"),
               bar("2025-06-03 09:30:00", close=102), bar("2025-06-03 09:35:00"),
               bar("2025-06-03 09:40:00", volume=float("nan")),
               {**bar("2025-06-03 09:45:00"), "high": 99},
               {**bar("2025-06-03 09:50:00"), "low": 0},
               {"date": "2025-06-03 09:55:00"}, bar("invalid"),
               bar("2025-06-01 09:30:00"), bar("2025-06-03 08:00:00"),
               bar("2025-06-03 09:36:00")]
    result = normalize_intraday(records, local("2025-06-03T10:00:00"))
    assert result.frame.timestamp.dt.strftime("%H:%M").tolist() == ["09:35"]
    assert result.frame.attrs["incomplete_sessions"] == [date(2025, 6, 3)]
    for text in ("完全重复", "冲突", "OHLCV 无效", "时间无效", "非正常交易时段", "未对齐", "缺口"):
        assert any(text in issue for issue in result.issues)


def test_partial_session_warns_but_complete_prefix_with_forming_bar_does_not():
    records = [bar(f"2025-06-03 {time}:00") for time in ("09:30", "09:35", "09:40")]
    good = normalize_intraday(records, local("2025-06-03T09:42:00"))
    assert good.frame.attrs["incomplete_sessions"] == []
    missing = normalize_intraday([records[0], records[2]], local("2025-06-03T09:42:00"))
    assert missing.frame.attrs["incomplete_sessions"] == [date(2025, 6, 3)]


@pytest.mark.parametrize("normalizer", [normalize_daily, normalize_intraday])
def test_empty_schema_is_stable(normalizer):
    frame = normalizer([], local("2025-06-03T10:00:00")).frame
    assert frame.empty
    assert list(frame.columns) == COLUMNS
    assert str(frame.timestamp.dtype) == "datetime64[ns, America/New_York]"
    assert frame.complete.dtype == bool


@pytest.mark.parametrize("symbol,expected", [("nvda", "NVDA"), (" BRK.B ", "BRK.B"), ("BRK-B", "BRK-B")])
def test_ticker_validation(symbol, expected):
    assert validate_symbol(symbol) == expected


@pytest.mark.parametrize("symbol", ["", "NVDA,AAPL", "../NVDA", "AAPL?apikey=x", "A A", "ABCDEFZZZZ"])
def test_invalid_ticker_never_requests(symbol):
    session = Session()
    with pytest.raises(DataError) as error:
        FMPClient(Settings(api_key=SECRET), session).fetch_daily(symbol)
    assert error.value.code == "invalid_symbol"
    assert session.calls == []


def test_missing_key_never_requests():
    session = Session()
    with pytest.raises(DataError) as error:
        FMPClient(Settings(api_key=""), session).fetch_daily("NVDA")
    assert error.value.code == "missing_key"
    assert session.calls == []


def test_cache_defensive_copy_manual_refresh_ttl_and_request_windows(monkeypatch):
    tick = [100.0]
    monkeypatch.setattr(fmp_client.time, "monotonic", lambda: tick[0])
    session = Session()
    client = FMPClient(Settings(api_key=SECRET, cache_ttl_seconds=30), session)
    now = local("2025-06-03T10:00:00")
    first = client.fetch_daily("nvda", now)
    first.records[0]["close"] = -1
    cached = client.fetch_daily("NVDA", now)
    assert cached.records[0]["close"] == 101
    assert len(session.calls) == 1
    assert cached.retrieved_at == now
    assert SECRET not in repr(cached)
    params = session.calls[0][1]["params"]
    assert params["from"] == "2023-06-04"
    assert params["to"] == "2025-06-03"
    client.fetch_daily("NVDA", now, refresh=True)
    assert len(session.calls) == 2
    tick[0] += 31
    client.fetch_daily("NVDA", now)
    assert len(session.calls) == 3
    client.fetch_intraday("NVDA", now)
    assert session.calls[-1][1]["params"]["from"] == "2025-04-29"
    assert session.calls[-1][1]["params"]["extended"] == "false"
    assert session.calls[-1][0].endswith("/stable/historical-chart/5min")
    assert session.trust_env is False
    assert session.calls[-1][1]["verify"] is True
    assert session.calls[-1][1]["allow_redirects"] is False


def test_proxy_auth_stays_in_explicit_header():
    session = Session()
    client = FMPClient(Settings(api_key=SECRET, base_url="https://gateway.example", auth_mode="proxy"), session)
    client.fetch_daily("NVDA", local("2025-06-03T10:00:00"))
    url, kwargs = session.calls[0]
    assert url == "https://gateway.example/stable/historical-price-eod/full"
    assert kwargs["headers"] == {"X-Proxy-Key": SECRET}
    assert "apikey" not in kwargs["params"]


@pytest.mark.parametrize("payload", [
    [{**bar("2025-06-03"), "symbol": "NVDA"}],
    {"symbol": "NVDA", "historical": [bar("2025-06-03")]},
])
def test_explicit_source_ticker_mismatch_is_rejected(payload):
    with pytest.raises(DataError) as error:
        FMPClient(Settings(api_key=SECRET), Session(Response(payload))).fetch_daily("AAPL")
    assert error.value.code == "symbol_mismatch"
    assert SECRET not in str(error.value)
    assert "NVDA" not in str(error.value)


@pytest.mark.parametrize("source", ["BRK.B", "BRK-B", "brk.b", None, ""])
def test_source_class_share_aliases_and_absent_labels_are_supported(source):
    payload = [{**bar("2025-06-03"), "symbol": source}]
    result = FMPClient(Settings(api_key=SECRET), Session(Response(payload))).fetch_daily("BRK.B")
    assert len(result.records) == 1


def test_cache_is_bounded_across_many_symbols():
    session = Session()
    client = FMPClient(Settings(api_key=SECRET), session)
    now = local("2025-06-03T10:00:00")
    for index in range(129):
        client.fetch_daily(f"A{index:04}", now)
    client.fetch_daily("A0000", now)
    assert len(session.calls) == 130
    client.fetch_daily("A0128", now)
    assert len(session.calls) == 130


@pytest.mark.parametrize("status,code", [(401, "unauthorized"), (402, "permission_denied"),
                                          (403, "permission_denied"), (429, "rate_limited"),
                                          (500, "request_failed"), (302, "request_failed")])
def test_http_errors_are_sanitized(status, code):
    session = Session(Response({"Error Message": SECRET}, status))
    with pytest.raises(DataError) as error:
        FMPClient(Settings(api_key=SECRET), session).fetch_daily("NVDA")
    assert error.value.code == code
    assert SECRET not in str(error.value)
    assert error.value.__cause__ is None


@pytest.mark.parametrize("response,code", [
    (Response({"Error Message": f"Invalid API KEY {SECRET}"}), "unauthorized"),
    (Response({"Error Message": f"Premium subscription {SECRET}"}), "permission_denied"),
    (Response({"Error Message": f"Limit reached {SECRET}"}), "rate_limited"),
    (Response([]), "no_data"), (Response("invalid"), "malformed_response"),
    (Response([None]), "malformed_response"), (Response([{"bad": SECRET}]), "malformed_response"),
    (Response(bad_json=True), "malformed_response"),
])
def test_empty_malformed_and_http200_provider_errors_are_sanitized(response, code):
    with pytest.raises(DataError) as error:
        FMPClient(Settings(api_key=SECRET), Session(response)).fetch_intraday("NVDA")
    assert error.value.code == code
    assert SECRET not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__ or error.value.__context__ is None


@pytest.mark.parametrize("failure,code", [
    (requests.exceptions.SSLError(SECRET), "tls_failed"),
    (requests.exceptions.Timeout(f"https://example.test?apikey={SECRET}"), "network_error"),
])
def test_network_errors_never_chain_secret_exceptions(failure, code):
    with pytest.raises(DataError) as error:
        FMPClient(Settings(api_key=SECRET), Session(failure=failure)).fetch_daily("NVDA")
    assert error.value.code == code
    assert SECRET not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__
