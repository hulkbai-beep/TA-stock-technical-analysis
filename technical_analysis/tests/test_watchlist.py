"""Watchlist storage, batch analysis and the home page, all offline."""
from dataclasses import replace
from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from config import Settings
from data import normalize
from data.fmp_client import FMPClient
from data.watchlist import (DEFAULT_SYMBOLS, MAX_SYMBOLS, WatchItem, add_symbol, load_watchlist,
                            remove_symbols, save_watchlist, set_holding)
from models import DataError
from scripts.benchmark_performance import _OfflineSession
from services import analysis_service
from services.watchlist_service import analyze_watchlist, summarize_result
from test_app import _daily_only_result

NY = ZoneInfo("America/New_York")
ROUTER = Path(__file__).resolve().parents[1] / "app.py"
NOW = datetime(2026, 10, 12, 17, tzinfo=NY)
_APPS = []


@pytest.fixture(autouse=True)
def isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("WATCHLIST_FILE", str(tmp_path / "watchlist.json"))
    monkeypatch.setenv("FMP_API_KEY", "offline-test-key")
    monkeypatch.setenv("FMP_BASE_URL", "https://financialmodelingprep.com")
    monkeypatch.setenv("FMP_AUTH_MODE", "query")
    monkeypatch.setenv("FMP_CA_BUNDLE", "")
    st.cache_resource.clear()
    yield tmp_path / "watchlist.json"
    for app in _APPS:
        try:
            app.session_state["ticker_requests"].close()
        except KeyError:
            pass
    _APPS.clear()
    st.cache_resource.clear()


def _saved(path):
    return json.loads(path.read_text(encoding="utf-8"))["tickers"]


# ---- storage -------------------------------------------------------------

def test_missing_file_starts_with_default_list(isolated):
    items, problem = load_watchlist()
    assert [item.symbol for item in items] == list(DEFAULT_SYMBOLS) and problem is None
    assert not isolated.exists()


def test_save_load_round_trip_with_holding_and_removal(isolated):
    items = add_symbol([WatchItem("NVDA")], "brk.b", holding=True)
    items = set_holding(items, "NVDA", True)
    save_watchlist(items)
    loaded, _ = load_watchlist()
    assert loaded == [WatchItem("NVDA", True), WatchItem("BRK.B", True)]
    save_watchlist(remove_symbols(loaded, ["NVDA"]))
    assert _saved(isolated) == [{"symbol": "BRK.B", "holding": True}]


@pytest.mark.parametrize("symbol,code", [("NVDA", "duplicate_symbol"), ("not a ticker", "invalid_symbol")])
def test_add_rejects_duplicates_and_invalid_symbols(symbol, code):
    with pytest.raises(DataError) as error:
        add_symbol([WatchItem("NVDA")], symbol)
    assert error.value.code == code


def test_add_rejects_more_than_the_limit():
    items = [WatchItem(f"A{index}") for index in range(MAX_SYMBOLS)]
    with pytest.raises(DataError) as error:
        add_symbol(items, "ZZZ")
    assert error.value.code == "watchlist_full"


def test_unreadable_file_is_reported_and_never_overwritten(isolated):
    isolated.write_text("{not json", encoding="utf-8")
    items, problem = load_watchlist()
    assert items == [] and problem
    assert isolated.read_text(encoding="utf-8") == "{not json"


# ---- batch analysis --------------------------------------------------------

def _offline_client(symbols):
    session = _OfflineSession(symbols)
    return FMPClient(Settings(api_key="offline"), session=session), session


def test_batch_analysis_uses_daily_bars_only_and_isolates_failures():
    client, session = _offline_client(["NVDA", "AAPL"])
    urls = []
    original_get = session.get

    def tracking_get(url, **kwargs):
        urls.append(url)
        return original_get(url, **kwargs)

    session.get = tracking_get
    progress = []
    rows = analyze_watchlist(["NVDA", "AAPL", "ZZZZ"], client, now=NOW,
                             progress=lambda done, total, symbol: progress.append((done, total)))
    assert list(rows) == ["NVDA", "AAPL", "ZZZZ"]
    assert rows["ZZZZ"].error and rows["NVDA"].error is None
    assert rows["NVDA"].score is not None and rows["NVDA"].price > 0
    assert not any(url.endswith("historical-chart/5min") for url in urls)
    assert sorted(progress) == [(1, 3), (2, 3), (3, 3)]


def test_row_switches_between_entry_and_holding_guidance_without_new_data():
    client, _ = _offline_client(["NVDA"])
    result = analysis_service.analyze("NVDA", client, now=NOW, include_intraday=False)
    row = summarize_result(result, NOW)
    assert row.action(False)[0] == result.assessment.entry.code
    assert row.action(True)[0] == result.assessment.position.code
    assert row.action(False)[2] and row.action(True)[2]


def test_unexpected_failure_names_the_error_and_logs_the_traceback(monkeypatch, caplog):
    def stale_analyze(symbol, client, now=None, refresh=False):  # an older signature
        raise AssertionError("never called with these arguments")

    monkeypatch.setattr(analysis_service, "analyze", stale_analyze)
    rows = analyze_watchlist(["NVDA"], client=None, now=NOW)
    assert "TypeError" in rows["NVDA"].error
    assert "Watchlist analysis failed for NVDA" in caplog.text and "include_intraday" in caplog.text


# ---- home page -------------------------------------------------------------

def _fake_analysis(calls):
    template = _daily_only_result()

    def analyze(symbol, client, *, now=None, refresh=False, include_intraday=True):
        calls.append((symbol, refresh, include_intraday))
        result = replace(template, symbol=symbol)
        result.assessment = analysis_service.assessment_for(result, now=NOW)
        return result
    return analyze


def _home():
    app = AppTest.from_file(str(ROUTER)).run(timeout=30)
    _APPS.append(app)
    assert not app.exception
    return app


def _button(app, label):
    return next(button for button in app.button if button.label == label)


def test_home_loads_every_ticker_once_and_shows_a_sorted_table(monkeypatch):
    calls = []
    monkeypatch.setattr(analysis_service, "analyze", _fake_analysis(calls))
    app = _home()
    assert sorted(calls) == sorted((symbol, False, False) for symbol in DEFAULT_SYMBOLS)
    table = app.dataframe[0].value
    assert list(table["代码"]) == sorted(DEFAULT_SYMBOLS)
    assert {"操作参考", "综合评分", "主要原因", "距吊灯止损 %", "下次财报"} <= set(table.columns)
    app.run(timeout=30)
    assert len(calls) == len(DEFAULT_SYMBOLS), "a rerun must reuse the session's rows"


def test_add_ticker_saves_and_analyzes_only_the_new_symbol(monkeypatch, isolated):
    calls = []
    monkeypatch.setattr(analysis_service, "analyze", _fake_analysis(calls))
    app = _home()
    calls.clear()
    app.text_input[0].input("tsla")
    _button(app, "＋ 添加股票").click().run(timeout=30)
    assert not app.exception
    assert calls == [("TSLA", False, False)]
    assert [row["symbol"] for row in _saved(isolated)] == [*DEFAULT_SYMBOLS, "TSLA"]
    assert "TSLA" in set(app.dataframe[0].value["代码"])


def test_invalid_or_duplicate_ticker_warns_without_saving(monkeypatch, isolated):
    monkeypatch.setattr(analysis_service, "analyze", _fake_analysis([]))
    app = _home()
    app.text_input[0].input("NVDA")
    _button(app, "＋ 添加股票").click().run(timeout=30)
    assert any("已在自选股中" in warning.value for warning in app.warning)
    assert not isolated.exists()


def test_refresh_all_bypasses_caches_for_every_ticker(monkeypatch):
    calls = []
    monkeypatch.setattr(analysis_service, "analyze", _fake_analysis(calls))
    app = _home()
    calls.clear()
    _button(app, "↻ 刷新全部").click().run(timeout=30)
    assert sorted(calls) == sorted((symbol, True, False) for symbol in DEFAULT_SYMBOLS)


def test_auto_refresh_runs_the_whole_list_when_due_during_market_hours(monkeypatch):
    calls = []
    monkeypatch.setattr(analysis_service, "analyze", _fake_analysis(calls))

    def open_schedule(start, end):
        import pandas as pd
        opening = pd.Timestamp(start, tz=NY)
        return pd.DataFrame({"market_open": [opening], "market_close": [opening + pd.Timedelta(days=1)]},
                            index=[start])

    monkeypatch.setattr(normalize, "session_schedule", open_schedule)
    app = _home()
    calls.clear()
    app.run(timeout=30)
    assert calls == [], "not due yet"
    app.session_state["watchlist_clock"] = -1e9
    app.run(timeout=30)
    assert sorted(calls) == sorted((symbol, True, False) for symbol in DEFAULT_SYMBOLS)


def test_unreadable_watchlist_disables_editing(monkeypatch, isolated):
    isolated.write_text("{broken", encoding="utf-8")
    monkeypatch.setattr(analysis_service, "analyze", _fake_analysis([]))
    app = _home()
    assert app.error and _button(app, "＋ 添加股票").disabled
    assert isolated.read_text(encoding="utf-8") == "{broken"


def test_missing_key_reports_once_without_analysis(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "")
    calls = []
    monkeypatch.setattr(analysis_service, "analyze", _fake_analysis(calls))
    app = _home()
    assert calls == [] and any("FMP_API_KEY" in error.value for error in app.error)


def test_opening_a_row_loads_that_ticker_with_its_holding_state(monkeypatch, isolated):
    save_watchlist([WatchItem("NVDA"), WatchItem("AAPL", True)])
    calls = []
    monkeypatch.setattr(analysis_service, "analyze", _fake_analysis(calls))
    app = _home()
    calls.clear()
    app.session_state["open_ticker"] = ("AAPL", True)
    app.switch_page("views/detail.py").run(timeout=30)
    assert not app.exception
    assert app.text_input(key="ticker_input").value == "AAPL"
    assert app.toggle(key="holding_status").value is True
    for _ in range(50):
        if calls:
            break
        app.run(timeout=30)
    assert calls and calls[0][0] == "AAPL"


def test_detail_page_can_add_to_watchlist_and_syncs_holding(monkeypatch, isolated):
    save_watchlist([WatchItem("NVDA")])
    calls = []
    monkeypatch.setattr(analysis_service, "analyze", _fake_analysis(calls))
    app = _home()
    app.session_state["open_ticker"] = ("TSLA", False)
    app.switch_page("views/detail.py").run(timeout=30)
    for _ in range(50):
        try:
            if app.session_state["analysis_result"].symbol == "TSLA":
                break
        except KeyError:
            pass
        app.run(timeout=30)
    app.run(timeout=30)
    _button(app, "☆ 加入自选股").click().run(timeout=30)
    assert [row["symbol"] for row in _saved(isolated)] == ["NVDA", "TSLA"]
    app.toggle(key="holding_status").set_value(True).run(timeout=30)
    assert _saved(isolated)[-1] == {"symbol": "TSLA", "holding": True}
