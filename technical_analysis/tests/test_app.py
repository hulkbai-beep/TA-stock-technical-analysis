"""UI smoke checks use offline fixtures; no test calls the market-data API."""
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from threading import Event
from time import monotonic, perf_counter
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from analysis.indicators import daily_indicators
from analysis.summary import SummaryItem
from config import Settings
from data import fmp_client, market_store, normalize
from data.fmp_client import validate_symbol
from data.normalize import session_schedule
from models import DataError
from services import analysis_service
from services.analysis_service import AnalysisResult, Snapshot

APP = Path(__file__).resolve().parents[1] / "views" / "detail.py"
NEW_YORK = ZoneInfo("America/New_York")
_APP_TESTS = []


@pytest.fixture(autouse=True)
def isolated_app_configuration(monkeypatch, tmp_path):
    """Never consume a real local credential, gateway or watchlist in tests."""
    monkeypatch.setenv("WATCHLIST_FILE", str(tmp_path / "watchlist.json"))
    monkeypatch.setenv("FMP_API_KEY", "offline-test-key")
    monkeypatch.setenv("FMP_BASE_URL", "https://financialmodelingprep.com")
    monkeypatch.setenv("FMP_AUTH_MODE", "query")
    monkeypatch.setenv("FMP_CA_BUNDLE", "")
    st.cache_resource.clear()
    yield
    for app in _APP_TESTS:
        try:
            app.session_state["ticker_requests"].close()
        except KeyError:
            pass
    _APP_TESTS.clear()
    st.cache_resource.clear()


def _button(app: AppTest, label: str):
    return next(button for button in app.button if button.label == label)


def _new_app():
    app = AppTest.from_file(str(APP)).run(timeout=20)
    _APP_TESTS.append(app)
    return app


def _settle(app: AppTest):
    """AppTest does not schedule periodic fragments; rerun to publish workers."""
    deadline = monotonic() + 5
    while monotonic() < deadline:
        app.run(timeout=20)
        assert not app.exception
        requests = app.session_state["ticker_requests"]
        state = requests.poll()
        try:
            result = app.session_state["analysis_result"]
        except KeyError:
            result = None
        if state.status == "idle" or (state.status == "error" and app.error):
            return app
        if state.status == "ready" and result is not None and result.symbol == state.symbol:
            return app
        Event().wait(0.005)
    pytest.fail("Background analysis did not settle")


def _daily_table(app: AppTest):
    return next(table.value for table in app.dataframe
                if {"交易日", "sma20", "sma50", "rsi14"}.issubset(table.value.columns))


def _daily_only_result() -> AnalysisResult:
    schedule = session_schedule("2026-05-01", "2026-10-09")
    close = pd.Series([100 + index * 0.2 for index in range(len(schedule))])
    daily = pd.DataFrame({
        "timestamp": list(schedule["market_close"]),
        "bar_end": list(schedule["market_close"]),
        "session": list(schedule.index),
        "complete": True,
        "open": close - 0.1,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": 1000.0,
    })
    daily = daily_indicators(daily)
    last = daily.iloc[-1]
    return AnalysisResult(
        symbol="NVDA",
        daily=daily,
        intraday=daily.iloc[:0].copy(),
        snapshot=Snapshot(
            price=float(last["close"]), change_percent=0.2,
            session=last["session"], market_timestamp=last["timestamp"].to_pydatetime(),
            source="日线收盘", provisional=False, volume=1000,
            timestamp_note="由交易日日历推定的收盘时刻。", volume_note="已完成日线成交量。",
        ),
        summaries=[SummaryItem("日线趋势", "已完成日线价格高于 SMA20。", "测试规则及数值。", "test.daily")],
        warnings=[],
        daily_retrieved_at=datetime(2026, 10, 12, 12, tzinfo=NEW_YORK),
        intraday_error="当前 FMP 账户无权访问此数据。日线功能仍可使用。",
        confirmed_daily_session=last["session"],
    )


def test_missing_key_stops_before_analysis(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "")
    calls = []

    def unexpected_analysis(*args, **kwargs):
        calls.append(True)
        raise AssertionError("No-key UI must not call the analysis service")

    monkeypatch.setattr(analysis_service, "analyze", unexpected_analysis)
    app = _new_app()
    assert not app.exception
    _button(app, "分析").click().run(timeout=20)
    _settle(app)
    assert not app.exception
    assert any("FMP_API_KEY" in error.value for error in app.error)
    assert calls == []
    assert len(app.get("plotly_chart")) == 0


def test_invalid_ticker_is_actionable_without_network(monkeypatch):
    def offline_analysis(symbol, client, **kwargs):
        validate_symbol(symbol)
        raise AssertionError("Invalid ticker unexpectedly passed validation")

    monkeypatch.setattr(analysis_service, "analyze", offline_analysis)
    app = _new_app()
    app.text_input[0].set_value("not a ticker").run(timeout=20)
    _settle(app)
    assert not app.exception
    assert any("有效的单个美股代码" in error.value for error in app.error)
    assert len(app.get("plotly_chart")) == 0


def test_daily_only_range_changes_reuse_result_and_refresh_is_explicit(monkeypatch):
    result = _daily_only_result()
    calls = []

    def offline_analysis(symbol, client, *, refresh=False):
        calls.append((validate_symbol(symbol), refresh))
        return result

    monkeypatch.setattr(analysis_service, "analyze", offline_analysis)
    app = _new_app()
    _button(app, "分析").click().run(timeout=20)
    _settle(app)
    assert not app.exception
    assert not app.error
    assert calls == [("NVDA", False)]
    assert not _button(app, "↻ 刷新数据").disabled
    assert {"最新可用价 · USD", "RSI14 · 已完成日线", "快照交易日", "该交易日累计成交量 · 股"}.issubset(
        {metric.label for metric in app.metric})
    assert len(app.get("plotly_chart")) >= 2
    assert any("无权访问" in item.value for item in app.info)
    assert any("日线趋势" in item.value for item in app.markdown)

    six_month = _daily_table(app).copy()
    app.radio[0].set_value("3M").run(timeout=20)
    assert not app.exception
    three_month = _daily_table(app).copy()
    assert 0 < len(three_month) < len(six_month)
    overlap = six_month.loc[six_month["交易日"].isin(three_month["交易日"])]
    pd.testing.assert_frame_equal(
        overlap[["交易日", "sma20", "sma50", "rsi14"]].reset_index(drop=True),
        three_month[["交易日", "sma20", "sma50", "rsi14"]].reset_index(drop=True),
    )
    app.radio[0].set_value("1Y").run(timeout=20)
    assert not app.exception
    assert calls == [("NVDA", False)]
    _button(app, "↻ 刷新数据").click().run(timeout=20)
    _settle(app)
    assert not app.exception
    assert calls == [("NVDA", False), ("NVDA", True)]


def test_sr_controls_recompute_from_cached_history_without_fetching(monkeypatch):
    result = _daily_only_result()
    calls = []

    def offline_analysis(symbol, client, *, refresh=False):
        calls.append((symbol, refresh))
        return result

    monkeypatch.setattr(analysis_service, "analyze", offline_analysis)
    app = _new_app()
    _button(app, "分析").click().run(timeout=20)
    _settle(app)
    assert not app.exception
    assert app.selectbox(key="sr_window").value == "12 个月 · 252 个交易日"
    assert app.selectbox(key="sr_count").value == 3
    assert [item.value for item in app.checkbox] == [True, True, True, False]
    charts_with_profile = len(app.get("plotly_chart"))
    app.checkbox[1].set_value(False).run(timeout=20)
    assert not app.exception
    assert len(app.get("plotly_chart")) == charts_with_profile - 1
    app.selectbox(key="sr_window").set_value("6 个月 · 126 个交易日").run(timeout=20)
    app.selectbox(key="sr_count").set_value(5).run(timeout=20)
    app.checkbox[0].set_value(False).run(timeout=20)
    app.checkbox[2].set_value(False).run(timeout=20)
    app.checkbox[3].set_value(True).run(timeout=20)
    assert not app.exception
    assert any("综合评分" in metric.label for metric in app.metric)
    assert calls == [("NVDA", False)]
    assert any("最近 126 个交易日" in item.value for item in app.caption)


def test_holding_status_changes_guidance_without_refetching_or_changing_bias(monkeypatch):
    result = _daily_only_result()
    calls = []
    original_assessment_for = analysis_service.assessment_for

    def offline_analysis(symbol, client, *, refresh=False):
        calls.append((symbol, refresh))
        return result

    def fixed_clock_assessment(data, config=None):
        return original_assessment_for(data, now=datetime(2026, 10, 12, 12, tzinfo=NEW_YORK), config=config)

    monkeypatch.setattr(analysis_service, "analyze", offline_analysis)
    monkeypatch.setattr(analysis_service, "assessment_for", fixed_clock_assessment)
    app = _new_app()
    _button(app, "分析").click().run(timeout=20)
    _settle(app)
    assert not app.exception
    initial = {metric.label: metric.value for metric in app.metric}
    assert initial["操作参考"] == "等待"
    assert initial["盘中判断"] == "不可用"
    app.toggle[0].set_value(True).run(timeout=20)
    assert not app.exception
    holding = {metric.label: metric.value for metric in app.metric}
    assert holding["操作参考"] == "持有观察"
    assert holding["日线判断"] == initial["日线判断"]
    assert holding["盘中判断"] == initial["盘中判断"]
    assert calls == [("NVDA", False)]
    app.toggle[0].set_value(False).run(timeout=20)
    assert not app.exception
    assert {metric.label: metric.value for metric in app.metric}["操作参考"] == "等待"
    assert calls == [("NVDA", False)]


def test_ticker_change_automatically_loads_without_analyze_click(monkeypatch):
    template = _daily_only_result()
    calls = []

    def offline_analysis(symbol, client, *, refresh=False):
        selected = validate_symbol(symbol)
        calls.append((selected, refresh))
        return replace(template, symbol=selected)

    monkeypatch.setattr(analysis_service, "analyze", offline_analysis)
    app = _new_app()
    assert calls == []
    app.text_input[0].set_value(" aapl ").run(timeout=20)
    _settle(app)
    assert not app.error
    assert calls == [("AAPL", False)]
    assert app.session_state["analysis_result"].symbol == "AAPL"
    assert any("AAPL · 市场快照" in item.value for item in app.subheader)
    assert app.session_state["ticker_requests"].current_symbol == "AAPL"
    app.text_input[0].set_value("MSFT").run(timeout=20)
    _settle(app)
    assert calls == [("AAPL", False), ("MSFT", False)]
    assert app.session_state["analysis_result"].symbol == "MSFT"
    assert any("MSFT · 市场快照" in item.value for item in app.subheader)


@pytest.mark.parametrize("stale_error", [False, True])
def test_rapid_ticker_switch_hides_old_results_and_ignores_stale_completion(monkeypatch, stale_error):
    template = _daily_only_result()
    started, release = Event(), Event()
    calls = []

    def offline_analysis(symbol, client, *, refresh=False):
        selected = validate_symbol(symbol)
        calls.append((selected, refresh))
        if selected == "AAPL":
            started.set()
            assert release.wait(15), "Slow test request was not released"
            if stale_error:
                raise RuntimeError("SECRET_PROVIDER_RESPONSE")
        return replace(template, symbol=selected)

    monkeypatch.setattr(analysis_service, "analyze", offline_analysis)
    app = _new_app()
    try:
        _button(app, "分析").click().run(timeout=20)
        _settle(app)
        app.toggle[0].set_value(True).run(timeout=20)
        app.text_input[0].set_value("AAPL").run(timeout=20)
        assert started.wait(1)
        requests = app.session_state["ticker_requests"]
        slow_id = requests.poll().request_id
        assert requests.poll().status == "loading"
        assert requests.poll().result is None
        assert len(app.get("plotly_chart")) == 0
        assert app.toggle[0].value is False
        app.text_input[0].set_value("MSFT").run(timeout=20)
        _settle(app)
        winning_id = requests.poll().request_id
        assert winning_id > slow_id
        assert app.session_state["analysis_result"].symbol == "MSFT"
        release.set()
        deadline = monotonic() + 3
        while monotonic() < deadline:
            with requests._lock:
                if not requests._inflight:
                    break
            Event().wait(0.005)
        else:
            pytest.fail("Stale request did not complete")
        app.run(timeout=20)
        assert not app.exception and not app.error
        assert app.session_state["analysis_result"].symbol == "MSFT"
        assert requests.poll().request_id == winning_id
        assert any("MSFT · 市场快照" in item.value for item in app.subheader)
        assert not any("AAPL · 市场快照" in item.value for item in app.subheader)
        assert calls == [("NVDA", False), ("AAPL", False), ("MSFT", False)]
    finally:
        release.set()


@pytest.mark.parametrize("retry_cache_valid", [False, True])
def test_new_daily_close_refreshes_automatically_and_preserves_controls(monkeypatch, retry_cache_valid):
    """A calendar close checks local data; provider-delay retry caches suppress I/O."""
    template = replace(_daily_only_result(), market_versions=("daily-old", None))
    epoch = {"value": pd.Timestamp("2026-10-09 16:00", tz=NEW_YORK)}
    next_close = pd.Timestamp("2026-10-12 16:00", tz=NEW_YORK)
    new_bar = template.daily.iloc[-1].to_dict()
    new_bar.update(timestamp=next_close.tz_convert("UTC"), bar_end=next_close.tz_convert("UTC"),
                   session=next_close.date())
    for column in ("open", "high", "low", "close"):
        new_bar[column] += 0.2
    updated_daily = daily_indicators(pd.concat([template.daily, pd.DataFrame([new_bar])], ignore_index=True))
    updated = replace(template, daily=updated_daily, confirmed_daily_session=next_close.date(),
                      market_versions=("daily-new", None),
                      snapshot=replace(template.snapshot, session=next_close.date(),
                                       market_timestamp=next_close.to_pydatetime(), price=new_bar["close"]))
    calls, peeks = [], []

    class OfflineClient:
        def peek_validated_daily(self, symbol, now=None):
            peeks.append(symbol)
            return object() if retry_cache_valid else None

    client = OfflineClient()

    def offline_analysis(symbol, received_client, *, refresh=False):
        assert received_client is client
        calls.append((symbol, refresh))
        return template if len(calls) == 1 else updated

    monkeypatch.setattr(fmp_client, "FMPClient", lambda settings: client)
    monkeypatch.setattr(market_store, "freshness_epoch", lambda timeframe, now: epoch["value"])
    monkeypatch.setattr(analysis_service, "analyze", offline_analysis)
    app = _new_app()
    _button(app, "分析").click().run(timeout=20)
    _settle(app)
    app.toggle[0].set_value(True).run(timeout=20)
    app.radio[0].set_value("1Y").run(timeout=20)
    app.selectbox(key="sr_window").set_value("6 个月 · 126 个交易日").run(timeout=20)
    app.selectbox(key="sr_count").set_value(5).run(timeout=20)
    app.checkbox[2].set_value(False).run(timeout=20)
    app.checkbox[3].set_value(True).run(timeout=20)
    assert not app.exception
    assert any("综合评分" in metric.label for metric in app.metric)
    assert calls == [("NVDA", False)] and peeks == []

    epoch["value"] = next_close
    app.run(timeout=20)
    _settle(app)
    assert not app.exception and not app.error
    assert app.text_input[0].value == "NVDA"
    assert app.toggle[0].value is True
    assert app.radio[0].value == "1Y"
    assert app.selectbox(key="sr_window").value == "6 个月 · 126 个交易日"
    assert app.selectbox(key="sr_count").value == 5
    assert app.checkbox[2].value is False
    assert peeks and set(peeks) == {"NVDA"}
    if retry_cache_valid:
        assert calls == [("NVDA", False)]
        assert app.session_state["analysis_result"].confirmed_daily_session == template.confirmed_daily_session
    else:
        assert calls == [("NVDA", False), ("NVDA", False)]
        assert app.session_state["analysis_result"].confirmed_daily_session == next_close.date()
    app.run(timeout=20)
    assert len(calls) == (1 if retry_cache_valid else 2)


def test_failed_daily_close_refresh_has_bounded_retry_cooldown(monkeypatch):
    template = replace(_daily_only_result(), market_versions=("daily-old", None))
    epoch = {"value": pd.Timestamp("2026-10-09 16:00", tz=NEW_YORK)}
    calls, peeks = [], []

    class OfflineClient:
        def peek_validated_daily(self, symbol, now=None):
            peeks.append(symbol)
            return None

    def offline_analysis(symbol, client, *, refresh=False):
        calls.append((symbol, refresh))
        if len(calls) > 1:
            raise DataError("行情服务暂时不可用。", "network_error")
        return template

    monkeypatch.setattr(fmp_client, "FMPClient", lambda settings: OfflineClient())
    monkeypatch.setattr(market_store, "freshness_epoch", lambda timeframe, now: epoch["value"])
    monkeypatch.setattr(analysis_service, "analyze", offline_analysis)
    app = _new_app()
    _button(app, "分析").click().run(timeout=20)
    _settle(app)
    epoch["value"] = pd.Timestamp("2026-10-12 16:00", tz=NEW_YORK)
    app.run(timeout=20)
    _settle(app)
    assert not app.exception
    assert any("行情服务暂时不可用" in item.value for item in app.error)
    assert calls == [("NVDA", False), ("NVDA", False)]
    request_id = app.session_state["ticker_requests"].poll().request_id
    for _ in range(3):
        app.run(timeout=20)
        assert not app.exception
    assert app.session_state["ticker_requests"].poll().request_id == request_id
    assert calls == [("NVDA", False), ("NVDA", False)]
    assert peeks == ["NVDA"]
    assert app.session_state["analysis_result"].confirmed_daily_session == template.confirmed_daily_session

    # Advance only the recorded retry clock, without real waits or network I/O.
    app.session_state["daily_close_attempt"] = ("NVDA", str(epoch["value"]), perf_counter() - 31)
    app.run(timeout=20)
    _settle(app)
    assert len(calls) == 3 and peeks == ["NVDA", "NVDA"]


def test_daily_close_uses_newer_shared_validated_cache_without_http(monkeypatch):
    """Another browser's completed candle updates this browser from shared data."""
    template = _daily_only_result()
    epoch = {"value": pd.Timestamp("2026-10-09 16:00", tz=NEW_YORK)}
    calls, http_calls = [], []

    class OfflineSession:
        def get(self, *args, **kwargs):
            http_calls.append(True)
            raise AssertionError("A warmed validated cache must not use HTTP")

    client = fmp_client.FMPClient(Settings(api_key="offline-test-key"), session=OfflineSession())

    def warm_cache(frame, completed_close):
        dataset = market_store.MarketDataset(
            frame=frame, issues=[], retrieved_at=completed_close.to_pydatetime(),
            endpoint="historical-price-eod/full", latest_session=completed_close.date(),
            version=market_store.content_version(frame, "daily"),
        )
        client._market_store.put("NVDA", "daily", market_store.ADJUSTMENT_POLICY, dataset,
                                 completed_close, completed_close.to_pydatetime(), 300)
        return dataset

    initial = warm_cache(template.daily, epoch["value"])

    def cached_analysis(symbol, received_client, *, refresh=False):
        assert received_client is client and refresh is False
        cached = received_client.fetch_validated_daily(symbol)
        assert cached.cache_hit and cached.api_seconds == 0
        calls.append((symbol, cached.version))
        last = cached.frame.iloc[-1]
        return replace(template, daily=daily_indicators(cached.frame),
                       confirmed_daily_session=cached.latest_session, market_versions=(cached.version, None),
                       snapshot=replace(template.snapshot, session=cached.latest_session,
                                        market_timestamp=last["bar_end"].to_pydatetime(), price=last["close"]))

    monkeypatch.setattr(fmp_client, "FMPClient", lambda settings: client)
    monkeypatch.setattr(market_store, "freshness_epoch", lambda timeframe, now: epoch["value"])
    monkeypatch.setattr(fmp_client, "freshness_epoch", lambda timeframe, now: epoch["value"])
    monkeypatch.setattr(analysis_service, "analyze", cached_analysis)
    app = _new_app()
    _button(app, "分析").click().run(timeout=20)
    _settle(app)
    assert calls == [("NVDA", initial.version)] and http_calls == []

    next_close = pd.Timestamp("2026-10-12 16:00", tz=NEW_YORK)
    new_bar = template.daily.iloc[-1].to_dict()
    new_bar.update(timestamp=next_close.tz_convert("UTC"), bar_end=next_close.tz_convert("UTC"),
                   session=next_close.date())
    for column in ("open", "high", "low", "close"):
        new_bar[column] += 0.2
    updated_frame = pd.concat([template.daily, pd.DataFrame([new_bar])], ignore_index=True)
    updated = warm_cache(updated_frame, next_close)
    epoch["value"] = next_close
    assert updated.version != initial.version
    assert client.peek_validated_daily("NVDA").version == updated.version
    app.run(timeout=20)
    _settle(app)
    assert not app.exception and not app.error
    assert calls == [("NVDA", initial.version), ("NVDA", updated.version)]
    assert app.session_state["analysis_result"].market_versions[0] == updated.version
    assert app.session_state["analysis_result"].confirmed_daily_session == next_close.date()
    app.run(timeout=20)
    assert len(calls) == 2 and http_calls == []


def _open_market_clock(monkeypatch):
    # Only the app's new import is patched; signal/normalization calendars remain real.
    def open_schedule(start, end):
        opening = pd.Timestamp(start, tz=NEW_YORK)
        return pd.DataFrame({"market_open": [opening],
                             "market_close": [opening + pd.Timedelta(days=1)]}, index=[start])
    monkeypatch.setattr(normalize, "session_schedule", open_schedule)


def test_auto_refresh_fetches_new_data_and_preserves_controls(monkeypatch):
    template = _daily_only_result()
    calls = []
    def offline_analysis(symbol, client, *, refresh=False):
        calls.append((symbol, refresh))
        return replace(template, snapshot=replace(template.snapshot, price=100 + len(calls)))
    monkeypatch.setattr(analysis_service, "analyze", offline_analysis)
    _open_market_clock(monkeypatch)
    app = _new_app()
    assert app.selectbox(key="auto_refresh_seconds").value == 60
    _button(app, "分析").click().run(timeout=20)
    _settle(app)
    app.toggle[0].set_value(True).run(timeout=20)
    app.radio[0].set_value("1Y").run(timeout=20)
    app.selectbox(key="sr_window").set_value("6 个月 · 126 个交易日").run(timeout=20)
    app.selectbox(key="sr_count").set_value(5).run(timeout=20)
    app.checkbox[2].set_value(False).run(timeout=20)
    assert calls == [("NVDA", False)]

    app.session_state["refresh_clock"] = perf_counter() - 61
    app.run(timeout=20)
    _settle(app)
    assert not app.exception and not app.error
    assert calls == [("NVDA", False), ("NVDA", True)]
    assert app.session_state["analysis_result"].snapshot.price == 102
    assert app.text_input[0].value == "NVDA"
    assert app.toggle[0].value is True and app.radio[0].value == "1Y"
    assert app.selectbox(key="sr_window").value == "6 个月 · 126 个交易日"
    assert app.selectbox(key="sr_count").value == 5 and app.checkbox[2].value is False
    app.run(timeout=20)
    assert len(calls) == 2


@pytest.mark.parametrize("mode", ["off", "closed", "interval"])
def test_auto_refresh_obeys_off_market_hours_and_interval(monkeypatch, mode):
    template = _daily_only_result()
    calls = []
    def offline_analysis(symbol, client, *, refresh=False):
        calls.append((symbol, refresh))
        return template
    monkeypatch.setattr(analysis_service, "analyze", offline_analysis)
    _open_market_clock(monkeypatch)
    app = _new_app()
    _button(app, "分析").click().run(timeout=20)
    _settle(app)
    if mode == "off":
        app.selectbox(key="auto_refresh_seconds").set_value(0).run(timeout=20)
        # Disabling also suppresses the existing missing-daily-close watcher.
        app.session_state["analysis_result"] = replace(template, market_versions=("old", None))
    elif mode == "closed":
        monkeypatch.setattr(normalize, "session_schedule", lambda start, end: pd.DataFrame())
    else:
        app.selectbox(key="auto_refresh_seconds").set_value(300).run(timeout=20)
    app.session_state["refresh_clock"] = perf_counter() - 61
    app.run(timeout=20)
    assert not app.exception
    assert calls == [("NVDA", False)]
    # Manual refresh works even when automatic requests are paused/disabled.
    _button(app, "↻ 刷新数据").click().run(timeout=20)
    _settle(app)
    assert calls == [("NVDA", False), ("NVDA", True)]


def test_auto_refresh_does_not_overlap_and_backs_off_after_failure(monkeypatch):
    template = _daily_only_result()
    calls, started, release = [], Event(), Event()
    def offline_analysis(symbol, client, *, refresh=False):
        calls.append((symbol, refresh))
        if len(calls) == 2:
            started.set()
            assert release.wait(15)
            raise DataError("行情服务暂时不可用。", "network_error")
        return template
    monkeypatch.setattr(analysis_service, "analyze", offline_analysis)
    _open_market_clock(monkeypatch)
    app = _new_app()
    try:
        _button(app, "分析").click().run(timeout=20)
        _settle(app)
        app.session_state["refresh_clock"] = perf_counter() - 61
        app.run(timeout=20)
        assert started.wait(1)
        app.session_state["refresh_clock"] = perf_counter() - 61
        app.run(timeout=20)
        assert len(calls) == 2
        release.set()
        _settle(app)
        assert app.error and not app.exception
        assert app.session_state["analysis_result"].snapshot == template.snapshot
        app.run(timeout=20)
        assert len(calls) == 2
        app.session_state["refresh_clock"] = perf_counter() - 61
        app.run(timeout=20)
        _settle(app)
        assert len(calls) == 3 and not app.error
    finally:
        release.set()


def test_assessment_expires_on_fragment_tick_even_with_fetching_off(monkeypatch):
    """A daily close that becomes stale withdraws the Buy without any new fetch."""
    from inspect import getclosurevars
    from streamlit.testing.v1 import local_script_runner

    template = _daily_only_result()
    daily = template.daily.copy()
    for index, close, sma20, sma50 in zip(daily.index[-2:], [100.5, 101], [99.8, 100], [94.8, 95]):
        daily.loc[index, ["open", "high", "low", "close", "sma20", "sma50", "rsi14"]] = [
            close, close + 0.5, close - 0.5, close, sma20, sma50, 60]
    daily.loc[daily.index[-1], ["atr14", "prev_high", "pullback_atr5", "high22", "updown_volume20", "sma200"]] = [
        2.0, 100.8, 0.2, 102.0, 1.4, 90.0]
    stamps = pd.DatetimeIndex(["2026-10-12 09:50", "2026-10-12 09:55"], tz=NEW_YORK)
    intraday = pd.DataFrame({
        "session": stamps.date, "timestamp": stamps, "bar_end": stamps + pd.Timedelta("5min"),
        "complete": True, "open": 101, "high": 101.5, "low": 100.5, "close": [101, 101.2],
        "volume": 1000, "vwap": [100.7, 100.8], "session_open": 100.6,
        "coverage_complete": True, "vwap_available": True,
    })
    clock = {"now": datetime(2026, 10, 12, 10, 2, tzinfo=NEW_YORK)}
    template = replace(template, daily=daily, intraday=intraday, intraday_error=None,
                       intraday_session=stamps[0].date(), daily_retrieved_at=clock["now"],
                       intraday_retrieved_at=clock["now"])
    calls = []
    def offline_analysis(symbol, client, *, refresh=False):
        calls.append(symbol)
        return template
    assess = analysis_service.assessment_for
    monkeypatch.setattr(analysis_service, "analyze", offline_analysis)
    monkeypatch.setattr(analysis_service, "assessment_for", lambda data, config=None: assess(data, now=clock["now"], config=config))
    app = _new_app()
    _button(app, "分析").click().run(timeout=20)
    _settle(app)
    app.selectbox(key="auto_refresh_seconds").set_value(0).run(timeout=20)
    assert next(metric.value for metric in app.metric if metric.label == "操作参考") == "买入"

    # AppTest does not schedule timers. Dispatch only the registered assessment
    # fragment through its runner, as the browser's periodic event would do.
    fragment_id = next(key for key, fn in app._fragment_storage._fragments.items()
                       if getclosurevars(fn).nonlocals["non_optional_func"].__name__ == "render_current_assessment")
    clock["now"] = datetime(2026, 10, 13, 16, 5, tzinfo=NEW_YORK)
    rerun_data = local_script_runner.RerunData
    with monkeypatch.context() as patch:
        patch.setattr(local_script_runner, "RerunData",
                      lambda **kwargs: rerun_data(**kwargs, fragment_id=fragment_id))
        app.run(timeout=20)
    assert not app.exception
    assert next(metric.value for metric in app.metric if metric.label == "操作参考") == "数据不足"
    assert calls == ["NVDA"]
