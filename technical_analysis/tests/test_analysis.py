"""Focused numerical and confirmation tests. All fixture prices are test-only."""
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from analysis.indicators import daily_indicators, session_vwap, visible_daily, wilder_rsi
from analysis.summary import generate_summaries
from data.normalize import normalize_intraday, session_schedule
from models import DataError, RawDataset
from services.analysis_service import analyze

NY = ZoneInfo("America/New_York")


def _raw_bar(stamp, close=100.0, volume=100):
    return {"date": str(stamp), "open": close, "high": close + 1, "low": close - 1, "close": close, "volume": volume}


def _intraday(stamps, prices, volumes=None):
    frame = pd.DataFrame({
        "timestamp": pd.DatetimeIndex(stamps),
        "open": prices,
        "high": np.array(prices) + 1,
        "low": np.array(prices) - 1,
        "close": prices,
        "volume": volumes or [10] * len(prices),
    })
    frame["session"] = frame["timestamp"].dt.date
    frame["bar_end"] = frame["timestamp"] + pd.Timedelta("5min")
    frame["complete"] = True
    return frame


def test_rsi_matches_wilder_reference_after_arithmetic_seed():
    closes = pd.Series([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.42, 45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00, 46.03, 46.41, 46.22])
    actual = wilder_rsi(closes)
    assert actual.iloc[:14].isna().all()
    assert actual.iloc[14:].tolist() == pytest.approx([70.46413502, 66.24961855, 66.48094183, 69.34685316, 66.29471266], abs=1e-7)


@pytest.mark.parametrize("closes,expected", [(list(range(15)), 100), (list(range(15, 0, -1)), 0), ([20] * 15, 50)])
def test_rsi_one_sided_and_flat_states(closes, expected):
    assert wilder_rsi(pd.Series(closes)).iloc[-1] == expected


def test_missing_close_resets_rsi_and_requires_full_new_seed():
    actual = wilder_rsi(pd.Series(list(range(20)) + [np.nan] + list(range(15))))
    assert actual.iloc[20:35].isna().all()
    assert actual.iloc[35] == 100


def test_vwap_volume_weights_and_resets_at_session_open():
    stamps = [pd.Timestamp("2025-05-01 09:30", tz=NY), pd.Timestamp("2025-05-01 09:35", tz=NY), pd.Timestamp("2025-05-02 09:30", tz=NY)]
    frame = _intraday(stamps, [100, 110, 200], [10, 30, 5])
    result = session_vwap(frame)
    assert result["vwap"].tolist() == pytest.approx([100, 107.5, 200])
    assert result["session_volume"].tolist() == [10, 40, 5]


@pytest.mark.parametrize("times", [["09:35", "09:40"], ["09:30", "09:40"]])
def test_partial_session_withholds_entire_vwap_open_and_volume(times):
    frame = _intraday([pd.Timestamp(f"2025-05-01 {time}", tz=NY) for time in times], [100, 101])
    result = session_vwap(frame)
    assert result[["vwap", "session_open", "session_volume"]].isna().all().all()
    assert not result["coverage_complete"].any()


def test_missing_elapsed_tail_withholds_vwap_and_zero_volume_is_unavailable():
    records = [_raw_bar("2025-05-01 09:30:00"), _raw_bar("2025-05-01 09:35:00")]
    normalized = normalize_intraday(records, now=datetime(2025, 5, 1, 10, 0, tzinfo=NY))
    assert session_vwap(normalized.frame)["vwap"].isna().all()
    zero = _intraday([pd.Timestamp("2025-05-01 09:30", tz=NY)], [100], [0])
    assert pd.isna(session_vwap(zero).iloc[0]["vwap"])


def test_daily_display_crop_never_recalculates_indicators():
    stamps = pd.date_range("2024-01-01", "2025-05-02", freq="B", tz=NY)
    frame = pd.DataFrame({"session": stamps.date, "close": 100 + np.arange(len(stamps)) / 10})
    calculated = daily_indicators(frame)
    ranges = {name: visible_daily(calculated, name) for name in ("3M", "6M", "1Y")}
    for name in ("6M", "1Y"):
        overlap = ranges[name].loc[ranges["3M"].index]
        pd.testing.assert_frame_equal(overlap, ranges["3M"])
    assert ranges["3M"].iloc[0]["session"] >= date(2025, 2, 2)


def test_summaries_use_confirmed_daily_and_state_numeric_conflict():
    daily = pd.DataFrame([
        {"session": date(2025, 5, 1), "complete": True, "close": 119, "sma20": 109, "sma50": 99, "rsi14": 68},
        {"session": date(2025, 5, 2), "complete": True, "close": 120, "sma20": 110, "sma50": 100, "rsi14": 72},
        {"session": date(2025, 5, 5), "complete": False, "close": 50, "sma20": 105, "sma50": 99, "rsi14": 10},
    ])
    intraday = session_vwap(_intraday([pd.Timestamp("2025-05-05 09:30", tz=NY), pd.Timestamp("2025-05-05 09:35", tz=NY)], [110, 100]))
    items = {item.rule_id: item for item in generate_summaries(daily, intraday, previous_close=120, previous_close_session=date(2025, 5, 2))}
    assert "2025-05-02" in items["daily.order_and_slopes"].text
    assert "收盘 120.00" in items["daily.order_and_slopes"].text
    assert "ΔSMA20=+1.0000" in items["daily.order_and_slopes"].evidence
    assert "72.00" in items["rsi.bands_and_change"].text
    assert "上升 4.00" in items["rsi.bands_and_change"].text
    assert "必然反转" in items["rsi.bands_and_change"].text
    assert "50.00" in items["daily.provisional"].text
    assert "方向冲突" in items["agreement.direction"].text


class FakeClient:
    def __init__(self, daily, intraday, retrieved_at, unavailable=False):
        self.daily, self.intraday, self.retrieved_at = daily, intraday, retrieved_at
        self.unavailable = unavailable

    def fetch_daily(self, symbol, now=None, refresh=False):
        return RawDataset(self.daily, self.retrieved_at, "test-daily")

    def fetch_intraday(self, symbol, now=None, refresh=False):
        if self.unavailable:
            raise DataError("5 分钟数据权限不足。", "permission_denied")
        return RawDataset(self.intraday, self.retrieved_at, "test-intraday")


def test_cache_clock_passage_cannot_confirm_forming_daily_or_intraday():
    retrieved = datetime(2025, 5, 5, 10, 3, tzinfo=NY)
    daily = [_raw_bar("2025-05-02", 100), _raw_bar("2025-05-05", 140)]
    stamps = pd.date_range("2025-05-05 09:30", "2025-05-05 10:00", freq="5min")
    client = FakeClient(daily, [_raw_bar(stamp, 110) for stamp in stamps], retrieved)
    result = analyze("TEST", client, now=datetime(2025, 5, 5, 16, 30, tzinfo=NY))
    assert not bool(result.daily.iloc[-1]["complete"])
    assert not bool(result.intraday.iloc[-1]["complete"])
    assert result.confirmed_daily_session == date(2025, 5, 2)
    assert result.snapshot.provisional
    assert result.snapshot.market_timestamp.hour == 10
    summary = next(item for item in result.summaries if item.rule_id == "intraday.completed_vs_vwap")
    assert "10:00" in summary.text  # last completed interval ends at 10:00


def test_daily_stays_usable_when_intraday_permissions_fail():
    client = FakeClient([_raw_bar("2025-05-02", 100)], [], datetime(2025, 5, 5, 10, 3, tzinfo=NY), unavailable=True)
    result = analyze("TEST", client, now=datetime(2025, 5, 5, 10, 3, tzinfo=NY))
    assert not result.daily.empty
    assert result.intraday.empty
    assert result.intraday_error == "5 分钟数据权限不足。"
    assert result.snapshot.price == 100
    assert result.assessment is not None
    assert result.assessment.daily.code == "insufficient_data"
    assert result.assessment.intraday.code == "unavailable"


def test_exact_previous_trading_session_is_required_for_percentage():
    # Monday May 26 is Memorial Day: Tuesday May 27's previous session is Friday.
    client = FakeClient([_raw_bar("2025-05-22", 90), _raw_bar("2025-05-27", 100)], [], datetime(2025, 5, 27, 17, 0, tzinfo=NY))
    result = analyze("TEST", client, now=datetime(2025, 5, 27, 17, 0, tzinfo=NY))
    assert result.snapshot.change_percent is None
    assert result.snapshot.session == date(2025, 5, 27)


def test_missing_daily_session_does_not_bridge_indicator_history():
    sessions = list(session_schedule(date(2025, 1, 2), date(2025, 5, 2)).index)
    missing = sessions[-3]
    records = [_raw_bar(session, 100 + index) for index, session in enumerate(sessions) if session != missing]
    client = FakeClient(records, [], datetime(2025, 5, 2, 17, 0, tzinfo=NY))
    result = analyze("TEST", client, now=datetime(2025, 5, 2, 17, 0, tzinfo=NY))
    assert len(result.daily) == len(records)
    assert result.daily.iloc[-1][["sma20", "sma50", "rsi14"]].isna().all()
    assert any("重新预热" in warning for warning in result.warnings)
