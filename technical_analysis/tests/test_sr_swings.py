"""Offline preparation, ATR, pivot-confirmation and clustering checks."""
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from analysis.support_resistance.models import Pivot, SRConfig
from analysis.support_resistance.prepare import prepare_daily
from analysis.support_resistance.swings import cluster_pivots, detect_pivots
from data.normalize import session_schedule


def daily_bars(count=100, *, scale=1.0, volume=1000.0):
    schedule = session_schedule("2024-01-02", "2026-12-31").iloc[:count]
    close = (100 + 4 * np.sin(np.arange(count) * np.pi / 7)) * scale
    return pd.DataFrame({
        "session": schedule.index.tolist(),
        "timestamp": schedule["market_close"].tolist(),
        "bar_end": schedule["market_close"].tolist(),
        "complete": True,
        "open": close,
        "high": close + 0.5 * scale,
        "low": close - 0.5 * scale,
        "close": close,
        "volume": volume,
    })


def pivot_frame():
    frame = daily_bars(18)
    frame["high"] = 11.0
    frame["low"] = 10.0
    frame["atr"] = 1.0
    frame["segment_id"] = 0
    frame.loc[6, "high"] = 15.0
    frame.loc[11, "low"] = 5.0
    frame.loc[16, "high"] = 20.0
    return frame


def test_full_history_wilder_seed_and_recurrence_before_window_crop():
    frame = daily_bars(320)
    config = SRConfig()
    prepared = prepare_daily(frame, config)
    recent = prepare_daily(frame, replace(config, lookback_sessions=126))
    assert prepared.available and recent.available
    assert len(prepared.frame) == 252
    assert len(recent.frame) == 126
    assert prepared.frame.index.equals(pd.RangeIndex(252))
    assert prepared.latest_atr == recent.latest_atr
    assert recent.frame["atr"].notna().all()
    full = prepare_daily(frame, replace(config, lookback_sessions=400)).frame
    true_ranges = [float(frame.iloc[0]["high"] - frame.iloc[0]["low"])]
    for index in range(1, len(frame)):
        row, previous = frame.iloc[index], frame.iloc[index - 1]
        true_ranges.append(max(row.high - row.low, abs(row.high - previous.close), abs(row.low - previous.close)))
    assert full.iloc[:13]["atr"].isna().all()
    assert full.iloc[13]["atr"] == pytest.approx(np.mean(true_ranges[:14]))
    assert full.iloc[14]["atr"] == pytest.approx((full.iloc[13]["atr"] * 13 + true_ranges[14]) / 14)


def test_missing_session_resets_atr_and_preserves_unfilled_calendar_gap():
    source = daily_bars(110)
    missing_date = source.iloc[70]["session"]
    result = prepare_daily(source.drop(index=70), SRConfig())
    assert result.available
    assert result.missing_sessions == 1
    assert missing_date not in result.frame["session"].tolist()
    assert result.frame.iloc[70]["missing_before"] == 1
    assert bool(result.frame.iloc[70]["gap_before"])
    assert result.frame.iloc[69]["segment_id"] == 0
    assert result.frame.iloc[70]["segment_id"] == 1
    assert result.frame.iloc[70:83]["atr"].isna().all()
    assert np.isfinite(result.frame.iloc[83]["atr"])
    assert any("不填补价格" in warning for warning in result.warnings)


def test_gap_at_end_withholds_positive_atr_instead_of_bridging_history():
    source = daily_bars(100).drop(index=90)
    result = prepare_daily(source, SRConfig())
    assert not result.available
    assert result.latest_atr is None
    assert "连续" in result.reason
    assert result.frame.iloc[-1]["atr"] != result.frame.iloc[-1]["atr"]


def test_sorting_exact_deduplication_and_conflicting_date_removal():
    source = daily_bars(100)
    duplicate = source.iloc[[20]].copy()
    duplicate["sma20"] = 321.0
    conflict = source.iloc[[40]].copy()
    conflict["close"] += 0.1
    dirty = pd.concat([source, duplicate, conflict]).sample(frac=1, random_state=3)
    result = prepare_daily(dirty, SRConfig())
    assert result.available
    assert len(result.frame) == 99
    assert result.frame["session"].is_monotonic_increasing
    assert source.iloc[40]["session"] not in result.frame["session"].tolist()
    assert result.frame["session"].tolist().count(source.iloc[20]["session"]) == 1
    assert any("完全相同" in warning for warning in result.warnings)
    assert any("冲突" in warning for warning in result.warnings)
    assert result.missing_sessions == 1


@pytest.mark.parametrize("flag", [False, np.nan, "True", 1, None])
def test_forming_or_invalid_completion_flag_cannot_confirm_a_bar(flag):
    source = daily_bars(100)
    source["complete"] = source["complete"].astype(object)
    source.loc[99, "complete"] = flag
    source.loc[99, ["open", "high", "low", "close"]] = [10000, 10000, 10000, 10000]
    result = prepare_daily(source, SRConfig())
    assert result.available
    assert len(result.frame) == 99
    assert result.frame.iloc[-1]["session"] == source.iloc[-2]["session"]
    assert result.latest_atr < 10


@pytest.mark.parametrize("field,value", [
    ("close", np.nan), ("close", np.inf), ("close", 0),
    ("high", 1), ("low", 1000), ("volume", -1), ("volume", True),
])
def test_invalid_ohlcv_is_rejected_without_mutating_source(field, value):
    source = daily_bars(100)
    source[field] = source[field].astype(object)
    source.loc[50, field] = value
    before = source.copy(deep=True)
    result = prepare_daily(source, SRConfig())
    assert len(result.frame) == 99
    assert source.iloc[50]["session"] not in result.frame["session"].tolist()
    assert result.missing_sessions == 1
    pd.testing.assert_frame_equal(source, before)


def test_exchange_calendar_and_scheduled_close_are_checked():
    source = daily_bars(100)
    saturday = source.iloc[[0]].copy()
    saturday["session"] = date(2024, 1, 6)
    saturday["timestamp"] = pd.Timestamp("2024-01-06 16:00", tz="America/New_York")
    saturday["bar_end"] = saturday["timestamp"]
    source.loc[20, "bar_end"] = source.loc[20, "bar_end"] - pd.Timedelta("1h")
    result = prepare_daily(pd.concat([source, saturday]), SRConfig())
    assert len(result.frame) == 99
    assert any("非 NYSE" in warning for warning in result.warnings)
    assert any("收盘时间" in warning for warning in result.warnings)


def test_utc_and_timezone_aware_session_fields_are_supported():
    source = daily_bars(100)
    expected = prepare_daily(source, SRConfig())
    source["session"] = source["timestamp"].dt.tz_convert("UTC")
    source["timestamp"] = source["timestamp"].dt.tz_convert("UTC")
    source["bar_end"] = source["bar_end"].dt.tz_convert("UTC")
    actual = prepare_daily(source, SRConfig())
    assert actual.available
    assert actual.frame["session"].tolist() == expected.frame["session"].tolist()
    assert str(actual.frame["timestamp"].dt.tz) == "America/New_York"
    assert actual.latest_atr == expected.latest_atr


def test_insufficient_history_and_flat_zero_atr_are_explicit():
    short = prepare_daily(daily_bars(59), SRConfig())
    assert not short.available
    assert "至少 60" in short.reason
    assert any("部分历史窗口" in warning for warning in short.warnings)
    flat = daily_bars(100)
    flat[["open", "high", "low", "close"]] = 100.0
    result = prepare_daily(flat, SRConfig())
    assert not result.available
    assert result.latest_atr is None
    assert "固定金额" in result.reason
    assert result.frame.iloc[-1]["atr"] == 0


def test_zero_volume_is_valid_price_history_and_missing_fields_are_unavailable():
    assert prepare_daily(daily_bars(volume=0), SRConfig()).available
    result = prepare_daily(daily_bars().drop(columns="bar_end"), SRConfig())
    assert not result.available
    assert "缺少必要" in result.reason
    assert {"atr", "segment_id", "gap_before", "missing_before"}.issubset(result.frame.columns)


def test_large_close_jump_warns_without_inventing_split_or_adjusting_ohlc():
    source = daily_bars(100)
    source.loc[70:, ["open", "high", "low", "close"]] *= 0.5
    result = prepare_daily(source, SRConfig())
    assert result.available
    assert result.frame.iloc[70]["close"] == source.iloc[70]["close"]
    assert result.frame["segment_id"].nunique() == 1
    assert result.frame.iloc[70]["atr"] > result.frame.iloc[69]["atr"]
    assert any("可能涉及价格调整" in warning and "不推测拆股比例" in warning for warning in result.warnings)


def test_confirmed_pivots_wait_for_right_window_and_exclude_latest_three_bars():
    frame = pivot_frame()
    config = SRConfig()
    assert not any(pivot.index == 6 for pivot in detect_pivots(frame.iloc[:9], config))
    confirmed = detect_pivots(frame.iloc[:10], config)
    high = next(pivot for pivot in confirmed if pivot.index == 6 and pivot.kind == "high")
    assert high.confirmed_index == 9
    assert high.session == frame.iloc[6]["session"]
    assert high.confirmed_session == frame.iloc[9]["session"]
    all_pivots = detect_pivots(frame, config)
    assert {pivot.kind for pivot in all_pivots} == {"high", "low"}
    assert not any(pivot.index == 16 for pivot in all_pivots)


def test_tied_extrema_missing_atr_and_calendar_gaps_cannot_create_pivots():
    frame = pivot_frame()
    frame.loc[7, "high"] = 15.0
    assert not any(pivot.index == 6 for pivot in detect_pivots(frame, SRConfig()))
    frame = pivot_frame()
    frame.loc[6, "atr"] = np.nan
    assert not any(pivot.index == 6 for pivot in detect_pivots(frame, SRConfig()))
    frame = pivot_frame()
    frame.loc[7:, "segment_id"] = 1
    assert not any(pivot.index == 6 for pivot in detect_pivots(frame, SRConfig()))


def test_significance_requires_reaction_on_both_sides_and_completed_windows():
    frame = pivot_frame()
    frame.loc[3:5, "low"] = 14.8
    assert not any(pivot.index == 6 for pivot in detect_pivots(frame, SRConfig()))
    frame = pivot_frame()
    frame.loc[8, "complete"] = False
    assert not any(pivot.index == 6 for pivot in detect_pivots(frame, SRConfig()))


def test_outside_bar_can_supply_both_pivot_kinds_with_one_swing_method_family():
    frame = pivot_frame()
    frame.loc[6, "low"] = 5.0
    pivots = detect_pivots(frame, SRConfig())
    both = [pivot for pivot in pivots if pivot.index == 6]
    assert {pivot.kind for pivot in both} == {"high", "low"}
    assert len({pivot.pivot_id for pivot in both}) == 2
    group = tuple([make_pivot(20, 15.1, "high"), *both])
    candidates = cluster_pivots(group, 1.0, SRConfig())
    assert len(candidates) == 1
    assert candidates[0].methods == ("swing",)


def make_pivot(index, price, kind="low"):
    sessions = daily_bars(100)["session"].tolist()
    return Pivot(f"{kind}-{index}", kind, index, index + 3, sessions[index],
                 sessions[index + 3], price, 1.0, 2.0)


def test_complete_link_clustering_prevents_price_chain_drift():
    pivots = tuple(make_pivot(index, price) for index, price in ((10, 100.0), (20, 100.7), (30, 101.4), (40, 101.5)))
    config = SRConfig()
    candidates = cluster_pivots(pivots, 1.0, config)
    assert len(candidates) == 2
    assert candidates[0].lower == pytest.approx(99.8)
    assert candidates[0].upper == pytest.approx(100.9)
    assert candidates[0].center == pytest.approx(100.35)
    assert candidates[0].methods == ("swing",)
    assert candidates[0].available_index == 23
    assert candidates[0].available_index != pivots[0].confirmed_index
    assert "不等于独立测试次数" in candidates[0].evidence[0]
    assert str(pivots[0].session) in candidates[0].evidence[0]
    assert str(pivots[0].confirmed_session) in candidates[0].evidence[0]


def test_single_pivots_are_only_anchors_and_duplicate_ids_do_not_create_zones():
    pivot = make_pivot(10, 100)
    assert cluster_pivots((pivot,), 1.0, SRConfig()) == ()
    assert cluster_pivots((pivot, pivot), 1.0, SRConfig()) == ()
    assert cluster_pivots((pivot, make_pivot(20, 100.1)), 0, SRConfig()) == ()


def test_final_cluster_geometry_is_available_only_after_last_member_confirms():
    early = make_pivot(10, 100.0)
    middle = make_pivot(20, 100.1)
    late = make_pivot(30, 100.7)
    initial = cluster_pivots((early, middle), 1.0, SRConfig())[0]
    final = cluster_pivots((early, middle, late), 1.0, SRConfig())[0]
    assert final.upper > initial.upper
    assert initial.available_index == middle.confirmed_index
    assert final.available_index == late.confirmed_index
    assert final.available_index > initial.available_index > early.confirmed_index


def test_latest_atr_cannot_change_existing_historical_cluster_geometry():
    pivots = (make_pivot(10, 100.0), make_pivot(20, 100.7),
              make_pivot(30, 101.4), make_pivot(40, 101.5))
    small_latest_atr = cluster_pivots(pivots, 0.1, SRConfig())
    large_latest_atr = cluster_pivots(pivots, 50.0, SRConfig())
    assert len(small_latest_atr) == 2
    assert small_latest_atr == large_latest_atr
    assert any("历史 ATR 中位数" in text for text in small_latest_atr[0].evidence)


def test_cluster_distance_and_padding_use_proposed_constituent_median_atr():
    first = make_pivot(10, 100.0)
    second = replace(make_pivot(20, 101.2), atr=3.0)
    third = make_pivot(30, 102.0)
    zones = cluster_pivots((first, second, third), 100.0, SRConfig())
    assert len(zones) == 1
    # The two-member median is 2 ATR, allowing a 1.2 price span. Adding the
    # third would reduce the median to 1 ATR, so the 2.0 span is rejected.
    assert zones[0].pivot_ids == (first.pivot_id, second.pivot_id)
    assert zones[0].lower == pytest.approx(99.6)
    assert zones[0].upper == pytest.approx(101.6)


def test_adjusted_price_scale_invariance_for_atr_pivots_and_clusters():
    config = SRConfig()
    original = prepare_daily(daily_bars(180), config)
    adjusted = prepare_daily(daily_bars(180, scale=0.1), config)
    assert original.available and adjusted.available
    assert adjusted.latest_atr == pytest.approx(original.latest_atr * 0.1)
    original_pivots = detect_pivots(original.frame, config)
    adjusted_pivots = detect_pivots(adjusted.frame, config)
    assert len(original_pivots) > 4
    assert [(pivot.kind, pivot.index, pivot.confirmed_index) for pivot in original_pivots] == [
        (pivot.kind, pivot.index, pivot.confirmed_index) for pivot in adjusted_pivots]
    assert [pivot.reaction_atr for pivot in adjusted_pivots] == pytest.approx([pivot.reaction_atr for pivot in original_pivots])
    zones = cluster_pivots(original_pivots, original.latest_atr, config)
    adjusted_zones = cluster_pivots(adjusted_pivots, adjusted.latest_atr, config)
    assert len(zones) >= 2
    assert len(zones) == len(adjusted_zones)
    assert [zone.lower for zone in adjusted_zones] == pytest.approx([zone.lower * 0.1 for zone in zones])
    assert [zone.upper for zone in adjusted_zones] == pytest.approx([zone.upper * 0.1 for zone in zones])
    assert cluster_pivots(tuple(reversed(original_pivots)), original.latest_atr, config) == zones
