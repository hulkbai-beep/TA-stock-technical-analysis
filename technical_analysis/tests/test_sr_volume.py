"""Numerical daily-profile/AVWAP fixtures; these are not trading backtests."""
from datetime import date, timedelta
import math

import numpy as np
import pandas as pd
import pytest

from analysis.support_resistance.avwap import anchored_vwaps
from analysis.support_resistance.models import Pivot, SRConfig
from analysis.support_resistance.volume_profile import estimated_volume_profile, profile_candidates


def _frame(ranges, volumes):
    return pd.DataFrame({
        "high": [high for low, high in ranges],
        "low": [low for low, high in ranges],
        "close": [(low + high) / 2 for low, high in ranges],
        "volume": volumes,
        "session": [date(2025, 1, 2) + timedelta(days=index) for index in range(len(ranges))],
        "segment_id": 0,
    })


def _pivot(frame, kind="low", index=0, confirmed=1):
    return Pivot(f"{kind}.{index}", kind, index, confirmed, frame.iloc[index]["session"],
                 frame.iloc[confirmed]["session"], float(frame.iloc[index]["close"]), 1.0, 2.0)


def test_profile_exact_overlap_mass_poc_nodes_and_contiguous_value_area():
    frame = _frame([(10, 20), (20, 30), (15, 15)], [100, 300, 50])
    profile = estimated_volume_profile(frame, SRConfig(volume_bin_strategy="fixed_count", volume_bins=20))
    assert profile.available
    assert [item.volume for item in profile.bins] == pytest.approx([10] * 5 + [60] + [10] * 4 + [30] * 10)
    assert math.fsum(item.volume for item in profile.bins) == pytest.approx(450)
    assert math.fsum(item.fraction for item in profile.bins) == pytest.approx(1)
    assert profile.poc == 15.5
    assert [(node.lower, node.upper, node.volume) for node in profile.hvn] == [(15, 16, 60)]
    assert [(node.lower, node.upper, node.volume) for node in profile.lvn] == [(16, 20, 40)]
    assert profile.hvn[0].relative_density == pytest.approx(60 / 22.5)
    assert profile.hvn[0].fraction == pytest.approx(60 / 450)
    assert profile.val == 10
    assert profile.vah == 26
    assert profile.value_area_fraction == pytest.approx(330 / 450)
    candidates = profile_candidates(profile, available_index=42)
    assert len(candidates) == 1
    assert candidates[0].available_index == 42
    assert candidates[0].methods == ("volume_profile",)
    assert "HVN" in candidates[0].evidence[0]


def test_uniform_distribution_poc_tie_uses_lower_price_without_forcing_hvn():
    profile = estimated_volume_profile(_frame([(10, 30)], [200]), SRConfig(volume_bin_strategy="fixed_count", volume_bins=20))
    assert profile.poc == 10.5
    assert not profile.hvn
    assert not profile.lvn
    assert profile.val == 10
    assert profile.vah == 24
    assert profile.value_area_fraction == pytest.approx(0.7)
    assert not profile_candidates(profile, 0)


def test_overnight_gap_receives_no_volume_and_internal_hole_is_lvn():
    frame = _frame([(10, 12), (28, 30)], [100, 100])
    profile = estimated_volume_profile(frame, SRConfig(volume_bin_strategy="fixed_count", volume_bins=20))
    assert all(item.volume == 0 for item in profile.bins if 12 <= item.lower and item.upper <= 28)
    assert [(node.lower, node.upper, node.volume) for node in profile.lvn] == [(12, 28, 0)]
    assert math.fsum(item.volume for item in profile.bins) == pytest.approx(200)


def test_point_bar_at_upper_bound_uses_one_actual_containing_bin():
    profile = estimated_volume_profile(_frame([(10, 30), (30, 30)], [0, 80]), SRConfig(volume_bin_strategy="fixed_count", volume_bins=20))
    assert [item.volume for item in profile.bins] == [0] * 19 + [80]
    assert profile.poc == 29.5
    assert profile.vah == 30
    assert profile.val == 29
    assert profile.value_area_fraction == 1
    assert not profile.lvn  # Boundary low-density tails are not internal nodes.


def test_flat_price_profile_does_not_fabricate_a_price_range():
    profile = estimated_volume_profile(_frame([(42, 42)] * 3, [0, 5, 15]), SRConfig(), latest_atr=0)
    assert profile.available
    assert len(profile.bins) == 1
    assert profile.bins[0].lower == profile.bins[0].upper == 42
    assert profile.poc == profile.val == profile.vah == 42
    assert profile.total_volume == 20
    assert profile.value_area_fraction == 1
    assert profile.warnings
    assert not profile.hvn and not profile.lvn


def test_all_zero_volume_is_explicitly_unavailable():
    profile = estimated_volume_profile(_frame([(10, 20), (10, 30)], [0, 0]), SRConfig(), 2)
    assert not profile.available
    assert profile.warnings
    assert not profile_candidates(profile, 1)


def test_very_low_volume_is_computable_and_scaled_volume_changes_no_price_statistics():
    frame = _frame([(10, 20), (20, 30), (15, 15)], [1e-8, 3e-8, 5e-9])
    config = SRConfig(volume_bin_strategy="fixed_count", volume_bins=20)
    tiny = estimated_volume_profile(frame, config)
    frame["volume"] *= 1e10
    large = estimated_volume_profile(frame, config)
    assert tiny.available
    assert tiny.total_volume == pytest.approx(4.5e-8, rel=1e-12, abs=0)
    assert [item.fraction for item in tiny.bins] == pytest.approx([item.fraction for item in large.bins])
    assert (tiny.poc, tiny.val, tiny.vah) == (large.poc, large.val, large.vah)
    assert [(item.lower, item.upper) for item in tiny.hvn] == [(item.lower, item.upper) for item in large.hvn]


@pytest.mark.parametrize("atr,expected", [(10, 20), (2, 20), (0.2, 120)])
def test_atr_bin_count_has_deterministic_bounds(atr, expected):
    profile = estimated_volume_profile(_frame([(10, 30)], [100]), SRConfig(), atr)
    assert len(profile.bins) == expected
    assert profile.bins[0].lower == 10
    assert profile.bins[-1].upper == 30


def test_missing_atr_uses_documented_fixed_count_and_preserves_mass():
    profile = estimated_volume_profile(_frame([(10.01, 17.31), (12.99, 20.07)], [123.45, 678.9]), SRConfig())
    assert len(profile.bins) == 60
    assert profile.warnings
    assert math.fsum(item.volume for item in profile.bins) == pytest.approx(802.35)
    assert profile.value_area_fraction >= 0.7


def test_fractional_range_overlap_is_uniform_density_with_exact_mass():
    profile = estimated_volume_profile(_frame([(10, 30), (10.25, 11.25)], [0, 100]), SRConfig(volume_bin_strategy="fixed_count", volume_bins=20))
    assert [item.volume for item in profile.bins] == pytest.approx([75, 25] + [0] * 18)
    assert profile.total_volume == 100


def test_invalid_profile_input_is_unavailable_instead_of_silently_dropped():
    frame = _frame([(10, 20)], [100])
    frame.loc[0, "volume"] = np.nan
    assert not estimated_volume_profile(frame, SRConfig()).available


def test_avwap_matches_manual_hlc3_weights_and_persists_across_sessions():
    frame = _frame([(9, 12), (18, 24), (27, 36)], [10, 30, 10])
    frame["close"] = [12, 24, 36]
    anchor = anchored_vwaps(frame, (_pivot(frame),), SRConfig())[0]
    assert [point.value for point in anchor.points] == pytest.approx([19.25, 22])
    assert anchor.latest_value == 22  # (11*10 + 22*30 + 33*10) / 50
    assert anchor.cumulative_volume == 50
    assert anchor.anchor_session == frame.iloc[0]["session"]
    assert anchor.confirmed_session == frame.iloc[1]["session"]
    assert "日线近似" in anchor.label and "确认" in anchor.label


def test_avwap_points_never_backfill_before_confirmation():
    frame = _frame([(10, 12)] * 6, [10] * 6)
    pivot = _pivot(frame, index=1, confirmed=4)
    assert anchored_vwaps(frame.iloc[:4], (pivot,), SRConfig()) == ()
    confirmed = anchored_vwaps(frame.iloc[:5], (pivot,), SRConfig())[0]
    assert [point.session for point in confirmed.points] == [frame.iloc[4]["session"]]
    assert confirmed.cumulative_volume == 40
    extended = anchored_vwaps(frame, (pivot,), SRConfig())[0]
    assert extended.points[:1] == confirmed.points
    assert extended.cumulative_volume == 50


def test_avwap_selects_latest_low_high_and_configurable_one_anchor():
    frame = _frame([(10, 12)] * 10, [10] * 10)
    pivots = (_pivot(frame, "low", 0, 2), _pivot(frame, "low", 3, 5), _pivot(frame, "high", 5, 7))
    anchors = anchored_vwaps(frame, pivots, SRConfig())
    assert [anchor.anchor_id for anchor in anchors] == ["high.5", "low.3"]
    assert len(anchored_vwaps(frame, pivots, SRConfig(max_anchors=1))) == 1
    assert anchored_vwaps(frame, pivots, SRConfig(max_anchors=0)) == ()


def test_avwap_initial_zero_volume_waits_for_positive_volume_and_later_zero_keeps_value():
    frame = _frame([(10, 10), (20, 20), (30, 30), (100, 100)], [0, 0, 5, 0])
    anchor = anchored_vwaps(frame, (_pivot(frame),), SRConfig())[0]
    assert [point.session for point in anchor.points] == [frame.iloc[2]["session"], frame.iloc[3]["session"]]
    assert [point.value for point in anchor.points] == [30, 30]
    frame["volume"] = 0
    assert anchored_vwaps(frame, (_pivot(frame),), SRConfig()) == ()


def test_avwap_withholds_anchor_crossing_missing_session_segment():
    frame = _frame([(10, 10), (20, 20), (30, 30), (40, 40)], [10] * 4)
    frame["segment_id"] = [0, 0, 1, 1]
    assert anchored_vwaps(frame, (_pivot(frame),), SRConfig()) == ()
    later = _pivot(frame, "high", 2, 3)
    assert anchored_vwaps(frame, (later,), SRConfig())[0].latest_value == 35


def test_split_adjusted_price_and_volume_scale_preserves_profile_and_avwap_structure():
    frame = _frame([(100, 120), (120, 140), (110, 110), (130, 150)], [100, 250, 100, 80])
    config = SRConfig()
    original_profile = estimated_volume_profile(frame, config, latest_atr=4)
    original_anchor = anchored_vwaps(frame, (_pivot(frame),), config)[0]
    adjusted = frame.copy()
    adjusted[["high", "low", "close"]] /= 4
    adjusted["volume"] *= 4
    adjusted_profile = estimated_volume_profile(adjusted, config, latest_atr=1)
    adjusted_anchor = anchored_vwaps(adjusted, (_pivot(adjusted),), config)[0]
    assert [item.center for item in adjusted_profile.bins] == pytest.approx([item.center / 4 for item in original_profile.bins])
    assert [item.fraction for item in adjusted_profile.bins] == pytest.approx([item.fraction for item in original_profile.bins])
    assert adjusted_profile.poc == pytest.approx(original_profile.poc / 4)
    assert adjusted_profile.vah == pytest.approx(original_profile.vah / 4)
    assert adjusted_profile.val == pytest.approx(original_profile.val / 4)
    assert adjusted_anchor.latest_value == pytest.approx(original_anchor.latest_value / 4)
    assert adjusted_anchor.cumulative_volume == original_anchor.cumulative_volume * 4
