"""Offline consolidation, state, scoring, and integration checks."""
from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from analysis.support_resistance import engine
from analysis.support_resistance.engine import (
    _consolidate_candidates, _historical_interactions, _score_components,
    _volume_metrics, _zone_state, analyze_support_resistance,
)
from analysis.support_resistance.models import (
    AnchoredVWAP, AVWAPPoint, CandidateZone, PreparedData, ProfileBin,
    ScoreWeights, SRConfig, VolumeProfile,
)
from data.normalize import session_schedule


def daily_bars(count=100):
    schedule = session_schedule("2024-01-02", "2026-12-31").iloc[:count]
    pattern = np.array([0, 1, 2, 4, 6, 4, 2, 1, 0, -1, -2, -4, -6, -4, -2, -1])
    close = 100 + np.resize(pattern, count)
    return pd.DataFrame({
        "session": schedule.index.tolist(), "timestamp": schedule["market_close"].tolist(),
        "bar_end": schedule["market_close"].tolist(), "complete": True,
        "open": close, "high": close + 0.5, "low": close - 0.5,
        "close": close, "volume": 1000.0,
    })


def prepared_frame(count=80):
    frame = daily_bars(count)
    frame["open"] = frame["close"] = 105.0
    frame["high"], frame["low"] = 106.0, 104.0
    frame["atr"], frame["segment_id"] = 2.0, 0
    return frame


def candidate(lower=99, upper=101, *, available=0, name="swing.a", pivots=("p1", "p2")):
    return CandidateZone(name, lower, upper, (lower + upper) / 2, ("swing",), available,
                         pivots, (f"Original evidence {name}",))


def touch(frame, index, *, close=102.0):
    frame.loc[index, ["open", "close", "high", "low"]] = [close, close, max(close, 102), 100]


def mock_analysis(monkeypatch, frame, candidates, profile=None, anchors=()):
    monkeypatch.setattr(engine, "prepare_daily", lambda daily, config: PreparedData(frame, latest_atr=2.0, available=True))
    monkeypatch.setattr(engine, "detect_pivots", lambda daily, config: ())
    monkeypatch.setattr(engine, "cluster_pivots", lambda pivots, atr, config: candidates)
    monkeypatch.setattr(engine, "estimated_volume_profile", lambda daily, config, latest_atr: profile or VolumeProfile(False))
    monkeypatch.setattr(engine, "profile_candidates", lambda profile, available_index: ())
    monkeypatch.setattr(engine, "anchored_vwaps", lambda daily, pivots, config: anchors)


def test_complete_link_merging_preserves_unique_evidence_and_blocks_chains():
    first = candidate(90, 92, name="a")
    overlap = candidate(91, 93, name="b", pivots=("p2", "p3"))
    distant = candidate(93.4, 94, name="c")
    result = _consolidate_candidates((distant, overlap, first, first), 2.0, SRConfig())
    assert len(result) == 2
    assert result[0].lower == 90 and result[0].upper == 93
    assert result[0].methods == ("swing",)
    assert result[0].pivot_ids == ("p1", "p2", "p3")
    assert result[0].evidence == first.evidence + overlap.evidence
    assert result[0].source_id == "a+b"
    assert _consolidate_candidates(tuple(reversed((first, overlap, distant))), 2.0, SRConfig()) == result


def test_invalid_and_overwide_zones_are_excluded():
    bad = (candidate(-1, 1), candidate(101, 100), candidate(90, 100), candidate(float("nan"), 100))
    assert _consolidate_candidates(bad, 2.0, SRConfig()) == ()


def test_final_zone_volume_is_integrated_once_instead_of_summing_candidates():
    profile = VolumeProfile(True, (
        ProfileBin(90, 100, 95, 200, 0.2, 0.4),
        ProfileBin(100, 110, 105, 800, 0.8, 1.6),
    ), total_volume=1000)
    merged = _consolidate_candidates((candidate(98, 103, name="a"), candidate(101, 105, name="b")), 4.0, SRConfig())[0]
    fraction, density = _volume_metrics(merged, profile)
    assert fraction == pytest.approx(0.44)
    assert density == pytest.approx((440 / 7) / (1000 / 20))
    assert fraction < sum(_volume_metrics(item, profile)[0] for item in (candidate(98, 103), candidate(101, 105)))
    assert _volume_metrics(merged, VolumeProfile(False)) == (None, None)


def test_consecutive_and_nearby_touches_are_one_test_with_full_reaction_horizon():
    frame = prepared_frame()
    for index in (10, 11, 14, 20, 79):
        touch(frame, index)
    interactions = _historical_interactions(frame, candidate(), SRConfig())
    assert [(item.start_index, item.end_index) for item in interactions] == [(10, 14), (20, 20), (79, 79)]
    assert interactions[0].reaction_atr == pytest.approx(2.5)
    assert interactions[1].reaction_atr == pytest.approx(2.5)
    assert interactions[-1].reaction_atr is None


def test_prior_close_sets_reaction_direction_and_inside_is_explicitly_unavailable():
    frame = prepared_frame(30)
    touch(frame, 10)
    frame.loc[9, "close"] = 100
    assert _historical_interactions(frame, candidate(), SRConfig())[0].reaction_atr is None
    frame.loc[9, "close"] = 98
    frame.loc[11:15, ["open", "close", "high", "low"]] = [96, 96, 98, 94]
    result = _historical_interactions(frame, candidate(), SRConfig())
    assert result[0].reaction_atr == pytest.approx(2.5)


def test_missing_segment_cannot_manufacture_five_known_outside_sessions():
    frame = prepared_frame(40)
    touch(frame, 10)
    touch(frame, 20)
    frame.loc[18:, "segment_id"] = 1
    result = _historical_interactions(frame, candidate(), SRConfig())
    assert len(result) == 1
    assert result[0].end_index == 20 and result[0].reaction_atr is None
    frame.loc[15:, "segment_id"] = 1
    assert len(_historical_interactions(frame, candidate(), SRConfig())) == 2


def test_retrospective_touches_are_distinct_from_causal_candidate_availability():
    frame = prepared_frame(40)
    for index in (2, 5, 10, 20):
        touch(frame, index)
    result = _historical_interactions(frame, candidate(available=10), SRConfig())
    assert len(result) == 2 and result[0].start_index == 2
    assert _historical_interactions(frame, candidate(available=len(frame) - 1), SRConfig()) == result


def test_overlapping_profile_adds_evidence_without_expanding_known_swing_geometry():
    swing = candidate(99, 101, available=10)
    profile = CandidateZone("profile.hvn.1", 98, 102, 100, ("volume_profile",), 79)
    merged = _consolidate_candidates((profile, swing), 2, SRConfig())[0]
    assert (merged.lower, merged.upper, merged.available_index) == (99, 101, 10)
    assert merged.methods == ("swing", "volume_profile")
    assert "profile.hvn.1" in merged.source_id
    nearby = replace(profile, lower=101.2, upper=102, center=101.6)
    expanded = _consolidate_candidates((nearby, swing), 2, SRConfig())[0]
    assert (expanded.lower, expanded.upper, expanded.available_index) == (99, 102, 79)
    broad = replace(profile, lower=90, upper=110)
    bounded = _consolidate_candidates((broad, swing), 2, SRConfig())[0]
    assert (bounded.lower, bounded.upper, bounded.available_index) == (99, 101, 10)
    assert "volume_profile" in bounded.methods
    assert _consolidate_candidates((broad,), 2, SRConfig()) == ()


def test_state_cannot_use_final_zone_boundaries_before_formation(monkeypatch):
    frame = breakout_frame()
    final = candidate(available=29)
    assert _zone_state(frame, final, SRConfig())[:3] == ("active", "resistance", False)
    assert _zone_state(frame, final, SRConfig())[3] is None
    mock_analysis(monkeypatch, frame, (final,))
    zone = analyze_support_resistance(frame).zones[0]
    assert zone.available_session == frame.iloc[29]["session"]
    assert zone.breakout_session is None and zone.retest_session is None
    assert any("形成前" in item for item in zone.evidence)


def test_current_atr_merge_does_not_backdate_bounds_if_formation_atr_disallowed_it():
    frame = prepared_frame()
    frame.loc[10, "atr"] = 0.5
    first = candidate(99, 100, available=5, name="first")
    second = candidate(100.4, 101, available=10, name="second")
    merged = _consolidate_candidates((first, second), 2.0, SRConfig(), frame)[0]
    assert merged.available_index == len(frame) - 1
    frame.loc[10, "atr"] = 2.0
    assert _consolidate_candidates((first, second), 2.0, SRConfig(), frame)[0].available_index == 10


def breakout_frame():
    frame = prepared_frame(30)
    frame.loc[10:11, ["open", "close", "high", "low"]] = [98, 98, 98.5, 97]
    frame.loc[12:, ["open", "close", "high", "low"]] = [97, 97, 98, 96]
    touch(frame, 14, close=100)
    frame.loc[16:, ["open", "close", "high", "low"]] = [98, 98, 98.5, 97]
    frame.loc[16, "high"] = 100
    return frame


def test_break_requires_two_completed_closes_and_retest_confirms_role_reversal():
    frame = breakout_frame()
    assert _zone_state(frame.iloc[:11], candidate(), SRConfig())[:3] == ("active", "support", False)
    broken = _zone_state(frame.iloc[:12], candidate(), SRConfig())
    assert broken[:3] == ("broken", "support", False)
    assert broken[3] == frame.iloc[11]["session"]
    assert _zone_state(frame.iloc[:15], candidate(), SRConfig())[:3] == ("under_retest", "support", False)
    reversed_state = _zone_state(frame, candidate(), SRConfig())
    assert reversed_state[:3] == ("active", "resistance", True)
    assert reversed_state[4] == frame.iloc[16]["session"]


def test_gap_breaks_confirmation_and_cancels_retest_continuity():
    frame = breakout_frame()
    frame.loc[11:, "segment_id"] = 1
    assert _zone_state(frame.iloc[:12], candidate(), SRConfig())[0] == "active"
    frame = breakout_frame()
    frame.loc[12:, "segment_id"] = 1
    assert _zone_state(frame, candidate(), SRConfig())[:3] == ("broken", "support", False)


def test_late_retest_cannot_confirm_role_reversal():
    frame = breakout_frame()
    frame.loc[14, ["open", "close", "high", "low"]] = [97, 97, 98, 96]
    frame.loc[16, "high"] = 98.5
    touch(frame, 29, close=98)
    assert _zone_state(frame, candidate(), replace(SRConfig(), retest_window_sessions=10))[:3] == ("broken", "support", False)


def test_failed_break_recovers_old_role_after_two_buffered_closes_and_gap_resets():
    frame = breakout_frame().iloc[:14].copy()
    frame.loc[12:13, ["open", "close", "high", "low"]] = [105, 105, 106, 104]
    state = _zone_state(frame, candidate(), SRConfig())
    assert state[:3] == ("active", "support", False)
    assert state[3] == frame.iloc[11]["session"]
    assert state[4] is None and state[5] == frame.iloc[13]["session"]
    frame.loc[13:, "segment_id"] = 1
    assert _zone_state(frame, candidate(), SRConfig())[:3] == ("broken", "support", False)


def test_new_break_cycle_clears_stale_retest_and_recovery_dates():
    frame = breakout_frame()
    frame.loc[20:, ["open", "close", "high", "low"]] = [104, 104, 105, 103]
    state = _zone_state(frame, candidate(), SRConfig())
    assert state[:3] == ("broken", "resistance", False)
    assert state[3] == frame.iloc[21]["session"]
    assert state[4] is None and state[5] is None


def test_provisional_quote_only_changes_position_and_never_confirmed_role(monkeypatch):
    frame = prepared_frame()
    mock_analysis(monkeypatch, frame, (candidate(),))
    normal = analyze_support_resistance(frame, current_price=105)
    snapshot = analyze_support_resistance(frame, current_price=95, reference_provisional=True)
    inside = analyze_support_resistance(frame, current_price=100, reference_provisional=True)
    assert normal.zones[0].kind == "support"
    assert snapshot.zones[0].kind == "resistance"
    assert inside.zones[0].kind == "interaction" and inside.interactions
    assert snapshot.reference_provisional
    for result in (normal, snapshot, inside):
        assert result.zones[0].confirmed_role == "support"
        assert result.zones[0].status == "active"
        assert not result.zones[0].role_reversed
    assert normal.zones[0].strength_score == snapshot.zones[0].strength_score


def test_broken_zones_remain_diagnostic_and_excluded_from_valid_lists(monkeypatch):
    frame = breakout_frame().iloc[:12]
    mock_analysis(monkeypatch, frame, (candidate(),))
    result = analyze_support_resistance(frame)
    assert result.zones[0].status == "broken"
    assert result.supports == result.resistances == result.interactions == ()
    assert result.reason


def test_avwap_is_confluence_only_and_correlated_anchor_bonus_is_capped(monkeypatch):
    frame = prepared_frame()
    session = frame.iloc[10]["session"]
    anchor = lambda name, value: AnchoredVWAP(name, "low", session, session, value,
                                            (AVWAPPoint(session, value),), 1000, name)
    mock_analysis(monkeypatch, frame, (candidate(),), anchors=(anchor("p1", 100.5), anchor("p2", 101.8)))
    result = analyze_support_resistance(frame)
    zone = result.zones[0]
    assert zone.methods.count("anchored_vwap") == 1
    assert next(item.value for item in zone.score_breakdown if item.name == "structure") == pytest.approx(0.6)
    assert any("p2" in item and "0.8000" in item for item in zone.evidence)
    mock_analysis(monkeypatch, frame, (), anchors=(anchor("p1", 100.5),))
    assert analyze_support_resistance(frame).zones == ()
    mock_analysis(monkeypatch, frame, (candidate(),), anchors=(anchor("far", 110),))
    assert "anchored_vwap" not in analyze_support_resistance(frame).zones[0].methods


def test_missing_score_components_are_excluded_with_coverage_and_no_artificial_zero():
    metrics = {"tests": 0.5, "volume": None, "structure": 0.4, "reaction": 0.6, "recency": 1.0}
    score, breakdown, coverage = _score_components(metrics, SRConfig())
    assert coverage == pytest.approx(0.75)
    assert score == pytest.approx((0.5 * 0.3 + 0.4 * 0.2 + 0.6 * 0.15 + 0.1) / 0.75 * 100)
    missing = next(item for item in breakdown if item.name == "volume")
    assert missing.value is None and missing.contribution is None and missing.effective_weight == 0
    assert sum(item.effective_weight for item in breakdown) == pytest.approx(1)
    assert sum(item.contribution or 0 for item in breakdown) == pytest.approx(score)
    assert _score_components({}, SRConfig())[0] is None
    assert _score_components({"tests": 0, "structure": 0.4}, SRConfig())[0] is None


def test_nearby_zone_can_outrank_stronger_distant_zone(monkeypatch):
    frame = prepared_frame()
    frame["close"] = frame["open"] = 100.0
    frame["low"], frame["high"] = 99.5, 100.5
    profile = VolumeProfile(True, (
        ProfileBin(40, 60, 50, 800, 0.8, 2.8), ProfileBin(60, 110, 85, 200, 0.2, 0.28),
    ), total_volume=1000)
    near, far = candidate(98, 99, name="near"), candidate(49, 51, name="far")
    mock_analysis(monkeypatch, frame, (near, far), profile)
    result = analyze_support_resistance(frame, replace(SRConfig(), max_zones_per_side=2))
    assert result.supports[0].center == near.center
    assert result.supports[0].strength_score < result.supports[1].strength_score


def test_rank_selection_omits_redundant_overlap_without_forcing_an_overwide_merge(monkeypatch):
    frame = prepared_frame()
    sources = (candidate(90, 91, name="a"), candidate(90.5, 94.5, name="b"), candidate(94, 95, name="c"))
    mock_analysis(monkeypatch, frame, sources)
    result = analyze_support_resistance(frame)
    assert len(result.zones) == 2
    assert result.zones[0].upper > result.zones[1].lower
    assert len(result.supports) == 1
    selected = result.supports + result.resistances + result.interactions
    assert all(min(first.upper, second.upper) <= max(first.lower, second.lower)
               for index, first in enumerate(selected) for second in selected[index + 1:])


def test_sideways_market_has_reproducible_horizontal_support_and_resistance():
    frame = daily_bars(180)
    first = analyze_support_resistance(frame, current_price=100)
    second = analyze_support_resistance(frame.copy(deep=True), current_price=100)
    assert first == second
    assert first.available and first.supports and first.resistances
    assert all(zone.upper < 100 for zone in first.supports)
    assert all(zone.lower > 100 for zone in first.resistances)
    assert len(first.supports) <= 3 and len(first.resistances) <= 3
    assert all(zone.independent_tests >= 1 for zone in first.supports + first.resistances)
    assert [zone.zone_id for zone in first.zones] == [f"Z{index:02d}" for index in range(1, len(first.zones) + 1)]


def test_strong_trend_does_not_force_three_zones_or_unconfirmed_pivots():
    frame = daily_bars(100)
    close = 50 + np.arange(100) * 0.5
    frame["open"] = frame["close"] = close
    frame["high"], frame["low"] = close + 0.2, close - 0.2
    result = analyze_support_resistance(frame)
    assert result.available and result.pivots == () and result.anchors == ()
    assert len(result.supports) < 3 and len(result.resistances) < 3
    if not result.zones:
        assert result.reason


def test_insufficient_history_produces_no_manufactured_levels():
    result = analyze_support_resistance(daily_bars(59))
    assert not result.available and result.reason
    assert result.supports == result.resistances == result.zones == ()


def test_forming_extreme_bar_does_not_change_confirmed_outputs():
    frame = daily_bars(181)
    last = frame.iloc[-1].copy()
    frame = frame.iloc[:-1].copy()
    before = analyze_support_resistance(frame)
    last[["open", "close", "high", "low"]] = [500, 500, 1000, 1]
    last["complete"] = False
    after = analyze_support_resistance(pd.concat([frame, last.to_frame().T], ignore_index=True))
    assert before.zones == after.zones
    assert before.pivots == after.pivots
    assert before.anchors == after.anchors
    assert before.profile == after.profile
    assert before.analysis_session == after.analysis_session


def test_history_window_and_new_completed_bar_update_the_snapshot():
    frame = daily_bars(321)
    frame.loc[160:, ["open", "high", "low", "close"]] += 50
    full = analyze_support_resistance(frame.iloc[:-1])
    recent = analyze_support_resistance(frame.iloc[:-1], replace(SRConfig(), lookback_sessions=126))
    next_day = analyze_support_resistance(frame)
    assert full.actual_sessions == 252 and recent.actual_sessions == 126
    assert any(zone.center < 120 for zone in full.zones)
    assert all(zone.center > 120 for zone in recent.zones)
    assert next_day.analysis_session != full.analysis_session
    assert next_day.profile != full.profile


def test_low_volume_is_missing_evidence_instead_of_neutral_or_positive():
    frame = daily_bars(180)
    frame["volume"] = 0
    result = analyze_support_resistance(frame, current_price=100)
    assert result.available and result.zones
    assert not result.profile.available and not result.anchors
    assert all(zone.volume_fraction is None and zone.relative_volume is None for zone in result.zones)
    assert all(next(item for item in zone.score_breakdown if item.name == "volume").value is None
               for zone in result.zones)


def test_custom_weights_change_score_without_changing_market_zone_geometry():
    frame = daily_bars(180)
    initial = analyze_support_resistance(frame, current_price=100)
    changed = analyze_support_resistance(frame, replace(SRConfig(), weights=ScoreWeights(0.1, 0.1, 0.6, 0.1, 0.1)), current_price=100)
    assert [(zone.lower, zone.upper, zone.status) for zone in initial.zones] == [(zone.lower, zone.upper, zone.status) for zone in changed.zones]
    assert any(one.strength_score != two.strength_score for one, two in zip(initial.zones, changed.zones))
