"""Deterministic consolidation, completed-bar interactions, and relative scoring.

This snapshot is descriptive. Retrospective tests use the final zone bounds;
they are not a reconstruction of historically tradable signals or a backtest.
"""
from __future__ import annotations

from datetime import datetime
import math

import numpy as np
import pandas as pd

from analysis.indicators import _fibonacci_leg
from analysis.strategy import FIB_ENTRY_RATIOS, FibLevels, fib_levels

from .avwap import anchored_vwaps
from .models import (CandidateZone, Interaction, ScoreComponent, SRConfig,
                     SupportResistanceResult, VolumeProfile, Zone)
from .prepare import prepare_daily
from .swings import cluster_pivots, detect_pivots
from .volume_profile import estimated_volume_profile, profile_candidates


_METHOD_ORDER = {"swing": 0, "volume_profile": 1, "anchored_vwap": 2, "fibonacci": 3}
# Structure family weights: two independent geometry sources, then two
# derived confirmations. Fibonacci has the weakest evidence, so it only
# adds confluence and never creates a zone on its own.
_STRUCTURE_WEIGHTS = {"swing": 0.35, "volume_profile": 0.35, "anchored_vwap": 0.15, "fibonacci": 0.15}
_SCORE_LABELS = {"tests": "独立测试", "volume": "估算成交量密度", "structure": "结构方法",
                 "reaction": "历史反应幅度", "recency": "最近测试"}


def _finite_positive(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and value > 0


def _interval_gap(first: CandidateZone, second: CandidateZone) -> float:
    return max(first.lower - second.upper, second.lower - first.upper, 0.0)


def _consolidate_candidates(candidates: tuple[CandidateZone, ...], atr: float,
                            config: SRConfig, frame: pd.DataFrame | None = None) -> tuple[CandidateZone, ...]:
    """Merge only complete-link nearby intervals with a bounded combined width.

    Every member pair must overlap or be within merge_distance_atr. A chain
    cannot connect distant members merely through intermediate candidates.
    Original source IDs, pivot IDs, and evidence are retained once each.
    """
    valid = [item for item in candidates if
             all(_finite_positive(value) for value in (item.lower, item.upper, item.center)) and
             item.lower < item.upper and item.lower <= item.center <= item.upper and
             (item.upper - item.lower <= config.max_zone_width_atr * atr or
              "volume_profile" in item.methods)]
    # Start from swing geometry. An overlapping current profile can add
    # evidence without retrospectively expanding a known swing interval.
    ordered = sorted(valid, key=lambda item: ("swing" not in item.methods,
                                              item.lower, item.upper, item.source_id))
    def geometry(group):
        swings = [member for member in group if "swing" in member.methods]
        if swings:
            core_lower = min(member.lower for member in swings)
            core_upper = max(member.upper for member in swings)
            profiles = [member for member in group if "swing" not in member.methods]
            if all(min(core_upper, member.upper) >= max(core_lower, member.lower)
                   for member in profiles):
                return core_lower, core_upper, max(member.available_index for member in swings)
        return (min(member.lower for member in group), max(member.upper for member in group),
                max(member.available_index for member in group))
    groups: list[list[CandidateZone]] = []
    for item in ordered:
        for group in groups:
            combined_lower, combined_upper, _ = geometry(group + [item])
            if (combined_upper - combined_lower <= config.max_zone_width_atr * atr and
                    all(_interval_gap(item, member) <= config.merge_distance_atr * atr
                        for member in group)):
                group.append(item)
                break
        else:
            if item.upper - item.lower <= config.max_zone_width_atr * atr:
                groups.append([item])
    result = []
    for group in groups:
        lower, upper, available_index = geometry(group)
        geometry_members = [item for item in group if "swing" in item.methods]
        if (not geometry_members or any(
                min(upper, item.upper) < max(lower, item.lower)
                for item in group if "swing" not in item.methods)):
            geometry_members = group
        # Current ATR can select a combination today, but cannot backdate its
        # availability unless the same merge also passed the then-known ATR.
        if frame is not None and len({item.source_id for item in geometry_members}) > 1:
            historical_atr = float(frame.iloc[min(max(available_index, 0), len(frame) - 1)]["atr"])
            historically_mergeable = (_finite_positive(historical_atr) and
                upper - lower <= config.max_zone_width_atr * historical_atr and
                all(_interval_gap(first, second) <= config.merge_distance_atr * historical_atr
                    for index, first in enumerate(geometry_members)
                    for second in geometry_members[index + 1:]))
            if not historically_mergeable:
                available_index = len(frame) - 1
        methods = tuple(sorted({method for item in group for method in item.methods},
                               key=lambda method: (_METHOD_ORDER.get(method, 99), method)))
        result.append(CandidateZone(
            "+".join(sorted({item.source_id for item in group})), lower, upper,
            (lower + upper) / 2, methods, available_index,
            tuple(sorted({pivot for item in group for pivot in item.pivot_ids})),
            tuple(dict.fromkeys(evidence for item in group for evidence in item.evidence)),
        ))
    return tuple(sorted(result, key=lambda item: (item.lower, item.upper, item.source_id)))


def _volume_metrics(zone: CandidateZone, profile: VolumeProfile) -> tuple[float | None, float | None]:
    """Integrate each final interval/bin intersection once, then normalize width."""
    if not profile.available or not _finite_positive(profile.total_volume) or not profile.bins:
        return None, None
    profile_lower = min(item.lower for item in profile.bins)
    profile_upper = max(item.upper for item in profile.bins)
    if profile_upper <= profile_lower or zone.upper <= zone.lower:
        return None, None
    masses = []
    for item in profile.bins:
        if item.upper <= item.lower:
            continue
        overlap = max(0.0, min(zone.upper, item.upper) - max(zone.lower, item.lower))
        masses.append(item.volume * overlap / (item.upper - item.lower))
    mass = math.fsum(masses)
    fraction = min(max(mass / profile.total_volume, 0.0), 1.0)
    relative = (mass / (zone.upper - zone.lower)) / (profile.total_volume / (profile_upper - profile_lower))
    return fraction, max(relative, 0.0)


def _historical_interactions(frame: pd.DataFrame, zone: CandidateZone,
                             config: SRConfig) -> tuple[Interaction, ...]:
    """Regroup full-window OHLC overlaps using current final zone boundaries.

    Five fully outside completed sessions separate independent tests. A
    missing-session segment resets the observed outside streak. Nearby
    touches remain one event; an event spanning a gap has no reaction metric.
    The next full five-bar continuous horizon is required for reaction data.
    """
    events: list[list[int]] = []
    outside_streak = 0
    previous_segment = None
    for index, row in enumerate(frame.itertuples(index=False)):
        segment = row.segment_id
        if segment != previous_segment:
            outside_streak = 0
        previous_segment = segment
        touches = row.high >= zone.lower and row.low <= zone.upper
        if not touches:
            outside_streak += 1
            continue
        if not events or outside_streak >= config.test_cooldown_sessions:
            events.append([index, index])
        else:
            events[-1][1] = index
        outside_streak = 0
    interactions = []
    for event_start, event_end in events:
        reaction = None
        horizon_end = event_end + config.reaction_horizon_sessions
        event = frame.iloc[event_start:event_end + 1]
        future = frame.iloc[event_end + 1:horizon_end + 1]
        atr = float(frame.iloc[event_end]["atr"])
        continuous = (event_start > 0 and horizon_end < len(frame) and
                      event["segment_id"].nunique() == 1 and
                      len(future) == config.reaction_horizon_sessions and
                      future["segment_id"].eq(frame.iloc[event_end]["segment_id"]).all() and
                      frame.iloc[event_start - 1]["segment_id"] == frame.iloc[event_start]["segment_id"])
        if continuous and _finite_positive(atr):
            incoming = float(frame.iloc[event_start - 1]["close"])
            if incoming > zone.upper:
                reaction = max(float(future["high"].max()) - zone.upper, 0.0) / atr
            elif incoming < zone.lower:
                reaction = max(zone.lower - float(future["low"].min()), 0.0) / atr
        # A test held when no close during the event, or on the bar that left
        # it, landed beyond the far boundary. Unknown incoming side: no verdict.
        held = None
        if event_start > 0:
            incoming = float(frame.iloc[event_start - 1]["close"])
            closes = frame.iloc[event_start:min(event_end + 2, len(frame))]["close"].astype(float)
            if incoming > zone.upper:
                held = bool((closes >= zone.lower).all())
            elif incoming < zone.lower:
                held = bool((closes <= zone.upper).all())
        interactions.append(Interaction(
            frame.iloc[event_start]["session"], frame.iloc[event_end]["session"],
            event_start, event_end, reaction,
            event_end >= len(frame) - config.recent_sessions, held,
        ))
    return tuple(interactions)


def _zone_state(frame: pd.DataFrame, zone: CandidateZone,
                config: SRConfig) -> tuple[str, str, bool, object, object, object]:
    """Confirm breaks and later role reversals entirely with completed bars.

    Two consecutive closes beyond the opposite buffered boundary confirm a
    break. A later overlap closing on the new side within 20 continuous
    sessions confirms a retest and role reversal. A quote never enters here.
    """
    first = min(max(zone.available_index, 0), len(frame) - 1)
    initial = float(frame.iloc[first]["close"])
    role = "support" if initial > zone.upper else "resistance" if initial < zone.lower else "interaction"
    status, reversed_role = "active", False
    breakout_session = retest_session = recovery_session = None
    breakout_index = None
    retest_eligible = False
    consecutive = recovery_count = 0
    previous_segment = frame.iloc[first]["segment_id"]
    for index, row in enumerate(frame.iloc[first + 1:].itertuples(index=False), first + 1):
        close = float(row.close)
        segment = row.segment_id
        if segment != previous_segment:
            consecutive = 0
            recovery_count = 0
            retest_eligible = False
        previous_segment = segment
        if role == "interaction":
            if close > zone.upper:
                role = "support"
            elif close < zone.lower:
                role = "resistance"
            continue
        if status != "active":
            atr = float(row.atr)
            buffer = config.breakout_buffer_atr * atr if _finite_positive(atr) else None
            old_side = (buffer is not None and
                        (close > zone.upper + buffer if role == "support"
                         else close < zone.lower - buffer))
            recovery_count = recovery_count + 1 if old_side else 0
            if recovery_count >= config.breakout_confirm_bars:
                status = "active"
                recovery_session = row.session
                retest_eligible = False
                consecutive = recovery_count = 0
                continue
            within_window = (retest_eligible and breakout_index is not None and
                             index - breakout_index <= config.retest_window_sessions)
            overlap = row.high >= zone.lower and row.low <= zone.upper
            new_side = close < zone.lower if role == "support" else close > zone.upper
            if within_window and overlap and new_side:
                role = "resistance" if role == "support" else "support"
                reversed_role, status = True, "active"
                retest_session = row.session
                consecutive = recovery_count = 0
            elif within_window and zone.lower <= close <= zone.upper:
                status = "under_retest"
            else:
                status = "broken"
            continue
        atr = float(row.atr)
        if not _finite_positive(atr):
            consecutive = 0
            continue
        threshold = config.breakout_buffer_atr * atr
        opposite_close = (close < zone.lower - threshold if role == "support"
                          else close > zone.upper + threshold)
        consecutive = consecutive + 1 if opposite_close else 0
        if consecutive >= config.breakout_confirm_bars:
            status = "broken"
            breakout_session, breakout_index = row.session, index
            retest_session = recovery_session = None
            reversed_role = False
            retest_eligible = True
            consecutive = recovery_count = 0
    return status, role, reversed_role, breakout_session, retest_session, recovery_session


def _score_components(metrics: dict[str, float | None], config: SRConfig
                      ) -> tuple[float | None, tuple[ScoreComponent, ...], float]:
    """Exclude missing metrics explicitly and withhold scores below 60% coverage."""
    weights = vars(config.weights)
    total = math.fsum(weights.values())
    available_weight = math.fsum(weight for name, weight in weights.items()
                                 if metrics.get(name) is not None)
    coverage = available_weight / total
    components = []
    for name, weight in weights.items():
        value = metrics.get(name)
        if value is not None:
            value = min(max(float(value), 0.0), 1.0)
        effective = weight / available_weight if value is not None and available_weight > 0 else 0.0
        contribution = value * effective * 100 if value is not None else None
        explanation = (f"{_SCORE_LABELS[name]}可计算，归一化值 {value:.3f}。"
                       if value is not None else f"{_SCORE_LABELS[name]}不可可靠计算；排除并披露权重覆盖。")
        components.append(ScoreComponent(name, value, weight, effective, contribution, explanation))
    score = (math.fsum(item.contribution or 0.0 for item in components)
             if coverage >= 0.60 and available_weight > 0 else None)
    return score, tuple(components), coverage


def _has_hvn(zone: CandidateZone, profile: VolumeProfile, config: SRConfig) -> bool:
    return "volume_profile" in zone.methods and any(
        node.relative_density >= config.hvn_density_multiple and
        min(zone.upper, node.upper) > max(zone.lower, node.lower) for node in profile.hvn)


def _window_fibonacci(frame: pd.DataFrame, config: SRConfig) -> FibLevels | None:
    """Retracement levels of the latest high/low leg at the window end."""
    lookback = min(config.fib_lookback_sessions, len(frame))
    if lookback < 20:
        return None
    leg = _fibonacci_leg(frame["high"].astype(float), frame["low"].astype(float), lookback)
    return fib_levels(leg.iloc[-1])


def _build_zone(zone_id: str, candidate: CandidateZone, frame: pd.DataFrame,
                profile: VolumeProfile, anchors: tuple, reference: float, atr: float,
                config: SRConfig, fib: FibLevels | None = None) -> Zone:
    methods = list(candidate.methods)
    evidence = list(candidate.evidence)
    evidence.append(f"原始候选标识：{candidate.source_id}。")
    if candidate.pivot_ids:
        evidence.append("原始摆动标识：" + "、".join(candidate.pivot_ids) + "。")
    for anchor in anchors:
        gap = max(candidate.lower - anchor.latest_value, anchor.latest_value - candidate.upper, 0.0)
        if gap <= config.avwap_proximity_atr * atr:
            if "anchored_vwap" not in methods:
                methods.append("anchored_vwap")
            evidence.append(f"锚点 {anchor.anchor_id}（{anchor.anchor_session}，确认 {anchor.confirmed_session}）"
                            f"日线近似 AVWAP {anchor.latest_value:.4f}；与区间距离 {gap:.4f}（{gap / atr:.3f} ATR）。")
    if fib is not None:
        for ratio, level in fib.levels:
            if ratio not in FIB_ENTRY_RATIOS:
                continue
            gap = max(candidate.lower - level, level - candidate.upper, 0.0)
            if gap <= config.fib_proximity_atr * atr:
                if "fibonacci" not in methods:
                    methods.append("fibonacci")
                evidence.append(f"斐波那契 {ratio:.1%} 回撤位 {level:.2f}（波段 {fib.low:.2f}–{fib.high:.2f}，"
                                f"{'上涨' if fib.up else '下跌'}波段）与区间距离 {gap:.2f}。")
    tests = _historical_interactions(frame, candidate, config)
    latest = tests[-1] if tests else None
    sessions_since = len(frame) - 1 - latest.end_index if latest is not None else None
    reactions = [item.reaction_atr for item in tests if item.reaction_atr is not None]
    reaction = math.fsum(reactions) / len(reactions) if reactions else None
    fraction, relative = _volume_metrics(candidate, profile)
    structure = ((_STRUCTURE_WEIGHTS["swing"] if "swing" in methods and len(candidate.pivot_ids) >= 2 else 0.0) +
                 (_STRUCTURE_WEIGHTS["volume_profile"] if _has_hvn(candidate, profile, config) else 0.0) +
                 (_STRUCTURE_WEIGHTS["anchored_vwap"] if "anchored_vwap" in methods else 0.0) +
                 (_STRUCTURE_WEIGHTS["fibonacci"] if "fibonacci" in methods else 0.0))
    # Held tests add strength (recent ones more); a test that closed through
    # the zone subtracts, since repeated breaks show the level absorbing less.
    held_count = sum(item.held is True for item in tests)
    failed_count = sum(item.held is False for item in tests)
    weighted_tests = math.fsum((1.5 if item.recent else 1.0) * (-0.5 if item.held is False else 1.0)
                               for item in tests)
    metrics = {"tests": min(max(weighted_tests / 6, 0.0), 1.0),
               "volume": min(relative / 3, 1.0) if relative is not None else None,
               "structure": structure,
               "reaction": min(reaction / 3, 1.0) if reaction is not None else None,
               "recency": math.exp(-sessions_since / (config.recent_sessions / 2))
               if sessions_since is not None else None}
    score, breakdown, coverage = _score_components(metrics, config)
    status, role, reversed_role, breakout, retest, recovery = _zone_state(frame, candidate, config)
    kind = ("support" if candidate.upper < reference else
            "resistance" if candidate.lower > reference else "interaction")
    distance = (candidate.center / reference - 1) * 100
    rank = (score or 0.0) / (1 + abs(candidate.center - reference) / atr / 5)
    recent_count = sum(item.recent for item in tests)
    status_text = {"active": "有效", "broken": "已突破，尚未完成有效回测", "under_retest": "回测观察中"}[status]
    reaction_text = (f"完整后续 {config.reaction_horizon_sessions} 根日线的平均反应为 {reaction:.2f} ATR"
                     if reaction is not None else "历史反应幅度尚无完整连续后续窗口，或入区方向不明确")
    volume_text = (f"估算成交量占比 {fraction:.1%}，按区间宽度归一的相对密度 {relative:.2f}"
                   if fraction is not None else "估算成交量指标不可用")
    score_text = f"评分权重覆盖 {coverage:.0%}"
    if score is None:
        score_text += "，低于 60%，不展示强度分数"
    explanation = (f"{len(tests)} 次回顾独立互动（近 {config.recent_sessions} 个交易日 {recent_count} 次；"
                   f"守住 {held_count} 次，收盘穿越 {failed_count} 次）；"
                   f"{reaction_text}；{volume_text}。状态：{status_text}；{score_text}。强度分数不是成功概率。")
    availability = frame.iloc[min(max(candidate.available_index, 0), len(frame) - 1)]["session"]
    evidence.append(f"当前静态区间边界形成日：{availability}；突破、回测与恢复仅使用该日起已知区间和已完成日线。"
                    "回顾独立互动按当前最终区间统计整个窗口，包含区间形成前的历史；并非历史实时可交易记录。")
    if "volume_profile" in methods:
        evidence.append("估算成交量分布使用整个当前分析窗口；其区间与历史测试的重合属于回顾性证据。"
                        "成交量与 AVWAP 的完整方法汇合仅代表当前快照，不表示历史当时已确认。")
    evidence.append("突破质量未单独评分；反应分量仅使用已观测的后续价格幅度。"
                    "各方法族仅计一次；结构分量中摆动 35%、成交量 35%、AVWAP 15%、斐波那契 15%。")
    if breakout is not None:
        evidence.append(f"最近已确认突破日：{breakout}；"
                        + (f"有效回测确认日：{retest}，角色已反转。" if retest else
                           f"假突破恢复日：{recovery}，恢复原角色，没有角色反转。" if recovery else
                           "尚无有效回测或假突破恢复确认。"))
    if role != "interaction":
        evidence.append(f"已完成日线确认的角色：{'支撑' if role == 'support' else '阻力'}；"
                        "当前报价只决定区间位置和距离，不确认突破或角色反转。")
    return Zone(zone_id, kind, candidate.lower, candidate.upper, candidate.center, distance,
                tuple(methods), len(tests), recent_count, latest.end_session if latest else None,
                reaction, fraction, relative, sessions_since, score, breakdown, status, role,
                reversed_role, breakout, retest, rank, explanation, tuple(evidence), tests,
                available_session=availability, recovery_session=recovery,
                held_tests=held_count, failed_tests=failed_count)


def _select_ranked_zones(zones: tuple[Zone, ...], reference: float, limit: int
                         ) -> tuple[tuple[Zone, ...], tuple[Zone, ...], tuple[Zone, ...]]:
    """Select globally nonoverlapping valid intervals, preserving diagnostics.

    Complete-link consolidation can leave overlapping groups when joining
    their extremes would be too broad. Keep both in the diagnostic result,
    but display the higher-ranked interval only, with a per-kind cap.
    """
    eligible = [zone for zone in zones if
                zone.status == "active" or (zone.kind == "interaction" and zone.status == "under_retest")]
    ordered = sorted(eligible, key=lambda zone: (-zone.ranking_score, abs(zone.center - reference), zone.zone_id))
    selected: list[Zone] = []
    counts = {"support": 0, "resistance": 0, "interaction": 0}
    for zone in ordered:
        if counts[zone.kind] >= limit:
            continue
        if any(min(zone.upper, existing.upper) > max(zone.lower, existing.lower) for existing in selected):
            continue
        selected.append(zone)
        counts[zone.kind] += 1
    return tuple(tuple(zone for zone in selected if zone.kind == kind)
                 for kind in ("support", "resistance", "interaction"))


def analyze_support_resistance(daily: pd.DataFrame, config: SRConfig | None = None, *,
                               current_price: float | None = None,
                               reference_timestamp: datetime | None = None,
                               reference_provisional: bool = False) -> SupportResistanceResult:
    """Return ranked daily zones without fetching data or using forming bars.

    A separate quote may classify distance/position, including a provisional
    quote, but completed daily bars exclusively determine validity and role.
    """
    config = config or SRConfig()
    prepared = prepare_daily(daily, config)
    frame = prepared.frame
    warnings = list(prepared.warnings)
    if not prepared.available:
        return SupportResistanceResult(config, False, None, reference_timestamp,
                                       reference_provisional, frame.iloc[-1]["session"] if len(frame) else None,
                                       len(frame), prepared.latest_atr, warnings=tuple(warnings), reason=prepared.reason)
    atr = float(prepared.latest_atr)
    reference_is_quote = _finite_positive(current_price)
    if current_price is not None and not reference_is_quote:
        warnings.append("当前报价无效，使用最新已完成日线收盘价作为距离参考。")
    reference = float(current_price) if reference_is_quote else float(frame.iloc[-1]["close"])
    if not reference_is_quote:
        reference_provisional = False
        if reference_timestamp is None:
            last = frame.iloc[-1]
            value = last.get("bar_end", last.get("timestamp"))
            reference_timestamp = pd.Timestamp(value).to_pydatetime() if pd.notna(value) else None
    pivots = detect_pivots(frame, config)
    profile = estimated_volume_profile(frame, config, latest_atr=atr)
    anchors = anchored_vwaps(frame, pivots, config)
    warnings.extend(profile.warnings)
    if not anchors and config.max_anchors > 0:
        warnings.append("没有可用的已确认连续数据锚点；日线近似 AVWAP 不可用。")
    raw = cluster_pivots(pivots, atr, config) + profile_candidates(profile, len(frame) - 1)
    consolidated = _consolidate_candidates(raw, atr, config, frame)
    supported = [candidate for candidate in consolidated if
                 ("swing" in candidate.methods and len(candidate.pivot_ids) >= 2) or
                 _has_hvn(candidate, profile, config)]
    fib = _window_fibonacci(frame, config)
    zones = tuple(_build_zone(f"Z{index:02d}", candidate, frame, profile, anchors, reference, atr, config, fib)
                  for index, candidate in enumerate(supported, start=1))
    supports, resistances, interactions = _select_ranked_zones(zones, reference, config.max_zones_per_side)
    reason = None if supports or resistances or interactions else "有效支撑/阻力证据不足；未强制生成区间。"
    return SupportResistanceResult(config, True, reference, reference_timestamp, reference_provisional,
                                   frame.iloc[-1]["session"], len(frame), atr, supports, resistances,
                                   interactions, zones, pivots, anchors, profile, tuple(dict.fromkeys(warnings)), reason,
                                   fib)
