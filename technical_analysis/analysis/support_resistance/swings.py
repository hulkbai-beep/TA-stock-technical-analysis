"""Confirmed strict extrema and deterministic complete-link price clusters."""
from __future__ import annotations

from statistics import median

import numpy as np
import pandas as pd

from .models import CandidateZone, Pivot, SRConfig


def detect_pivots(frame: pd.DataFrame, config: SRConfig) -> tuple[Pivot, ...]:
    """Confirm unique extrema after right-hand completed candles have arrived.

    Significance is the smaller excursion into the left and right windows,
    measured from the extremum to their opposite price extremes, divided by
    ATR at the pivot. A full window must belong to one calendar segment.
    """
    required = {"session", "high", "low", "atr", "segment_id"}
    if frame.empty or not required.issubset(frame.columns):
        return ()
    frame = frame.reset_index(drop=True)
    atr_values = pd.to_numeric(frame["atr"], errors="coerce").to_numpy(dtype=float)
    high_values = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=float)
    low_values = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=float)
    segments = frame["segment_id"].to_numpy()
    missing_segments = pd.isna(segments)
    sessions = frame["session"].to_numpy()
    completed = (np.array([isinstance(value, (bool, np.bool_)) and value
                           for value in frame["complete"]], dtype=bool)
                 if "complete" in frame else None)
    pivots: list[Pivot] = []
    left, right = config.pivot_left, config.pivot_right
    comparisons = np.arange(left + right + 1) != left
    for index in range(left, len(frame) - right):
        # Extract columns once; window slices remain views and preserve the
        # same completed-bar, unique-extremum and gap rules without creating
        # hundreds of intermediate DataFrames and Series.
        start, end = index - left, index + right + 1
        if (missing_segments[start:end].any() or
                not (segments[start:end] == segments[index]).all()):
            continue
        if completed is not None and not completed[start:end].all():
            continue
        atr = atr_values[index]
        highs, lows = high_values[start:end], low_values[start:end]
        if not np.isfinite(atr) or atr <= 0 or not np.isfinite(highs).all() or not np.isfinite(lows).all():
            continue
        for kind, values in (("high", highs), ("low", lows)):
            price = float(values[left])
            if kind == "high":
                unique = (price > values[comparisons]).all()
                excursion = min(price - lows[:left].min(), price - lows[left + 1:].min())
            else:
                unique = (price < values[comparisons]).all()
                excursion = min(highs[:left].max() - price, highs[left + 1:].max() - price)
            reaction = float(excursion / atr)
            if not unique or reaction < config.pivot_min_reaction_atr:
                continue
            session = sessions[index]
            confirmed = sessions[index + right]
            pivots.append(Pivot(f"{kind}-{session}", kind, index, index + right,
                                session, confirmed, price, atr, reaction))
    return tuple(pivots)


def cluster_pivots(pivots: tuple[Pivot, ...], latest_atr: float, config: SRConfig) -> tuple[CandidateZone, ...]:
    """Partition sorted pivot prices with a bounded total cluster span.

    Complete-link span prevents a chain of nearby pivots from drifting into
    one wide zone. At least two distinct pivots are required for a static
    swing zone. Final bounds become available only after every constituent
    pivot confirms. Clustering and padding use the median constituent pivot
    ATR, known at formation; latest_atr is only a sanity gate. Pivot counts
    are evidence counts, never independent tests.
    """
    if latest_atr is None or not np.isfinite(latest_atr) or latest_atr <= 0:
        return ()
    unique: dict[str, Pivot] = {}
    for pivot in pivots:
        if (np.isfinite(pivot.price) and pivot.price > 0 and
                np.isfinite(pivot.atr) and pivot.atr > 0):
            unique.setdefault(pivot.pivot_id, pivot)
    ordered = sorted(unique.values(), key=lambda pivot: (pivot.price, pivot.session, pivot.kind, pivot.pivot_id))
    groups: list[list[Pivot]] = []
    for pivot in ordered:
        proposed = [*groups[-1], pivot] if groups else [pivot]
        historical_atr = median(item.atr for item in proposed)
        distance = config.cluster_distance_atr * historical_atr
        if not groups or pivot.price - groups[-1][0].price > distance:
            groups.append([pivot])
        else:
            groups[-1].append(pivot)
    candidates = []
    for group in groups:
        if len(group) < 2:
            continue
        historical_atr = median(pivot.atr for pivot in group)
        padding = config.zone_halfwidth_atr * historical_atr
        lower = group[0].price - padding
        upper = group[-1].price + padding
        if lower <= 0 or not lower < upper:
            continue
        ids = tuple(sorted(pivot.pivot_id for pivot in group))
        dates = "；".join(f"{pivot.session} 的{'高点' if pivot.kind == 'high' else '低点'}于 {pivot.confirmed_session} 确认" for pivot in sorted(group, key=lambda pivot: (pivot.session, pivot.kind)))
        formation_session = max(pivot.confirmed_session for pivot in group)
        evidence = (
            f"{len(group)} 个已确认摆动点聚类：{dates}。摆动点数量不等于独立测试次数。",
            f"聚类距离和区间扩展使用成员摆动点的历史 ATR 中位数 {historical_atr:.4f}；完整区间最早于 {formation_session} 确认，不随最新 ATR 追溯改变。",
        )
        candidates.append(CandidateZone(
            source_id="swing:" + ",".join(ids), lower=float(lower), upper=float(upper),
            center=float(median(pivot.price for pivot in group)), methods=("swing",),
            available_index=max(pivot.confirmed_index for pivot in group),
            pivot_ids=ids, evidence=evidence,
        ))
    return tuple(candidates)
