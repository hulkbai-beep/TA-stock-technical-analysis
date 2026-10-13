"""Estimated daily-bar volume distribution; never exact trade-by-price data.

Each positive-volume daily bar contributes uniform density over its own
observed [low, high] range. Exact bin overlaps allocate its volume once; gaps
between bars receive no volume merely because the previous close differs.
Price inputs retain the ingestion layer's split-adjusted price basis and
vendor-reported volume convention. This module neither infers nor reapplies
splits.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .models import CandidateZone, ProfileBin, ProfileNode, SRConfig, VolumeProfile


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive consecutive runs without counting a bin twice."""
    indexes = np.flatnonzero(mask)
    if not len(indexes):
        return []
    groups = np.split(indexes, np.flatnonzero(np.diff(indexes) > 1) + 1)
    return [(int(group[0]), int(group[-1])) for group in groups]


def _node(edges: np.ndarray, volumes: np.ndarray, start: int, end: int,
          total: float, global_density: float) -> ProfileNode:
    volume = math.fsum(float(value) for value in volumes[start:end + 1])
    lower, upper = float(edges[start]), float(edges[end + 1])
    centers = (edges[start:end + 1] + edges[start + 1:end + 2]) / 2
    center = (float(np.dot(centers, volumes[start:end + 1]) / volume)
              if volume > 0 else (lower + upper) / 2)
    return ProfileNode(lower, upper, center, volume, volume / total,
                       volume / (upper - lower) / global_density)


def estimated_volume_profile(frame: pd.DataFrame, config: SRConfig,
                             latest_atr: float | None = None) -> VolumeProfile:
    """Compute an estimated profile over the supplied completed daily window.

    ATR binning uses ceil(observed range / (volume_bin_atr * latest ATR)),
    clipped to 20..120 bins. Fixed-count binning uses the configured count,
    also clipped to 20..120. If ATR is unavailable, fixed-count is an explicit
    fallback. Bounds are actual observed extremes, including zero-volume bars.

    HVNs are maximal consecutive runs with density >= the configured multiple
    of global mean density. LVNs use <= their configured multiple, excluding
    runs touching the profile boundary. POC is the greatest-volume bin, with
    equal estimates resolved toward the lower price. The contiguous value
    area starts there and adds the larger adjacent bin (ties downward) until
    its actual volume fraction reaches the configured target.
    """
    if frame.empty or not {"low", "high", "volume"}.issubset(frame.columns):
        return VolumeProfile(False, warnings=("估算成交量分布缺少有效日线数据。",))
    data = frame.loc[:, ["low", "high", "volume"]].apply(pd.to_numeric, errors="coerce")
    valid = (np.isfinite(data).all(axis=1) & (data["low"] > 0) &
             (data["high"] >= data["low"]) & (data["volume"] >= 0))
    if not valid.all():
        return VolumeProfile(False, warnings=("估算成交量分布收到无效数据，未计算。",))
    total = math.fsum(float(value) for value in data["volume"] if value > 0)
    if not np.isfinite(total) or total <= 0:
        return VolumeProfile(False, warnings=("全部有效日线成交量为零，估算成交量分布不可用。",))
    lower, upper = float(data["low"].min()), float(data["high"].max())
    if upper == lower:
        # A point mass has no finite density per price unit. Its normalized
        # relative density is defined as 1; no HVN/LVN inference is made.
        point = ProfileBin(lower, upper, lower, total, 1.0, 1.0)
        return VolumeProfile(True, (point,), poc=lower, vah=upper, val=lower,
                             total_volume=total, value_area_fraction=1.0,
                             warnings=("全部日线价格相同：仅显示实际价格点，无法估算每价格单位密度。",))
    warnings: list[str] = []
    atr_valid = latest_atr is not None and np.isfinite(latest_atr) and latest_atr > 0
    if config.volume_bin_strategy == "atr" and atr_valid:
        count = int(np.clip(math.ceil((upper - lower) / (config.volume_bin_atr * latest_atr)), 20, 120))
    else:
        count = int(np.clip(config.volume_bins, 20, 120))
        if config.volume_bin_strategy == "atr":
            warnings.append("最新 ATR 不可用，估算成交量分布使用固定数量价格区间。")
    # Unique edges avoid zero-width bins when floating-point price resolution
    # cannot represent every requested subdivision.
    edges = np.unique(np.linspace(lower, upper, count + 1))
    widths = np.diff(edges)
    volumes = np.zeros(len(widths), dtype=float)
    for low, high, volume in data.itertuples(index=False, name=None):
        if volume <= 0:
            continue
        if high == low:
            index = int(np.clip(np.searchsorted(edges, low, side="right") - 1, 0, len(volumes) - 1))
            volumes[index] += volume
            continue
        overlap = np.maximum(0.0, np.minimum(edges[1:], high) - np.maximum(edges[:-1], low))
        allocation = overlap / (high - low) * volume
        covered = np.flatnonzero(overlap > 0)
        # Repair rounding only in an actually observed part of this bar.
        allocation[covered[-1]] += volume - math.fsum(float(value) for value in allocation)
        volumes += allocation
    maximum = float(volumes.max())
    poc_index = int(np.flatnonzero(np.isclose(volumes, maximum, rtol=1e-12, atol=0))[0])
    volumes[poc_index] += total - math.fsum(float(value) for value in volumes)
    density = total / (upper - lower)
    relative = volumes / widths / density
    hvn_runs = _runs(relative >= config.hvn_density_multiple)
    lvn_runs = [(start, end) for start, end in _runs(relative <= config.lvn_density_multiple)
                if start > 0 and end < len(volumes) - 1]
    hvn = tuple(_node(edges, volumes, start, end, total, density) for start, end in hvn_runs)
    lvn = tuple(_node(edges, volumes, start, end, total, density) for start, end in lvn_runs)
    labels = ["normal"] * len(volumes)
    for label, runs in (("hvn", hvn_runs), ("lvn", lvn_runs)):
        for start, end in runs:
            labels[start:end + 1] = [label] * (end - start + 1)
    left = right = poc_index
    area_volume = float(volumes[poc_index])
    while area_volume / total < config.value_area_fraction and (left > 0 or right < len(volumes) - 1):
        left_volume = float(volumes[left - 1]) if left > 0 else -1.0
        right_volume = float(volumes[right + 1]) if right < len(volumes) - 1 else -1.0
        if left_volume >= right_volume or np.isclose(left_volume, right_volume, rtol=1e-12, atol=0):
            left -= 1
        else:
            right += 1
        area_volume = math.fsum(float(value) for value in volumes[left:right + 1])
    bins = tuple(ProfileBin(float(edges[index]), float(edges[index + 1]),
                            float((edges[index] + edges[index + 1]) / 2), float(volume),
                            float(volume / total), float(relative[index]), labels[index])
                 for index, volume in enumerate(volumes))
    return VolumeProfile(True, bins, hvn, lvn, bins[poc_index].center,
                         float(edges[right + 1]), float(edges[left]), total,
                         area_volume / total, tuple(warnings))


def profile_candidates(profile: VolumeProfile, available_index: int) -> tuple[CandidateZone, ...]:
    """Expose meaningful HVN intervals without promoting POC-only evidence.

    These full-window estimates become available only at the supplied final
    index. POC always remains a reported statistic, but a maximum-volume bin
    does not by itself establish a reliable static price zone. In particular,
    uniform and flat-price profiles produce no static profile candidates.
    Only the engine should calculate a merged zone's final volume fraction;
    adding candidate fractions would double-count overlapping evidence.
    """
    if not profile.available:
        return ()
    candidates = [CandidateZone(f"profile.hvn.{index}", node.lower, node.upper,
                                node.center, ("volume_profile",), available_index,
                                evidence=(f"估算成交量分布 HVN：成交量占比 {node.fraction:.1%}，相对密度 {node.relative_density:.2f}。",))
                  for index, node in enumerate(profile.hvn)]
    return tuple(candidates)
