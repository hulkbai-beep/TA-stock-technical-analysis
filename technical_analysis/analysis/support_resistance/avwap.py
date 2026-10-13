"""Cumulative daily HLC3 anchored VWAP for confirmed major swing events."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .models import AnchoredVWAP, AVWAPPoint, Pivot, SRConfig


def anchored_vwaps(frame: pd.DataFrame, pivots: tuple[Pivot, ...],
                   config: SRConfig) -> tuple[AnchoredVWAP, ...]:
    """Select at most the newest confirmed swing low and high.

    The anchor is the actual pivot session, while chart values are emitted
    only from its confirmation session. Thus a retrospective line cannot
    imply the pivot was known beforehand. Accumulation persists across daily
    boundaries. Zero-volume bars contribute no weight; initial zero-volume
    values remain unavailable. An anchor crossing missing-session segments
    is withheld because the full cumulative volume cannot be reconstructed.

    Inputs preserve ingestion's split-adjusted prices and reported volume;
    this function does not infer splits, execution prices, or breakout events.
    """
    required = {"high", "low", "close", "volume", "session"}
    if frame.empty or config.max_anchors == 0 or not required.issubset(frame.columns):
        return ()
    last_index = len(frame) - 1
    selected: list[Pivot] = []
    for kind in ("low", "high"):
        choices = [pivot for pivot in pivots if pivot.kind == kind and
                   0 <= pivot.index <= pivot.confirmed_index <= last_index]
        if choices:
            selected.append(max(choices, key=lambda pivot: (pivot.index, pivot.confirmed_index, pivot.pivot_id)))
    # Configuring one anchor retains the most recent actual structure event.
    selected.sort(key=lambda pivot: (-pivot.index, pivot.kind, pivot.pivot_id))
    result: list[AnchoredVWAP] = []
    for pivot in selected[:config.max_anchors]:
        group = frame.iloc[pivot.index:]
        if "segment_id" in group and (group["segment_id"].isna().any() or group["segment_id"].nunique() != 1):
            continue
        values = group.loc[:, ["high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(values).all(axis=1).all() or (values["volume"] < 0).any():
            continue
        typical_price = (values["high"] + values["low"] + values["close"]) / 3.0
        cumulative_volume = values["volume"].cumsum()
        estimated = (typical_price * values["volume"]).cumsum() / cumulative_volume.where(cumulative_volume > 0)
        if pd.isna(estimated.iloc[-1]) or not np.isfinite(estimated.iloc[-1]):
            continue
        # Keep pandas cumulative arithmetic unchanged, then read its finished
        # values directly instead of constructing a Series for every point.
        estimates = estimated.to_numpy()
        sessions = group["session"].to_numpy()
        points = tuple(AVWAPPoint(sessions[index], float(estimates[index]))
                       for index in range(pivot.confirmed_index - pivot.index, len(group))
                       if np.isfinite(estimates[index]))
        kind_label = "低点" if pivot.kind == "low" else "高点"
        label = (f"日线近似 AVWAP（摆动{kind_label} {pivot.session.isoformat()}；"
                 f"确认 {pivot.confirmed_session.isoformat()}）")
        result.append(AnchoredVWAP(pivot.pivot_id, pivot.kind, pivot.session,
                                  pivot.confirmed_session, float(estimated.iloc[-1]),
                                  points, float(cumulative_volume.iloc[-1]), label))
    return tuple(result)
