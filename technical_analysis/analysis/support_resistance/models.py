"""Typed contracts for deterministic daily support/resistance analysis."""
from dataclasses import dataclass, field
from datetime import date, datetime

import pandas as pd


@dataclass(frozen=True)
class ScoreWeights:
    tests: float = 0.30
    volume: float = 0.25
    structure: float = 0.20
    reaction: float = 0.15
    recency: float = 0.10

    def __post_init__(self):
        values = tuple(vars(self).values())
        if any(not 0 <= value <= 1 for value in values) or sum(values) <= 0:
            raise ValueError("Score weights must be finite, nonnegative fractions with a positive sum")


@dataclass(frozen=True)
class SRConfig:
    lookback_sessions: int = 252
    recent_sessions: int = 126
    min_history_sessions: int = 60
    atr_period: int = 14
    pivot_left: int = 3
    pivot_right: int = 3
    pivot_min_reaction_atr: float = 1.0
    cluster_distance_atr: float = 0.75
    zone_halfwidth_atr: float = 0.20
    merge_distance_atr: float = 0.25
    max_zone_width_atr: float = 2.5
    test_cooldown_sessions: int = 5
    reaction_horizon_sessions: int = 5
    breakout_confirm_bars: int = 2
    breakout_buffer_atr: float = 0.25
    retest_window_sessions: int = 20
    volume_bin_strategy: str = "atr"
    volume_bins: int = 60
    volume_bin_atr: float = 0.5
    value_area_fraction: float = 0.70
    hvn_density_multiple: float = 1.5
    lvn_density_multiple: float = 0.5
    avwap_proximity_atr: float = 0.5
    fib_lookback_sessions: int = 126
    fib_proximity_atr: float = 0.25
    max_anchors: int = 2
    max_zones_per_side: int = 3
    weights: ScoreWeights = field(default_factory=ScoreWeights)

    def __post_init__(self):
        if self.lookback_sessions < 20 or self.min_history_sessions < self.atr_period:
            raise ValueError("History must cover at least 20 sessions and the ATR warm-up")
        positive = (self.recent_sessions, self.atr_period, self.pivot_left, self.pivot_right,
                    self.cluster_distance_atr, self.zone_halfwidth_atr, self.max_zone_width_atr,
                    self.test_cooldown_sessions, self.reaction_horizon_sessions,
                    self.breakout_confirm_bars, self.retest_window_sessions, self.volume_bins,
                    self.volume_bin_atr, self.hvn_density_multiple)
        if any(not 0 < value < float("inf") for value in positive):
            raise ValueError("Detection periods and scales must be finite and positive")
        if (self.volume_bin_strategy not in {"atr", "fixed_count"} or
                not 0 < self.value_area_fraction < 1 or not 0 <= self.lvn_density_multiple < 1 or
                self.max_zones_per_side not in {2, 3, 5} or not 0 <= self.max_anchors <= 2 or
                any(not 0 <= value < float("inf") for value in
                    (self.merge_distance_atr, self.breakout_buffer_atr, self.avwap_proximity_atr, self.pivot_min_reaction_atr))):
            raise ValueError("Invalid support/resistance configuration")


@dataclass
class PreparedData:
    frame: pd.DataFrame
    warnings: tuple[str, ...] = ()
    latest_atr: float | None = None
    available: bool = False
    reason: str | None = None
    missing_sessions: int = 0


@dataclass(frozen=True)
class Pivot:
    pivot_id: str
    kind: str
    index: int
    confirmed_index: int
    session: date
    confirmed_session: date
    price: float
    atr: float
    reaction_atr: float


@dataclass(frozen=True)
class CandidateZone:
    source_id: str
    lower: float
    upper: float
    center: float
    methods: tuple[str, ...]
    available_index: int
    pivot_ids: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class ProfileBin:
    lower: float
    upper: float
    center: float
    volume: float
    fraction: float
    relative_density: float
    node: str = "normal"


@dataclass(frozen=True)
class ProfileNode:
    lower: float
    upper: float
    center: float
    volume: float
    fraction: float
    relative_density: float


@dataclass(frozen=True)
class VolumeProfile:
    available: bool
    bins: tuple[ProfileBin, ...] = ()
    hvn: tuple[ProfileNode, ...] = ()
    lvn: tuple[ProfileNode, ...] = ()
    poc: float | None = None
    vah: float | None = None
    val: float | None = None
    total_volume: float = 0.0
    value_area_fraction: float | None = None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class AVWAPPoint:
    session: date
    value: float


@dataclass(frozen=True)
class AnchoredVWAP:
    anchor_id: str
    kind: str
    anchor_session: date
    confirmed_session: date
    latest_value: float
    points: tuple[AVWAPPoint, ...]
    cumulative_volume: float
    label: str


@dataclass(frozen=True)
class Interaction:
    start_session: date
    end_session: date
    start_index: int
    end_index: int
    reaction_atr: float | None
    recent: bool
    held: bool | None = None


@dataclass(frozen=True)
class ScoreComponent:
    name: str
    value: float | None
    configured_weight: float
    effective_weight: float
    contribution: float | None
    explanation: str


@dataclass(frozen=True)
class Zone:
    zone_id: str
    kind: str
    lower: float
    upper: float
    center: float
    distance_percent: float
    methods: tuple[str, ...]
    independent_tests: int
    recent_tests: int
    latest_test: date | None
    reaction_atr: float | None
    volume_fraction: float | None
    relative_volume: float | None
    sessions_since_test: int | None
    strength_score: float | None
    score_breakdown: tuple[ScoreComponent, ...]
    status: str
    confirmed_role: str
    role_reversed: bool
    breakout_session: date | None
    retest_session: date | None
    ranking_score: float
    explanation: str
    evidence: tuple[str, ...]
    interactions: tuple[Interaction, ...] = ()
    available_session: date | None = None
    recovery_session: date | None = None
    held_tests: int = 0
    failed_tests: int = 0


@dataclass(frozen=True)
class SupportResistanceResult:
    config: SRConfig
    available: bool
    reference_price: float | None
    reference_timestamp: datetime | None
    reference_provisional: bool
    analysis_session: date | None
    actual_sessions: int
    latest_atr: float | None
    supports: tuple[Zone, ...] = ()
    resistances: tuple[Zone, ...] = ()
    interactions: tuple[Zone, ...] = ()
    zones: tuple[Zone, ...] = ()
    pivots: tuple[Pivot, ...] = ()
    anchors: tuple[AnchoredVWAP, ...] = ()
    profile: VolumeProfile = field(default_factory=lambda: VolumeProfile(False))
    warnings: tuple[str, ...] = ()
    reason: str | None = None
    fibonacci: object | None = None
