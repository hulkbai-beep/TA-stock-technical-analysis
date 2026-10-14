# Multi-method daily support and resistance

This engine extends the existing dashboard, using its already-fetched FMP daily
OHLCV history. It makes no additional market-data requests and requires no
intraday subscription, database, LLM, or new dependency. The service exposes
`SupportResistanceResult`; the UI renders that result without detection formulas.
The existing technical-assessment action rules remain a separate module.

## Windows and data preparation

The default window is the latest **252 completed NYSE sessions**, approximating
12 months. The alternative is 126 sessions, approximating six months. These are
session counts, rather than exact calendar-month cutoffs. The latest 126 sessions
receive extra relevance weight in the 252-session window. A minimum of 60 valid
completed sessions and a positive latest ATR are required. Shorter-than-requested
windows are disclosed; fewer than 60 bars produce an unavailable result.

Preparation rechecks finite positive and internally consistent OHLC, nonnegative
volume, session dates and scheduled closes. Exact duplicates collapse; conflicting
duplicates are discarded. Forming bars are excluded. Missing NYSE dates are not
filled, and separate contiguous segments prevent ATR, pivot confirmation windows,
and cumulative AVWAP from silently bridging gaps.

ATR14 uses `TR = max(high-low, abs(high-previous_close), abs(low-previous_close))`.
The first bar of each contiguous segment uses `high-low`. The first 14 true ranges
seed an arithmetic mean; subsequent values use `(previous_ATR*13 + TR)/14`.
ATR is computed over the full cached history **before** cropping the analysis
window. Zero/unknown ATR is not replaced by a fixed-dollar threshold.

Prices use the existing FMP split-adjusted OHLC convention. The engine does not
guess split ratios or adjust prices a second time. An adjacent valid close change
of at least 40% produces a discontinuity/adjustment warning, not a declaration of
a stock split; real overnight gaps remain real observations. Volume weights use
the vendor's returned volume. Its cross-split volume adjustment convention has
not been independently verified, so weighted results must not be described as
reconstructed tick-level history.

## A. Confirmed swings and clustering

Default pivots require three completed sessions on each side. A high must be
strictly above every neighboring high; a low strictly below every neighboring low.
Tied plateaus do not create multiple pivots. The smaller excursion toward the
opposite extreme in the left/right windows must be at least one pivot-date ATR.
A pivot at `i` becomes usable at `i+3`, not at its original date. Every window must
belong to one continuous segment.

Sorted pivot prices form deterministic complete-link clusters: the entire proposed
cluster span must be at most **0.75 times its constituents' median pivot-date ATR**.
At least two distinct pivots are required. Boundaries are minimum/maximum pivot
prices padded by **0.20 times that same historical ATR**; the initial reference is
the median pivot price. Using constituent ATR avoids sizing historical geometry
with a volatility value that arrived later. Final constituent geometry cannot be
considered available before the last member's confirmation date. Pivot counts are
evidence counts, not independent interaction counts.

## B. Estimated Volume Profile

Each positive-volume day's volume is distributed uniformly across its observed
`[low,high]`: a bin receives `volume * overlap_length / (high-low)`. A zero-range bar
puts all its volume into the bin containing its observed price. Zero-volume bars
contribute zero weight. No volume is distributed across an overnight gap between
two bars. Bin mass sums to the actual input volume, subject to floating precision.

Default bin count is `ceil(window_price_range / (0.5*latest_ATR))`, bounded to
20-120 bins. A configurable fixed-count strategy defaults to 60 bins. All bins
cover the actual minimum low to maximum high. A fully flat profile has one
observed-price point, rather than an invented price range.

- POC is the center of the greatest-volume bin; ties prefer the lower-price bin.
- HVNs are maximal contiguous runs with density at least **1.5 times** the full
  profile's average density. They can contribute static zone candidates.
- LVNs are internal contiguous runs with density at most **0.5 times** the average;
  boundary-touching runs are excluded. LVNs are displayed as profile information,
  rather than automatically promoted to support/resistance.
- The value area starts at POC and expands to the larger-volume adjacent bin until
  it covers at least **70%** of volume; ties prefer the lower side. VAL/VAH are the
  outer boundaries. The actual covered fraction is reported because bins can
  overshoot 70%.

A uniform profile can have a POC without a meaningful HVN; it does not force a
static zone. Profile-only boundaries are known at the window end. This is explicitly
**Estimated Volume Profile**, not exact trade-by-price or buy/sell volume.

## C. Anchored VWAP

At most two automatic anchors are selected: the most recent confirmed significant
swing low and swing high. Setting the maximum to one keeps the more recent event.
No additional breakout/gap anchor is invented: existing action rules do not define
a high-volume structure event suitable for that purpose.

For each anchor, accumulate `HLC3=(high+low+close)/3` times volume from the actual
pivot session and divide by cumulative volume. Accumulation does not reset each
day. Chart points appear **only from the pivot confirmation date**, even though
their cumulative calculation includes the earlier anchor bars. Initial zero-volume
values remain unavailable. An anchor crossing a missing-session segment is withheld.

Latest AVWAP contributes dynamic confluence only when it lies inside a static
zone or within **0.50 latest ATR** of its nearest boundary. It never creates a
standalone static zone. Its source pivot and daily-bar approximation are disclosed.

## D. Fibonacci confluence

The window-end Fibonacci leg is the highest high and lowest low of the latest
**126 sessions** (or the window, if shorter); the leg is an advance when the high
is the more recent extreme. Retracements 23.6/38.2/50/61.8/78.6% are reported and
can be drawn on the chart. A 38.2/50/61.8% level inside a zone or within
**0.25 latest ATR** of its nearest boundary adds the `fibonacci` method and its
structure weight. Like AVWAP, Fibonacci never creates or widens a zone: its
published evidence is weaker than swing or volume evidence, so it only confirms.

## Consolidation, interactions, and validity

Nearby candidates use a **0.25 ATR** merge tolerance and a **2.5 ATR** maximum
combined width, with complete-link checks preventing long chains of overlapping
levels. Overlapping volume evidence can attach to existing swing geometry without
widening its bounds; newly expanded geometry has its own availability date. Original
pivot/source evidence remains attached. AVWAP evidence does not alter static bounds.
Candidates are processed swing-first, in ascending boundary/source order. Expanded
geometry uses the latest constituent availability; if a multi-candidate merge would
not also have passed the distance/width checks using ATR at that date, availability
is advanced to the current window end. Its consolidated representative is the
boundary midpoint. This prevents today's larger volatility tolerance from backdating
a newly merged interval. Initially, a completed close above/below the available zone
assigns support/resistance; a close inside assigns interaction until a later completed
close leaves it. Break/role-reversal confirmation follows the stricter rules below.

Independent historical interactions are explicitly **retrospective**: apply today's
final boundaries to observed completed historical bars. Consecutive touches belong
to one event. Five fully observed outside sessions are needed before another test;
missing dates cannot supply that separation. Reaction magnitude requires five full
continuous completed sessions after the event, and a known incoming direction.
It measures favorable excursion beyond the relevant boundary divided by event ATR.
Insufficient horizons or ambiguous direction produce unavailable reaction values.
Each test also records whether it **held**: approached from above (support), no
completed close during the event or on the bar that left it fell below the lower
boundary; approached from below (resistance), none rose above the upper boundary.
Earlier versions counted every touch as strength, so a level that price kept
slicing through scored as well as one that kept holding. Now a test that closed
through the zone subtracts half a test, and the panel shows held/failed counts.
These descriptive counts do not imply the zone was known or tradable at those dates.

Break/retest validity is separate and starts only once static geometry is available.
Two consecutive completed closes beyond the opposite boundary plus **0.25 ATR**
confirm a break. A later range overlap closing on the new side within 20 continuous
sessions confirms a retest and role reversal. Missing-session gaps cancel retest
eligibility. Two buffered closes back on the old side confirm failed-break recovery,
restoring the original role. Provisional quotes do not enter this state calculation.

Zone position is relative to the separately labeled latest available snapshot:
entirely below is Support, entirely above is Resistance, and containing the price
is an active price-interaction zone. A provisional reference can move that position,
but cannot confirm a breakout or role reversal. Broken zones remain diagnostic and
are excluded from valid support/resistance rankings.

## Explainable score and ranking

All components are normalized to `[0,1]`; defaults are provisional, configurable
`ScoreWeights`, not fitted parameters:

| Component | Weight | Normalization |
| --- | --- | --- |
| Independent interactions | 30% | `clip((1.5*recent + older)/6, 0, 1)`, where a held test counts +1 and a test that closed through the zone counts -0.5 |
| Estimated volume concentration | 25% | `min(relative_volume_density/3, 1)` |
| Structural confluence | 20% | Swing family 0.35 + HVN family 0.35 + AVWAP family 0.15 + Fibonacci family 0.15 |
| Observed reaction magnitude | 15% | `min(mean_observed_reaction_ATR/3, 1)` |
| Recency | 10% | `exp(-sessions_since_latest_test/(recent_sessions/2))` |

Final-zone volume integrates each intersecting profile bin once and normalizes
by zone width against full-profile density. Merging candidates never adds duplicate
bin volumes or sums their interaction counts. Multiple pivots/anchors count once
per method family. The smaller AVWAP structure weight acknowledges its derivation
from swing/volume inputs; it is not an independent statistical confirmation.

Missing reaction, recency, or volume components are disclosed and excluded; remaining
weights are renormalized. At least **60% of configured weight** must be calculable
to show a strength score. Observing zero interactions is a genuine zero count, while
having no valid volume data is missing. Each result preserves original/effective
weights, normalized values, contributions, and evidence. Breakout status is displayed
but breakout quality is not separately scored in this version.

Nearby relevance uses `ranking_score = strength_score / (1 + distance_in_ATR/5)`;
ties prefer the nearer zone, then stable zone ID. Unknown scores rank conservatively
and remain explicitly unavailable. The default selects up to three valid zones per
side, with two/five alternatives. Counts are never forced. Intersecting zones are
reported separately. A final greedy ranking pass suppresses overlapping displayed
zones across all three position categories, while retaining full diagnostic evidence.
Scores are **not probabilities or measured trading performance**.

## Integration and checks

`analysis/support_resistance/` separates typed models, preparation, swings, volume
profile, AVWAP, consolidation, state and scoring. `support_resistance_for()` in the
existing service reuses cached daily frames when controls change. The daily Plotly
price subplot shows green/red zones, amber interactions, subtle AVWAP curves and
optional POC/VAL/VAH references. A separate histogram shows estimated volume by
price; out-of-visible-range zones remain in the table without stretching candles.

The Chinese panel includes boundaries, distance, score, methods, retrospective test
count, last interaction, status, anchor/confirmation dates, HVN/LVN statistics and
score breakdowns, held/failed test counts and the Fibonacci table. Controls change
analysis window, maximum zones and display layers, including Fibonacci lines.

Tests cover each method, horizontal levels, trends, consolidation, gaps, low/zero
volume, missing/duplicate/invalid data, adjusted-price scale invariance, confirmation
timing, overlap conservation, break/retest/recovery, reproducibility, window updates,
service request reuse, and exact backend-to-chart coordinates. Synthetic data exists
only in tests. These are correctness checks, not a historical profitability test.

Delivery verification on October 12, 2026: **170 checks passed**. The real NVDA
workflow used 252 completed daily sessions from the existing cached-history
request and produced 3 selected support zones, 1 resistance zone, an available
estimated profile and 2 AVWAP anchors, with no support/resistance data warnings.
No additional endpoint or subscription was used. Actual tick-level distribution,
cross-split volume-adjustment semantics and predictive performance remain
unverified; there is no outstanding implementation dependency for this release.

### Added and changed files

| Files | Change |
| --- | --- |
| `analysis/support_resistance/__init__.py`, `models.py` | New module and typed/configurable contracts |
| `analysis/support_resistance/prepare.py` | New validation, session segmentation and ATR preparation |
| `analysis/support_resistance/swings.py` | New confirmed extrema and historical-volatility clustering |
| `analysis/support_resistance/volume_profile.py` | New estimated distribution, nodes and value area |
| `analysis/support_resistance/avwap.py` | New confirmed daily swing anchors and cumulative curves |
| `analysis/support_resistance/engine.py` | New consolidation, interactions, state, scoring and ranking |
| `services/analysis_service.py` | Exposes default results and cached-frame recalculation |
| `app.py` | Integrates controls, overlays and result section |
| `ui/charts.py`, `ui/support_resistance.py` | Extends daily Plotly layers; adds evidence/profile presentation |
| `scripts/check_live.py` | Reports sanitized real-ticker zone/profile/anchor availability |
| `tests/test_sr_swings.py`, `test_sr_volume.py`, `test_sr_engine.py` | New calculation and state checks |
| `tests/test_sr_integration.py`, `test_app.py` | New chart/service checks; extends UI regression coverage |
| `docs/support-resistance.md`, `docs/technical-assessment-rules.md`, `README.md` | Adds parameters/limitations and reconciles documentation |

Concept references: [Fidelity ATR](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/atr),
[TradingView volume-profile concepts](https://www.tradingview.com/support/solutions/43000502040-volume-profile-indicators-basic-concepts/),
and [TradingView Anchored VWAP](https://www.tradingview.com/support/solutions/43000669764-anchored-vwap-drawing-tool/).
The daily-volume approximation, thresholds, scoring and consolidation rules above
are this project's implementation choices; those sources do not validate them.
