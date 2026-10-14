"""Deterministic assessment checks using explicit, precomputed test indicators."""
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from analysis.signals import MarketContext, evaluate_signals


NY = ZoneInfo("America/New_York")
NOW = datetime(2025, 5, 5, 10, 2, tzinfo=NY)


def _daily():
    """Two consecutive completed NYSE sessions after indicator warm-up."""
    sessions = [date(2025, 5, 1), date(2025, 5, 2)]
    return pd.DataFrame({
        "session": sessions,
        "timestamp": [pd.Timestamp(f"{session} 16:00", tz=NY) for session in sessions],
        "bar_end": [pd.Timestamp(f"{session} 16:00", tz=NY) for session in sessions],
        "complete": [True, True],
        "close": [100.5, 101.0],
        "sma20": [99.8, 100.0],
        "sma50": [94.8, 95.0],
        "rsi14": [58.0, 60.0],
        "sma200": [89.9, 90.0],
        "atr14": [2.0, 2.0],
        "high22": [102.0, 102.0],
        "prev_high": [100.2, 100.8],
        "pullback_atr5": [0.1, 0.2],
        "updown_volume20": [1.4, 1.4],
        "ret63": [0.12, 0.12],
        "ret126": [0.20, 0.20],
        "mkt_close": [500.0, 500.0],
        "mkt_sma50": [490.0, 490.0],
        "mkt_sma200": [470.0, 470.0],
        "mkt_ret63": [0.05, 0.05],
        "mkt_ret126": [0.08, 0.08],
    })


def _intraday():
    """A covered session with its newest two completed five-minute intervals."""
    stamps = pd.DatetimeIndex(["2025-05-05 09:50", "2025-05-05 09:55"], tz=NY)
    return pd.DataFrame({
        "session": stamps.date,
        "timestamp": stamps,
        "bar_end": stamps + pd.Timedelta("5min"),
        "complete": [True, True],
        "close": [101.0, 101.2],
        "vwap": [100.7, 100.8],
        "session_open": [100.6, 100.6],
        "coverage_complete": [True, True],
        "vwap_available": [True, True],
    })


def _assess(daily=None, intraday=None, *, now=NOW, retrieved_at=None, context=None):
    return evaluate_signals(
        _daily() if daily is None else daily,
        _intraday() if intraday is None else intraday,
        now=now,
        daily_retrieved_at=retrieved_at or now,
        intraday_retrieved_at=retrieved_at or now,
        context=context or MarketContext(earnings_checked=True),
    )


def test_confirmed_bullish_setup_separates_entry_and_position_actions():
    daily, intraday = _daily(), _intraday()
    original_daily, original_intraday = daily.copy(deep=True), intraday.copy(deep=True)
    result = _assess(daily, intraday)

    assert result.daily.code == "strong_bullish"
    assert result.intraday.code == "bullish"
    assert result.entry.code == "buy"
    assert result.position.code == "hold"
    assert result.daily.session == date(2025, 5, 2)
    assert result.daily.as_of == datetime(2025, 5, 2, 16, tzinfo=NY)
    assert result.intraday.as_of == datetime(2025, 5, 5, 10, tzinfo=NY)
    assert result.entry.timeframe
    assert result.entry.trigger_status
    assert result.entry.conditions and result.entry.invalidation
    assert any("100.00" in item for item in result.daily.evidence)
    pd.testing.assert_frame_equal(daily, original_daily)
    pd.testing.assert_frame_equal(intraday, original_intraday)


def test_bearish_daily_and_oversold_rsi_produce_reduce_not_automatic_buy():
    daily = _daily()
    daily["close"] = [99.0, 98.0]
    daily["sma20"] = [101.2, 101.0]
    daily["sma50"] = [105.2, 105.0]
    daily["rsi14"] = [28.0, 25.0]

    result = _assess(daily)

    assert result.daily.code == "strong_bearish"
    assert result.entry.code == "wait"
    assert result.position.code == "reduce_sell"
    assert result.position.invalidation


def test_bearish_intraday_is_a_timing_note_not_an_entry_blocker():
    intraday = _intraday()
    intraday["close"] = [100.4, 100.2]

    result = _assess(intraday=intraday)

    assert result.daily.code == "strong_bullish"
    assert result.intraday.code == "bearish"
    assert result.entry.code == "buy"
    assert result.position.code == "hold"
    assert any("盘中偏空" in item for item in result.entry.conflicts)


def test_overbought_uptrend_remains_strong_with_wait_and_hold():
    daily = _daily()
    daily["rsi14"] = [73.0, 75.0]

    result = _assess(daily)

    assert result.daily.code == "strong_bullish"
    assert result.entry.code == "wait"
    assert result.position.code == "hold"
    assert result.entry_attractiveness != _assess().entry_attractiveness
    assert result.momentum_label


@pytest.mark.parametrize("missing", ["empty", "sma50", "rsi14"])
def test_insufficient_daily_history_never_manufactures_action(missing):
    daily = _daily()
    if missing == "empty":
        daily = daily.iloc[:0].copy()
    else:
        daily.loc[daily.index[-1], missing] = np.nan

    result = _assess(daily)

    assert result.daily.code == "insufficient_data"
    assert result.entry.code == "insufficient_data"
    assert result.position.code == "insufficient_data"


def test_loose_bullish_trend_does_not_automatically_offer_entry():
    daily = _daily()
    daily["sma20"] = [100.2, 100.0]

    result = _assess(daily)

    assert result.daily.code == "bullish"
    assert result.entry.code == "wait"
    assert result.position.code == "hold"


def test_unavailable_intraday_preserves_daily_entry_decision():
    result = _assess(intraday=_intraday().iloc[:0].copy())

    assert result.daily.code == "strong_bullish"
    assert result.intraday.code == "unavailable"
    assert result.entry.code == "buy"
    assert result.position.code == "hold"


def test_forming_daily_observation_cannot_flip_confirmed_assessment():
    daily = _daily()
    baseline = _assess(daily)
    forming = daily.iloc[[-1]].copy()
    forming["session"] = date(2025, 5, 5)
    forming["timestamp"] = pd.Timestamp("2025-05-05 16:00", tz=NY)
    forming["bar_end"] = forming["timestamp"]
    forming["complete"] = False
    forming["close"] = 70.0
    forming["sma20"] = 98.0
    forming["sma50"] = 94.0
    forming["rsi14"] = 20.0

    result = _assess(pd.concat([daily, forming], ignore_index=True))

    assert result.daily == baseline.daily
    assert result.entry.code == baseline.entry.code == "buy"
    assert result.position.code == "hold"
    assert any("临时观察" in note for note in result.provisional_notes)


def test_forming_intraday_bar_cannot_supply_second_confirmation():
    intraday = _intraday()
    intraday.loc[0, "close"] = 100.5
    baseline = _assess(intraday=intraday)
    forming = intraday.iloc[[-1]].copy()
    forming["timestamp"] = pd.Timestamp("2025-05-05 10:00", tz=NY)
    forming["bar_end"] = pd.Timestamp("2025-05-05 10:05", tz=NY)
    forming["complete"] = False
    forming["close"] = 102.0

    result = _assess(intraday=pd.concat([intraday, forming], ignore_index=True))

    assert result.intraday == baseline.intraday
    assert result.intraday.code == "neutral"
    assert result.entry.code == baseline.entry.code
    assert any("临时观察" in note for note in result.provisional_notes)


def test_finite_indicators_do_not_bridge_nonconsecutive_daily_sessions():
    daily = _daily()
    daily.loc[0, "session"] = date(2025, 4, 30)
    daily.loc[0, "timestamp"] = pd.Timestamp("2025-04-30 16:00", tz=NY)
    daily.loc[0, "bar_end"] = daily.loc[0, "timestamp"]

    result = _assess(daily)

    assert result.daily.code == "insufficient_data"
    assert result.entry.code == "insufficient_data"
    assert result.position.code == "insufficient_data"


def test_entry_trigger_requires_reclaim_or_pullback_then_resumption():
    daily = _daily()
    daily.loc[0, "close"] = 101.5
    daily.loc[1, "prev_high"] = 101.8
    pending = _assess(daily)
    assert pending.daily.code == "strong_bullish"
    assert pending.entry.code == "wait"

    daily.loc[0, "close"] = 99.5
    reclaimed = _assess(daily)
    assert reclaimed.daily.code == "strong_bullish"
    assert reclaimed.entry.code == "buy"
    assert reclaimed.entry.trigger_status != pending.entry.trigger_status


@pytest.mark.parametrize("now,stamps", [
    (NOW, ["2025-05-02 15:50", "2025-05-02 15:55"]),
    (datetime(2025, 5, 5, 10, 12, tzinfo=NY), ["2025-05-05 09:50", "2025-05-05 09:55"]),
    (datetime(2025, 5, 2, 16, 3, tzinfo=NY), ["2025-05-02 15:50", "2025-05-02 15:55"]),
], ids=["historical-session", "stale-twelve-minutes", "after-hours"])
def test_entry_is_decided_on_completed_daily_bars_whatever_the_intraday_age(now, stamps):
    intraday = _intraday()
    stamps = pd.DatetimeIndex(stamps, tz=NY)
    intraday["session"] = stamps.date
    intraday["timestamp"] = stamps
    intraday["bar_end"] = stamps + pd.Timedelta("5min")

    result = _assess(intraday=intraday, now=now)

    assert result.entry.code == "buy"
    assert result.position.code == "hold"


def test_stale_daily_data_withholds_both_actions():
    result = _assess(now=datetime(2025, 5, 6, 10, 2, tzinfo=NY))

    assert result.daily.code == "strong_bullish"
    assert result.entry.code == "insufficient_data"
    assert result.position.code == "insufficient_data"


def test_current_gap_up_is_unattractive_even_when_daily_close_was_near_sma20():
    intraday = _intraday()
    intraday["close"] = [104.0, 104.5]

    result = _assess(intraday=intraday)

    assert result.daily.code == "strong_bullish"
    assert result.intraday.code == "bullish"
    assert result.entry.code == "wait"
    assert result.position.code == "hold"


def test_rsi70_is_excluded_but_the_extension_limit_is_inclusive():
    daily = _daily()
    daily.loc[1, "rsi14"] = 70.0
    assert _assess(daily).entry.code == "wait"

    daily.loc[1, "rsi14"] = 69.9
    daily["close"] = [102.5, 103.0]
    intraday = _intraday()
    intraday["close"] = [102.9, 103.0]
    assert _assess(daily, intraday).entry.code == "buy"
    assert _assess(daily, intraday, now=datetime(2025, 5, 5, 10, 10, tzinfo=NY)).entry.code == "buy"


def test_missing_session_coverage_is_unavailable_rather_than_neutral():
    intraday = _intraday()
    intraday["coverage_complete"] = False

    result = _assess(intraday=intraday)

    assert result.intraday.code == "unavailable"


def test_two_daily_closes_below_sma50_reduce_even_without_bearish_ma_order():
    daily = _daily()
    daily["close"] = [94.0, 94.5]

    result = _assess(daily)

    assert result.daily.code == "neutral"
    assert result.entry.code == "wait"
    assert result.position.code == "reduce_sell"


def test_pullback_trigger_needs_a_recent_touch_of_sma20():
    daily = _daily()
    daily.loc[1, "pullback_atr5"] = 1.2

    result = _assess(daily)

    assert result.decision.trigger is None
    assert result.entry.code == "wait"


def test_weak_market_blocks_new_entries_but_not_holding():
    daily = _daily()
    daily["mkt_close"] = [460.0, 460.0]

    result = _assess(daily)

    assert result.entry.code == "wait"
    assert result.entry_attractiveness == "大盘弱势，暂缓新开仓"
    assert result.position.code == "hold"


def test_missing_market_data_is_reported_not_silently_passed():
    daily = _daily()
    for column in ("mkt_close", "mkt_sma50", "mkt_sma200", "mkt_ret63", "mkt_ret126"):
        daily[column] = np.nan

    result = _assess(daily)

    assert result.entry.code == "buy"
    assert any("大盘环境" in item for item in result.entry.conflicts)
    regime = next(item for item in result.score.components if item.name == "market_regime")
    assert regime.value is None and regime.effective_weight == 0


def test_upcoming_earnings_blocks_entry():
    context = MarketContext(next_earnings=date(2025, 5, 7), earnings_checked=True)

    result = _assess(context=context)

    assert result.entry.code == "wait"
    assert result.entry_attractiveness == "临近财报，等待事件落地"


def test_price_below_sma200_blocks_entry():
    daily = _daily()
    daily["sma200"] = [102.0, 102.0]

    assert _assess(daily).entry.code == "wait"


def test_low_score_blocks_entry():
    daily = _daily()
    daily["ret63"] = [-0.30, -0.30]
    daily["ret126"] = [-0.30, -0.30]
    daily["updown_volume20"] = [0.5, 0.5]
    daily["mkt_sma50"] = [480.0, 460.0]
    daily["rsi14"] = [51.0, 52.0]

    result = _assess(daily)

    assert result.score.score < 60
    assert result.entry.code == "wait"


def test_score_weights_sum_and_breakdown_are_reported():
    result = _assess()

    assert result.score.score is not None and 60 <= result.score.score <= 100
    assert abs(sum(item.effective_weight for item in result.score.components) - 1) < 1e-9
    assert {item.name for item in result.score.components} == {
        "trend", "relative_strength", "market_regime", "momentum", "location", "volume", "reward_risk"}


def test_two_closes_below_sma20_reduce_without_selling():
    daily = _daily()
    daily["close"] = [99.5, 99.6]

    result = _assess(daily)

    assert result.position.code == "reduce"


def test_chandelier_stop_sells_inside_an_intact_ma_structure():
    daily = _daily()
    daily["high22"] = [108.0, 108.0]

    result = _assess(daily)

    assert result.daily.code == "strong_bullish"
    assert result.position.code == "reduce_sell"
    assert any("吊灯止损" in item for item in result.position.conditions)
