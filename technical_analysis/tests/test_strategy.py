"""Checks for the added daily features, benchmarks, context fetch and backtest."""
from datetime import date, datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from analysis.backtest import run_backtest, summarize
from analysis.indicators import daily_indicators, wilder_atr
from analysis.strategy import (StrategyConfig, attach_benchmarks, entry_limits, evaluate_bar, fib_levels,
                               reward_risk, technical_score)
from analysis.support_resistance.engine import _historical_interactions
from analysis.support_resistance.models import CandidateZone, SRConfig
from config import Settings
from data.fmp_client import FMPClient, _next_earnings
from data.normalize import session_schedule
from services.analysis_service import calendar_indicators

NY = ZoneInfo("America/New_York")
SECRET = "secret-that-must-never-appear"


def _bars(start="2023-01-03", end="2025-06-30", seed=3, drift=0.0007):
    schedule = session_schedule(start, end)
    rng = np.random.default_rng(seed)
    count = len(schedule)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.015, count) + 0.004 * np.sin(np.arange(count) / 12)))
    opening = close * np.exp(rng.normal(0, 0.004, count))
    return pd.DataFrame({
        "session": list(schedule.index), "timestamp": list(schedule["market_close"]),
        "bar_end": list(schedule["market_close"]), "complete": True, "open": opening,
        "high": np.maximum(opening, close) * 1.006, "low": np.minimum(opening, close) * 0.994,
        "close": close, "volume": rng.integers(1_000_000, 3_000_000, count).astype(float),
    })


def test_wilder_atr_seeds_with_mean_then_smooths_and_restarts_after_gap():
    high = pd.Series([11.0, 12, 13, np.nan, 12, 13, 14])
    low = pd.Series([9.0, 10, 11, np.nan, 10, 11, 12])
    close = pd.Series([10.0, 11, 12, np.nan, 11, 12, 13])
    atr = wilder_atr(high, low, close, period=2)
    assert np.isnan(atr[0])
    assert atr[1] == pytest.approx((2 + 2) / 2)
    assert atr[2] == pytest.approx((2 * 1 + 2) / 2)
    assert np.isnan(atr[3]) and np.isnan(atr[4])
    assert atr[5] == pytest.approx(2.0)


def test_fibonacci_leg_and_levels_follow_the_latest_swing():
    high = [10, 12, 15, 20, 18, 16]
    frame = pd.DataFrame({"close": high, "high": high, "low": [8, 9, 11, 17, 15, 14]})
    row = daily_indicators(frame, fib_lookback=5).iloc[-1]
    assert (row["fib_high"], row["fib_low"], bool(row["fib_up"])) == (20, 9, True)
    levels = dict(fib_levels(row).levels)
    assert levels[0.5] == pytest.approx(14.5)
    assert levels[0.618] == pytest.approx(20 - 0.618 * 11)


def test_updown_volume_ratio_and_relative_volume_use_only_prior_average():
    close = pd.Series(np.r_[100, np.tile([101.0, 100.0], 15)])
    volume = pd.Series(np.r_[1000, np.tile([3000.0, 1000.0], 15)])
    result = daily_indicators(pd.DataFrame({"close": close, "volume": volume}))
    assert result["updown_volume20"].iloc[-1] == pytest.approx(3.0)
    assert result["vol_avg20"].iloc[-1] == pytest.approx(volume.iloc[-21:-1].mean())


def test_calendar_indicators_add_features_without_bridging_gaps():
    bars = _bars()
    gapped = bars.drop(index=300).reset_index(drop=True)
    result, missing = calendar_indicators(gapped)
    assert missing == 1
    after_gap = result.iloc[300:305]
    assert after_gap["atr14"].isna().all() and after_gap["sma200"].isna().all()
    assert result.iloc[-1][["sma200", "atr14", "updown_volume20", "fib_high", "pullback_atr5"]].notna().all()


def test_benchmarks_ignore_forming_bars():
    stock, _ = calendar_indicators(_bars())
    market = _bars(seed=9)
    market.loc[market.index[-1], ["close", "complete"]] = [1.0, False]
    attached = attach_benchmarks(stock, market, None)
    assert np.isnan(attached["mkt_close"].iloc[-1])
    assert attached["mkt_sma200"].iloc[-2] > 0
    assert attached["sec_ret63"].isna().all()


def _row(**values):
    return pd.Series(values, dtype=float)


def test_prior_highs_and_volume_ratio_exclude_today():
    count = 70
    high = pd.Series(np.r_[np.full(count - 1, 100.0), 130.0])
    volume = pd.Series(np.r_[np.full(count - 5, 1000.0), np.full(5, 2000.0)])
    frame = pd.DataFrame({"close": high - 1, "high": high, "low": high - 2, "volume": volume})
    row = daily_indicators(frame).iloc[-1]
    assert row["prior_high20"] == row["prior_high60"] == 100.0
    assert np.isnan(row["prior_high126"]) and np.isnan(row["prior_high252"])
    assert row["vol_ratio5_20"] == pytest.approx(2000 / 1250)


def test_reward_risk_measures_to_the_nearest_prior_high_above():
    row = _row(close=100.0, atr14=2.0, prior_high20=99.0, prior_high60=106.0, prior_high126=110.0, prior_high252=120.0)
    rr = reward_risk(row, 97.0)
    assert (rr.resistance, rr.lookback, rr.ratio) == (106.0, 60, pytest.approx(2.0))
    score = technical_score(row, row, StrategyConfig(), chandelier=97.0)
    component = next(item for item in score.components if item.name == "reward_risk")
    assert component.value == pytest.approx(0.5)
    assert "106.00" in component.detail and "60" in component.detail


def test_reward_risk_without_resistance_or_stop():
    clear = _row(close=130.0, prior_high20=120.0, prior_high60=125.0, prior_high126=np.nan, prior_high252=np.nan)
    rr = reward_risk(clear, 120.0)
    assert rr.resistance is None and rr.ratio is None and rr.checked == 60
    score = technical_score(clear, clear, StrategyConfig(), chandelier=120.0)
    assert next(item for item in score.components if item.name == "reward_risk").value == 1.0
    assert reward_risk(clear, None) is None
    assert reward_risk(_row(close=100.0), 95.0) is None


def _setup_rows(**latest):
    """Two bars in a strict uptrend; latest fields can be overridden per test."""
    prior = _row(close=100.0, sma20=95.8, sma50=95.0)
    values = dict(close=101.0, sma20=99.5, sma50=95.5, rsi14=60.0, atr14=2.0, high22=102.0, prev_high=100.5,
                  pullback_atr5=0.3, prior_high20=102.0, prior_high60=102.0, prior_high126=110.0,
                  prior_high252=120.0, vol_ratio5_20=1.0)
    values.update(latest)
    return prior, _row(**values)


def _gate(decision, name):
    return next(gate for gate in decision.gates if gate.name == name)


GATED = StrategyConfig(use_reward_risk_gate=True)


def test_reward_risk_gate_blocks_entries_too_close_to_a_prior_high():
    # Stop 102 - 3*2 = 96, risk 5; the 20-day high at 102 leaves only 1 of upside.
    prior, latest = _setup_rows()
    decision = evaluate_bar(prior, latest, GATED)
    assert decision.trigger == "pullback"
    gate = _gate(decision, "reward_risk")
    assert gate.status == "fail" and gate.required and "0.20" in gate.detail
    assert decision.reward_risk.resistance == 102.0
    assert not decision.buy
    # Above the 20/60-day highs the next one is the 126-day high at 112: 11 / 5 = 2.2.
    prior, latest = _setup_rows(prior_high20=100.9, prior_high60=100.9, prior_high126=112.0)
    assert _gate(evaluate_bar(prior, latest, GATED), "reward_risk").status == "pass"
    prior, latest = _setup_rows(prior_high20=100.9, prior_high60=100.9, prior_high126=np.nan, prior_high252=np.nan)
    assert _gate(evaluate_bar(prior, latest, GATED), "reward_risk").status == "pass"
    prior, latest = _setup_rows(atr14=np.nan)
    assert _gate(evaluate_bar(prior, latest, GATED), "reward_risk").status == "unknown"


def test_reward_risk_is_scored_but_not_required_by_default():
    prior, latest = _setup_rows()
    decision = evaluate_bar(prior, latest)
    assert "reward_risk" not in {gate.name for gate in decision.gates}
    assert decision.reward_risk.ratio == pytest.approx(0.2)
    assert next(item for item in decision.score.components if item.name == "reward_risk").value == 0.0


def test_volume_breakout_gets_its_own_trigger_and_wider_limits():
    # Close 101 above the 60-day high 100 on 1.8x volume, RSI 76, 2.5 ATR above SMA20.
    hot = dict(sma20=96.0, rsi14=76.0, prior_high20=100.0, prior_high60=100.0, prior_high126=113.0,
               vol_ratio5_20=1.8)
    prior, latest = _setup_rows(**hot)
    decision = evaluate_bar(prior, latest, GATED)
    assert decision.trigger == "breakout"
    assert {name: _gate(decision, name).status for name in ("rsi_range", "extension", "trigger", "reward_risk")} == \
        {"rsi_range": "pass", "extension": "pass", "trigger": "pass", "reward_risk": "pass"}
    assert "1.80" in _gate(decision, "trigger").detail
    assert entry_limits(StrategyConfig(), "breakout") == (80.0, 3.0)
    assert entry_limits(StrategyConfig(), "pullback") == (70.0, 1.5)
    # Without enough volume the same bar is an ordinary pullback, too hot and too extended.
    prior, latest = _setup_rows(**{**hot, "vol_ratio5_20": 1.2})
    quiet = evaluate_bar(prior, latest)
    assert quiet.trigger == "pullback"
    assert (_gate(quiet, "rsi_range").status, _gate(quiet, "extension").status) == ("fail", "fail")
    prior, latest = _setup_rows(**hot)
    assert evaluate_bar(prior, latest, StrategyConfig(use_breakout_setup=False)).trigger == "pullback"


def test_chandelier_exit_can_be_switched_off_for_the_ablation():
    prior, latest = _setup_rows(high22=110.0)  # stop 110 - 3*2 = 104 is above the close
    assert any("吊灯止损" in reason for reason in evaluate_bar(prior, latest).exit_reasons)
    relaxed = evaluate_bar(prior, latest, StrategyConfig(use_chandelier_exit=False))
    assert not any("吊灯止损" in reason for reason in relaxed.exit_reasons)
    assert relaxed.chandelier == pytest.approx(104.0)


def test_disabled_gates_are_omitted():
    stock, _ = calendar_indicators(_bars())
    prior, latest = stock.iloc[-2], stock.iloc[-1]
    names = {gate.name for gate in evaluate_bar(prior, latest).gates}
    assert {"long_trend", "market_regime", "earnings"} <= names
    config = StrategyConfig(use_long_trend_gate=False, use_market_gate=False, use_earnings_gate=False)
    assert not {"long_trend", "market_regime", "earnings"} & {gate.name for gate in evaluate_bar(prior, latest, config).gates}


def test_backtest_never_uses_future_bars():
    stock, _ = calendar_indicators(_bars())
    stock = attach_benchmarks(stock, _bars(seed=11, drift=0.0004), None)
    full = run_backtest("TEST", stock)
    assert full.trades, "fixture should trade"
    cut = full.trades[0].exit_session or full.trades[0].entry_session
    altered = _bars()
    later = altered["session"] > cut
    altered.loc[later, ["open", "high", "low", "close"]] *= 0.5
    altered_stock, _ = calendar_indicators(altered)
    altered_stock = attach_benchmarks(altered_stock, _bars(seed=11, drift=0.0004), None)
    changed = run_backtest("TEST", altered_stock)
    assert changed.trades[0].entry_session == full.trades[0].entry_session
    assert changed.trades[0].entry_price == full.trades[0].entry_price


def test_backtest_summary_reports_trades_and_score_buckets():
    stock, _ = calendar_indicators(_bars())
    results = [run_backtest("TEST", attach_benchmarks(stock, _bars(seed=11), None), cost_bps=10)]
    summary = summarize(results)
    assert summary["symbols"].iloc[0]["trades"] == len(results[0].trades)
    assert summary["score_buckets"]["observations"].sum() > 100
    assert set(summary["trades"].columns) >= {"entry_session", "exit_session", "weighted_return", "exit_reason", "setup"}
    setups = summary["setups"]
    assert list(setups["setup"]) == ["放量突破", "站回 SMA20", "回踩"]
    assert setups["trades"].sum() == len(results[0].trades)
    assert all(trade.setup in {"breakout", "reclaim", "pullback"} for trade in results[0].trades)


def test_support_tests_record_whether_the_zone_held():
    zone = CandidateZone("z", 99.0, 101.0, 100.0, ("swing",), 0)
    closes = [105, 104, 100.5, 104, 105, 106, 107, 108, 104, 99.5, 97]
    frame = pd.DataFrame({
        "session": pd.date_range("2025-01-01", periods=len(closes)).date,
        "close": closes, "high": [value + 0.5 for value in closes], "low": [value - 0.6 for value in closes],
        "atr": 1.0, "segment_id": 0,
    })
    frame.loc[2, "low"] = 100.0
    tests = _historical_interactions(frame, zone, SRConfig(test_cooldown_sessions=2, reaction_horizon_sessions=1))
    assert [item.held for item in tests] == [True, False]


def test_next_earnings_picks_the_first_future_date():
    rows = [{"date": "2025-01-30"}, {"date": "2025-04-30"}, {"date": "bad"}, {"date": "2025-07-30"}]
    assert _next_earnings(rows, date(2025, 2, 1)) == date(2025, 4, 30)
    assert _next_earnings(rows, date(2025, 8, 1)) is None


class _Response:
    def __init__(self, payload, status=200):
        self.payload, self.status_code = payload, status

    def json(self):
        return self.payload


class _RoutingSession:
    def __init__(self, routes):
        self.routes, self.calls, self.trust_env = routes, [], True

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs.get("params", {}).get("symbol")))
        endpoint = url.split("/stable/")[1]
        return self.routes(endpoint, kwargs["params"]["symbol"])


def test_fetch_context_combines_spy_sector_and_earnings_without_leaking_keys():
    daily = [{"date": str(day), "open": 100, "high": 101, "low": 99, "close": 100, "volume": 1000}
             for day in session_schedule("2025-03-03", "2025-05-02").index]

    def routes(endpoint, symbol):
        if endpoint == "profile":
            return _Response([{"symbol": symbol, "sector": "Technology", "apiKey": SECRET}])
        if endpoint == "earnings":
            return _Response([{"symbol": symbol, "date": "2025-05-08"}, {"symbol": symbol, "date": "2025-01-30"}])
        return _Response(daily)

    session = _RoutingSession(routes)
    bundle = FMPClient(Settings(api_key=SECRET), session).fetch_context(
        "NVDA", now=datetime(2025, 5, 5, 10, tzinfo=NY))
    assert bundle.context.sector == "Technology" and bundle.context.sector_etf == "XLK"
    assert bundle.context.next_earnings == date(2025, 5, 8) and bundle.context.earnings_checked
    assert not bundle.market.empty and not bundle.sector.empty
    assert {symbol for _, symbol in session.calls} == {"SPY", "NVDA", "XLK"}
    assert SECRET not in repr(bundle.context)


def test_fetch_context_degrades_when_endpoints_are_denied():
    session = _RoutingSession(lambda endpoint, symbol: _Response({"Error Message": "premium"}, 403))
    bundle = FMPClient(Settings(api_key=SECRET), session).fetch_context(
        "NVDA", now=datetime(2025, 5, 5, 10, tzinfo=NY))
    assert bundle.market is None and bundle.sector is None
    assert not bundle.context.earnings_checked
    assert len(bundle.context.notes) == 3
    assert all(SECRET not in note for note in bundle.context.notes)
