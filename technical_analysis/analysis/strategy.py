"""Shared bar-level rules: daily bias, entry gates, exits and the weighted score.

The live assessment (`analysis/signals.py`) and the backtest
(`scripts/backtest.py`) both call `evaluate_bar`, so a historical test runs
exactly the rules the dashboard shows. Every threshold and weight lives in
`StrategyConfig`. Weights are reasoned priors, not fitted parameters; use the
backtest to check them before trusting them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import math

import pandas as pd

from analysis.indicators import BREAKOUT_LOOKBACK, RESISTANCE_LOOKBACKS, daily_indicators


FIB_RATIOS = (0.236, 0.382, 0.5, 0.618, 0.786)
FIB_ENTRY_RATIOS = (0.382, 0.5, 0.618)

SECTOR_ETFS = {
    "Technology": "XLK", "Healthcare": "XLV", "Financial Services": "XLF",
    "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP", "Energy": "XLE",
    "Industrials": "XLI", "Basic Materials": "XLB", "Real Estate": "XLRE",
    "Utilities": "XLU", "Communication Services": "XLC",
}


@dataclass(frozen=True)
class ScoreWeights:
    trend: float = 0.25
    relative_strength: float = 0.20
    market_regime: float = 0.15
    momentum: float = 0.10
    location: float = 0.10
    volume: float = 0.10
    reward_risk: float = 0.10

    def __post_init__(self):
        values = tuple(vars(self).values())
        if any(not 0 <= value <= 1 for value in values) or sum(values) <= 0:
            raise ValueError("Score weights must be nonnegative fractions with a positive sum")


@dataclass(frozen=True)
class StrategyConfig:
    strong_rsi: float = 55.0
    weak_rsi: float = 45.0
    entry_rsi_low: float = 50.0
    entry_rsi_high: float = 70.0
    max_extension_atr: float = 1.5
    fallback_max_extension: float = 0.03
    pullback_touch_atr: float = 0.5
    chandelier_atr: float = 3.0
    earnings_blackout_days: int = 7
    buy_min_score: float = 60.0
    min_score_coverage: float = 0.5
    fib_proximity_atr: float = 0.5
    rs_full_scale: float = 0.10
    breakout_volume_ratio: float = 1.5
    breakout_rsi_high: float = 80.0
    breakout_max_extension_atr: float = 3.0
    min_reward_risk: float = 2.0
    use_breakout_setup: bool = True
    use_reward_risk_gate: bool = True
    use_long_trend_gate: bool = True
    use_market_gate: bool = True
    use_earnings_gate: bool = True
    use_chandelier_exit: bool = True
    weights: ScoreWeights = field(default_factory=ScoreWeights)


TRIGGER_LABELS = {
    "breakout": f"放量突破 {BREAKOUT_LOOKBACK} 日前高",
    "reclaim": "重新站上 SMA20",
    "pullback": "回踩 SMA20 后突破前一日高点",
}
SETUP_SHORT = {"breakout": "放量突破", "reclaim": "站回 SMA20", "pullback": "回踩"}


def entry_limits(config: StrategyConfig, trigger: str | None) -> tuple[float, float]:
    """RSI ceiling and maximum ATR distance above SMA20 for this setup.

    A breakout is expected to run hotter and further from SMA20 than a
    pullback entry, so it gets its own (wider) limits.
    """
    if trigger == "breakout":
        return config.breakout_rsi_high, config.breakout_max_extension_atr
    return config.entry_rsi_high, config.max_extension_atr


SCORE_LABELS = {
    "trend": "趋势结构", "relative_strength": "相对强度", "market_regime": "大盘环境",
    "momentum": "RSI 动量", "location": "回调位置", "volume": "量价配合", "reward_risk": "盈亏比",
}


@dataclass(frozen=True)
class ScoreComponent:
    name: str
    label: str
    value: float | None
    weight: float
    effective_weight: float
    contribution: float | None
    detail: str


@dataclass(frozen=True)
class TechnicalScore:
    score: float | None
    coverage: float
    grade: str
    components: tuple[ScoreComponent, ...]


@dataclass(frozen=True)
class Gate:
    name: str
    status: str  # "pass", "fail" or "unknown"
    required: bool
    detail: str


@dataclass(frozen=True)
class FibLevels:
    high: float
    low: float
    up: bool
    levels: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class RewardRisk:
    """Room to the nearest prior high above the close versus the drop to the stop.

    `resistance` is None when the close is above every prior high checked
    (`checked` sessions back); `ratio` is then None as well.
    """
    stop: float
    resistance: float | None
    lookback: int | None
    checked: int
    ratio: float | None


@dataclass(frozen=True)
class BarDecision:
    daily_code: str
    strict_up: bool
    strict_down: bool
    trigger: str | None
    gates: tuple[Gate, ...]
    buy: bool
    exit: str | None
    exit_reasons: tuple[str, ...]
    score: TechnicalScore
    fib: FibLevels | None
    chandelier: float | None
    reward_risk: RewardRisk | None = None


def _num(row, name: str) -> float | None:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError, OverflowError):
        return None
    return value if math.isfinite(value) else None


def _clip(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def attach_benchmarks(daily: pd.DataFrame, market: pd.DataFrame | None = None,
                      sector: pd.DataFrame | None = None) -> pd.DataFrame:
    """Map completed benchmark closes and their trend/returns onto stock sessions.

    Only completed benchmark bars are used, so a forming SPY candle never
    enters a confirmed comparison. Sessions the benchmark lacks stay NaN.
    """
    result = daily.copy()
    for prefix, frame, columns in (("mkt", market, ("close", "sma50", "sma200", "ret63", "ret126")),
                                   ("sec", sector, ("close", "ret63", "ret126"))):
        for column in columns:
            result[f"{prefix}_{column}"] = float("nan")
        if frame is None or frame.empty or not {"session", "close"}.issubset(frame.columns):
            continue
        completed = frame.loc[frame["complete"].astype(bool)] if "complete" in frame else frame
        if completed.empty:
            continue
        calculated = daily_indicators(completed[["session", "close"]].reset_index(drop=True)).set_index("session")
        for column in columns:
            result[f"{prefix}_{column}"] = result["session"].map(calculated[column]).astype(float)
    return result


def fib_levels(row) -> FibLevels | None:
    high, low, up = _num(row, "fib_high"), _num(row, "fib_low"), _num(row, "fib_up")
    if high is None or low is None or up is None or not high > low > 0:
        return None
    span = high - low
    levels = tuple((ratio, high - ratio * span if up else low + ratio * span) for ratio in FIB_RATIOS)
    return FibLevels(high, low, bool(up), levels)


def classify_daily(prior, latest, config: StrategyConfig) -> tuple[str, bool, bool]:
    """Return the daily bias code plus strict-uptrend and strict-downtrend flags."""
    close, sma20, sma50, rsi = (float(latest[name]) for name in ("close", "sma20", "sma50", "rsi14"))
    slope20 = sma20 - float(prior["sma20"])
    slope50 = sma50 - float(prior["sma50"])
    strict_up = close > sma20 > sma50 and slope20 > 0 and slope50 > 0
    strict_down = close < sma20 < sma50 and slope20 < 0 and slope50 < 0
    if strict_up and rsi >= config.strong_rsi:
        return "strong_bullish", strict_up, strict_down
    if close > sma50 and sma20 > sma50 and slope50 >= 0:
        return "bullish", strict_up, strict_down
    if strict_down and rsi <= config.weak_rsi:
        return "strong_bearish", strict_up, strict_down
    if close < sma50 and sma20 < sma50 and slope50 <= 0:
        return "bearish", strict_up, strict_down
    return "neutral", strict_up, strict_down


def reward_risk(latest, chandelier: float | None) -> RewardRisk | None:
    """Nearest prior high above the close (20/60/126/252 sessions) against the chandelier stop.

    Prior highs are the resistance here, rather than the support/resistance
    zones, so the backtest can evaluate exactly the same rule bar by bar.
    """
    close = _num(latest, "close")
    known = [(period, _num(latest, f"prior_high{period}")) for period in RESISTANCE_LOOKBACKS]
    known = [(period, level) for period, level in known if level is not None]
    if close is None or chandelier is None or not known:
        return None
    checked = max(period for period, _ in known)
    above = [(period, level) for period, level in known if level > close]
    if not above:
        return RewardRisk(chandelier, None, None, checked, None)
    lookback, level = min(above, key=lambda item: item[1])
    risk = close - chandelier
    return RewardRisk(chandelier, level, lookback, checked, (level - close) / risk if risk > 0 else 0.0)


def _trend_value(prior, latest) -> tuple[float | None, str]:
    close, sma20, sma50, sma200 = (_num(latest, name) for name in ("close", "sma20", "sma50", "sma200"))
    p20, p50 = _num(prior, "sma20"), _num(prior, "sma50")
    checks = {
        "收盘>SMA20": None if None in (close, sma20) else close > sma20,
        "SMA20>SMA50": None if None in (sma20, sma50) else sma20 > sma50,
        "收盘>SMA200": None if None in (close, sma200) else close > sma200,
        "SMA20上升": None if None in (sma20, p20) else sma20 > p20,
        "SMA50上升": None if None in (sma50, p50) else sma50 > p50,
    }
    known = {name: value for name, value in checks.items() if value is not None}
    if len(known) < 3:
        return None, "均线数据不足"
    passed = [name for name, value in known.items() if value]
    return len(passed) / len(known), f"满足 {len(passed)}/{len(known)}：" + ("、".join(passed) or "无")


def _relative_strength_value(latest, config: StrategyConfig) -> tuple[float | None, str]:
    pairs = []
    for benchmark, label in (("mkt", "SPY"), ("sec", "行业ETF")):
        for period in (63, 126):
            own, other = _num(latest, f"ret{period}"), _num(latest, f"{benchmark}_ret{period}")
            if own is not None and other is not None:
                pairs.append((f"{period}日相对{label} {own - other:+.1%}", own - other))
    if not pairs:
        return None, "缺少基准指数数据"
    excess = sum(value for _, value in pairs) / len(pairs)
    return _clip(0.5 + excess / (2 * config.rs_full_scale)), "；".join(text for text, _ in pairs)


def _regime_value(latest) -> tuple[float | None, str]:
    close, sma50, sma200 = (_num(latest, f"mkt_{name}") for name in ("close", "sma50", "sma200"))
    if None in (close, sma50, sma200):
        return None, "缺少 SPY 200 日均线"
    value = (0.6 if close > sma200 else 0.0) + (0.4 if sma50 > sma200 else 0.0)
    return value, f"SPY {close:.2f}，SMA50 {sma50:.2f}，SMA200 {sma200:.2f}"


def _momentum_value(latest) -> tuple[float | None, str]:
    rsi = _num(latest, "rsi14")
    if rsi is None:
        return None, "RSI 不可用"
    if 55 <= rsi <= 65:
        value = 1.0
    elif 40 < rsi < 55:
        value = (rsi - 40) / 15
    elif 65 < rsi < 80:
        value = (80 - rsi) / 15
    else:
        value = 0.0
    return value, f"RSI14 {rsi:.1f}（55–65 最佳）"


def _location_value(latest, fib: FibLevels | None, config: StrategyConfig) -> tuple[float | None, str]:
    close, sma20, atr = _num(latest, "close"), _num(latest, "sma20"), _num(latest, "atr14")
    if None in (close, sma20, atr) or atr <= 0:
        return None, "ATR 或 SMA20 不可用"
    distance = (close - sma20) / atr
    if distance < -0.5:
        value = 0.0
    elif distance < 0:
        value = 0.6 * (distance + 0.5) / 0.5
    elif distance <= 0.75:
        value = 1.0
    elif distance < 2.0:
        value = (2.0 - distance) / 1.25
    else:
        value = 0.0
    detail = f"距 SMA20 {distance:+.2f} ATR"
    if fib is not None and fib.up:
        near = [ratio for ratio, level in fib.levels
                if ratio in FIB_ENTRY_RATIOS and abs(close - level) <= config.fib_proximity_atr * atr]
        if near:
            value = _clip(value + 0.25)
            detail += f"；位于斐波那契 {near[0]:.1%} 回撤附近"
    return value, detail


def _volume_value(latest) -> tuple[float | None, str]:
    ratio = _num(latest, "updown_volume20")
    if ratio is None:
        return None, "成交量数据不足"
    return _clip((ratio - 0.8) / 0.7), f"20 日上涨日/下跌日成交量比 {ratio:.2f}"


def describe_reward_risk(latest, rr: RewardRisk | None) -> str:
    if rr is None:
        return "缺少 ATR、22 日最高价或前高数据"
    close = _num(latest, "close")
    if close is not None and close <= rr.stop:
        return f"收盘 {close:.2f} 已在吊灯止损 {rr.stop:.2f} 下方"
    if rr.resistance is None:
        return f"收盘高于近 {rr.checked} 个交易日的所有前高，上方无前高阻力；吊灯止损 {rr.stop:.2f}"
    return f"上方前高 {rr.resistance:.2f}（近 {rr.lookback} 日），吊灯止损 {rr.stop:.2f}，盈亏比 {rr.ratio:.2f}"


def _reward_risk_value(latest, rr: RewardRisk | None) -> tuple[float | None, str]:
    if rr is None:
        return None, describe_reward_risk(latest, rr)
    close = _num(latest, "close")
    if close is not None and close <= rr.stop:
        value = 0.0
    elif rr.ratio is None:
        value = 1.0
    else:
        value = _clip((rr.ratio - 1) / 2)
    return value, describe_reward_risk(latest, rr)


def technical_score(prior, latest, config: StrategyConfig, fib: FibLevels | None = None,
                    chandelier: float | None = None) -> TechnicalScore:
    raw = {
        "trend": _trend_value(prior, latest),
        "relative_strength": _relative_strength_value(latest, config),
        "market_regime": _regime_value(latest),
        "momentum": _momentum_value(latest),
        "location": _location_value(latest, fib, config),
        "volume": _volume_value(latest),
        "reward_risk": _reward_risk_value(latest, reward_risk(latest, chandelier)),
    }
    weights = vars(config.weights)
    total = math.fsum(weights.values())
    available = math.fsum(weight for name, weight in weights.items() if raw[name][0] is not None)
    coverage = available / total if total else 0.0
    components = []
    for name, weight in weights.items():
        value, detail = raw[name]
        effective = weight / available if value is not None and available > 0 else 0.0
        contribution = value * effective * 100 if value is not None else None
        components.append(ScoreComponent(name, SCORE_LABELS[name], value, weight, effective, contribution, detail))
    score = (math.fsum(item.contribution or 0.0 for item in components)
             if coverage >= config.min_score_coverage and available > 0 else None)
    grade = ("不可用" if score is None else "强" if score >= 75 else "偏强" if score >= 60
             else "中性" if score >= 40 else "偏弱")
    return TechnicalScore(score, coverage, grade, tuple(components))


def evaluate_bar(prior, latest, config: StrategyConfig | None = None, *,
                 next_earnings: date | None = None, earnings_checked: bool = False) -> BarDecision:
    """Apply the entry gates, exit rules and score to two consecutive completed bars.

    The caller guarantees both rows are valid, consecutive completed sessions.
    Required gates must pass. Conditional gates (SMA200, market, earnings)
    block only when their data shows a failure; missing data is reported as
    unknown rather than silently treated as a pass.
    """
    config = config or StrategyConfig()
    code, strict_up, strict_down = classify_daily(prior, latest, config)
    close, sma20, sma50, rsi = (float(latest[name]) for name in ("close", "sma20", "sma50", "rsi14"))
    prior_close, prior_sma20, prior_sma50 = (float(prior[name]) for name in ("close", "sma20", "sma50"))
    atr = _num(latest, "atr14")
    atr = atr if atr is not None and atr > 0 else None
    fib = fib_levels(latest)
    high22 = _num(latest, "high22")
    chandelier = high22 - config.chandelier_atr * atr if high22 is not None and atr is not None else None

    reclaim = prior_close <= prior_sma20 and close > sma20
    prev_high, pullback = _num(latest, "prev_high"), _num(latest, "pullback_atr5")
    resumed = (prior_close > prior_sma20 and close > sma20 and prev_high is not None and close > prev_high
               and pullback is not None and pullback <= config.pullback_touch_atr)
    breakout_level, volume_ratio = _num(latest, f"prior_high{BREAKOUT_LOOKBACK}"), _num(latest, "vol_ratio5_20")
    breakout = (config.use_breakout_setup and breakout_level is not None and volume_ratio is not None
                and close > breakout_level and volume_ratio >= config.breakout_volume_ratio)
    trigger = "breakout" if breakout else "reclaim" if reclaim else "pullback" if resumed else None
    rsi_high, max_extension = entry_limits(config, trigger)

    gates = [Gate("strict_trend", "pass" if strict_up else "fail", True,
                  "收盘 > SMA20 > SMA50，且两条均线均上升")]
    gates.append(Gate("rsi_range", "pass" if config.entry_rsi_low <= rsi < rsi_high else "fail", True,
                      f"RSI14 {rsi:.2f}，需在 [{config.entry_rsi_low:g}, {rsi_high:g})"
                      + ("（突破形态上限）" if trigger == "breakout" else "")))
    if atr is not None:
        extension = (close - sma20) / atr
        gates.append(Gate("extension", "pass" if 0 <= extension <= max_extension else "fail", True,
                          f"距 SMA20 {extension:+.2f} ATR，需在 0–{max_extension:g} ATR"
                          + ("（突破形态上限）" if trigger == "breakout" else "")))
    else:
        extension = close / sma20 - 1
        gates.append(Gate("extension", "pass" if 0 <= extension <= config.fallback_max_extension else "fail", True,
                          f"距 SMA20 {extension:+.2%}（ATR 不可用，按 {config.fallback_max_extension:.0%} 上限）"))
    if trigger == "breakout":
        trigger_detail = (f"收盘 {close:.2f} 突破前 {BREAKOUT_LOOKBACK} 日最高价 {breakout_level:.2f}，"
                          f"5 日均量为 20 日均量的 {volume_ratio:.2f} 倍")
    elif trigger:
        trigger_detail = TRIGGER_LABELS[trigger]
    else:
        waiting = ["重新站上 SMA20", "回踩 SMA20 后收盘突破前一日高点"]
        if config.use_breakout_setup:
            waiting.append(f"放量（5 日均量 ≥ {config.breakout_volume_ratio:g} × 20 日均量）突破 {BREAKOUT_LOOKBACK} 日前高")
        trigger_detail = "等待" + "、".join(waiting[:-1]) + "，或" + waiting[-1]
    gates.append(Gate("trigger", "pass" if trigger else "fail", True, trigger_detail))
    rr = reward_risk(latest, chandelier)
    if config.use_reward_risk_gate:
        if rr is None:
            rr_status = "unknown"
        elif close <= rr.stop:
            rr_status = "fail"
        elif rr.ratio is None:
            rr_status = "pass"
        else:
            rr_status = "pass" if rr.ratio >= config.min_reward_risk else "fail"
        rr_detail = describe_reward_risk(latest, rr)
        if rr is not None and rr.ratio is not None and close > rr.stop:
            rr_detail += f"，需至少 {config.min_reward_risk:g}"
        gates.append(Gate("reward_risk", rr_status, True, rr_detail))
    sma200 = _num(latest, "sma200")
    if config.use_long_trend_gate:
        gates.append(Gate("long_trend", "unknown" if sma200 is None else "pass" if close > sma200 else "fail", False,
                          "SMA200 不可用" if sma200 is None else f"收盘 {close:.2f} 对 SMA200 {sma200:.2f}"))
    market_close, market_sma200 = _num(latest, "mkt_close"), _num(latest, "mkt_sma200")
    if config.use_market_gate:
        gates.append(Gate("market_regime",
                          "unknown" if None in (market_close, market_sma200) else
                          "pass" if market_close > market_sma200 else "fail", False,
                          "SPY 数据不可用" if None in (market_close, market_sma200) else
                          f"SPY {market_close:.2f} 对其 SMA200 {market_sma200:.2f}"))
    session = latest["session"] if "session" in latest else None
    if not config.use_earnings_gate:
        pass
    elif not earnings_checked or session is None:
        gates.append(Gate("earnings", "unknown", False, "财报日期不可用"))
    elif next_earnings is None:
        gates.append(Gate("earnings", "pass", False, "未查到近期财报日期"))
    else:
        days = (next_earnings - session).days
        inside = 0 <= days <= config.earnings_blackout_days
        gates.append(Gate("earnings", "fail" if inside else "pass", False,
                          f"下次财报 {next_earnings}（距 {days} 个自然日；{config.earnings_blackout_days} 日内不新开仓）"))
    score = technical_score(prior, latest, config, fib, chandelier)
    gates.append(Gate("score", "unknown" if score.score is None else
                      "pass" if score.score >= config.buy_min_score else "fail", True,
                      "综合评分不可用" if score.score is None else
                      f"综合评分 {score.score:.0f}，需至少 {config.buy_min_score:.0f}"))
    buy = all(gate.status == "pass" if gate.required else gate.status != "fail" for gate in gates)

    sell_reasons = []
    if code in ("bearish", "strong_bearish"):
        sell_reasons.append("日线判断转为偏空")
    if prior_close < prior_sma50 and close < sma50:
        sell_reasons.append("连续两日收盘低于 SMA50")
    if config.use_chandelier_exit and chandelier is not None and close < chandelier:
        sell_reasons.append(f"收盘 {close:.2f} 跌破吊灯止损 {chandelier:.2f}（22 日最高价 − {config.chandelier_atr:g} ATR）")
    if sell_reasons:
        exit_kind, reasons = "sell", sell_reasons
    elif prior_close < prior_sma20 and close < sma20:
        exit_kind, reasons = "reduce", ["连续两日收盘低于 SMA20，短线动能减弱"]
    else:
        exit_kind, reasons = None, []
    return BarDecision(code, strict_up, strict_down, trigger, tuple(gates), buy, exit_kind,
                       tuple(reasons), score, fib, chandelier, rr)


__all__ = ["BarDecision", "FibLevels", "Gate", "RewardRisk", "SECTOR_ETFS", "SETUP_SHORT", "ScoreComponent",
           "ScoreWeights", "StrategyConfig", "TRIGGER_LABELS", "TechnicalScore", "attach_benchmarks",
           "classify_daily", "describe_reward_risk", "entry_limits", "evaluate_bar", "fib_levels", "reward_risk",
           "technical_score"]
