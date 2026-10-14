"""Deterministic technical heuristics over completed, dated market bars.

Daily trend, RSI momentum, and current-session timing remain separate; no
portfolio facts are inferred. The bar-level rules and the weighted score live
in `analysis/strategy.py`, shared with the backtest. The caller supplies
already-calculated daily features (see `daily_indicators`) and session VWAP.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from math import isfinite

import pandas as pd

from analysis.strategy import (TRIGGER_LABELS, BarDecision, StrategyConfig, TechnicalScore, classify_daily,
                               entry_limits, evaluate_bar)
from data.normalize import session_schedule


TIMEFRAME = "日线／波段观察（约 5–20 个交易日）；按已完成日线决策，5 分钟仅用于避免追高。"
REFERENCE_NOTE = "均线与 VWAP 是随新K线更新的计算参考位，不是固定止损价或目标价。"


@dataclass(frozen=True)
class BiasAssessment:
    code: str
    label: str
    explanation: str
    evidence: tuple[str, ...]
    conflicts: tuple[str, ...]
    as_of: datetime | None
    session: date | None


@dataclass(frozen=True)
class ActionGuidance:
    code: str
    label: str
    explanation: str
    timeframe: str
    trigger_status: str
    conditions: tuple[str, ...]
    invalidation: tuple[str, ...]
    conflicts: tuple[str, ...]


@dataclass(frozen=True)
class TechnicalAssessment:
    daily: BiasAssessment
    intraday: BiasAssessment
    momentum_label: str
    entry_attractiveness: str
    entry: ActionGuidance
    position: ActionGuidance
    evaluated_at: datetime
    provisional_notes: tuple[str, ...]
    score: TechnicalScore | None = None
    decision: BarDecision | None = None
    context_notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class MarketContext:
    """Event and benchmark facts gathered outside the stock's own bars."""
    next_earnings: date | None = None
    earnings_checked: bool = False
    sector: str | None = None
    sector_etf: str | None = None
    notes: tuple[str, ...] = ()


def _local(value: datetime | pd.Timestamp) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("America/New_York") if stamp.tzinfo is None else stamp.tz_convert("America/New_York")


def _time(value: datetime | pd.Timestamp) -> str:
    return _local(value).strftime("%Y-%m-%d %H:%M %Z")


def _completed(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Never promote a cached or future observation solely by wall-clock time."""
    if frame.empty or not {"complete", "bar_end", "session"}.issubset(frame.columns):
        return frame.iloc[0:0].copy()
    result = frame.loc[frame["complete"].eq(True)].copy()
    if result.empty:
        return result
    ends = pd.to_datetime(result["bar_end"], errors="coerce", utc=True)
    result = result.loc[ends.notna() & (ends <= cutoff.tz_convert("UTC"))]
    return result.sort_values("bar_end").reset_index(drop=True)


def _positive(row: pd.Series, columns: tuple[str, ...]) -> bool:
    try:
        return all(isfinite(float(row[column])) and float(row[column]) > 0 for column in columns)
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _valid_daily(row: pd.Series) -> bool:
    try:
        rsi = float(row["rsi14"])
        return _positive(row, ("close", "sma20", "sma50")) and isfinite(rsi) and 0 <= rsi <= 100
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def _momentum(rsi: float) -> str:
    if rsi >= 70:
        return "偏强但超买"
    if rsi >= 55:
        return "偏强"
    if rsi >= 45:
        return "中性"
    if rsi > 30:
        return "偏弱"
    return "偏弱且超卖"


_BIAS_TEXT = {
    "strong_bullish": ("强烈偏多", "收盘价高于 SMA20 和 SMA50，两条均线均上升，RSI14 至少为 55。"),
    "bullish": ("偏多", "收盘价与 SMA20 均高于 SMA50，且 SMA50 未下降；尚未满足全部强势偏多条件。"),
    "strong_bearish": ("强烈偏空", "收盘价低于 SMA20 和 SMA50，两条均线均下降，RSI14 不高于 45。"),
    "bearish": ("偏空", "收盘价与 SMA20 均低于 SMA50，且 SMA50 未上升；尚未满足全部强势偏空条件。"),
    "neutral": ("中性", "价格与均线结构未形成规则定义的明确多头或空头排列。"),
}


def _daily_bias(bars: pd.DataFrame, expected_session: date | None,
                config: StrategyConfig) -> tuple[BiasAssessment, dict | None]:
    latest = None if bars.empty else bars.iloc[-1]
    stamp = None if latest is None else _local(latest["bar_end"]).to_pydatetime()
    session = None if latest is None else latest["session"]
    unavailable = BiasAssessment(
        "insufficient_data", "数据不足",
        "需要最近两个相邻 NYSE 交易日的已完成日线，并且两根K线的收盘价、SMA20、SMA50 和 RSI14 均有效。",
        () if stamp is None else (f"最近已完成日线：{_time(stamp)}。",), (), stamp, session,
    )
    if len(bars) < 2:
        return unavailable, None
    prior, latest = bars.iloc[-2], bars.iloc[-1]
    if not _valid_daily(prior) or not _valid_daily(latest):
        return unavailable, None
    schedule = session_schedule(prior["session"], latest["session"])
    if list(schedule.index) != [prior["session"], latest["session"]]:
        return unavailable, None
    close, sma20, sma50, rsi = (float(latest[column]) for column in ("close", "sma20", "sma50", "rsi14"))
    slope20 = sma20 - float(prior["sma20"])
    slope50 = sma50 - float(prior["sma50"])
    code, strict_up, strict_down = classify_daily(prior, latest, config)
    label, reason = _BIAS_TEXT[code]
    evidence = tuple(
        f"已完成日线 {_time(row['bar_end'])}：收盘 {float(row['close']):.2f}，SMA20 {float(row['sma20']):.2f}，SMA50 {float(row['sma50']):.2f}，RSI14 {float(row['rsi14']):.2f}。"
        for row in (prior, latest)
    ) + (f"相邻交易日 SMA20 变化 {slope20:+.4f}；SMA50 变化 {slope50:+.4f}。",)
    conflicts = []
    if rsi >= 70:
        conflicts.append(f"RSI14 {rsi:.2f} 已超买；趋势可以保持偏多，但当前入场需等待动量与价格距离重新符合条件。")
    elif rsi <= 30:
        conflicts.append(f"RSI14 {rsi:.2f} 已超卖；超卖本身不能确认反转或触发买入。")
    if code in ("bullish", "strong_bullish") and rsi < 50:
        conflicts.append("均线结构偏多，但 RSI14 低于 50，动量尚未确认入场。")
    if code in ("bearish", "strong_bearish") and rsi > 50:
        conflicts.append("均线结构偏空，但 RSI14 高于 50，动量与趋势方向不同。")
    if code == "neutral":
        conflicts.append("趋势条件尚不一致；不能仅凭 RSI 或单一均线位置确认方向。")
    if session != expected_session:
        reason += f" 这是 {session} 的历史判断；缺少最新应完成交易日 {expected_session} 的有效数据，操作参考不可用。"
    facts = {
        "prior": prior, "latest": latest, "close": close, "sma20": sma20, "sma50": sma50, "rsi": rsi,
        "strict_up": strict_up, "strict_down": strict_down, "gap": close / sma20 - 1,
        "fresh": session == expected_session,
    }
    return BiasAssessment(code, label, reason, evidence, tuple(conflicts), stamp, session), facts


def _intraday_bias(frame: pd.DataFrame, cutoff: pd.Timestamp, now: pd.Timestamp, market_open: bool) -> tuple[BiasAssessment, dict | None]:
    missing = BiasAssessment("unavailable", "不可用", "需要同一交易时段内连续两根已完成 5 分钟K线，以及完整覆盖的 session VWAP 和开盘价。", (), (), None, None)
    if frame.empty or "session" not in frame:
        return missing, None
    latest_session = frame["session"].max()
    bars = _completed(frame.loc[frame["session"] == latest_session], cutoff)
    if len(bars) < 2:
        return missing, None
    prior, latest = bars.iloc[-2], bars.iloc[-1]
    required_flags = ("coverage_complete", "vwap_available")
    valid = all(_positive(row, ("close", "vwap", "session_open")) and
                all(column in row and pd.notna(row[column]) and bool(row[column]) for column in required_flags)
                for row in (prior, latest))
    if not valid or _local(latest["bar_end"]) - _local(prior["bar_end"]) != pd.Timedelta("5min"):
        return missing, None
    close, opening = float(latest["close"]), float(latest["session_open"])
    above = all(float(row["close"]) > float(row["vwap"]) for row in (prior, latest))
    below = all(float(row["close"]) < float(row["vwap"]) for row in (prior, latest))
    if above and close > opening:
        code, label, reason = "bullish", "偏多", "连续两根已完成 5 分钟K线收盘均高于各自 VWAP，且最新收盘高于当日开盘价。"
    elif below and close < opening:
        code, label, reason = "bearish", "偏空", "连续两根已完成 5 分钟K线收盘均低于各自 VWAP，且最新收盘低于当日开盘价。"
    else:
        code, label, reason = "neutral", "中性", "连续两根K线的 VWAP 位置与当日开盘价尚未共同确认方向。"
    last_end = _local(latest["bar_end"])
    recent = pd.Timedelta(0) <= now - last_end <= pd.Timedelta("10min")
    current = latest_session == now.date() and market_open and recent
    conflicts = []
    if latest_session != now.date():
        conflicts.append(f"这是历史交易时段 {latest_session} 的观察，不能作为今日盘中确认。")
    elif not market_open:
        conflicts.append("当前不在 NYSE 正常交易时段内；这组K线不能触发当日新入场。")
    elif not recent:
        conflicts.append(f"最新已完成 5 分钟K线结束于 {_time(last_end)}，距评估时刻超过 10 分钟，盘中确认已过期。")
    if conflicts:
        reason += " " + conflicts[0]
    evidence = tuple(
        f"已完成 5 分钟K线 {_time(row['bar_end'])}：收盘 {float(row['close']):.2f}，session VWAP {float(row['vwap']):.2f}，当日开盘 {float(row['session_open']):.2f}；时段覆盖完整。"
        for row in (prior, latest)
    )
    facts = {"latest": latest, "close": close, "current": current}
    return BiasAssessment(code, label, reason, evidence, tuple(conflicts), last_end.to_pydatetime(), latest_session), facts


def _unavailable_actions(reason: str, daily: BiasAssessment, intraday: BiasAssessment) -> tuple[ActionGuidance, ActionGuidance]:
    guidance = ActionGuidance(
        "insufficient_data", "数据不足", reason, TIMEFRAME, "不可用",
        ("取得最新应完成交易日及其相邻前一交易日的有效收盘、SMA20、SMA50、RSI14 后重新评估。",),
        ("数据不足时不确认买入、持有或退出信号。",), daily.conflicts + intraday.conflicts,
    )
    return guidance, guidance


_GATE_MARK = {"pass": "✓", "fail": "✗", "unknown": "？"}
_GATE_NAMES = {
    "strict_trend": "严格多头排列", "rsi_range": "RSI 入场区间", "extension": "距 SMA20 不追高",
    "trigger": "日线触发", "reward_risk": "盈亏比（前高空间 ÷ 止损风险）", "long_trend": "SMA200 长期趋势",
    "market_regime": "大盘环境（SPY）", "earnings": "财报事件风险", "score": "综合评分",
}
_GATE_WAIT = {
    "strict_trend": "等待日线收盘 > SMA20 > SMA50，且两条均线均较前一交易日上升。",
    "rsi_range": "等待已完成日线 RSI14 回到入场区间。",
    "extension": "价格离 SMA20 过远，等待回调后再考虑，避免追高。",
    "trigger": "等待日线触发：重新站上 SMA20、回踩 SMA20 后收盘突破前一日高点，或放量突破 60 日前高。",
    "reward_risk": "上方前高的空间不足止损风险的要求倍数，等待回调到更低位置，或放量突破前高后再考虑。",
    "long_trend": "股价低于 SMA200，长期趋势未确认，暂不新开仓。",
    "market_regime": "SPY 低于其 SMA200，大盘处于弱势环境，暂不新开仓。",
    "earnings": "临近财报，波段持仓会承担财报跳空风险，等待财报公布后再评估。",
    "score": "综合评分不足，等待更多指标共振。",
}


def _actions(daily: BiasAssessment, intraday: BiasAssessment, facts: dict | None, intra_facts: dict | None,
             market_open: bool, expected_session: date | None, decision: BarDecision | None,
             config: StrategyConfig) -> tuple[str, ActionGuidance, ActionGuidance]:
    if facts is None or decision is None:
        entry, position = _unavailable_actions("已完成日线历史或指标不足，无法生成操作参考。", daily, intraday)
        return "数据不足", entry, position
    if not facts["fresh"]:
        entry, position = _unavailable_actions(f"最新日线为 {daily.session}；应完成交易日为 {expected_session}。请刷新或补足数据后再判断。", daily, intraday)
        return "数据不足", entry, position
    close, sma20, sma50, rsi = (facts[name] for name in ("close", "sma20", "sma50", "rsi"))
    atr = facts["latest"].get("atr14") if hasattr(facts["latest"], "get") else None
    atr = float(atr) if atr is not None and pd.notna(atr) and float(atr) > 0 else None
    conditions = [f"日线参考 {_time(daily.as_of)}：收盘 {close:.2f}，SMA20 {sma20:.2f}，SMA50 {sma50:.2f}，RSI14 {rsi:.2f}，距 SMA20 {facts['gap'] * 100:+.2f}%。"]
    for gate in decision.gates:
        suffix = "" if gate.required else "（数据缺失时不阻止，但需自行核实）" if gate.status == "unknown" else ""
        conditions.append(f"{_GATE_MARK[gate.status]} {_GATE_NAMES[gate.name]}：{gate.detail}{suffix}")
    blockers = [_GATE_WAIT[gate.name] for gate in decision.gates
                if gate.status == "fail" or (gate.required and gate.status == "unknown")]
    conflicts = list(daily.conflicts + intraday.conflicts)
    conflicts.extend(f"{_GATE_NAMES[gate.name]}无法核实：{gate.detail}。" for gate in decision.gates
                     if gate.status == "unknown" and not gate.required)
    if intra_facts is not None:
        intra_close = intra_facts["close"]
        intra_gap = (intra_close - sma20) / atr if atr else intra_close / sma20 - 1
        limit = entry_limits(config, decision.trigger)[1] if atr else config.fallback_max_extension
        gap_text = f"{intra_gap:+.2f} ATR" if atr else f"{intra_gap:+.2%}"
        conditions.append(f"盘中参考 {_time(intraday.as_of)}：已完成收盘 {intra_close:.2f}，距日线 SMA20 {gap_text}。")
        if intra_facts["current"] and intra_gap > limit:
            blockers.append(f"今日盘中价格已偏离日线 SMA20 {gap_text}，超过不追高上限；等待回落。")
        if intraday.code == "bearish" and daily.code in ("bullish", "strong_bullish"):
            conflicts.append("日线偏多但盘中偏空：入场时机可等待盘中转强；盘中信号不改变日线判断。")
    if intraday.code == "bullish" and daily.code in ("bearish", "strong_bearish"):
        conflicts.append("盘中偏多与日线偏空相冲突；盘中反弹不覆盖日线退出条件。")
    conditions.append(REFERENCE_NOTE)
    rsi_high, max_extension = entry_limits(config, decision.trigger)
    if decision.trigger:
        conditions.insert(1, f"入场形态：{TRIGGER_LABELS[decision.trigger]}。")
    entry_invalidation = (
        f"后续已完成日线收盘跌回 SMA20（当前参考 {sma20:.2f}）下方，或严格多头排列消失。",
        f"RSI14 离开 [{config.entry_rsi_low:g}, {rsi_high:g})，或价格偏离 SMA20 超过 {max_extension:g} ATR。",
        "SPY 跌破其 SMA200、临近财报，或综合评分跌破入场门槛。",
    )
    if not blockers:
        attractiveness = "入场条件已满足"
        checked = "趋势、动量、位置、触发、" + ("盈亏比、" if config.use_reward_risk_gate else "") + "大盘和综合评分"
        entry = ActionGuidance("buy", "买入", f"已完成日线的{checked}均满足规则，可在当前或下一交易时段考虑入场。",
                               TIMEFRAME, "已触发", tuple(conditions), entry_invalidation, tuple(conflicts))
    else:
        failed = {gate.name for gate in decision.gates if gate.status == "fail"}
        if rsi >= rsi_high:
            attractiveness = "超买，等待入场条件"
        elif "extension" in failed:
            attractiveness = "偏离 SMA20 过远，等待回调"
        elif "reward_risk" in failed:
            attractiveness = "上方空间不足，等待更好的价格"
        elif "market_regime" in failed:
            attractiveness = "大盘弱势，暂缓新开仓"
        elif "earnings" in failed:
            attractiveness = "临近财报，等待事件落地"
        else:
            attractiveness = "等待入场确认"
        entry = ActionGuidance("wait", "等待", blockers[0] + " 技术强弱与入场吸引力分开判断。", TIMEFRAME, "等待确认",
                               tuple(conditions), entry_invalidation, tuple(conflicts))
    stop_text = f"吊灯止损参考 {decision.chandelier:.2f}（22 日最高价 − {config.chandelier_atr:g} ATR）" if decision.chandelier else "吊灯止损不可用（缺少 ATR 或 22 日最高价）"
    position_conditions = [f"已完成日线依据 {_time(daily.as_of)}；SMA20 {sma20:.2f}，SMA50 {sma50:.2f}；{stop_text}。"]
    if decision.exit == "sell":
        position_conditions.extend(f"卖出触发已发生：{reason}。" for reason in decision.exit_reasons)
        invalidation = (f"后续已完成日线重新站上 SMA50（当前 {sma50:.2f}）与吊灯止损，且不再符合偏空规则时，重新评估。",)
        position = ActionGuidance("reduce_sell", "卖出", "已完成日线触发趋势破坏或止损规则，可考虑卖出；不依据单独的超买或 VWAP 位置退出。",
                                  TIMEFRAME, "已触发", tuple(position_conditions) + (REFERENCE_NOTE,), invalidation, tuple(conflicts))
    elif decision.exit == "reduce":
        position_conditions.extend(f"减仓触发已发生：{reason}。" for reason in decision.exit_reasons)
        invalidation = (f"收盘重新站上 SMA20（当前 {sma20:.2f}）后可恢复持有观察；若继续跌破 SMA50 或吊灯止损则转为卖出。",)
        position = ActionGuidance("reduce", "减仓", "短线动能减弱但中期趋势尚未破坏，可考虑部分减仓以锁定利润、降低风险。",
                                  TIMEFRAME, "已触发", tuple(position_conditions) + (REFERENCE_NOTE,), invalidation, tuple(conflicts))
    else:
        position_conditions.append("退出条件尚未触发：观察是否转为偏空、连续两日收于 SMA50 下方、跌破吊灯止损（卖出），或连续两日收于 SMA20 下方（减仓）。")
        invalidation = (f"连续两日收于 SMA20 下方转为减仓；偏空结构、连续两日收于 SMA50（{sma50:.2f}）下方或跌破吊灯止损转为卖出。",)
        position = ActionGuidance("hold", "持有观察", "已完成日线尚未触发退出规则；盘中偏空或 RSI 超买本身不触发卖出。", TIMEFRAME,
                                  "持有参考成立；退出未触发", tuple(position_conditions) + (REFERENCE_NOTE,), invalidation, tuple(conflicts))
    return attractiveness, entry, position


def evaluate_signals(
    daily: pd.DataFrame,
    intraday: pd.DataFrame,
    *,
    now: datetime,
    daily_retrieved_at: datetime,
    intraday_retrieved_at: datetime | None = None,
    context: MarketContext | None = None,
    config: StrategyConfig | None = None,
) -> TechnicalAssessment:
    """Return both position-independent market bias and two action contexts.

    Entry and exit decisions come from `analysis.strategy.evaluate_bar` on the
    two latest completed daily bars. Intraday bars no longer confirm an entry;
    during an active session they only block chasing a gap beyond the
    extension limit.
    """
    config = config or StrategyConfig()
    context = context or MarketContext()
    local_now = _local(now)
    daily_cutoff = min(local_now, _local(daily_retrieved_at))
    schedule = session_schedule(local_now.date() - timedelta(days=20), local_now.date())
    ended = schedule.loc[schedule["market_close"] <= local_now]
    expected = None if ended.empty else ended.index[-1]
    today = local_now.date()
    market_open = today in schedule.index and schedule.loc[today, "market_open"] <= local_now < schedule.loc[today, "market_close"]
    daily_bias, facts = _daily_bias(_completed(daily, daily_cutoff), expected, config)
    if intraday_retrieved_at is None:
        intraday_bias, intra_facts = _intraday_bias(intraday.iloc[0:0], local_now, local_now, bool(market_open))
    else:
        intra_cutoff = min(local_now, _local(intraday_retrieved_at))
        intraday_bias, intra_facts = _intraday_bias(intraday, intra_cutoff, local_now, bool(market_open))
    decision = None if facts is None else evaluate_bar(
        facts["prior"], facts["latest"], config,
        next_earnings=context.next_earnings, earnings_checked=context.earnings_checked)
    attractiveness, entry, position = _actions(daily_bias, intraday_bias, facts, intra_facts, bool(market_open),
                                               expected, decision, config)
    notes = []
    for frame, name, cutoff in ((daily, "日线", daily_cutoff), (intraday, "5 分钟K线", min(local_now, _local(intraday_retrieved_at)) if intraday_retrieved_at is not None else local_now)):
        if not frame.empty and "complete" in frame and frame["complete"].eq(False).any():
            notes.append(f"存在未完成{name}，仅作临时观察，未用于已确认的技术判断或触发条件；数据有效时刻 {_time(cutoff)}。")
    return TechnicalAssessment(daily_bias, intraday_bias, "不可用" if facts is None else _momentum(facts["rsi"]),
                               attractiveness, entry, position, local_now.to_pydatetime(), tuple(notes),
                               None if decision is None else decision.score, decision, context.notes)


__all__ = ["BiasAssessment", "ActionGuidance", "MarketContext", "TechnicalAssessment", "evaluate_signals"]
