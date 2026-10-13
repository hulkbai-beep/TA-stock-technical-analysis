"""Deterministic technical heuristics over completed, dated market bars.

These initial rules are not backtested strategies. Daily trend, RSI momentum,
and current-session timing remain separate; no portfolio facts are inferred.
The caller supplies already-calculated SMA20, SMA50, RSI14, and session VWAP.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from math import isfinite

import pandas as pd

from data.normalize import session_schedule


TIMEFRAME = "日线／波段观察（约 5–20 个交易日）；5 分钟仅用于当日入场时机。"
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


def _daily_bias(bars: pd.DataFrame, expected_session: date | None) -> tuple[BiasAssessment, dict | None]:
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
    strict_up = close > sma20 > sma50 and slope20 > 0 and slope50 > 0
    strict_down = close < sma20 < sma50 and slope20 < 0 and slope50 < 0
    if strict_up and rsi >= 55:
        code, label, reason = "strong_bullish", "强烈偏多", "收盘价高于 SMA20 和 SMA50，两条均线均上升，RSI14 至少为 55。"
    elif close > sma50 and sma20 > sma50 and slope50 >= 0:
        code, label, reason = "bullish", "偏多", "收盘价与 SMA20 均高于 SMA50，且 SMA50 未下降；尚未满足全部强势偏多条件。"
    elif strict_down and rsi <= 45:
        code, label, reason = "strong_bearish", "强烈偏空", "收盘价低于 SMA20 和 SMA50，两条均线均下降，RSI14 不高于 45。"
    elif close < sma50 and sma20 < sma50 and slope50 <= 0:
        code, label, reason = "bearish", "偏空", "收盘价与 SMA20 均低于 SMA50，且 SMA50 未上升；尚未满足全部强势偏空条件。"
    else:
        code, label, reason = "neutral", "中性", "价格与均线结构未形成规则定义的明确多头或空头排列。"
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
    prior_close, prior_sma20 = float(prior["close"]), float(prior["sma20"])
    reclaim = prior_close <= prior_sma20 and close > sma20
    continuation = prior_close > prior_sma20 and close > sma20 and close > prior_close
    facts = {
        "prior": prior, "latest": latest, "close": close, "sma20": sma20, "sma50": sma50, "rsi": rsi,
        "strict_up": strict_up, "strict_down": strict_down, "gap": close / sma20 - 1,
        "reclaim": reclaim, "continuation": continuation, "fresh": session == expected_session,
        "two_below50": prior_close < float(prior["sma50"]) and close < sma50,
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


def _actions(daily: BiasAssessment, intraday: BiasAssessment, facts: dict | None, intra_facts: dict | None,
             market_open: bool, expected_session: date | None) -> tuple[str, ActionGuidance, ActionGuidance]:
    if facts is None:
        entry, position = _unavailable_actions("已完成日线历史或指标不足，无法生成操作参考。", daily, intraday)
        return "数据不足", entry, position
    if not facts["fresh"]:
        entry, position = _unavailable_actions(f"最新日线为 {daily.session}；应完成交易日为 {expected_session}。请刷新或补足数据后再判断。", daily, intraday)
        return "数据不足", entry, position
    close, sma20, sma50, rsi = (facts[name] for name in ("close", "sma20", "sma50", "rsi"))
    daily_trigger = facts["reclaim"] or facts["continuation"]
    conditions = [f"日线参考 {_time(daily.as_of)}：收盘 {close:.2f}，SMA20 {sma20:.2f}，SMA50 {sma50:.2f}，RSI14 {rsi:.2f}，距 SMA20 {facts['gap'] * 100:+.2f}%。"]
    if facts["reclaim"]:
        conditions.append("日线触发已发生：相邻两根已完成日线由前收盘不高于前 SMA20，转为最新收盘高于最新 SMA20（重新站上）。")
    elif facts["continuation"]:
        conditions.append("日线触发已发生：相邻两根已完成日线均收于各自 SMA20 上方，且最新收盘高于前收盘（延续确认）。")
    else:
        conditions.append("日线触发待确认：相邻两根已完成日线须确认重新站上 SMA20，或均在各自 SMA20 上方且最新收盘高于前收盘。")
    blockers = []
    if not facts["strict_up"]:
        blockers.append("等待日线收盘 > SMA20 > SMA50，且两条均线均较前一交易日上升。")
    if not 50 <= rsi < 70:
        blockers.append(f"等待已完成日线 RSI14 回到 50（含）至 70（不含）；当前 {rsi:.2f}。")
    if not sma20 <= close <= sma20 * 1.03:
        blockers.append(f"等待日线收盘处于其 SMA20 {sma20:.2f} 上方 0%–3% 范围；当前距离 {facts['gap'] * 100:+.2f}%。")
    if not daily_trigger:
        blockers.append("等待上述由连续已完成日线验证的价格触发。")
    if not market_open:
        blockers.append("等待 NYSE 正常交易时段内的当日确认。")
    if intra_facts is None:
        blockers.append("盘中确认不可用：等待完整覆盖、VWAP 有效且连续的两根已完成 5 分钟K线。")
    else:
        if not intra_facts["current"]:
            blockers.append("等待今日最新已完成 5 分钟K线，且结束时刻距评估不超过 10 分钟。")
        if intraday.code != "bullish":
            blockers.append("等待连续两根已完成 5 分钟收盘高于各自 VWAP，且最新收盘高于当日开盘。")
        intra_gap = intra_facts["close"] / sma20 - 1
        conditions.append(f"盘中参考 {_time(intraday.as_of)}：已完成收盘 {intra_facts['close']:.2f}，距日线 SMA20 {sma20:.2f} 为 {intra_gap * 100:+.2f}%。")
        intra_confirmed = intra_facts["current"] and intraday.code == "bullish" and sma20 < intra_facts["close"] <= sma20 * 1.03
        conditions.append("盘中入场触发已发生：今日新鲜的连续两根已完成 5 分钟K线确认偏多，且最新收盘高于日线 SMA20、偏离不超过 3%。" if intra_confirmed else "盘中入场触发待确认：需在当日正常交易时段内同时满足数据新鲜、连续两根K线偏多及 SMA20 价格距离条件。")
        if not sma20 < intra_facts["close"] <= sma20 * 1.03:
            blockers.append(f"等待已完成 5 分钟收盘高于日线 SMA20 {sma20:.2f} 且偏离不超过 3%，避免追逐跳空。")
    conditions.extend(blockers)
    conditions.append(REFERENCE_NOTE)
    conflicts = list(daily.conflicts + intraday.conflicts)
    if intraday.code == "bearish" and daily.code in ("bullish", "strong_bullish"):
        conflicts.append("日线偏多但盘中偏空：新入场需等待盘中确认，盘中信号不改变日线判断。")
    if intraday.code == "bullish" and daily.code in ("bearish", "strong_bearish"):
        conflicts.append("盘中偏多与日线偏空相冲突；盘中反弹不覆盖日线退出条件。")
    entry_invalidation = (
        f"后续已完成日线收盘不再高于其 SMA20（当前参考 {sma20:.2f}），或严格多头排列／两条均线同时上升条件消失。",
        "已完成日线 RSI14 离开 50（含）至 70（不含）范围，或日线／盘中价格偏离日线 SMA20 超过 3%。",
        f"最新已完成 5 分钟收盘回到日线 SMA20（当前参考 {sma20:.2f}）或下方，盘中入场价格条件失效。",
        "当日连续两根已完成 5 分钟K线不再同时位于各自 VWAP 上方，最新收盘不再高于当日开盘，或盘中数据过期／不可用。",
    )
    if not blockers:
        attractiveness = "入场条件已满足"
        entry = ActionGuidance("buy", "买入", "日线趋势、动量、价格距离与当日盘中确认共同满足初始规则，可考虑技术性入场。", TIMEFRAME, "已触发", tuple(conditions), entry_invalidation, tuple(conflicts))
    else:
        if rsi >= 70:
            attractiveness = "超买，等待入场条件"
        elif close > sma20 * 1.03:
            attractiveness = "日线偏离 SMA20 超过 3%，等待确认"
        elif intra_facts is not None and intra_facts["close"] > sma20 * 1.03:
            attractiveness = "盘中偏离 SMA20 超过 3%，等待确认"
        else:
            attractiveness = "等待入场确认"
        entry = ActionGuidance("wait", "等待", blockers[0] + " 技术强弱与入场吸引力分开判断。", TIMEFRAME, "等待确认", tuple(conditions), entry_invalidation, tuple(conflicts))
    exit_triggered = daily.code in ("bearish", "strong_bearish") or facts["two_below50"]
    position_conditions = [f"已完成日线依据 {_time(daily.as_of)}；当前 SMA50 参考 {sma50:.2f}。"]
    if exit_triggered:
        if daily.code in ("bearish", "strong_bearish"):
            position_conditions.append("退出触发已发生：已完成日线符合偏空或强势偏空规则。")
        if facts["two_below50"]:
            position_conditions.append("退出触发已发生：相邻两根已完成日线收盘均低于各自 SMA50。")
        invalidation = (f"后续已完成日线收盘回到其 SMA50（当前 {sma50:.2f}）或上方，且不再符合偏空规则，也不再连续两日收于各自 SMA50 下方时，重新评估减仓／卖出依据。",)
        position = ActionGuidance("reduce_sell", "减仓或卖出", "已完成日线已触发趋势转弱退出规则，可考虑减仓或卖出；不依据单独的超买或 VWAP 位置退出。", TIMEFRAME, "已触发", tuple(position_conditions) + (REFERENCE_NOTE,), invalidation, tuple(conflicts))
    else:
        position_conditions.append("退出条件尚未触发：观察后续已完成日线是否变为偏空，或相邻两日收盘均低于各自 SMA50。")
        invalidation = (f"后续已完成日线出现偏空／强势偏空结构，或相邻两根已完成日线收盘均低于各自 SMA50（当前 {sma50:.2f}）时，持有观察依据失效，转为减仓／卖出参考。",)
        position = ActionGuidance("hold", "持有观察", "已完成日线尚未触发退出规则；盘中偏空或 RSI 超买本身不触发卖出。", TIMEFRAME, "持有参考成立；退出未触发", tuple(position_conditions) + (REFERENCE_NOTE,), invalidation, tuple(conflicts))
    return attractiveness, entry, position


def evaluate_signals(
    daily: pd.DataFrame,
    intraday: pd.DataFrame,
    *,
    now: datetime,
    daily_retrieved_at: datetime,
    intraday_retrieved_at: datetime | None = None,
) -> TechnicalAssessment:
    """Return both position-independent market bias and two action contexts.

    Entry requires strict daily uptrend, RSI in [50, 70), daily SMA20 distance
    in [0%, 3%], a verified daily reclaim/continuation, and fresh RTH intraday
    bullish confirmation with price in (0%, 3%] above daily SMA20. A daily
    bearish bias or two consecutive closes below SMA50 takes exit precedence.
    """
    local_now = _local(now)
    daily_cutoff = min(local_now, _local(daily_retrieved_at))
    schedule = session_schedule(local_now.date() - timedelta(days=20), local_now.date())
    ended = schedule.loc[schedule["market_close"] <= local_now]
    expected = None if ended.empty else ended.index[-1]
    today = local_now.date()
    market_open = today in schedule.index and schedule.loc[today, "market_open"] <= local_now < schedule.loc[today, "market_close"]
    daily_bias, facts = _daily_bias(_completed(daily, daily_cutoff), expected)
    if intraday_retrieved_at is None:
        intraday_bias, intra_facts = _intraday_bias(intraday.iloc[0:0], local_now, local_now, bool(market_open))
    else:
        intra_cutoff = min(local_now, _local(intraday_retrieved_at))
        intraday_bias, intra_facts = _intraday_bias(intraday, intra_cutoff, local_now, bool(market_open))
    attractiveness, entry, position = _actions(daily_bias, intraday_bias, facts, intra_facts, bool(market_open), expected)
    notes = []
    for frame, name, cutoff in ((daily, "日线", daily_cutoff), (intraday, "5 分钟K线", min(local_now, _local(intraday_retrieved_at)) if intraday_retrieved_at is not None else local_now)):
        if not frame.empty and "complete" in frame and frame["complete"].eq(False).any():
            notes.append(f"存在未完成{name}，仅作临时观察，未用于已确认的技术判断或触发条件；数据有效时刻 {_time(cutoff)}。")
    return TechnicalAssessment(daily_bias, intraday_bias, "不可用" if facts is None else _momentum(facts["rsi"]), attractiveness, entry, position, local_now.to_pydatetime(), tuple(notes))


__all__ = ["BiasAssessment", "ActionGuidance", "TechnicalAssessment", "evaluate_signals"]
