"""Deterministic Chinese observations with explicit numeric evidence."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import math

import pandas as pd


@dataclass(frozen=True)
class SummaryItem:
    title: str
    text: str
    evidence: str
    rule_id: str
    tone: str = "neutral"


def _number(value: object) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def _relation(left: float, right: float) -> str:
    return "高于" if left > right else "低于" if left < right else "等于"


def _sign(value: float) -> str:
    return "上升" if value > 0 else "下降" if value < 0 else "持平"


def generate_summaries(
    daily: pd.DataFrame,
    intraday: pd.DataFrame,
    *,
    previous_close: float | None = None,
    previous_close_session: date | None = None,
    comparable_sessions: bool = True,
) -> list[SummaryItem]:
    """Only completed bars establish directional rules.

    Daily direction requires price/MA ordering AND both MA slopes to agree.
    Intraday direction is the last completed close versus estimated VWAP.
    These classifications describe observations; they are not forecasts.
    """
    items: list[SummaryItem] = []
    completed = daily.loc[daily["complete"].astype(bool)]
    daily_direction: str | None = None
    daily_session = None
    if completed.empty:
        items.append(SummaryItem("日线趋势", "尚无已完成日线，无法确认日线趋势。", "规则：仅使用 complete=True 的日线。", "daily.no_completed", "warning"))
    else:
        latest = completed.iloc[-1]
        daily_session = latest["session"]
        close = float(latest["close"])
        sma20, sma50 = _number(latest.get("sma20")), _number(latest.get("sma50"))
        previous = completed.iloc[-2] if len(completed) > 1 else None
        slope20 = slope50 = None
        if previous is not None:
            previous20, previous50 = _number(previous.get("sma20")), _number(previous.get("sma50"))
            if sma20 is not None and previous20 is not None:
                slope20 = sma20 - previous20
            if sma50 is not None and previous50 is not None:
                slope50 = sma50 - previous50
        if sma20 is None or sma50 is None:
            available = ", ".join(f"SMA{period}={float(latest[f'sma{period}']):.2f}" if _number(latest.get(f"sma{period}")) is not None else f"SMA{period}=不可用" for period in (20, 50))
            items.append(SummaryItem("日线趋势", f"{daily_session} 已完成收盘 {close:.2f}；均线历史不足，趋势规则暂不可用。", f"收盘={close:.2f}；{available}。规则：SMA20 / SMA50 分别需要 20 / 50 根有效日线。", "daily.warmup", "warning"))
        else:
            text = f"{daily_session} 已完成收盘 {close:.2f}，{_relation(close, sma20)} SMA20（{sma20:.2f}），{_relation(close, sma50)} SMA50（{sma50:.2f}）；SMA20 {_relation(sma20, sma50)} SMA50。"
            evidence = f"收盘={close:.4f}；SMA20={sma20:.4f}；SMA50={sma50:.4f}。"
            if slope20 is not None and slope50 is not None:
                text += f" 较上一已完成数据日 {previous['session']}，SMA20 {_sign(slope20)}、SMA50 {_sign(slope50)}。"
                evidence += f" ΔSMA20={slope20:+.4f}；ΔSMA50={slope50:+.4f}。"
                if close > sma20 > sma50 and slope20 > 0 and slope50 > 0:
                    daily_direction = "positive"
                    text += " 价格、均线排列及均线方向同时偏强。"
                elif close < sma20 < sma50 and slope20 < 0 and slope50 < 0:
                    daily_direction = "negative"
                    text += " 价格、均线排列及均线方向同时偏弱。"
                else:
                    daily_direction = "mixed"
                    text += " 价格、均线排列与均线方向未同时满足单一方向规则。"
            else:
                text += " 前一数据日的均线不足，均线方向暂不可用。"
            evidence += " 方向规则：收盘>SMA20>SMA50 且两均线斜率>0 为偏强；全部反向为偏弱；其余为混合。"
            items.append(SummaryItem("日线趋势", text, evidence, "daily.order_and_slopes", daily_direction or "neutral"))
        rsi = _number(latest.get("rsi14"))
        if rsi is None:
            items.append(SummaryItem("RSI14 动量", f"{daily_session} 的 RSI14 不可用，连续有效历史不足。", "规则：14 个价格变动的算术平均初始化，随后采用 Wilder 递推；缺失价格重新预热。", "rsi.warmup", "warning"))
        else:
            band = "超买区间" if rsi >= 70 else "超卖区间" if rsi <= 30 else "中间区间"
            text = f"{daily_session} 已完成日线 RSI14={rsi:.2f}，处于{band}。"
            evidence = f"RSI14={rsi:.4f}。规则：≥70 为超买，≤30 为超卖，其余为中间区间。"
            previous_rsi = _number(previous.get("rsi14")) if previous is not None else None
            if previous_rsi is not None:
                delta = rsi - previous_rsi
                text += f" 较 {previous['session']} {_sign(delta)} {abs(delta):.2f} 点。"
                evidence += f" 前值={previous_rsi:.4f}；变化={delta:+.4f}。"
            text += " 超买或超卖不代表必然反转。"
            items.append(SummaryItem("RSI14 动量", text, evidence, "rsi.bands_and_change"))
    provisional = daily.loc[~daily["complete"].astype(bool)]
    if not provisional.empty:
        row = provisional.iloc[-1]
        items.append(SummaryItem("未完成日线", f"{row['session']} 未完成日线的最新可用价格为 {float(row['close']):.2f}；仅作临时观察，不参与已确认日线趋势。", "规则：complete=False 的日线不用于确认趋势或 RSI 动量结论。", "daily.provisional", "warning"))
    if intraday.empty:
        items.append(SummaryItem("盘中观察", "5 分钟数据不可用，无法生成盘中观察。", "规则：盘中结论必须来自真实的 RTH 5 分钟K线。", "intraday.unavailable", "warning"))
        return items
    actual_session = intraday.iloc[-1]["session"]
    complete_intraday = intraday.loc[intraday["complete"].astype(bool)]
    if complete_intraday.empty:
        items.append(SummaryItem("盘中观察", f"{actual_session} 暂无已完成的 5 分钟K线；未完成K线仅展示为临时数据。", "规则：盘中结论仅使用 complete=True 的K线。", "intraday.no_completed", "warning"))
        return items
    row = complete_intraday.iloc[-1]
    close = float(row["close"])
    ending = pd.Timestamp(row["bar_end"]).strftime("%H:%M")
    vwap, opening, volume = (_number(row.get(name)) for name in ("vwap", "session_open", "session_volume"))
    coverage = bool(row.get("coverage_complete", False))
    intraday_direction = None
    text = f"{actual_session} 截至 {ending}（纽约时间）的已完成 5 分钟K线收盘为 {close:.2f}。"
    evidence = f"收盘={close:.4f}；K线结束={row['bar_end']}。"
    if coverage and vwap is not None:
        text += f" 价格{_relation(close, vwap)}当日估算 VWAP（{vwap:.2f}）。"
        intraday_direction = "positive" if close > vwap else "negative" if close < vwap else "neutral"
        evidence += f" VWAP={vwap:.4f}；规则：Σ[HLC3×成交量]/Σ成交量，自 09:30 重置；价格>VWAP 为偏强，<VWAP 为偏弱。"
    elif not coverage:
        text += " 缺少 09:30 开盘K线或期间有缺口，完整交易时段的 VWAP、开盘对比及累计成交量不可用。"
        evidence += " 覆盖规则：必须含 09:30 且每个 5 分钟时点连续。"
    else:
        text += " 累计成交量为零，估算 VWAP 不可用。"
        evidence += " 规则：累计成交量必须大于零才计算 VWAP。"
    if coverage and opening is not None:
        text += f" 价格{_relation(close, opening)}当日开盘（{opening:.2f}）。"
        evidence += f" 09:30 开盘={opening:.4f}。"
    if coverage and volume is not None:
        text += f" 截至该K线累计成交量 {volume:,.0f}；未作高低量判断。"
        evidence += f" 累计成交量={volume:.0f}。"
    if previous_close is not None and previous_close_session is not None:
        change = (close / previous_close - 1.0) * 100.0
        text += f" 较上一交易日 {previous_close_session} 收盘（{previous_close:.2f}）变化 {change:+.2f}%。"
        evidence += f" 前收={previous_close:.4f}；变化=({close:.4f}/{previous_close:.4f}-1)×100={change:+.4f}%。"
    else:
        text += " 缺少相同拆股复权口径的确切上一交易日收盘，前收对比不可用。"
        evidence += " 前收规则：只接受 NYSE 日历中确切上一交易日的已完成日线。"
    items.append(SummaryItem("盘中观察", text, evidence, "intraday.completed_vs_vwap", intraday_direction or "warning"))
    if not comparable_sessions:
        items.append(SummaryItem("跨周期信号", f"日线确认日为 {daily_session}，盘中数据日为 {actual_session}；日期衔接不完整，暂不判断两者是否一致。", "规则：盘中日期必须等于日线确认日或其确切下一交易日。", "agreement.date_mismatch", "warning"))
    elif daily_direction in ("positive", "negative") and intraday_direction in ("positive", "negative"):
        aligned = daily_direction == intraday_direction
        items.append(SummaryItem("跨周期信号", f"{daily_session} 的已确认日线规则与 {actual_session} 的盘中价格/VWAP 规则{'方向一致' if aligned else '方向冲突'}。", "规则：日线使用价格/均线排列和两条均线斜率；盘中使用最后已完成K线价格相对估算 VWAP 的方向。", "agreement.direction", "neutral" if aligned else "warning"))
    else:
        items.append(SummaryItem("跨周期信号", "至少一个周期未形成明确方向，暂不判断方向一致或冲突。", "规则：只有日线与盘中均满足明确偏强或偏弱条件时才比较方向。", "agreement.insufficient"))
    return items
