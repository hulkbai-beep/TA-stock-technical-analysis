"""Analyze every watchlist ticker and condense each result into one table row.

Each ticker runs the same `analyze` service as the detail page, with daily
bars only: every action rule uses completed daily bars, so the 5-minute
request would add cost without changing a decision. Rows keep both the
new-entry and the holding guidance, so toggling "held" needs no new request.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timezone
import logging
from typing import Callable

from analysis.strategy import SETUP_SHORT, Gate
from models import DataError
from services import analysis_service

_GATE_SHORT = {
    "strict_trend": "趋势未成多头排列", "rsi_range": "RSI 不在入场区间", "extension": "离 SMA20 过远",
    "trigger": "未出现日线触发", "reward_risk": "盈亏比不足", "long_trend": "低于 SMA200", "market_regime": "大盘弱势",
    "earnings": "临近财报", "score": "评分不足",
}
_LOG = logging.getLogger(__name__)
ACTION_ORDER = {"buy": 0, "reduce_sell": 1, "reduce": 2, "hold": 3, "wait": 4, "insufficient_data": 5, "error": 6}


@dataclass(frozen=True)
class WatchlistRow:
    symbol: str
    analyzed_at: datetime
    error: str | None = None
    price: float | None = None
    change_percent: float | None = None
    session: date | None = None
    provisional: bool = False
    confirmed_session: date | None = None
    daily_label: str | None = None
    score: float | None = None
    grade: str | None = None
    entry_code: str = "error"
    entry_label: str = "错误"
    entry_reason: str = ""
    position_code: str = "error"
    position_label: str = "错误"
    position_reason: str = ""
    chandelier: float | None = None
    stop_distance: float | None = None
    next_earnings: date | None = None
    relative_strength: float | None = None
    setup: str | None = None
    reward_risk: str | None = None

    def action(self, holding: bool) -> tuple[str, str, str]:
        if holding:
            return self.position_code, self.position_label, self.position_reason
        return self.entry_code, self.entry_label, self.entry_reason


def _entry_reason(gates: tuple[Gate, ...], code: str) -> str:
    if code == "buy":
        return "全部入场条件满足"
    blocked = [_GATE_SHORT[gate.name] for gate in gates
               if gate.status == "fail" or (gate.required and gate.status == "unknown")]
    return "、".join(blocked) or "等待确认"


def summarize_result(result, analyzed_at: datetime) -> WatchlistRow:
    assessment = result.assessment
    decision = assessment.decision
    entry, position = assessment.entry, assessment.position
    latest = result.daily.loc[result.daily["complete"].astype(bool)].iloc[-1] if result.confirmed_daily_session else None
    relative = None
    if latest is not None:
        own, market = latest.get("ret63"), latest.get("mkt_ret63")
        if own is not None and market is not None and own == own and market == market:
            relative = float(own) - float(market)
    chandelier = decision.chandelier if decision is not None else None
    setup = SETUP_SHORT.get(decision.trigger) if decision is not None else None
    rr = decision.reward_risk if decision is not None else None
    close = None if latest is None else float(latest["close"])
    if rr is None or close is None:
        reward_risk = None
    elif close <= rr.stop:
        reward_risk = "止损下方"
    elif rr.ratio is None:
        reward_risk = "上方无前高"
    else:
        reward_risk = f"{rr.ratio:.1f}"
    price = result.snapshot.price
    if decision is None:
        entry_reason = position_reason = entry.explanation
    else:
        entry_reason = _entry_reason(decision.gates, entry.code)
        position_reason = "；".join(decision.exit_reasons) or "退出条件未触发"
    if entry.code == "insufficient_data":
        entry_reason = position_reason = entry.explanation
    return WatchlistRow(
        symbol=result.symbol, analyzed_at=analyzed_at, price=price,
        change_percent=result.snapshot.change_percent, session=result.snapshot.session,
        provisional=result.snapshot.provisional, confirmed_session=result.confirmed_daily_session,
        daily_label=assessment.daily.label,
        score=None if assessment.score is None else assessment.score.score,
        grade=None if assessment.score is None else assessment.score.grade,
        entry_code=entry.code, entry_label=entry.label, entry_reason=entry_reason,
        position_code=position.code, position_label=position.label, position_reason=position_reason,
        chandelier=chandelier,
        stop_distance=None if chandelier is None or not price else (price - chandelier) / price,
        next_earnings=result.context.next_earnings, relative_strength=relative,
        setup=setup, reward_risk=reward_risk,
    )


def analyze_watchlist(symbols: list[str], client, *, now: datetime | None = None, refresh: bool = False,
                      max_workers: int = 3,
                      progress: Callable[[int, int, str], None] | None = None) -> dict[str, WatchlistRow]:
    """Analyze tickers concurrently; one failure becomes an error row, not a crash.

    `progress` is called from the calling thread only, after each ticker.
    """
    rows: dict[str, WatchlistRow] = {}
    if not symbols:
        return rows

    def one(symbol: str) -> WatchlistRow:
        stamp = datetime.now(timezone.utc)
        try:
            result = analysis_service.analyze(symbol, client, now=now, refresh=refresh, include_intraday=False)
            return summarize_result(result, stamp)
        except DataError as error:
            return WatchlistRow(symbol, stamp, error=str(error))
        except Exception as error:
            # Unexpected failures are bugs, not data gaps: keep the traceback.
            _LOG.exception("Watchlist analysis failed for %s", symbol)
            return WatchlistRow(symbol, stamp,
                                error=f"分析失败（{type(error).__name__}），详细错误见运行 Streamlit 的终端窗口。")

    with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(symbols))),
                            thread_name_prefix="watchlist") as executor:
        futures = {executor.submit(one, symbol): symbol for symbol in symbols}
        for done, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows[row.symbol] = row
            if progress is not None:
                progress(done, len(symbols), row.symbol)
    return {symbol: rows[symbol] for symbol in symbols if symbol in rows}


__all__ = ["ACTION_ORDER", "WatchlistRow", "analyze_watchlist", "summarize_result"]
