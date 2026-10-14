"""Walk-forward historical test of the shared `evaluate_bar` rules.

Each decision uses only completed bars up to that session. Orders fill at
the next session's open, with a per-side cost. One position per symbol;
Reduce sells half of what is held, Sell closes the rest. This measures how
the rules would have behaved on past data. It does not model slippage beyond
the cost, gaps through stops, taxes, or survivorship bias in the symbol list.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from bisect import bisect_left
import math
from typing import Callable

import numpy as np
import pandas as pd

from analysis.strategy import StrategyConfig, evaluate_bar
from data.normalize import session_schedule

FORWARD_SESSIONS = 10
SCORE_BUCKETS = ((0, 40, "<40"), (40, 60, "40–60"), (60, 75, "60–75"), (75, 101, "≥75"))


@dataclass(frozen=True)
class Trade:
    symbol: str
    entry_session: date
    entry_price: float
    exit_session: date | None
    exit_price: float | None
    weighted_return: float
    sessions_held: int
    entry_score: float | None
    exit_reason: str


@dataclass
class BacktestResult:
    symbol: str
    trades: list[Trade] = field(default_factory=list)
    equity: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    buy_and_hold: float | None = None
    exposure: float = 0.0
    observations: pd.DataFrame = field(default_factory=pd.DataFrame)


def _valid(row) -> bool:
    try:
        values = [float(row[name]) for name in ("open", "close", "sma20", "sma50", "rsi14")]
    except (KeyError, TypeError, ValueError):
        return False
    return all(math.isfinite(value) and value > 0 for value in values[:4]) and 0 <= values[4] <= 100


def run_backtest(symbol: str, daily: pd.DataFrame, config: StrategyConfig | None = None, *,
                 earnings_dates: tuple[date, ...] | None = None, cost_bps: float = 5.0,
                 support_resistance_for: Callable[[pd.DataFrame], object] | None = None,
                 start: date | None = None) -> BacktestResult:
    """Simulate the entry/exit rules over a featured daily frame.

    `daily` must already carry `calendar_indicators` features and, when
    available, `attach_benchmarks` columns. `earnings_dates` is the sorted
    set of known report dates; None means the earnings gate is unknown.
    """
    config = config or StrategyConfig()
    frame = daily.loc[daily["complete"].astype(bool)].reset_index(drop=True)
    result = BacktestResult(symbol)
    if len(frame) < 3:
        return result
    schedule = session_schedule(frame["session"].min(), frame["session"].max())
    position_of = {session: index for index, session in enumerate(schedule.index)}
    cost = cost_bps / 10_000
    earnings = sorted(earnings_dates) if earnings_dates is not None else None
    held_fraction = 0.0
    entry_index = entry_price = entry_score = None
    realized = 0.0
    pending: tuple[str, str] | None = None
    base = 1.0
    equity = []
    observations = []
    in_market = 0
    first = 1
    if start is not None:
        first = max(1, int(np.searchsorted(frame["session"].to_numpy(), start)))
    closes = frame["close"].astype(float).to_numpy()
    for index in range(first, len(frame)):
        row = frame.iloc[index]
        price_open = float(row["open"])
        # Execute yesterday's decision at today's open.
        if pending is not None and held_fraction > 0 and pending[0] in ("sell", "reduce"):
            sold = held_fraction if pending[0] == "sell" else held_fraction / 2
            realized += sold * (price_open * (1 - cost) / entry_price - 1)
            held_fraction -= sold
            if held_fraction <= 1e-9:
                result.trades.append(Trade(symbol, frame.iloc[entry_index]["session"], entry_price, row["session"],
                                           price_open, realized, index - entry_index, entry_score, pending[1]))
                base *= 1 + realized
                held_fraction, realized = 0.0, 0.0
        elif pending is not None and pending[0] == "buy" and held_fraction == 0:
            entry_index, entry_price, held_fraction, realized = index, price_open * (1 + cost), 1.0, 0.0
        pending = None
        if held_fraction > 0:
            in_market += 1
        mark = (held_fraction * (float(row["close"]) / entry_price - 1) + realized) if held_fraction > 0 else 0.0
        equity.append((row["session"], base * (1 + mark)))
        prior = frame.iloc[index - 1]
        consecutive = position_of.get(row["session"], -2) - position_of.get(prior["session"], -9) == 1
        if not (consecutive and _valid(prior) and _valid(row)):
            continue
        next_earnings = None
        if earnings is not None:
            position = bisect_left(earnings, row["session"])
            next_earnings = earnings[position] if position < len(earnings) else None
        sr = support_resistance_for(frame.iloc[:index + 1]) if support_resistance_for else None
        decision = evaluate_bar(prior, row, config, support_resistance=sr, next_earnings=next_earnings,
                                earnings_checked=earnings is not None)
        forward = (closes[index + FORWARD_SESSIONS] / closes[index] - 1
                   if index + FORWARD_SESSIONS < len(frame) else np.nan)
        observations.append((row["session"], decision.score.score, decision.buy, forward))
        if held_fraction > 0 and decision.exit:
            pending = (decision.exit, "；".join(decision.exit_reasons))
        elif held_fraction == 0 and decision.buy and index + 1 < len(frame):
            pending = ("buy", "")
            entry_score = decision.score.score
    if held_fraction > 0:
        last = frame.iloc[-1]
        result.trades.append(Trade(symbol, frame.iloc[entry_index]["session"], entry_price, None, float(last["close"]),
                                   realized + held_fraction * (float(last["close"]) / entry_price - 1),
                                   len(frame) - 1 - entry_index, entry_score, "仍持有（按最后收盘估值）"))
    result.observations = pd.DataFrame(observations, columns=["session", "score", "buy", "forward_return"])
    result.equity = pd.Series([value for _, value in equity], index=[session for session, _ in equity], dtype=float)
    traded = frame.iloc[first:]
    if len(traded) > 1:
        result.buy_and_hold = float(traded.iloc[-1]["close"]) / float(traded.iloc[0]["open"]) - 1
        result.exposure = in_market / len(traded)
    return result


def summarize(results: list[BacktestResult]) -> dict[str, pd.DataFrame]:
    """Per-symbol trade statistics and score-bucket forward returns."""
    rows = []
    for item in results:
        returns = np.array([trade.weighted_return for trade in item.trades], dtype=float)
        wins, losses = returns[returns > 0], returns[returns <= 0]
        curve = item.equity
        drawdown = float((curve / curve.cummax() - 1).min()) if len(curve) else np.nan
        rows.append({
            "symbol": item.symbol, "trades": len(returns),
            "win_rate": float((returns > 0).mean()) if len(returns) else np.nan,
            "avg_return": float(returns.mean()) if len(returns) else np.nan,
            "median_return": float(np.median(returns)) if len(returns) else np.nan,
            "avg_win": float(wins.mean()) if len(wins) else np.nan,
            "avg_loss": float(losses.mean()) if len(losses) else np.nan,
            "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() < 0 else np.nan,
            "avg_sessions": float(np.mean([trade.sessions_held for trade in item.trades])) if item.trades else np.nan,
            "strategy_return": float(curve.iloc[-1] - 1) if len(curve) else np.nan,
            "max_drawdown": drawdown, "exposure": item.exposure, "buy_and_hold": item.buy_and_hold,
        })
    observations = pd.concat([item.observations for item in results if not item.observations.empty],
                             ignore_index=True) if any(not item.observations.empty for item in results) else pd.DataFrame(
                                 columns=["session", "score", "buy", "forward_return"])
    buckets = []
    scored = observations.dropna(subset=["score", "forward_return"])
    baseline = float(scored["forward_return"].mean()) if len(scored) else np.nan
    for low, high, label in SCORE_BUCKETS:
        chunk = scored.loc[(scored["score"] >= low) & (scored["score"] < high), "forward_return"]
        buckets.append({"score": label, "observations": len(chunk),
                        f"avg_{FORWARD_SESSIONS}d_return": float(chunk.mean()) if len(chunk) else np.nan,
                        "hit_rate": float((chunk > 0).mean()) if len(chunk) else np.nan,
                        "vs_all_bars": float(chunk.mean() - baseline) if len(chunk) else np.nan})
    trades = pd.DataFrame([vars(trade) for item in results for trade in item.trades])
    return {"symbols": pd.DataFrame(rows), "score_buckets": pd.DataFrame(buckets), "trades": trades}


__all__ = ["BacktestResult", "Trade", "run_backtest", "summarize"]
