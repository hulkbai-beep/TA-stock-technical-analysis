"""Backtest the dashboard's entry/exit rules on real FMP daily history.

Usage (from the technical_analysis directory):

    python scripts/backtest.py                       # default symbol list, 6 years
    python scripts/backtest.py NVDA AAPL MSFT --years 8
    python scripts/backtest.py --ablation            # also rerun with each gate removed
    python scripts/backtest.py --with-sr NVDA        # include support/resistance (slow)

Results print to the console and are saved as CSV files under
`artifacts/backtest/`. No API key, URL or provider body is printed.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd  # noqa: E402

from analysis.backtest import run_backtest, summarize  # noqa: E402
from analysis.strategy import SECTOR_ETFS, StrategyConfig, attach_benchmarks  # noqa: E402
from analysis.support_resistance.engine import analyze_support_resistance  # noqa: E402
from config import load_settings  # noqa: E402
from data.fmp_client import MARKET_BENCHMARK, FMPClient  # noqa: E402
from data.normalize import normalize_daily  # noqa: E402
from models import DataError  # noqa: E402
from services.analysis_service import calendar_indicators  # noqa: E402

DEFAULT_SYMBOLS = ("AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM", "UNH", "HD",
                   "COST", "KO", "CAT", "AMD", "NFLX")
INDICATOR_WARMUP_DAYS = 330  # covers SMA200 and the 126-session Fibonacci leg

ABLATIONS = {
    "full rules": {},
    "no score gate": {"buy_min_score": 0.0},
    "no SPY gate": {"use_market_gate": False},
    "no SMA200 gate": {"use_long_trend_gate": False},
    "no earnings gate": {"use_earnings_gate": False},
    "no chandelier stop": {"chandelier_atr": 1e9},
}


def _bars(client: FMPClient, symbol: str, now: datetime) -> pd.DataFrame:
    raw = client.fetch_daily(symbol, now=now)
    return normalize_daily(raw.records, now).frame


def _earnings(client: FMPClient, symbol: str):
    try:
        rows = client._reference_rows("earnings", symbol)
    except DataError:
        return None
    dates = set()
    for row in rows:
        try:
            dates.add(datetime.fromisoformat(str(row.get("date"))[:10]).date())
        except ValueError:
            continue
    return tuple(sorted(dates)) or None


def _sector_etf(client: FMPClient, symbol: str) -> str | None:
    try:
        profile = client._reference_rows("profile", symbol)
    except DataError:
        return None
    return SECTOR_ETFS.get(next((row.get("sector") for row in profile if row.get("sector")), None))


_RATES = {"win_rate", "hit_rate", "exposure"}


def _percent(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    shown = frame.copy()
    for column in columns:
        if column in shown:
            pattern = "{:.1%}" if column in _RATES else "{:+.1%}"
            shown[column] = shown[column].map(lambda value: "" if pd.isna(value) else pattern.format(value))
    return shown


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest the technical-assessment rules.")
    parser.add_argument("symbols", nargs="*", help="US tickers (default: a 15-stock large-cap list)")
    parser.add_argument("--years", type=float, default=6.0, help="years of tested history (default 6)")
    parser.add_argument("--cost-bps", type=float, default=5.0, help="cost per side in basis points (default 5)")
    parser.add_argument("--with-sr", action="store_true", help="recompute support/resistance each day (slow)")
    parser.add_argument("--ablation", action="store_true", help="rerun with each gate removed")
    arguments = parser.parse_args(argv)
    symbols = tuple(symbol.upper() for symbol in arguments.symbols) or DEFAULT_SYMBOLS
    settings = load_settings()
    if not settings.api_key:
        print("FMP_API_KEY is not configured.")
        return 2
    days = int(arguments.years * 365) + INDICATOR_WARMUP_DAYS
    client = FMPClient(replace(settings, daily_history_days=days))
    now = datetime.now(ZoneInfo("America/New_York"))
    start = (pd.Timestamp(now.date()) - pd.Timedelta(days=int(arguments.years * 365))).date()
    try:
        market = _bars(client, MARKET_BENCHMARK, now)
    except DataError as error:
        print(f"SPY history unavailable ({error.code}); the market gate will be unknown.")
        market = None
    prepared, sector_cache = {}, {}
    for symbol in symbols:
        try:
            frame, _ = calendar_indicators(_bars(client, symbol, now))
        except DataError as error:
            print(f"{symbol}: skipped ({error.code})")
            continue
        etf = _sector_etf(client, symbol)
        if etf and etf not in sector_cache:
            try:
                sector_cache[etf] = _bars(client, etf, now)
            except DataError:
                sector_cache[etf] = None
        prepared[symbol] = (attach_benchmarks(frame, market, sector_cache.get(etf)), _earnings(client, symbol), etf)
        print(f"{symbol}: {len(frame)} sessions, sector ETF {etf or 'unknown'}, "
              f"earnings dates {'found' if prepared[symbol][1] else 'unavailable'}")
    if not prepared:
        print("No symbols could be loaded.")
        return 1
    sr = (lambda history: analyze_support_resistance(history)) if arguments.with_sr else None
    output = PROJECT_DIR / "artifacts" / "backtest"
    output.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 160)
    variants = ABLATIONS if arguments.ablation else {"full rules": {}}
    comparison = []
    for name, overrides in variants.items():
        config = replace(StrategyConfig(), **overrides)
        results = [run_backtest(symbol, frame, config, earnings_dates=earnings, cost_bps=arguments.cost_bps,
                                support_resistance_for=sr, start=start)
                   for symbol, (frame, earnings, _) in prepared.items()]
        summary = summarize(results)
        symbols_table, trades = summary["symbols"], summary["trades"]
        all_returns = trades["weighted_return"] if not trades.empty else pd.Series(dtype=float)
        comparison.append({
            "variant": name, "trades": len(all_returns),
            "win_rate": float((all_returns > 0).mean()) if len(all_returns) else float("nan"),
            "avg_trade": float(all_returns.mean()) if len(all_returns) else float("nan"),
            "avg_strategy_return": float(symbols_table["strategy_return"].mean()),
            "avg_buy_and_hold": float(symbols_table["buy_and_hold"].mean()),
            "avg_max_drawdown": float(symbols_table["max_drawdown"].mean()),
            "avg_exposure": float(symbols_table["exposure"].mean()),
        })
        if name == "full rules":
            print(f"\n=== Per-symbol results, {arguments.years:g} years, {arguments.cost_bps:g} bps per side ===")
            print(_percent(symbols_table, ("win_rate", "avg_return", "median_return", "avg_win", "avg_loss",
                                           "strategy_return", "max_drawdown", "exposure", "buy_and_hold"))
                  .to_string(index=False, float_format=lambda value: f"{value:.2f}"))
            print("\n=== Does a higher score predict better 10-session returns? ===")
            print(_percent(summary["score_buckets"], ("avg_10d_return", "hit_rate", "vs_all_bars"))
                  .to_string(index=False))
            symbols_table.to_csv(output / "symbols.csv", index=False)
            summary["score_buckets"].to_csv(output / "score_buckets.csv", index=False)
            trades.to_csv(output / "trades.csv", index=False)
    table = pd.DataFrame(comparison)
    if arguments.ablation:
        print("\n=== Ablation: what each gate contributes ===")
        print(_percent(table, ("win_rate", "avg_trade", "avg_strategy_return", "avg_buy_and_hold",
                               "avg_max_drawdown", "avg_exposure")).to_string(index=False))
        table.to_csv(output / "ablation.csv", index=False)
    print(f"\nCSV files saved in {output}")
    print("Past results on a hand-picked list of today's large caps carry survivorship bias; "
          "treat them as a sanity check, not a forecast.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
