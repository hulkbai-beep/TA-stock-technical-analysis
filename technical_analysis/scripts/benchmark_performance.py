"""Measure analysis stages and cached switching without logging credentials.

Default fixtures are isolated synthetic benchmark inputs, never dashboard data.
Only --live loads local settings and allows genuine FMP network requests.
Server UI generation is measured separately; browser paint is not instrumented.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys
from threading import Lock
import time
from zoneinfo import ZoneInfo

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

SAFE_ERROR_CODES = {
    "missing_key", "unauthorized", "permission_denied", "rate_limited",
    "network_error", "tls_failed", "invalid_configuration", "invalid_symbol",
    "malformed_response", "provider_error", "request_failed", "empty_daily",
    "no_data", "symbol_mismatch",
}
TIMING_DEFINITIONS = {
    "daily_api_ms": "Daily endpoint HTTP request and JSON decoding; zero on a cache hit.",
    "intraday_api_ms": "Intraday endpoint HTTP request and JSON decoding; zero on a cache hit.",
    "data_wall_ms": "Elapsed time waiting for both data loads, which may overlap.",
    "validation_ms": "Sum of normalization and fingerprint work for uncached datasets.",
    "compute_ms": "Elapsed indicator, assessment and support/resistance calculation time.",
    "support_resistance_ms": "Support/resistance subset of compute_ms; do not add twice.",
    "total_ms": "Service analysis elapsed time; excludes browser rendering.",
    "caller_wall_ms": "Elapsed analyze() invocation measured by this CLI.",
    "render_ms": "Unavailable in this CLI; server UI generation is measured separately; browser paint is not instrumented.",
}


class SafeArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        # Unknown arguments can accidentally contain a pasted credential.
        self.exit(2, "Invalid benchmark arguments. Use --help for usage.\n")


class _OfflineResponse:
    status_code = 200

    def __init__(self, records):
        self._records = records

    def json(self):
        return deepcopy(self._records)


class _OfflineSession:
    """Inject local responses into the real adapter without opening a socket."""

    def __init__(self, symbols):
        import numpy as np
        import pandas as pd
        from data.normalize import session_schedule

        schedule = session_schedule("2024-01-02", "2026-10-12").tail(500)
        self._records = {}
        self._lock = Lock()
        self.requests = 0
        for symbol in symbols:
            scale = 1 + sum(map(ord, symbol)) % 7 / 10
            close = (100 + 8 * np.cos(np.arange(len(schedule)) * np.pi / 10)) * scale
            daily = [
                {"date": session.isoformat(), "open": float(price - 0.2 * scale),
                 "high": float(price + scale), "low": float(price - scale),
                 "close": float(price), "volume": 1000.0}
                for session, price in zip(schedule.index, close)
            ]
            latest = schedule.iloc[-1]
            intraday = [
                {"date": stamp.strftime("%Y-%m-%d %H:%M:%S"),
                 "open": float(close[-1]), "high": float(close[-1] + 0.1 * scale),
                 "low": float(close[-1] - 0.1 * scale),
                 "close": float(close[-1]), "volume": 100.0}
                for stamp in pd.date_range(latest.market_open, latest.market_close,
                                           freq="5min", inclusive="left")
            ]
            self._records[symbol] = {"daily": daily, "intraday": intraday}

    def get(self, url, *, params, **kwargs):
        with self._lock:
            self.requests += 1
        timeframe = "intraday" if url.endswith("/historical-chart/5min") else "daily"
        return _OfflineResponse(self._records[params["symbol"]][timeframe])


def _summarize(runs):
    import numpy as np

    summary = {}
    for phase in ("first_visit", "cached_switch"):
        matching = [run for run in runs if run["phase"] == phase and run["success"]]
        metrics = {}
        for name in TIMING_DEFINITIONS:
            values = [run[name] for run in matching if isinstance(run.get(name), (int, float))]
            if values:
                metrics[name] = {
                    "median_ms": round(statistics.median(values), 3),
                    "p95_ms": round(float(np.percentile(values, 95)), 3),
                }
        summary[phase] = {"successful_runs": len(matching), "metrics": metrics}
    return summary


def main(argv=None):
    parser = SafeArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Use genuine FMP requests and local credentials.")
    parser.add_argument("--symbols", nargs="+", default=["NVDA", "AAPL"])
    parser.add_argument("--repeats", type=int, default=10, help="Cached visits per symbol (default: 10).")
    parser.add_argument("--output", type=Path, help="Also save the sanitized JSON report to this file.")
    args = parser.parse_args(argv)
    if not 1 <= args.repeats <= 100:
        parser.error("repeats must be between 1 and 100")

    try:
        from models import DataError
        from config import Settings, load_settings
        from data.fmp_client import FMPClient, validate_symbol
        from services.analysis_service import analyze
    except Exception:
        print("Benchmark unavailable: install the project dependencies first.")
        return 1
    try:
        symbols = list(dict.fromkeys(validate_symbol(symbol) for symbol in args.symbols))
        offline = None
        if args.live:
            settings = load_settings()
            if not settings.api_key or settings.api_key == "your_fmp_api_key":
                print("Live benchmark unavailable: FMP_API_KEY is not configured.")
                return 2
            client, observed_now = FMPClient(settings), None
        else:
            offline = _OfflineSession(symbols)
            client = FMPClient(Settings(api_key="offline-benchmark"), session=offline)
            observed_now = datetime(2026, 10, 12, 17, tzinfo=ZoneInfo("America/New_York"))

        runs = []
        successful_symbols = []
        for phase in ("first_visit", "cached_switch"):
            sequence = symbols if phase == "first_visit" else successful_symbols * args.repeats
            for symbol in sequence:
                started = time.perf_counter()
                try:
                    result = analyze(symbol, client, now=observed_now, refresh=phase == "first_visit")
                    caller_wall = (time.perf_counter() - started) * 1000
                    row = {"phase": phase, "symbol": symbol, "success": True,
                           **asdict(result.timings), "caller_wall_ms": caller_wall,
                           "render_ms": None, "daily_bars": len(result.daily),
                           "intraday_bars": len(result.intraday),
                           "intraday_available": not result.intraday.empty,
                           "daily_session": (result.confirmed_daily_session.isoformat()
                                             if result.confirmed_daily_session else None)}
                    runs.append(row)
                    if phase == "first_visit":
                        successful_symbols.append(symbol)
                except DataError as error:
                    code = (error.code if isinstance(error.code, str) and error.code in SAFE_ERROR_CODES
                            else "request_failed")
                    runs.append({"phase": phase, "symbol": symbol, "success": False,
                                 "error_code": code,
                                 "caller_wall_ms": (time.perf_counter() - started) * 1000})

        report = {
            "mode": "live_fmp" if args.live else "offline_synthetic_fixture",
            "measured_at_utc": datetime.now(timezone.utc).isoformat(),
            "symbols": symbols, "cached_visits_per_symbol": args.repeats,
            "timing_definitions": TIMING_DEFINITIONS,
            "notes": [
                "First visits bypass data caches; later switches reuse one process and client.",
                "Calendar initialization may contribute to first-visit time.",
                "Concurrent API/validation times overlap; stage values are not all additive.",
                "Offline API timings describe local fixture transport, not FMP latency.",
                "Browser rendering is excluded; performance targets are not guarantees.",
            ],
            "offline_transport_requests": offline.requests if offline else None,
            "summary": _summarize(runs), "runs": runs,
        }
        serialized = json.dumps(report, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized + "\n", encoding="utf-8")
        print(serialized)
        return 0 if all(run["success"] for run in runs) else 1
    except DataError as error:
        code = (error.code if isinstance(error.code, str) and error.code in SAFE_ERROR_CODES
                else "request_failed")
        print(f"Benchmark unavailable ({code}); no provider response was logged.")
        return 1
    except Exception:
        print("Benchmark failed; no settings, provider response, or exception message was logged.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
