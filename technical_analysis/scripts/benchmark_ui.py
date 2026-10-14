"""Measure server UI generation with preloaded, isolated benchmark fixtures.

No credential is loaded and no network request is made. AppTest runs the actual
dashboard component/chart code. It does not run a browser or measure paint.
"""
from dataclasses import replace
import json
from pathlib import Path
import statistics
import sys
from time import perf_counter
from datetime import datetime
from zoneinfo import ZoneInfo

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def main():
    import numpy as np
    from streamlit.testing.v1 import AppTest
    from config import Settings
    from data.fmp_client import FMPClient
    from services.analysis_service import analyze
    from scripts.benchmark_performance import _OfflineSession

    symbols = ["NVDA", "AAPL"]
    client = FMPClient(Settings(api_key="offline-benchmark"), session=_OfflineSession(symbols))
    now = datetime(2026, 10, 12, 17, tzinfo=ZoneInfo("America/New_York"))
    # Disable the live daily-close watcher for isolated fixed-time fixtures.
    results = {symbol: replace(analyze(symbol, client, now=now), market_versions=None)
               for symbol in symbols}
    app = AppTest.from_file(str(PROJECT / "views" / "detail.py"))
    rows = []
    try:
        for index, symbol in enumerate(symbols * 6):
            app.session_state["analysis_result"] = results[symbol]
            app.session_state["ticker_input"] = symbol
            started = perf_counter()
            app.run(timeout=30)
            if app.exception:
                raise RuntimeError("UI benchmark failed")
            rows.append({"symbol": symbol, "warmup": index < 2,
                         "server_ui_ms": app.session_state["last_ui_render_ms"],
                         "app_test_run_ms": (perf_counter() - started) * 1000})
        measured = [row for row in rows if not row["warmup"]]
        summary = {name: {"median_ms": round(statistics.median(row[name] for row in measured), 3),
                          "p95_ms": round(float(np.percentile([row[name] for row in measured], 95)), 3)}
                   for name in ("server_ui_ms", "app_test_run_ms")}
        report = {"mode": "offline_preloaded_full_dashboard", "samples": len(measured),
                  "daily_bars": 500, "intraday_bars": 78,
                  "notes": ["server_ui_ms includes chart construction, serialization and UI emission.",
                            "app_test_run_ms includes the full script and AppTest harness overhead.",
                            "Neither duration measures browser paint or transport.",
                            "Two initial runs are excluded as warmup; no API requests are made."],
                  "summary": summary, "runs": rows}
        output = PROJECT / "artifacts" / "performance-ui.json"
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"samples": len(measured), **summary}, indent=2))
    finally:
        try:
            app.session_state["ticker_requests"].close()
        except KeyError:
            pass


if __name__ == "__main__":
    main()
