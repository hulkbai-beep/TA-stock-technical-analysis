"""Run one explicit real NVDA integration check without exposing credentials."""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import sys
import traceback


# Allow execution as `python scripts/check_live.py` from any working directory.
PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))

SAFE_ERROR_CODES = {
    "authentication",
    "authentication_failed",
    "unauthorized",
    "permission_denied",
    "forbidden",
    "rate_limited",
    "rate_limit",
    "request_failed",
    "network_error",
    "timeout",
    "empty_data",
    "empty_daily",
    "no_data",
    "invalid_data",
    "invalid_response",
    "invalid_configuration",
    "malformed_response",
    "provider_error",
    "tls_failed",
    "invalid_symbol",
    "daily_unavailable",
    "missing_key",
}


def _date_text(value: object) -> str:
    """Only display expected typed dates, never arbitrary provider strings."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return "unavailable"


def _diagnostics(error: Exception) -> None:
    """Print stack locations only, excluding messages, source text, and locals."""
    print(f"Exception class: {type(error).__name__}")
    for frame, line in traceback.walk_tb(error.__traceback__):
        print(f"  {Path(frame.f_code.co_filename).name}:{line}:{frame.f_code.co_name}")


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if any(argument not in {"--diagnostics", "--help", "-h"} for argument in arguments):
        # Do not echo unknown arguments, which might contain a pasted credential.
        print("Usage: python scripts/check_live.py [--diagnostics]")
        return 2
    if any(argument in {"--help", "-h"} for argument in arguments):
        print("Usage: python scripts/check_live.py [--diagnostics]")
        print("Check real NVDA data. Diagnostics show exception class and stack locations only.")
        return 0
    diagnostics = "--diagnostics" in arguments
    try:
        from config import load_settings
        from data.fmp_client import FMPClient
        from models import DataError
        from services.analysis_service import analyze
    except Exception as error:
        print("NVDA check unavailable: install the project dependencies first.")
        if diagnostics:
            _diagnostics(error)
        return 1

    try:
        settings = load_settings()
        if not settings.api_key or settings.api_key == "your_fmp_api_key":
            print("NVDA check skipped: FMP_API_KEY is not configured.")
            return 2

        # This is a fresh process; refresh also explicitly bypasses any cache.
        result = analyze("NVDA", FMPClient(settings), refresh=True)
        if result.daily.empty:
            print("NVDA daily: unavailable (no valid daily bars).")
            return 1

        print("NVDA daily: available")
        print(f"Daily rows: {len(result.daily)}")
        print(f"Confirmed daily session: {_date_text(result.confirmed_daily_session)}")
        print(f"Daily retrieved at: {_date_text(result.daily_retrieved_at)}")
        print(f"Snapshot session: {_date_text(result.snapshot.session)}")
        print(f"Snapshot market time: {_date_text(result.snapshot.market_timestamp)}")
        print(f"Snapshot provisional: {bool(result.snapshot.provisional)}")
        if result.assessment is not None:
            print(f"Daily technical bias: {result.assessment.daily.code}")
            print(f"Intraday technical bias: {result.assessment.intraday.code}")
            print(f"Entry guidance: {result.assessment.entry.code}")
            print(f"Holding guidance: {result.assessment.position.code}")

        if result.intraday_error or result.intraday.empty:
            print("NVDA intraday: unavailable; daily features remain usable (partial result).")
        else:
            print("NVDA intraday: available")
            print(f"Intraday rows: {len(result.intraday)}")
            print(f"Intraday session: {_date_text(result.intraday_session)}")
            print(f"Intraday retrieved at: {_date_text(result.intraday_retrieved_at)}")
        print(f"Data-quality warnings: {len(result.warnings)}")
        zones = getattr(result, "support_resistance", None)
        if zones is not None:
            print(f"Support/resistance available: {zones.available}")
            print(f"Support/resistance completed sessions: {zones.actual_sessions}")
            print(f"Selected zones: {len(zones.supports)} support, {len(zones.resistances)} resistance, {len(zones.interactions)} interaction")
            print(f"Estimated volume profile available: {zones.profile.available}")
            print(f"Confirmed AVWAP anchors: {len(zones.anchors)}")
            print(f"Support/resistance warnings: {len(zones.warnings)}")
        return 0
    except DataError as error:
        # Only allow known constant identifiers; never print raw exception text.
        code = error.code if isinstance(error.code, str) and error.code in SAFE_ERROR_CODES else "request_failed"
        print(f"NVDA daily: unavailable ({code}).")
        return 1
    except Exception as error:
        print("NVDA check failed; no provider response or exception message was logged.")
        if diagnostics:
            _diagnostics(error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
