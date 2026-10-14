"""The user's watchlist: a small local JSON file next to `.env`.

The file holds only ticker symbols and a "currently held" flag. It never
contains credentials, prices or position sizes. Set `WATCHLIST_FILE` to keep
it elsewhere.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import tempfile

from config import PROJECT_DIR
from data.fmp_client import validate_symbol
from models import DataError

DEFAULT_SYMBOLS = ("NVDA", "AAPL", "MSFT")
MAX_SYMBOLS = 50
_VERSION = 1


@dataclass(frozen=True)
class WatchItem:
    symbol: str
    holding: bool = False


def watchlist_path() -> Path:
    configured = os.environ.get("WATCHLIST_FILE", "").strip()
    return Path(configured) if configured else PROJECT_DIR / "watchlist.json"


def load_watchlist(path: Path | None = None) -> tuple[list[WatchItem], str | None]:
    """Return the saved list, or the starter list when no file exists yet.

    An unreadable file is left untouched and reported, so a typo made by
    hand never silently wipes the user's list.
    """
    path = path or watchlist_path()
    if not path.exists():
        return [WatchItem(symbol) for symbol in DEFAULT_SYMBOLS], None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload["tickers"] if isinstance(payload, dict) else payload
        items, seen = [], set()
        for row in rows:
            symbol = validate_symbol(row["symbol"] if isinstance(row, dict) else row)
            if symbol in seen:
                continue
            seen.add(symbol)
            items.append(WatchItem(symbol, bool(row.get("holding", False)) if isinstance(row, dict) else False))
        return items[:MAX_SYMBOLS], None
    except (OSError, ValueError, KeyError, TypeError, DataError):
        return [], f"自选股文件 {path.name} 无法读取，已暂停保存以免覆盖；请检查或删除该文件。"


def save_watchlist(items: list[WatchItem], path: Path | None = None) -> None:
    """Write atomically so an interrupted save cannot leave a half-written file."""
    path = path or watchlist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": _VERSION, "tickers": [{"symbol": item.symbol, "holding": item.holding} for item in items]}
    descriptor, temporary = tempfile.mkstemp(prefix=".watchlist-", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def add_symbol(items: list[WatchItem], symbol: str, holding: bool = False) -> list[WatchItem]:
    cleaned = validate_symbol(symbol)
    if any(item.symbol == cleaned for item in items):
        raise DataError(f"{cleaned} 已在自选股中。", "duplicate_symbol")
    if len(items) >= MAX_SYMBOLS:
        raise DataError(f"自选股最多 {MAX_SYMBOLS} 只；请先移除部分股票。", "watchlist_full")
    return [*items, WatchItem(cleaned, holding)]


def remove_symbols(items: list[WatchItem], symbols) -> list[WatchItem]:
    removed = set(symbols)
    return [item for item in items if item.symbol not in removed]


def set_holding(items: list[WatchItem], symbol: str, holding: bool) -> list[WatchItem]:
    return [replace(item, holding=holding) if item.symbol == symbol else item for item in items]


__all__ = ["DEFAULT_SYMBOLS", "MAX_SYMBOLS", "WatchItem", "add_symbol", "load_watchlist", "remove_symbols",
           "save_watchlist", "set_holding", "watchlist_path"]
