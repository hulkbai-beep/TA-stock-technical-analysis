"""Rule switches the user sets in the app, saved in `preferences.json` next to the watchlist.

The file holds only on/off choices; it never contains credentials or prices.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from data.watchlist import watchlist_path, write_json_atomic


@dataclass(frozen=True)
class Preferences:
    # Require at least 2:1 reward/risk before a Buy. Off by default; see
    # docs/technical-assessment-rules.md for the backtest behind that choice.
    reward_risk_gate: bool = False


def preferences_path() -> Path:
    return watchlist_path().with_name("preferences.json")


def load_preferences(path: Path | None = None) -> Preferences:
    """A missing or unreadable file means the defaults; nothing else depends on it."""
    try:
        payload = json.loads((path or preferences_path()).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Preferences()
    if not isinstance(payload, dict):
        return Preferences()
    return Preferences(reward_risk_gate=payload.get("reward_risk_gate") is True)


def save_preferences(preferences: Preferences, path: Path | None = None) -> None:
    write_json_atomic(path or preferences_path(), {"reward_risk_gate": preferences.reward_risk_gate})


__all__ = ["Preferences", "load_preferences", "preferences_path", "save_preferences"]
