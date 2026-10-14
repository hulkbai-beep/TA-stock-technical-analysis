"""Offline smoke test of the backtest command with a routed fake FMP session."""
import importlib.util
from pathlib import Path

import numpy as np

from data.normalize import session_schedule
from data import fmp_client


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "backtest.py"
    spec = importlib.util.spec_from_file_location("backtest_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Response:
    def __init__(self, payload):
        self.payload, self.status_code = payload, 200

    def json(self):
        return self.payload


class _Session:
    trust_env = True

    def __init__(self):
        days = session_schedule("2020-01-02", "2025-12-31").index
        self.daily = {}
        for seed, symbol in enumerate(("SPY", "XLK", "AAA")):
            rng = np.random.default_rng(seed)
            close = 100 * np.exp(np.cumsum(rng.normal(0.0006, 0.015, len(days)) + 0.004 * np.sin(np.arange(len(days)) / 12)))
            self.daily[symbol] = [{"date": str(day), "open": c, "high": c * 1.006, "low": c * 0.994, "close": c,
                                   "volume": 1_000_000} for day, c in zip(days, close)]

    def get(self, url, params=None, **kwargs):
        endpoint, symbol = url.split("/stable/")[1], params["symbol"]
        if endpoint == "profile":
            return _Response([{"symbol": symbol, "sector": "Technology"}])
        if endpoint == "earnings":
            return _Response([{"symbol": symbol, "date": "2023-05-01"}, {"symbol": symbol, "date": "2024-05-01"}])
        rows = [row for row in self.daily[symbol] if params["from"] <= row["date"] <= params["to"]]
        return _Response(rows[::-1])


def test_backtest_command_runs_offline_and_writes_csv(monkeypatch, tmp_path, capsys):
    module = _load_script()
    session = _Session()
    monkeypatch.setenv("FMP_API_KEY", "offline-test-key")
    monkeypatch.setattr(module, "PROJECT_DIR", tmp_path)
    original = fmp_client.FMPClient.__init__
    monkeypatch.setattr(module.FMPClient, "__init__", lambda self, settings, http=None: original(self, settings, session))
    assert module.main(["AAA", "--years", "3", "--ablation"]) == 0
    output = capsys.readouterr().out
    assert "Per-symbol results" in output and "Ablation" in output and "offline-test-key" not in output
    assert "Trades by entry setup" in output
    assert "no reward/risk gate" in output and "no breakout setup" in output
    for name in ("symbols.csv", "score_buckets.csv", "setups.csv", "trades.csv", "ablation.csv"):
        assert (tmp_path / "artifacts" / "backtest" / name).is_file()
