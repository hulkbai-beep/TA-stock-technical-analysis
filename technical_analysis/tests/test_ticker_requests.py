"""Deterministic concurrency checks without network calls or UI state access."""
from concurrent.futures import CancelledError
from dataclasses import FrozenInstanceError
from threading import Event
from time import monotonic
from types import SimpleNamespace

import pytest

from models import DataError
from services.ticker_requests import LatestTickerRequest


@pytest.fixture
def coordinator():
    instance = LatestTickerRequest()
    yield instance
    instance.close()


def controlled_loader(symbol, started, release, *, error=None, marker=None):
    def load():
        started.set()
        assert release.wait(3), "Test failed to release its worker"
        if error is not None:
            raise error
        return SimpleNamespace(symbol=symbol, marker=marker)
    return load


def await_status(coordinator, status="ready"):
    deadline = monotonic() + 3
    while monotonic() < deadline:
        state = coordinator.poll()
        if state.status == status:
            return state
        Event().wait(0.001)
    pytest.fail(f"Coordinator did not reach {status}")


def await_workers(coordinator):
    deadline = monotonic() + 3
    while monotonic() < deadline:
        with coordinator._lock:
            if not coordinator._inflight:
                return
        Event().wait(0.001)
    pytest.fail("Workers did not finish their completion callbacks")


def test_slow_a_cannot_overwrite_fast_b(coordinator):
    started, release = Event(), Event()
    try:
        first = coordinator.submit(" aapl ", controlled_loader("AAPL", started, release))
        assert started.wait(1)
        second = coordinator.submit("MSFT", lambda: SimpleNamespace(symbol="MSFT"))
        state = await_status(coordinator)
        assert state.request_id == second > first
        assert state.result.symbol == coordinator.current_symbol == "MSFT"
        release.set()
        await_workers(coordinator)
        assert coordinator.poll() == state
        assert not coordinator.is_current(first, "AAPL")
    finally:
        release.set()


@pytest.mark.parametrize("failure", [DataError("Safe error", "network"), RuntimeError("SECRET_API_KEY")])
def test_stale_error_is_ignored(coordinator, failure):
    started, release = Event(), Event()
    try:
        coordinator.submit("AAPL", controlled_loader("AAPL", started, release, error=failure))
        assert started.wait(1)
        stale = next(iter(coordinator._inflight))
        coordinator.submit("MSFT", lambda: SimpleNamespace(symbol="MSFT"))
        state = await_status(coordinator)
        release.set()
        with pytest.raises(type(failure)):
            stale.result(timeout=1)
        await_workers(coordinator)
        assert coordinator.poll() == state
        assert coordinator.poll().error is None
    finally:
        release.set()


def test_a_b_a_uses_generation_not_only_symbol(coordinator):
    first_started, first_release = Event(), Event()
    second_started, second_release = Event(), Event()
    try:
        first = coordinator.submit("AAPL", controlled_loader("AAPL", first_started, first_release, marker="old"))
        assert first_started.wait(1)
        second = coordinator.submit("MSFT", controlled_loader("MSFT", second_started, second_release))
        assert second_started.wait(1)
        third = coordinator.submit("AAPL", lambda: SimpleNamespace(symbol="AAPL", marker="new"))
        assert first < second < third
        assert coordinator.poll().result is None
        first_release.set()
        state = await_status(coordinator)
        assert state.request_id == third and state.result.marker == "new"
        assert not coordinator.is_current(first, "AAPL")
        second_release.set()
        await_workers(coordinator)
        assert coordinator.poll().request_id == third
    finally:
        first_release.set()
        second_release.set()


def test_pending_requests_are_replaced_without_executor_backlog(coordinator):
    first_started, second_started, release = Event(), Event(), Event()
    calls = []
    try:
        coordinator.submit("AAPL", controlled_loader("AAPL", first_started, release))
        assert first_started.wait(1)
        coordinator.submit("MSFT", controlled_loader("MSFT", second_started, release))
        assert second_started.wait(1)
        for symbol in ("META", "AMZN", "NVDA", "TSLA"):
            coordinator.submit(symbol, lambda selected=symbol: calls.append(selected) or SimpleNamespace(symbol=selected))
        assert len(coordinator._inflight) == 2
        assert calls == []
        release.set()
        state = await_status(coordinator)
        assert state.symbol == "TSLA"
        assert calls == ["TSLA"]
    finally:
        release.set()


def test_invalid_symbol_supersedes_old_work_without_calling_loader(coordinator):
    started, release = Event(), Event()
    calls = []
    try:
        old_id = coordinator.submit("AAPL", controlled_loader("AAPL", started, release))
        assert started.wait(1)
        stale = next(iter(coordinator._inflight))
        invalid_id = coordinator.submit("INVALID/INPUT", lambda: calls.append(True))
        state = coordinator.poll()
        assert invalid_id > old_id
        assert state.status == "error" and state.error.code == "invalid_symbol"
        assert state.result is None and state.symbol is None and calls == []
        release.set()
        stale.result(timeout=1)
        await_workers(coordinator)
        assert coordinator.poll() == state
    finally:
        release.set()


@pytest.mark.parametrize("failure", [RuntimeError, CancelledError])
def test_current_unexpected_failure_is_sanitized(coordinator, failure):
    def fail():
        raise failure("https://provider/?apikey=SECRET_API_KEY")
    coordinator.submit("AAPL", fail)
    state = await_status(coordinator, "error")
    assert state.error.code == "analysis_failed"
    assert "SECRET" not in str(state.error)
    assert state.error.__cause__ is None and state.error.__context__ is None


def test_current_data_error_preserves_safe_code(coordinator):
    def fail():
        raise DataError("Rate limited", "rate_limited")
    coordinator.submit("AAPL", fail)
    assert await_status(coordinator, "error").error.code == "rate_limited"


def test_same_symbol_is_new_request_and_clears_old_result(coordinator):
    first = coordinator.submit("AAPL", lambda: SimpleNamespace(symbol="AAPL"))
    await_status(coordinator)
    started, release = Event(), Event()
    try:
        second = coordinator.submit("AAPL", controlled_loader("AAPL", started, release), refresh=True)
        assert second > first
        assert coordinator.poll().status == "loading"
        assert coordinator.poll().result is None and coordinator.poll().error is None
    finally:
        release.set()


def test_mismatched_loader_result_is_never_published(coordinator):
    coordinator.submit("AAPL", lambda: SimpleNamespace(symbol="MSFT"))
    state = await_status(coordinator, "error")
    assert state.result is None and state.error.code == "symbol_mismatch"


def test_browser_sessions_are_isolated(coordinator):
    other = LatestTickerRequest()
    try:
        coordinator.submit("AAPL", lambda: SimpleNamespace(symbol="AAPL"))
        other.submit("MSFT", lambda: SimpleNamespace(symbol="MSFT"))
        assert await_status(coordinator).symbol == "AAPL"
        assert await_status(other).symbol == "MSFT"
        assert coordinator.current_symbol == "AAPL"
    finally:
        other.close()


def test_close_drops_pending_work_and_late_completions(coordinator):
    first_started, second_started, release = Event(), Event(), Event()
    calls = []
    try:
        coordinator.submit("AAPL", controlled_loader("AAPL", first_started, release))
        assert first_started.wait(1)
        coordinator.submit("MSFT", controlled_loader("MSFT", second_started, release))
        assert second_started.wait(1)
        running = tuple(coordinator._inflight)
        coordinator.submit("NVDA", lambda: calls.append(True))
        coordinator.close()
        state = coordinator.poll()
        assert state.status == "idle" and state.result is None
        release.set()
        for future in running:
            future.result(timeout=1)
        await_workers(coordinator)
        assert coordinator.poll() == state and calls == []
        coordinator.close()
        with pytest.raises(RuntimeError, match="closed"):
            coordinator.submit("TSLA", lambda: None)
    finally:
        release.set()


def test_state_is_immutable_and_starts_idle(coordinator):
    state = coordinator.poll()
    assert state.status == "idle" and state.request_id == 0
    with pytest.raises(FrozenInstanceError):
        state.symbol = "AAPL"
