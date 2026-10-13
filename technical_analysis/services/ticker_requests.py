"""Per-session, bounded background loading with latest-request-wins publishing."""
from __future__ import annotations

from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
from dataclasses import dataclass
from threading import RLock
from typing import TYPE_CHECKING, Callable, Literal

from data.fmp_client import validate_symbol
from models import DataError

if TYPE_CHECKING:
    from services.analysis_service import AnalysisResult


@dataclass(frozen=True)
class RequestState:
    request_id: int = 0
    symbol: str | None = None
    status: Literal["idle", "loading", "ready", "error"] = "idle"
    result: AnalysisResult | None = None
    error: DataError | None = None


class LatestTickerRequest:
    """Own one browser session's state; workers never access Streamlit state.

    At most ``max_workers`` loaders are submitted, plus one replaceable pending
    loader. HTTP already in progress cannot be interrupted: its completion may
    warm service caches, but cannot publish a stale result or stale exception.
    """

    def __init__(self, max_workers: int = 2):
        if max_workers not in (2, 3):
            raise ValueError("max_workers must be 2 or 3")
        self._lock = RLock()
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="ticker-analysis")
        self._max_workers = max_workers
        self._state = RequestState()
        self._inflight: set[Future] = set()
        self._latest_future: Future | None = None
        self._pending: tuple[int, str, Callable[[], AnalysisResult]] | None = None
        self._closed = False

    @property
    def current_symbol(self) -> str | None:
        return self.poll().symbol

    def poll(self) -> RequestState:
        with self._lock:
            return self._state

    def is_current(self, request_id: int, symbol: str) -> bool:
        """Guard UI commits against a selection made since its last poll."""
        with self._lock:
            return not self._closed and self._state.request_id == request_id and self._state.symbol == symbol

    def submit(self, symbol: str, loader: Callable[[], AnalysisResult], *, refresh: bool = False) -> int:
        """Supersede every earlier request, including another for the same symbol.

        ``loader`` owns refresh/cache semantics; the optional flag documents the
        caller's intent without causing additional submissions or API requests.
        Validation failures supersede old work without invoking the loader.
        """
        with self._lock:
            if self._closed:
                raise RuntimeError("Request coordinator is closed")
            request_id = self._state.request_id + 1
            self._state = RequestState(request_id=request_id)
            self._pending = None
            if self._latest_future is not None:
                self._latest_future.cancel()
            try:
                selected = validate_symbol(symbol)
            except DataError as error:
                self._state = RequestState(request_id=request_id, status="error", error=error)
                return request_id
            self._state = RequestState(request_id, selected, "loading")
            self._pending = (request_id, selected, loader)
            self._start_pending()
            return request_id

    def _start_pending(self) -> None:
        """Called with the lock held; do not enqueue an unbounded executor backlog."""
        if self._closed or self._pending is None or len(self._inflight) >= self._max_workers:
            return
        request_id, symbol, loader = self._pending
        self._pending = None
        future = self._executor.submit(loader)
        self._inflight.add(future)
        self._latest_future = future
        future.add_done_callback(lambda completed: self._completed(request_id, symbol, completed))

    def _completed(self, request_id: int, symbol: str, future: Future) -> None:
        result, error = None, None
        try:
            result = future.result()
            if result.symbol != symbol:
                result = None
                error = DataError("分析结果与当前股票代码不一致，请重新分析。", "symbol_mismatch")
        except CancelledError:
            if not future.cancelled():
                error = DataError("分析暂时无法完成，请重试。", "analysis_failed")
        except DataError as failure:
            result = None
            error = DataError(str(failure), failure.code)
        except Exception:
            result = None
            error = DataError("分析暂时无法完成，请重试。", "analysis_failed")
        with self._lock:
            self._inflight.discard(future)
            if self.is_current(request_id, symbol) and not future.cancelled():
                self._state = RequestState(request_id, symbol, "error" if error else "ready", result, error)
            self._start_pending()

    def close(self) -> None:
        """Drop pending work; let bounded, already-running requests finish safely."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._pending = None
            self._state = RequestState(request_id=self._state.request_id + 1)
            for future in tuple(self._inflight):
                future.cancel()
        self._executor.shutdown(wait=False, cancel_futures=True)
