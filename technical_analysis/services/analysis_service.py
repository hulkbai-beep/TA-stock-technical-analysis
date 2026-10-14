"""Coordinate genuine market data, calculations, and deterministic summaries."""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta, timezone
from threading import RLock
from time import perf_counter
from typing import Protocol
from weakref import WeakKeyDictionary

import pandas as pd

from analysis.indicators import daily_indicators, session_vwap, visible_daily
from analysis.summary import SummaryItem, generate_summaries
from analysis.signals import MarketContext, TechnicalAssessment, evaluate_signals
from analysis.strategy import attach_benchmarks
from analysis.support_resistance.engine import analyze_support_resistance
from analysis.support_resistance.models import SRConfig, SupportResistanceResult
from data.normalize import normalize_daily, normalize_intraday, session_schedule
from data.market_store import ADJUSTMENT_POLICY, MarketDataset, content_version
from data.fmp_client import validate_symbol
from models import DataError, RawDataset


class MarketDataClient(Protocol):
    def fetch_daily(self, symbol: str, now: datetime | None = None, refresh: bool = False) -> RawDataset: ...
    def fetch_intraday(self, symbol: str, now: datetime | None = None, refresh: bool = False) -> RawDataset: ...


@dataclass(frozen=True)
class Snapshot:
    price: float
    change_percent: float | None
    session: date
    market_timestamp: datetime | None
    source: str
    provisional: bool
    volume: float | None = None
    timestamp_note: str = ""
    volume_note: str = ""


@dataclass(frozen=True)
class AnalysisTimings:
    """Measured durations, never estimates of browser paint or provider freshness."""
    daily_api_ms: float = 0.0
    intraday_api_ms: float = 0.0
    data_wall_ms: float = 0.0
    validation_ms: float = 0.0
    compute_ms: float = 0.0
    support_resistance_ms: float = 0.0
    total_ms: float = 0.0
    daily_cache_hit: bool = False
    intraday_cache_hit: bool = False
    derived_cache_hit: bool = False


@dataclass
class AnalysisResult:
    symbol: str
    daily: pd.DataFrame
    intraday: pd.DataFrame
    snapshot: Snapshot
    summaries: list[SummaryItem]
    warnings: list[str]
    daily_retrieved_at: datetime
    intraday_retrieved_at: datetime | None = None
    intraday_error: str | None = None
    confirmed_daily_session: date | None = None
    intraday_session: date | None = None
    assessment: TechnicalAssessment | None = None
    support_resistance: SupportResistanceResult | None = None
    timings: AnalysisTimings = field(default_factory=AnalysisTimings)
    sr_variants: dict[SRConfig, SupportResistanceResult] = field(default_factory=dict, repr=False)
    context: MarketContext = field(default_factory=MarketContext)
    market_versions: tuple[str, str | None] | None = field(default=None, repr=False)


# CPU calculations are small and mostly Python work. Parallelize the independent
# HTTP/validation paths; keep numerical rules sequential and deterministic.
_DATA_WORKERS = ThreadPoolExecutor(max_workers=4, thread_name_prefix="market-data")
_DERIVED: WeakKeyDictionary = WeakKeyDictionary()
_DERIVED_LOCK = RLock()
_MAX_DERIVED_PER_CLIENT = 64


def _derived_cache(client) -> OrderedDict | None:
    """Client identity scopes provider credentials without putting them in keys."""
    try:
        with _DERIVED_LOCK:
            return _DERIVED.setdefault(client, OrderedDict())
    except TypeError:
        # Unhashable/non-weakrefable third-party adapters still work without caching.
        return None


def _load_market(client, symbol: str, timeframe: str, now, refresh: bool) -> MarketDataset:
    method = getattr(client, f"fetch_validated_{timeframe}", None)
    if callable(method):
        return method(symbol, now=now, refresh=refresh)
    raw = getattr(client, f"fetch_{timeframe}")(symbol, now=now, refresh=refresh)
    started = perf_counter()
    asof = min(_local_timestamp(now), _local_timestamp(raw.retrieved_at))
    normalizer = normalize_daily if timeframe == "daily" else normalize_intraday
    normalized = normalizer(raw.records, now=asof.to_pydatetime())
    frame = normalized.frame
    return MarketDataset(frame, normalized.issues, raw.retrieved_at, raw.endpoint,
                         None if frame.empty else frame.iloc[-1]["session"],
                         content_version(frame, timeframe), raw.cache_hit,
                         raw.api_seconds, perf_counter() - started)


def _local_timestamp(value: datetime | None = None) -> pd.Timestamp:
    stamp = pd.Timestamp(value or datetime.now(timezone.utc))
    return stamp.tz_localize("America/New_York") if stamp.tzinfo is None else stamp.tz_convert("America/New_York")


def assessment_for(result: AnalysisResult, now: datetime | None = None) -> TechnicalAssessment:
    """Evaluate cached completed bars without I/O or any holding-state input.

    Rechecking the clock on UI reruns prevents an old entry confirmation from
    staying actionable after its freshness window or trading session ends.
    Bar completion itself remains frozen at the original data retrieval time.
    """
    return evaluate_signals(
        result.daily, result.intraday,
        now=_local_timestamp(now).to_pydatetime(),
        daily_retrieved_at=result.daily_retrieved_at,
        intraday_retrieved_at=result.intraday_retrieved_at,
        context=getattr(result, "context", None),
        support_resistance=getattr(result, "support_resistance", None),
    )


def support_resistance_for(result: AnalysisResult, config: SRConfig | None = None) -> SupportResistanceResult:
    """Analyze the already-retrieved daily history; controls never fetch data."""
    selected = config or SRConfig()
    existing = getattr(result, "support_resistance", None)
    if existing is not None and existing.config == selected:
        return existing
    if selected in result.sr_variants:
        return result.sr_variants[selected]
    calculated = analyze_support_resistance(
        result.daily, selected, current_price=result.snapshot.price,
        reference_timestamp=result.snapshot.market_timestamp,
        reference_provisional=result.snapshot.provisional,
    )
    if len(result.sr_variants) >= 8:
        result.sr_variants.pop(next(iter(result.sr_variants)))
    result.sr_variants[selected] = calculated
    return calculated


def _previous_session(session: date) -> date | None:
    schedule = session_schedule(session - timedelta(days=20), session - timedelta(days=1))
    return schedule.index[-1] if len(schedule) else None


def _previous_close(daily: pd.DataFrame, session: date) -> tuple[float | None, date | None]:
    previous = _previous_session(session)
    matched = daily.loc[(daily["session"] == previous) & daily["complete"].astype(bool)]
    if matched.empty:
        return None, None
    return float(matched.iloc[-1]["close"]), previous


_FEATURES = ("sma20", "sma50", "sma200", "rsi14", "ret63", "ret126", "atr14", "high22", "prev_high",
             "pullback_atr5", "fib_high", "fib_low", "fib_up", "vol_avg20", "rel_volume", "updown_volume20")


def calendar_indicators(daily: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Use missing-session NaNs only for calculations, never as market bars."""
    schedule = session_schedule(daily["session"].min(), daily["session"].max())
    columns = [column for column in ("close", "high", "low", "volume") if column in daily.columns]
    bars = daily.drop_duplicates("session", keep="last").set_index("session")[columns].reindex(schedule.index)
    calculations = daily_indicators(bars)
    result = daily.copy()
    for column in _FEATURES:
        if column in calculations:
            result[column] = result["session"].map(calculations[column])
    return result, int(bars["close"].isna().sum())


_calendar_indicators = calendar_indicators


def _load_context(client, symbol: str, now) -> tuple[MarketContext, pd.DataFrame | None, pd.DataFrame | None, tuple]:
    """Benchmarks and events are optional; any failure leaves them unknown."""
    method = getattr(client, "fetch_context", None)
    if not callable(method):
        return MarketContext(notes=("当前数据源未提供大盘、行业或财报信息；相关条件显示为无法核实。",)), None, None, ()
    try:
        context = method(symbol, now=now)
    except Exception:
        return MarketContext(notes=("大盘、行业或财报信息读取失败；相关条件显示为无法核实。",)), None, None, ()
    return (context.context, context.market, context.sector, context.versions)


def _snapshot(daily: pd.DataFrame, intraday: pd.DataFrame) -> Snapshot:
    latest_daily = daily.iloc[-1]
    chosen = latest_daily
    intraday_source = False
    if not intraday.empty:
        latest_intraday = intraday.iloc[-1]
        # A date-only unfinished daily observation cannot prove it is newer
        # than an intraday bar on the same date. Prefer the timed observation.
        if (latest_intraday["session"] > latest_daily["session"] or
                (latest_intraday["session"] == latest_daily["session"] and
                 (not bool(latest_daily["complete"]) or
                  latest_intraday["bar_end"] > latest_daily["bar_end"]))):
            chosen = latest_intraday
            intraday_source = True
    complete = bool(chosen["complete"])
    session = chosen["session"]
    price = float(chosen["close"])
    previous_close, _ = _previous_close(daily, session)
    change_percent = (price / previous_close - 1) * 100 if previous_close is not None else None
    if intraday_source:
        # Bar interval times are source market-data times; retrieval is kept
        # separately. No last-trade timestamp is invented from a fetch time.
        timestamp = pd.Timestamp(chosen["bar_end"] if complete else chosen["timestamp"]).to_pydatetime()
        note = "5 分钟K线区间结束时刻；不代表逐笔成交时刻。" if complete else "未完成 5 分钟K线的区间开始时刻；最新成交时刻未知。"
        volume = chosen.get("session_volume")
        volume = None if pd.isna(volume) else float(volume)
        return Snapshot(price, change_percent, session, timestamp, "5 分钟K线", not complete, volume, note, "截至所示K线的交易时段累计成交量（含未完成K线时为临时值）。" if volume is not None else "缺少完整交易时段数据，累计成交量不可用。")
    timestamp = pd.Timestamp(chosen["bar_end"]).to_pydatetime() if complete else None
    note = "由交易日日历推定的正常时段收盘时刻；日线接口未提供逐笔成交时刻。" if complete else "日线接口仅提供交易日，未完成日线的最新成交时刻未知。"
    return Snapshot(price, change_percent, session, timestamp, "日线收盘" if complete else "未完成日线", not complete, float(chosen["volume"]), note, "已完成日线成交量。" if complete else "未完成日线累计成交量，为临时值。")


def analyze(
    symbol: str,
    client: MarketDataClient,
    now: datetime | None = None,
    refresh: bool = False,
) -> AnalysisResult:
    """Daily failures stop analysis; unavailable intraday remains partial success.

    A cached response is normalized at min(request time, retrieval time).
    Consequently a cached unfinished candle cannot become a confirmed candle
    merely because the user opens the page later.
    """
    started = perf_counter()
    symbol = validate_symbol(symbol)
    local_now = _local_timestamp(now)
    daily_future = _DATA_WORKERS.submit(_load_market, client, symbol, "daily", now, refresh)
    intraday_future = _DATA_WORKERS.submit(_load_market, client, symbol, "intraday", now, refresh)
    context_future = _DATA_WORKERS.submit(_load_context, client, symbol, now)
    daily_data = daily_future.result()
    if daily_data.frame.empty:
        raise DataError("未取得有效的正常交易时段日线数据。请检查股票代码、数据权限或稍后刷新。", "empty_daily")
    intraday_data = None
    intraday_error = None
    error_code = None
    intraday_api_ms = 0.0
    intraday_cache_hit = False
    try:
        intraday_data = intraday_future.result()
    except DataError as error:
        intraday_error, error_code = str(error), error.code
        intraday_api_ms = getattr(error, "api_seconds", 0.0) * 1000
        intraday_cache_hit = getattr(error, "cache_hit", False)
    context, market_frame, sector_frame, context_versions = context_future.result()
    data_wall_ms = (perf_counter() - started) * 1000
    daily_retrieved = _local_timestamp(daily_data.retrieved_at).to_pydatetime()
    intraday_retrieved = None if intraday_data is None else _local_timestamp(intraday_data.retrieved_at).to_pydatetime()
    if intraday_data is not None:
        intraday_api_ms = intraday_data.api_seconds * 1000
        intraday_cache_hit = intraday_data.cache_hit
    timing_base = AnalysisTimings(
        daily_api_ms=daily_data.api_seconds * 1000, intraday_api_ms=intraday_api_ms,
        data_wall_ms=data_wall_ms,
        validation_ms=(daily_data.validation_seconds + (0 if intraday_data is None else intraday_data.validation_seconds)) * 1000,
        daily_cache_hit=daily_data.cache_hit, intraday_cache_hit=intraday_cache_hit,
    )
    # A content revision or changed completion flags necessarily selects new work.
    # Retrieval time alone does not invalidate unchanged numerical calculations.
    key = (symbol, ADJUSTMENT_POLICY, daily_data.latest_session, daily_data.version,
           tuple(daily_data.issues), None if intraday_data is None else intraday_data.latest_session,
           None if intraday_data is None else intraday_data.version,
           () if intraday_data is None else tuple(intraday_data.issues), error_code, context_versions,
           context.next_earnings, context.earnings_checked, context.notes)
    cache = _derived_cache(client)
    compute_started = perf_counter()
    with _DERIVED_LOCK:
        if refresh and cache is not None:
            for old_key in [item for item in cache if item[0] == symbol]:
                del cache[old_key]
        cached = None if cache is None else cache.get(key)
        if cached is not None:
            cache.move_to_end(key)
    if cached is not None:
        result = replace(cached, daily=cached.daily.copy(), intraday=cached.intraday.copy(),
                         summaries=list(cached.summaries), warnings=list(cached.warnings),
                         sr_variants=dict(cached.sr_variants), daily_retrieved_at=daily_retrieved,
                         intraday_retrieved_at=intraday_retrieved, intraday_error=intraday_error or cached.intraday_error)
        result.assessment = assessment_for(result, now=local_now.to_pydatetime())
        result.timings = replace(timing_base, compute_ms=(perf_counter() - compute_started) * 1000,
                                 total_ms=(perf_counter() - started) * 1000, derived_cache_hit=True)
        return result
    daily, missing_count = calendar_indicators(daily_data.frame)
    daily = attach_benchmarks(daily, market_frame, sector_frame)
    warnings = list(daily_data.issues)
    if missing_count:
        warnings.append(f"日线历史缺少 {missing_count} 个 NYSE 交易日；指标计算在缺口处重新预热，不填补价格。")
    intraday = daily_data.frame.iloc[0:0].copy()
    if intraday_data is not None:
        warnings.extend(intraday_data.issues)
        calculated = session_vwap(intraday_data.frame)
        if calculated.empty:
            intraday_error = "未取得可用的正常交易时段 5 分钟数据。日线功能仍可使用。"
        else:
            latest_session = calculated["session"].max()
            intraday = calculated.loc[calculated["session"] == latest_session].copy()
    completed = daily.loc[daily["complete"].astype(bool)]
    confirmed_session = None if completed.empty else completed.iloc[-1]["session"]
    intraday_session = None if intraday.empty else intraday.iloc[-1]["session"]
    previous_close, previous_close_session = (None, None) if intraday_session is None else _previous_close(daily, intraday_session)
    comparable = (confirmed_session is not None and intraday_session is not None and
                  (intraday_session == confirmed_session or _previous_session(intraday_session) == confirmed_session))
    summaries = generate_summaries(daily, intraday, previous_close=previous_close, previous_close_session=previous_close_session, comparable_sessions=comparable)
    snapshot = _snapshot(daily, intraday)
    if intraday_session is not None and intraday_session != snapshot.session:
        warnings.append(f"日内图来自 {intraday_session}，与市场快照交易日 {snapshot.session} 不同；请按各自日期解读。")
    if snapshot.change_percent is None:
        warnings.append("缺少确切上一交易日的已完成日线，涨跌幅不可用。")
    sr_started = perf_counter()
    sr_result = analyze_support_resistance(
        daily, current_price=snapshot.price, reference_timestamp=snapshot.market_timestamp,
        reference_provisional=snapshot.provisional,
    )
    sr_ms = (perf_counter() - sr_started) * 1000
    warnings.extend(context.notes)
    result = AnalysisResult(
        symbol=symbol,
        daily=daily,
        intraday=intraday,
        snapshot=snapshot,
        summaries=summaries,
        warnings=list(dict.fromkeys(warnings)),
        daily_retrieved_at=daily_retrieved,
        intraday_retrieved_at=intraday_retrieved,
        intraday_error=intraday_error,
        confirmed_daily_session=confirmed_session,
        intraday_session=intraday_session,
        assessment=evaluate_signals(
            daily, intraday, now=local_now.to_pydatetime(),
            daily_retrieved_at=daily_retrieved,
            intraday_retrieved_at=intraday_retrieved,
            context=context, support_resistance=sr_result,
        ),
        support_resistance=sr_result,
        context=context,
        market_versions=(daily_data.version, None if intraday_data is None else intraday_data.version),
    )
    result.timings = replace(timing_base, compute_ms=(perf_counter() - compute_started) * 1000,
                             support_resistance_ms=sr_ms, total_ms=(perf_counter() - started) * 1000)
    if cache is not None:
        with _DERIVED_LOCK:
            cache[key] = replace(result, daily=result.daily.copy(), intraday=result.intraday.copy(),
                                 summaries=list(result.summaries), warnings=list(result.warnings), sr_variants={})
            cache.move_to_end(key)
            while len(cache) > _MAX_DERIVED_PER_CLIENT:
                cache.popitem(last=False)
    return result


__all__ = ["AnalysisTimings", "AnalysisResult", "MarketContext", "calendar_indicators", "Snapshot", "SummaryItem", "TechnicalAssessment", "SRConfig", "SupportResistanceResult", "analyze", "assessment_for", "support_resistance_for", "visible_daily"]
