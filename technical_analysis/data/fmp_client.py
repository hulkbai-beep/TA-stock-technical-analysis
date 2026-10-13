"""Small FMP OHLCV adapter with bounded process-local caching.

Never use provider bodies, request URLs, or underlying exceptions as display text.
The two selected stable endpoints use split-adjusted prices; adjClose is unused.
"""
from collections import OrderedDict
from concurrent.futures import Future
from copy import deepcopy
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import re
from threading import RLock, local
import time
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import requests

from config import Settings
from data.market_store import (ADJUSTMENT_POLICY, MarketDataset, MarketStore,
                               content_version, freshness_epoch)
from data.normalize import normalize_daily, normalize_intraday
from models import DataError, RawDataset

NEW_YORK = ZoneInfo("America/New_York")
_SYMBOL = re.compile(r"[A-Z][A-Z0-9]{0,5}(?:[.-][A-Z0-9]{1,2})?\Z")
_ERROR_KEYS = {"error", "error message", "errormessage", "message", "information", "note"}
_MAX_CACHE_ENTRIES = 128


def validate_symbol(symbol: str) -> str:
    """Accept one conventional US equity symbol, including BRK.B and BRK-B."""
    cleaned = symbol.strip().upper() if isinstance(symbol, str) else ""
    if not _SYMBOL.fullmatch(cleaned):
        raise DataError("请输入有效的单个美股代码，例如 NVDA 或 BRK.B。", "invalid_symbol")
    return cleaned


def _local_now(now: datetime | None) -> datetime:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=NEW_YORK)
    return value.astimezone(NEW_YORK)


def _provider_error(payload: object) -> DataError | None:
    """Classify known FMP error envelopes without reflecting their contents."""
    if not isinstance(payload, dict):
        return None
    messages = [str(value).lower() for key, value in payload.items()
                if str(key).lower() in _ERROR_KEYS and value]
    if not messages:
        return None
    message = " ".join(messages)
    if any(word in message for word in ("limit", "rate", "too many")):
        return DataError("FMP 请求额度或速率限制已触发，请稍后重试。", "rate_limited")
    if any(word in message for word in ("apikey", "api key", "invalid key", "unauthorized")):
        return DataError("FMP 身份验证失败，请检查本地 API 配置。", "unauthorized")
    if any(word in message for word in ("subscription", "premium", "restricted", "permission", "upgrade", "payment")):
        return DataError("当前 FMP 账户无权访问此数据。", "permission_denied")
    return DataError("FMP 返回数据错误，暂时无法获取此行情。", "provider_error")


def _check_source_symbol(source: object, requested: str) -> None:
    """Reject explicitly mislabeled data; class-share separators are aliases."""
    if source is None or (isinstance(source, str) and not source.strip()):
        return
    if (not isinstance(source, str) or
            source.strip().upper().replace(".", "-") != requested.replace(".", "-")):
        raise DataError("FMP 返回的行情代码与请求不一致，已停止该数据分析。", "symbol_mismatch")


class FMPClient:
    """Keep credentials in the adapter, never in cache keys or result metadata."""

    def __init__(self, settings: Settings, session: requests.Session | None = None):
        parsed = urlsplit(settings.base_url)
        if (parsed.scheme != "https" or not parsed.hostname or parsed.username or
                parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}
                or settings.auth_mode not in {"query", "proxy"}):
            raise DataError("FMP 连接配置无效，请检查本地配置。", "invalid_configuration")
        self._settings = settings
        self._session = session or requests.Session()
        # Ignore ambient proxy/netrc credentials. Custom gateways use explicit auth.
        self._session.trust_env = False
        self._injected_session = session is not None
        self._sessions = local()
        self._sessions.value = self._session
        self._cache: OrderedDict[tuple[str, str, str, str], tuple[float, RawDataset]] = OrderedDict()
        self._lock = RLock()
        self._raw_inflight: dict[tuple, Future] = {}
        self._validated_inflight: dict[tuple, Future] = {}
        self._generations: dict[tuple[str, str], int] = {}
        self._errors: OrderedDict[tuple, tuple[float, str, str]] = OrderedDict()
        self._market_store = MarketStore()

    def _http_session(self):
        if self._injected_session:
            return self._session
        if not hasattr(self._sessions, "value"):
            self._sessions.value = requests.Session()
            self._sessions.value.trust_env = False
        return self._sessions.value

    def fetch_daily(self, symbol: str, now: datetime | None = None, refresh: bool = False) -> RawDataset:
        return self._fetch("historical-price-eod/full", symbol,
                           self._settings.daily_history_days, now, refresh)

    def fetch_intraday(self, symbol: str, now: datetime | None = None, refresh: bool = False) -> RawDataset:
        return self._fetch("historical-chart/5min", symbol,
                           self._settings.intraday_history_days, now, refresh)

    def fetch_validated_daily(self, symbol: str, now: datetime | None = None,
                              refresh: bool = False) -> MarketDataset:
        return self._fetch_validated("daily", symbol, now, refresh)

    def fetch_validated_intraday(self, symbol: str, now: datetime | None = None,
                                 refresh: bool = False) -> MarketDataset:
        return self._fetch_validated("5min", symbol, now, refresh)

    def peek_validated_daily(self, symbol: str, now: datetime | None = None) -> MarketDataset | None:
        return self._peek_validated("daily", symbol, now)

    def peek_validated_intraday(self, symbol: str, now: datetime | None = None) -> MarketDataset | None:
        return self._peek_validated("5min", symbol, now)

    def _peek_validated(self, timeframe: str, symbol: str, now: datetime | None) -> MarketDataset | None:
        symbol, local_now = validate_symbol(symbol), _local_now(now)
        return self._market_store.get(symbol, timeframe, ADJUSTMENT_POLICY,
                                      freshness_epoch(timeframe, local_now), local_now)

    def _invalidate(self, endpoint: str, symbol: str) -> None:
        timeframe = "daily" if endpoint == "historical-price-eod/full" else "5min"
        with self._lock:
            pair = (endpoint, symbol)
            self._generations[pair] = self._generations.get(pair, 0) + 1
            for key in [key for key in self._cache if key[:2] == pair]:
                del self._cache[key]
            for key in [key for key in self._errors if key[:2] == pair]:
                del self._errors[key]
            self._market_store.invalidate(symbol, timeframe)

    def _fetch_validated(self, timeframe: str, symbol: str, now: datetime | None,
                         refresh: bool) -> MarketDataset:
        symbol, local_now = validate_symbol(symbol), _local_now(now)
        endpoint = "historical-price-eod/full" if timeframe == "daily" else "historical-chart/5min"
        epoch = freshness_epoch(timeframe, local_now)
        if refresh:
            self._invalidate(endpoint, symbol)
        with self._lock:
            cached = self._market_store.get(symbol, timeframe, ADJUSTMENT_POLICY, epoch, local_now)
            if cached is not None:
                return cached
            generation = self._generations.get((endpoint, symbol), 0)
            flight_key = (endpoint, symbol, str(epoch), generation)
            future = self._validated_inflight.get(flight_key)
            owner = future is None
            if owner:
                future = self._validated_inflight[flight_key] = Future()
        if not owner:
            try:
                result = future.result()
            except DataError as error:
                raise DataError(str(error), error.code, cache_hit=True) from None
            return deepcopy(replace(result, cache_hit=True, api_seconds=0.0,
                                    validation_seconds=0.0))
        try:
            fetcher = self.fetch_daily if timeframe == "daily" else self.fetch_intraday
            raw = fetcher(symbol, now=now, refresh=False)
            # Cache age can never promote a previously forming candle to confirmed.
            normalized_at = min(local_now, raw.retrieved_at)
            started = time.perf_counter()
            normalizer = normalize_daily if timeframe == "daily" else normalize_intraday
            bars = normalizer(raw.records, normalized_at)
            version = content_version(bars.frame, timeframe)
            validation_seconds = time.perf_counter() - started
            latest = None if bars.frame.empty else bars.frame.iloc[-1]["session"]
            result = MarketDataset(bars.frame, bars.issues, raw.retrieved_at, endpoint,
                                   latest, version, raw.cache_hit, raw.api_seconds,
                                   validation_seconds)
            completed = bars.frame.loc[bars.frame["complete"]]
            latest_end = None if completed.empty else completed.iloc[-1]["bar_end"]
            ttl = min(self._settings.cache_ttl_seconds, 300)
            # A scheduled candle may arrive late. Retain its actual observed date
            # and retry soon, without inventing data or issuing immediate retries.
            if epoch is not None and (latest_end is None or latest_end < epoch):
                ttl = min(ttl, 30)
            with self._lock:
                if generation == self._generations.get((endpoint, symbol), 0):
                    self._market_store.put(symbol, timeframe, ADJUSTMENT_POLICY, result,
                                           epoch, normalized_at, ttl)
                    raw_key = (endpoint, symbol, ADJUSTMENT_POLICY, str(epoch))
                    if raw_key in self._cache:
                        expiry, cached_raw = self._cache[raw_key]
                        self._cache[raw_key] = (min(expiry, time.monotonic() + ttl), cached_raw)
            future.set_result(result)
            return deepcopy(result)
        except BaseException as error:
            future.set_exception(error)
            raise
        finally:
            with self._lock:
                self._validated_inflight.pop(flight_key, None)

    def _fetch(self, endpoint: str, symbol: str, days: int,
               now: datetime | None, refresh: bool) -> RawDataset:
        symbol = validate_symbol(symbol)
        if not self._settings.api_key:
            raise DataError("尚未配置 FMP_API_KEY，请设置环境变量或本地 .env 文件。", "missing_key")
        local_now = _local_now(now)
        end = local_now.date().isoformat()
        start = (local_now.date() - timedelta(days=days)).isoformat()
        timeframe = "daily" if endpoint == "historical-price-eod/full" else "5min"
        epoch = freshness_epoch(timeframe, local_now)
        cache_key = (endpoint, symbol, ADJUSTMENT_POLICY, str(epoch))
        if refresh:
            self._invalidate(endpoint, symbol)
        with self._lock:
            instant = time.monotonic()
            # Expire all old entries, so a client never accumulates ticker history.
            expired = [key for key, (until, _) in self._cache.items() if until <= instant]
            for key in expired:
                del self._cache[key]
            if cache_key in self._cache:
                self._cache.move_to_end(cache_key)
                return deepcopy(replace(self._cache[cache_key][1], api_seconds=0.0, cache_hit=True))
            generation = self._generations.get((endpoint, symbol), 0)
            flight_key = (*cache_key, generation)
            for key in [key for key, entry in self._errors.items() if entry[0] <= instant]:
                del self._errors[key]
            if flight_key in self._errors:
                _, message, code = self._errors[flight_key]
                self._errors.move_to_end(flight_key)
                raise DataError(message, code, cache_hit=True) from None
            future = self._raw_inflight.get(flight_key)
            owner = future is None
            if owner:
                future = self._raw_inflight[flight_key] = Future()
        if not owner:
            try:
                result = future.result()
            except DataError as error:
                raise DataError(str(error), error.code, cache_hit=True) from None
            return deepcopy(replace(result, api_seconds=0.0, cache_hit=True))
        try:
            params = {"symbol": symbol, "from": start, "to": end}
            headers = {}
            if endpoint == "historical-chart/5min":
                params["extended"] = "false"
            if self._settings.auth_mode == "proxy":
                headers["X-Proxy-Key"] = self._settings.api_key
            else:
                params["apikey"] = self._settings.api_key
            started = time.perf_counter()
            try:
                response = self._http_session().get(
                    self._settings.base_url.rstrip("/") + "/stable/" + endpoint,
                    params=params, headers=headers,
                    timeout=self._settings.request_timeout_seconds,
                    allow_redirects=False, verify=self._settings.ca_bundle or True,
                )
            except requests.exceptions.SSLError:
                raise DataError("FMP 连接的 TLS 证书验证失败，请检查主机和受信证书配置。", "tls_failed") from None
            except Exception:
                raise DataError("FMP 网络请求失败，请稍后重试。", "network_error") from None

            status = response.status_code
            if status == 401:
                raise DataError("FMP 身份验证失败，请检查本地 API 配置。", "unauthorized")
            if status in {402, 403}:
                raise DataError("当前 FMP 账户无权访问此数据。", "permission_denied")
            if status == 429:
                raise DataError("FMP 请求额度或速率限制已触发，请稍后重试。", "rate_limited")
            if not 200 <= status < 300:
                raise DataError("FMP 请求失败，暂时无法获取此行情。", "request_failed")
            try:
                payload = response.json()
            except Exception:
                raise DataError("FMP 返回的行情格式无效。", "malformed_response") from None
            api_seconds = time.perf_counter() - started
            error = _provider_error(payload)
            if error:
                raise error from None
            if isinstance(payload, dict):
                _check_source_symbol(payload.get("symbol"), symbol)
                # Stable responses are arrays; also accept the documented legacy wrapper.
                payload = payload.get("historical", payload.get("data"))
            if not isinstance(payload, list) or any(not isinstance(row, dict) for row in payload):
                raise DataError("FMP 返回的行情格式无效。", "malformed_response")
            if not payload:
                raise DataError("FMP 未返回该代码的可用行情，请检查代码、日期范围或账户权限。", "no_data")
            for row in payload:
                error = _provider_error(row)
                if error:
                    raise error from None
                _check_source_symbol(row.get("symbol"), symbol)
            if not any("date" in row for row in payload):
                raise DataError("FMP 返回的行情格式无效。", "malformed_response")
            # Store only chart inputs, never arbitrary provider metadata or auth fields.
            records = [{name: row[name] for name in ("date", "open", "high", "low", "close", "volume")
                        if name in row} for row in payload]
            retrieved_at = local_now if now is not None else _local_now(None)
            result = RawDataset(records=records, retrieved_at=retrieved_at, endpoint=endpoint,
                                api_seconds=api_seconds, cache_hit=False)
            with self._lock:
                if generation == self._generations.get((endpoint, symbol), 0):
                    self._cache[cache_key] = (time.monotonic() + min(self._settings.cache_ttl_seconds, 300), result)
                    self._cache.move_to_end(cache_key)
                    while len(self._cache) > _MAX_CACHE_ENTRIES:
                        self._cache.popitem(last=False)
            future.set_result(result)
            return deepcopy(result)
        except BaseException as error:
            if isinstance(error, DataError):
                error.api_seconds = time.perf_counter() - started
                # Store only our sanitized display text, never provider objects,
                # request arguments, underlying exceptions or exception traces.
                ttl = min(self._settings.cache_ttl_seconds,
                          300 if error.code == "permission_denied" else 30)
                with self._lock:
                    if generation == self._generations.get((endpoint, symbol), 0):
                        self._errors[flight_key] = (time.monotonic() + ttl, str(error), error.code)
                        self._errors.move_to_end(flight_key)
                        while len(self._errors) > _MAX_CACHE_ENTRIES:
                            self._errors.popitem(last=False)
            future.set_exception(error)
            raise
        finally:
            with self._lock:
                self._raw_inflight.pop(flight_key, None)
