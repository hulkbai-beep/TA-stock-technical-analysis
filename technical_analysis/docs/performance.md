# Ticker switching and performance

The performance goals are under 500 ms for cached analysis and under 3 seconds
for a new ticker, subject to provider latency and local rendering cost. They are
targets, not guarantees. Measurements below separate service analysis from UI
work and do not claim browser paint timings.

## Data and calculation reuse

- The adapter caches validated OHLCV by ticker, timeframe (`daily` or `5min`),
  adjustment policy (`split_adjusted`), and the latest **observed** trading
  session. Each provider client owns its caches; credentials are never keys.
- A separate NYSE calendar epoch determines freshness: the latest scheduled
  completed daily close or regular-session five-minute bar end. A new epoch
  makes old data ineligible. Holidays and early closes use the same calendar.
- TTL expiry and manual refresh also cause a reload. If the provider has not
  delivered the expected candle yet, retain its actual observed session and
  retry after a short expiry; never invent a candle or change its completion
  state just because cached data has aged.
- Identical requests already in progress share their work. Daily and intraday
  data paths run concurrently. The small CPU calculations remain sequential;
  array/tuple access removes unnecessary pandas row/window allocation.
- Derived results are scoped to a client, ticker, adjustment policy, observed
  sessions, validated content fingerprints, and data-quality issues. Changing
  OHLCV or completion flags invalidates derived work. Unchanged content reuses
  daily indicators, session VWAP, and all three support/resistance methods.
  Assessment freshness is reevaluated using the current clock.
- Changing support/resistance controls reuses the same daily data. Repeated
  configurations reuse a bounded set of calculated variants. Existing chart
  layers, holding status, and layout retain their behavior.
- Background loads carry monotonically increasing request IDs. Only the
  currently selected ticker/request may publish a result or error. Queued work
  is canceled or replaced when possible. An HTTP request already running may
  finish and populate its cache, but its stale result cannot replace the UI.

The caches are bounded and process-local; restarting the application clears
them. Cached validation does not infer split ratios or change vendor volume.

## Timing boundaries

`AnalysisResult.timings` exposes these measured durations in milliseconds:

| Field | Work included |
|---|---|
| `daily_api_ms`, `intraday_api_ms` | Endpoint HTTP request through JSON decoding, including client/network latency; zero for a cache hit |
| `data_wall_ms` | Elapsed time waiting for both market-data paths |
| `validation_ms` | Sum of uncached normalization and content-fingerprint work |
| `compute_ms` | Indicators, summaries, assessment and support/resistance calculations, or assessment freshness checks on a derived cache hit |
| `support_resistance_ms` | Support/resistance portion of `compute_ms`; zero when reused |
| `total_ms` | Full service analysis; excludes UI and browser rendering |

Concurrent endpoint and validation durations overlap. They must not be added
to `data_wall_ms` or summed to reconstruct total wall time. Cold calendar
initialization can contribute to first-visit latency.

The dashboard separately records server-side chart construction, serialization,
and Streamlit UI emission. This is **not browser paint**. Browser layout, Plotly
paint, transport, and the user's visible completion time are not measured by the
CLI benchmark. UI polling can also contribute to visible switching latency.
The loading fragment polls every 100 ms while a request is pending. A separate
30-second clock/cache check refreshes an open dashboard when a newly completed
daily session is missing; delayed provider candles follow the short retry TTL.

## Reproduce measurements

From the project directory:

```powershell
# No network: isolated synthetic inputs, 500 daily bars and 78 intraday bars.
.\.venv\Scripts\python scripts/benchmark_performance.py --repeats 10 --output artifacts/performance-offline.json

# Genuine FMP requests, using the existing local settings.
.\.venv\Scripts\python scripts/benchmark_performance.py --live --symbols NVDA AAPL --repeats 10 --output artifacts/performance-live.json

# No network: measure the actual dashboard with preloaded benchmark fixtures.
.\.venv\Scripts\python scripts/benchmark_ui.py
```

Each ticker's first visit explicitly bypasses caches. Ten subsequent visits per
ticker alternate through the same client. The JSON includes each run, cache-hit
flags, stage definitions, and phase median/p95 values. Failed first visits are
not retried by the cached-switch loop. Logs exclude credentials, configuration
representations, provider payloads, URLs, and raw exception messages. Offline API
durations describe local fixture transport, not FMP latency.

## Observed results

Measured on this workstation on **October 12, 2026 (America/New_York)** through
the locally configured FMP connection. Both real tickers returned 500 daily and
78 intraday bars. These two cold samples are observations, not a latency SLA.

| First visit | Daily API | Intraday API | Validation sum | Computation | Service total |
|---|---:|---:|---:|---:|---:|
| NVDA | 325.6 ms | 402.6 ms | 968.5 ms | 389.6 ms | 1,459.1 ms |
| AAPL | 202.1 ms | 32.8 ms | 309.1 ms | 101.2 ms | 452.5 ms |

Across 20 genuine cached ticker switches, all daily, intraday and derived caches
hit. API and validation time were zero. Computation median/p95 was
**3.85 / 4.21 ms**; service total median/p95 was **5.55 / 5.88 ms**. Browser
rendering is excluded. See `artifacts/performance-live.json` for the samples.

A separate isolated 500-bar, 252-session support/resistance benchmark (15 warm
runs, no API or UI work) improved from **139.63 / 178.97 ms** median/p95 to
**50.49 / 59.05 ms**. The whole result matched the pre-optimization snapshot
exactly, and the existing 81 support/resistance tests passed.

Ten warm full-dashboard server UI runs with preloaded, isolated 500-daily/
78-intraday fixtures measured **148.6 / 186.4 ms** median/p95. This includes
chart construction, serialization and Streamlit emission. The enclosing
AppTest script/harness runs measured **168.0 / 208.5 ms**. Two initial runs were
excluded as warmup. These are rendering-side server measurements, not a browser
paint or end-to-end switching claim; API and service analysis are excluded.
Samples are in `artifacts/performance-ui.json`.

The final offline suite passed **222 checks**, including rapid-switch stale
success/error handling, explicit refresh and data revisions, regular and early
close invalidation, delayed-candle retry cooldowns, and consuming a new daily
candle already cached by another session without making HTTP requests.
