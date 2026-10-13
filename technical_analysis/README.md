# Stock Technical Analysis Dashboard MVP

A standalone, single-process Streamlit dashboard for US equity price, volume, and technical indicators. The UI and deterministic analysis summaries use Simplified Chinese; source code and documentation use English. This project does not import the Earnings Research project, access its database, or share its workflows. There is no database, task queue, LLM, or agent orchestration in the application.

## Start on Windows

Run these commands in PowerShell from this repository's root:

```powershell
Set-Location .\technical_analysis
py -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements-dev.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

Replace the placeholder in this project's `.env` with your FMP API key:

```dotenv
FMP_API_KEY=your_fmp_api_key
```

Then start the dashboard:

```powershell
.\.venv\Scripts\python -m streamlit run app.py
```

Open the local address printed by Streamlit, enter a ticker such as `NVDA`, and click **分析**. Python 3.11 or newer is recommended. Activation of the virtual environment is optional because the commands use its Python executable directly.

`FMP_API_KEY` in the process environment takes precedence over `.env`. Configuration loads only the `.env` next to `config.py`; it does not search sibling projects. The real `.env`, virtual environment, local artifacts, and Streamlit secrets are ignored by Git. Do not paste keys into source files, screenshots, logs, or issues. The adapter exposes fixed error messages instead of provider response bodies or credential-bearing request URLs.

The default host is the official FMP API and uses its query-parameter authentication. If your existing FMP access uses a gateway, configure the matching `FMP_BASE_URL` and `FMP_AUTH_MODE=proxy`; proxy mode sends the credential in `X-Proxy-Key`. Keep a gateway credential paired with its intended host. `FMP_CA_BUNDLE` can specify a trusted CA bundle for that gateway. HTTPS certificate verification stays enabled; the app does not disable verification to make a request succeed.

## What version 1 implements

- Latest available bar price, percentage change when a valid previous-session comparison exists, actual trading date, market-data time, and separate retrieval times.
- Daily candlesticks and volume with SMA20 and SMA50; a separate RSI14 panel; 3M, 6M, and 1Y display ranges.
- The most recent available regular-hours session of 5-minute candlesticks, volume, and estimated session VWAP.
- A prominent technical assessment with separate daily strength, RSI momentum, intraday bias, conditional action guidance, supporting/conflicting evidence, trigger status, and invalidation conditions.
- Multi-method support/resistance zones from confirmed swing clusters, Estimated Volume Profile, and daily-bar Anchored VWAP, with ranked bands, retrospective interactions, and score breakdowns.
- A holding-status control selecting entry guidance (Buy/Wait) or position-management guidance (Hold/Reduce or Sell) without changing the underlying market assessment.
- Rule-based Chinese summaries with values and rule identifiers covering completed-day trend, RSI momentum, intraday observations, and signal disagreement.
- Validation, partial-data warnings, validated OHLCV and analysis caches, automatic ticker switching, and manual refresh.

The daily view remains usable when the account lacks intraday permission or the intraday response is empty or invalid. No synthetic market data is substituted. Price comes from the available OHLCV bars, so the application does not require quote-endpoint access. The dashboard provides conditional technical Buy/Wait or Hold/Reduce or Sell guidance for a defined timeframe. It uses no portfolio, cost-basis, position-size, or investment-suitability information and does not produce targets, stops, or success probabilities.

## Technical assessment

The daily/swing assessment uses completed daily bars for a **5-20 trading-day** interpretation horizon. It reports Strong Bullish / Bullish / Neutral / Bearish / Strong Bearish separately from entry attractiveness. Intraday Bullish / Neutral / Bearish / Unavailable uses two consecutive completed 5-minute bars versus their own estimated session VWAP and the session open. Unfinished-bar observations are labeled provisional and do not establish confirmed signals.

For a new entry, Buy requires a strict rising `close > SMA20 > SMA50` trend, RSI14 in `[50, 70)`, no more than 3% daily SMA20 extension, a verified completed-bar SMA20 reclaim or rising continuation, and fresh Bullish confirmation during the current regular session. Otherwise the result is Wait. For a current holding, Bearish/Strong Bearish daily bias or two consecutive completed closes below their own SMA50 produces Reduce or Sell; otherwise the daily assessment is Hold. Intraday disagreement and RSI at/above 70 do not independently create a holding exit. Insufficient required daily history or a missing latest expected completed daily session withholds action guidance.

The UI shows the actual indicator values, timestamps, occurred/pending triggers, conflicts, and conditions for reassessment. See [the exact rule definitions](docs/technical-assessment-rules.md) for thresholds, boundaries, lookbacks, freshness, and precedence. These initial heuristics have not been historically validated or backtested; no profitability or prediction-accuracy claim is made.

## Support and resistance

The daily chart automatically includes support/resistance bands and confirmed
Anchored VWAP curves. **支撑/阻力设置** selects a 12-month (252-session) or
6-month (126-session) analysis window, a maximum of 2/3/5 zones per side, and
show/hide controls for zones, estimated volume profile, and AVWAP. All methods
use existing cached daily bars; these controls make no additional FMP requests.

The panel shows actual boundaries, distance from the labeled snapshot, relative
strength, methods, retrospective interactions, last test, validity, profile
statistics and anchor dates. Intersecting zones are separate; broken zones remain
diagnostic. Touch counts are descriptive, while breakout/retest state respects
geometry availability and completed bars. Scores are not calibrated probabilities.

See [detection, scoring, defaults and limitations](docs/support-resistance.md).
Daily volume profile and AVWAP are approximations; cross-split volume adjustment
depends on the vendor. No historical profitability or prediction accuracy is claimed.

## Data and indicator conventions

Daily history is requested over a fixed **730-calendar-day** window independent of the selected chart range. Indicators are calculated before slicing the visible range, so switching 3M/6M/1Y does not change a value on the same date within one analysis result. A future refresh can change provider data or the historical window. A short listing or limited API response can still leave insufficient warm-up; those indicators remain unavailable.

SMA20 and SMA50 require 20 and 50 closing prices. RSI14 uses Wilder smoothing: the first average gain and loss are arithmetic means of the first 14 close-to-close changes; subsequent averages use `(previous_average * 13 + current_value) / 14`. RSI is 100 when average loss alone is zero, 0 when average gain alone is zero, and 50 when both are zero. The initial 14 changes are required. RSI at or above 70 or at or below 30 describes momentum conditions and does not promise a reversal.

Estimated VWAP uses each 5-minute bar's typical price, `(high + low + close) / 3`, weighted by volume. Its cumulative numerator and denominator reset for each New York trading session. It is a bar-based estimate, not trade-by-trade VWAP. The app requires the 09:30 opening bar and every elapsed completed interval as of the response retrieval time (up to scheduled close); otherwise it withholds that session's VWAP, session-open comparison, and cumulative volume with a coverage warning. Zero cumulative volume cannot produce a VWAP. Forming bars are distinguished; confirmed intraday summaries use only completed bars and their cumulative estimates. Cached unfinished bars remain provisional until a new response confirms completion.

Records are sorted chronologically and checked for duplicate timestamps, required fields, finite positive prices, nonnegative volume, and OHLC consistency. Invalid rows are discarded with warnings. Gaps are not filled with invented candles or volumes. A cumulative volume value represents the available regular-hours bars; the app does not call it unusually high or low without a comparable historical same-time baseline.

US sessions use `America/New_York` and the NYSE calendar through `pandas-market-calendars`, including holidays, early closes, and daylight-saving transitions. Premarket and after-hours bars are excluded. Weekends, holidays, and sessions without available data display the latest actual session date. Confirmed daily statements use the latest completed daily session; observations from the unfinished session are labeled provisional.

FMP daily records have a session date rather than an exchange event timestamp. The displayed daily market timestamp is **derived from that session's scheduled close**, not claimed to be a provider tick time. A 5-minute timestamp is treated as the interval start; completion requires the full interval to have elapsed within the regular session. This interval-start convention is an application assumption: the public FAQ does not fully specify interval labeling. The documented geographic exchange timezone wording is interpreted as New York local time, including DST, rather than a fixed UTC-5 offset. If FMP changes its convention, update `data/normalize.py` and its boundary tests.

FMP documents daily `close` as split-adjusted, while `adjClose` additionally adjusts dividends; intraday prices are also split-adjusted. The application uses daily OHLC/`close`, not `adjClose`, for a consistent price basis across timeframes. FMP describes endpoint timezones by exchange location and limits intraday records per request. These are provider conventions, not a guarantee that every response is complete or timely. See the [FMP FAQ](https://site.financialmodelingprep.com/faqs).

Percentage changes and previous-close statements require the exact previous scheduled trading session in the returned daily data; an older available row is not silently substituted. Intraday behavior is never inferred from a daily bar. A delayed or missing session can therefore leave a comparison unavailable.

## Refresh and permissions

Committing a changed ticker (Enter or leaving the field) automatically analyzes all three support/resistance methods. The Analyze button remains available. Each browser session publishes only its latest request generation; queued stale work is replaced, and completed stale requests cannot overwrite the selected ticker. A lightweight loading caption appears while work runs. Chart-range, holding and support/resistance display controls reuse retrieved data.

Validated OHLCV caches are bounded and scoped to one provider client, keyed by ticker, timeframe, split-adjustment policy and actual observed latest session. Entries have a maximum 300-second TTL. NYSE completed-close epochs invalidate daily data at regular or early closes; intraday entries also invalidate at completed five-minute boundaries. A lightweight 30-second check refreshes an open dashboard when a new completed daily session is missing. Delayed new candles retry after a short cache TTL without fabricating a session. Manual refresh bypasses caches. Stable content fingerprints reuse derived calculations until data or completion flags change. A cache does not establish real-time freshness; check market and retrieval timestamps.

Independent daily and intraday retrieval run concurrently. The support/resistance loops are optimized without changing detection or scoring rules. See [performance behavior and measurements](docs/performance.md) for API, validation, computation and server UI timing boundaries, reproducible benchmarks and limitations. The 500 ms cached and 3 s uncached goals are targets, not guarantees.

The endpoints `/stable/historical-price-eod/full` and `/stable/historical-chart/5min` are requested separately with explicit date ranges. Intraday retrieval covers 35 calendar days and requests `extended=false`, with local regular-session filtering as a second check. Actual responses and account permissions determine availability. A missing key stops API requests. Authentication, entitlement, rate-limit, network, malformed-response, and empty-data failures appear as sanitized messages. If daily data fails, the analysis cannot proceed; if only intraday fails, daily charts and summaries remain available with an explicit intraday-unavailable notice.

## Verification

The performance and ticker-switching update passed **222 offline checks**, covering cached reuse, concurrent loads, revisions, session rollover, rapid-switch stale success/error suppression and automatic daily-close refresh, alongside the existing indicator, assessment and support/resistance checks. The real NVDA check on October 12, 2026 used 252 completed daily sessions and returned 3 support zones, 1 resistance zone, an estimated volume profile and 2 confirmed AVWAP anchors. Live NVDA/AAPL API and cached-switch timings are documented separately. These checks verify application behavior, not trading performance.

From the `technical_analysis` directory:

```powershell
.\.venv\Scripts\python -m pytest -q
.\.venv\Scripts\python scripts\check_live.py
```

The focused offline checks use clearly separated synthetic fixtures to verify indicator arithmetic, Wilder initialization, VWAP session resets, data validation, session boundaries, unfinished-bar behavior, and deterministic assessment rules. Assessment cases include bullish, bearish, conflicting, overbought-in-an-uptrend, insufficient/stale data, and holding-status guidance. Fixtures are test inputs only; the production UI has no demo-data fallback. These correctness checks are not a backtest.

`check_live.py` performs a real **NVDA** analysis using this project's configuration. It prints sanitized availability, dates, row counts, and retrieval times. Exit code 0 means usable daily data, including a partial result when intraday is unavailable; 1 means daily analysis failed; 2 means no API key is configured. No API key, request URL, raw response body, or raw exception is printed. Live availability and latency must be checked with the actual local account; passing offline tests does not verify FMP subscription permissions.

For an unexpected implementation failure, `python scripts/check_live.py --diagnostics` additionally prints the exception class and stack locations (filename, line number, function). It omits exception messages, source text, local variables, provider bodies, and request URLs. Unknown command arguments are not echoed.

The real-ticker check on **2026-10-12** succeeded through the locally configured FMP gateway: NVDA returned 500 valid daily bars and 78 regular-hours 5-minute bars, with both latest sessions dated 2026-10-12. This verifies those endpoints for the configured account at that time; it does not establish real-time latency or future subscription permissions.

## Modules and the next increment

| Module | Responsibility |
| --- | --- |
| `app.py` | Streamlit presentation and Plotly charts consuming structured results |
| `config.py` | Project-local settings and credentials |
| `data/fmp_client.py` | FMP requests, caching, and sanitized failures |
| `data/market_store.py` | Bounded validated OHLCV snapshots, content versions and exchange-calendar freshness |
| `data/normalize.py` | Validation, timestamps, calendar sessions, and bar completion |
| `analysis/indicators.py` | SMA, Wilder RSI, and session VWAP calculations |
| `analysis/signals.py` | Structured market assessment and conditional entry/holding rules |
| `analysis/support_resistance/` | Daily preparation, filtered swings, estimated volume profile, AVWAP, consolidation, validity and scoring |
| `ui/support_resistance.py` | Zone evidence, profile statistics and score presentation |
| `analysis/summary.py` | Supplementary Chinese indicator observations and evidence |
| `services/analysis_service.py` | Coordinates one ticker and returns structured analysis |
| `services/ticker_requests.py` | Per-session bounded background requests and latest-generation publishing |
| `docs/technical-assessment-rules.md` | Exact assessment thresholds, lookbacks, and precedence |
| `docs/support-resistance.md` | Support/resistance methods, parameters, scoring and limitations |
| `scripts/check_live.py` | Explicit real-ticker integration check |
| `scripts/benchmark_performance.py` | Separate API/validation/computation timings and cached-switch benchmark |
| `scripts/benchmark_ui.py` | Offline full-dashboard server UI generation benchmark |
| `tests/` | Focused offline correctness checks |

Add future calculations to `analysis/indicators.py` and assessment rules to `analysis/signals.py`, document them, and expose structured output through the service before adding a UI panel. Keep trading-rule logic out of the UI. A future historical-validation module can consume normalized bars without changing retrieval or chart code; historical validation is outside this update.
