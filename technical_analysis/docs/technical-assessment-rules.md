# Technical assessment rules

These are initial, deterministic heuristics for interpreting the existing indicators. They have not been historically validated, backtested, or calibrated for profitability, prediction accuracy, or a success probability. The assessment describes one stock over a defined timeframe and uses no account, position size, purchase price, portfolio, or suitability information.

`analysis/signals.py` owns the rules and structured results. The analysis service supplies calculated, normalized bars; the UI renders the results and selects the guidance for the holding-status control. Changing that control changes the action guidance, not the daily bias, momentum, intraday bias, or supporting market values.

## Timeframes, data, and reference values

- Daily bias and position management use completed daily bars, for an intended **5-20 trading-day** daily/swing timeframe. This is an interpretation horizon, not a promised holding period or prediction.
- Entry timing uses completed **5-minute regular-session** bars. Intraday evidence is evaluated separately and cannot change the daily bias or, by itself, cause a holding to be sold.
- Sessions follow the NYSE calendar in `America/New_York`, including holidays, early closes, and daylight-saving time. The latest expected completed daily session is the latest scheduled session whose close has elapsed at assessment time. Before today's scheduled close, it is the previous scheduled session.
- A confirmed daily assessment needs the latest two completed bars to be consecutive scheduled trading sessions, with finite positive closes/SMA20/SMA50 and finite RSI14 on both. Thus SMA50 and its one-session slope require at least **51 consecutive valid session closes**. Daily indicator calculations insert missing-session NaNs for calculation only, so a history gap restarts warm-up rather than bridging missing prices. RSI14 must lie in `[0, 100]`.
- SMA20/SMA50 are arithmetic means of 20/50 session closes. RSI14 uses the existing Wilder calculation: 14 close-to-close changes initialize arithmetic average gain/loss, then each average is updated as `(previous * 13 + current) / 14`. Loss-only, gain-only, and flat states return 0, 100, and 50 respectively.
- Estimated session VWAP is `sum(((high + low + close) / 3) * volume) / sum(volume)`, reset at 09:30 New York. Opening coverage, every elapsed completed 5-minute interval, and positive cumulative volume are required. It is an estimate from bars, not trade-by-trade VWAP.
- Unfinished daily and 5-minute bars may be shown as **provisional** observations. They never satisfy a confirmed rule or trigger. Cached bars retain completion status from the earlier of retrieval time and analysis time; merely waiting does not confirm a cached unfinished bar.
- Prices, SMA values, RSI values, session open, VWAP, session dates, and bar end times are the actual calculated inputs. Daily timestamps derive from the scheduled session close; intraday times are interval boundaries, not last-trade timestamps. Retrieval times remain separate from market-data times.

Chart-range selection does not alter the full-history calculations. These action rules do not use the separate support/resistance engine or infer targets, stop-loss levels, or calibrated probabilities. An SMA or VWAP reference is a changing calculated level, not an execution order or guaranteed fill price.

## Daily trend and strength

Let `C`, `M20`, `M50`, and `R` denote the latest completed daily close, SMA20, SMA50, and RSI14. Let `P`, `P20`, and `P50` denote the previous consecutive completed session's close and moving averages. One-session slopes are `d20 = M20 - P20` and `d50 = M50 - P50`; these are price-unit differences, not fitted or annualized slopes.

Evaluate the following in order:

| Priority | Daily technical bias | Exact condition |
| --- | --- | --- |
| 1 | Insufficient data | Required two-bar inputs or session continuity are missing/invalid. |
| 2 | Strong Bullish | `C > M20 > M50`, `d20 > 0`, `d50 > 0`, and `R >= 55`. |
| 3 | Bullish | The same strict uptrend with `R < 55`, or `C > M50`, `M20 > M50`, and `d50 >= 0`. |
| 4 | Strong Bearish | `C < M20 < M50`, `d20 < 0`, `d50 < 0`, and `R <= 45`. |
| 5 | Bearish | The same strict downtrend with `R > 45`, or `C < M50`, `M20 < M50`, and `d50 <= 0`. |
| 6 | Neutral | All required data exists, but none of the preceding directional conditions is met. |

The strict uptrend/downtrend tests require both moving-average slopes to agree. The broader Bullish/Bearish tests preserve the longer-term ordering and SMA50 direction but do not establish the strict entry setup. Equalities do not satisfy strict price ordering. A rising SMA20 alone, price above VWAP alone, or an extreme RSI alone does not establish a directional daily assessment.

### Momentum is separate

| RSI14 range | Momentum description |
| --- | --- |
| `R >= 70` | Overbought |
| `55 <= R < 70` | Bullish momentum |
| `45 <= R < 55` | Neutral momentum |
| `30 < R < 45` | Bearish momentum |
| `R <= 30` | Oversold |

An overbought strict uptrend can remain **Strong Bullish**, while new-entry guidance is **Wait**. RSI at or below 30 never creates a Buy by itself. RSI at or above 70 never creates a holding exit by itself.

Context: [Fidelity's RSI guide](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/RSI) explains that RSI can stay overbought or oversold during strong trends. That general indicator behavior does not validate this application's particular thresholds, combinations, action rules, or precedence; those are the initial heuristics defined here.

## Intraday bias and confirmation

Use the latest actual intraday session, with its date displayed explicitly. Its two latest completed 5-minute bars must be consecutive intervals in that session. Each must have finite positive close/VWAP, complete session coverage, and valid session-open data.

| Intraday technical bias | Exact condition |
| --- | --- |
| Unavailable | Two valid consecutive completed bars, valid session VWAP, or opening coverage are missing. Missing evidence is never counted as Neutral. |
| Bullish | Both completed closes exceed their own cumulative session VWAP, and the latest close exceeds the session open. |
| Bearish | Both completed closes are below their own cumulative session VWAP, and the latest close is below the session open. |
| Neutral | Required evidence exists, but neither directional condition is met. |

A historical session may still have a dated intraday bias. It cannot confirm an entry for the current session. Entry confirmation additionally requires an **active current regular-hours NYSE session**, matching intraday date, and a latest completed intraday bar ending no more than **10 minutes** before assessment time. Before open, after close, on holidays/weekends, with stale intraday bars, or with a prior-session intraday result, the entry trigger remains pending. At an intraday boundary, only fully elapsed intervals can qualify.

## Action precedence

First check daily data adequacy and freshness for both holding states. If the confirmed daily inputs are insufficient, or the latest completed daily session differs from the latest expected completed NYSE session, action guidance is **Insufficient data**. A dated historical daily bias may still be displayed, but stale evidence does not authorize a current action. Provisional daily observations cannot repair missing completed evidence.

### No current holding: Buy or Wait

**Buy** requires every condition below; otherwise the result is **Wait**, with the failed/pending conditions stated:

1. Daily inputs are sufficient and the latest completed daily session is the latest expected completed session.
2. A strict uptrend exists: `C > M20 > M50`, `d20 > 0`, and `d50 > 0`.
3. Momentum is in the entry range: `50 <= R < 70`. This intentionally differs from the strength threshold of 55.
4. Daily SMA20 extension is within range: `0 <= (C / M20 - 1) * 100 <= 3`.
5. One of two daily triggers has occurred on consecutive completed bars:
   - **SMA20 reclaim:** `P <= P20` and `C > M20`.
   - **Rising continuation:** `P > P20`, `C > M20`, and `C > P`.
6. The current NYSE regular session is active, its intraday bias is Bullish, the intraday session is today's scheduled session, and the latest completed 5-minute bar is no more than 10 minutes old.
7. That latest completed intraday close is above the confirmed daily SMA20 and no more than 3% above it: `0 < (intraday_close / M20 - 1) * 100 <= 3`.

The daily trigger and intraday confirmation are shown independently as occurred or pending. A daily uptrend with bearish, neutral, unavailable, stale, or prior-session intraday evidence therefore remains Wait for a new entry. A strong uptrend with excessive SMA20 extension or RSI14 at/above 70 also remains Wait; strength and entry attractiveness are separate outputs.

### Current holding: Hold or Reduce/Sell

After the same daily adequacy/freshness gate, evaluate:

1. **Reduce or Sell** if the daily bias is Bearish or Strong Bearish, **or** the two consecutive completed daily closes are both below their own SMA50: `P < P50` and `C < M50`.
2. Otherwise **Hold**. This is a completed-day position-management assessment, including a Neutral daily bias when no defined exit condition has occurred. It does not imply that adding to the holding has an attractive entry.

For holding management, RSI14 at/above 70, an unfinished daily decline, and bearish intraday evidence do not independently override the daily Hold. Conflicting intraday evidence remains visible as a risk/entry-timing condition. A new-entry Wait and a holding Hold can therefore coexist for the same market assessment.

## Triggers, invalidation, and conflicting evidence

The panel includes the intended timeframe, values/dates for both daily bars, one-session SMA slopes, RSI14, SMA20 extension, the two intraday bars/VWAPs and their ending times, session open, and relevant freshness status. It states the daily trigger, intraday confirmation, and holding exit condition separately, so a condition that has not occurred is not described as a confirmed crossover or reclaim.

- For a bullish daily setup, a subsequent completed daily close at/below the then-current SMA20 or loss of the strict uptrend invalidates that strict setup. RSI14 leaving `[50, 70)` or daily/intraday price exceeding the 3% SMA20-extension limit also invalidates the entry conditions. Loss of the broader SMA50/MA ordering undermines the directional bias. Recompute the moving averages from the new completed data; today's printed level is only the current reference.
- Intraday entry confirmation is invalidated by a completed intraday close at/below the confirmed daily SMA20, loss of the defined two-bar Bullish condition against each bar's session VWAP and the session open, or expiry of the freshness/current-session requirements. A single provisional bar cannot confirm or invalidate a completed-bar trigger.
- For holding exits, the actual bearish classification or verified two-bar SMA50 breakdown is identified as occurred; otherwise those exit conditions remain pending. A recovery requires evaluating new consecutive completed bars and their recalculated moving averages.
- The result explicitly reports contradictory momentum, overbought/oversold conditions, daily/intraday disagreement, excessive extension, missing confirmation, historical-session evidence, and provisional observations when applicable.

These conditions describe when to reassess. They do not supply invented stops, targets, support/resistance, guaranteed execution prices, or personalized trading suitability.

## Focused verification

`tests/test_signals.py` uses isolated synthetic fixtures to cover bullish, bearish, conflicting, overbought-uptrend, insufficient/stale daily, unavailable intraday, and provisional-bar cases. It also checks that holding status selects different guidance without changing the shared market assessment. Fixtures verify implementation against the rules; they measure neither historical trading performance nor prediction accuracy.
