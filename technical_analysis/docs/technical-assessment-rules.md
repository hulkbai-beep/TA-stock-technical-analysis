# Technical assessment rules

These are deterministic heuristics for interpreting the indicators. The thresholds and score weights are reasoned priors, not fitted parameters, and they have not yet been validated on real history: run `scripts/backtest.py` (see [Backtesting](#backtesting)) before relying on them. The assessment describes one stock over a defined timeframe and uses no account, position size, purchase price, portfolio, or suitability information.

`analysis/strategy.py` owns the bar-level rules (daily bias, entry gates, exits and the weighted score) through `evaluate_bar`, which both the dashboard and the backtest call, so a historical test runs exactly the rules the dashboard shows. `analysis/signals.py` adds session freshness, intraday context and the Chinese explanations. The analysis service supplies calculated, normalized bars; the UI renders the results and selects the guidance for the holding-status control. Changing that control changes the action guidance, not the daily bias, momentum, intraday bias, score, or supporting market values.

## Timeframes, data, and reference values

- Daily bias, entries and position management use completed daily bars, for an intended **5-20 trading-day** daily/swing timeframe. This is an interpretation horizon, not a promised holding period or prediction.
- Completed **5-minute regular-session** bars are context only. They cannot change the daily bias, create a Buy, or cause a holding to be sold. During an active session they can only hold back a Buy when price has already run beyond the extension limit (no chasing a gap).
- Sessions follow the NYSE calendar in `America/New_York`, including holidays, early closes, and daylight-saving time. The latest expected completed daily session is the latest scheduled session whose close has elapsed at assessment time. Before today's scheduled close, it is the previous scheduled session.
- A confirmed daily assessment needs the latest two completed bars to be consecutive scheduled trading sessions, with finite positive closes/SMA20/SMA50 and finite RSI14 on both. Thus SMA50 and its one-session slope require at least **51 consecutive valid session closes**. Daily indicator calculations insert missing-session NaNs for calculation only, so a history gap restarts warm-up rather than bridging missing prices. RSI14 must lie in `[0, 100]`.
- SMA20/SMA50/SMA200 are arithmetic means of 20/50/200 session closes. RSI14 uses the Wilder calculation: 14 close-to-close changes initialize arithmetic average gain/loss, then each average is updated as `(previous * 13 + current) / 14`. Loss-only, gain-only, and flat states return 0, 100, and 50 respectively.
- Estimated session VWAP is `sum(((high + low + close) / 3) * volume) / sum(volume)`, reset at 09:30 New York. Opening coverage, every elapsed completed 5-minute interval, and positive cumulative volume are required. It is an estimate from bars, not trade-by-trade VWAP.
- Unfinished daily and 5-minute bars may be shown as **provisional** observations. They never satisfy a confirmed rule or trigger. Cached bars retain completion status from the earlier of retrieval time and analysis time; merely waiting does not confirm a cached unfinished bar.
- Prices, SMA values, RSI values, session open, VWAP, session dates, and bar end times are the actual calculated inputs. Daily timestamps derive from the scheduled session close; intraday times are interval boundaries, not last-trade timestamps. Retrieval times remain separate from market-data times.

Chart-range selection does not alter the full-history calculations. An SMA, VWAP or chandelier reference is a changing calculated level, not an execution order or guaranteed fill price.

### Additional daily features

All are calculated over the calendar-reindexed history, so a missing session restarts warm-up instead of bridging the gap (`analysis/indicators.py`).

| Feature | Definition | Used for |
| --- | --- | --- |
| SMA200 | 200-session arithmetic mean of closes | long-term trend gate, trend score |
| ATR14 | Wilder ATR: first 14 true ranges averaged, then `(prev*13 + TR)/14` | extension limit, location score, chandelier stop |
| 22-session high | highest high of the last 22 sessions | chandelier stop |
| Prior highs | highest high of the 20/60/126/252 sessions before today (today excluded) | reward/risk score (and the optional gate), breakout trigger |
| 5/20 volume ratio | 5-session average volume ÷ 20-session average volume | breakout trigger |
| Pullback depth | lowest `(low - SMA20)/ATR14` of the last 5 sessions | pullback trigger |
| Up/down volume ratio | 20-session volume on up closes ÷ volume on down closes, capped at 5 | volume score |
| Relative volume | volume ÷ average of the previous 20 sessions | displayed in the data table |
| 63/126-session return | `close / close[n sessions ago] - 1` | relative strength |
| Fibonacci leg | highest high and lowest low of the last 126 sessions, and which came last | Fibonacci levels |
| SPY close, SMA50, SMA200, returns | from completed SPY bars, matched by session | market gate, regime and relative-strength scores |
| Sector ETF returns | from the stock's SPDR sector ETF (FMP profile sector) | relative-strength score |
| Next earnings date | first FMP earnings date on or after today | earnings gate |

SPY, the sector ETF, the company profile and the earnings calendar are fetched separately (`FMPClient.fetch_context`). Any of them can be unavailable for an account; the stock's own analysis still runs and the dependent conditions are shown as unverifiable.

Fibonacci retracements (23.6/38.2/50/61.8/78.6%) are measured on the 126-session leg: down from the high when the high is the more recent extreme, otherwise up from the low. Published evidence for Fibonacci levels is weak, so they are used only as confluence: a small bonus in the location score and an extra method on support/resistance zones. They never create a signal or a zone by themselves.

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

Context: [Fidelity's RSI guide](https://www.fidelity.com/learning-center/trading-investing/technical-analysis/technical-indicator-guide/RSI) explains that RSI can stay overbought or oversold during strong trends. That general indicator behavior does not validate this application's particular thresholds, combinations, action rules, or precedence.

## Intraday bias

Use the latest actual intraday session, with its date displayed explicitly. Its two latest completed 5-minute bars must be consecutive intervals in that session. Each must have finite positive close/VWAP, complete session coverage, and valid session-open data.

| Intraday technical bias | Exact condition |
| --- | --- |
| Unavailable | Two valid consecutive completed bars, valid session VWAP, or opening coverage are missing. Missing evidence is never counted as Neutral. |
| Bullish | Both completed closes exceed their own cumulative session VWAP, and the latest close exceeds the session open. |
| Bearish | Both completed closes are below their own cumulative session VWAP, and the latest close is below the session open. |
| Neutral | Required evidence exists, but neither directional condition is met. |

Earlier versions required a fresh Bullish intraday reading, no more than 10 minutes old, for every Buy. That tied a multi-day decision to minute-level noise and made the signal impossible to act on outside a short window, so it was removed. A Bearish intraday reading against a bullish daily setup is now reported as a timing conflict. During an active session with fresh intraday bars (latest completed bar no more than 10 minutes old), a latest completed 5-minute close more than 1.5 ATR above the daily SMA20 holds the entry at Wait.

## Weighted technical score

`technical_score` combines seven components, each normalized to `[0, 1]`, into a 0-100 score. Weights are configurable in `StrategyConfig.weights`:

| Component | Weight | Normalization |
| --- | --- | --- |
| Trend structure | 25% | share of `close>SMA20`, `SMA20>SMA50`, `close>SMA200`, rising SMA20, rising SMA50 that hold (needs 3 known) |
| Relative strength | 20% | average 63/126-session excess return vs SPY and the sector ETF; `0.5 + excess/0.20`, clipped (+10% → 1, -10% → 0) |
| Market regime | 15% | 0.6 if SPY > its SMA200, plus 0.4 if SPY SMA50 > SMA200 |
| RSI momentum | 10% | 1.0 for RSI 55-65, linear to 0 at 40 and 80 |
| Pullback location | 10% | 1.0 within 0-0.75 ATR above SMA20, linear to 0 at 2 ATR above or 0.5 ATR below; +0.25 within 0.5 ATR of a 38.2/50/61.8% retracement of an up leg |
| Volume confirmation | 10% | up/down volume ratio: 0 at 0.8, 1 at 1.5 |
| Reward/risk room | 10% | (nearest prior high above the close − close) ÷ (close − chandelier stop); 1:1 → 0, 3:1 → 1; 1 when no prior high is above the close, 0 at or below the stop |

Why these weights: trend and relative strength carry the most published evidence at multi-week horizons (time-series and cross-sectional momentum), so together they get 45%. Market regime is a strong filter but changes slowly, so it acts mostly as a gate and is 15% of the score. RSI, location and volume refine timing and get 10% each. Reward/risk room uses prior highs rather than the support/resistance zones so that the backtest can replay exactly the same rule bar by bar (recomputing zones every day took about 0.2 s per bar). MACD and Stochastics are deliberately absent: they largely restate RSI and moving-average information and would double-count it.

Missing components are excluded and the remaining weights are renormalized; at least 50% of the configured weight must be calculable or the score is unavailable. Grades: ≥75 strong, ≥60 favorable, ≥40 neutral, otherwise weak. The score is a ranking aid, not a probability. The backtest's score-bucket table shows whether higher scores have actually preceded better returns; adjust the weights if they have not.

## Action precedence

First check daily data adequacy and freshness for both holding states. If the confirmed daily inputs are insufficient, or the latest completed daily session differs from the latest expected completed NYSE session, action guidance is **Insufficient data**. A dated historical daily bias may still be displayed, but stale evidence does not authorize a current action. Provisional daily observations cannot repair missing completed evidence.

### No current holding: Buy or Wait

**Buy** requires every required gate to pass and no conditional gate to fail; otherwise the result is **Wait**. Each gate is listed in the panel as passed (✓), failed (✗) or unverifiable (？):

| Gate | Type | Condition |
| --- | --- | --- |
| Strict uptrend | required | `C > M20 > M50`, `d20 > 0`, `d50 > 0` |
| RSI entry range | required | `50 <= R < 70` |
| No chasing | required | `0 <= (C - M20)/ATR14 <= 1.5` (falls back to 0-3% when ATR is unavailable) |
| Daily trigger | required | **Volume breakout**: `C` above the highest high of the previous 60 sessions, with 5-session average volume ≥ 1.5 × the 20-session average; or **SMA20 reclaim**: `P <= P20` and `C > M20`; or **pullback resumption**: `P > P20`, `C > M20`, the lowest low of the last 5 sessions came within 0.5 ATR of SMA20 (or below it), and `C` closed above the previous session's high |
| Reward/risk | optional, off by default | (nearest prior high above `C` − `C`) ÷ (`C` − chandelier stop) ≥ 2 (`min_reward_risk`). When switched on (the **买入需盈亏比 ≥ 2** switch on either page, saved in `preferences.json`; `use_reward_risk_gate=True` in code) it is required. Prior highs are the highest highs of the previous 20, 60, 126 and 252 sessions. Passes when `C` is above all of them; fails when `C` is at or below the stop; unverifiable without ATR, the 22-session high or any prior high |
| Score | required | weighted score available and at least 60 |
| Long-term trend | conditional | `C > SMA200` |
| Market regime | conditional | SPY close > SPY SMA200 |
| Earnings | conditional | no earnings report within the next 7 calendar days |

Conditional gates block only when their data shows a failure. Missing data is shown as unverifiable rather than silently passed.

When the trigger is a volume breakout, the RSI range widens to `50 <= R < 80` and the no-chasing limit to 3 ATR, because a genuine breakout runs hotter and further from SMA20 than a pullback; the panel labels these as breakout limits. The trigger shown first wins when several apply (breakout, then reclaim, then pullback). The entry setup is listed in the panel and in the watchlist's 入场形态 column.

The reward/risk gate was meant to stop a Buy just below an obvious ceiling, where the nearest prior high leaves less than twice the room of the drop to the chandelier stop. It is off by default because the first real-data test did not support it. On 15 large caps over 6 years (run October 13, 2026), the rules with the gate made 78 trades averaging +1.26% each; without it they made 250 trades averaging +1.27%. The blocked entries did about as well as the kept ones, so the gate cut exposure without improving trade quality, and a 1:1 version was worse (+0.90% per trade). The ratio is still shown in the panel and the watchlist, and it remains a 10% score component. Run `--ablation` again on your own list before switching it on.

The old "rising continuation" trigger (any up close above SMA20 in an uptrend) fired on most days of a trend, so it was not really a trigger. Pullback resumption requires an actual test of the average followed by a close above the prior high. The old fixed 3% extension limit treated a utility and a semiconductor stock alike; the ATR limit scales with each stock's volatility.

A daily uptrend with bearish or unavailable intraday evidence can still be a Buy. A strong uptrend with excessive extension or RSI14 at/above 70 remains Wait; strength and entry attractiveness are separate outputs.

### Current holding: Hold, Reduce or Sell

After the same daily adequacy/freshness gate, evaluate:

1. **Sell** (code `reduce_sell`) if any of:
   - the daily bias is Bearish or Strong Bearish;
   - two consecutive completed closes are below their own SMA50 (`P < P50` and `C < M50`);
   - the close is below the **chandelier stop**: 22-session highest high − 3 × ATR14.
2. **Reduce** (code `reduce`) if no Sell condition holds but two consecutive completed closes are below their own SMA20 (`P < P20` and `C < M20`): a partial trim when short-term momentum fades inside an intact medium-term trend.
3. Otherwise **Hold**.

The old rules sold only on a bearish bias or two closes below SMA50. For a 5-20 day horizon those react slowly and can give back a large part of a gain. The chandelier stop trails the recent high by a volatility-scaled distance, and the SMA20 Reduce step responds earlier without exiting the whole position. RSI14 at/above 70, an unfinished daily decline, and bearish intraday evidence still do not independently override Hold. A new-entry Wait and a holding Hold can coexist for the same market assessment.

## Triggers, invalidation, and conflicting evidence

The panel includes the intended timeframe, values/dates for both daily bars, one-session SMA slopes, RSI14, SMA20 extension, every entry gate with its value, the score breakdown, the intraday bars/VWAPs and their ending times, the chandelier level and relevant freshness status.

- For a bullish daily setup, a subsequent completed daily close below the then-current SMA20 or loss of the strict uptrend invalidates that setup. RSI14 leaving `[50, 70)`, price running more than 1.5 ATR above SMA20, SPY falling below its SMA200, an approaching earnings date, or the score dropping below 60 also invalidates the entry conditions. Recompute the moving averages from the new completed data; today's printed level is only the current reference.
- For holding exits, an occurred Sell or Reduce condition is identified as occurred; otherwise those exit conditions remain pending. A recovery requires evaluating new consecutive completed bars and their recalculated moving averages.
- The result explicitly reports contradictory momentum, overbought/oversold conditions, daily/intraday disagreement, excessive extension, unverifiable conditional gates, historical-session evidence, and provisional observations when applicable.

These conditions describe when to reassess. The chandelier level is a rule-based trailing reference, not a guaranteed execution price, and nothing here supplies targets or personalized trading suitability.

## Backtesting

`scripts/backtest.py` replays `evaluate_bar` over real FMP daily history (default: 15 large caps, 6 years):

```powershell
.\.venv\Scripts\python scripts\backtest.py
.\.venv\Scripts\python scripts\backtest.py NVDA AAPL MSFT --years 8 --ablation
```

Each decision uses only completed bars up to that session. Orders fill at the next session's open with a per-side cost (default 5 bps). Reduce sells half, Sell closes the rest. The report shows, per symbol, trade count, win rate, average/median trade, profit factor, strategy return, max drawdown, time in market and buy-and-hold return. It also buckets every session by score and reports the average 10-session forward return per bucket, which shows whether a higher score has actually preceded better returns. A per-setup table splits the trades by entry trigger (volume breakout, SMA20 reclaim, pullback). `--ablation` reruns the test with each gate or setup removed (score, SPY, SMA200, earnings, breakout setup, chandelier stop) and with the optional reward/risk gate switched on at 2:1 and at 1:1, so you can see what each one contributes. CSV files are written to `artifacts/backtest/`.

Limits: the default symbol list is today's large caps, so results carry survivorship bias. The test ignores slippage beyond the cost, overnight gaps through a stop, taxes and position sizing. Treat it as a check on whether the rules behave sensibly, not a forecast.

## Focused verification

`tests/test_signals.py` uses isolated synthetic fixtures to cover bullish, bearish, conflicting, overbought-uptrend, insufficient/stale daily, unavailable intraday, provisional-bar, pullback-trigger, reward/risk, market-gate, earnings-gate, SMA200, score, Reduce and chandelier cases. It also checks that holding status selects different guidance without changing the shared market assessment. `tests/test_strategy.py` covers ATR, Fibonacci legs, volume ratios, prior highs, the breakout trigger and its limits, the reward/risk gate, gap handling, benchmark alignment, the context fetch, held/failed zone tests and a no-look-ahead backtest check; `tests/test_backtest_script.py` runs the backtest command offline. Fixtures verify implementation against the rules; they measure neither historical trading performance nor prediction accuracy.
