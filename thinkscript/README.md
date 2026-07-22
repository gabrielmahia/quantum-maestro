# TeriQuantumOsc — ThinkScript Suite

Chart-side decision support for Quantum Maestro. These are **proxies**: the
canonical regime engine lives in the app (`qm/regime.py`) and sees breadth,
oil, and credit that ThinkScript cannot. On disagreement, the app wins. None
of these place orders, read balances, or size from account equity — by design.

All constants mirror `qm/config.py`. If a limit changes, change the repo
config first, then mirror it here.

## The suite

| File | Type | Install location | What it does |
|---|---|---|---|
| `TeriQuantumOsc_v4.ts` | Lower study | Charts → Studies → Edit → new | **Chart-adaptive.** Change the chart timeframe and everything recomputes on it (like RSI). Runs on any timeframe incl. weekly. Regime/Location/Permission split. Use when you want "regime at the timeframe I'm looking at." |
| `TeriQuantumOsc_v3.ts` | Lower study | Charts → Studies → Edit → new | **Fixed-daily.** Always reads daily regime regardless of chart — a stable macro backdrop that doesn't wobble as you zoom. Breaks on weekly charts (daily secondary-agg illegal there). |
| `TeriQuantumOsc_v2.ts` | Lower study | Charts → Studies → Edit → new | Simplest single-composite oscillator. Legacy; kept for reference. |
| `TQO_RegimeColumn_v4.ts` | Watchlist column | Watchlist gear → Customize → Scripts → new | **Adaptive per-name regime.** Each row's regime score at the column's aggregation. Set column agg to Daily (stable) or Weekly (swing scan). Amber = volatility stress. |
| `TQO_ChartOverlay.ts` | Upper study | Charts → Studies → Edit → new | Buyer/seller zones (daily + weekly), ATR extension band, on-chart setup card (entry/stop/targets/R:R) |
| `TQO_GuardRails.ts` | Study | Charts → Studies → Edit → new | Settlement/event/DTE warnings: 0DTE-ban notice, event lockout, physical-settlement late-day close reminder, anti-repair reminder |
| `TQO_WatchlistColumn.ts` | Watchlist column | Watchlist gear → Customize → Scripts → new | The regime score compressed to a colored cell — scan the whole Teri list at once |
| `TQO_LevelProximity.ts` | Watchlist column | Watchlist gear → Customize → Scripts → new | "Is this name AT a level with room?" — green = favorable-R:R setup candidate, gray = mid-range no-chase |

## Recommended layout

- **Chart:** `TeriQuantumOsc_v2` (lower) + `TQO_ChartOverlay` (upper) + `TQO_GuardRails`.
- **Watchlist (Teri's list):** add `TQO_WatchlistColumn` and `TQO_LevelProximity` as columns. Sort by LevelProximity to surface names sitting at a level with favorable room — that is the daily scan.

## Install notes

1. `.ts` is just text — open the file, copy all, paste into the ThinkScript editor.
2. If a study won't save, ThinkOrSwim is reporting a compile error on a specific line; fix that line (common: `imp_volatility()` on index symbols — switch the symbol input to SPY).
3. Set the chart to **daily** for the intended calibration. Weekly works but every lookback then means weeks, which is a different (slower) instrument.
4. `eventLockout` / `isPhysicallySettled` are **manual inputs** — ThinkScript can't read the economic or earnings calendar. Flip them yourself for FOMC/CPI weeks and for equity (vs SPX) options.

## Doctrine reminders baked into the code

- **0DTE is banned** (MIN_DTE = 7). The studies surface this as a warning; they never emit a 0DTE structure suggestion regardless of score.
- **DEFENSIVE/NO-TRADE never suggest net-short premium** beyond one-sided defined-risk at 0.3x with confirmed headroom.
- **Symmetric measurement, asymmetric policy:** the score measures downtrends as fully as uptrends; any bullish prior lives in the play label, not the scorer.
- **Journal discipline:** an acted-upon signal (including No-Trade) should become a decision-journal row in the app. The chart is where you notice; the app is where you record.

## v3 vs v2 — why both exist

v3 incorporates an external engineering review. Adopted, because they were right:

1. **Timeframe stability** — v2's moving averages/levels/IV inherited the *chart's* aggregation, so on a 15-min chart `sma200` meant 200 fifteen-minute bars. v3 pulls regime data at an explicit `regimeAggregation` (default DAY). This was v2's biggest real flaw.
2. **Regime vs Permission separated** — v2 corrupted the score during event blackout (`Min(score, 0)`). v3 keeps the regime measurement honest and expresses restrictions as a separate PERMISSION output (BLOCKED / DEFENSIVE / SELECTIVE). This is "symmetric measurement, asymmetric policy" — the same rule as the engineer handoff doc.
3. **Prior-bar levels** — `Highest(high[1], N)` so a live intraday high doesn't instantly become "resistance."
4. **De-duplicated momentum** — RSI + MACD only (dropped stochastic, which re-measured the same thing a third time).
5. **Missing-data handling** — absent IV/VIX suppresses the recommendation instead of faking a neutral 50.

**The one place the review was wrong (and v3 corrects):** it proposed detecting a weekly chart and "falling back to the chart timeframe." ThinkScript enforces the secondary-aggregation rule at **compile time** — an illegal `close(period=DAY)` on a weekly chart is rejected whole; a runtime conditional cannot rescue it. So v3 instead pulls regime data at `regimeAggregation` unconditionally, and shows a loud WARNING label when the chart timeframe is >= the regime aggregation. **On weekly/monthly charts, either set `regimeAggregation` to WEEK/MONTH, or use v2.** That's why v2 is retained rather than deleted.

**Kept configurable against the review:** `requiredHeadroomATR` stays at 0.5 (the review silently raised it to 1.0). 0.5 keeps equity-name setups tradable; raise to 1.0 for stricter index-only premium. A deliberate knob, not a hidden default.

## v4 (adaptive) vs v3 (fixed-daily) — which lower study to run

Both are correct; they answer different questions.

**v4 — chart-adaptive (recommended for everyday use).** Every calculation inherits the chart's timeframe, exactly like a native RSI. Switch to weekly and trend/momentum/levels/IV recompute weekly; switch to 15-min and they recompute 15-min. Runs on *any* timeframe including weekly/monthly. Use it when the lower panel should describe "the regime **at the timeframe I'm trading**" — weekly swings off a weekly chart, intraday off a 15-min. Tradeoff: a weekly score and a 5-min score are different instruments; don't compare their absolute values.

**v3 — fixed-daily.** Ignores the chart and always reads *daily* regime. Correct behavior for the question "what is the macro backdrop today," whose answer shouldn't change based on whether you're staring at a 5-min or a daily chart. Cost: it uses daily secondary aggregation, which ThinkScript forbids on weekly/monthly charts (compile error there).

**Rule of thumb:** run **v4** as your lower study since it matches how you navigate timeframes. Keep v3's *concept* for the app's canonical engine, which is genuinely daily-fixed for the right reason. The app engine — not any chart study — remains authoritative for sizing and the promotion gate.

**On very low timeframes (1-5 min):** `imp_volatility()` and VIX9D get sparse, so v4's IV Rank / 9D30D labels may read "unavailable" / "N/A" more often. That's the missing-data suppression working — a blank beats a fabricated number.

### Adaptive watchlist column

`TQO_RegimeColumn_v4` shows each name's regime score in a single colored cell, computed on that row's own symbol at the column's aggregation. Set the column aggregation to **Daily** for a stable per-name regime, or **Weekly** to scan weekly-swing regime across Teri's whole list at once. Amber cell = volatility stress present (act defensively regardless of score). Pair with `TQO_LevelProximity` and sort to surface names that are both in-regime and sitting at a level with room.

## v4.1 fix — the "all N/A" bug (2026-07)

Symptom: on an equity chart (e.g. AAPL/NASDAQ), every label read N/A — Score N/A, VIX unavailable, IV Rank N/A, Headroom N/A — even though the study loaded.

Root cause: the `$SPX.X` / `$VIX.X` / `$VIX9D.X` symbol form did not resolve/align as a secondary symbol on the equity chart, returning `NaN`. Because every label was gated on a `*DataOK` flag, one failed pull cascaded the *entire* study to N/A.

Two fixes:
1. **Bare symbol strings** (`"SPX"`, `"VIX"`, `"VIX9D"`) — these resolve reliably as secondaries (matching how working ThinkScript community studies reference `close("vix")` etc.). Applied across v3, v4, and the regime column.
2. **Graceful degradation** — trend and momentum are price-driven and now always compute. If the VIX/VIX9D/IV feeds are missing, only those components drop to 0 and their own labels say "unavailable"; the regime score still works off price. A missing vol feed no longer blanks the whole study. v4 also falls back to the chart symbol (with a visible NOTE label) if `marketSymbol` itself won't resolve.

If VIX/VIX9D still read "unavailable" in your data package, the study remains fully functional on price alone — the volatility contribution simply stays neutral until the feed is available.

All label strings are ASCII-only (em-dashes → hyphens) to avoid ToS string-render quirks.
