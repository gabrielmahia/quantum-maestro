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
| `TeriQuantumOsc_v2.ts` | Lower study | Charts → Studies → Edit → new | The oscillator: symmetric regime score histogram, IV-rank premium bias, VIX9D/VIX stress, level-headroom-gated play label |
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
