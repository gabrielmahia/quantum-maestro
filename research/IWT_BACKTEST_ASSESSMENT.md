# IWT Backtest Integration — Assessment & Provenance

External work (a VIP-archive-derived backtest engine and three workbooks) was
reviewed and selectively integrated. This document records what was adopted,
what was rejected, and what remains unproven — so the provenance is auditable
and nobody later mistakes a proxy for a validated edge.

## What the external engine gets right (adopted)

1. **The eight-point odds enhancer is a real model, not a slogan.** It decomposes
   "a good zone" into four measurable factors — base tightness, departure speed,
   freshness, and reward:risk — each 0-2, summing to 0-8. That is genuinely more
   coherent than "buy at a buyer level," and it is now `qm/iwt_zones.py`
   (unit-tested) rather than living only in a script.

2. **Long/short symmetry and the ATR stop buffer** are faithful to the method:
   longs at buyer zones targeting the prior seller zone, shorts mirrored, stop
   beyond the distal boundary by an ATR buffer.

3. **Boundary survival is the honest options proxy.** Daily bars cannot price a
   spread, so instead of fabricating option P&L the engine asks the answerable
   question: within N days, did price breach the short-strike neighborhood? That
   is a probability, not a fill — and stating it that way is the correct
   epistemic posture. Ported to `qm/iwt_zones.boundary_survival`.

4. **Daily-bar ambiguity handled conservatively.** A bar spanning both stop and
   target is counted STOP-FIRST and labelled `stop_ambiguous` so the ambiguity
   is countable, never hidden. Ported as `resolve_daily_outcome`.

## What was hardened (real fragilities in the original)

- **SPY hard-crash.** The original raised and died if SPY didn't download,
  losing the whole run. `research/iwt_backtest_v2.py` now degrades to a disabled
  market filter with a loud warning instead of crashing.
- **Single free data source.** Stooq returns HTML (not CSV) under rate limiting
  or from some IPs, which fails every symbol. This is noted as a limitation; a
  production run needs a real feed or cached data. The core LOGIC was verified
  independently on synthetic bars.

## What is NOT proven (do not skip this section)

1. **Mechanical zones are a PROXY for Teri's manual zones — unvalidated.** The
   archive itself never numerically defines "fast departure" or the exact base
   construction. The engine uses transparent, adjustable thresholds. **The
   decisive validation — comparing proxy zones against a set of hand-marked
   Teri-class examples — has not been done.** Until it is, backtest statistics
   describe the proxy, not the method.

2. **No options P&L exists.** Only boundary probabilities. Any statement about
   "credit-spread returns" would be fabrication until historical option chains
   are supplied.

3. **Survivorship and point-in-time gaps.** No earnings calendar, borrow costs,
   dividends on shorts, or point-in-time index membership. The 2021-dated index
   constituent weights in the archive are stale and were NOT treated as current.

4. **Free daily data is not an exchange feed.** Splits/adjustments and bad prints
   are not audited.

## How this connects to the promotion gate

This is the validation layer the gate has been waiting for — but with a caveat
that matters: a positive backtest here is **necessary, not sufficient**, evidence
because of item (1) above. The gate still requires live SHADOW-mode journaled
decisions with positive after-cost expectancy. Backtest expectancy that is
negative is a strong kill signal; backtest expectancy that is positive earns the
right to keep paper-trading, nothing more. This ordering is deliberate: it is
much easier to make a backtest look good than to make a live account compound.

## Files

- `qm/iwt_zones.py` — scoring + boundary survival, pure & unit-tested (the keeper).
- `research/iwt_backtest_v2.py` — full engine (external, hardened), for offline runs.
- Workbooks stay out of the repo (large binaries); their logic is captured above.
