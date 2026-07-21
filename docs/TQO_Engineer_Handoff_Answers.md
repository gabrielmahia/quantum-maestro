# TeriQuantumOsc — Product Owner Answers to §21 (Engineer Handoff)

Prepared for the implementing engineer. Authoritative source for all constants:
`qm/config.py` in gabrielmahia/quantum-maestro (public). Where this document and
the design doc disagree, this document wins; where this document and the repo's
config disagree, the repo wins.

---

**Q1. What pivot/touch rules produced the current Weekly [Buy P] / [Seller P] / [Seller D] levels?**
They are hand-drawn from discretionary IWT analysis — there is no existing algorithm to reverse-engineer. Treat the currently plotted levels (Weekly Seller D 7620.9, Seller P 7500, Buy P 7230.12, Buy D 7046.55; Daily Sell D 7648.75, Sell P 7581.25, Buy P 7431 on /ES) as the gold-standard labeled data. Calibrate §8's LevelConfidence weights so the algorithm reproduces these zones within ±0.25 ATR before tuning anything else. Success criterion: the algorithm finds these levels without being told them.

**Q2. Zones or single prices? How many visible?**
Zones (cluster width ≈ 0.25–0.5 ATR), labeled at zone midpoint. Maximum visible: nearest 2 per side per timeframe (daily + weekly = 8 max). Anything more is clutter; the trade only ever cares about the nearest opposing level and the one behind it.

**Q3. Authoritative timeframes per horizon?**
Weekly levels are strategic (position framing); daily levels are tactical (entry/invalidation). For the 5–15 DTE primary window: daily levels govern entry and stop, weekly levels govern targets and the "room" calculation. Intraday levels are display-only in v1 — no scoring weight.

**Q4. How does the user indicate event lockouts?**
Manual boolean input in v1 (as built today), flipped for FOMC/CPI/NFP weeks and per-symbol earnings. v2 may add a hardcoded known-dates array updated monthly. The lockout must cap the score at ≤ 0 and force the play label to the lockout message — a penalty is insufficient; it must be a ceiling.

**Q5. Minimum reward-to-risk?**
Directional: 2.0 minimum (hard, from config). Defined-risk credit structures: credit ≥ 25% of max loss. Configurable upward only — the floor is not user-adjustable. This mirrors `LIMITS.MIN_REWARD_RISK` / `MIN_REWARD_RISK_INCOME`.

**Q6. Symmetric scores, or long-run bullish prior for indexes?**
Symmetric MEASUREMENT, asymmetric POLICY. The scorer must produce −3 trend in a fully inverted MA stack exactly as it produces +3 in a stacked one (the v1 asymmetry was a defect, not a prior). The bullish prior for broad indexes is then expressed in policy only: bearish permission on SPX/SPY/QQQ requires TradeScore ≤ −4 where bullish requires ≥ +3, and bearish plays default one size notch lower. Rationale: passive-flow persistence justifies cautious shorting policy; it never justifies mismeasuring a downtrend.

**Q7. Which short-vol proxy is available?**
VIX9D is available in this ToS environment and is in production use (current reading 0.95 vs VIX). Keep VIX9D/VIX as primary; fall back to 5-day VIX rate-of-change if VIX9D returns NaN. Stress threshold: ratio > 1.02 = −2; calm: < 0.90 = +1 (matches deployed v2 study).

**Q8. Play messages: advisory only, or strategy classes by DTE?**
Strategy CLASSES only, never strikes or specific orders, per the doc's own §12 — with two hard overrides: (a) the 0DTE bucket NEVER emits a structure suggestion; its play label is fixed to "BANNED by risk doctrine — observation only" regardless of score (the execution system enforces MIN_DTE = 7; the chart must not contradict it); (b) DEFENSIVE and NO-TRADE states never suggest net-short-premium structures beyond one-sided defined-risk at 0.3x with confirmed level headroom.

**Q9. Alert limits intraday?**
Regime-change alerts: once per bar, minimum 15 minutes between repeats of the same transition. Level-test alerts: once per level per session. Vol-stress breach: once per session unless it recrosses calm first. All alerts respect a global maximum of ~6/session — an alert system that fires constantly is muted within a week and then protects nothing.

**Q10. Gold-standard historical examples?**
(a) Counterexample: the July 2026 AAPL 0DTE 325/327.5 call credit spread — near-max loss; the system must show NO-TRADE/banned for this entry. (b) The June–July 2026 SPX advance and current pullback — study should show OFFENSIVE through the June leg, degrading to NEUTRAL→DEFENSIVE across mid-July (the deployed v2 histogram already renders this sequence; preserve it). (c) Validation regimes per §18: 2020 crash weeks must produce DEFENSIVE/NO-TRADE before the largest down bars, not after — walk-forward, no look-ahead.

---

## Additional binding constraints (not asked, but required)

1. **Constants unification:** NEUTRAL = 0.6x (not 0.5x), DEFENSIVE = 0.3x (not 0.25–0.3x), matching `qm/config.py REGIME_MULTIPLIER`. If analysis later justifies different values, change the repo config first, then mirror here — never the reverse.
2. **Scale unification:** normalize the composite to the repo's regime vocabulary and thresholds; the study header comment must state which config version it mirrors.
3. **Journal discipline in acceptance criteria:** every acted-upon alert (including No-Trade decisions) is expected to produce a decision-journal row in the Quantum Maestro app. Add to Sprint 4 acceptance: play labels include the reminder "journal this decision" on regime transitions.
4. **Graceful degrade:** /ES–/NQ confirmation and imp_volatility() must fail silently to neutral (0 contribution) with a visible "degraded inputs" marker — never a script error, never a stale value presented as live.
5. **The study is a proxy, not the canon.** The app's regime engine (breadth, oil, credit inputs the chart cannot see) is authoritative for sizing and the promotion gate. On disagreement, the app wins. This hierarchy appears in the study's header comment.
