# Spec Conformance Map

Maps the canonical functional specification (`docs/SYSTEM_SPECIFICATION.md`)
to what is actually implemented, so the gap between design and code is never
guessed at. Status is honest: IMPLEMENTED / PARTIAL / NOT BUILT.

| § | Capability | Status | Where |
|---|---|---|---|
| 2 | Buyer/seller zone model, proximal/distal | IMPLEMENTED | `qm/iwt_zones.py`, ThinkScript v6 |
| 3 | Canonical long (entry/stop/target/RR/qty) | IMPLEMENTED | `qm/iwt_canonical.iwt_long_trade` — validated to the cent vs course sheet |
| 4 | Canonical short (symmetric) | IMPLEMENTED | `qm/iwt_canonical.iwt_short_trade` |
| 5 | Odd Enhancers (0–8, cohorts) | IMPLEMENTED | `qm/iwt_zones.odds_enhancer` — matches course table incl. "1+ visits"=0 |
| 6 | Price confirmation layer | PARTIAL | breakout/turn confirmation in ThinkScript; VWAP + breadth confirmation NOT built |
| 7 | Macro/regime overlay | PARTIAL | `qm/regime.py` (trend/VIX/breadth/oil/credit); rates, liquidity, gamma, flows NOT built |
| 8 | Cross-index divergence detector | NOT BUILT | concept documented; no live implementation |
| 9 | Direction-selection state machine | PARTIAL | agents + risk gate approximate it; not an explicit FSM |
| 10 | Instrument-selection philosophy | IMPLEMENTED | `qm/iwt_options.py` + `qm/execution_governance.route_strategy` |
| 11 | Long-option philosophy (45–90 DTE) | PARTIAL | DTE floor + doctrine notes in `iwt_options`; greeks modelling NOT built |
| 12 | Short-premium philosophy | IMPLEMENTED | naked shorts hard-blocked; credit requires rich IV + confirmation |
| 13 | Defined-risk structure preference | IMPLEMENTED | `route_strategy` prefers debit on event-heavy sessions |
| 14 | Account-aware routing | IMPLEMENTED | `qm/execution_governance.ACCOUNT_CAPABILITIES` |
| 15 | Earnings no-trade window | IMPLEMENTED | ThinkScript ER gate; `iwt_canonical` worksheet block |
| 16 | Time-of-day model | NOT BUILT | session windows not encoded |
| 17 | Event gate (minutes-to-event) | PARTIAL | event blackout flag exists; no minutes-to-event clock |
| 18 | Intent-versus-order check | IMPLEMENTED | `qm/execution_governance.validate_intent` |
| 19 | Position sizing (planned vs catastrophic) | IMPLEMENTED | `qm/sizing.py` + `iwt_canonical` cascade |
| 20 | Tranche model | IMPLEMENTED | `TranchePlan`, `may_add_tranche` (blocks rescue adds) |
| 21 | 0DTE regime | IMPLEMENTED (as prohibition) | 0DTE banned by doctrine — stricter than spec |
| 22 | Managed vertical / recycling accounting | IMPLEMENTED | `ManagedVerticalLedger` — separates management alpha from ordinary decay |
| 23 | Probable path, never certainty | IMPLEMENTED | epistemic labelling throughout docs//journal |
| 24 | Regime transitions (relationships) | PARTIAL | multi-factor scoring exists; causal-combination reasoning NOT built |
| 25 | Position-management hierarchy | NOT BUILT | documented order of questions, no code |
| 26 | Journaling / attribution schema | PARTIAL | `qm/journal.py` logs decisions incl. NO_TRADE; full schema fields not all captured |
| 27 | Learning engine (segmented expectancy) | PARTIAL | expectancy + promotion gate exist; segmentation by regime/score/time NOT built |
| 28 | Morning decision engine | NOT BUILT | no scheduled macro scan |
| 29 | Non-morning change detector | NOT BUILT | |
| 31 | Deterministic-vs-LLM boundary | IMPLEMENTED | risk/sizing/gates are code; `GOVERNANCE.md` formalises it |

## The honest headline

Roughly two-thirds of the spec is implemented, and every **safety-critical**
layer now is: sizing, risk cascade, intent validation, account permissions,
naked-short prohibition, earnings/event blackout, tranche discipline.

What remains unbuilt is mostly **context enrichment** (time-of-day, cross-index
divergence, morning scan, segmented learning) rather than execution safety.

The one gap that matters more than its size suggests is **§6 confirmation** —
the spec's insistence that a zone is only a *candidate location* and price must
demonstrate the reaction before acting. Partial implementation means the system
can still say "at a good zone" without fully answering "is it actually turning?"

And the standing caveat: mechanical zone detection remains unvalidated against
hand-marked zones. Every downstream calculation is provably correct; whether the
detector finds the *right* zones is still unproven.
