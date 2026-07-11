# Quantum Maestro — Dual-System Architecture

Two goals, two systems, one doctrine.

## Goal A — Quantum Maestro (this repo): Streamlit terminal + Tradier execution
Audience: newbie → Bloomberg-terminal pro. The app teaches, scores, gates,
previews, and (optionally, deliberately) executes through Tradier.

## Goal B — Robinhood agentic laboratory (robinhood_agent/): Claude via MCP
A fenced experimental account operated by Claude through Robinhood's MCP.
Equities only (current beta), tiny capital, phased autonomy. Nothing in this
repo executes against Robinhood; the "code" is the directive + guardrails +
audit procedure.

**The two systems never share credentials, accounts, or execution paths.**

---

## Layered decision pipeline (both systems, same doctrine)

    Macro / regime → events → volatility → IWT setup quality
      → instrument selection → deterministic risk gates → preview
      → approval (advisory / review / autonomous) → execute → journal

## New package (this branch)

    quantum_maestro/
      execution/order_models.py     canonical OrderIntent — the only way to express a trade
      execution/tradier_adapter.py  preview-first Tradier client; refuses unapproved intents
      risk/permission_engine.py     deterministic GREEN/YELLOW/RED gates; LLM-proof

Key invariants enforced in code, not prose:
1. No exit plan → structurally invalid order.
2. `risk_approved` can only be set by the permission engine.
3. Live submission additionally requires `QM_LIVE_CONFIRM=I_UNDERSTAND_LIVE_RISK`
   set per session — a human speed bump between sandbox and live.
4. Duplicate intent submission is blocked in the adapter.
5. `human_impaired` session flag forces advisory-only (the "never trade
   tired" rule as code — e.g., newborn months).

## Migration plan for app.py (6,367 lines → modules)
Phase 1 (this branch): new package lands alongside app.py; nothing breaks.
Phase 2: app.py's `tdr_*` functions delegate to TradierAdapter; the Trade tab
builds OrderIntents and shows the Verdict card (color + reasons) before any
submit button is enabled.
Phase 3: extract indicators/IWT scoring into quantum_maestro/strategy/;
app.py becomes UI-only (~1,500 lines).
Phase 4: shadow-mode logging (intents created vs hypothetical outcomes) to
build the evidence base autonomy requires.

## Honesty fix
README claimed "does not execute trades" while shipping order-placement
functions. Updated: execution exists, is optional, sandbox-default, preview-
first, and risk-gated. Tools that execute must say so.
