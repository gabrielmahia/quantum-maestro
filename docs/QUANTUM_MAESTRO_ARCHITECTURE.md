# Quantum Maestro — Architecture & Roadmap

## Core pattern: AI proposes, deterministic engine disposes

```
Operator / (future) LLM agents
        │  propose
        ▼
┌─────────────────────┐
│ Seven-Agent Stack   │  any agent vetoes; none initiates; ≥5/7 required
└────────┬────────────┘
         ▼
┌─────────────────────┐
│ Deterministic Risk  │  11 hard rules, no override path (qm/risk.py)
│ Engine (Agent 7)    │
└────────┬────────────┘
         ▼
┌─────────────────────┐     ┌──────────────────┐
│ Journal (SQLite)    │────▶│ Promotion Gate   │  SHADOW → LIVE
│ every decision      │     │ (qm/journal.py)  │
└────────┬────────────┘     └──────────────────┘
         ▼
   SHADOW: paper record          LIVE (future): broker adapter
```

This is the same architecture as PVoC-as-a-Service and Agri-Trace data validation:
probabilistic intelligence upstream, deterministic compliance gate downstream.
Build the pattern once as a philosophy; reuse it everywhere.

## Module map

| Module | Responsibility | Deterministic? |
|---|---|---|
| `qm/config.py` | Hard limits + gate criteria ("the constitution") | Yes — frozen dataclasses |
| `qm/regime.py` | Portfolio State Engine; yfinance autofill w/ manual override | Yes — transparent scoring |
| `qm/risk.py` | 11-rule veto engine | Yes |
| `qm/sizing.py` | Fixed-risk (Ijeoma) + fractional Kelly (Thorp), regime-scaled | Yes |
| `qm/agents.py` | Checklist agents + aggregation rule + strategy selector | Aggregation yes; scoring human (v1) |
| `qm/journal.py` | SQLite decision log, expectancy, gate evaluation | Yes |
| `qm/wisdom.py` | Global masters library | Content |

## Design decisions and their reasons

1. **Shadow-first, gate-locked live mode.** Live capital before demonstrated expectancy is donation. The gate reads only the journal — never memory, never narrative.
2. **No-trade and veto logging.** A prevented bad trade is a positive-expectancy event; systems that only log fills can't measure their most valuable output.
3. **Human agents before LLM agents.** 60–90 manually-completed stacks = the labeled evaluation dataset for future AI agents. Sequencing: validation before infrastructure.
4. **Kelly locked behind 20 closed trades.** Edge estimation from small samples is how quarter-Kelly becomes full-Martingale.
5. **No streaming P&L on screen.** Darvas traded by telegram; distance from the ticker was the edge. The app shows decisions and expectancy, not ticks.
6. **Limits are code, not settings.** A limit you can change in a UI during a drawdown is not a limit.

## Roadmap (strictly sequenced)

**Phase 1 — now:** Daily regime reads + full-stack paper decisions. Target: 50 decisions / 30 closed shadow trades.

**Phase 2 — persistence:** Swap SQLite → Supabase/Postgres (one-file change in `qm/journal.py`) so Streamlit Cloud redeploys stop resetting the journal.

**Phase 3 — LLM agents (only after Phase 1 dataset exists):** Each agent = one Anthropic API call with a structured JSON verdict, graded against the human-labeled dataset before its votes count. Aggregation rule unchanged: AI may veto, never initiate.

**Phase 4 — broker adapter (only after gate passes):** Tradier sandbox → preview-only order staging → manual confirm → (much later) auto-execution of gate-approved structures only. The adapter imports `qm/risk.py`; an order the engine hasn't approved cannot be constructed.

**Kill criteria (falsification):** if after 60 closed shadow trades expectancy < 0 net of modeled costs, the system as configured has no edge. Response is not "add indicators" — it is: archive, write the post-mortem, and either change ONE module at a time with fresh out-of-sample tracking, or conclude the honest thing: the edge isn't there, and index + T-bills is the professional-grade decision.
