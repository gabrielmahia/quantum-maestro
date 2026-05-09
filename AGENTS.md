# AGENTS.md — EasyStockTrader (Quantum Maestro)

This file gives AI coding agents (Cursor, GitHub Copilot, Claude Code, Devin, etc.)
the context needed to work reliably in this repository.

---

## What this is

**EasyStockTrader V2 — SPX Income Engine**

A Streamlit-based trade planning simulator for SPX cash-settled vertical credit spreads.
Default mode: defined-risk premium selling (put/call vertical credit spreads).
Secondary modes: long stock/ETF, cash-secured put, covered call, NSE equities.

**Educational simulation only — no broker connectivity, no trade execution.**

**Live app:** https://easystocktrader.streamlit.app  
**Portfolio:** https://gabrielmahia.github.io  
**Methodology:** docs/methodology/SPX_INCOME_PHILOSOPHY.md

---

## Repository structure

```
app.py                                  ← Single-file Streamlit app (all logic)
requirements.txt                        ← Python dependencies (no pandas_ta; pure numpy/pandas indicators)
docs/
  methodology/
    SPX_INCOME_PHILOSOPHY.md            ← Trading philosophy, math, decision rules
tests/
  test_smoke.py                         ← Import + basic run path checks
  test_live_data.py                     ← Indicator and spread math validation
PATCH_NOTES_SPX_VERTICALS.md           ← V1 patch notes (SPX vertical math)
PATCH_NOTES_SPX_VERTICALS_V2.md        ← V2 patch notes (Income Engine)
.streamlit/config.toml                  ← Dark theme, toolbar config
```

---

## Core functions — V2 additions

| Function | Purpose |
|----------|---------|
| `calc_vertical_credit_spread(...)` | Spread P&L math: max profit, max loss, breakeven, credit efficiency |
| `contracts_for_defined_risk(...)` | Size position by max-loss budget |
| `estimate_expected_move(...)` | EM ≈ SPX × IV × √(DTE/365) |
| `expected_move_check(...)` | Flag short-strike placement vs EM boundary |
| `approx_pop_from_delta(...)` | POP ≈ 1 − δ, POT ≈ 2δ |
| `rank_trade_days(...)` | Mon–Thu ranking by VIX, flow, event risk, PDT budget |
| `pdt_guidance(...)` | PDT budget: Legacy / New Intraday Margin / Conservative |
| `_calc_indicators(df)` | Pure numpy/pandas indicator library (replaces pandas_ta) |

---

## Critical rules — do not change without understanding

### 1. pandas_ta is removed — do not re-add it
The app uses a custom `_calc_indicators()` function built entirely with numpy/pandas.
`pandas_ta` depended on `numba`/`llvmlite` which breaks on Python 3.12+. Adding it back
will break Streamlit Cloud deployment.

### 2. Trust integrity banners — do not remove
The app opens with a disclaimer modal. All strategy outputs include "SIMULATION ONLY" markers.
These are not cosmetic — they're required trust-integrity disclosures.

### 3. DEMO vs REAL labelling
- Market data from yfinance = REAL (but delayed)
- IV, Greeks, POP = APPROXIMATION (manual inputs)
- Trade execution = NOT CONNECTED

Do not remove or weaken these distinctions in UI output.

### 4. Credit efficiency scoring
Vertical spreads are scored by credit efficiency, NOT directional reward/risk:
```python
credit_ratio = credit / spread_width
# Preferred: >= 0.25
# Acceptable: >= 0.20
# Thin: < 0.20 → penalty
```
Do not substitute stock-style R/R logic for spread scoring.

### 5. Expected move formula
```python
em = underlying_price * (iv_percent / 100) * math.sqrt(dte / 365)
```
This is a first-order Black-Scholes approximation. Volatility skew is not modelled.
Do not change the formula without updating the methodology doc.

### 6. PDT logic is intentionally conservative
Three modes: Legacy PDT, New Intraday Margin, Conservative.
Do not remove the Conservative mode or allow the engine to assume broker permission.
The operator must explicitly confirm framework.

### 7. Mon–Thu trade planner — do not add Friday
Fridays are intentionally excluded as preferred trade days (gamma risk).
Do not add Friday to the "Best Trade Day" category regardless of other conditions.

### 8. Mobile-first UI
All UI is designed mobile-first:
- `min-height: 44px` for tap targets
- `font-size: 16px` minimum for body text
- Single-column layouts as base
- Desktop enhancements are additive

### 9. CSS dark mode — dual selectors
Every custom HTML class has:
- `@media (prefers-color-scheme: dark)` AND
- `[data-theme="dark"]` selectors
Removing either breaks Streamlit's theme toggle.

---

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Running tests

```bash
pytest tests/ -v
# Expected: 12 passed, 0 failed
```

---

## Adding new features — checklist

- [ ] New math functions: add corresponding test in `tests/test_smoke.py`
- [ ] New strategy mode: add credit efficiency or equivalent scoring
- [ ] New data source: label REAL vs DEMO vs APPROXIMATION in UI
- [ ] New UI element: mobile-first, 44px tap targets, 16px body text
- [ ] New methodology: update `docs/methodology/SPX_INCOME_PHILOSOPHY.md`

---

## Deployment

**Streamlit Cloud** — auto-deploys from `main` branch push.  
App URL: https://easystocktrader.streamlit.app

No secrets required for base functionality (yfinance is public).

---

## License

CC BY-NC-ND 4.0 — personal and educational use permitted. Commercial use prohibited.  
Contact: contact@aikungfu.dev  
Copyright © 2026 Gabriel Mahia. All Rights Reserved.
