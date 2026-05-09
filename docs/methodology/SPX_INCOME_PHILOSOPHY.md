# SPX Income Engine — Trading Philosophy & Methodology

## Overview

EasyStockTrader V2 implements a **defined-risk, income-first** approach to systematic options trading on the S&P 500 (SPX). This document captures the philosophy, mathematics, and decision rules that drive every component of the scoring engine.

---

## Core Philosophy

### 1. Income Over Direction

The foundational insight of this strategy is the asymmetry between **predicting** and **collecting**. Directional trading requires being right about where the market goes. Premium selling requires only that the market does *not* reach a specific level.

> "You don't need to know where the market is going. You need to know where it is *not* going."

**Implication for the engine:** Every scoring component is calibrated to penalise directional overconfidence and reward probability-positive positioning.

---

### 2. Defined Risk as Operational Discipline

Vertical credit spreads cap maximum loss at entry. This is not merely a risk-management technique — it is an **operational discipline** that enables:

- Pre-trade position sizing (max loss is known before clicking)
- Portfolio-level exposure limits (not stop-loss dependent)
- PDT budget management (knowing max loss helps decide whether a position warrants a day-trade to close)

**Math:**

```
Max profit  = credit × 100 × contracts
Max loss    = (spread_width − credit) × 100 × contracts
Max loss %  = max_loss / account_value  →  must be < trade_risk_budget
Breakeven   = short_strike − credit        (put credit spread)
Breakeven   = short_strike + credit        (call credit spread)
```

Credit efficiency threshold:
```
Preferred:   credit ≥ 0.25 × spread_width
Acceptable:  credit ≥ 0.20 × spread_width
Thin:        credit < 0.20 × spread_width  →  penalty applied
```

---

### 3. Expected Move as the Probability Anchor

The expected move (EM) is the market's own estimate of one-standard-deviation price range over the remaining DTE. Selling outside the EM means the market assigns a probability of < ~16% to the short strike being breached (one tail of a normal distribution).

**Math:**

```
EM ≈ SPX_price × (IV% / 100) × √(DTE / 365)
```

Strike placement relative to EM:
- **Outside EM**: POP approximately ≥ 84% (1-sigma edge)
- **Inside EM**: Market assigns meaningful probability to short strike being reached — penalty applied

This is a first-order approximation. Real-world skew (volatility smile) means put spreads at the same EM distance as call spreads carry different actual probabilities. This is flagged as a manual verification item.

---

### 4. POP and Touch-Risk from Delta

Probability of profit (POP) and probability of touch (POT) are approximated from the short-strike delta:

```
POP ≈ 1 − |delta|
POT ≈ 2 × |delta|
```

**Scoring thresholds:**
```
POP ≥ 0.70  →  Strong
POP ≥ 0.65  →  Acceptable
POP < 0.65  →  Penalty applied
```

**Why POT matters:** A spread can expire worthless even if price touches the short strike, but touching the short strike typically forces active management (buying back early), which costs premium. POT gives a realistic picture of active management frequency.

---

### 5. Event Risk as Binary Discontinuity

Most options-pricing models assume continuous underlying movement. Major macro events (FOMC, CPI, NFP, earnings) create **discontinuous jumps** that can move SPX through a vertical spread's entire width in a single session. Defined risk does not protect against loss of the full spread width — it only defines that width.

**Engine rules:**
- New premium entry within 48h of major event: **penalised**
- Holding short premium through event (planned): **strong warning issued**
- Event risk does not block the trade — it informs the operator

---

### 6. Mon–Thu Preference (Gamma Management)

Friday gamma risk is the most dangerous period for short options. Short-dated SPX options (< 7 DTE) experience rapid gamma expansion on Fridays, making position management less predictable and closing costs more expensive.

**Trade day ranking logic:**
```
Best Trade Days:    Mon, Tue, Wed (enter new positions, manage existing)
Conditional:        Thu (monitor closely; avoid new short-dated entries)
Avoid:              Fri (do not sell new short-dated premium)
```

Additional overlay: passive flow windows (1st–5th of month) increase intraday volatility in broad ETFs, which affects SPX indirectly. These windows are flagged as conditional, not blocked.

---

### 7. PDT Budget as Ammunition Rationing

For accounts below $25,000 (Legacy PDT regime), the IRS/FINRA Pattern Day Trader rule limits intraday round-trips to 3 in any rolling 5-business-day window. Exceeding this triggers a 90-day restriction.

Under the new FINRA intraday margin framework (if broker-enabled), this restriction may be relaxed — but the engine treats this as an operator-confirmed input, not an assumption.

**Operational rule:**
```
Legacy PDT:
  Day trades remaining = 3 − day_trades_used (rolling 5 days)
  If same-day exit planned → consumes 1 day trade
  Reserve ≥ 1 day trade for A+ management decisions

New Intraday Margin (broker-confirmed):
  More permissive — engine still tracks and displays budget

Conservative:
  All intraday exits treated as scarce regardless of framework
```

---

### 8. IWT 7-Step Verification Integration

Teri Ijeoma's (Invest With Teri) 7-step trade verification framework is implemented as a scoring overlay on top of the vertical spread math. The IWT checklist verifies:

1. **Freshness**: Setup is based on recent price action, not stale signals
2. **Time-in-zone**: Price has been in the decision zone long enough to confirm intent
3. **Speed-out**: There is a clear, fast move away from the zone (momentum confirmation)
4. **Volume**: Volume confirms the move (institutional participation signal)
5. **R/R threshold**: Risk/reward meets minimum standard (adapted to credit efficiency for spreads)
6. **Trend alignment**: Trade direction aligns with the higher-timeframe trend
7. **Macro alignment**: Macro conditions (VIX, rates, passive flows) support the trade

For vertical spreads, steps 5 and 6 are adapted:
- Step 5 → credit efficiency (credit ≥ 25% of spread width)
- Step 6 → trade direction relative to VIX regime and EM placement

---

### 9. Cash Settlement Advantage of SPX

SPX options are:
- **European-style**: Can only be exercised at expiration (no early assignment risk)
- **Cash-settled**: No underlying shares are delivered (no pin risk, no large margin call at expiration)
- **1256 contracts**: 60% long-term / 40% short-term tax treatment (US-specific, verify with a tax advisor)

These structural advantages make SPX the preferred vehicle for mechanical premium-selling programs.

---

## Model Limitations

| Limitation | Status |
|-----------|--------|
| Live IV/Greeks from broker | Not integrated — manual input required |
| Real option chain pricing | Not integrated — manual entry |
| Volatility skew adjustment | Not implemented — EM is symmetric approximation |
| Assignment / pin risk modelling | N/A (SPX is cash-settled) |
| Broker connectivity / execution | Not implemented — simulation only |

---

## Trust Integrity Statement

All outputs from this engine are **educational simulations**. No trade is executed. No broker is connected. The PDT guidance, expected move, POP/POT, and scoring outputs are approximations for planning purposes. Broker rules, real-time IV, and actual option chain data govern live trading decisions.

---

*Document version: 2.0.0 — 2026-05-08*  
*Author: Gabriel Mahia — contact@aikungfu.dev*  
*License: CC BY-NC-ND 4.0*
