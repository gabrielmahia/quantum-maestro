# Teri Ijeoma IWT — Organized Curriculum

A followable path through the Invest With Teri material, reorganized **by
capability** rather than by calendar week, and mapped to the Quantum Maestro
tools that operationalize each piece. Work top to bottom; don't skip Phase I.

> Source: Teri Ijeoma's Trade & Travel / IWT course, the VIP archive workbooks,
> and the eight-point odds enhancer. This is a study guide, not the course
> itself. IWT is execution discipline; Teri's income claims are marketing —
> your own journaled expectancy is the only admissible evidence.

---

## PHASE I — Foundations (do not skip)

### Module A — Market mechanics
Orientation, order types, brokerage mechanics, market participants, and
trading psychology. Goal: understand what actually moves price before trying
to predict it.

### Module B — Risk (the most important module)
Position sizing, reward:risk, portfolio sizing, stop losses, probability,
capital preservation. **Everything downstream depends on this.** Teri's core:
risk is fixed, position size varies.
→ *Quantum Maestro:* `qm/sizing.py` (fixed-risk), the risk engine's hard caps.

---

## PHASE II — Reading price

### Module C — Technical analysis
Trend, support, resistance, buyer levels, seller levels, candles, momentum.
The vocabulary of the chart.
→ *App page 1 (Regime), ThinkScript TeriQuantumOsc (trend/levels).*

### Module D — The 7-Step IWT Process (the execution checklist)
1. **Select the stock** — liquid, institutional names.
2. **Find the buyers** — where institutions accumulate (buyer zone).
3. **Find the sellers** — where they distribute (seller zone).
4. **Calculate reward:risk** — only favorable ratios justify entry.
5. **Position size** — from the dollars actually at risk.
6. **Execute** — predefined entry, stop, target. No emotional changes.
7. **Exit** — never hope, never average down, never move a stop emotionally.
→ *App page 2 (seven-agent stack) and page 3 (risk engine + sizer) ARE this
checklist, enforced in code. ThinkScript setup card mirrors steps 2-4.*

---

## PHASE III — Zone quality (the eight-point odds enhancer)

Not every buyer level is equal. Score each zone 0-8:

| Factor | 2 points | 1 point | 0 points |
|---|---|---|---|
| Base candles | 1-2 | 3-4 | 5+ |
| Departure speed | fast (>=1.5 ATR) | average (>=0.75 ATR) | slow |
| Freshness | 0 revisits | 1 revisit | 2+ revisits |
| Reward:risk | 3.0+ | 2.0-2.99 | <2.0 |

- **7-8 = PRIMARY** cohort — full size, limit at the proximal line.
- **5-6 = SECONDARY** — half size, require confirmation (a close out of the zone).
- **0-4 = SKIP.**

Log the two cohorts **separately** — never combine, or a strong cohort's edge
gets diluted by a weak one.
→ *App page 10 (IWT Zone Scorer), `qm/iwt_zones.py`, ThinkScript v5 on-chart
"ZONE ODDS" label.*

---

## PHASE IV — Trade structures

### Module E — Short selling
Bearish trades at seller zones targeting the prior buyer zone. Borrowing,
the asymmetry of short risk.

### Module F — Gaps & Globex (the morning workflow)
Overnight futures, gaps, international markets, the open. The 8am read:
Asia → Europe → ES futures → currencies → VIX → economic calendar. Costs
nothing, informs whether there's an edge today.

### Module G — Options (the income engine)
- **Long premium:** long calls, long puts (direction + timing bets).
- **Defined-risk credit:** put credit spreads (bullish), call credit spreads
  (bearish) — the primary income structures.
- **Premium selling mechanics:** theta, probability, IV.
→ *App page 3 strategy selector; ThinkScript premium-rich/cheap bias.*
→ **Hard rule:** defined-risk only, no 0DTE, no short premium into events.

---

## PHASE V — Market context (beyond Teri)

Regimes (bull/bear/sideways/volatile), macro (Fed, liquidity, dollar, oil),
institutional flows (passive, buybacks, quarter-end, dealer gamma), portfolio
construction, and performance analytics (win rate, expectancy, Sharpe,
drawdown, Kelly). Then the AI layer: the regime → agents → risk → journal
stack that decides *whether* a Teri setup should be taken at all.
→ *This is what Quantum Maestro adds on top of IWT.*

---

## The order to actually learn this (5-year arc)

1. Long-term investing foundations (Buffett, Marks, Graham) — temperament.
2. IWT process (Modules A-D) — execution discipline.
3. Defined-risk premium selling (Module G) — the income engine.
4. Portfolio construction — cash, hedges, diversification.
5. Macro regime analysis — Fed, liquidity, credit, currencies.
6. Futures — as an information source first, trading instrument much later.
7. Volatility & options pricing — Greeks, skew, term structure.
8. Behavioral finance & market history — runs in parallel with all of it; it's
   the discipline that keeps the other seven funded.

What's deliberately missing: chasing ever more exotic strategies. The edge is
in doing the boring parts consistently, which is exactly what the journal and
the hard rules exist to enforce.

---

## Canonical source integration (from Teri's course documents)

The following are drawn *directly* from the Trade & Travel course files
(Personal Trading Plan, RR spreadsheet, IWT Stock-Pick worksheet, Key Terms),
and are now enforced in code (`qm/iwt_canonical.py`, app page 11):

**The risk-plan cascade.** Teri's plan specifies more than per-trade risk — it
is a full cascade. Stocks/Options defaults on a $100k account: per-trade $1,000
(1%), daily $3,000 (3%), weekly $5,000 (5%), monthly $10,000 (10%), and trade
counts of 2/day, 8/week, 20/month. Stored as percentages so they scale to any
account. **Where Teri's 3% daily and Quantum Maestro's 2% doctrine disagree, the
stricter 2% wins** — the newborn-period impaired-window rule and "protect
downside first" both argue for the smaller daily bleed.

**The RR formula chain** (validated to the cent against the course spreadsheet):
long stop = distal_BZ − 20%·ATR, entry = proximal_BZ, target = proximal_SZ,
shares = risk_$ ÷ risk. The eight-point odds enhancer was also confirmed exactly
against the "Odd Enhancers: Testing Your Level Strength" worksheet.

**The pre-trade worksheet** as a scored gate: volume > 1M, uptrend, 52-week and
3-month range position, earnings distance, best-in-breed (relative strength),
"$1/day" mover, and news. Hard blocks (liquidity, earnings blackout) stop a pick
before charting; soft warnings (range extremes, not-a-leader) inform it.

What stays deliberately un-encoded: the order-placement mechanics (limit/market/
stop/GTC, etc.) are educational — Quantum Maestro's broker adapter is limit-only
by doctrine, which is the correct subset for defined-risk entries.
