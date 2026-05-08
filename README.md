# 📊 EasyStockTrader — SPX Income Engine

**Defined-risk vertical credit spread planning terminal for SPX. IWT-discipline, small-account survivability.**  
Simulation only — does not execute trades or connect to a broker. For education and trade planning.

[![Live App](https://img.shields.io/badge/Live%20App-easystocktrader.streamlit.app-FF4B4B?logo=streamlit)](https://easystocktrader.streamlit.app)
[![Live Data](https://img.shields.io/badge/Live%20Data-Yahoo%20Finance%20%C2%B7%20open.er-api.com%20%C2%B7%20NDMA%20%C2%B7%20World%20Bank-00b4d8)](#features)
[![License](https://img.shields.io/badge/license-CC%20BY--NC--ND%204.0-red)](LICENSE)
[![Status](https://img.shields.io/badge/status-Simulation%20Only-orange)](#disclaimer)
[![Version](https://img.shields.io/badge/version-v2.0.0--spx--income-blue)](#changelog)

> **EasyStockTrader** — SPX cash-settled defined-risk income, Teri Ijeoma-style discipline.

> ⚠️ **SIMULATION ONLY** — This tool analyses market signals and scores trade setups. It does not execute trades, connect to any broker, or manage real funds. Not financial advice.

---

## What it does

EasyStockTrader implements a structured workflow for SPX vertical credit spread planning — the primary use case being **defined-risk premium selling on the S&P 500** using cash-settled options. The V2 engine formalises five practitioner frameworks into a unified decision scorecard:

**Vertical Credit Spread Math** — Full defined-risk accounting: max profit, max loss, breakeven, position sizing, and commission load per contract. Credit efficiency scoring replaces stock-style reward/risk.

**Expected Move Engine** — Computes the approximate expected move for SPX (EM ≈ SPX × IV × √(DTE/365)) and flags whether the chosen short strike sits inside or outside the EM boundary.

**POP / Touch-Risk Approximation** — Short-delta input drives probability-of-profit (≈ 1 − δ) and probability-of-touch (≈ 2δ). Scores penalise setups with POP below 65%.

**Event Risk Blockers** — Checkboxes flag major macro events within 48 hours. The scoring engine penalises new premium selling near catalysts and issues strong warnings when the position would survive through event risk.

**Mon–Thu Trade Planner** — Weekly trade-day ranking engine considers VIX regime, passive-flow windows, event days, and remaining PDT budget to rank each weekday as Best / Conditional / Avoid.

**Passive Flow Windows** — Detects the 1st–5th of the month when institutional passive funds (pension, 401k) deploy capital systematically, creating observable volume anomalies in broad market ETFs.

**Fed/Rates Filter** — Penalises growth/directional trades when the 10-year Treasury yield rises meaningfully above a recent baseline, implementing rate-sensitivity impact on equity multiples.

**IWT 7-Step Verification** — Formalises Teri Ijeoma's trade entry checklist as an algorithmic scorecard. Setups scoring 7–8/8 qualify; below 5 are blocked.

**VIX Regime Sizing** — Adjusts position size via Kelly Criterion and volatility scaling. High-volatility regimes automatically reduce size or block new entries.

---

## Core capabilities

| Feature | What it does |
|---------|-------------|
| **SPX Vertical Spread Math** | Max profit/loss, breakeven, credit efficiency scoring |
| **Expected Move Engine** | EM approximation, short-strike placement check |
| **POP / Touch Risk** | Delta-based probability scoring for premium sellers |
| **Event Risk Blocker** | Macro-event proximity penalty and hold-through warning |
| **Mon–Thu Trade Planner** | Daily trade-day ranking with VIX, flow, and PDT context |
| **PDT / Day-Trade Budget** | Legacy PDT · New Intraday Margin · Conservative modes |
| **Macro Audit** | VIX level, yield curve, dollar index, DAX/Nikkei correlation |
| **Pattern Detection** | Support/resistance, divergences, gap analysis |
| **Position Sizing** | Kelly Criterion + volatility scaling + slippage/commission |
| **IWT Scorecard** | 8-point setup verification — GREEN/YELLOW/RED verdict |
| **Portfolio Guard** | Blocks new entries when daily P&L goal is met |
| **Journal 2.0** | CSV audit trail for paper and live trades |
| **NSE Coverage** | Safaricom, Equity Bank, KCB, EABL, and others |
| **WarrenAI Export** | Copy block with full context for AI-assisted trade review |

---

## Strategy modes

| Mode | Best for |
|------|---------|
| **Income – SPX Vertical Credit Spread** *(default)* | Defined-risk put/call credit spreads on SPX |
| Long Stock / ETF | Directional equity position sizing |
| Cash-Secured Put | CSP assignment exposure & income math |
| Covered Call | Covered call income calculation |
| NSE Equities | Nairobi Securities Exchange trade scoring |

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

A disclaimer screen appears on launch — accept to proceed. All market data is fetched live from Yahoo Finance.

---

## Methodology

**IWT (Invest With Teri)** — Teri Ijeoma's 7-step trade verification system, formalised for algorithmic scoring. Credit: [investwithteri.com](https://www.investwithteri.com)

**Vertical Credit Spread Discipline** — Defined-risk SPX premium selling: short strike outside expected move, POP ≥ 65%, credit ≥ 25% of spread width as preferred threshold. Based on IWT income strategy principles.

**Passive Inflow Windows** — Based on documented end-of-month / beginning-of-month institutional rebalancing. Reference: Kamstra et al. (2017) on calendar-based institutional flows.

**Fed/Rates Filter** — Generalised implementation of yield-curve sensitivity in equity valuations. Rising long-end yields compress growth multiples and directional equity risk/reward.

**PDT Framework** — Conservative treatment of Pattern Day Trader rules for sub-$25k accounts. Three modes: Legacy PDT (3 day trades in rolling 5 business days), New Intraday Margin (broker-permissioned relaxation), and Conservative (treat all intraday exits as scarce).

---

## Changelog

### v2.0.0 — SPX Income Engine (2026-05-08)
- SPX verticals set as **default strategy mode**
- Expected move engine (IV × √(DTE/365) approximation)
- POP and touch-risk from short delta
- Event risk blocker with hold-through penalty
- Mon–Thu trade planner with VIX/flow/PDT ranking
- PDT framework selector (Legacy / New Intraday / Conservative)
- WarrenAI export expanded with EM, POP, event risk, PDT plan
- Fed/rates filter generalised (removed hardcoded "Warsh" label)
- Kenya macro GDP/CPI display bug fixed
- Tests: 12/12 passing

### v1.1.0 — SPX Verticals Patch (2026-05-08)
- Defined-risk vertical spread calculator
- Credit efficiency scoring (replaces directional R/R for spreads)
- Cash-secured put math: assignment vs technical stop separated
- PDT day-trade budget guidance (conservative baseline)

### v1.0.0 — Initial release (2026-05-01)
- IWT 7-step scorecard, VIX regime sizing, NSE coverage
- Kenya macro dashboard, Journal 2.0, Portfolio Guard

---

## Important limitations

- Does not fetch live option-chain prices, IV rank, Greeks, or POP from a broker
- IV/VIX inputs must be entered manually from broker / InvestingPro / WarrenAI
- PDT/intraday-margin section is intentionally conservative — broker implementation varies
- All outputs are educational simulations; no trade execution

---

## Disclaimer

This software is for educational use and trade simulation only. It does not constitute financial advice. Trading involves substantial risk of loss — past performance does not guarantee future results. Market data is provided by Yahoo Finance and may have delays or inaccuracies. Always validate signals with your own analysis and consult a qualified financial advisor before risking capital.

The PDT rules, intraday margin frameworks, and trade-budget guidance are approximations for planning purposes only. Your broker's rules govern — always confirm with your broker before trading.

---

## License

CC BY-NC-ND 4.0 — personal and educational use permitted. Commercial use prohibited. Contact: [contact@aikungfu.dev](mailto:contact@aikungfu.dev)

Copyright © 2026 Gabriel Mahia. All Rights Reserved.

---

## Portfolio

Part of a suite of civic and community tools built by [Gabriel Mahia](https://github.com/gabrielmahia):

| App | What it does |
|-----|-------------|
| [🌊 Mafuriko](https://floodwatch-kenya.streamlit.app) | Flood risk & policy enforcement tracker — Kenya |
| [💧 WapiMaji](https://wapimaji.streamlit.app) | Water stress & drought intelligence — 47 counties |
| [🏛️ Macho ya Wananchi](https://macho-ya-wananchi.streamlit.app) | MP voting records, CDF spending, bill tracker |
| [🌾 JuaMazao](https://juamazao.streamlit.app) | Live food price intelligence for smallholders |
| [🏦 ChaguaSacco](https://chaguasacco.streamlit.app) | Compare Kenya SACCOs on dividends & loan rates |
| [🛡️ Hesabu](https://hesabu.streamlit.app) | County budget absorption tracker |
| [🗺️ Hifadhi](https://hifadhi.streamlit.app) | Riparian encroachment & Water Act compliance map |
| [💰 Hela](https://helaismoney.streamlit.app) | Chama management for the 21st century |
| [💸 Peleka](https://tumapesa.streamlit.app) | True cost remittance comparison — diaspora to Kenya |
| [📊 EasyStockTrader](https://easystocktrader.streamlit.app) | SPX income engine — defined-risk vertical spreads |
| [🦁 Dagoretti](https://dagoretti-high-school-community-app.streamlit.app) | Alumni atlas & community hub for Dagoretti High |
| [⛪ Jumuia](https://jumuia.streamlit.app) | Catholic parish tools — church finder, pastoral care |
