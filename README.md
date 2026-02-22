# 🏛️ Macro Risk Desk — Quantum Maestro

**Global macro trading analysis terminal with IWT execution discipline.**  
Simulation only — does not execute trades. For education and trade planning.

[![License](https://img.shields.io/badge/license-CC%20BY--NC--ND%204.0-red)](LICENSE)
[![Status](https://img.shields.io/badge/status-Simulation%20Only-orange)](#disclaimer)
[![Data](https://img.shields.io/badge/data-Yahoo%20Finance%20(live)-blue)](https://finance.yahoo.com)

> ⚠️ **SIMULATION ONLY** — This tool analyses market signals and scores trade setups. It does not execute trades, connect to any broker, or manage real funds. Not financial advice.

---

## What it does

Quantum Maestro implements a structured framework for evaluating trade setups across US equities, ETFs, and Nairobi Securities Exchange stocks. It formalises three practitioner frameworks into a single analytical workflow:

**Passive Flow Windows** — Detects the 1st–5th of the month when institutional passive funds (pension, 401k) deploy capital systematically, creating observable volume anomalies in broad market ETFs.

**Warsh Filter** — Penalises growth/tech positions when the 10-year Treasury yield rises more than 1% above a recent baseline, implementing the rate-sensitivity insight that rising long-end yields compress growth stock multiples disproportionately.

**IWT 7-Step Verification** — Implements Teri Ijeoma's (Invest With Teri) trade entry checklist as an algorithmic scorecard: freshness, time-in-zone, speed-out, volume confirmation, R/R threshold, trend alignment, macro alignment, and regime filter. Setups scoring 7–8/8 qualify; below 5 are blocked.

**VIX Regime Sizing** — Adjusts position size via Kelly Criterion and volatility scaling based on VIX level. High-volatility regimes automatically reduce size or block new entries.

---

## Core capabilities

| Feature | What it does |
|---------|-------------|
| Macro Audit | VIX level, yield curve, dollar index, DAX/Nikkei correlation |
| Pattern Detection | Support/resistance, divergences, gap analysis |
| Position Sizing | Kelly Criterion + volatility scaling + slippage/commission |
| IWT Scorecard | 8-point setup verification — GREEN/YELLOW/RED verdict |
| Portfolio Guard | Blocks new entries when daily P&L goal is met |
| Journal 2.0 | CSV audit trail for paper and live trades |
| NSE Coverage | Safaricom, Equity Bank, KCB, EABL, and 4 others |

---

## Quick start

```bash
pip install -r requirements.txt
streamlit run app.py
```

A disclaimer screen appears on launch — accept to proceed. All market data is fetched live from Yahoo Finance.

---

## Methodology

The three core frameworks implemented are:

**IWT (Invest With Teri)** — Teri Ijeoma's 7-step trade verification system, formalised for algorithmic screening. Credit: [investwithteri.com](https://www.investwithteri.com)

**Passive Inflow Windows** — Based on documented end-of-month / beginning-of-month institutional rebalancing behaviour. Reference: Kamstra et al. (2017) on calendar-based institutional flows.

**Warsh Filter** — Named after former Federal Reserve Governor Kevin Warsh's analysis of yield-curve sensitivity in growth equity valuations.

---

## Disclaimer

This software is for educational use and trade simulation only. It does not constitute financial advice. Trading involves substantial risk of loss — past performance does not guarantee future results. Market data is provided by Yahoo Finance and may have delays or inaccuracies. Always validate signals with your own analysis and consult a qualified financial advisor before risking capital.

---

## License

CC BY-NC-ND 4.0 — personal and educational use permitted. Commercial use prohibited. Contact: [contact@aikungfu.dev](mailto:contact@aikungfu.dev)

Copyright © 2026 Gabriel Mahia. All Rights Reserved.
