# 📊 EasyStockTrader — IWT-Disciplined Trading Terminal

**Defined-risk SPX income + IWT framework implementation. Small-account survivability.**  
Educational simulation — does not execute trades. For trade planning and discipline enforcement.

[![Live App](https://img.shields.io/badge/Live%20App-easystocktrader.streamlit.app-FF4B4B?logo=streamlit)](https://easystocktrader.streamlit.app)
[![IWT Framework](https://img.shields.io/badge/Framework-Invest%20With%20Teri%20(IWT)-6c8cff)](https://investwithteri.com)
[![License](https://img.shields.io/badge/license-CC%20BY--NC--ND%204.0-red)](LICENSE)
[![Status](https://img.shields.io/badge/status-Simulation%20Only-orange)](#disclaimer)
[![Version](https://img.shields.io/badge/version-v4.0.0--iwt--complete-blue)](#changelog)

> ⚠️ **SIMULATION ONLY** — Analyses signals, scores setups. Does not execute trades or manage real funds. Not financial advice.

---

## IWT Method — What This App Implements

EasyStockTrader is structured around Teri Ijeoma's 7-step Invest With Teri (IWT) trading methodology, with each lesson surfaced as a concrete UI gate or calculation:

| IWT Lesson | Implementation |
|------------|----------------|
| **Lesson 1** — Pick good companies, not just good stories | Company quality ≠ trade quality: IWT scorecard separates fundamental score from entry timing |
| **Lesson 2** — Never trade without a stop loss | Structural stop (0.5× ATR below support) — not arbitrary % |
| **Lesson 3** — Don't chase plays | Late-entry gate: flags when price is >2% above the level |
| **Lesson 5** — 1% of capital as the daily target | Session P&L tracker shows progress vs 1% goal in real time |
| **Lesson 6** — Chart reading: candlesticks, formations, buyer/seller zones | Buyers zone / sellers zone displayed with every ticker scan |
| **Lesson 7** — Emotional discipline through systems | Hard NO-TRADE gate blocks entries when conditions fail — system decides, not emotion |
| **Lesson 8** — Trade a curated watchlist | IWT 30-stock watchlist enforcement: off-watchlist tickers trigger a warning and 2-week paper period |
| **Lesson 9** — Use a simulator before going live | Paper trading gate for Beginners — experience level blocks real trading until confidence is built |
| **Lesson 10** — Define risk/reward BEFORE entry | IWT Trade Setup Card: R:R, entry zone, stop, target, shares, max loss displayed before any action |

---

## Key Features

### IWT Trade Setup Card (new in v4)
Every trade now generates a structured card BEFORE execution:

```
ENTRY ZONE:  $95.00–$95.48  (at support, not above it)
STOP LOSS:   $94.25         (0.5× ATR below support — structural)
TARGET:      $115.00        (resistance = natural seller zone)
R:R RATIO:   4.00:1         ✅ (minimum 2:1 required)
SHARES:      10             (so max loss = $100 = 1% of $10,000)
PROFIT:      $1,980         (198% of daily 1% goal)
```

### Watchlist Discipline Gate
Warns when trading off the IWT 30-stock watchlist. Deep familiarity
with how a stock behaves is worth more than scanning 500 tickers.

### Session P&L as % of Capital
Real-time tracker showing daily P&L vs the 1% capital target ($100 on $10k).
Stop-trading signals fire automatically at:
- ✅ Daily goal met → close platform, protect the win
- 🛑 2× daily goal in losses → stand aside, review tomorrow

### No-Trade Hard Gate
System refuses entries when conditions fail:
- VIX > 35 (crisis mode)
- Major event within 48h
- R:R < 1.5:1
- Price chasing (>2% from level)

### SPX Income Engine
- Live strikes via yfinance + BSM (scipy)
- IWT $0.50/share credit minimum filter
- Tradier broker integration for live execution
- At-expiry P&L payoff diagram

---

## Architecture

```
app.py (5,800+ lines)
├── IWT Trade Setup Card      ← R:R, stop, target, shares, 1% goal
├── Watchlist Discipline Gate ← 30-stock IWT universe check
├── Paper Trading Gate        ← experience-level enforcement
├── Capital Progress Tracker  ← 1% daily goal, loss limit
├── No-Trade Hard Gate        ← event risk, VIX, R:R blockers
├── Market Weather Engine     ← regime classifier (RISK-ON/OFF)
├── IWT Universe Scanner      ← 34-stock batch scan
├── SPX Daily Plan            ← live BSM strikes
├── Backtest Engine           ← 1yr real data, credit spread + long call
├── Options Intelligence      ← yfinance/Tradier Greeks, IVR, EM
├── Kelly Criterion           ← position sizing math
├── Futures Calculator        ← CME spec database
├── Platform Translator       ← TOS/Fidelity/IBKR/TastyTrade strings
└── Research Tab              ← 6 analyst skills (equity, earnings, sector)

ml_signals.py
└── RandomForest classifier  ← rolling 252d window, 5d fwd return target
```

---

## Quick Start

```bash
git clone https://github.com/gabrielmahia/quantum-maestro
cd quantum-maestro
pip install -r requirements.txt
streamlit run app.py
```

Or open the live app: **easystocktrader.streamlit.app**

---

## Validation

```bash
python -m pytest tests/ -v       # 20+ tests including IWT math verification
ruff check app.py                # lint
python -c "import ast; ast.parse(open('app.py').read())"  # AST gate
```

---

## Framework Credit

The IWT methodology and Invest With Teri system are the work of [Teri Ijeoma](https://investwithteri.com).
This application is an independent educational implementation of publicly documented trading principles.
No affiliation with Invest With Teri LLC.

---

## License & Disclaimer

**CC BY-NC-ND 4.0** — Non-commercial. No derivatives. Attribution required.  
**Not financial advice.** Simulation only. Past performance does not guarantee future results.  
All data from Yahoo Finance (delayed). Options involve risk of loss.

See [SECURITY.md](SECURITY.md) for vulnerability reporting.

---

## 🎼 Quantum Maestro Decision Engine (v5.1 — new page)

The app now ships a second page — **Quantum Maestro Engine** — implementing the full decision stack upstream of any trade idea:

**Regime Engine** (Offensive/Neutral/Defensive/Lockdown, auto-filled from live data, manually overridable) → **Seven-Agent Pre-Trade Stack** (any agent vetoes, none initiates) → **Deterministic Risk Engine** (11 hard rules, no override path: 0DTE ban, defined-risk only, 1% × regime multiplier, 3% heat cap, no repair trades, event blackout, cooldown, −2% daily circuit breaker) → **Sizing** (fixed-risk IWT + quarter-Kelly locked until ≥20 journaled trades) → **Decision Journal** (SQLite; logs trades, no-trades AND vetoes; expectancy + equity curve) → **Promotion Gate** (SHADOW→LIVE only after ≥50 decisions, ≥30 closed trades, positive net expectancy, zero rule violations, ≥60 days).

Plus **Playbooks** (FOMC, CPI, geopolitics, earnings clusters, summer liquidity, drawdown recovery) and a **global Masters Library** (Ijeoma, PTJ, Buffett, Dalio, Soros, Simons, Thorp, Kostolany, Darvas, Jhunjhunwala, Li Lu, Kotegawa; Livermore as the cautionary null hypothesis).

Architecture and roadmap: [`docs/QUANTUM_MAESTRO_ARCHITECTURE.md`](docs/QUANTUM_MAESTRO_ARCHITECTURE.md). Package: [`qm/`](qm/) — pure-Python, CI-tested (`tests/test_qm.py`). Journal state is local (`data/*.db`, gitignored); export CSV from the Journal page on Streamlit Cloud (ephemeral storage).

> Still **SIMULATION / SHADOW ONLY**. The engine's core doctrine: *determine whether to trade before determining what to trade.* A prevented bad trade is a positive-expectancy event — and it gets journaled as one.

### 🎫 Paper Desk — Tradier Sandbox (page 8)

Sandbox-only execution: the adapter (`qm/broker_tradier.py`) pins its endpoint to `sandbox.tradier.com` — the production host does not exist in the codebase (CI-enforced test). Orders are **gated in code**: `place_order()` requires an approved `RiskVerdict`; a vetoed or missing verdict raises before any HTTP call. Vetoed submissions are journaled as `VETOED` decisions.

Setup (Streamlit Cloud → App → Settings → Secrets):
```toml
TRADIER_SANDBOX_TOKEN = "your-sandbox-key"
TRADIER_SANDBOX_ACCOUNT = "VAxxxxxxxx"   # optional; auto-resolved from profile if omitted
```
Sandbox constraints (per Tradier): $100k virtual funds, 15-minute delayed data, no streaming. If you regenerate the sandbox key, update the secret — nothing in the repo changes.
