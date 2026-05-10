# Open-Source Data Resources — Running Thread

This document tracks all free / open-source data sources discovered during development.
When a quality resource is found, it gets evaluated and integrated automatically.
**Last updated: 2026-05-10**

---

## Currently Integrated

### yfinance (yfinance.readthedocs.io)
**Status: LIVE — used in production**
- Options chains with **implied volatility** (from real market bid/ask prices)
- Price history for any ticker (1 day to max)
- Macroeconomic proxies: ^VIX, ^TNX, DX-Y.NYB, ES=F, GC=F, CL=F, NG=F
- No API key required. Delayed data (15-20 min for US markets).
- Limitation: No historical IV snapshots. Current chain only.

### scipy.stats.norm (scipy.org)
**Status: LIVE — BSM Greeks engine**
- Computes exact Black-Scholes-Merton Greeks (δ, Γ, Θ, ν, ρ)
- Inputs: spot, strike, time, risk-free rate, implied vol
- Already in requirements.txt. No new dependency.
- Formula: exact (not approximated). Matches textbook BSM to 6+ decimal places.

### NDMA Kenya (ndma.go.ke)
**Status: LIVE — Kenya civic data**
- Drought alerts and commodity pressure signals
- Used in Kenya macro intelligence overlay

### World Bank API (api.worldbank.org)
**Status: LIVE — Kenya macro**
- GDP per capita, CPI inflation via open REST API
- No API key required

---

## Evaluated — Conditionally Available

### CBOE Delayed Quotes API (cdn.cboe.com)
**Status: INTERMITTENT — 503 errors on some days**
- Endpoint: `https://cdn.cboe.com/api/global/delayed_quotes/options/SPY.json`
- When live: full options chain with bid/ask for SPY, SPX, major ETFs
- Has Greeks directly in some responses
- **Integration plan**: Add as fallback/enrichment when available. Fall back to yfinance+BSM when 503.

### Tradier (tradier.com)
**Status: REQUIRES FREE API KEY**
- Free production account available (paper trading tier)
- Has live options chains with actual Greeks (delta, gamma, theta, vega)
- Rate limit: 1 req/sec on free tier
- **Integration plan**: When user provides Tradier API key, use for real-time Greeks (much better quality).
- Signup: tradier.com/create/account

### Alpha Vantage (alphavantage.co)
**Status: REQUIRES FREE API KEY**
- Free tier: 25 API calls/day, 5/minute
- Has: realtime stock quotes, fundamental data, some economic indicators
- Does NOT have options chains directly
- **Integration plan**: Optional enrichment for sector/fundamental data

### polygon.io
**Status: REQUIRES FREE API KEY (then paid for options)**
- Options chains available on Starter plan ($29/mo)
- Delayed data on free tier (no options)
- **Integration plan**: Paid tier only. Not prioritised.

---

## Evaluated — Not Used

### Interactive Brokers (IBKR)
- Best-in-class real-time Greeks and IV surface
- Requires brokerage account + ibapi Python library
- **Decision**: Not integrated in Streamlit Cloud (requires local IB Gateway/TWS running)
- **For local use**: ibapi is free once you have an account

### OpenBB Terminal (github.com/OpenBB-finance/OpenBBTerminal)
- Open-source financial data aggregator
- Has CBOE options data provider
- Size: ~200MB install
- **Decision**: Too heavy for Streamlit Cloud free tier. Viable for self-hosted deployments.

### ORATS (orats.com)
- Highest quality historical IV and Greeks data
- Cost: $200+/month
- **Decision**: Paid. Out of scope for this app.

### OptionStack (optionstack.com)
- Historical backtesting with real options prices
- Cost: $50+/month
- **Decision**: Paid. Suitable for strategy backtesting engine (future feature).

### Barchart (barchart.com)
- Has free delayed IV data on web
- API: paid tier only
- **Decision**: Web scraping fragile. Not integrated.

---

## IVR Proxy Methodology

**For SPX/VIX-based instruments (exact):**
```
IVR = (VIX_current - VIX_52w_low) / (VIX_52w_high - VIX_52w_low) × 100
```
VIX IS SPX's 30-day implied volatility — this formula is exact for SPX options.

**For individual stocks (proxy — clearly labelled):**
```
IVR_proxy = (ATM_IV_current - RealizedVol_52w_low) / (RealizedVol_52w_high - RealizedVol_52w_low) × 100
```
RV (realized vol) used as a proxy for historical IV because:
- IV typically tracks RV + a 2-3 percentage point risk premium
- Free historical IV requires paid sources (ORATS, OptionStack, CBOE LiveVol)
- This proxy correctly identifies HIGH vs LOW IV environments in most market conditions
- Clearly labelled as approximation in all UI displays

**To get true IVR for equities (future upgrade):**
1. Connect Tradier free API (provides current and some historical IV)
2. Store daily ATM IV snapshots locally after user scans (builds up history over time)
3. Use stored history for IVR (self-builds accuracy over weeks)

---

## Architecture Principle

> "If quality open-source or free data exists anywhere in the world, find it and use it.
> Never make up data. Always label what's real vs approximated."
> — Project principle

When a new free source is discovered:
1. Test quality vs known benchmarks
2. Document here
3. Integrate if quality is acceptable
4. Add graceful fallback if the source is intermittent
5. Label data provenance in all UI displays

---

## Next Sources to Investigate

- [ ] **quant.stackexchange.com datasets** — community-maintained financial datasets
- [ ] **Fed FRED API (fred.stlouisfed.org)** — macro data, Treasury yields, credit spreads. Free.
- [ ] **SEC EDGAR API (data.sec.gov)** — fundamental data, 10-K/10-Q filings. Free.
- [ ] **finviz.com** — screener data (options volume, IV percentile on web). Scrape-fragile.
- [ ] **Tastytrade API** — paper account provides real-time Greeks. Pending free tier confirmation.
- [ ] **Self-building IV history** — store daily ATM IV from yfinance chains to build own IVR database over time

---
*Maintained automatically as part of the Quantum Maestro development thread.*
*contact@aikungfu.dev | github.com/gabrielmahia*
