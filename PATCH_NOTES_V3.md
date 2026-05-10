# EasyStockTrader / Quantum Maestro — V3 Patch Notes

## Release: V3.0.0 — IWT Full Playbook + High-Quality Algorithm Integration

### Purpose
V3 incorporates Teri Ijeoma's complete options trading playbook — not just the income/spread side,
but also her core teaching that long options (60+ DTE, DITM) are the correct vehicle for capturing
significant directional market moves when IV is low. Combined with institutional algorithm best
practices from TastyTrade, Kelly Criterion research, and volatility regime management.

---

## Architecture: The Two-Weapon System

V3 establishes a clear binary framework driven by IV environment:

```
IV Rank HIGH (>=50%)  →  SELL PREMIUM  →  SPX Vertical Credit Spreads
IV Rank LOW  (<30%)   →  BUY OPTIONS   →  Long Options 60+ DTE, DITM
IV Rank MID  (30-50%) →  SELECTIVE     →  Require A+ setup quality
```

Before V3, the app defaulted to premium selling regardless of IV environment.
The IV Arbiter now makes this the first decision point before any trade is evaluated.

---

## New Functions

### 1. `calc_long_option_iwt()`
IWT-compliant long options calculator.
- Enforces 60 DTE minimum (Teri Ijeoma's hard rule)
- Grades delta selection: DITM (>=0.70) preferred for leverage plays
- Calculates leverage ratio vs owning stock outright
- Profit targets: 50% and 100% of premium paid (IWT exit rules)
- Hard stop: 50% of premium paid (IWT's "never lose more than half" rule)
- Breakeven at expiration for CALL and PUT
- Time value and intrinsic value decomposition
- Daily theta burn approximation

### 2. `classify_iv_environment(ivr, vix_level)`
IV Rank arbiter — the master strategy selector.
- Grade A (IVR >= 50): Sell premium — IV is elevated, options are expensive to buy
- Grade B (IVR 30-49): Selective — require A+ setup quality for either direction
- Grade C (IVR < 30): Buy options (60+ DTE) — IV is cheap, leverage is inexpensive
- VIX overlay adds crisis/complacency context on top of IVR regime

### 3. `theta_decay_profile()`
Theta decay tracker for open credit spread positions.
- Models: value remaining ≈ credit × √(DTE_remaining / DTE_entry)
- Flags 50% profit management trigger (TastyTrade empirical rule)
- Shows daily decay rate and management signal

### 4. `trade_management_engine()`
Systematic position management decision tree for both credit spreads and long options.

Credit spread rules:
- Close at 50% max profit (TastyTrade standard)
- Close if spread doubles in cost (hard loss stop)
- Close gamma risk < 7 DTE if not at 50% profit
- Consider closing at 25% profit with < 21 DTE

Long option rules:
- Close at 100% gain (IWT full target)
- Scale out at 50% gain (IWT partial exit)
- Hard stop at 50% loss (IWT stop rule — no exceptions)
- Roll or close at < 30 DTE with < 20% gain

### 5. `kelly_and_ruin()`
Kelly Criterion position sizing with risk-of-ruin estimate.
- Full Kelly (theoretical maximum growth rate)
- Half Kelly (institutional compromise — 75% of growth, ~50% of drawdown)
- Quarter Kelly (most conservative institutional default)
- Edge per trade and expected monthly return
- Risk-of-ruin approximation using a 50% drawdown threshold

---

## New UI Sections

### IV Environment Arbiter Panel
Prominent expander at top of Market Intelligence Dashboard.
Displays IV regime, recommendation, strategy list, and VIX context.
Updates dynamically based on user-entered IVR.

### IWT Long Option (60+ DTE) Strategy Mode
New entry in strategy selector. Sidebar inputs:
- Direction (CALL/PUT)
- Underlying price
- Strike price
- Premium paid
- Option delta
- DTE (with 60-day enforcement warning)
- Contracts

Output: cost, leverage, profit targets, hard stop, theta burn, breakeven.

### IVR Input (Universal)
New sidebar input: IV Rank (0-100%). Drives the IV Arbiter for all strategies.

### Trade Management Engine
New expandable section. Handles both credit spreads and long options.
Includes integrated theta decay tracker for spread positions.

### Kelly Criterion + Risk-of-Ruin Calculator
New expandable section in performance area.
Win rate slider, avg win/loss inputs, monthly trades.
Outputs Full/Half/Quarter Kelly and ruin probability.

---

## Philosophy Upgrades

### Two-Weapon Framework
Premium selling and option buying are not competing strategies — they are
complementary weapons appropriate for different IV regimes:
- High IV: the seller has the edge (mean reversion in their favour)
- Low IV: the buyer has the edge (cheap leverage on directional moves)

The IV Arbiter implements this framework systematically.

### Decision Hierarchy (V3)
```
1. What is the IV regime? (Arbiter)
2. What is the macro backdrop? (VIX, rates, flows)
3. Do I have permission to trade? (Goal met? Portfolio limit?)
4. Does this setup meet A+ quality?
5. Is IV favourable for this specific structure?
6. What is my defined max loss?
7. What is my management plan before entry?
```

### Teri Ijeoma 60+ DTE Rule — Full Context
"Buying options" in Teri's system means DITM options with long DTE.
This is leverage, not speculation:
- DITM delta (0.70+) makes the option behave like owning 70-80 shares per contract
- 60+ DTE gives the underlying time to make the expected move
- 50% profit target and 50% stop rule makes the risk/reward mechanical and emotion-free

This is explicitly NOT the same as buying cheap OTM options or lottery tickets.

---

## Tests
All existing tests: 12/12 passing
New helper functions: 4/4 test suites passing (logic validation)
AST parse: clean (2,667 lines)

---

## Important Limitations
All new features remain educational simulations. IVR requires manual entry from
broker platform, Barchart, or InvestingPro. The app does not connect to an
options chain or IV data feed.
