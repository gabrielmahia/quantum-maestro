# EasyStockTrader / Quantum Maestro — SPX Income Engine V2 Patch

## Purpose
Rebuilt the app toward the workflow discussed: SPX cash-settled vertical credit spreads for income, Teri Ijeoma-style discipline, small-account survivability, and Mon–Thu trade planning under legacy/new PDT-style constraints.

## What changed

### 1. SPX verticals are now the default strategy
- The strategy selector now defaults to **Income (SPX Vertical Credit Spread)**.
- This reflects the actual trading use case: defined-risk premium selling rather than directional stock buys.

### 2. Better vertical spread math
- Keeps defined-risk math:
  - Max profit = credit × 100 × contracts
  - Max loss = (spread width − credit) × 100 × contracts
  - Breakeven = short strike − credit for puts, short strike + credit for calls
- Contracts are rounded down so theoretical max loss stays under the selected trade-risk and portfolio-risk budgets.

### 3. Expected move engine
- Added SPX reference price, IV/VIX proxy, DTE, and short-delta inputs.
- Calculates approximate expected move:
  - Expected Move ≈ SPX × IV × sqrt(DTE / 365)
- Flags whether the short strike is inside or outside expected move.
- Penalizes short strikes inside expected move.

### 4. POP / touch-risk approximation
- Added short-strike delta input.
- Approximate POP = 1 − delta.
- Approximate probability of touch ≈ 2 × delta.
- Penalizes POP below 65%.

### 5. Event risk blockers
- Added event-risk checkboxes:
  - Major event within 48h?
  - Would hold through event?
- Penalizes new premium selling near macro event risk.
- Strongly warns when holding short premium through event risk.

### 6. Day-trade / PDT-style trade budget
- Added **Day-Trade Rule Framework** selector:
  - Legacy PDT
  - New Intraday Margin
  - Broker Unknown / Conservative
- Added account-type selector and rolling 5-business-day trade tracker.
- The app now treats small-account day trades as scarce bullets unless the user selects new intraday margin and broker permission is assumed.

### 7. Mon–Thu trade planner
- Added preferred trade-day selector for Monday–Thursday.
- Added major event-day selector for weekly planning.
- Ranks Mon–Thu as:
  - Best Trade Day
  - Conditional
  - Avoid
- Ranking considers VIX regime, passive-flow windows, event risk, and remaining trade budget.

### 8. Fed/rates filter generalized
- Replaced the old “Warsh” language with a generalized **Fed/Rates** penalty.
- Rising 10Y yield now penalizes growth-directional trades without hardcoding one person/regime.

### 9. WarrenAI export improved
The copy block now includes:
- PDT framework
- day-trade plan
- expected move
- EM status
- approximate POP
- event-risk flag
- vertical spread max loss

## Tests
- `python -m pytest -q`
- Result: 12 passed

## Notes
This remains an educational simulation and does not execute trades or connect to a broker. The PDT/new intraday-margin section is intentionally conservative because broker implementation varies.
