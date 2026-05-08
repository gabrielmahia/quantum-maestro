# Quantum Maestro Patch Notes — SPX Vertical Income Math

## Purpose
This patch updates the Streamlit app from primarily stock-directional math toward the user's stated strategy: SPX cash-settled, defined-risk vertical credit spreads for income, consistent with risk-first IWT-style discipline.

## Key changes

### 1. Added defined-risk vertical spread calculator
New helper functions:
- `calc_vertical_credit_spread(...)`
- `contracts_for_defined_risk(...)`
- `pdt_guidance(...)`

Vertical spread math now uses:
- Spread width = `abs(short_strike - long_strike)`
- Max profit per contract = `credit * 100`
- Max loss per contract = `(spread_width - credit) * 100`
- Breakeven:
  - Put credit spread = `short_strike - credit`
  - Call credit spread = `short_strike + credit`
- Commissions = two legs per contract

### 2. Added strategy mode: Income (SPX Vertical Credit Spread)
The sidebar now supports:
- Put or call credit spread
- Short strike
- Long strike
- Net credit
- DTE

The app validates correct strike order:
- Put credit spread: short strike above long strike
- Call credit spread: short strike below long strike

### 3. Corrected score logic for vertical spreads
Stock-style reward/risk does not fit short verticals. Instead, verticals now score the risk/reward component by credit efficiency:
- Preferred: credit >= 25% of spread width
- Acceptable: credit >= 20% of spread width
- Thin: credit < 20% of spread width

### 4. Added conservative small-account day-trade budget guidance
The app now includes:
- Account type
- Day trades used in rolling 5 business days
- Planned same-day exit checkbox

This does not replace broker rules, but it encourages preserving limited day trades for A+ setups.

### 5. Improved cash-secured put math
Cash-secured puts now display assignment/cash-secured exposure separately from technical stop risk.

### 6. Fixed Kenya macro display bug
The prior display could show inflation as GDP. The app now displays:
- GDP per capita
- CPI inflation
using the correct World Bank indicator codes.

### 7. Updated close-position logic for SPX verticals
For vertical spreads, enter the closing debit / remaining spread value in index points when closing the position. Example: if you sold a spread for 1.00 and buy it back for 0.35, enter `0.35`.

## Validation
- Python compile check passed.
- Test suite passed: 12/12 tests.

## Important limitation
This app remains educational/simulation-only. It does not fetch live option-chain prices, IV rank, Greeks, or POP directly. Those should be entered manually from the broker/InvestingPro/WarrenAI until an option-chain data provider is integrated.
