# AGENTS.md — EasyStockTrader (Quantum Maestro)

Context for AI coding agents (Cursor, GitHub Copilot, Claude Code, etc.)

---

## What this is

**EasyStockTrader v4.0 — IWT-Disciplined Trading Terminal**

A Streamlit-based educational trading simulator implementing the Invest With Teri (IWT) methodology.
Primary use case: SPX cash-settled vertical credit spread planning with full discipline enforcement.
Secondary modes: long stock/ETF, long options, short sell, NSE equities, futures planning.

**Not a broker — does not execute trades.**  
**Live app:** https://easystocktrader.streamlit.app

---

## Repository structure

```
app.py                          ← Single-file Streamlit app (~6,300 lines)
ml_signals.py                   ← RandomForest signal layer (rolling 252d window)
requirements.txt                ← Deps: streamlit, yfinance, scipy, sklearn, pytz
tests/
  test_smoke.py                 ← 20+ tests: IWT math, constants, smoke
  test_live_data.py             ← Indicator and spread math validation
docs/methodology/               ← Trading philosophy and math documentation
.streamlit/config.toml          ← Dark theme, toolbar minimal
PATCH_NOTES_V4.md               ← Changelog
```

---

## Key function inventory (v4.0 complete)

### IWT Core (Lessons 2,3,5,8,9,10)
| Function | Lesson | Purpose |
|----------|--------|---------|
| `calc_iwt_trade_setup(price, support, resistance, atr, capital)` | 2,3,10 | R:R card: entry zone, stop (structural ATR), target, shares, dollar risk |
| `render_iwt_setup_card(ticker, price, ...)` | 10 | Renders the setup card to Streamlit UI |
| `check_late_entry(price, support, threshold_pct=2.0)` | 3 | Flags when price is >2% from the level (chasing) |
| `calc_capital_progress(daily_pnl, capital, monthly_goal)` | 5 | 1% daily goal tracker, stop-trading signals |
| `check_ticker_vs_iwt_watchlist(ticker)` | 8 | 30-stock IWT watchlist gate |
| `paper_trading_gate(experience_level)` | 9 | Beginner → paper only gate |
| `IWT_WATCHLIST` | 8 | 30-stock core trading universe |

### IWT VIP Curriculum
| Function | Source | Purpose |
|----------|--------|---------|
| `detect_gap_trap(gap_pct, rvol, trend, bb_squeeze, globex_high, globex_low)` | VIP 2023 | Gap trap probability and safe entry rule |
| `levels_plus_em_strike_placement(spx, iv_pct, dte, support, resistance)` | VIP May 2022 | Double-confirmation: EM AND level both must clear |
| `short_or_not_score(trend, vix, macro, metrics, rsi, rvol)` | VIP 2022-23 | 5-condition short framework |
| `calc_covered_call_yield(stock_price, call_strike, call_premium, dte)` | VIP 2020 | Annualized yield + IWT rules |
| `calc_six_figure_plan(monthly_goal, account_size, win_rate)` | Coaching 1/9/2023 | Backward-plan from income goal |
| `options_playbook_router(vix, ivr, trend, pre_event, post_event_selloff)` | Options 101 | Maps conditions to structures |
| `gap_trade_playbook(gap_pct, gap_type, rvol, trend)` | Gaps coaching | Fade/ride/stop-level strategies |
| `troubleshoot_trading(responses)` | Coaching 1/4/2023 | 7-symptom diagnostic + IWT fixes |
| `calc_cc_cost_basis_reducer(stock_price, purchase_price, calls_sold)` | Options 101 | Tracks cost basis reduction over time |

### Market Intelligence
| Function | Purpose |
|----------|---------|
| `compute_market_weather(macro)` | RISK-ON/NEUTRAL/CAUTIOUS/OFF regime |
| `compute_market_health_score()` | 0-100 breadth score (RSP vs SPY, sectors, VIX, IWM, yield curve) |
| `compute_no_trade_gate(macro, metrics, strategy)` | HARD_NO/SOFT_NO/CONDITIONAL/YES |
| `classify_market_regime(macro, metrics)` | Full regime classification |
| `compute_trade_grade(iwt_score, regime_weight, event_risk, vix)` | A-F trade grade |
| `generate_spx_daily_plan(dte_target, min_credit)` | Live BSM strikes with IWT $0.50 filter |
| `batch_scan_teri_universe(universe_list, top_n)` | 34-stock scan with IWT scorecard |
| `run_global_scan()` | Asia → Europe → US pre-market data |
| `classify_gap_type(gap_pct, rvol, trend, bb_width_ratio)` | Common/Breakaway/Runaway/Exhaustion |

### Options & Spreads
| Function | Purpose |
|----------|---------|
| `calc_vertical_credit_spread(short_strike, long_strike, credit)` | Full spread P&L math |
| `calc_long_option_iwt(underlying_price, strike, premium_paid, delta, dte)` | 60+ DTE DITM analysis |
| `classify_iv_environment(ivr, vix_level)` | Buy vs sell premium decision |
| `bsm_greeks(S, K, T, r, sigma, flag)` | Exact BSM Greeks (scipy) |
| `live_options_greeks_v2(ticker, dte_target)` | Tradier-first / yfinance fallback |
| `theta_decay_profile(credit, dte_at_entry, dte_remaining)` | √(DTE remaining) decay model |
| `trade_management_engine(structure, entry_value, current_value, dte_remaining)` | 50% rule + gamma rules |
| `kelly_and_ruin(win_rate, avg_win_R, avg_loss_R)` | Kelly + risk-of-ruin |
| `plain_spread_explanation(...)` | TOS/Fidelity/IBKR/TastyTrade order strings |

### Backtest Engine (v7)
| Function | Purpose |
|----------|---------|
| `load_backtest_data()` | 1-year SPY/VIX/TNX real data + rolling IVR/SMA |
| `backtest_credit_spread(df, ...)` | Weekly put credit spread backtest |
| `backtest_long_call(df, ...)` | IWT DITM long call backtest |
| `compute_backtest_stats(trades_df, initial_capital, strategy_name)` | Win rate, EV, drawdown, profit factor |

### Broker Integration (Tradier)
| Function | Purpose |
|----------|---------|
| `tdr_get_account_id()` | Primary account number |
| `tdr_get_balances(account_id)` | Equity, cash, buying power, PDT flag |
| `tdr_get_positions(account_id)` | Open positions |
| `tdr_get_orders(account_id)` | Recent orders |
| `tdr_place_equity_order(...)` | Stock/ETF order execution |
| `tdr_place_option_order(...)` | Single-leg option order |
| `tdr_place_spread_order(...)` | Multi-leg spread order |
| `build_option_symbol(underlying, expiry_str, option_type, strike)` | OCC symbol builder |
| `trade_safety_check(balances, max_loss, qty)` | Pre-trade BP + Kelly gate |

### Research (6 Analyst Skills)
Accessible in Tab 5 (🔬 Research):
- Daily Market Brief, Equity Research, Earnings Review
- Sector Comparison, Comparable Companies, Pre-Earnings Prep

---

## App UI structure (5 tabs)

| Tab | Label | Key content |
|-----|-------|-------------|
| T1 | 🎯 Setup | Experience level → Account/risk → Ticker + watchlist gate + paper gate → IWT Scorecard → GO/NO-GO verdict |
| T2 | 🔍 Scan | Universe scan (34 stocks) → SPX Daily Plan → Market Health Score |
| T3 | 📊 Trade | IWT Setup Card → Credit spread builder → Payoff diagram → Signal details |
| T4 | 📓 Review | Journal → Capital progress (1% goal) → Income blueprint → Troubleshoot |
| T5 | 🔬 Research | 6 analyst skills → Structured reports with confidence labels |
| PW | ⚡ Power Tools | Options playbook · Backtest UI · Covered call yield · Macro brief · Account A/B · Live positions |

---

## Critical rules — do not change

### 1. pandas_ta is removed permanently
`_calc_indicators()` is a pure numpy/pandas implementation. Do not re-add pandas_ta.

### 2. Trust integrity banners
`SIMULATION ONLY` markers are required disclosures. Do not remove.

### 3. IWT function contracts
- `calc_iwt_trade_setup`: stop must always be below support for LONG (ATR-structural)
- `render_iwt_setup_card`: must show R:R before any execution UI
- `calc_capital_progress`: daily loss limit = 2× daily goal (not arbitrary)
- `check_late_entry`: threshold 2% default — do not lower

### 4. Mobile-first UI (permanent)
44px tap targets, 16px body min, single-column base, desktop is additive.

### 5. AST gate is hard (CI enforces it)
`python -c "import ast; ast.parse(open('app.py').read())"` must pass before every push.

### 6. Tradier API is form-encoded POST
`application/x-www-form-urlencoded` — not JSON. Multi-leg: `class=multileg`.

### 7. yfinance MultiIndex
Always use `df['Close'].iloc[:,0]` check via isinstance(df.columns, pd.MultiIndex).

### 8. No invented data anywhere
All math uses real formulas. All data labeled: REAL (yfinance) / APPROXIMATION (BSM) / SIMULATION.

---

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Tests

```bash
pytest tests/ -v   # 20+ tests — all must pass before push
ruff check app.py ml_signals.py
python -c "import ast; ast.parse(open('app.py').read())"
```

## License

CC BY-NC-ND 4.0 — Non-commercial. No derivatives. Attribution required.
contact@aikungfu.dev | © 2026 Gabriel Mahia
