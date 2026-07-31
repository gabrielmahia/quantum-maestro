"""
Quantum Maestro — easystocktrader.streamlit.app
================================================
An institutional-grade decision engine for a retail account.

Philosophy: determine WHETHER to trade before WHAT to trade.
Mode: SHADOW by default. Live execution does not exist in this codebase
until the Promotion Gate passes — by design, not by omission.

This tool is decision-support software, not financial advice. All trading
involves risk of loss. The author of a losing trade is always the operator.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import glob
import streamlit as st
import pandas as pd

from qm.config import LIMITS, GATE, DEFAULT_MODE, WATCHLIST_CORE, EVENT_TYPES
from qm import regime as regime_mod
from qm.regime import RegimeInputs, score_regime, try_autofill_inputs
from qm.risk import TradeProposal, evaluate
from qm.sizing import final_size, kelly_fraction
from qm.agents import AGENTS, SCORES, aggregate, select_strategy
from qm.wisdom import MASTERS
from qm import journal

# page config owned by main app.py (this file runs as a Streamlit page)

# ---------------------------------------------------------------- sidebar
st.sidebar.title("🎼 Quantum Maestro")
st.sidebar.caption("Decide whether. Then what. Then how much.")
mode = st.sidebar.radio("Mode", ["SHADOW", "LIVE (locked)"], index=0,
                        help="LIVE unlocks only when the Promotion Gate passes. See Gate page.")
MODE = "SHADOW"
if mode.startswith("LIVE"):
    g = journal.evaluate_gate()
    if g["promotable"]:
        st.sidebar.success("Gate PASSED. Live execution may be wired via a broker adapter — see docs/ARCHITECTURE.md.")
        MODE = "LIVE"
    else:
        st.sidebar.error("Promotion Gate not passed. Mode remains SHADOW.")

equity = st.sidebar.number_input("Account equity ($)", min_value=100.0, value=10000.0, step=500.0)
st.sidebar.divider()
st.sidebar.caption(f"Hard limits (code-locked): {LIMITS.MAX_RISK_PCT_PER_TRADE:.0%}/trade · "
                   f"{LIMITS.MAX_PORTFOLIO_HEAT:.0%} heat · min {LIMITS.MIN_DTE} DTE · "
                   f"max {LIMITS.MAX_CONCURRENT_POSITIONS} positions · -{LIMITS.DAILY_LOSS_CIRCUIT_BREAKER:.0%} daily stop")

page = st.sidebar.radio("Navigate", [
    "1 · Regime (Can I trade?)",
    "2 · Pre-Trade Stack (Should I?)",
    "3 · Strategy & Sizing (How?)",
    "4 · Journal & Expectancy",
    "5 · Promotion Gate",
    "6 · Playbooks",
    "7 · Masters Library",
    "8 · Paper Desk (Tradier Sandbox)",
    "9 · ThinkScript Suite",
    "10 · IWT Zone Scorer",
    "11 · Risk Plan + Pre-Trade",
])

# ---------------------------------------------------------------- regime state
if "regime_result" not in st.session_state:
    st.session_state.regime_result = score_regime(RegimeInputs())

# ================================================================ PAGE 1
if page.startswith("1"):
    st.header("Portfolio State Engine — Can I trade?")
    st.caption("Deterministic scoring over observable inputs. Auto-fill pulls live data; every field is overridable so the logic stays inspectable.")

    c1, c2 = st.columns([1, 2])
    with c1:
        if st.button("⚡ Auto-fill from market data"):
            auto = try_autofill_inputs()
            if auto:
                st.session_state.auto_inputs = auto
                st.success("Fetched. Review and adjust the manual flags below.")
            else:
                st.warning("Market data unavailable (offline/rate-limited). Enter inputs manually.")
    base = st.session_state.get("auto_inputs", RegimeInputs())

    col1, col2, col3 = st.columns(3)
    with col1:
        vix = st.number_input("VIX level", 5.0, 90.0, float(base.vix_level))
        vix5 = st.number_input("VIX 5-day change (%)", -60.0, 200.0, float(base.vix_5d_change_pct))
        spx = st.number_input("SPX vs 50DMA (%)", -25.0, 25.0, float(base.spx_vs_50dma_pct))
    with col2:
        breadth = st.number_input("Breadth: RSP/SPY 20d rel. (%)", -15.0, 15.0, float(base.breadth_rsp_spy_20d_pct))
        oil5 = st.number_input("Oil 5-day change (%)", -40.0, 60.0, float(base.oil_5d_change_pct))
        credit = st.selectbox("Credit spreads", [0, 1, 2], format_func=lambda i: ["Calm", "Widening", "Stress"][i])
    with col3:
        event = st.checkbox("Major macro event within 5 days (FOMC/CPI/NFP/mega-cap earnings)")
        geo = st.checkbox("Active geopolitical escalation (oil/shipping-relevant)")
        dd = st.checkbox("Account in drawdown / recent ≥1R loss")

    res = score_regime(RegimeInputs(vix, vix5, spx, breadth, oil5, credit, event, geo, dd))
    st.session_state.regime_result = res

    color = {"OFFENSIVE": "green", "NEUTRAL": "blue", "DEFENSIVE": "orange", "LOCKDOWN": "red"}[res["regime"]]
    st.markdown(f"## Regime: :{color}[{res['regime']}]  (score {res['score']:+d})")
    st.info(res["guidance"])
    st.caption(f"Sizing multiplier: **{LIMITS.REGIME_MULTIPLIER[res['regime']]}x** → effective max risk/trade: "
               f"**${equity * LIMITS.MAX_RISK_PCT_PER_TRADE * LIMITS.REGIME_MULTIPLIER[res['regime']]:,.0f}**")
    st.dataframe(pd.DataFrame(res["factors"].items(), columns=["Factor", "Score (-2..+2)"]), hide_index=True)

    if st.button("📓 Log today's regime read to journal"):
        journal.log_decision(mode=MODE, decision_type="NO_TRADE", underlying="(regime read)",
                             structure="-", direction="-", regime=res["regime"],
                             regime_score=res["score"], thesis="Daily regime assessment", status="N/A")
        st.success("Logged. Daily regime reads build the calibration dataset.")

# ================================================================ PAGE 2
elif page.startswith("2"):
    st.header("Seven-Agent Pre-Trade Stack — Should I trade?")
    reg = st.session_state.regime_result
    st.caption(f"Current regime: **{reg['regime']}** (set on page 1). "
               "Aggregation rule: any agent can VETO; no agent can initiate; ≥5/7 neutral-or-better required.")

    underlying = st.selectbox("Underlying", WATCHLIST_CORE + ["Other…"])
    if underlying == "Other…":
        underlying = st.text_input("Ticker", "SPY").upper()
    thesis = st.text_area("Thesis (falsifiable — what invalidates it?)",
                          placeholder="e.g., NVDA held institutional accumulation at 172 on 3x volume; invalidation below 169.5…")

    scores = {}
    cols = st.columns(2)
    agent_names = [a for a in AGENTS if a != "Risk"]
    for i, name in enumerate(agent_names):
        with cols[i % 2]:
            with st.expander(f"🤖 {name} Agent — {AGENTS[name]['question']}", expanded=False):
                for p in AGENTS[name]["prompts"]:
                    st.caption(f"· {p}")
                scores[name] = st.select_slider(f"{name} verdict", SCORES, value="NEUTRAL", key=f"ag_{name}")

    agg = aggregate(scores)
    st.divider()
    if agg["passed"]:
        st.success(f"✅ Agent stack PASSED — {len(agg['supportive_or_better'])}/6 human agents neutral-or-better, no vetoes. "
                   "Proceed to the Risk Engine (Agent 7) on page 3.")
    else:
        st.error(f"❌ Agent stack FAILED. Vetoes: {agg['vetoes'] or 'none'} · "
                 f"Neutral-or-better: {len(agg['supportive_or_better'])}/6 (need ≥5 of 7 incl. Risk). "
                 "The correct trade is no trade.")
        if st.button("📓 Log NO-TRADE decision (this is a win)"):
            journal.log_decision(mode=MODE, decision_type="NO_TRADE", underlying=underlying,
                                 structure="-", direction="-", regime=reg["regime"], regime_score=reg["score"],
                                 thesis=thesis, agent_notes=str(scores), status="N/A")
            st.success("Logged. Prevented bad trades are positive expectancy.")

# ================================================================ PAGE 3
elif page.startswith("3"):
    st.header("Strategy Selection, Risk Engine & Sizing — How?")
    reg = st.session_state.regime_result

    st.subheader("a) Structure selector")
    c1, c2 = st.columns(2)
    with c1:
        view = st.selectbox("Directional view", ["BULLISH", "BEARISH", "NEUTRAL", "BIG-MOVE-UNSURE-DIRECTION"])
    with c2:
        iv = st.selectbox("IV state (vs HV / IV rank)", ["RICH", "CHEAP", "NORMAL"])
    sel = select_strategy(view, iv, reg["regime"])
    st.info(f"**Suggested structure: {sel['structure']}** — {sel['why']}")

    st.subheader("b) Deterministic Risk Engine (Agent 7 — final authority)")
    with st.form("risk_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            underlying = st.text_input("Underlying", "SPY").upper()
            structure = st.text_input("Structure", sel["structure"])
            is_option = st.checkbox("Options trade", True)
            short_prem = st.checkbox("Short premium (credit)", "Credit" in sel["structure"] or "Condor" in sel["structure"])
            defined = st.checkbox("Defined risk", True)
            dte = st.number_input("DTE", 0, 365, 30)
        with c2:
            max_loss = st.number_input("Max loss per contract / stop-risk per share ($)", 0.0, 100000.0, 150.0)
            max_gain = st.number_input("Max gain per contract / target-gain per share ($)", 0.0, 100000.0, 350.0)
            risk_dollars = st.number_input("Proposed total risk ($)", 0.0, 1000000.0,
                                           float(round(equity * 0.01 * LIMITS.REGIME_MULTIPLIER[reg["regime"]], 0)))
            open_pos = st.number_input("Currently open positions", 0, 20, 0)
            heat = st.number_input("Current open risk (% of equity)", 0.0, 20.0, 0.0) / 100
        with c3:
            losing_same = st.checkbox("Open LOSING position in this underlying?")
            hours_event = st.number_input("Hours to next major event (this underlying)", 0.0, 999.0, 999.0)
            today_pnl = st.number_input("Today's P&L (%)", -50.0, 50.0, 0.0)
            cooldown = st.checkbox("≥1R loss within last session?")
            thesis = st.text_area("Thesis", height=68)
        submitted = st.form_submit_button("⚖️ Run Risk Engine")

    if submitted:
        p = TradeProposal(underlying=underlying, direction=view, structure=structure, is_option=is_option,
                          is_short_premium=short_prem, is_defined_risk=defined, dte=int(dte),
                          max_loss_per_unit=max_loss, max_gain_per_unit=max_gain, account_equity=equity,
                          proposed_risk_dollars=risk_dollars, open_positions=int(open_pos),
                          open_portfolio_heat_pct=heat, has_open_losing_position_same_underlying=losing_same,
                          hours_to_next_major_event=hours_event, todays_pnl_pct=today_pnl,
                          last_full_loss_within_cooldown=cooldown, regime=reg["regime"], thesis=thesis)
        verdict = evaluate(p)
        if verdict.approved:
            st.success(f"✅ APPROVED ({MODE}). Regime-adjusted max risk: ${verdict.adjusted_max_risk_dollars:,.0f}")
        else:
            st.error("⛔ VETOED. The engine's verdict is final — there is no override button, on purpose.")
            for x in verdict.vetoes:
                st.markdown(f"- 🔴 {x}")
        for w in verdict.warnings:
            st.warning(w)
        with st.expander("Rules passed"):
            st.write(verdict.passes)

        dtype = "TRADE" if verdict.approved else "VETOED"
        if st.button(f"📓 Log this {dtype} decision"):
            journal.log_decision(mode=MODE, decision_type=dtype, underlying=underlying, structure=structure,
                                 direction=view, regime=reg["regime"], regime_score=reg["score"], thesis=thesis,
                                 risk_dollars=risk_dollars, veto_reasons="; ".join(verdict.vetoes),
                                 status="OPEN" if verdict.approved else "N/A")
            st.success("Logged.")

    st.subheader("c) Position sizing")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Fixed-risk (Ijeoma)** — options/defined-risk")
        mlpc = st.number_input("Max loss per contract ($)", 1.0, 50000.0, 150.0, key="sz1")
        fs = final_size(equity, reg["regime"], max_loss_per_contract=mlpc)
        st.json(fs)
    with c2:
        st.markdown("**Fractional Kelly (Thorp)** — from journal stats only")
        s = journal.stats()
        if s.get("closed_trades", 0) >= 20 and "win_rate" in s:
            k = kelly_fraction(s["win_rate"], s["avg_win_r"], max(s["avg_loss_r"], 0.01))
            st.json(k)
        else:
            st.warning(f"Kelly locked until ≥20 closed trades in the journal "
                       f"(currently {s.get('closed_trades', 0)}). Thorp counted cards before sizing bets — "
                       "so does this system. Until then: fixed-risk 1% × regime multiplier only.")

# ================================================================ PAGE 4
elif page.startswith("4"):
    st.header("Decision Journal & Expectancy")
    st.caption("The journal is the MVP. Everything else is decoration until this table proves an edge.")

    s = journal.stats()
    c = st.columns(6)
    c[0].metric("Decisions", s.get("decisions_logged", 0))
    c[1].metric("Trades", s.get("trades_taken", 0))
    c[2].metric("No-trades", s.get("no_trades", 0))
    c[3].metric("Vetoes", s.get("vetoes", 0))
    c[4].metric("Win rate", f"{s.get('win_rate', 0):.0%}" if "win_rate" in s else "—")
    c[5].metric("Expectancy (R)", s.get("expectancy_r", "—"))

    df = journal.load()
    if len(df):
        closed = df[(df.decision_type == "TRADE") & (df.status == "CLOSED") & df.realized_r.notna()].sort_values("id")
        if len(closed) >= 2:
            import plotly.express as px
            eq = closed.realized_r.cumsum().reset_index(drop=True)
            st.plotly_chart(px.line(eq, labels={"value": "Cumulative R", "index": "Closed trade #"},
                                    title="Equity curve (R units)"), use_container_width=True)
        st.dataframe(df, hide_index=True, use_container_width=True)
        st.download_button("⬇️ Export CSV", df.to_csv(index=False), "quantum_maestro_journal.csv")
    else:
        st.info("No decisions logged yet. Start with a daily regime read (page 1).")

    st.subheader("Close a trade")
    with st.form("close"):
        did = st.number_input("Decision ID", 1, 10_000_000, 1)
        exitp = st.number_input("Exit price", 0.0, 1_000_000.0, 0.0)
        rr = st.number_input("Realized R (net of costs; loss = negative)", -10.0, 20.0, 0.0)
        lesson = st.text_area("Lesson (mandatory for losses — Dalio: pain + reflection = progress)")
        if st.form_submit_button("Close trade"):
            if rr < 0 and len(lesson.strip()) < 15:
                st.error("Losses require a written lesson. That's the tuition receipt.")
            else:
                journal.close_trade(int(did), exitp, rr, lesson)
                st.success("Closed and recorded.")

# ================================================================ PAGE 5
elif page.startswith("5"):
    st.header("Promotion Gate — SHADOW → LIVE")
    st.caption("All criteria must pass, evaluated from the journal only. One hard-rule violation resets the clock. "
               "Simons doctrine: never override the model — including this one.")
    g = journal.evaluate_gate()
    for check, passed in g["checks"].items():
        st.markdown(f"{'✅' if passed else '❌'} {check}")
    st.divider()
    if g["promotable"]:
        st.success("Gate PASSED. Next step (deliberate, human, offline): wire a broker adapter per docs/ARCHITECTURE.md, "
                   "starting with preview-only order staging.")
    else:
        st.warning("Gate NOT passed. This is the system working, not the system failing. "
                   "Live capital deployed before demonstrated expectancy is donation, not trading.")
    with st.expander("Current journal stats"):
        st.json(g["stats"])

# ================================================================ PAGE 6
elif page.startswith("6"):
    st.header("Playbooks")
    files = sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "playbooks", "*.md")))
    if not files:
        st.info("No playbooks found.")
    for f in files:
        with open(f) as fh:
            content = fh.read()
        title = content.splitlines()[0].lstrip("# ")
        with st.expander(f"📖 {title}"):
            st.markdown(content)

# ================================================================ PAGE 7
elif page.startswith("7"):
    st.header("Masters Library — a global game")
    st.caption("Principles mapped to the module they govern. Cautions included: every master is also a cautionary tale about something.")
    for m in MASTERS:
        with st.expander(f"🌍 {m['name']} — {m['origin']}"):
            st.markdown(f"**Principle:** {m['principle']}")
            st.markdown(f"**Governs in this system:** {m['governs']}")
            st.markdown(f"**Caution:** _{m['caution']}_")


# ================================================================ PAGE 8
elif page.startswith("8"):
    st.header("Paper Desk — Tradier Sandbox")
    st.caption("Endpoint pinned to sandbox.tradier.com (production does not exist in this codebase). "
               "$100k virtual funds · 15-min delayed data. Every submitted order passes the deterministic "
               "risk engine — the submit path physically requires an approved verdict.")

    from qm.broker_tradier import TradierSandbox, TradierAuthError, OrderNotApprovedError

    @st.cache_resource(show_spinner=False)
    def _client():
        return TradierSandbox()

    try:
        tc = _client()
        acct = tc.resolve_account()
    except TradierAuthError as e:
        st.error(f"Sandbox connection failed: {e}")
        st.code('# App -> Settings -> Secrets\nTRADIER_SANDBOX_TOKEN = "<your sandbox key>"\nTRADIER_SANDBOX_ACCOUNT = "<VAxxxxxxxx>"  # optional', language="toml")
        st.stop()
    except Exception as e:
        st.error(f"Sandbox unreachable: {e}")
        st.stop()

    reg = st.session_state.regime_result
    try:
        bal = tc.balances()
        c = st.columns(4)
        c[0].metric("Account", acct)
        c[1].metric("Total equity", f"${float(bal.get('total_equity', 0) or 0):,.0f}")
        c[2].metric("Option BP", f"${float(bal.get('option_buying_power', 0) or 0):,.0f}")
        c[3].metric("Regime", reg["regime"])
    except Exception as e:
        st.warning(f"Balances unavailable: {e}")

    tab_pos, tab_ticket = st.tabs(["📋 Positions & Orders", "🎫 Vertical Spread Ticket"])

    with tab_pos:
        try:
            pos = tc.positions()
            st.dataframe(pd.DataFrame(pos) if pos else pd.DataFrame(), use_container_width=True, hide_index=True)
            st.caption("Open orders")
            odf = tc.orders()
            st.dataframe(pd.DataFrame(odf) if odf else pd.DataFrame(), use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Positions/orders unavailable: {e}")

    with tab_ticket:
        st.caption("Defined-risk verticals only — matching the hard rules. Build → risk engine → preview → submit.")
        c1, c2, c3 = st.columns(3)
        with c1:
            sym = st.text_input("Underlying", "SPY").upper().strip()
            kind = st.selectbox("Structure", ["Put Credit Spread", "Call Credit Spread",
                                              "Bull Call Spread (debit)", "Bear Put Spread (debit)"])
        try:
            exps = tc.expirations(sym)
        except Exception:
            exps = []
        with c2:
            exp = st.selectbox("Expiration", exps or ["(unavailable)"])
        chain = []
        if exps and exp and exp != "(unavailable)":
            try:
                chain = tc.option_chain(sym, exp)
            except Exception as e:
                st.warning(f"Chain unavailable: {e}")
        opt_type = "put" if "Put" in kind else "call"
        legs_pool = [o for o in chain if o.get("option_type") == opt_type]
        strikes = sorted({float(o["strike"]) for o in legs_pool})
        with c3:
            qty = st.number_input("Contracts", 1, 100, 1)
            limit = st.number_input("Limit price (net credit/debit per spread)", 0.01, 500.0, 0.50, step=0.01)

        if strikes:
            sc1, sc2 = st.columns(2)
            with sc1:
                short_k = st.selectbox("Short strike" if "Credit" in kind else "Long strike", strikes,
                                       index=min(len(strikes)//2, len(strikes)-1))
            with sc2:
                long_k = st.selectbox("Long strike" if "Credit" in kind else "Short strike", strikes,
                                      index=min(len(strikes)//2 + 2, len(strikes)-1))
            width = abs(float(long_k) - float(short_k))
            is_credit = "Credit" in kind
            max_loss = (width - limit) * 100 if is_credit else limit * 100
            max_gain = limit * 100 if is_credit else (width - limit) * 100
            from datetime import date
            dte = ( date.fromisoformat(exp) - date.today() ).days if exp and exp != "(unavailable)" else 0

            def occ(strike, right):
                d = exp.replace("-", "")[2:]
                return f"{sym}{d}{right.upper()[0]}{int(round(float(strike)*1000)):08d}"

            if is_credit:
                legs = [{"option_symbol": occ(short_k, opt_type), "side": "sell_to_open", "quantity": int(qty)},
                        {"option_symbol": occ(long_k, opt_type), "side": "buy_to_open", "quantity": int(qty)}]
            else:
                legs = [{"option_symbol": occ(short_k, opt_type), "side": "buy_to_open", "quantity": int(qty)},
                        {"option_symbol": occ(long_k, opt_type), "side": "sell_to_open", "quantity": int(qty)}]
            order = {"class": "multileg", "symbol": sym, "type": "credit" if is_credit else "debit",
                     "duration": "day", "price": round(float(limit), 2), "legs": legs}

            st.caption(f"Width ${width:.2f} · Max loss/contract ${max_loss:.0f} · Max gain/contract ${max_gain:.0f} · {dte} DTE")
            colA, colB = st.columns(2)
            thesis = st.text_area("Thesis (required by the engine)", key="pd_thesis",
                                  placeholder="Level, invalidation, why now…")

            proposal = TradeProposal(
                underlying=sym, direction="NEUTRAL" if is_credit else "LONG", structure=kind,
                is_option=True, is_short_premium=is_credit, is_defined_risk=True, dte=int(dte),
                max_loss_per_unit=max_loss, max_gain_per_unit=max_gain, account_equity=equity,
                proposed_risk_dollars=max_loss * int(qty), open_positions=len(tc.positions() or []),
                regime=reg["regime"], thesis=thesis or "")

            with colA:
                if st.button("🔍 Preview (ungated)"):
                    try:
                        st.json(tc.preview_order(order))
                    except Exception as e:
                        st.error(f"Preview failed: {e}")
            with colB:
                if st.button("⚖️➡️📤 Run Risk Engine & Submit"):
                    verdict = evaluate(proposal)
                    if not verdict.approved:
                        st.error("⛔ VETOED — order refused before it reached the broker.")
                        for x in verdict.vetoes:
                            st.markdown(f"- 🔴 {x}")
                        journal.log_decision(mode="SHADOW", decision_type="VETOED", underlying=sym,
                                             structure=kind, direction=proposal.direction, regime=reg["regime"],
                                             regime_score=reg["score"], thesis=thesis,
                                             risk_dollars=proposal.proposed_risk_dollars,
                                             veto_reasons="; ".join(verdict.vetoes), status="N/A")
                    else:
                        try:
                            resp = tc.place_order(order, verdict)
                            st.success("✅ Sandbox order submitted.")
                            st.json(resp)
                            journal.log_decision(mode="SHADOW", decision_type="TRADE", underlying=sym,
                                                 structure=kind, direction=proposal.direction,
                                                 regime=reg["regime"], regime_score=reg["score"], thesis=thesis,
                                                 risk_dollars=proposal.proposed_risk_dollars, status="OPEN")
                        except OrderNotApprovedError as e:
                            st.error(str(e))
                        except Exception as e:
                            st.error(f"Submit failed: {e}")
        else:
            st.info("Enter a symbol with an options chain to build a vertical (sandbox data is 15-min delayed).")


# ================================================================ PAGE 9
elif page.startswith("9"):
    st.header("ThinkScript Suite — chart-side proxies")
    st.caption("Copy-paste studies for ThinkOrSwim. These are PROXIES — the app's regime engine "
               "(breadth/oil/credit) is canonical; on disagreement, the app wins. None place orders "
               "or read balances. Constants mirror qm/config.py.")

    import glob as _glob
    ts_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "thinkscript"))
    meta = {
        "TeriQuantumMaestro_v6_ChartDraw.ts": ("Chart draw-only (Bollinger-style)", "Charts → Studies → Edit → Create",
            "The RECOMMENDED pairing: draws zones + entry/stop/target on price and stays quiet, like Bollinger Bands. Run TeriQuantumOsc_v4_6 in the lower panel for regime/permission/odds/setup readouts. One info line only; no double alerts."),
        "TeriQuantumMaestro_v6.ts": ("ALL-IN-ONE (upper)", "Charts → Studies → Edit → Create",
            "THE culmination: one upper study - draws zones on price + regime + permission + eight-point odds + relative strength + earnings/ex-div gates + canonical order-type intent. Run TeriQuantumOsc_v4_6 alongside for the histogram panel. Mechanical proxy, not validated vs hand-marked zones."),
        "TeriQuantumZones_v5_1.ts": ("Upper study (zones)", "Charts → Studies → Edit → Create",
            "Draws buyer/seller ZONES on price with rec-state persistence, U/Chair approach detection, distinct-revisit counting, and canonical 7-8/5-6/0-4 cohorts. Bug-fixed from a v5.0 draft (base>=5, thresholds, secondary cohort, repainting). Complements the lower TeriQuantumOsc; mechanical PROXY, not validated."),
        "TeriQuantumOsc_v4_6.ts": ("Lower study (definitive)", "Charts → Studies → Edit → Create",
            "v4.4 + v5 odds enhancer + canonical alignment: freshness matches the course table (1+ visits=0), on-chart order-type intent (entry/exit LIMIT, stop STOP-MKT), and the >3 TAKE THE TRADE reward:risk gate. The definitive lower-panel study."),
        "TeriQuantumOsc_v5.ts": ("Lower study (latest)", "Charts → Studies → Edit → Create",
            "v4.4 + on-chart eight-point ZONE ODDS enhancer: scores the most recent zone base/departure/freshness/reward-risk into PRIMARY(7-8)/SECONDARY(5-6)/SKIP. Mirrors qm/iwt_zones.py and page 10."),
        "TeriQuantumOsc_v4.ts": ("Lower study (adaptive)", "Charts → Studies → Edit → Create",
            "Chart-adaptive: change the chart timeframe and everything recomputes on it (like RSI). Runs on any timeframe "
            "including weekly. Regime/Location/Permission split. Use when you want 'regime at the timeframe I'm looking at.'"),
        "TeriQuantumOsc_v3.ts": ("Lower study (fixed-daily)", "Charts → Studies → Edit → Create",
            "Always reads daily regime regardless of chart — a stable macro backdrop that doesn't wobble as you zoom. "
            "Breaks on weekly charts (daily secondary-aggregation is illegal there); use v4 on weekly/monthly."),
        "TeriQuantumOsc_v2.ts": ("Lower study (legacy)", "Charts → Studies → Edit → Create",
            "Simplest single-composite oscillator. Kept for reference."),
        "TQO_RegimeColumn_v4.ts": ("Watchlist column (adaptive)", "Watchlist gear → Customize → Scripts → Create",
            "Per-name regime score at the column's aggregation. Set column agg to Daily (stable) or Weekly (swing scan). "
            "Amber cell = volatility stress. Pair with LevelProximity and sort to find in-regime names sitting at a level."),
        "TQO_ChartOverlay.ts": ("Upper study", "Charts → Studies → Edit → Create",
            "Buyer/seller zones (daily+weekly), ATR extension band, on-chart setup card (entry/stop/targets/R:R)."),
        "TQO_GuardRails.ts": ("Study", "Charts → Studies → Edit → Create",
            "Settlement/event/DTE warnings: 0DTE-ban notice, event lockout, physical-settlement late-day close, anti-repair."),
        "TQO_WatchlistColumn.ts": ("Watchlist column", "Watchlist gear → Customize → Scripts → Create",
            "Regime score compressed to a colored cell — scan the whole Teri list at once."),
        "TQO_LevelProximity.ts": ("Watchlist column", "Watchlist gear → Customize → Scripts → Create",
            "Is this name AT a level with room? Green = favorable-R:R candidate, gray = mid-range no-chase."),
    }
    readme = os.path.join(ts_dir, "README.md")
    if os.path.exists(readme):
        with st.expander("📋 Recommended layout & install notes"):
            st.markdown(open(readme).read())

    files = sorted(_glob.glob(os.path.join(ts_dir, "*.ts")))
    if not files:
        st.info("No ThinkScript files found in /thinkscript.")
    for f in files:
        base = os.path.basename(f)
        kind, where, desc = meta.get(base, ("Study", "Charts → Studies", ""))
        with st.expander(f"📈 {base}  ·  {kind}"):
            st.caption(f"**Install:** {where}")
            st.markdown(desc)
            st.code(open(f).read(), language="c")


# ================================================================ PAGE 10
elif page.startswith("10"):
    st.header("IWT Zone Scorer - eight-point odds enhancer")
    st.caption("Teri's zone-quality decomposition as a checklist. Score 7-8 = primary cohort (full size); "
               "5-6 = secondary (half size, needs confirmation); 0-4 = skip. Research proxy - see "
               "research/IWT_BACKTEST_ASSESSMENT.md for what this does and doesn't prove.")
    from qm.iwt_zones import odds_enhancer, frame_trade

    st.subheader("a) Score the zone")
    c1, c2 = st.columns(2)
    with c1:
        base_candles = st.number_input("Base candles (tighter = stronger)", 1, 12, 2)
        departure = st.slider("Departure strength (x ATR)", 0.0, 3.0, 1.6, 0.1,
                              help="Body of the departure candle as a multiple of ATR. >=1.5 fast, >=0.75 average.")
    with c2:
        prior_visits = st.number_input("Prior visits (BEFORE this entry touch)", 0, 10, 0,
                                       help="Look-ahead control: do not count the current entry touch.")
        st.caption("Reward:risk is computed from the trade frame below.")

    st.subheader("b) Frame the trade (for reward:risk)")
    direction = st.radio("Direction", ["long", "short"], horizontal=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1: proximal = st.number_input("Proximal (entry edge)", value=100.0)
    with c2: distal = st.number_input("Distal (far edge)", value=98.0 if direction=="long" else 102.0)
    with c3: target = st.number_input("Target (opposing zone)", value=110.0 if direction=="long" else 90.0)
    with c4: atr = st.number_input("ATR", value=1.0, min_value=0.01)
    atr_buffer = st.slider("ATR stop buffer", 0.0, 1.0, 0.20, 0.05)

    trade = frame_trade(direction, proximal, distal, target, atr, atr_buffer)
    if not trade.is_valid():
        st.error("Trade geometry is invalid (for a long need stop < entry < target). Check your levels.")
    else:
        result = odds_enhancer(base_candles, departure, prior_visits, trade.reward_risk)
        color = {"PRIMARY": "green", "SECONDARY": "orange", "REJECTED": "red"}[result["cohort"]]
        st.markdown(f"### Score: :{color}[{result['score']}/8]  -  {result['cohort']} cohort")
        st.info(f"{result['action']}   (size multiplier {result['size_multiplier']}x)")
        pc = result["parts"]
        st.dataframe(pd.DataFrame([
            {"Factor": "Base candles", "Value": base_candles, "Points": pc["base"]},
            {"Factor": "Departure (xATR)", "Value": round(departure, 2), "Points": pc["departure"]},
            {"Factor": "Freshness (prior visits)", "Value": prior_visits, "Points": pc["freshness"]},
            {"Factor": "Reward:risk", "Value": round(trade.reward_risk, 2), "Points": pc["reward_risk"]},
        ]), hide_index=True, use_container_width=True)
        st.caption(f"Entry {trade.entry:.2f} - Stop {trade.stop:.2f} - Target {trade.target:.2f} - "
                   f"R:R {trade.reward_risk:.2f} - risk/share {trade.risk:.2f}")

        if result["cohort"] != "REJECTED":
            st.subheader("c) Cohort-aware position size")
            from qm.sizing import final_size
            reg = st.session_state.regime_result
            fs = final_size(equity, reg["regime"], entry=trade.entry, stop=trade.stop,
                            cohort=result["cohort"])
            st.caption(f"Effective risk = 1% x regime {fs['regime_multiplier']} ({reg['regime']}) "
                       f"x cohort {fs['cohort_multiplier']} ({result['cohort']}) "
                       f"= **{fs['effective_risk_pct']*100:.2f}%** of ${equity:,.0f} "
                       f"= ${equity*fs['effective_risk_pct']:,.0f} at risk")
            if "fixed_risk" in fs and "shares" in fs.get("fixed_risk", {}):
                st.caption(f"Shares: {fs['fixed_risk']['shares']} (fixed-risk, regime+cohort scaled)")
            st.warning("Cohort discipline: log 7-8 and 5-6 trades as SEPARATE cohorts in the journal - never "
                       "combine them, or a strong cohort's edge gets diluted by a weak one. This scorer is a "
                       "PROXY for Teri's manual zones; that proxy is not yet validated against hand-marked examples.")


# ================================================================ PAGE 11
elif page.startswith("11"):
    st.header("Risk Plan + Pre-Trade Worksheet")
    st.caption("Canonical from Teri's Personal Trading Plan and IWT Stock-Pick worksheet. "
               "The daily ceiling uses the STRICTER of Teri's 3% and Quantum Maestro's 2% doctrine.")
    from qm.iwt_canonical import IWTRiskPlan, check_risk_plan, PreTradeWorksheet, iwt_long_trade

    tab1, tab2, tab3 = st.tabs(["Risk-plan cascade", "Pre-trade worksheet", "Canonical RR"])

    with tab1:
        st.subheader("Where am I against my loss ceilings?")
        eq = st.number_input("Account equity ($)", value=100000.0, min_value=1000.0, key="rp_eq")
        c1, c2, c3 = st.columns(3)
        with c1:
            pnl_d = st.number_input("Realized P&L today ($)", value=0.0, key="rp_d")
            n_d = st.number_input("Trades today", 0, 50, 0, key="rp_nd")
        with c2:
            pnl_w = st.number_input("Realized P&L this week ($)", value=0.0, key="rp_w")
            n_w = st.number_input("Trades this week", 0, 200, 0, key="rp_nw")
        with c3:
            pnl_m = st.number_input("Realized P&L this month ($)", value=0.0, key="rp_m")
            n_m = st.number_input("Trades this month", 0, 500, 0, key="rp_nm")
        risk = st.number_input("Proposed trade's max loss ($)", value=1000.0, min_value=0.0, key="rp_risk")

        r = check_risk_plan(eq, pnl_d, pnl_w, pnl_m, n_d, n_w, n_m, risk)
        cc = r["ceilings"]
        st.dataframe(pd.DataFrame([
            {"Ceiling": "Per trade", "Limit": f"${cc['per_trade']:,.0f}"},
            {"Ceiling": f"Daily ({r['daily_ceiling_source']})", "Limit": f"${cc['daily']:,.0f}"},
            {"Ceiling": "Weekly", "Limit": f"${cc['weekly']:,.0f}"},
            {"Ceiling": "Monthly", "Limit": f"${cc['monthly']:,.0f}"},
        ]), hide_index=True, use_container_width=True)
        if r["allowed"]:
            st.success("WITHIN PLAN — trade permitted by the risk cascade.")
        else:
            st.error("BLOCKED by the risk plan:")
            for b in r["breaches"]:
                st.write(f"- {b}")

    with tab2:
        st.subheader("Score a pick before you chart it")
        sym = st.text_input("Symbol", "AAPL", key="ws_sym")
        c1, c2 = st.columns(2)
        with c1:
            vol = st.number_input("Avg volume (shares/day)", value=2_000_000, key="ws_vol")
            up = st.radio("In an uptrend?", ["Yes", "No", "Unknown"], horizontal=True, key="ws_up")
            dte = st.number_input("Days to earnings (-1 = unknown)", -1, 400, -1, key="ws_dte")
            price = st.number_input("Current price", value=100.0, key="ws_px")
        with c2:
            hi = st.number_input("52-week high", value=120.0, key="ws_hi")
            lo = st.number_input("52-week low", value=60.0, key="ws_lo")
            bib = st.radio("Best in breed (leader)?", ["Yes", "No", "Unknown"], horizontal=True, key="ws_bib")
            dad = st.radio("~$1/day mover?", ["Yes", "No", "Unknown"], horizontal=True, key="ws_dad")

        tri = lambda v: True if v == "Yes" else (False if v == "No" else None)
        w = PreTradeWorksheet(sym, volume=vol, up_trend=tri(up),
                              days_to_earnings=(None if dte < 0 else dte),
                              price=price, high_52w=hi, low_52w=lo,
                              best_in_breed=tri(bib), dollar_a_day=tri(dad))
        out = w.evaluate()
        if out["ready_to_chart"]:
            st.success(f"{sym}: READY TO CHART — no hard blocks.")
        else:
            st.error(f"{sym}: NOT READY — hard blocks:")
            for b in out["hard_blocks"]:
                st.write(f"- {b}")
        if out["range_pos_52w"] is not None:
            st.caption(f"52-week range position: {out['range_pos_52w']}%")
        for wn in out["warnings"]:
            st.warning(wn)
        for nt in out["notes"]:
            st.info(nt)

    with tab3:
        st.subheader("Canonical RR (matches the course spreadsheet)")
        st.caption("Long: stop = distal_BZ - 20% ATR, target = proximal_SZ, entry = proximal_BZ.")
        c1, c2, c3 = st.columns(3)
        with c1:
            d_bz = st.number_input("Distal (buyer zone)", value=179.23, key="rr_dbz")
            p_bz = st.number_input("Proximal (buyer zone)", value=184.87, key="rr_pbz")
        with c2:
            p_sz = st.number_input("Proximal (seller zone/target)", value=202.82, key="rr_psz")
            atr_v = st.number_input("ATR", value=9.07, key="rr_atr")
        with c3:
            risk_tol = st.number_input("Risk tolerance ($)", value=1000.0, key="rr_tol")
        t = iwt_long_trade(d_bz, p_bz, p_sz, atr_v, risk_tolerance_dollars=risk_tol)
        st.dataframe(pd.DataFrame([
            {"Field": "Entry", "Value": t["entry"]},
            {"Field": "Stop", "Value": t["stop"]},
            {"Field": "Target", "Value": t["target"]},
            {"Field": "Reward:Risk", "Value": t["reward_risk"]},
            {"Field": "Shares", "Value": t["shares"]},
            {"Field": "Max profit", "Value": t["max_profit"]},
            {"Field": "Max loss", "Value": t["max_loss"]},
        ]), hide_index=True, use_container_width=True)
        if t["reward_risk"] < 3.0:
            st.warning(f"R:R {t['reward_risk']} is below the worksheet's >3:1 gate.")
        else:
            st.success(f"R:R {t['reward_risk']} clears the >3:1 gate.")

st.sidebar.divider()
st.sidebar.caption("Not financial advice. Decision-support software for a system in SHADOW validation. "
                   "Trading involves substantial risk of loss.")
