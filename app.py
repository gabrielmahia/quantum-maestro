# Copyright (c) 2026 Gabriel Mahia. All Rights Reserved.
# Unauthorized copying of this file, via any medium, is strictly prohibited.
# Proprietary and confidential.
# Written by Gabriel Mahia, 2026
# app.py
# 🏛️ QUANTUM MAESTRO — INSTITUTIONAL EDITION (V13 + Newbie Help Notes + WarrenAI Macro Prompt)
# Educational simulation tool — not financial advice.

# =========================
# ✅ QUICK START (NEWBIES)
# =========================
# 1) Install dependencies (example):
#    pip install streamlit yfinance pandas pandas_ta mplfinance numpy scipy pytz
# 2) Run:
#    streamlit run app.py
# 3) Workflow:
#    (A) Set Capital + Risk in Sidebar
#    (B) Scan Macro (VIX / Risk-on-off)
#    (C) Scan a Ticker (VIP list recommended)
#    (D) Score the setup (IWT)
#    (E) If GREEN/YELLOW and math checks out → log as PAPER first

import streamlit as st
import yfinance as yf
import pandas_ta as ta
import mplfinance as mpf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from datetime import datetime, time, timedelta
import pytz

# -------------------------
# 1) CONFIGURATION
# -------------------------
VIP_TICKERS = ["NVDA", "AAPL", "AMZN", "GOOGL", "TSLA", "MSFT", "META", "AMD", "NFLX", "SPY", "QQQ", "IWM", "GLD", "SLV", "USO"]
GROWTH_TICKERS = ["NVDA", "AAPL", "AMZN", "GOOGL", "TSLA", "MSFT", "META", "AMD", "NFLX", "QQQ", "ARKK", "COIN", "SHOP", "SQ"]
COMMODITY_TICKERS = ["GLD", "SLV", "GDX", "USO", "XLE", "FCX"]
VALUE_TICKERS = ["JPM", "BAC", "XOM", "CVX", "BRK.B", "JNJ", "PG"]

SECTOR_MAP = {
    "NVDA": "Tech", "AMD": "Tech", "MSFT": "Tech", "AAPL": "Tech", "META": "Tech", "GOOGL": "Tech",
    "TSLA": "Auto", "AMZN": "Consumer", "NFLX": "Media", "SPY": "Index", "QQQ": "Tech-Index", "IWM": "Index",
    "GLD": "Commodity", "SLV": "Commodity", "GDX": "Mining", "USO": "Energy", "XLE": "Energy",
    "JPM": "Finance", "BAC": "Finance", "XOM": "Energy", "CVX": "Energy", "BRK.B": "Conglomerate",
    "JNJ": "Healthcare", "PG": "Staples", "ARKK": "Thematic", "COIN": "Crypto", "SHOP": "Tech", "SQ": "Fintech", "FCX": "Materials"
}

# Realistic cost assumptions (for net R/R)
COMMISSION_PER_SHARE = 0.005   # e.g., $0.005/share
SLIPPAGE_BPS = 5              # 5 bps = 0.05%

# -------------------------
# 2) PAGE CONFIG + STYLE
# -------------------------
st.set_page_config(
    page_title="Quantum Maestro — Institutional Edition",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏛️"
)

st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 6px; height: 3em; font-weight: 700; letter-spacing: .3px; }
    div[data-testid="stMetric"] { border: 1px solid #dee2e6; border-radius: 8px; padding: 10px 14px; }
    @media (prefers-color-scheme: dark) {
        div[data-testid="stMetric"] { border: 1px solid #30333d; }
    }
    .risk-alert { background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 6px; border-left: 5px solid #ffc107; margin: 10px 0; }
    .risk-critical { background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 6px; border-left: 5px solid #dc3545; margin: 10px 0; }
    .success-box { background-color: #d4edda; color: #155724; padding: 10px; border-radius: 6px; border-left: 5px solid #28a745; margin: 10px 0; }
    .methodology-text { font-size: .9em; opacity: .85; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# -------------------------
# 3) LEGAL + ONBOARDING GATE
# -------------------------
st.title("🏛️ Quantum Maestro [TradingBot] — Institutional Edition")
st.caption("Portfolio Risk Architecture | Volatility Regimes | Passive Flows Thinking | IWT Execution Discipline")

with st.expander("⚠️ READ FIRST: Legal Disclaimer & Risk Warning", expanded=True):
    st.markdown("""
**1) Educational Use Only:** Simulation + training tool — **not** financial advice.  
**2) No Affiliation:** Independent; not affiliated with Trade & Travel or any institution.  
**3) Risk Warning:** Trading can result in significant losses. Past performance ≠ future results.  
**4) Data Disclaimer:** Yahoo Finance data may be delayed/inaccurate.  
""")
    agree = st.checkbox("✅ I understand this is educational only and involves real financial risk.")

if not agree:
    st.warning("🛑 Please accept the disclaimer above to continue.")
    st.stop()

# -------------------------
# 4) NEWBIE HELP NOTES (EXPANDER)
# -------------------------
with st.expander("🎓 Beginner's Guide (Read This First)", expanded=False):
    st.markdown("""
### 🎓 Beginner's Guide to Using Quantum Maestro

**Step 1: Understand Your Risk**
- Set **Total Capital** to your actual account size
- Set **Risk per Trade** to ~1% of capital (ex: $10,000 → $100)
- Never risk more than 2% per trade

**Step 2: Check Macro Conditions FIRST**
- Scan **VIX** before scanning individual tickers
- If VIX is **HIGH/CRISIS** → reduce size or don’t trade
- If Risk-Off (Gold + VIX rising) → avoid aggressive longs

**Step 3: Scan a Stock**
- Use **VIP List** (safest liquidity) or enter a ticker manually
- Wait for indicators to load

**Step 4: Score the IWT Setup**
- Freshness (fresh zone beats stale)
- Time in zone (fast rejection beats lingering)
- Speed out (impulsive beats grind)
- R/R must be ≥ 2.0 (prefer ≥ 3.0)

**Step 5: Verdict Discipline**
- 7–8 → GREEN (allowed)
- 5–6 → YELLOW (confirmation + smaller size)
- 0–4 → RED (no trade)

**Golden Rules**
1) Stop trading when you hit your daily goal  
2) Don’t stack too many positions (overtrading kills accounts)  
3) Trade WITH the trend (moving averages / structure)  
4) High VIX = smaller size or sit out  
5) Journal every trade (wins + losses)
""")

# -------------------------
# 5) WARRENAI “GLOBAL CONFLUENCE” PROMPT (COPY/PASTE)
# -------------------------
WARRENAI_PROMPT = r"""
IDENTITY & CORE DIRECTIVE
You are Quantum Maestro, an Institutional Macro Strategist. You combine:
• The scenario depth of WarrenAI (Investing.com / institutional lens)
• The execution discipline of Teri Ijeoma’s IWT method
Your goal is NOT to predict. Your goal is to FILTER for high-probability trades with risk-first discipline.

PHASE 1: THE GLOBAL CONFLUENCE (MANDATORY)
1) The Index Correlation:
   Compare SPY (S&P 500), DIA (Dow), QQQ (Nasdaq).
   Decide: “Broadening” (Dow/S&P leading) vs “Concentration” (Nasdaq/Mag7 leading).

2) The Foreign Echo:
   Review DAX (Europe) and Nikkei (Japan) relative to the US open.
   If foreign markets sold off overnight → prioritize bearish scenarios; be cautious with longs.

3) Passive Flow Audit:
   Identify if today is a high-flow day (1st/15th payroll/401k flows; quarter-end rebalancing).
   High passive flows can create a “floor” but also increase midday chop.

PHASE 2: INSTITUTIONAL ANALYSIS (WARRENAI STYLE)
1) Sector Rotation: Where is money flowing (out of tech into defensives/energy/etc)?
2) Volatility Regime: Use VIX trend.
   If VIX > 20 → widen stops and reduce size; if VIX > 30 → default to defense.
3) Institutional Footprints: Are large players accumulating or distributing near key levels?

PHASE 3: THE IWT EXECUTION FILTER
Rules:
- Asset must be VIP-grade liquidity
- Reward/Risk minimum 3:1 (strict) or reject
- Zone quality: fresh demand/supply preferred
- If daily goal is met → STOP (no new trades)

PHASE 4: VERDICT + MATH
Output:
- Verdict (GREEN / YELLOW / RED)
- One-sentence reason referencing macro + structure
- Exact levels (entry/stop/targets)
- Position size using max risk $X (user-defined)
- Reminder: “Consistency beats intensity.”
"""

with st.sidebar:
    st.header("💼 Portfolio Settings")

    # Session state init
    if "journal" not in st.session_state: st.session_state.journal = []
    if "open_positions" not in st.session_state: st.session_state.open_positions = []
    if "closed_trades" not in st.session_state: st.session_state.closed_trades = []
    if "daily_pnl" not in st.session_state: st.session_state.daily_pnl = 0.0
    if "goal_met" not in st.session_state: st.session_state.goal_met = False
    if "total_risk_deployed" not in st.session_state: st.session_state.total_risk_deployed = 0.0
    if "consecutive_losses" not in st.session_state: st.session_state.consecutive_losses = 0

    capital = st.number_input(
        "Total Capital ($)",
        value=10000,
        min_value=100,
        help="Your total account size. Example: $10,000 means ten thousand dollars in your account."
    )

    risk_per_trade = st.number_input(
        "Risk per Trade ($)",
        value=100,
        min_value=10,
        help="Max $ loss on ONE trade. Recommended ~1% of capital (ex: $10,000 → $100)."
    )

    max_portfolio_risk = st.number_input(
        "Max Portfolio Risk (%)",
        value=6.0,
        min_value=1.0,
        max_value=20.0,
        step=0.5,
        help="Max total risk across ALL open positions combined (prevents overtrading). Typical: 5–10%."
    )

    daily_goal = capital * 0.01
    st.caption(f"🎯 Daily Goal (1%): **${daily_goal:.2f}**")
    st.caption("💡 The 1% rule is a discipline rule: stop when you hit it.")

    portfolio_risk_pct = (st.session_state.total_risk_deployed / capital) * 100 if capital else 0
    if portfolio_risk_pct > max_portfolio_risk:
        st.error(f"⚠️ Portfolio Risk: {portfolio_risk_pct:.1f}% (OVER LIMIT)")
    else:
        st.info(f"📊 Portfolio Risk: {portfolio_risk_pct:.1f}% / {max_portfolio_risk:.1f}%")

    pnl_pct = (st.session_state.daily_pnl / capital) * 100 if capital else 0
    if st.session_state.goal_met:
        st.success(f"✅ Goal Met: +${st.session_state.daily_pnl:.2f} ({pnl_pct:.2f}%)")
    else:
        st.info(f"📈 Daily P&L: ${st.session_state.daily_pnl:.2f} ({pnl_pct:.2f}%)")

    if st.session_state.consecutive_losses > 0:
        st.warning(f"⚠️ Losing Streak: {st.session_state.consecutive_losses} trades")
        st.caption("💡 After 3 losses, reduce size automatically in real trading. Protect capital.")

    st.divider()
    st.header("🎯 Asset Selection")

    input_mode = st.radio(
        "Input Mode:",
        ["VIP List", "Manual"],
        help="VIP List = safest high-liquidity tickers. Manual = any ticker (higher risk of illiquidity/spreads)."
    )

    if input_mode == "VIP List":
        ticker = st.selectbox(
            "Ticker",
            VIP_TICKERS,
            help="High-liquidity favorites (tighter spreads / safer execution)."
        )
    else:
        ticker = st.text_input(
            "Ticker (Manual)",
            value="SPY",
            help="Enter any ticker symbol. Warning: low volume tickers can trap you with spreads and slippage."
        ).upper().strip()

    timeframe = st.selectbox(
        "Timeframe",
        ["1d", "1h", "15m"],
        index=0,
        help="1D = swing bias. 1H/15m = intraday bias. Lower timeframe = more noise."
    )

    stop_mode = st.selectbox(
        "Stop Mode (ATR Multiplier)",
        ["Swing (1.0 ATR)", "IWT Tight (0.2 ATR)"],
        index=0,
        help="Swing stops survive noise. Tight stops increase win-rate pressure and whipsaw risk."
    )
    atr_mult = 1.0 if "1.0" in stop_mode else 0.2

    st.divider()
    with st.expander("📋 Copy/Paste: WarrenAI Macro Prompt", expanded=False):
        st.code(WARRENAI_PROMPT, language="text")

# -------------------------
# 6) ENGINE
# -------------------------
class Engine:
    def __init__(self):
        self.vix_regimes = {
            "COMPLACENT": (0, 12),
            "NORMAL": (12, 20),
            "ELEVATED": (20, 30),
            "HIGH": (30, 40),
            "CRISIS": (40, 100)
        }

    def market_phase(self):
        et = pytz.timezone("US/Eastern")
        now = datetime.now(et).time()
        if now < time(9, 30): return "PRE_MARKET", "⏰ Pre-Market (low liquidity)"
        if now < time(10, 0): return "OPENING", "⚡ Opening Range (volatility window)"
        if now < time(12, 0): return "MORNING", "📈 Morning Trend Window"
        if now < time(14, 0): return "LUNCH", "🦀 Lunch Chop (low edge)"
        if now < time(15, 30): return "AFTERNOON", "🧭 Afternoon Resolution"
        if now < time(16, 0): return "CLOSE", "🪓 Close / Stop Hunts"
        return "AFTER_HOURS", "🌙 After Hours (thin liquidity)"

    def vix_value(self):
        try:
            vix = yf.Ticker("^VIX").history(period="5d", interval="1d")
            if vix.empty: return None
            return float(vix["Close"].iloc[-1])
        except:
            return None

    def vix_regime(self, v):
        if v is None: return "UNKNOWN"
        for k, (lo, hi) in self.vix_regimes.items():
            if lo <= v < hi: return k
        return "UNKNOWN"

    def fetch(self, symbol: str, tf: str):
        tf_map = {"1d": ("1y", "1d"), "1h": ("60d", "1h"), "15m": ("30d", "15m")}
        period, interval = tf_map.get(tf, ("1y", "1d"))
        df = yf.Ticker(symbol).history(period=period, interval=interval)
        if df is None or df.empty: return None

        df.ta.atr(length=14, append=True)
        df.ta.sma(length=20, append=True)
        df.ta.sma(length=50, append=True)
        st_df = df.ta.supertrend(length=10, multiplier=3)
        # pandas_ta returns columns like SUPERT_10_3.0 and SUPERTd_10_3.0
        df["ST_VAL"] = st_df[st_df.columns[0]]
        df["ST_DIR"] = st_df[st_df.columns[1]]

        # Structure
        closes = df["Close"].values
        df["Min"] = df.iloc[argrelextrema(closes, np.less_equal, order=5)[0]]["Close"]
        df["Max"] = df.iloc[argrelextrema(closes, np.greater_equal, order=5)[0]]["Close"]
        return df

    def auto_zones(self, df: pd.DataFrame):
        px = float(df["Close"].iloc[-1])
        supports = df["Min"][df["Min"] < px * 0.99]
        resists = df["Max"][df["Max"] > px * 1.01]
        demand = float(supports.iloc[-1]) if not supports.empty else float(df["Low"].min())
        supply = float(resists.iloc[-1]) if not resists.empty else float(df["High"].max())
        return demand, supply

engine = Engine()

# -------------------------
# 7) TOP ROW: MACRO
# -------------------------
colA, colB, colC, colD = st.columns(4)
phase_code, phase_label = engine.market_phase()
vix = engine.vix_value()
vix_reg = engine.vix_regime(vix)

with colA:
    st.metric("Market Phase (ET)", phase_label)

with colB:
    st.metric("VIX", "N/A" if vix is None else f"{vix:.2f}", vix_reg)

with colC:
    st.metric("Daily Goal (1%)", f"${daily_goal:.2f}")

with colD:
    st.metric("Risk Deployed", f"${st.session_state.total_risk_deployed:.2f}", f"{portfolio_risk_pct:.1f}%")

if st.session_state.goal_met:
    st.markdown('<div class="risk-critical"><b>🛑 STOP:</b> Daily goal met. No new trades.</div>', unsafe_allow_html=True)

# -------------------------
# 8) SCAN + CHART
# -------------------------
st.divider()
st.subheader("📡 Scan + Levels")

scan = st.button("🔎 SCAN TICKER", type="primary")
df = None
if scan:
    df = engine.fetch(ticker, timeframe)
    if df is None:
        st.error("❌ No data returned. Check ticker or try a different timeframe.")
    else:
        st.success(f"✅ Loaded {ticker} ({timeframe}) — rows: {len(df)}")

if df is None:
    # Auto-load once for convenience (safe)
    df = engine.fetch(ticker, timeframe)

if df is not None and not df.empty:
    px = float(df["Close"].iloc[-1])
    atr = float(df["ATRr_14"].iloc[-1]) if "ATRr_14" in df.columns else float("nan")
    sma20 = float(df["SMA_20"].iloc[-1]) if "SMA_20" in df.columns else float("nan")
    st_dir = "Bullish" if float(df["ST_DIR"].iloc[-1]) > 0 else "Bearish"

    demand, supply = engine.auto_zones(df)

    # Quick phase
    if not np.isnan(sma20) and px > sma20 and st_dir == "Bullish":
        market_phase = "🚀 STRONG UPTREND"
    elif not np.isnan(sma20) and px < sma20 and st_dir == "Bearish":
        market_phase = "📉 STRONG DOWNTREND"
    else:
        market_phase = "🦀 CHOP / TRANSITION"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"${px:.2f}")
    c2.metric("ATR(14)", "N/A" if np.isnan(atr) else f"{atr:.2f}")
    c3.metric("SuperTrend", st_dir)
    c4.metric("Regime", market_phase)

    st.caption("🤖 Auto Zones (structure-based):")
    z1, z2 = st.columns(2)
    buy_zone = z1.number_input("Demand / Buy Zone", value=float(demand), help="Support area where buyers previously defended price.")
    sell_zone = z2.number_input("Supply / Target Zone", value=float(supply), help="Resistance area where sellers previously rejected price.")

    # Chart (last N bars)
    with st.expander("📉 Chart (with SMA20 + SuperTrend + Zones)", expanded=True):
        plot_df = df.tail(120).copy()
        apds = []
        if "SMA_20" in plot_df.columns:
            apds.append(mpf.make_addplot(plot_df["SMA_20"]))
        if "ST_VAL" in plot_df.columns:
            apds.append(mpf.make_addplot(plot_df["ST_VAL"], linestyle="--"))
        hlines = dict(hlines=[buy_zone, sell_zone], linestyle='-.')
        try:
            fig, _ = mpf.plot(
                plot_df,
                type="candle",
                volume=True,
                addplot=apds if apds else None,
                hlines=hlines,
                returnfig=True,
                figratio=(16, 7),
                title=f"{ticker} ({timeframe})"
            )
            st.pyplot(fig)
        except Exception as e:
            st.info("Chart renderer unavailable in this environment — data + levels still valid.")

    # -------------------------
    # 9) IWT SCORECARD + TRADE MATH
    # -------------------------
    st.divider()
    st.subheader("🧮 IWT Scorecard + Execution Math")

    colS1, colS2, colS3, colS4 = st.columns(4)
    freshness = colS1.selectbox("Freshness", [2, 1, 0], index=1, help="2=fresh zone (0 touches), 1=okay (1 touch), 0=stale (2+ touches).")
    time_in_zone = colS2.selectbox("Time in Zone", [2, 1, 0], index=1, help="2=fast rejection (<3 candles), 1=medium, 0=linger.")
    speed_out = colS3.selectbox("Speed Out", [2, 1, 0], index=1, help="2=impulsive / gap, 1=normal, 0=grind.")
    pattern = colS4.selectbox("Pattern", ["Consolidation", "Bull Flag", "Double Bottom", "Parabolic"], index=0)

    # Direction selection (simple)
    direction = st.radio("Primary Scenario", ["Bullish", "Bearish"], horizontal=True, help="Pick the scenario you’re evaluating; the system scores ONE primary plan.")

    if np.isnan(atr) or atr <= 0:
        st.error("❌ ATR not available — cannot compute stop distance reliably.")
        st.stop()

    if direction == "Bullish":
        entry = float(buy_zone)
        stop = entry - (atr * atr_mult)
        target = float(sell_zone)
    else:
        entry = float(sell_zone)  # for shorts, you “enter” at supply
        stop = entry + (atr * atr_mult)
        target = float(buy_zone)

    risk_per_share = abs(entry - stop)
    reward_per_share = abs(target - entry)
    rr = (reward_per_share / risk_per_share) if risk_per_share > 0 else 0.0

    score_rr = 2 if rr >= 3 else 1 if rr >= 2 else 0
    total_score = int(freshness + time_in_zone + speed_out + score_rr)

    # Costs estimate
    est_slippage = entry * (SLIPPAGE_BPS / 10000.0)
    est_commission = COMMISSION_PER_SHARE
    net_reward_per_share = max(0.0, reward_per_share - est_slippage - est_commission)
    net_rr = (net_reward_per_share / risk_per_share) if risk_per_share > 0 else 0.0

    # Sizing
    shares = int(risk_per_trade / risk_per_share) if risk_per_share > 0 else 0
    total_trade_risk = shares * risk_per_share

    # Discipline gates
    can_trade = True
    reasons = []

    if st.session_state.goal_met:
        can_trade = False
        reasons.append("Daily goal already met → STOP.")
    if portfolio_risk_pct > max_portfolio_risk:
        can_trade = False
        reasons.append("Portfolio risk over limit.")
    if shares <= 0:
        can_trade = False
        reasons.append("Position size is 0 (risk too tight / too small $ risk).")

    verdict = "🔴 RED"
    if can_trade:
        if total_score >= 7:
            verdict = "🟢 GREEN"
        elif total_score >= 5:
            verdict = "🟡 YELLOW"
        else:
            verdict = "🔴 RED"

    # Show results
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("Score (0–8)", f"{total_score}/8")
    v2.metric("R/R (Gross)", f"{rr:.2f}")
    v3.metric("R/R (Net est.)", f"{net_rr:.2f}")
    v4.metric("Verdict", verdict)

    st.markdown(f"**Levels (Exact):** Entry **{entry:.4f}** | Stop **{stop:.4f}** | Target **{target:.4f}**")
    st.caption(f"Stop distance: {risk_per_share:.4f} | Shares: {shares} | Total risk: ${total_trade_risk:.2f}")

    if vix is not None and vix_reg in ["HIGH", "CRISIS"]:
        st.markdown('<div class="risk-alert"><b>⚠️ Volatility Note:</b> VIX is high — expect whipsaws. Consider smaller size or no-trade.</div>', unsafe_allow_html=True)

    if reasons:
        st.markdown('<div class="risk-critical"><b>Trade Blocked:</b><br>' + "<br>".join(reasons) + "</div>", unsafe_allow_html=True)

    # -------------------------
    # 10) EXECUTION + JOURNAL (WITH HELP NOTES)
    # -------------------------
    if can_trade and total_score >= 5:
        st.divider()
        st.subheader("⚡ Trade Execution")
        st.caption("💡 PAPER = practice log (no P&L). LIVE = counts as an open position. Only log LIVE if you actually executed with your broker.")

        colE1, colE2 = st.columns(2)

        def make_trade_record(status: str):
            return {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ticker": ticker,
                "sector": SECTOR_MAP.get(ticker, "Unknown"),
                "timeframe": timeframe,
                "pattern": pattern,
                "scenario": direction,
                "entry": float(entry),
                "stop": float(stop),
                "target": float(target),
                "shares": int(shares),
                "score": int(total_score),
                "risk": float(total_trade_risk),
                "rr_gross": float(rr),
                "rr_net_est": float(net_rr),
                "status": status
            }

        with colE1:
            if st.button("📝 Log as PAPER TRADE", type="secondary", help="Logs setup for learning. Does NOT affect P&L or portfolio risk."):
                rec = make_trade_record("PAPER")
                st.session_state.journal.append(rec)
                st.success("📋 Paper trade logged.")

        with colE2:
            if st.button("💵 Log as LIVE TRADE", type="primary", help="Logs as open position (affects portfolio risk). Only if you actually entered the trade."):
                rec = make_trade_record("OPEN")
                st.session_state.journal.append(rec)
                st.session_state.open_positions.append(rec)
                st.session_state.total_risk_deployed += float(total_trade_risk)
                st.success("✅ Live position logged.")
                st.rerun()

    # -------------------------
    # 11) OPEN POSITIONS (WITH HELP NOTES)
    # -------------------------
    if st.session_state.open_positions:
        st.divider()
        st.subheader("📊 Open Positions (Live Trades)")
        st.caption("💡 These are trades you logged as LIVE. Close them here only after you exit in your broker.")

        pos_df = pd.DataFrame(st.session_state.open_positions)
        st.dataframe(pos_df[["timestamp", "ticker", "scenario", "entry", "stop", "target", "shares", "risk", "score"]], use_container_width=True)

        st.markdown("**Close a Position**")
        st.caption("💡 Enter the real exit price from your broker. The system will compute realized P&L.")

        close_ticker = st.selectbox("Select position to close", [p["ticker"] for p in st.session_state.open_positions])
        exit_price = st.number_input("Exit Price", min_value=0.0, value=0.0, help="Use the actual fill price from your broker (not an estimate).")

        if st.button("✅ CLOSE POSITION"):
            if exit_price <= 0:
                st.error("❌ Enter a valid exit price.")
            else:
                for i, p in enumerate(st.session_state.open_positions):
                    if p["ticker"] == close_ticker:
                        direction_mult = 1.0 if p["scenario"] == "Bullish" else -1.0
                        gross_pnl = (exit_price - p["entry"]) * p["shares"] * direction_mult

                        # Apply costs (simple estimate)
                        slippage_cost = p["entry"] * (SLIPPAGE_BPS / 10000.0) * p["shares"]
                        commission_cost = COMMISSION_PER_SHARE * p["shares"]
                        actual_pnl = gross_pnl - slippage_cost - commission_cost

                        closed = dict(p)
                        closed["exit"] = float(exit_price)
                        closed["actual_pnl"] = float(actual_pnl)
                        closed["status"] = "CLOSED"

                        st.session_state.closed_trades.append(closed)
                        st.session_state.daily_pnl += float(actual_pnl)
                        st.session_state.total_risk_deployed -= float(p["risk"])

                        # losing streak tracking
                        if actual_pnl < 0:
                            st.session_state.consecutive_losses += 1
                        else:
                            st.session_state.consecutive_losses = 0

                        if st.session_state.daily_pnl >= daily_goal:
                            st.session_state.goal_met = True

                        st.session_state.open_positions.pop(i)
                        st.success(f"✅ Closed {close_ticker}: P&L = ${actual_pnl:.2f}")
                        st.rerun()
                        break

    # -------------------------
    # 12) JOURNAL + PERFORMANCE (BASIC)
    # -------------------------
    if st.session_state.journal:
        st.divider()
        st.subheader("📓 Session Journal")
        st.caption("💡 Journal everything. Patterns emerge after 20–50 trades, not after 2.")
        jdf = pd.DataFrame(st.session_state.journal)
        st.dataframe(jdf.tail(50), use_container_width=True)

    if st.session_state.closed_trades:
        st.divider()
        st.subheader("📈 Performance Analytics (Simple)")
        closed_df = pd.DataFrame(st.session_state.closed_trades)

        wins = int((closed_df["actual_pnl"] > 0).sum())
        total = len(closed_df)
        win_rate = (wins / total * 100) if total else 0.0
        avg_rr = float(closed_df["rr_gross"].mean()) if total else 0.0
        total_pnl = float(closed_df["actual_pnl"].sum()) if total else 0.0

        a1, a2, a3 = st.columns(3)
        a1.metric("Win Rate", f"{win_rate:.1f}%", f"{wins}/{total}")
        a2.metric("Avg R/R (gross)", f"{avg_rr:.2f}")
        a3.metric("Total P&L", f"${total_pnl:.2f}", f"{(total_pnl/capital)*100:.2f}%")

st.caption("🏛️ Quantum Maestro [TradingBot] — Educational Use Only. Consistency beats intensity.")
