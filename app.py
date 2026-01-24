# app.py
import streamlit as st
import yfinance as yf
import pandas_ta as ta
import mplfinance as mpf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Quantum Maestro Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏛️"
)

# --- 2. CSS STYLING ---
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 4px;
        height: 3em;
        font-weight: 600;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] {
        background-color: #f0f2f6;
        border: 1px solid #d6d6d6;
        border-radius: 6px;
        padding: 10px 15px;
    }
    @media (prefers-color-scheme: dark) {
        div[data-testid="stMetric"] {
            background-color: #1e2127;
            border: 1px solid #30333d;
        }
    }
    .audit-pass { color: #00e676; font-weight: bold; }
    .audit-fail { color: #ff1744; font-weight: bold; }
    .audit-warn { color: #ff9100; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🏛️ Quantum Maestro: Mentor Edition")

# --- 3. SESSION STATE ---
if 'data' not in st.session_state: st.session_state.data = None
if 'metrics' not in st.session_state: st.session_state.metrics = {}
if 'macro' not in st.session_state: st.session_state.macro = None

# --- 4. SIDEBAR ---
with st.sidebar:
    st.header("1. Configuration")
    c1, c2 = st.columns(2)
    with c1:
        ticker = st.text_input("Ticker", value="NVDA").upper()
    with c2:
        risk_per_trade = st.number_input("Risk $", value=100)
    capital = st.number_input("Total Capital ($)", value=10000)

    st.divider()
    st.header("2. Strategy Board")
    strategy = st.selectbox("Execution Mode", ['Long (Buy Stock)', 'Short (Sell Stock)', 'Sell Puts (Income)', 'Sell Calls (Income)'])
    stop_mode = st.selectbox("Stop Type", [1.0, 0.2], format_func=lambda x: "Safe Swing (1.0 ATR)" if x == 1.0 else "IWT Tight (0.2 ATR)")

    premium = 0.0
    if "Income" in strategy:
        st.success("💰 Income Mode")
        premium = st.number_input("Option Premium ($)", value=0.0, step=0.05)

    st.divider()
    st.header("3. IWT Scorecard")
    c3, c4 = st.columns(2)
    with c3:
        fresh = st.selectbox("Freshness", [2, 1, 0], format_func=lambda x: {2:'2-Fresh', 1:'1-Used', 0:'0-Stale'}[x])
        speed = st.selectbox("Speed Out", [2, 1, 0], format_func=lambda x: {2:'2-Fast', 1:'1-Avg', 0:'0-Slow'}[x])
    with c4:
        time_zone = st.selectbox("Time in Zone", [2, 1, 0], format_func=lambda x: {2:'2-Short', 1:'1-Med', 0:'0-Long'}[x])
        pattern = st.selectbox("Pattern", ['Consolidation', 'Bull Flag', 'Double Bottom', 'Parabolic', 'Gap Fill'])

# --- 5. LOGIC ENGINE ---
class Analyst:
    def fetch_data(self, t):
        try:
            data = yf.Ticker(t).history(period="1y")
            if data.empty: return None
            data.ta.atr(length=14, append=True)
            data.ta.sma(length=20, append=True)
            data.ta.sma(length=50, append=True)
            st_data = data.ta.supertrend(length=10, multiplier=3)
            data['ST_VAL'] = st_data[st_data.columns[0]]
            data['ST_DIR'] = st_data[st_data.columns[1]]
            vol_sma = data['Volume'].rolling(20).mean()
            data['RVOL'] = data['Volume'] / vol_sma
            data['Min'] = data.iloc[argrelextrema(data.Close.values, np.less_equal, order=5)[0]]['Close']
            data['Max'] = data.iloc[argrelextrema(data.Close.values, np.greater_equal, order=5)[0]]['Close']
            return data
        except: return None

    def get_macro(self):
        tickers = {"S&P 500": "ES=F", "VIX": "^VIX", "Gold": "GC=F"}
        try:
            df = yf.download(list(tickers.values()), period="5d", progress=False)['Close']
            return {
                "sp_change": ((df["ES=F"].iloc[-1] - df["ES=F"].iloc[-2])/df["ES=F"].iloc[-2])*100,
                "vix": df["^VIX"].iloc[-1],
                "gold_change": ((df["GC=F"].iloc[-1] - df["GC=F"].iloc[-2])/df["GC=F"].iloc[-2])*100
            }
        except: return None

engine = Analyst()

# --- 6. CONTROL PANEL ---
c_macro, c_scan = st.columns([1, 1])
with c_macro:
    if st.button("🌍 1. CHECK MACRO", type="secondary"):
        with st.spinner("Scanning Global Sensors..."):
            st.session_state.macro = engine.get_macro()
with c_scan:
    if st.button("🔎 2. SCAN TICKER", type="primary"):
        with st.spinner(f"Parsing {ticker}..."):
            df = engine.fetch_data(ticker)
            if df is not None:
                st.session_state.data = df
                price = df['Close'].iloc[-1]
                supp = df['Min'][df['Min'] < price * 0.99].iloc[-1] if not df['Min'][df['Min'] < price * 0.99].empty else price * 0.9
                res = df['Max'][df['Max'] > price * 1.01].iloc[-1] if not df['Max'][df['Max'] > price * 1.01].empty else price * 1.1
                st.session_state.metrics = {
                    "price": price, "atr": df['ATRr_14'].iloc[-1],
                    "phase": "🚀 UPTREND" if price > df['SMA_20'].iloc[-1] else "📉 DOWNTREND",
                    "supp": supp, "res": res, "rvol": df['RVOL'].iloc[-1]
                }
            else:
                st.error("Ticker not found.")

# --- 7. DASHBOARD ---
if st.session_state.macro:
    m = st.session_state.macro
    with st.expander("🌍 Global Context", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Futures", f"{m['sp_change']:.2f}%")
        c2.metric("VIX", f"{m['vix']:.2f}")
        c3.metric("Gold", f"{m['gold_change']:.2f}%")
        if m['sp_change'] < -0.5 and m['vix'] > 20: c4.error("🐻 BEAR")
        elif m['sp_change'] > 0.5 and m['vix'] < 20: c4.success("🐂 BULL")
        else: c4.warning("🦀 CHOP")

if st.session_state.data is not None:
    df = st.session_state.data
    m = st.session_state.metrics
    
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"${m['price']:.2f}")
    c2.metric("Trend", m['phase'])
    c3.metric("Volume", f"{m['rvol']:.1f}x")
    
    # Strategy Calculation
    if "Short" in strategy:
        entry, target = m['res'], m['supp']
        stop = entry + (m['atr'] * stop_mode)
        risk, reward = stop - entry, entry - target
    elif "Income" in strategy:
        entry = m['supp'] if "Puts" in strategy else m['res']
        stop = entry; risk = entry * 0.1; reward = premium
    else: # Long
        entry, target = m['supp'], m['res']
        stop = entry - (m['atr'] * stop_mode)
        risk, reward = entry - stop, target - entry
    
    rr = reward / risk if risk > 0 else 0
    c4.metric("R/R Ratio", f"{rr:.2f}", delta="Target > 2.0" if rr >=2 else "Low", delta_color="normal" if rr>=2 else "inverse")

    # Chart
    chart_slice = df.iloc[-60:]
    fig, ax = mpf.plot(
        chart_slice, type='candle', style='yahoo', volume=True,
        addplot=[mpf.make_addplot(chart_slice['SMA_20'], color='blue'), mpf.make_addplot(chart_slice['ST_VAL'], color='orange')],
        hlines=dict(hlines=[entry, target if "Income" not in strategy else entry], colors=['green','red'], linestyle='-.'),
        returnfig=True, figsize=(12, 5), fontscale=0.8
    )
    st.pyplot(fig)

    # --- THE AUDIT SYSTEM (New Feature) ---
    st.markdown("### 📝 The Quantum Verdict")
    score_rr = 2 if rr >= 3 or ("Income" in strategy and rr > 0.1) else 1 if rr >= 2 else 0
    total_score = fresh + time_zone + speed + score_rr
    
    col_verdict, col_audit = st.columns([1, 1])
    
    with col_verdict:
        # 1. The Big Verdict
        if total_score >= 7:
            st.success(f"## 🟢 GREEN LIGHT\n**Score: {total_score}/8**")
            st.caption("Perfect setup. High probability.")
        elif total_score >= 5:
            st.warning(f"## 🟡 YELLOW LIGHT\n**Score: {total_score}/8**")
            st.caption("Decent setup, but flawed. Reduce size.")
        else:
            st.error(f"## 🔴 RED LIGHT\n**Score: {total_score}/8**")
            st.caption("Low probability. Stay away.")

    with col_audit:
        # 2. The Detailed Checklist
        st.markdown("**📋 Setup Audit:**")
        
        # Freshness Check
        if fresh == 2: st.markdown("✅ **Freshness:** Perfect (2/2)")
        elif fresh == 1: st.markdown("⚠️ **Freshness:** Used level (1/2)")
        else: st.markdown("❌ **Freshness:** Stale/Dirty (0/2)")
        
        # Time Check
        if time_zone == 2: st.markdown("✅ **Time:** Fast Rejection (2/2)")
        elif time_zone == 1: st.markdown("⚠️ **Time:** Lingered (1/2)")
        else: st.markdown("❌ **Time:** Stuck in zone (0/2)")
        
        # R/R Check
        if score_rr == 2: st.markdown(f"✅ **Reward/Risk:** Excellent ({rr:.2f})")
        elif score_rr == 1: st.markdown(f"⚠️ **Reward/Risk:** Acceptable ({rr:.2f})")
        else: st.markdown(f"❌ **Reward/Risk:** Poor (<2.0)")

    # --- BROKER SLIP ---
    st.markdown("---")
    st.subheader("🧾 Trade Execution")
    
    if "Income" in strategy:
        contracts = int((capital / entry) // 100) if entry > 0 else 0
        roi = (premium/entry)*100 if entry>0 else 0
        st.success(f"**INCOME PLAN:** Selling {contracts} Contracts | **Income:** ${contracts*100*premium:.2f} | **ROI:** {roi:.2f}%")
        st.code(f"SELL TO OPEN: {ticker} {entry:.2f} Strike | Expiration: 30 Days Out")
        
    else:
        shares = int(risk_per_trade / risk) if risk > 0 else 0
        order_type = "BUY" if "Long" in strategy else "SELL SHORT"
        
        col_slip_1, col_slip_2 = st.columns([2, 1])
        with col_slip_1:
            st.code(f"""
            ACTION:      {order_type}
            SHARES:      {shares}
            LIMIT PRICE: ${entry:.2f}
            ---------------------------
            STOP LOSS:   ${stop:.2f}
            TAKE PROFIT: ${target:.2f}
            """, language="yaml")
        with col_slip_2:
            st.info(f"**RISK:** ${risk_per_trade:.2f}\n\n**REWARD:** ${(shares*(target-entry) if 'Long' in strategy else shares*(entry-target)):.2f}")

else:
    st.info("👈 Please enter a Ticker on the left and click '2. SCAN TICKER'")
