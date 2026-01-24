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

# --- 2. THE "BEAUTIFIER" (Custom CSS) ---
# This block forces the app to look like a Pro Terminal
st.markdown("""
<style>
    /* Main Background & Text */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background-color: #1e2127;
        border: 1px solid #30333d;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.9em;
        color: #a0a0a0;
    }
    /* Custom Buttons */
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        height: 3em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    /* Dividers */
    hr {
        margin-top: 0.5em;
        margin-bottom: 0.5em;
        border: 0;
        border-top: 1px solid #30333d;
    }
    /* Verdict Cards */
    .verdict-box {
        padding: 20px;
        border-radius: 10px;
        margin-top: 10px;
        border: 1px solid #ffffff20;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏛️ Quantum Maestro: Pro Terminal")

# --- 3. SESSION STATE ---
if 'data' not in st.session_state:
    st.session_state.data = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}
if 'macro' not in st.session_state:
    st.session_state.macro = None

# --- 4. SIDEBAR INPUTS ---
with st.sidebar:
    st.header("1. CONFIGURATION")
    c1, c2 = st.columns(2)
    with c1:
        ticker = st.text_input("Ticker", value="NVDA").upper()
    with c2:
        risk_per_trade = st.number_input("Risk $", value=100)
    capital = st.number_input("Total Capital ($)", value=10000)

    st.divider()
    st.header("2. STRATEGY BOARD")
    strategy = st.selectbox(
        "Execution Mode",
        ['Long (Buy Stock)', 'Short (Sell Stock)', 'Sell Puts (Income)', 'Sell Calls (Income)']
    )
    stop_mode = st.selectbox("Stop Type", [1.0, 0.2], format_func=lambda x: "Safe Swing (1.0 ATR)" if x == 1.0 else "IWT Tight (0.2 ATR)")

    premium = 0.0
    if "Income" in strategy:
        st.success("💰 Income Mode Active")
        premium = st.number_input("Option Premium ($)", value=0.0, step=0.05)

    st.divider()
    st.header("3. IWT SCORECARD")
    # Using columns for tighter layout
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
            
            # Math
            data.ta.atr(length=14, append=True)
            data.ta.sma(length=20, append=True)
            data.ta.sma(length=50, append=True)
            st_data = data.ta.supertrend(length=10, multiplier=3)
            data['ST_VAL'] = st_data[st_data.columns[0]]
            data['ST_DIR'] = st_data[st_data.columns[1]]
            
            # RVOL
            vol_sma = data['Volume'].rolling(20).mean()
            data['RVOL'] = data['Volume'] / vol_sma
            
            # SciPy Structure
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

# --- 6. ACTION PANEL ---
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
                    "price": price,
                    "atr": df['ATRr_14'].iloc[-1],
                    "phase": "🚀 UPTREND" if price > df['SMA_20'].iloc[-1] else "📉 DOWNTREND",
                    "supp": supp,
                    "res": res,
                    "rvol": df['RVOL'].iloc[-1]
                }
            else:
                st.error("Ticker not found.")

# --- 7. DISPLAY LAYER ---

# A. MACRO HEADER
if st.session_state.macro:
    m = st.session_state.macro
    with st.container():
        st.markdown("#### 🌍 Global Atmosphere")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("S&P Futures", f"{m['sp_change']:.2f}%", delta_color="normal")
        c2.metric("VIX (Fear)", f"{m['vix']:.2f}", delta_color="inverse")
        c3.metric("Gold", f"{m['gold_change']:.2f}%")
        
        # Macro Logic
        if m['sp_change'] < -0.5 and m['vix'] > 20:
            c4.error("🐻 BEAR BIAS")
            st.caption("Advice: Selling pressure is high. Favor Shorts or Cash.")
        elif m['sp_change'] > 0.5 and m['vix'] < 20:
            c4.success("🐂 BULL BIAS")
            st.caption("Advice: Trend is healthy. Buy Dips.")
        else:
            c4.warning("🦀 CHOPPY")
            st.caption("Advice: Market is sideways. Be selective.")
    st.divider()

# B. TICKER DATA
if st.session_state.data is not None:
    df = st.session_state.data
    m = st.session_state.metrics
    
    # 1. Ticker Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Current Price", f"${m['price']:.2f}")
    c2.metric("Trend Phase", m['phase'])
    c3.metric("Volume (RVOL)", f"{m['rvol']:.1f}x")
    
    # Strategy Calculation
    if "Short" in strategy:
        entry, target = m['res'], m['supp']
        stop = entry + (m['atr'] * stop_mode)
        risk = stop - entry
        reward = entry - target
    elif "Income" in strategy:
        entry = m['supp'] if "Puts" in strategy else m['res']
        stop = entry 
        risk = entry * 0.1 
        reward = premium
    else: # Long
        entry, target = m['supp'], m['res']
        stop = entry - (m['atr'] * stop_mode)
        risk = entry - stop
        reward = target - entry
    
    rr = reward / risk if risk > 0 else 0
    c4.metric("Risk/Reward", f"{rr:.2f}")

    # 2. Chart
    chart_slice = df.iloc[-60:]
    fig, ax = mpf.plot(
        chart_slice, type='candle', style='nightclouds', volume=True,
        addplot=[
            mpf.make_addplot(chart_slice['SMA_20'], color='#2962ff'),
            mpf.make_addplot(chart_slice['ST_VAL'], color='#ff6d00')
        ],
        hlines=dict(hlines=[entry, target if "Income" not in strategy else entry], colors=['#00e676','#ff1744'], linestyle='-.'),
        returnfig=True, figsize=(12, 5), fontscale=0.8
    )
    st.pyplot(fig)

    # 3. THE EXPLAINER ENGINE (Verdict Logic)
    st.markdown("### 📝 The Quantum Verdict")
    
    # Calculate Score
    score_rr = 2 if rr >= 3 or ("Income" in strategy and rr > 0.1) else 1 if rr >= 2 else 0
    total_score = fresh + time_zone + speed + score_rr
    
    # Generate "Why" Description
    reasons = []
    if fresh < 2: reasons.append("Level is not fresh")
    if time_zone < 2: reasons.append("Spent too long in zone")
    if speed < 2: reasons.append("Slow momentum leaving zone")
    if score_rr == 0: reasons.append(f"R/R ({rr:.2f}) is below 2.0")
    
    reason_text = " • ".join(reasons) if reasons else "All Odd Enhancers aligned."

    # Render Verdict
    if total_score >= 7:
        st.success(f"### 🟢 GREEN LIGHT (Score: {total_score}/8)")
        st.write(f"**Confidence:** High. {reason_text}")
    elif total_score >= 5:
        st.warning(f"### 🟡 YELLOW LIGHT (Score: {total_score}/8)")
        st.write(f"**Wait for Confirmation.** Why? {reason_text}")
        st.write("*Tip: Wait for a reversal candlestick pattern before entering.*")
    else:
        st.error(f"### 🔴 RED LIGHT (Score: {total_score}/8)")
        st.write(f"**No Trade.** Why? {reason_text}")

    # 4. Execution Plan (Card Style)
    with st.expander("🏹 View Execution Plan", expanded=True):
        if "Income" in strategy:
            contracts = int((capital / entry) // 100) if entry > 0 else 0
            collateral = contracts * 100 * entry
            income = contracts * 100 * premium
            roi = (premium/entry)*100 if entry>0 else 0
            
            st.info(f"""
            **INCOME SETUP:**
            - **Sell:** {contracts} Contracts of the ${entry:.2f} Strike
            - **Collect:** ${income:.2f} Premium
            - **Collateral:** ${collateral:,.2f}
            - **ROI:** {roi:.2f}% Cash-on-Cash
            """)
        else:
            shares = int(risk_per_trade / risk) if risk > 0 else 0
            st.info(f"""
            **SWING SETUP:**
            - **Order:** {strategy} {shares} shares @ ${entry:.2f}
            - **Stop:** ${stop:.2f} (Risking ${risk_per_trade})
            - **Target:** ${target:.2f}
            """)

else:
    st.info("👈 Enter a ticker in the sidebar and click 'Scan Ticker' to begin.")
