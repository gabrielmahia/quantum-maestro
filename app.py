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
    initial_sidebar_state="expanded"
)

# --- 2. CSS STYLING (Make it look Professional) ---
st.markdown("""
<style>
    .metric-card {
        background-color: #0e1117;
        border: 1px solid #262730;
        border-radius: 5px;
        padding: 15px;
        text-align: center;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

st.title("🏛️ Quantum Maestro V8.1: Global Macro Terminal")

# --- 3. SESSION STATE (The Brain Upgrade) ---
# This keeps data alive between clicks
if 'data' not in st.session_state:
    st.session_state.data = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}
if 'macro' not in st.session_state:
    st.session_state.macro = None

# --- 4. SIDEBAR INPUTS ---
with st.sidebar:
    st.header("1. Configuration")
    ticker = st.text_input("Ticker Symbol", value="NVDA").upper()
    capital = st.number_input("Capital ($)", value=10000)
    risk_per_trade = st.number_input("Risk Amount ($)", value=100)

    st.header("2. Strategy")
    strategy = st.selectbox(
        "Select Strategy",
        ['Long (Buy Stock)', 'Short (Sell Stock)', 'Sell Puts (Income)', 'Sell Calls (Income)']
    )
    stop_mode = st.selectbox("Stop Loss Mode", [1.0, 0.2], format_func=lambda x: "Swing (1.0 ATR)" if x == 1.0 else "IWT Tight (0.2 ATR)")

    # Premium Input (Conditional)
    premium = 0.0
    if "Income" in strategy:
        st.info("💰 Income Mode Active")
        premium = st.number_input("Option Premium ($)", value=0.0, step=0.05)

    st.header("3. IWT Scorecard")
    fresh = st.selectbox("Freshness", [2, 1, 0], format_func=lambda x: f"{x} - {'Perfect' if x==2 else 'Okay' if x==1 else 'Bad'}")
    time_zone = st.selectbox("Time in Zone", [2, 1, 0], format_func=lambda x: f"{x} - {'Fast' if x==2 else 'Medium' if x==1 else 'Slow'}")
    speed = st.selectbox("Speed Out", [2, 1, 0], format_func=lambda x: f"{x} - {'Gap/Fast' if x==2 else 'Candle' if x==1 else 'Grind'}")
    pattern = st.selectbox("Pattern", ['Consolidation', 'Bull Flag', 'Double Bottom', 'Parabolic', 'Gap Fill'])

# --- 5. LOGIC ENGINE ---
class Analyst:
    def fetch_data(self, t):
        try:
            data = yf.Ticker(t).history(period="1y")
            if data.empty: return None
            
            # Indicators
            data.ta.atr(length=14, append=True)
            data.ta.sma(length=20, append=True)
            data.ta.sma(length=50, append=True)
            st_data = data.ta.supertrend(length=10, multiplier=3)
            data['ST_VAL'] = st_data[st_data.columns[0]]
            data['ST_DIR'] = st_data[st_data.columns[1]]
            
            # Volatility
            vol_sma = data['Volume'].rolling(20).mean()
            data['RVOL'] = data['Volume'] / vol_sma
            
            # Structure (SciPy)
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

# --- 6. ACTION BUTTONS (The Control Panel) ---
col_macro, col_scan = st.columns([1, 1])

with col_macro:
    if st.button("🌍 1. Check Macro"):
        with st.spinner("Scanning Global Sensors..."):
            st.session_state.macro = engine.get_macro()

with col_scan:
    if st.button("🔎 2. Scan Ticker"):
        with st.spinner(f"Analyzing {ticker}..."):
            df = engine.fetch_data(ticker)
            if df is not None:
                st.session_state.data = df
                # Calculate metrics once and save them
                price = df['Close'].iloc[-1]
                atr = df['ATRr_14'].iloc[-1]
                supp = df['Min'][df['Min'] < price * 0.99].iloc[-1] if not df['Min'][df['Min'] < price * 0.99].empty else price * 0.9
                res = df['Max'][df['Max'] > price * 1.01].iloc[-1] if not df['Max'][df['Max'] > price * 1.01].empty else price * 1.1
                
                st.session_state.metrics = {
                    "price": price,
                    "atr": atr,
                    "phase": "🚀 UPTREND" if price > df['SMA_20'].iloc[-1] else "📉 DOWNTREND",
                    "supp": supp,
                    "res": res,
                    "rvol": df['RVOL'].iloc[-1]
                }
            else:
                st.error("Ticker not found.")

# --- 7. DISPLAY LOGIC (Only runs if data exists in memory) ---

# A. MACRO SECTION
if st.session_state.macro:
    m = st.session_state.macro
    with st.expander("🌍 Global Macro Context", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("S&P 500 Futures", f"{m['sp_change']:.2f}%")
        c2.metric("VIX (Fear Gauge)", f"{m['vix']:.2f}")
        c3.metric("Gold (Chaos)", f"{m['gold_change']:.2f}%")
        
        if m['sp_change'] < -0.5 and m['vix'] > 20:
            st.error("🐻 BIAS: BEARISH. Cash is King.")
        elif m['sp_change'] > 0.5 and m['vix'] < 20:
            st.success("🐂 BIAS: BULLISH. Buying Dips Allowed.")
        else:
            st.warning("🦀 BIAS: CHOPPY. Be selective.")

# B. TICKER SECTION
if st.session_state.data is not None:
    df = st.session_state.data
    m = st.session_state.metrics
    
    st.divider()
    
    # 1. Dashboard Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ticker", ticker)
    c2.metric("Price", f"${m['price']:.2f}")
    c3.metric("Trend", m['phase'])
    c4.metric("Volume", f"{m['rvol']:.1f}x Avg")

    # 2. Strategy Engine (Live Calculation)
    # This runs every time you change a sidebar input because it uses stored data
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
    
    # 3. Charting (The Visuals)
    st.subheader(f"📊 Technical Structure: {ticker}")
    
    # Slice for speed and readability
    chart_slice = df.iloc[-60:]
    
    # Create the chart with Zones
    fig, ax = mpf.plot(
        chart_slice, 
        type='candle', 
        style='yahoo', 
        volume=True,
        addplot=[
            mpf.make_addplot(chart_slice['SMA_20'], color='blue', width=1.5),
            mpf.make_addplot(chart_slice['ST_VAL'], color='orange', linestyle='--')
        ],
        hlines=dict(hlines=[entry, target if "Income" not in strategy else entry], colors=['green','red'], linestyle='-.', linewidths=1.5),
        returnfig=True,
        figsize=(12, 6) # Bigger layout for Web
    )
    st.pyplot(fig)

    # 4. The Verdict Console
    st.divider()
    st.subheader("📝 The Quantum Verdict")
    
    score_rr = 2 if rr >= 3 or ("Income" in strategy and rr > 0.1) else 1 if rr >= 2 else 0
    total_score = fresh + time_zone + speed + score_rr
    
    col_verdict, col_plan = st.columns([1, 2])
    
    with col_verdict:
        if total_score >= 7:
            st.success(f"## 🟢 GREEN LIGHT\n**Score: {total_score}/8**")
        elif total_score >= 5:
            st.warning(f"## 🟡 YELLOW LIGHT\n**Score: {total_score}/8**")
        else:
            st.error(f"## 🔴 NO TRADE\n**Score: {total_score}/8**")
            
    with col_plan:
        st.markdown(f"#### 🏹 Execution Plan: {strategy}")
        if "Income" in strategy:
            shares = int(capital / entry) if entry > 0 else 0
            contracts = shares // 100
            collateral = contracts * 100 * entry
            income = contracts * 100 * premium
            roi = (premium/entry)*100 if entry>0 else 0
            
            st.info(f"""
            - **Strike (Entry):** ${entry:.2f}
            - **Sell Contracts:** {contracts}
            - **Collateral:** ${collateral:,.2f}
            - **Instant Income:** ${income:.2f} (ROI: {roi:.2f}%)
            """)
        else:
            shares = int(risk_per_trade / risk) if risk > 0 else 0
            st.info(f"""
            - **Entry:** ${entry:.2f}
            - **Stop Loss:** ${stop:.2f}
            - **Target:** ${target:.2f}
            - **Size:** {shares} shares (Risking ${risk_per_trade})
            - **R/R Ratio:** {rr:.2f}
            """)

else:
    st.info("👈 Enter a ticker in the sidebar and click 'Scan Ticker' to begin.")
