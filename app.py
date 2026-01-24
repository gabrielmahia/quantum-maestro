# app.py
import streamlit as st
import yfinance as yf
import pandas_ta as ta
import mplfinance as mpf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema

# --- PAGE CONFIG ---
st.set_page_config(page_title="Quantum Maestro V8", layout="wide")
st.title("🏛️ Quantum Maestro V8.0: Global Macro Terminal")

# --- SIDEBAR: INPUTS ---
st.sidebar.header("1. Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", value="NVDA").upper()
capital = st.sidebar.number_input("Capital ($)", value=10000)
risk_per_trade = st.sidebar.number_input("Risk Amount ($)", value=100)

st.sidebar.header("2. Strategy")
strategy = st.sidebar.selectbox(
    "Select Strategy",
    ['Long (Buy Stock)', 'Short (Sell Stock)', 'Sell Puts (Income)', 'Sell Calls (Income)']
)
stop_mode = st.sidebar.selectbox("Stop Loss Mode", [1.0, 0.2], format_func=lambda x: "Swing (1.0 ATR)" if x == 1.0 else "IWT Tight (0.2 ATR)")

# Premium input only shows if Income strategy is selected
premium = 0.0
if "Income" in strategy:
    premium = st.sidebar.number_input("Option Premium ($)", value=0.0, step=0.05)

st.sidebar.header("3. IWT Scorecard")
fresh = st.sidebar.selectbox("Freshness", [2, 1, 0], format_func=lambda x: f"{x} - {'Perfect' if x==2 else 'Okay' if x==1 else 'Bad'}")
time_zone = st.sidebar.selectbox("Time in Zone", [2, 1, 0], format_func=lambda x: f"{x} - {'Fast' if x==2 else 'Medium' if x==1 else 'Slow'}")
speed = st.sidebar.selectbox("Speed Out", [2, 1, 0], format_func=lambda x: f"{x} - {'Gap/Fast' if x==2 else 'Candle' if x==1 else 'Grind'}")
pattern = st.sidebar.selectbox("Pattern", ['Consolidation', 'Bull Flag', 'Double Bottom', 'Parabolic', 'Gap Fill'])

# --- LOGIC ENGINE ---
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

# --- MAIN DASHBOARD ---
engine = Analyst()

# 1. MACRO BANNER
macro = engine.get_macro()
if macro:
    c1, c2, c3 = st.columns(3)
    c1.metric("S&P 500 Fut", f"{macro['sp_change']:.2f}%", delta_color="normal")
    c2.metric("VIX (Fear)", f"{macro['vix']:.2f}", delta_color="inverse")
    c3.metric("Gold (Chaos)", f"{macro['gold_change']:.2f}%")
    
    if macro['sp_change'] < -0.5 and macro['vix'] > 20:
        st.error("🐻 MACRO ALERT: Bearish Bias. Cash or Shorts recommended.")
    elif macro['sp_change'] > 0.5 and macro['vix'] < 20:
        st.success("🐂 MACRO ALERT: Bullish Bias. Buying Dips allowed.")
    else:
        st.info("⚖️ MACRO ALERT: Mixed/Choppy. Stick to the chart.")

# 2. TICKER ANALYSIS
if st.sidebar.button("Run Analysis"):
    with st.spinner(f"Analyzing {ticker}..."):
        df = engine.fetch_data(ticker)
        
        if df is not None:
            # Current Price & Phase
            price = df['Close'].iloc[-1]
            atr = df['ATRr_14'].iloc[-1]
            phase = "🚀 UPTREND" if price > df['SMA_20'].iloc[-1] else "📉 DOWNTREND"
            
            # Auto Zones
            supp = df['Min'][df['Min'] < price * 0.99].iloc[-1] if not df['Min'][df['Min'] < price * 0.99].empty else price * 0.9
            res = df['Max'][df['Max'] > price * 1.01].iloc[-1] if not df['Max'][df['Max'] > price * 1.01].empty else price * 1.1
            
            # Strategy Logic
            if "Short" in strategy:
                entry, target = res, supp
                stop = entry + (atr * stop_mode)
                risk = stop - entry
                reward = entry - target
            elif "Income" in strategy:
                entry = supp if "Puts" in strategy else res
                stop = entry # Mental stop
                risk = entry * 0.1 # Contextual risk
                reward = premium
            else: # Long
                entry, target = supp, res
                stop = entry - (atr * stop_mode)
                risk = entry - stop
                reward = target - entry
            
            # Math
            rr = reward / risk if risk > 0 else 0
            
            # Display Metrics
            col1, col2, col3 = st.columns(3)
            col1.metric("Current Price", f"${price:.2f}")
            col2.metric("Market Phase", phase)
            col3.metric("Auto-Entry Zone", f"${entry:.2f}")
            
            # Chart
            st.subheader("Structure & Trends")
            chart_data = df.iloc[-60:]
            apds = [
                mpf.make_addplot(chart_data['SMA_20'], color='blue'),
                mpf.make_addplot(chart_data['ST_VAL'], color='orange')
            ]
            fig, ax = mpf.plot(chart_data, type='candle', style='yahoo', addplot=apds, 
                               hlines=dict(hlines=[entry, target if "Income" not in strategy else entry], colors=['green','red']),
                               returnfig=True, volume=True)
            st.pyplot(fig)
            
            # Scorecard & Verdict
            st.subheader("The Quantum Verdict")
            
            score_rr = 2 if rr >= 3 or ("Income" in strategy and rr > 0.1) else 1 if rr >= 2 else 0
            total_score = fresh + time_zone + speed + score_rr
            
            if total_score >= 7:
                st.success(f"🟢 GREEN LIGHT (Score: {total_score}/8)")
            elif total_score >= 5:
                st.warning(f"🟡 YELLOW LIGHT (Score: {total_score}/8)")
            else:
                st.error(f"🔴 NO TRADE (Score: {total_score}/8)")
                
            st.write(f"**Plan:** Entry ${entry:.2f} | Stop ${stop:.2f} | Target ${target if 'Income' not in strategy else 'Expire'}")
            
            if "Income" in strategy and premium > 0:
                roi = (premium / entry) * 100
                st.info(f"💰 Cash-on-Cash ROI: {roi:.2f}%")
                
        else:
            st.error("Ticker not found.")
