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

# --- 2. CLEAN UI (Mentor Mode) ---
st.markdown("""
<style>
    /* Button Styling */
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        height: 3em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    /* Metric Cards */
    div[data-testid="stMetric"] {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    /* Broker Slip Box */
    .broker-slip {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6; /* Light gray for contrast */
        border-left: 5px solid #2962ff;
        margin-top: 20px;
    }
    /* Dark mode override */
    @media (prefers-color-scheme: dark) {
        div[data-testid="stMetric"] { border: 1px solid #30333d; }
        .broker-slip { background-color: #1e2127; border-left: 5px solid #2962ff; }
    }
</style>
""", unsafe_allow_html=True)

st.title("🏛️ Quantum Maestro: Mentor Edition")

# --- 3. SESSION STATE ---
if 'data' not in st.session_state: st.session_state.data = None
if 'metrics' not in st.session_state: st.session_state.metrics = {}
if 'macro' not in st.session_state: st.session_state.macro = None

# --- 4. SIDEBAR (With Tooltips) ---
with st.sidebar:
    st.header("1. CONFIGURATION")
    c1, c2 = st.columns(2)
    with c1:
        ticker = st.text_input("Ticker", value="NVDA", help="Enter the stock symbol (e.g., TSLA, AMD).").upper()
    with c2:
        risk_per_trade = st.number_input("Risk $", value=100, help="The maximum amount you are willing to lose if the trade fails.")
    capital = st.number_input("Total Capital ($)", value=10000, help="Your total account balance/buying power.")

    st.divider()
    st.header("2. STRATEGY BOARD")
    strategy = st.selectbox(
        "Execution Mode",
        ['Long (Buy Stock)', 'Short (Sell Stock)', 'Sell Puts (Income)', 'Sell Calls (Income)'],
        help="Long: Profit if price goes UP. Short: Profit if price goes DOWN. Income: Get paid to wait (Options)."
    )
    stop_mode = st.selectbox(
        "Stop Type", 
        [1.0, 0.2], 
        format_func=lambda x: "Safe Swing (1.0 ATR)" if x == 1.0 else "IWT Tight (0.2 ATR)",
        help="Swing (1.0): Gives the stock room to breathe. Tight (0.2): Very close stop, high risk of stopping out early, but huge Reward/Risk ratio."
    )

    premium = 0.0
    if "Income" in strategy:
        st.success("💰 Income Mode Active")
        premium = st.number_input("Option Premium ($)", value=0.0, step=0.05, help="The price per share the market will pay you for the contract.")

    st.divider()
    st.header("3. IWT SCORECARD")
    st.caption("Rate the setup quality (0-2 points each)")
    
    c3, c4 = st.columns(2)
    with c3:
        fresh = st.selectbox(
            "Freshness", [2, 1, 0], 
            format_func=lambda x: {2:'2-Fresh', 1:'1-Used', 0:'0-Stale'}[x],
            help="Fresh: Price has NOT touched this level recently (Banks still have orders there). Stale: Price has bounced here many times (Orders are depleted)."
        )
        speed = st.selectbox(
            "Speed Out", [2, 1, 0], 
            format_func=lambda x: {2:'2-Fast', 1:'1-Avg', 0:'0-Slow'}[x],
            help="How fast did price leave this zone last time? Fast/Gap = Strong Institutional Buying."
        )
    with c4:
        time_zone = st.selectbox(
            "Time in Zone", [2, 1, 0], 
            format_func=lambda x: {2:'2-Short', 1:'1-Med', 0:'0-Long'}[x],
            help="Did price hang around? Short time = Strong rejection. Long time = Weak rejection."
        )
        pattern = st.selectbox(
            "Pattern", ['Consolidation', 'Bull Flag', 'Double Bottom', 'Parabolic', 'Gap Fill'],
            help="Select the specific chart formation you see."
        )

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
            # RVOL & Structure
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

# --- 6. ACTION PANEL ---
c_macro, c_scan = st.columns([1, 1])
with c_macro:
    if st.button("🌍 1. CHECK MACRO", type="secondary"):
        with st.spinner("Analyzing Global Markets..."):
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

# --- 7. DISPLAY LAYER ---
if st.session_state.macro:
    m = st.session_state.macro
    with st.container():
        st.markdown("#### 🌍 Global Context")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Futures", f"{m['sp_change']:.2f}%", help="S&P 500 Pre-market direction.")
        c2.metric("VIX", f"{m['vix']:.2f}", help="Fear Gauge. >20 means high risk/volatility.")
        c3.metric("Gold", f"{m['gold_change']:.2f}%")
        if m['sp_change'] < -0.5 and m['vix'] > 20:
            c4.error("🐻 BEAR")
        elif m['sp_change'] > 0.5 and m['vix'] < 20:
            c4.success("🐂 BULL")
        else:
            c4.warning("🦀 CHOP")
    st.divider()

if st.session_state.data is not None:
    df = st.session_state.data
    m = st.session_state.metrics
    
    # 1. Metrics with Explanations
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"${m['price']:.2f}")
    c2.metric("Trend", m['phase'], help="Based on 20-Day Moving Average.")
    c3.metric("RVOL", f"{m['rvol']:.1f}x", help="Relative Volume. >1.0 means higher than average interest.")
    
    # Strategy Calc
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
    c4.metric("R/R Ratio", f"{rr:.2f}", delta="Good (>2.0)" if rr >=2 else "Poor (<2.0)", delta_color="normal" if rr>=2 else "inverse", help="You must make at least $2 for every $1 you risk.")

    # 2. Chart
    chart_slice = df.iloc[-60:]
    fig, ax = mpf.plot(
        chart_slice, type='candle', style='yahoo', volume=True,
        addplot=[mpf.make_addplot(chart_slice['SMA_20'], color='blue'), mpf.make_addplot(chart_slice['ST_VAL'], color='orange')],
        hlines=dict(hlines=[entry, target if "Income" not in strategy else entry], colors=['green','red'], linestyle='-.'),
        returnfig=True, figsize=(12, 5), fontscale=0.8
    )
    st.pyplot(fig)

    # 3. VERDICT & EDUCATIONAL FEEDBACK
    st.markdown("### 📝 The Quantum Verdict")
    score_rr = 2 if rr >= 3 or ("Income" in strategy and rr > 0.1) else 1 if rr >= 2 else 0
    total_score = fresh + time_zone + speed + score_rr
    
    # Progress Bar Visualization
    st.progress(total_score / 8, text=f"Trade Quality: {int((total_score/8)*100)}%")

    reasons = []
    if fresh < 2: reasons.append("Banks have used this level before (Not Fresh)")
    if time_zone < 2: reasons.append("Price lingered too long (Weak Rejection)")
    if speed < 2: reasons.append("Exit momentum was slow")
    if score_rr == 0: reasons.append(f"Reward/Risk is only {rr:.2f} (Need 2.0+)")
    
    if total_score >= 7:
        st.success(f"### 🟢 GREEN LIGHT ({total_score}/8)")
        st.write("Excellent setup. All systems go.")
    elif total_score >= 5:
        st.warning(f"### 🟡 YELLOW LIGHT ({total_score}/8)")
        st.write(f"**Caution needed.** Issues: {', '.join(reasons)}.")
    else:
        st.error(f"### 🔴 RED LIGHT ({total_score}/8)")
        st.write(f"**Do not trade.** Issues: {', '.join(reasons)}.")

    # 4. THE BROKER SLIP (The Newbie Helper)
    st.markdown("---")
    st.subheader("🧾 Broker Execution Instructions")
    
    if "Income" in strategy:
        contracts = int((capital / entry) // 100) if entry > 0 else 0
        collateral = contracts * 100 * entry
        income = contracts * 100 * premium
        st.info(f"**INCOME STRATEGY:** Sell to Open (STO) **{contracts}** Contracts of **{ticker}** | Strike: **${entry:.2f}** | Expiration: (Choose 2-4 weeks out)")
    else:
        shares = int(risk_per_trade / risk) if risk > 0 else 0
        order_type = "BUY" if "Long" in strategy else "SELL SHORT"
        limit_price = f"${entry:.2f}"
        
        st.code(f"""
        ACTION:      {order_type}
        QUANTITY:    {shares} Shares
        ORDER TYPE:  LIMIT
        LIMIT PRICE: {limit_price}
        ---------------------------
        STOP LOSS:   ${stop:.2f} (Protect yourself)
        TAKE PROFIT: ${target:.2f} (Pay yourself)
        """, language="yaml")
        st.caption(f"Copy these exact numbers into your broker. You are risking approx ${risk_per_trade:.2f} to make ${(shares * (target-entry) if 'Long' in strategy else shares * (entry-target)):.2f}.")

else:
    st.info("👈 Please enter a Ticker on the left and click '2. SCAN TICKER'")
