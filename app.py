# app.py
import streamlit as st
import yfinance as yf
import pandas_ta as ta
import mplfinance as mpf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from datetime import datetime

# --- 1. CONFIGURATION & VIP WATCHLIST ---
VIP_TICKERS = ["NVDA", "AAPL", "AMZN", "GOOGL", "TSLA", "MSFT", "META", "AMD", "NFLX", "SPY", "QQQ", "IWM"]

st.set_page_config(
    page_title="Quantum Maestro Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏛️"
)

# --- 2. CSS STYLING ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 4px; height: 3em; font-weight: 600; letter-spacing: 0.5px; }
    div[data-testid="stMetric"] { background-color: #f0f2f6; border: 1px solid #d6d6d6; border-radius: 6px; padding: 10px 15px; }
    @media (prefers-color-scheme: dark) {
        div[data-testid="stMetric"] { background-color: #1e2127; border: 1px solid #30333d; }
    }
</style>
""", unsafe_allow_html=True)

# --- 3. TITLE & ONBOARDING ---
st.title("🏛️ Quantum TradeMaster Maestro: Mentor Edition")
st.markdown("""
**What is this?** An Algorithmic Assistant that automates the math for the **7-Step Financial Trading System**. 
It calculates Risk/Reward, identifies Demand Zones, and audits your trade setups.
""")

# --- 4. LEGAL DISCLAIMER & GATE ---
with st.expander("⚠️ READ FIRST: Legal Disclaimer & Risk Warning", expanded=True):
    st.markdown("""
    **1. No Affiliation:** This application is an independent educational tool. It is **not** affiliated with, endorsed by, or sponsored by Teri Ijeoma, Trade and Travel, or any associated entities.
    
    **2. Educational Use Only:** This tool provides technical analysis based on mathematical formulas. It is **not** financial advice. You are solely responsible for your own trading decisions.
    
    **3. Risk Warning:** Trading stocks and options involves significant risk and can result in the loss of your invested capital. 
    """)
    agree = st.checkbox("I understand this is not financial advice and I am using this tool for educational purposes.")

if not agree:
    st.warning("🛑 Please accept the disclaimer above to access the terminal.")
    st.stop() 

st.divider()

# --- 5. SESSION STATE (With Journal) ---
if 'data' not in st.session_state: st.session_state.data = None
if 'metrics' not in st.session_state: st.session_state.metrics = {}
if 'macro' not in st.session_state: st.session_state.macro = None
if 'journal' not in st.session_state: st.session_state.journal = []

# --- 6. SIDEBAR ---
with st.sidebar:
    st.header("1. VIP Selection")
    input_mode = st.radio("Input Mode", ["VIP Watchlist", "Manual Search"])
    
    if input_mode == "VIP Watchlist":
        ticker = st.selectbox("Select Ticker", VIP_TICKERS, help="Choose from Teri's High-Volume Favorites.")
    else:
        ticker = st.text_input("Enter Ticker", value="NVDA", help="Enter the stock symbol (e.g., TSLA, AMD).").upper()
        
    c1, c2 = st.columns(2)
    with c1:
        capital = st.number_input("Total Capital ($)", value=10000, help="Your total account balance/buying power.")
    with c2:
        risk_per_trade = st.number_input("Risk $", value=100, help="The maximum amount you are willing to lose per trade.")

    daily_goal = capital * 0.01
    st.caption(f"🎯 Daily Goal (1%): **${daily_goal:.2f}**")

    st.divider()
    st.header("2. Strategy Board")
    strategy = st.selectbox(
        "Execution Mode", 
        ['Long (Buy Stock)', 'Short (Sell Stock)', 'Sell Puts (Income)', 'Sell Calls (Income)'],
        help="Long: Profit if price goes UP. Short: Profit if price goes DOWN. Income: Get paid to wait (Options)."
    )
    
    # NEW: Manual Entry Option
    entry_mode = st.radio(
        "Entry Method", 
        ["Auto-Limit (Wait for Zone)", "Current Price (Market)", "Manual Entry (Override)"],
        help="Manual Entry allows you to type a specific price to test scenarios."
    )
    
    manual_entry_price = 0.0
    if entry_mode == "Manual Entry (Override)":
        manual_entry_price = st.number_input("Enter Entry Price ($)", value=0.0, step=0.01)
    
    stop_mode = st.selectbox(
        "Stop Type", 
        [1.0, 0.2], 
        format_func=lambda x: "Safe Swing (1.0 ATR)" if x == 1.0 else "IWT Tight (0.2 ATR)",
        help="Swing (1.0): Gives the stock room to breathe. Tight (0.2): Very close stop, high risk of stopping out early."
    )

    premium = 0.0
    if "Income" in strategy:
        st.success("💰 Income Mode")
        premium = st.number_input("Option Premium ($)", value=0.0, step=0.05, help="The price per share the market will pay you for the contract.")

    st.divider()
    st.header("3. IWT Scorecard")
    st.caption("Hover over the '?' for definitions.")
    c3, c4 = st.columns(2)
    with c3:
        fresh = st.selectbox(
            "Freshness", [2, 1, 0], 
            format_func=lambda x: {2:'2-Fresh', 1:'1-Used', 0:'0-Stale'}[x],
            help="Fresh: Price has NOT touched this level recently. Stale: Price has bounced here many times."
        )
        speed = st.selectbox(
            "Speed Out", [2, 1, 0], 
            format_func=lambda x: {2:'2-Fast', 1:'1-Avg', 0:'0-Slow'}[x],
            help="How fast did price leave this zone last time? Fast/Gap = Strong Buying."
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
        
    st.divider()
    st.caption("ℹ️ **Legal:** This tool is an independent project and is not affiliated with Trade and Travel.")

# --- 7. LOGIC ENGINE ---
class Analyst:
    def fetch_data(self, t):
        try:
            ticker_obj = yf.Ticker(t)
            data = ticker_obj.history(period="1y")
            
            if data.empty: return None, 0.0, None
            
            # Get Full Name (Try/Except to prevent crashes)
            try:
                full_name = ticker_obj.info.get('longName', t)
            except:
                full_name = t
            
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
            
            prev_close = data['Close'].iloc[-2]
            curr_open = data['Open'].iloc[-1]
            gap_pct = ((curr_open - prev_close) / prev_close) * 100
            
            return data, gap_pct, full_name
        except Exception as e:
            return None, 0.0, None

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

# --- 8. CONTROL PANEL ---
c_macro, c_scan = st.columns([1, 1])
with c_macro:
    if st.button("🌍 1. CHECK MACRO", type="secondary"):
        with st.spinner("Scanning Global Sensors..."):
            st.session_state.macro = engine.get_macro()
with c_scan:
    if st.button(f"🔎 2. SCAN {ticker}, AFTER you CONFIRM your selections for SECTIONS 2 and 3", type="primary"):
        with st.spinner(f"Parsing {ticker}..."):
            df, gap, fname = engine.fetch_data(ticker)
            
            # ERROR HANDLING: Custom Page for Missing Ticker
            if df is None:
                st.error(f"🚫 **Ticker '{ticker}' Not Found.** Please check spelling (e.g., 'GOOG' vs 'GOOGL').")
                st.session_state.data = None
            else:
                st.session_state.data = df
                price = df['Close'].iloc[-1]
                supp = df['Min'][df['Min'] < price * 0.99].iloc[-1] if not df['Min'][df['Min'] < price * 0.99].empty else price * 0.9
                res = df['Max'][df['Max'] > price * 1.01].iloc[-1] if not df['Max'][df['Max'] > price * 1.01].empty else price * 1.1
                st.session_state.metrics = {
                    "price": price, "atr": df['ATRr_14'].iloc[-1],
                    "phase": "🚀 UPTREND" if price > df['SMA_20'].iloc[-1] else "📉 DOWNTREND",
                    "supp": supp, "res": res, "rvol": df['RVOL'].iloc[-1], "gap": gap, "name": fname
                }

# --- 9. DASHBOARD ---
if st.session_state.macro:
    m = st.session_state.macro
    with st.expander("🌍 Global Context", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Futures", f"{m['sp_change']:.2f}%", help="Pre-market direction of S&P 500.")
        c2.metric("VIX", f"{m['vix']:.2f}", help="Fear Gauge. >20 = High Volatility.")
        c3.metric("Gold", f"{m['gold_change']:.2f}%")
        if m['sp_change'] < -0.5 and m['vix'] > 20: c4.error("🐻 BEAR")
        elif m['sp_change'] > 0.5 and m['vix'] < 20: c4.success("🐂 BULL")
        else: c4.warning("🦀 CHOP")

if st.session_state.data is not None:
    df = st.session_state.data
    m = st.session_state.metrics
    
    st.divider()
    
    # NEW: Full Name Display
    st.subheader(f"🏢 {m['name']} ({ticker})")
    st.markdown(f"📰 **Intelligence:** [Read News on Yahoo Finance](https://finance.yahoo.com/quote/{ticker}/news)")
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"${m['price']:.2f}")
    
    # NEW: Gap Coloring (Explicit)
    gap_color = "normal" # Normal means Green=Up, Red=Down
    c2.metric("Gap %", f"{m['gap']:.2f}%", delta=f"{m['gap']:.2f}%", delta_color=gap_color, help="Pre-market move. >2% is ideal for Gap & Go.")
    
    c3.metric("Volume", f"{m['rvol']:.1f}x", help="Relative Volume. >1.0 means High Interest.")
    
    # Strategy Calculation
    # LOGIC FOR MANUAL ENTRY
    if entry_mode == "Manual Entry (Override)":
        entry = manual_entry_price
        target = m['res'] if "Long" in strategy else m['supp']
        stop = entry - (m['atr'] * stop_mode) if "Long" in strategy else entry + (m['atr'] * stop_mode)
    elif "Short" in strategy:
        entry = m['price'] if entry_mode == "Current Price (Market)" else m['res']
        target = m['supp']
        stop = entry + (m['atr'] * stop_mode)
    elif "Income" in strategy:
        entry = m['supp'] if "Puts" in strategy else m['res']
        stop = entry; target = entry # Income targets are just the strike usually
    else: # Long Standard
        entry = m['price'] if entry_mode == "Current Price (Market)" else m['supp']
        target = m['res']
        stop = entry - (m['atr'] * stop_mode)

    # Risk/Reward Math
    if "Income" in strategy:
        risk = entry * 0.1
        reward = premium
    elif "Short" in strategy:
        risk = stop - entry
        reward = entry - target
    else: # Long
        risk = entry - stop
        reward = target - entry
    
    rr = reward / risk if risk > 0 else 0
    delta_msg = "Excellent (>3.0)" if rr >= 3 else "Good (>2.0)" if rr >= 2 else "Weak (<2.0)"
    delta_col = "normal" if rr >= 2 else "inverse"
    c4.metric("R/R Ratio", f"{rr:.2f}", delta=delta_msg, delta_color=delta_col, help="Must be > 3.0 for Green Light.")

    # Chart
    chart_slice = df.iloc[-60:]
    fig, ax = mpf.plot(
        chart_slice, type='candle', style='yahoo', volume=True,
        addplot=[mpf.make_addplot(chart_slice['SMA_20'], color='blue'), mpf.make_addplot(chart_slice['ST_VAL'], color='orange')],
        hlines=dict(hlines=[entry, target if "Income" not in strategy else entry], colors=['green','red'], linestyle='-.'),
        returnfig=True, figsize=(12, 5), fontscale=0.8
    )
    st.pyplot(fig)

    # --- VERDICT ---
    st.markdown("### 📝 The Quantum Verdict")
    score_rr = 2 if rr >= 3 or ("Income" in strategy and rr > 0.1) else 1 if rr >= 2 else 0
    total_score = fresh + time_zone + speed + score_rr
    
    col_verdict, col_audit = st.columns([1, 1])
    
    with col_verdict:
        if total_score >= 7: 
            st.success(f"## 🟢 GREEN LIGHT\n**Score: {total_score}/8**")
            st.caption("Action: Execute Trade.")
        elif total_score >= 5: 
            st.warning(f"## 🟡 YELLOW LIGHT\n**Score: {total_score}/8**")
            st.caption("Action: **Reduce Size** or **Wait for Confirmation**.")
        else: 
            st.error(f"## 🔴 RED LIGHT\n**Score: {total_score}/8**")
            st.caption("Action: Do Not Trade.")

    with col_audit:
        st.markdown("**📋 Setup Audit:**")
        if fresh == 2: st.markdown("✅ **Freshness:** Perfect (2/2)")
        elif fresh == 1: st.markdown("⚠️ **Freshness:** Used Level (1/2)")
        else: st.markdown("❌ **Freshness:** Stale / Dirty (0/2)")
        
        if score_rr == 2: st.markdown(f"✅ **R/R:** Excellent ({rr:.2f})")
        else: st.markdown(f"❌ **R/R:** Poor ({rr:.2f})")
        if abs(m['gap']) > 2.0: st.markdown(f"🚀 **Gap:** Large ({m['gap']:.2f}%)")

    # --- BROKER SLIP & JOURNAL ---
    st.markdown("---")
    st.subheader("🧾 Trade Execution")
    
    trade_record = {}
    
    if "Income" in strategy:
        contracts = int((capital / entry) // 100) if entry > 0 else 0
        potential_profit = contracts * 100 * premium
        
        goal_status = f"✅ GOAL MET (>${daily_goal:.2f})" if potential_profit >= daily_goal else f"❌ MISS (${potential_profit - daily_goal:.2f})"
        st.success(f"**DAILY GOAL:** {goal_status}")
        
        st.code(f"SELL TO OPEN: {ticker} {entry:.2f} Strike | Expiration: 30 Days Out")
        # FIXED: Score format to prevent Excel date bug
        trade_record = {"Ticker": ticker, "Type": "INCOME", "Entry": entry, "Profit": potential_profit, "Verdict": f"Score {total_score}", "Stop": entry, "Target": entry}
        
    else:
        shares = int(risk_per_trade / risk) if risk > 0 else 0
        potential_profit = shares * (target-entry) if 'Long' in strategy else shares * (entry-target)
        order_type = "BUY" if "Long" in strategy else "SELL SHORT"
        
        goal_status = f"✅ GOAL MET (Target > ${daily_goal:.0f})" if potential_profit >= daily_goal else f"❌ MISS (< ${daily_goal:.0f})"
        
        col_slip_1, col_slip_2 = st.columns([2, 1])
        with col_slip_1:
            st.code(f"""
            ACTION:      {order_type}
            SHARES:      {shares}
            PRICE:       ${entry:.2f} ({entry_mode})
            ---------------------------
            STOP LOSS:   ${stop:.2f}
            TAKE PROFIT: ${target:.2f}
            """, language="yaml")
        with col_slip_2:
            st.success(f"**GOAL:** {goal_status}")
            st.write(f"**Potential Reward:** ${potential_profit:.2f}")
            st.write(f"**Max Risk:** ${risk_per_trade:.2f}")
            
        # FIXED: Added Stop/Target and changed Verdict format
        trade_record = {
            "Ticker": ticker, "Type": order_type, "Entry": entry, 
            "Profit": potential_profit, "Verdict": f"Score {total_score}", 
            "Stop": stop, "Target": target
        }

    # --- SAVE TO JOURNAL FEATURE ---
    if st.button("💾 Log Trade to Journal"):
        trade_record["Date"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        st.session_state.journal.append(trade_record)
        st.success("Trade Logged!")

    if st.session_state.journal:
        st.divider()
        st.subheader("📓 Session Journal")
        journal_df = pd.DataFrame(st.session_state.journal)
        st.dataframe(journal_df)
        
        csv = journal_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Journal (CSV)", data=csv, file_name="trade_journal.csv", mime="text/csv")

else:
    st.info("👈 Please enter a Ticker on the left AND select your choices for SECTIONS 2 and 3. THEN click '2. SCAN TICKER'")
