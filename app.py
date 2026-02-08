# Copyright (c) 2026 Gabriel Mahia. All Rights Reserved.
# Unauthorized copying of this file, via any medium, is strictly prohibited.
# Proprietary and confidential.
# Written by Gabriel Mahia, 2026
# app.py
# 🏛️ QUANTUM MAESTRO: THE INSTITUTIONAL EDITION
# Integrity: V12 Risk Engine + V13 Methodology Notes + V11 Core Logic

import streamlit as st
import yfinance as yf
import pandas_ta as ta
import mplfinance as mpf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from datetime import datetime, time
import pytz

# --- 1. CONFIGURATION & ASSET UNIVERSES ---
VIP_TICKERS = ["NVDA", "AAPL", "AMZN", "GOOGL", "TSLA", "MSFT", "META", "AMD", "NFLX", "SPY", "QQQ", "IWM"]
GROWTH_TICKERS = ["NVDA", "AAPL", "AMZN", "GOOGL", "TSLA", "MSFT", "META", "AMD", "NFLX", "QQQ", "ARKK", "COIN"]

# Sector Map for Concentration Risk
SECTOR_MAP = {
    "NVDA": "Tech", "AMD": "Tech", "MSFT": "Tech", "AAPL": "Tech", "META": "Tech", "GOOGL": "Tech",
    "TSLA": "Auto", "AMZN": "Consumer", "NFLX": "Media", "SPY": "Index", "QQQ": "Tech-Index", "IWM": "Index"
}

st.set_page_config(
    page_title="Quantum Maestro Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏛️"
)

# Custom CSS for Institutional Styling
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 4px; height: 3em; font-weight: 600; letter-spacing: 0.5px; }
    div[data-testid="stMetric"] { background-color: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; padding: 10px 15px; }
    @media (prefers-color-scheme: dark) {
        div[data-testid="stMetric"] { background-color: #1e2127; border: 1px solid #30333d; }
    }
    .risk-alert { background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 5px; border-left: 5px solid #ffc107; font-size: 0.9em; margin-bottom: 10px; }
    .risk-critical { background-color: #f8d7da; color: #721c24; padding: 10px; border-radius: 5px; border-left: 5px solid #dc3545; font-size: 0.9em; margin-bottom: 10px; }
    .success-box { background-color: #d4edda; color: #155724; padding: 10px; border-radius: 5px; border-left: 5px solid #28a745; font-size: 0.9em; margin-bottom: 10px; }
    .methodology-text { font-size: 0.85em; color: #6c757d; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# --- 2. LEGAL & ONBOARDING ---
st.title("🏛️ Quantum Maestro [TradingBot]: Institutional Edition")
st.caption("Portfolio Risk Architecture | Volatility Regimes | Passive Flows | IWT Execution")

with st.expander("⚠️ READ FIRST: Legal Disclaimer & Risk Warning", expanded=True):
    st.markdown("""
    **1. Educational Use Only:** This is a simulation tool for risk analysis training. It is **not** financial advice.
    **2. No Affiliation:** Independent tool. Not affiliated with Trade and Travel or any specific institution.
    **3. Risk Warning:** Trading involves substantial risk of loss. Past performance is not indicative of future results.
    **4. Data:** Market data provided by Yahoo Finance (delayed).
    """)
    agree = st.checkbox("I understand this is not financial advice and I am using this tool for educational purposes.")

if not agree:
    st.warning("🛑 Please accept the disclaimer above.")
    st.stop()

st.divider()

# --- 3. SESSION STATE MANAGEMENT ---
if 'data' not in st.session_state: st.session_state.data = None
if 'metrics' not in st.session_state: st.session_state.metrics = {}
if 'macro' not in st.session_state: st.session_state.macro = None
if 'journal' not in st.session_state: st.session_state.journal = []
if 'open_positions' not in st.session_state: st.session_state.open_positions = []
if 'closed_trades' not in st.session_state: st.session_state.closed_trades = []
if 'goal_met' not in st.session_state: st.session_state.goal_met = False
if 'daily_pnl' not in st.session_state: st.session_state.daily_pnl = 0.0
if 'total_risk_deployed' not in st.session_state: st.session_state.total_risk_deployed = 0.0

# --- 4. INSTITUTIONAL ANALYST ENGINE ---
class InstitutionalAnalyst:
    
    def __init__(self):
        # V12/V13 Volatility Regimes
        self.vix_regimes = {
            "COMPLACENT": (0, 12),
            "NORMAL": (12, 20),
            "ELEVATED": (20, 30),
            "HIGH": (30, 40),
            "CRISIS": (40, 100)
        }

    def get_market_phase(self):
        """Determine Market Hours (EST)"""
        try:
            et = pytz.timezone('US/Eastern')
            now = datetime.now(et)
            curr = now.time()
            
            if curr < time(9, 30): return "PRE_MARKET", "⏰ Pre-Market (Low Liquidity)"
            elif curr < time(10, 0): return "OPENING", "🔔 Opening Range (Wait for Direction)"
            elif curr < time(12, 0): return "MORNING", "☀️ Morning Session (Prime Execution)"
            elif curr < time(14, 0): return "LUNCH", "🍴 Lunch Chop (Avoid New Entries)"
            elif curr < time(15, 0): return "AFTERNOON", "🌤️ Afternoon Trend"
            elif curr < time(16, 0): return "POWER_HOUR", "⚡ Power Hour (Inst. Flow)"
            else: return "AFTER_HOURS", "🌙 After Hours"
        except: return "UNKNOWN", "Unknown Time"

    def get_regime_guidance(self, regime):
        """V13 Risk Multipliers based on VIX"""
        guidance = {
            "COMPLACENT": {"size": 0.8, "stop": 1.2, "note": "Low VIX = Sudden Reversal Risk. Size Down slightly."},
            "NORMAL":     {"size": 1.0, "stop": 1.0, "note": "Ideal Conditions. Standard Sizing."},
            "ELEVATED":   {"size": 0.7, "stop": 1.3, "note": "Caution. Reduce Size 30%. Widen Stops."},
            "HIGH":       {"size": 0.5, "stop": 1.5, "note": "High Risk. Half Size. Defensive Posture."},
            "CRISIS":     {"size": 0.3, "stop": 2.0, "note": "Cash is a Position. Only A+ Setups."}
        }
        return guidance.get(regime, guidance["NORMAL"])

    def fetch_data(self, t):
        try:
            # 1. Fetch History
            ticker_obj = yf.Ticker(t)
            data = ticker_obj.history(period="1y")
            if data.empty or len(data) < 50: return None, 0.0, None, "Insufficient Data"
            
            try: full_name = ticker_obj.info.get('longName', t)
            except: full_name = t
            
            # 2. Institutional Indicators
            data.ta.atr(length=14, append=True)
            data.ta.rsi(length=14, append=True)
            data.ta.sma(length=20, append=True)
            data.ta.sma(length=50, append=True)
            data.ta.sma(length=200, append=True)
            
            try:
                st_data = data.ta.supertrend(length=10, multiplier=3)
                data['ST_VAL'] = st_data[st_data.columns[0]]
                data['ST_DIR'] = st_data[st_data.columns[1]]
            except: 
                data['ST_VAL'] = data['Close']; data['ST_DIR'] = 1

            # 3. Volume & Structure
            vol_sma = data['Volume'].rolling(20).mean()
            data['RVOL'] = data['Volume'] / vol_sma
            
            # 4. Support/Resistance (Local Extrema)
            data['Min'] = data['Low'].rolling(window=10).min()
            data['Max'] = data['High'].rolling(window=10).max()
            
            # 5. Gap Logic
            prev_close = data['Close'].iloc[-2]
            curr_open = data['Open'].iloc[-1]
            gap_pct = ((curr_open - prev_close) / prev_close) * 100
            
            # 6. Trend Audit
            price = data['Close'].iloc[-1]
            trend_score = 0
            if price > data['SMA_20'].iloc[-1]: trend_score += 1
            if price > data['SMA_50'].iloc[-1]: trend_score += 1
            if price > data['SMA_200'].iloc[-1]: trend_score += 1
            
            trend_str = "STRONG_BULL" if trend_score == 3 else "NEUTRAL" if trend_score == 2 else "BEAR"
            
            return data, gap_pct, full_name, {"trend": trend_str, "rsi": data['RSI_14'].iloc[-1]}
            
        except Exception as e: return None, 0.0, None, str(e)

    def get_macro(self):
        try:
            tickers = ["ES=F", "^VIX", "GC=F", "^GDAXI", "^N225", "^TNX", "DX-Y.NYB"]
            df = yf.download(tickers, period="5d", progress=False, timeout=10)['Close']
            
            if df.empty: return None
            
            # Extract Data (Robust Handling)
            try:
                sp = df["ES=F"]; vix = df["^VIX"]; gold = df["GC=F"]
                dax = df["^GDAXI"]; nikkei = df["^N225"]; tnx = df["^TNX"]
                dxy = df["DX-Y.NYB"] if "DX-Y.NYB" in df.columns else pd.Series([100]*len(df))
            except: return None
            
            # Calculations
            sp_chg = ((sp.iloc[-1]-sp.iloc[-2])/sp.iloc[-2])*100
            dax_chg = ((dax.iloc[-1]-dax.iloc[-2])/dax.iloc[-2])*100
            nikkei_chg = ((nikkei.iloc[-1]-nikkei.iloc[-2])/nikkei.iloc[-2])*100
            tnx_chg = ((tnx.iloc[-1]-tnx.iloc[-2])/tnx.iloc[-2])*100
            
            # V13 Passive Flow Logic (1st-5th OR 15th-20th)
            day = datetime.now().day
            passive_on = (1 <= day <= 5) or (15 <= day <= 20)
            
            return {
                "sp": sp_chg, "vix": vix.iloc[-1], "gold": gold.iloc[-1],
                "dax": dax_chg, "nikkei": nikkei_chg, "tnx": tnx.iloc[-1],
                "tnx_chg": tnx_chg, "dxy": dxy.iloc[-1], "passive": passive_on
            }
        except: return None

    def calculate_sharpe(self, trades):
        """Risk-Adjusted Return Metric"""
        if len(trades) < 5: return None
        returns = [t['pnl'] for t in trades]
        if np.std(returns) == 0: return 0
        return (np.mean(returns) / np.std(returns)) * np.sqrt(252)

engine = InstitutionalAnalyst()

# --- 5. SIDEBAR: CONTROL & EDUCATION ---
with st.sidebar:
    st.header("1. Portfolio Specs")
    capital = st.number_input("Capital ($)", 10000, help="Total Account Value")
    risk_per_trade = st.number_input("Risk Unit ($)", 100, help="Max loss per trade")
    max_portfolio_risk = st.number_input("Max Port Risk (%)", 5.0, step=0.5)
    
    daily_goal = capital * 0.01
    
    # Portfolio Status
    current_risk = st.session_state.total_risk_deployed
    risk_pct = (current_risk / capital) * 100
    pnl_pct = (st.session_state.daily_pnl / capital) * 100
    
    st.divider()
    if st.session_state.goal_met:
        st.success(f"✅ GOAL MET: +${st.session_state.daily_pnl:.2f}")
    else:
        st.info(f"📊 P&L: ${st.session_state.daily_pnl:.2f} | Risk: {risk_pct:.1f}%")

    st.header("2. Asset Selection")
    input_mode = st.radio("Input:", ["VIP List", "Manual"])
    if input_mode == "VIP List":
        ticker = st.selectbox("Ticker", VIP_TICKERS)
    else:
        ticker = st.text_input("Ticker", "NVDA").upper()

    st.header("3. Execution")
    strategy = st.selectbox("Mode", ['Long (Buy)', 'Short (Sell)', 'Income (Puts)'])
    entry_mode = st.radio("Entry", ["Auto-Limit", "Market", "Manual"])
    manual_price = 0.0
    if entry_mode == "Manual": manual_price = st.number_input("Price ($)", 0.0)
    
    stop_mode = st.selectbox("Stop", [1.0, 0.5, 0.2], format_func=lambda x: f"{x} ATR")
    
    # --- V13 HELP NOTES INTEGRATION ---
    with st.expander("📘 Institutional Playbook (V13 Rules)"):
        st.markdown("""
        **1. Passive Flows:**
        * **Dates:** 1st-5th & 15th-20th.
        * **Logic:** 401k/Pension inflows auto-bid the market. Bias is LONG.
        
        **2. The Warsh Effect:**
        * **Indicator:** 10Y Yields (`^TNX`).
        * **Rule:** If Yields rise >1%, Tech/Growth stocks reprice lower.
        
        **3. Global Correlation:**
        * **Sync:** If US, DAX (EU), and Nikkei (JP) move together = Stable.
        * **Break:** If US is Green but World is Red = Trap/Volatility.
        
        **4. Gap Rules:**
        * **Novice Gap:** >2% on Low Volume (Fade it).
        * **Pro Gap:** >2% on High Volume (Follow it).
        """)
        
    st.divider()
    if st.button("🔄 Reset Session"):
        st.session_state.daily_pnl = 0.0
        st.session_state.goal_met = False
        st.session_state.open_positions = []
        st.session_state.total_risk_deployed = 0.0
        st.rerun()

# --- 6. MAIN DASHBOARD ---
col_m, col_s = st.columns([1,1])
with col_m:
    if st.button("🌍 Macro Audit", use_container_width=True):
        st.session_state.macro = engine.get_macro()
with col_s:
    if st.button(f"🔎 Scan {ticker}", type="primary", use_container_width=True):
        df, gap, name, extra = engine.fetch_data(ticker)
        if df is None:
            st.error(f"Error: {extra}")
        else:
            st.session_state.data = df
            st.session_state.metrics = {
                "price": df['Close'].iloc[-1],
                "atr": df['ATRr_14'].iloc[-1],
                "supp": df['Min'].iloc[-1],
                "res": df['Max'].iloc[-1],
                "rvol": df['RVOL'].iloc[-1],
                "gap": gap,
                "name": name,
                "trend": extra['trend'],
                "rsi": extra['rsi']
            }

# --- 7. ANALYSIS & VERDICT ---
if st.session_state.macro:
    m = st.session_state.macro
    
    # VIX Regime
    vix_label = "NORMAL"
    for k, v in engine.vix_regimes.items():
        if v[0] <= m['vix'] < v[1]: vix_label = k
    
    regime = engine.get_regime_guidance(vix_label)
    
    # Market Phase
    phase, phase_desc = engine.get_market_phase()
    
    # Alerts
    if vix_label in ["HIGH", "CRISIS"]:
        st.markdown(f"<div class='risk-critical'>🚨 <strong>CRITICAL VOLATILITY ({vix_label}):</strong> {regime['note']}</div>", unsafe_allow_html=True)
    elif vix_label == "ELEVATED":
        st.markdown(f"<div class='risk-alert'>⚠️ <strong>ELEVATED VOLATILITY:</strong> {regime['note']}</div>", unsafe_allow_html=True)
    
    if phase == "LUNCH":
        st.markdown(f"<div class='risk-alert'>🍴 <strong>LUNCH CHOP:</strong> Volume is thin. Avoid new entries.</div>", unsafe_allow_html=True)

    # Passive Flow
    if m['passive']:
        st.markdown(f"<div class='success-box'>🌊 <strong>PASSIVE FLOW WINDOW OPEN:</strong> Institutional inflows (1st-5th/15th-20th) support dip-buying.</div>", unsafe_allow_html=True)

    # Macro Grid
    with st.expander("🌍 Macro Control Panel", expanded=True):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("SPY Fut", f"{m['sp']:.2f}%")
        c2.metric("DAX (EU)", f"{m['dax']:.2f}%")
        c3.metric("Nikkei (JP)", f"{m['nikkei']:.2f}%")
        
        tnx_col = "inverse" if m['tnx_chg'] > 1.0 else "normal"
        c4.metric("10Y Yield", f"{m['tnx']:.2f}%", delta=f"{m['tnx_chg']:.2f}%", delta_color=tnx_col)
        c5.metric("DXY ($)", f"{m['dxy']:.2f}")

if st.session_state.data is not None:
    m = st.session_state.metrics
    st.divider()
    st.header(f"📈 {m['name']} ({ticker})")
    
    # Metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"${m['price']:.2f}")
    
    # Gap Analysis (V13 Logic)
    gap_msg = "Normal"
    if abs(m['gap']) > 2.0:
        if m['rvol'] > 1.5: gap_msg = "🚀 PRO GAP"
        else: gap_msg = "⚠️ NOVICE GAP"
    c2.metric("Gap", f"{m['gap']:.2f}%", delta=gap_msg)
    
    c3.metric("RVOL", f"{m['rvol']:.1f}x", delta="Hot" if m['rvol']>1.2 else "Normal")
    c4.metric("Trend", m['trend'])
    
    # Chart
    df_slice = st.session_state.data.iloc[-60:]
    fig, ax = mpf.plot(df_slice, type='candle', style='yahoo', volume=True, 
                       hlines=dict(hlines=[m['supp'], m['res']], colors=['g','r']), returnfig=True)
    st.pyplot(fig)
    
    # --- LOGIC & SCORING ---
    # 1. Levels
    if entry_mode == "Manual": entry = manual_price
    elif "Short" in strategy: entry = m['res'] if "Auto" in entry_mode else m['price']
    else: entry = m['supp'] if "Auto" in entry_mode else m['price']
    
    # 2. Risk Adjustments (VIX)
    if st.session_state.macro:
        adj = regime['stop']
    else: adj = 1.0
    
    # 3. Stop/Target
    if "Short" in strategy:
        stop = entry + (m['atr'] * stop_mode * adj)
        target = m['supp']
        risk = stop - entry
        reward = entry - target
    elif "Income" in strategy:
        stop = entry
        target = entry
        risk = entry * 0.1
        reward = 0 # Premium handles this
    else: # Long
        stop = entry - (m['atr'] * stop_mode * adj)
        target = m['res']
        risk = entry - stop
        reward = target - entry
    
    rr = reward / risk if risk > 0 else 0
    
    # 4. Scoring
    # IWT Base
    with st.container():
        st.subheader("📝 IWT Scorecard")
        c_iwt1, c_iwt2, c_iwt3 = st.columns(3)
        fresh = c_iwt1.selectbox("Freshness", [2, 1, 0], format_func=lambda x: ["0-Stale","1-Used","2-Fresh"][x])
        speed = c_iwt2.selectbox("Speed", [2, 1, 0], format_func=lambda x: ["0-Slow","1-Avg","2-Fast"][x])
        time_z = c_iwt3.selectbox("Time", [2, 1, 0], format_func=lambda x: ["0-Long","1-Med","2-Short"][x])
        
    score_rr = 2 if rr >= 3 else 1 if rr >= 2 else 0
    total = fresh + speed + time_z + score_rr
    
    # 5. Penalties (The Architect)
    penalties = []
    
    # Warsh Penalty
    if st.session_state.macro and st.session_state.macro['tnx_chg'] > 1.0 and ticker in GROWTH_TICKERS and "Long" in strategy:
        total -= 2
        penalties.append("Warsh Effect (-2): Rates Spiking >1%")
    
    # Market Hours
    if phase in ["LUNCH", "PRE_MARKET", "AFTER_HOURS"]:
        total -= 1
        penalties.append(f"Market Phase (-1): {phase}")
        
    # Concentration
    sector = SECTOR_MAP.get(ticker, "Unknown")
    cnt = sum(1 for p in st.session_state.open_positions if SECTOR_MAP.get(p['ticker']) == sector)
    if cnt >= 2:
        total -= 1
        penalties.append(f"Concentration (-1): {cnt} {sector} positions open")

    # --- VERDICT ---
    st.divider()
    c_verdict, c_slip = st.columns([1,1])
    
    with c_verdict:
        st.subheader("🚦 Quantum Verdict")
        
        if st.session_state.goal_met:
            st.error("🛑 DAILY GOAL MET")
            st.caption("You have hit your 1% target. Close screens.")
        elif risk_pct + ((risk_per_trade/capital)*100) > max_portfolio_risk:
            st.error("🛑 RISK LIMIT EXCEEDED")
            st.caption(f"Portfolio Risk would exceed {max_portfolio_risk}%")
        else:
            if total >= 7:
                st.success(f"## 🟢 GREEN LIGHT ({total}/8)")
                st.caption("Action: Execute. A+ Setup.")
            elif total >= 5:
                st.warning(f"## 🟡 YELLOW LIGHT ({total}/8)")
                st.caption("Action: Half Size or Wait.")
            else:
                st.error(f"## 🔴 RED LIGHT ({total}/8)")
                st.caption("Action: Do Not Trade.")
                
        if penalties:
            st.markdown("**Penalties:**")
            for p in penalties: st.markdown(f"- {p}")
            
    with c_slip:
        st.subheader("🧾 Execution Slip")
        
        # Sizing
        if st.session_state.macro:
            size_mult = regime['size']
        else: size_mult = 1.0
        
        shares = int((risk_per_trade * size_mult) / risk) if risk > 0 else 0
        
        st.code(f"""
ACTION: {strategy}
SIZE:   {shares} (Adj: {size_mult}x)
ENTRY:  ${entry:.2f}
STOP:   ${stop:.2f}
TARGET: ${target:.2f}
RISK:   ${(shares*risk):.2f}
        """)
        
        if st.button("📝 Log Trade (Live)"):
            rec = {
                "Time": datetime.now().strftime("%H:%M"),
                "Ticker": ticker, "Action": strategy,
                "Entry": entry, "Stop": stop, "Target": target,
                "Shares": shares, "Risk": shares*risk,
                "Status": "OPEN", "Pnl": 0.0
            }
            st.session_state.open_positions.append(rec)
            st.session_state.journal.append(rec)
            st.session_state.total_risk_deployed += (shares*risk)
            st.success("Logged!")
            st.rerun()

# --- 8. PORTFOLIO & JOURNAL ---
if st.session_state.open_positions:
    st.divider()
    st.subheader("📊 Open Portfolio")
    df_open = pd.DataFrame(st.session_state.open_positions)
    st.dataframe(df_open)
    
    # Closing Logic
    c1, c2, c3 = st.columns(3)
    t_close = c1.selectbox("Close Ticker", df_open['Ticker'].unique())
    p_close = c2.number_input("Exit Price", 0.0)
    
    if c3.button("Close Position"):
        for i, p in enumerate(st.session_state.open_positions):
            if p['Ticker'] == t_close:
                # Calc PnL
                if "Long" in p['Action']: pnl = (p_close - p['Entry']) * p['Shares']
                else: pnl = (p['Entry'] - p_close) * p['Shares']
                
                # Update
                st.session_state.daily_pnl += pnl
                st.session_state.total_risk_deployed -= p['Risk']
                p['Status'] = "CLOSED"
                p['Pnl'] = pnl
                
                # Move to closed
                st.session_state.closed_trades.append(p)
                st.session_state.open_positions.pop(i)
                
                # Goal Check
                if st.session_state.daily_pnl >= daily_goal:
                    st.session_state.goal_met = True
                    st.balloons()
                
                st.rerun()
                break

if st.session_state.closed_trades:
    st.divider()
    st.subheader("📓 Trade History")
    df_closed = pd.DataFrame(st.session_state.closed_trades)
    st.dataframe(df_closed)
    
    sharpe = engine.calculate_sharpe(st.session_state.closed_trades)
    if sharpe: st.caption(f"📈 Session Sharpe Ratio: {sharpe:.2f}")

    csv = df_closed.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Journal", csv, "journal.csv", "text/csv")

st.divider()
st.caption("🏛️ Quantum  Maestro  Financial  Markets TradingBot | Multi-Algorithm Fusion | Educational Use Only")
