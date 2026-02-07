# app.py
import streamlit as st
import yfinance as yf
import pandas_ta as ta
import mplfinance as mpf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from datetime import datetime, time
import pytz

# --- 1. CONFIGURATION & LISTS ---
VIP_TICKERS = ["NVDA", "AAPL", "AMZN", "GOOGL", "TSLA", "MSFT", "META", "AMD", "NFLX", "SPY", "QQQ", "IWM"]
GROWTH_TICKERS = ["NVDA", "AAPL", "AMZN", "GOOGL", "TSLA", "MSFT", "META", "AMD", "NFLX", "QQQ", "ARKK", "COIN"]

# Sector classification for concentration risk
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

st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 4px; height: 3em; font-weight: 600; letter-spacing: 0.5px; }
    div[data-testid="stMetric"] { background-color: #f0f2f6; border: 1px solid #d6d6d6; border-radius: 6px; padding: 10px 15px; }
    @media (prefers-color-scheme: dark) {
        div[data-testid="stMetric"] { background-color: #1e2127; border: 1px solid #30333d; }
    }
    .risk-warning { background-color: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 10px 0; }
    .success-box { background-color: #d4edda; padding: 15px; border-left: 4px solid #28a745; margin: 10px 0; }
</style>
""", unsafe_allow_html=True)

# --- 2. LEGAL & ONBOARDING ---
st.title("🏛️ Quantum Maestro [TradingBot]: Institutional Edition ")
st.caption("Portfolio Risk Architecture | Volatility Regimes | Multi-Position Management | Performance Analytics")

with st.expander("⚠️ READ FIRST: Legal Disclaimer", expanded=True):
    st.markdown("""
    **1. No Affiliation:** Independent tool. Not affiliated with Trade and Travel.
    **2. Educational Use Only:** Not financial advice. For simulation and learning purposes.
    **3. Risk Warning:** Trading involves substantial risk of loss. Past performance does not guarantee future results.
    **4. Data Disclaimer:** Market data provided by Yahoo Finance. Delays and inaccuracies may occur.
    """)
    agree = st.checkbox("I understand this is not financial advice and I am using this tool for educational purposes.")

if not agree:
    st.warning("🛑 Please accept the disclaimer above.")
    st.stop()

st.divider()

# --- 3. SESSION STATE (Enhanced) ---
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
        self.vix_regimes = {
            "EXTREME_LOW": (0, 12),
            "LOW": (12, 15),
            "NORMAL": (15, 20),
            "ELEVATED": (20, 30),
            "HIGH": (30, 40),
            "EXTREME": (40, 100)
        }
    
    def get_market_hours_status(self):
        """Determine current market phase"""
        try:
            et = pytz.timezone('US/Eastern')
            now = datetime.now(et)
            current_time = now.time()
            
            market_open = time(9, 30)
            market_close = time(16, 0)
            opening_range_end = time(10, 0)
            lunch_start = time(12, 0)
            lunch_end = time(14, 0)
            power_hour_start = time(15, 0)
            
            if current_time < market_open:
                return "PRE_MARKET", "⏰ Pre-Market (Higher volatility, lower liquidity)"
            elif current_time < opening_range_end:
                return "OPENING", "🔔 Opening Range (Wait for direction, avoid chasing)"
            elif current_time < lunch_start:
                return "MORNING", "☀️ Morning Session (Prime trading window)"
            elif current_time < lunch_end:
                return "LUNCH", "🍴 Lunch Hour (Reduced volume, avoid new positions)"
            elif current_time < power_hour_start:
                return "AFTERNOON", "🌤️ Afternoon Session (Trend continuation)"
            elif current_time < market_close:
                return "POWER_HOUR", "⚡ Power Hour (Institutional positioning, high volume)"
            else:
                return "AFTER_HOURS", "🌙 After Hours (Extended hours risk)"
        except:
            return "UNKNOWN", "⚠️ Unable to determine market hours"
    
    def classify_vix_regime(self, vix_level):
        """Classify market volatility regime"""
        for regime, (low, high) in self.vix_regimes.items():
            if low <= vix_level < high:
                return regime
        return "EXTREME"
    
    def get_regime_guidance(self, regime):
        """Provide regime-specific trading rules"""
        guidance = {
            "EXTREME_LOW": {
                "desc": "Complacency Zone",
                "action": "⚠️ Reduce size. Market pricing in no risk. Potential for sudden reversals.",
                "size_multiplier": 0.7,
                "stop_multiplier": 1.2
            },
            "LOW": {
                "desc": "Calm Waters",
                "action": "✅ Normal conditions. Standard position sizing appropriate.",
                "size_multiplier": 1.0,
                "stop_multiplier": 1.0
            },
            "NORMAL": {
                "desc": "Healthy Volatility",
                "action": "✅ Ideal environment. Markets functioning normally.",
                "size_multiplier": 1.0,
                "stop_multiplier": 1.0
            },
            "ELEVATED": {
                "desc": "Heightened Uncertainty",
                "action": "⚠️ Reduce size by 30%. Widen stops. Expect intraday swings.",
                "size_multiplier": 0.7,
                "stop_multiplier": 1.3
            },
            "HIGH": {
                "desc": "Crisis Mode",
                "action": "🛑 Reduce size by 50%. Consider cash. Only highest-conviction setups.",
                "size_multiplier": 0.5,
                "stop_multiplier": 1.5
            },
            "EXTREME": {
                "desc": "Market Dislocation",
                "action": "🚨 EXTREME VOLATILITY. Close non-essential positions. Preserve capital.",
                "size_multiplier": 0.3,
                "stop_multiplier": 2.0
            }
        }
        return guidance.get(regime, guidance["NORMAL"])
    
    def fetch_data(self, t):
        """Enhanced data fetch with validation"""
        try:
            ticker_obj = yf.Ticker(t)
            data = ticker_obj.history(period="1y")
            if data.empty: 
                return None, 0.0, None, "ERROR: No data returned"
            
            # Validate data quality
            if len(data) < 50:
                return None, 0.0, None, f"ERROR: Insufficient data ({len(data)} days)"
            
            try: 
                full_name = ticker_obj.info.get('longName', t)
            except: 
                full_name = t
            
            # Technical indicators
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
                data['ST_VAL'] = data['Close']
                data['ST_DIR'] = 1
            
            # Volume analysis
            vol_sma = data['Volume'].rolling(20).mean()
            data['RVOL'] = data['Volume'] / vol_sma
            
            # Support/Resistance (more robust)
            data['Min'] = data['Low'].rolling(window=10).min()
            data['Max'] = data['High'].rolling(window=10).max()
            
            # Calculate swing levels using local extrema
            try:
                local_min_idx = argrelextrema(data['Close'].values, np.less_equal, order=5)[0]
                local_max_idx = argrelextrema(data['Close'].values, np.greater_equal, order=5)[0]
                
                if len(local_min_idx) > 0:
                    recent_mins = data.iloc[local_min_idx[-3:]]['Close'].values if len(local_min_idx) >= 3 else data.iloc[local_min_idx]['Close'].values
                    support_level = np.mean(recent_mins)
                else:
                    support_level = data['Low'].iloc[-20:].min()
                
                if len(local_max_idx) > 0:
                    recent_maxs = data.iloc[local_max_idx[-3:]]['Close'].values if len(local_max_idx) >= 3 else data.iloc[local_max_idx]['Close'].values
                    resistance_level = np.mean(recent_maxs)
                else:
                    resistance_level = data['High'].iloc[-20:].max()
                    
            except:
                support_level = data['Low'].iloc[-20:].min()
                resistance_level = data['High'].iloc[-20:].max()
            
            # Gap calculation
            if len(data) >= 2:
                prev_close = data['Close'].iloc[-2]
                curr_open = data['Open'].iloc[-1]
                gap_pct = ((curr_open - prev_close) / prev_close) * 100
            else:
                gap_pct = 0.0
            
            # Multi-timeframe trend alignment
            price = data['Close'].iloc[-1]
            sma20 = data['SMA_20'].iloc[-1]
            sma50 = data['SMA_50'].iloc[-1]
            sma200 = data['SMA_200'].iloc[-1]
            
            trend_score = 0
            if price > sma20: trend_score += 1
            if price > sma50: trend_score += 1
            if price > sma200: trend_score += 1
            if sma20 > sma50: trend_score += 1
            if sma50 > sma200: trend_score += 1
            
            trend_strength = "STRONG_BULL" if trend_score >= 4 else \
                           "BULL" if trend_score == 3 else \
                           "NEUTRAL" if trend_score == 2 else \
                           "BEAR" if trend_score == 1 else "STRONG_BEAR"
            
            return data, gap_pct, full_name, {
                "support": support_level,
                "resistance": resistance_level,
                "trend_strength": trend_strength,
                "rsi": data['RSI_14'].iloc[-1] if 'RSI_14' in data.columns else 50
            }
            
        except Exception as e:
            return None, 0.0, None, f"ERROR: {str(e)}"

    def get_macro(self):
        """Enhanced macro data with error handling"""
        try:
            tickers = ["ES=F", "^VIX", "GC=F", "^GDAXI", "^N225", "^TNX", "DX-Y.NYB"]
            df = yf.download(tickers, period="5d", progress=False, timeout=10)['Close']
            
            if df.empty: 
                return None
            
            # Handle MultiIndex and missing data
            try:
                sp = df["ES=F"].dropna()
                vix = df["^VIX"].dropna()
                gold = df["GC=F"].dropna()
                dax = df["^GDAXI"].dropna()
                nikkei = df["^N225"].dropna()
                tnx = df["^TNX"].dropna()
                dxy = df["DX-Y.NYB"].dropna() if "DX-Y.NYB" in df.columns else None
            except KeyError:
                return None
            
            # Calculate changes with validation
            sp_chg = ((sp.iloc[-1]-sp.iloc[-2])/sp.iloc[-2])*100 if len(sp) >= 2 else 0
            dax_chg = ((dax.iloc[-1]-dax.iloc[-2])/dax.iloc[-2])*100 if len(dax) >= 2 else 0
            nikkei_chg = ((nikkei.iloc[-1]-nikkei.iloc[-2])/nikkei.iloc[-2])*100 if len(nikkei) >= 2 else 0
            tnx_chg = ((tnx.iloc[-1]-tnx.iloc[-2])/tnx.iloc[-2])*100 if len(tnx) >= 2 else 0
            dxy_chg = ((dxy.iloc[-1]-dxy.iloc[-2])/dxy.iloc[-2])*100 if dxy is not None and len(dxy) >= 2 else 0
            
            # Passive flow logic
            day = datetime.now().day
            passive_on = (1 <= day <= 5) or (15 <= day <= 20)
            
            return {
                "sp": sp_chg, 
                "vix": vix.iloc[-1] if len(vix) > 0 else 20,
                "gold": gold.iloc[-1] if len(gold) > 0 else 2000,
                "gold_chg": ((gold.iloc[-1]-gold.iloc[-2])/gold.iloc[-2])*100 if len(gold) >= 2 else 0,
                "dax": dax_chg, 
                "nikkei": nikkei_chg, 
                "tnx": tnx.iloc[-1] if len(tnx) > 0 else 4.0,
                "tnx_chg": tnx_chg, 
                "dxy": dxy.iloc[-1] if dxy is not None and len(dxy) > 0 else 100,
                "dxy_chg": dxy_chg,
                "passive": passive_on,
                "data_quality": "GOOD"
            }
        except Exception as e:
            st.warning(f"⚠️ Macro data fetch failed: {str(e)}. Using cached/default values.")
            return {
                "sp": 0, "vix": 20, "gold": 2000, "gold_chg": 0,
                "dax": 0, "nikkei": 0, "tnx": 4.0, "tnx_chg": 0,
                "dxy": 100, "dxy_chg": 0, "passive": False,
                "data_quality": "DEGRADED"
            }
    
    def detect_correlation_break(self, macro_data):
        """Enhanced correlation break with threshold"""
        us = macro_data['sp']
        eu = macro_data['dax']
        jp = macro_data['nikkei']
        
        # Stronger divergence threshold
        divergence = (us > 1.0 and (eu < -1.0 or jp < -1.0)) or \
                     (us < -1.0 and (eu > 1.0 or jp > 1.0))
        
        return divergence
    
    def check_passive_intensity(self, day, rvol):
        """Passive flow strength validator"""
        passive_window = (1 <= day <= 5) or (15 <= day <= 20)
        
        if passive_window:
            if rvol > 1.5:
                return "STRONG"
            elif rvol > 1.0:
                return "MODERATE"
            else:
                return "WEAK"
        return "NEUTRAL"
    
    def calculate_position_size(self, capital, risk_per_trade, risk_distance, method="FIXED", volatility_mult=1.0):
        """Advanced position sizing with multiple methods"""
        if risk_distance <= 0:
            return 0
        
        if method == "FIXED":
            # Standard fixed risk
            shares = int((risk_per_trade * volatility_mult) / risk_distance)
        
        elif method == "VOLATILITY_ADJUSTED":
            # Reduce size in high volatility
            adjusted_risk = risk_per_trade * volatility_mult
            shares = int(adjusted_risk / risk_distance)
        
        elif method == "KELLY":
            # Kelly Criterion (simplified: assumes 55% win rate, 2:1 R/R)
            win_rate = 0.55
            win_loss_ratio = 2.0
            kelly_fraction = ((win_rate * win_loss_ratio) - (1 - win_rate)) / win_loss_ratio
            kelly_fraction = max(0.1, min(kelly_fraction * 0.5, 0.25))  # Cap at 25% account risk
            adjusted_risk = capital * kelly_fraction
            shares = int(adjusted_risk / risk_distance)
        
        return max(0, shares)
    
    def calculate_sharpe_ratio(self, trades):
        """Calculate Sharpe ratio from closed trades"""
        if len(trades) < 5:
            return None
        
        returns = [t['actual_pnl'] / t['entry'] for t in trades if 'actual_pnl' in t and t['entry'] > 0]
        if not returns:
            return None
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return None
        
        # Annualized Sharpe (assuming 252 trading days)
        sharpe = (mean_return / std_return) * np.sqrt(252)
        return sharpe

engine = InstitutionalAnalyst()

# --- 5. SIDEBAR (Enhanced) ---
with st.sidebar:
    st.header("1. Portfolio Settings")
    
    capital = st.number_input("Total Capital ($)", value=10000, min_value=100, help="Your total account size")
    risk_per_trade = st.number_input("Risk per Trade ($)", value=100, min_value=10, help="Maximum $ risk per position")
    max_portfolio_risk = st.number_input("Max Portfolio Risk (%)", value=6.0, min_value=1.0, max_value=20.0, step=0.5, 
                                         help="Maximum total risk across all open positions (recommended: 5-10%)")
    
    daily_goal = capital * 0.01
    st.caption(f"🎯 Daily Goal (1%): **${daily_goal:.2f}**")
    
    # Portfolio risk display
    portfolio_risk_pct = (st.session_state.total_risk_deployed / capital) * 100
    if portfolio_risk_pct > max_portfolio_risk:
        st.error(f"⚠️ Portfolio Risk: {portfolio_risk_pct:.1f}% (OVER LIMIT)")
    else:
        st.info(f"📊 Portfolio Risk: {portfolio_risk_pct:.1f}% / {max_portfolio_risk:.1f}%")
    
    # Goal progress
    pnl_pct = (st.session_state.daily_pnl / capital) * 100
    if st.session_state.goal_met:
        st.success(f"✅ Goal Achieved: +${st.session_state.daily_pnl:.2f} ({pnl_pct:.2f}%)")
    else:
        st.info(f"📈 Session P&L: ${st.session_state.daily_pnl:.2f} ({pnl_pct:.2f}%)")

    st.divider()
    st.header("2. Asset Selection")
    input_mode = st.radio("Input:", ["VIP List", "Manual Search"])
    if input_mode == "VIP List":
        ticker = st.selectbox("Ticker", VIP_TICKERS, help="High-liquidity institutional favorites")
    else:
        ticker = st.text_input("Ticker", "NVDA", help="Enter any stock symbol").upper()

    st.divider()
    st.header("3. Strategy & Execution")
    strategy = st.selectbox("Mode", ['Long (Buy)', 'Short (Sell)', 'Income (Puts)'],
                           help="Long: Bullish. Short: Bearish. Income: Sell premium for cash flow.")
    
    entry_mode = st.radio("Entry", ["Auto-Limit (Zone)", "Market (Now)", "Manual Override"],
                         help="Auto-Limit: Wait for best price. Market: Enter immediately. Manual: Custom price.")
    manual_price = 0.0
    if entry_mode == "Manual Override":
        manual_price = st.number_input("Entry Price ($)", value=0.0, step=0.01)
    
    stop_mode = st.selectbox("Stop Width", [1.0, 0.5, 0.2], 
                            format_func=lambda x: f"Wide ({x} ATR)" if x==1.0 else f"Medium ({x} ATR)" if x==0.5 else f"Tight ({x} ATR)",
                            help="Stop distance in ATR multiples. Tighter = more risk of premature exit.")
    
    position_sizing_method = st.selectbox("Position Sizing", ["FIXED", "VOLATILITY_ADJUSTED", "KELLY"],
                                         help="FIXED: Standard. VOLATILITY_ADJUSTED: Scales down in high VIX. KELLY: Mathematical optimal sizing.")
    
    premium = 0.0
    if "Income" in strategy:
        premium = st.number_input("Option Premium ($)", value=0.0, step=0.05, help="Per-share premium received")

    st.divider()
    st.header("4. IWT Scorecard")
    st.caption("Hover over the (?) for definitions")
    
    fresh = st.selectbox("Freshness", [2, 1, 0], 
                        format_func=lambda x: {2:'2-Fresh', 1:'1-Used', 0:'0-Stale'}[x],
                        help="Fresh: Untested level. Used: Touched 1-2x. Stale: Weak, tested 3+ times.")
    speed = st.selectbox("Speed Out", [2, 1, 0], 
                        format_func=lambda x: {2:'2-Fast', 1:'1-Avg', 0:'0-Slow'}[x],
                        help="How quickly price left the zone. Fast = strong hands, Slow = weak.")
    time_z = st.selectbox("Time in Zone", [2, 1, 0], 
                         format_func=lambda x: {2:'2-Short', 1:'1-Med', 0:'0-Long'}[x],
                         help="How long price lingered. Short = rejection strength, Long = indecision.")
    
    st.divider()
    if st.button("🔄 Reset Session"):
        st.session_state.goal_met = False
        st.session_state.daily_pnl = 0.0
        st.session_state.total_risk_deployed = 0.0
        st.session_state.open_positions = []
        st.rerun()

# --- 6. MAIN UI ---
st.subheader("📊 Market Intelligence Dashboard")

col_macro, col_scan = st.columns([1, 1])
with col_macro:
    if st.button("🌍 Macro Audit", use_container_width=True):
        with st.spinner("Scanning global markets..."):
            st.session_state.macro = engine.get_macro()

with col_scan:
    if st.button(f"🔎 Scan {ticker}", type="primary", use_container_width=True):
        with st.spinner(f"Analyzing {ticker}..."):
            df, gap, fname, extra = engine.fetch_data(ticker)
            
            if df is None:
                st.error(f"🚫 **{ticker} - Data Error:** {extra}")
                st.session_state.data = None
            else:
                price = df['Close'].iloc[-1]
                st.session_state.data = df
                st.session_state.metrics = {
                    "price": price, 
                    "atr": df['ATRr_14'].iloc[-1],
                    "supp": extra['support'], 
                    "res": extra['resistance'],
                    "rvol": df['RVOL'].iloc[-1], 
                    "gap": gap, 
                    "name": fname,
                    "trend": extra['trend_strength'],
                    "rsi": extra['rsi']
                }
                st.success(f"✅ {fname} loaded successfully")

# --- 7. MARKET CONTEXT DISPLAY ---
if st.session_state.macro:
    m = st.session_state.macro
    
    # Data quality warning
    if m.get('data_quality') == 'DEGRADED':
        st.warning("⚠️ **Data Quality Issue:** Using cached/estimated values. Results may be less accurate.")
    
    # VIX Regime Classification
    vix_regime = engine.classify_vix_regime(m['vix'])
    regime_guide = engine.get_regime_guidance(vix_regime)
    
    # Market Hours
    market_phase, phase_desc = engine.get_market_hours_status()
    
    # Display regime box
    if vix_regime in ["EXTREME", "HIGH"]:
        st.error(f"🚨 **VIX REGIME: {regime_guide['desc']} ({m['vix']:.1f})**")
        st.markdown(f"<div class='risk-warning'>{regime_guide['action']}</div>", unsafe_allow_html=True)
    elif vix_regime == "ELEVATED":
        st.warning(f"⚠️ **VIX REGIME: {regime_guide['desc']} ({m['vix']:.1f})**")
        st.info(regime_guide['action'])
    else:
        st.success(f"✅ **VIX REGIME: {regime_guide['desc']} ({m['vix']:.1f})**")
    
    # Market hours warning
    if market_phase in ["PRE_MARKET", "AFTER_HOURS"]:
        st.warning(f"⏰ **{phase_desc}**")
    elif market_phase == "LUNCH":
        st.info(f"{phase_desc}")
    else:
        st.caption(f"{phase_desc}")
    
    # Passive flow intensity
    flow_strength = engine.check_passive_intensity(
        datetime.now().day, 
        st.session_state.metrics.get('rvol', 0) if st.session_state.metrics else 0
    )
    
    if flow_strength == "STRONG":
        st.success("🌊 **STRONG PASSIVE INFLOWS** (Calendar + Volume): Institutional dip-buying dominant.")
    elif flow_strength == "MODERATE":
        st.info("🌊 **MODERATE PASSIVE INFLOWS** (Calendar window, moderate volume)")
    elif flow_strength == "WEAK":
        st.warning("🌊 **WEAK PASSIVE INFLOWS** (Calendar window but low volume - retail-driven)")
    else:
        st.info("⏸️ **PASSIVE FLOWS NEUTRAL:** Active trading environment.")
    
    # Correlation break
    if engine.detect_correlation_break(m):
        st.error("🌍 **GLOBAL CORRELATION BREAK:** US/EU/Asia diverging. Elevated volatility risk. Consider tighter stops or smaller size.")
    
    # Macro dashboard
    with st.expander("🌍 Global Macro Dashboard", expanded=True):
        col1, col2, col3, col4, col5 = st.columns(5)
        
        col1.metric("🇺🇸 S&P 500", f"{m['sp']:.2f}%", help="US Equity Futures")
        col2.metric("🇩🇪 DAX", f"{m['dax']:.2f}%", help="European Equities")
        col3.metric("🇯🇵 Nikkei", f"{m['nikkei']:.2f}%", help="Asian Equities")
        
        # Dollar index
        dxy_color = "inverse" if m['dxy_chg'] > 0 else "normal"
        col4.metric("💵 DXY", f"{m['dxy']:.1f}", delta=f"{m['dxy_chg']:.2f}%", delta_color=dxy_color, 
                   help="US Dollar Strength (↑ = headwind for commodities/intl stocks)")
        
        # Yields
        tnx_color = "inverse" if m['tnx_chg'] > 0 else "normal"
        col5.metric("📈 10Y Yield", f"{m['tnx']:.2f}%", delta=f"{m['tnx_chg']:.2f}%", delta_color=tnx_color,
                   help="Rising yields = headwind for growth stocks")

# --- 8. ASSET ANALYSIS ---
if st.session_state.data is not None:
    m = st.session_state.metrics
    df = st.session_state.data
    
    st.divider()
    st.header(f"📈 {m['name']} ({ticker})")
    
    # Key metrics row
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("Price", f"${m['price']:.2f}", help="Last close")
    
    # Enhanced gap analysis
    if abs(m['gap']) > 2.0:
        if m['rvol'] > 1.5:
            col2.metric("Gap", f"{m['gap']:.2f}%", delta="🚀 PROFESSIONAL", 
                       help="High-volume gap = institutional breakout, likely to persist")
        else:
            col2.metric("Gap", f"{m['gap']:.2f}%", delta="⚠️ NOVICE",
                       help="Low-volume gap = retail FOMO, likely to fill")
    else:
        col2.metric("Gap", f"{m['gap']:.2f}%")
    
    # Volume
    vol_status = "🔥 HOT" if m['rvol'] > 1.5 else "✅ NORMAL" if m['rvol'] > 0.8 else "💤 THIN"
    col3.metric("Volume", f"{m['rvol']:.1f}x", delta=vol_status)
    
    # Trend strength
    trend_emoji = {"STRONG_BULL": "🚀", "BULL": "📈", "NEUTRAL": "➡️", "BEAR": "📉", "STRONG_BEAR": "🔻"}
    col4.metric("Trend", m['trend'], delta=trend_emoji.get(m['trend'], "➡️"))
    
    # RSI
    rsi_status = "⚠️ OB" if m['rsi'] > 70 else "⚠️ OS" if m['rsi'] < 30 else "✅"
    col5.metric("RSI", f"{m['rsi']:.0f}", delta=rsi_status, help="Relative Strength Index")
    
    # Support/Resistance levels
    st.caption(f"**Key Levels:** Support ${m['supp']:.2f} | Resistance ${m['res']:.2f}")
    
    # Enhanced chart
    st.subheader("📊 Technical Chart")
    
    chart_data = df.iloc[-60:]
    
    # Create custom colors for volume
    colors = ['red' if chart_data['Close'].iloc[i] < chart_data['Open'].iloc[i] else 'green' 
              for i in range(len(chart_data))]
    
    # Plot with zones highlighted
    fig, axes = mpf.plot(
        chart_data,
        type='candle',
        style='yahoo',
        volume=True,
        addplot=[
            mpf.make_addplot(chart_data['SMA_20'], color='blue', width=1.5, label='SMA 20'),
            mpf.make_addplot(chart_data['SMA_50'], color='orange', width=1.5, label='SMA 50'),
            mpf.make_addplot(chart_data['ST_VAL'], color='purple', width=1, linestyle='--', label='SuperTrend')
        ],
        hlines=dict(
            hlines=[m['supp'], m['res']], 
            colors=['green', 'red'], 
            linestyle='-.',
            linewidths=2
        ),
        returnfig=True,
        figsize=(14, 7),
        title=f"{ticker} - Last 60 Days"
    )
    
    st.pyplot(fig)
    
    # --- TRADE CALCULATION ---
    st.divider()
    st.subheader("🎯 Trade Setup Calculator")
    
    # Entry logic
    if entry_mode == "Manual Override":
        entry = manual_price
    elif "Short" in strategy:
        entry = m['res'] if "Auto" in entry_mode else m['price']
    else:
        entry = m['supp'] if "Auto" in entry_mode else m['price']
    
    # Get volatility multiplier from regime
    if st.session_state.macro:
        vix_regime = engine.classify_vix_regime(st.session_state.macro['vix'])
        regime_guide = engine.get_regime_guidance(vix_regime)
        vol_multiplier = regime_guide['size_multiplier']
        stop_multiplier = regime_guide['stop_multiplier']
    else:
        vol_multiplier = 1.0
        stop_multiplier = 1.0
    
    # Stop/Target logic
    if "Short" in strategy:
        stop = entry + (m['atr'] * stop_mode * stop_multiplier)
        target = m['supp']
        risk = stop - entry
        reward = entry - target
    elif "Income" in strategy:
        stop = entry
        target = entry
        risk = entry * 0.1
        reward = premium
    else:  # Long
        stop = entry - (m['atr'] * stop_mode * stop_multiplier)
        target = m['res']
        risk = entry - stop
        reward = target - entry
    
    # Risk/Reward ratio
    rr = reward / risk if risk > 0 else 0
    
    # Position sizing
    shares = engine.calculate_position_size(
        capital, 
        risk_per_trade, 
        risk, 
        position_sizing_method, 
        vol_multiplier
    )
    
    # Calculate total risk
    total_trade_risk = shares * risk if shares > 0 else 0
    
    # Display setup
    col_setup1, col_setup2 = st.columns(2)
    
    with col_setup1:
        st.markdown("**📍 Price Levels**")
        st.code(f"""
Entry:   ${entry:.2f}
Stop:    ${stop:.2f}
Target:  ${target:.2f}
        """)
        
    with col_setup2:
        st.markdown("**💰 Position Details**")
        sizing_note = f" (VIX-adjusted: {vol_multiplier:.1f}x)" if vol_multiplier != 1.0 else ""
        st.code(f"""
Size:    {shares} {'contracts' if 'Income' in strategy else 'shares'}{sizing_note}
Risk:    ${total_trade_risk:.2f}
Reward:  ${shares * reward:.2f} (est.)
R/R:     {rr:.2f}
        """)
    
    # --- SCORING & VERDICT ---
    st.divider()
    st.subheader("🚦 The Quantum Verdict")
    
    # Base score
    score_rr = 2 if rr >= 3 or ("Income" in strategy and rr > 0.1) else 1 if rr >= 2 else 0
    total_score = fresh + time_z + speed + score_rr
    
    # Apply penalties
    penalties = []
    
    # Warsh penalty (structural yield moves)
    warsh_penalty = False
    if st.session_state.macro and st.session_state.macro['tnx_chg'] > 1.0 and ticker in GROWTH_TICKERS and "Long" in strategy:
        total_score -= 2
        warsh_penalty = True
        penalties.append("Warsh Penalty (-2): Rising yields >1%")
    
    # Market hours penalty
    market_phase, _ = engine.get_market_hours_status()
    if market_phase in ["LUNCH", "PRE_MARKET", "AFTER_HOURS"]:
        total_score -= 1
        penalties.append(f"Market Hours (-1): {market_phase}")
    
    # Trend misalignment penalty
    if "Long" in strategy and m['trend'] in ["BEAR", "STRONG_BEAR"]:
        total_score -= 1
        penalties.append("Trend Misalignment (-1): Long in downtrend")
    elif "Short" in strategy and m['trend'] in ["BULL", "STRONG_BULL"]:
        total_score -= 1
        penalties.append("Trend Misalignment (-1): Short in uptrend")
    
    # Portfolio concentration check
    sector = SECTOR_MAP.get(ticker, "Unknown")
    sector_exposure = sum(1 for p in st.session_state.open_positions if SECTOR_MAP.get(p['ticker'], '') == sector)
    if sector_exposure >= 2:
        total_score -= 1
        penalties.append(f"Concentration Risk (-1): {sector_exposure+1} positions in {sector}")
    
    # Display verdict
    col_verdict, col_analysis = st.columns([1, 1])
    
    with col_verdict:
        # Goal override check
        if st.session_state.goal_met:
            st.error("## 🛑 DAILY GOAL ACHIEVED")
            st.markdown("<div class='risk-warning'><strong>STOP TRADING.</strong> You've hit your 1% target. Consistency beats intensity. Close your terminal and protect your gains.</div>", unsafe_allow_html=True)
            can_trade = False
        
        # Portfolio risk check
        elif (st.session_state.total_risk_deployed + total_trade_risk) > (capital * max_portfolio_risk / 100):
            st.error("## 🛑 PORTFOLIO RISK LIMIT")
            st.markdown(f"<div class='risk-warning'>Adding this trade would exceed your {max_portfolio_risk}% portfolio risk limit. Close existing positions or reduce size.</div>", unsafe_allow_html=True)
            can_trade = False
        
        else:
            can_trade = True
            if total_score >= 7:
                st.success(f"## 🟢 GREEN LIGHT\n**Score: {total_score}/8**")
                st.caption("✅ **Action:** Execute with full confidence.")
            elif total_score >= 5:
                st.warning(f"## 🟡 YELLOW LIGHT\n**Score: {total_score}/8**")
                st.caption("⚠️ **Action:** Reduce size by 50% OR wait for confirmation.")
            else:
                st.error(f"## 🔴 RED LIGHT\n**Score: {total_score}/8**")
                st.caption("🛑 **Action:** DO NOT TRADE. Setup is flawed.")
        
        # Display penalties
        if penalties:
            st.markdown("**Penalties Applied:**")
            for p in penalties:
                st.caption(f"• {p}")
    
    with col_analysis:
        st.markdown("**📋 Setup Analysis**")
        
        # Checklist
        checks = []
        checks.append(("✅" if fresh == 2 else "⚠️" if fresh == 1 else "❌", f"Freshness: {['Stale', 'Used', 'Fresh'][fresh]}"))
        checks.append(("✅" if score_rr == 2 else "⚠️" if score_rr == 1 else "❌", f"R/R: {rr:.2f} ({['Poor', 'Acceptable', 'Excellent'][score_rr]})"))
        checks.append(("✅" if abs(m['gap']) > 2 else "➖", f"Gap: {m['gap']:.2f}% ({'Significant' if abs(m['gap']) > 2 else 'Normal'})"))
        checks.append(("✅" if m['rvol'] > 1.2 else "⚠️", f"Volume: {m['rvol']:.1f}x ({'Strong' if m['rvol'] > 1.2 else 'Weak'})"))
        
        for icon, text in checks:
            st.caption(f"{icon} {text}")
        
        # Warren AI export
        st.markdown("---")
        st.caption("**Copy for WarrenAI:**")
        ai_export = f"""
[ARCHITECT REVIEW - {ticker}]
Strategy: {strategy}
Score: {total_score}/8
Verdict: {'GREEN' if total_score>=7 else 'YELLOW' if total_score>=5 else 'RED'}
Entry: ${entry:.2f} | Stop: ${stop:.2f} | Target: ${target:.2f}
R/R: {rr:.2f} | Size: {shares} shares
VIX Regime: {vix_regime if st.session_state.macro else 'N/A'}
Passive Flow: {flow_strength}
10Y Yield: {'RISING >1%' if warsh_penalty else 'STABLE'}
        """
        st.code(ai_export.strip(), language='text')
    
    # --- TRADE EXECUTION ---
    if can_trade and total_score >= 5:  # Only show execution for tradeable setups
        st.divider()
        st.subheader("⚡ Trade Execution")
        
        col_exec1, col_exec2 = st.columns(2)
        
        with col_exec1:
            if st.button("📝 Log as PAPER TRADE", use_container_width=True, type="secondary"):
                trade_record = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ticker": ticker,
                    "action": strategy,
                    "entry": entry,
                    "stop": stop,
                    "target": target,
                    "shares": shares,
                    "score": total_score,
                    "risk": total_trade_risk,
                    "expected_reward": shares * reward,
                    "rr_ratio": rr,
                    "status": "PAPER"
                }
                st.session_state.journal.append(trade_record)
                st.success("📋 Paper trade logged for analysis!")
        
        with col_exec2:
            if st.button("💵 LOG AS LIVE TRADE", use_container_width=True, type="primary"):
                trade_record = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ticker": ticker,
                    "action": strategy,
                    "entry": entry,
                    "stop": stop,
                    "target": target,
                    "shares": shares,
                    "score": total_score,
                    "risk": total_trade_risk,
                    "expected_reward": shares * reward,
                    "rr_ratio": rr,
                    "status": "OPEN"
                }
                st.session_state.journal.append(trade_record)
                st.session_state.open_positions.append(trade_record)
                st.session_state.total_risk_deployed += total_trade_risk
                st.success("✅ Live trade logged! Position is now OPEN.")
                st.rerun()

# --- 9. PORTFOLIO MANAGEMENT ---
if st.session_state.open_positions:
    st.divider()
    st.subheader("📊 Open Positions")
    
    positions_df = pd.DataFrame(st.session_state.open_positions)
    positions_df = positions_df[['ticker', 'action', 'entry', 'stop', 'target', 'shares', 'risk', 'score']]
    st.dataframe(positions_df, use_container_width=True)
    
    # Close position interface
    st.markdown("**Close a Position:**")
    col_close1, col_close2, col_close3 = st.columns(3)
    
    with col_close1:
        position_to_close = st.selectbox("Select Position", [p['ticker'] for p in st.session_state.open_positions])
    
    with col_close2:
        exit_price = st.number_input("Exit Price ($)", value=0.0, step=0.01)
    
    with col_close3:
        if st.button("Close Position"):
            if exit_price > 0:
                # Find and close the position
                for i, pos in enumerate(st.session_state.open_positions):
                    if pos['ticker'] == position_to_close:
                        # Calculate actual P&L
                        if "Long" in pos['action']:
                            actual_pnl = (exit_price - pos['entry']) * pos['shares']
                        else:
                            actual_pnl = (pos['entry'] - exit_price) * pos['shares']
                        
                        # Update records
                        pos['exit_price'] = exit_price
                        pos['actual_pnl'] = actual_pnl
                        pos['status'] = 'CLOSED'
                        
                        st.session_state.closed_trades.append(pos)
                        st.session_state.daily_pnl += actual_pnl
                        st.session_state.total_risk_deployed -= pos['risk']
                        
                        # Check if goal met
                        if st.session_state.daily_pnl >= daily_goal:
                            st.session_state.goal_met = True
                        
                        # Remove from open positions
                        st.session_state.open_positions.pop(i)
                        st.success(f"✅ Closed {position_to_close}: P&L = ${actual_pnl:.2f}")
                        st.rerun()
                        break
            else:
                st.error("Please enter a valid exit price")

# --- 10. PERFORMANCE ANALYTICS ---
if st.session_state.closed_trades:
    st.divider()
    st.subheader("📈 Performance Analytics")
    
    closed_df = pd.DataFrame(st.session_state.closed_trades)
    
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    
    # Win rate
    wins = len(closed_df[closed_df['actual_pnl'] > 0])
    total_trades = len(closed_df)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    col_stats1.metric("Win Rate", f"{win_rate:.1f}%", 
                     delta=f"{wins}/{total_trades} trades",
                     help="Percentage of profitable trades")
    
    # Average R/R
    avg_rr = closed_df['rr_ratio'].mean()
    col_stats2.metric("Avg R/R", f"{avg_rr:.2f}",
                     help="Average risk/reward ratio of executed trades")
    
    # Total P&L
    total_pnl = closed_df['actual_pnl'].sum()
    col_stats3.metric("Total P&L", f"${total_pnl:.2f}",
                     delta=f"{(total_pnl/capital)*100:.2f}%",
                     help="Total profit/loss from closed trades")
    
    # Sharpe Ratio
    sharpe = engine.calculate_sharpe_ratio(st.session_state.closed_trades)
    if sharpe:
        sharpe_quality = "Excellent" if sharpe > 2 else "Good" if sharpe > 1 else "Poor"
        col_stats4.metric("Sharpe Ratio", f"{sharpe:.2f}",
                         delta=sharpe_quality,
                         help="Risk-adjusted return metric (>1 is good, >2 is excellent)")
    else:
        col_stats4.metric("Sharpe Ratio", "N/A", help="Need 5+ trades to calculate")
    
    # Trade history table
    with st.expander("📋 Trade History", expanded=False):
        history_df = closed_df[['timestamp', 'ticker', 'action', 'entry', 'exit_price', 'actual_pnl', 'score']]
        st.dataframe(history_df, use_container_width=True)

# --- 11. JOURNAL EXPORT ---
if st.session_state.journal:
    st.divider()
    st.subheader("📓 Trading Journal")
    
    journal_df = pd.DataFrame(st.session_state.journal)
    st.dataframe(journal_df, use_container_width=True)
    
    # Export options
    col_export1, col_export2 = st.columns(2)
    
    with col_export1:
        csv = journal_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Full Journal (CSV)",
            data=csv,
            file_name=f"trade_journal_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_export2:
        if st.session_state.closed_trades:
            closed_csv = pd.DataFrame(st.session_state.closed_trades).to_csv(index=False).encode('utf-8')
            st.download_button(
                "📊 Download Performance Report (CSV)",
                data=closed_csv,
                file_name=f"performance_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

else:
    st.info("👈 **Quick Start:** 1. Check Macro → 2. Scan a ticker → 3. Review verdict → 4. Log trade")

# --- 12. FOOTER ---
st.divider()
st.caption("Quantum Maestro V12.0 | Built for institutional-grade risk management | Educational purposes only")
