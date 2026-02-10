# Copyright (c) 2026 Gabriel Mahia. All Rights Reserved.
# Unauthorized copying of this file, via any medium, is strictly prohibited.
# Proprietary and confidential.
# Written by Gabriel Mahia, 2026
# app.py

# =============================================================================
# 🏛️ QUANTUM MAESTRO — ULTIMATE INSTITUTIONAL EDITION V13.0 FINAL
# =============================================================================
# Multi-Algorithm Fusion | Pattern Recognition | Adaptive Risk | Performance Analytics
# Combines: Teri Ijeoma's IWT + WarrenAI Macro + 15+ Technical Indicators
# Educational simulation tool — NOT financial advice.
# =============================================================================

import streamlit as st
import yfinance as yf
import pandas_ta as ta
import mplfinance as mpf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from datetime import datetime, time, timedelta
import pytz

# --- 1. CONFIGURATION ---
# 🇺🇸 US / GLOBAL VIPs
VIP_TICKERS = ["NVDA", "AAPL", "AMZN", "GOOGL", "TSLA", "MSFT", "META", "AMD", "NFLX", "SPY", "QQQ", "IWM", "GLD", "SLV", "USO"]
GROWTH_TICKERS = ["NVDA", "AAPL", "AMZN", "GOOGL", "TSLA", "MSFT", "META", "AMD", "NFLX", "QQQ", "ARKK", "COIN", "SHOP", "SQ"]
COMMODITY_TICKERS = ["GLD", "SLV", "GDX", "USO", "XLE", "FCX"]
VALUE_TICKERS = ["JPM", "BAC", "XOM", "CVX", "BRK.B", "JNJ", "PG"]

# 🇰🇪 KENYA VIPs (The "Zidii Trader" Watchlist - Nairobi Securities Exchange)
# SCOM: Safaricom (The Market Mover - M-Pesa parent) | EQTY: Equity Bank | KCB: KCB Group
# EABL: East African Breweries | COOP: Co-op Bank | ABSA: Absa Kenya | NCBA: NCBA Group | BAT: BAT Kenya
KENYA_TICKERS = ["SCOM.NR", "EQTY.NR", "KCB.NR", "EABL.NR", "COOP.NR", "ABSA.NR", "NCBA.NR", "BAT.NR"]

# 📊 COMBINED TICKER LIST (US + Kenya)
ALL_TICKERS = VIP_TICKERS + KENYA_TICKERS

SECTOR_MAP = {
    # 🇺🇸 US Stocks
    "NVDA": "Tech", "AMD": "Tech", "MSFT": "Tech", "AAPL": "Tech", "META": "Tech", "GOOGL": "Tech",
    "TSLA": "Auto", "AMZN": "Consumer", "NFLX": "Media", "SPY": "Index", "QQQ": "Tech-Index", "IWM": "Index",
    "GLD": "Commodity", "SLV": "Commodity", "GDX": "Mining", "USO": "Energy", "XLE": "Energy",
    "JPM": "Finance", "BAC": "Finance", "XOM": "Energy", "CVX": "Energy", "BRK.B": "Conglomerate",
    "JNJ": "Healthcare", "PG": "Staples", "ARKK": "Thematic", "COIN": "Crypto", "SHOP": "Tech", "SQ": "Fintech", "FCX": "Materials",
    # 🇰🇪 Kenya Stocks (Nairobi Securities Exchange)
    "SCOM.NR": "Telecom", "EQTY.NR": "Finance", "KCB.NR": "Finance",
    "EABL.NR": "Consumer", "COOP.NR": "Finance", "ABSA.NR": "Finance",
    "NCBA.NR": "Finance", "BAT.NR": "Consumer"
}

COMMISSION_PER_SHARE = 0.005
SLIPPAGE_BPS = 5

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
    .signal-bull { 
        background-color: #d4edda; 
        color: #155724;
        padding: 8px 12px; 
        border-radius: 4px; 
        margin: 5px 0;
        border: 1px solid #28a745;
        font-weight: 500;
    }
    .signal-bear { 
        background-color: #f8d7da; 
        color: #721c24;
        padding: 8px 12px; 
        border-radius: 4px; 
        margin: 5px 0;
        border: 1px solid #dc3545;
        font-weight: 500;
    }
    .signal-neutral { 
        background-color: #fff3cd; 
        color: #856404;
        padding: 8px 12px; 
        border-radius: 4px; 
        margin: 5px 0;
        border: 1px solid #ffc107;
        font-weight: 500;
    }
    @media (prefers-color-scheme: dark) {
        .signal-bull { 
            background-color: #1e4620; 
            color: #7dcea0;
            border-color: #28a745;
        }
        .signal-bear { 
            background-color: #4a1c1c; 
            color: #f1948a;
            border-color: #dc3545;
        }
        .signal-neutral { 
            background-color: #4a3f1a; 
            color: #f9e79f;
            border-color: #ffc107;
        }
    }
</style>
""", unsafe_allow_html=True)

# --- 2. LEGAL & ONBOARDING ---
st.title("🏛️ Quantum Maestro [TradingBot]: Institutional Edition")
st.caption("Portfolio Risk Architecture | Volatility Regimes | Multi-Algorithm Fusion | IWT Execution Discipline | Performance Analytics")

with st.expander("⚠️ READ FIRST: Legal Disclaimer", expanded=True):
    st.markdown("""
    **1. No Affiliation:** Independent tool. Not affiliated with Trade and Travel or any trading organization.
    **2. Educational Use Only:** Not financial advice. For simulation and learning purposes.
    **3. Risk Warning:** Trading involves substantial risk of loss. Past performance does not guarantee future results.
    **4. Data Disclaimer:** Market data provided by Yahoo Finance. Delays and inaccuracies may occur.
    """)
    agree = st.checkbox("✅ I understand this is not financial advice and I am using this tool for educational purposes.")

if not agree:
    st.warning("🛑 Please accept the disclaimer above.")
    st.stop()

st.divider()

# --- 3. SESSION STATE ---
if 'data' not in st.session_state: st.session_state.data = None
if 'metrics' not in st.session_state: st.session_state.metrics = {}
if 'macro' not in st.session_state: st.session_state.macro = None
if 'signals' not in st.session_state: st.session_state.signals = {}
if 'journal' not in st.session_state: st.session_state.journal = []
if 'open_positions' not in st.session_state: st.session_state.open_positions = []
if 'closed_trades' not in st.session_state: st.session_state.closed_trades = []
if 'goal_met' not in st.session_state: st.session_state.goal_met = False
if 'daily_pnl' not in st.session_state: st.session_state.daily_pnl = 0.0
if 'total_risk_deployed' not in st.session_state: st.session_state.total_risk_deployed = 0.0
if 'consecutive_losses' not in st.session_state: st.session_state.consecutive_losses = 0

# --- 4. INSTITUTIONAL ANALYST ENGINE ---
class InstitutionalAnalyst:
    
    def __init__(self):
        self.vix_regimes = {
            "EXTREME_LOW": (0, 12), "LOW": (12, 15), "NORMAL": (15, 20),
            "ELEVATED": (20, 30), "HIGH": (30, 40), "EXTREME": (40, 100)
        }
        self.fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    
    def get_market_hours_status(self):
        try:
            et = pytz.timezone('US/Eastern')
            now = datetime.now(et)
            current_time = now.time()
            
            if current_time < time(9, 30):
                return "PRE_MARKET", "⏰ Pre-Market (Higher volatility, lower liquidity)"
            elif current_time < time(10, 0):
                return "OPENING", "🔔 Opening Range (Wait for direction, avoid chasing)"
            elif current_time < time(12, 0):
                return "MORNING", "☀️ Morning Session (Prime trading window)"
            elif current_time < time(14, 0):
                return "LUNCH", "🍴 Lunch Hour (Reduced volume, avoid new positions)"
            elif current_time < time(15, 0):
                return "AFTERNOON", "🌤️ Afternoon Session (Trend continuation)"
            elif current_time < time(16, 0):
                return "POWER_HOUR", "⚡ Power Hour (Institutional positioning, high volume)"
            else:
                return "AFTER_HOURS", "🌙 After Hours (Extended hours risk)"
        except:
            return "UNKNOWN", "⚠️ Unable to determine market hours"
    
    def classify_vix_regime(self, vix_level):
        for regime, (low, high) in self.vix_regimes.items():
            if low <= vix_level < high:
                return regime
        return "EXTREME"
    
    def get_regime_guidance(self, regime):
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
        try:
            ticker_obj = yf.Ticker(t)
            data = ticker_obj.history(period="1y")
            
            if data.empty or len(data) < 50:
                return None, None, f"ERROR: Insufficient data for {t}. Need at least 50 trading days."
            
            try: 
                full_name = ticker_obj.info.get('longName', t)
                beta = ticker_obj.info.get('beta', 1.0)
            except: 
                full_name = t
                beta = 1.0
            
            # Core indicators
            data.ta.atr(length=14, append=True)
            data.ta.rsi(length=14, append=True)
            data.ta.macd(fast=12, slow=26, signal=9, append=True)
            data.ta.bbands(length=20, std=2, append=True)
            data.ta.stoch(k=14, d=3, append=True)
            data.ta.adx(length=14, append=True)
            data.ta.obv(append=True)
            data.ta.mfi(length=14, append=True)
            data.ta.willr(length=14, append=True)
            
            data.ta.sma(length=20, append=True)
            data.ta.sma(length=50, append=True)
            data.ta.sma(length=200, append=True)
            data.ta.ema(length=12, append=True)
            data.ta.ema(length=26, append=True)
            
            try:
                st_data = data.ta.supertrend(length=10, multiplier=3)
                data['ST_VAL'] = st_data.iloc[:, 0]
                data['ST_DIR'] = st_data.iloc[:, 1]
            except:
                data['ST_VAL'] = data['Close']
                data['ST_DIR'] = 1
            
            try:
                ich = data.ta.ichimoku()[0]
                data['ICH_SPAN_A'] = ich['ISA_9']
                data['ICH_SPAN_B'] = ich['ISB_26']
            except:
                data['ICH_SPAN_A'] = data['Close']
                data['ICH_SPAN_B'] = data['Close']
            
            vol_sma = data['Volume'].rolling(20).mean()
            data['RVOL'] = data['Volume'] / vol_sma
            
            # Support/Resistance
            swing_high = data['High'].rolling(20).max()
            swing_low = data['Low'].rolling(20).min()
            
            recent_high = swing_high.iloc[-1]
            recent_low = swing_low.iloc[-1]
            price_range = recent_high - recent_low
            
            fib_levels = {}
            for level in self.fib_levels:
                fib_levels[f"fib_{level}"] = recent_high - (price_range * level)
            
            try:
                local_min_idx = argrelextrema(data['Close'].values, np.less_equal, order=5)[0]
                local_max_idx = argrelextrema(data['Close'].values, np.greater_equal, order=5)[0]
                
                if len(local_min_idx) > 0:
                    recent_mins = data.iloc[local_min_idx[-3:]]['Close'].values if len(local_min_idx) >= 3 else data.iloc[local_min_idx]['Close'].values
                    support_level = np.mean(recent_mins)
                else:
                    support_level = swing_low.iloc[-1]
                
                if len(local_max_idx) > 0:
                    recent_maxs = data.iloc[local_max_idx[-3:]]['Close'].values if len(local_max_idx) >= 3 else data.iloc[local_max_idx]['Close'].values
                    resistance_level = np.mean(recent_maxs)
                else:
                    resistance_level = swing_high.iloc[-1]
            except:
                support_level = swing_low.iloc[-1]
                resistance_level = swing_high.iloc[-1]
            
            support_touches = 0
            resistance_touches = 0
            for i in range(max(0, len(data)-60), len(data)):
                if abs(data['Low'].iloc[i] - support_level) / support_level < 0.01:
                    support_touches += 1
                if abs(data['High'].iloc[i] - resistance_level) / resistance_level < 0.01:
                    resistance_touches += 1
            
            if len(data) >= 2:
                prev_close = data['Close'].iloc[-2]
                curr_open = data['Open'].iloc[-1]
                gap_pct = ((curr_open - prev_close) / prev_close) * 100
            else:
                gap_pct = 0.0
            
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
            
            trend_strength = "STRONG_BULL" if trend_score >= 4 else "BULL" if trend_score == 3 else \
                           "NEUTRAL" if trend_score == 2 else "BEAR" if trend_score == 1 else "STRONG_BEAR"
            
            patterns = self.detect_patterns(data)
            divergences = self.detect_divergences(data)
            
            metrics = {
                "support": support_level,
                "resistance": resistance_level,
                "support_touches": support_touches,
                "resistance_touches": resistance_touches,
                "trend_strength": trend_strength,
                "rsi": data['RSI_14'].iloc[-1] if 'RSI_14' in data.columns else 50,
                "macd": data['MACD_12_26_9'].iloc[-1] if 'MACD_12_26_9' in data.columns else 0,
                "macd_signal": data['MACDs_12_26_9'].iloc[-1] if 'MACDs_12_26_9' in data.columns else 0,
                "macd_hist": data['MACDh_12_26_9'].iloc[-1] if 'MACDh_12_26_9' in data.columns else 0,
                "bb_upper": data['BBU_20_2.0'].iloc[-1] if 'BBU_20_2.0' in data.columns else price * 1.02,
                "bb_lower": data['BBL_20_2.0'].iloc[-1] if 'BBL_20_2.0' in data.columns else price * 0.98,
                "bb_width": (data['BBU_20_2.0'].iloc[-1] - data['BBL_20_2.0'].iloc[-1]) if 'BBU_20_2.0' in data.columns else 0,
                "stoch_k": data['STOCHk_14_3_3'].iloc[-1] if 'STOCHk_14_3_3' in data.columns else 50,
                "stoch_d": data['STOCHd_14_3_3'].iloc[-1] if 'STOCHd_14_3_3' in data.columns else 50,
                "adx": data['ADX_14'].iloc[-1] if 'ADX_14' in data.columns else 20,
                "obv": data['OBV'].iloc[-1] if 'OBV' in data.columns else 0,
                "mfi": data['MFI_14'].iloc[-1] if 'MFI_14' in data.columns else 50,
                "willr": data['WILLR_14'].iloc[-1] if 'WILLR_14' in data.columns else -50,
                "ich_position": "ABOVE_CLOUD" if price > max(data['ICH_SPAN_A'].iloc[-1], data['ICH_SPAN_B'].iloc[-1]) else \
                               "BELOW_CLOUD" if price < min(data['ICH_SPAN_A'].iloc[-1], data['ICH_SPAN_B'].iloc[-1]) else "IN_CLOUD",
                "fib_levels": fib_levels,
                "beta": beta,
                "patterns": patterns,
                "divergences": divergences
            }
            
            return data, metrics, full_name
            
        except Exception as e:
            return None, None, f"ERROR: {str(e)}"
    
    def detect_patterns(self, data):
        patterns = []
        if len(data) < 3:
            return patterns
        
        last_3 = data.iloc[-3:]
        c0, c1, c2 = last_3.iloc[0], last_3.iloc[1], last_3.iloc[2]
        
        if c1['Close'] < c1['Open'] and c2['Close'] > c2['Open'] and \
           c2['Open'] < c1['Close'] and c2['Close'] > c1['Open']:
            patterns.append("🟢 Bullish Engulfing")
        
        if c1['Close'] > c1['Open'] and c2['Close'] < c2['Open'] and \
           c2['Open'] > c1['Close'] and c2['Close'] < c1['Open']:
            patterns.append("🔴 Bearish Engulfing")
        
        body = abs(c2['Close'] - c2['Open'])
        lower_wick = min(c2['Open'], c2['Close']) - c2['Low']
        upper_wick = c2['High'] - max(c2['Open'], c2['Close'])
        
        if lower_wick > 2 * body and upper_wick < body:
            patterns.append("🔨 Hammer (Bullish)")
        
        if upper_wick > 2 * body and lower_wick < body:
            patterns.append("💫 Shooting Star (Bearish)")
        
        if body < (c2['High'] - c2['Low']) * 0.1:
            patterns.append("➕ Doji (Indecision)")
        
        return patterns
    
    def detect_divergences(self, data):
        divergences = []
        if len(data) < 20:
            return divergences
        
        recent = data.iloc[-20:]
        
        if 'RSI_14' in recent.columns:
            price_highs = recent['High'].iloc[-10:].max()
            price_lows = recent['Low'].iloc[-10:].min()
            rsi_highs = recent['RSI_14'].iloc[-10:].max()
            rsi_lows = recent['RSI_14'].iloc[-10:].min()
            
            current_price = recent['Close'].iloc[-1]
            current_rsi = recent['RSI_14'].iloc[-1]
            
            if current_price < price_lows * 1.01 and current_rsi > rsi_lows * 1.05:
                divergences.append("🟢 RSI Bullish Divergence")
            
            if current_price > price_highs * 0.99 and current_rsi < rsi_highs * 0.95:
                divergences.append("🔴 RSI Bearish Divergence")
        
        if 'MACD_12_26_9' in recent.columns:
            macd_current = recent['MACD_12_26_9'].iloc[-1]
            macd_prev_high = recent['MACD_12_26_9'].iloc[-10:].max()
            
            price_current = recent['Close'].iloc[-1]
            price_prev_high = recent['High'].iloc[-10:].max()
            
            if price_current > price_prev_high * 0.99 and macd_current < macd_prev_high * 0.95:
                divergences.append("🔴 MACD Bearish Divergence")
        
        return divergences

    def get_macro(self):
        try:
            tickers = ["ES=F", "^VIX", "GC=F", "^GDAXI", "^N225", "^TNX", "DX-Y.NYB"]
            df = yf.download(tickers, period="5d", progress=False, timeout=10)['Close']
            
            if df.empty: 
                return None
            
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
            
            sp_chg = ((sp.iloc[-1]-sp.iloc[-2])/sp.iloc[-2])*100 if len(sp) >= 2 else 0
            dax_chg = ((dax.iloc[-1]-dax.iloc[-2])/dax.iloc[-2])*100 if len(dax) >= 2 else 0
            nikkei_chg = ((nikkei.iloc[-1]-nikkei.iloc[-2])/nikkei.iloc[-2])*100 if len(nikkei) >= 2 else 0
            tnx_chg = ((tnx.iloc[-1]-tnx.iloc[-2])/tnx.iloc[-2])*100 if len(tnx) >= 2 else 0
            dxy_chg = ((dxy.iloc[-1]-dxy.iloc[-2])/dxy.iloc[-2])*100 if dxy is not None and len(dxy) >= 2 else 0
            gold_chg = ((gold.iloc[-1]-gold.iloc[-2])/gold.iloc[-2])*100 if len(gold) >= 2 else 0
            
            day = datetime.now().day
            passive_on = (1 <= day <= 5) or (15 <= day <= 20)
            
            risk_off = gold_chg > 1.0 and vix.iloc[-1] > 25
            dollar_headwind = dxy_chg > 0.5 if dxy is not None else False
            
            return {
                "sp": sp_chg, 
                "vix": vix.iloc[-1] if len(vix) > 0 else 20,
                "gold": gold.iloc[-1] if len(gold) > 0 else 2000,
                "gold_chg": gold_chg,
                "dax": dax_chg, 
                "nikkei": nikkei_chg, 
                "tnx": tnx.iloc[-1] if len(tnx) > 0 else 4.0,
                "tnx_chg": tnx_chg, 
                "dxy": dxy.iloc[-1] if dxy is not None and len(dxy) > 0 else 100,
                "dxy_chg": dxy_chg,
                "passive": passive_on,
                "risk_off": risk_off,
                "dollar_headwind": dollar_headwind,
                "data_quality": "GOOD"
            }
        except Exception as e:
            return None
    
    def generate_signals(self, data, metrics, ticker):
        signals = {"bullish": [], "bearish": [], "neutral": [], "score": 0}
        
        if metrics['macd'] > metrics['macd_signal']:
            signals['bullish'].append("MACD: Bullish crossover")
            signals['score'] += 1
        elif metrics['macd'] < metrics['macd_signal']:
            signals['bearish'].append("MACD: Bearish crossover")
            signals['score'] -= 1
        
        if metrics['rsi'] < 30:
            signals['bullish'].append("RSI: Oversold (<30)")
            signals['score'] += 1
        elif metrics['rsi'] > 70:
            signals['bearish'].append("RSI: Overbought (>70)")
            signals['score'] -= 1
        
        price = data['Close'].iloc[-1]
        if price < metrics['bb_lower']:
            signals['bullish'].append("BB: Below lower band")
            signals['score'] += 1
        elif price > metrics['bb_upper']:
            signals['bearish'].append("BB: Above upper band")
            signals['score'] -= 1
        
        avg_bb_width = (data['BBU_20_2.0'] - data['BBL_20_2.0']).mean() if 'BBU_20_2.0' in data.columns else 0
        if metrics['bb_width'] < avg_bb_width * 0.7:
            signals['neutral'].append("BB: Squeeze detected")
        
        if metrics['stoch_k'] < 20:
            signals['bullish'].append("Stochastic: Oversold")
            signals['score'] += 1
        elif metrics['stoch_k'] > 80:
            signals['bearish'].append("Stochastic: Overbought")
            signals['score'] -= 1
        
        if metrics['adx'] > 25:
            if metrics['trend_strength'] in ["STRONG_BULL", "BULL"]:
                signals['bullish'].append(f"ADX: Strong uptrend ({metrics['adx']:.1f})")
                signals['score'] += 1
            elif metrics['trend_strength'] in ["STRONG_BEAR", "BEAR"]:
                signals['bearish'].append(f"ADX: Strong downtrend ({metrics['adx']:.1f})")
                signals['score'] -= 1
        else:
            signals['neutral'].append(f"ADX: Weak trend ({metrics['adx']:.1f})")
        
        if metrics['ich_position'] == "ABOVE_CLOUD":
            signals['bullish'].append("Ichimoku: Above cloud")
            signals['score'] += 1
        elif metrics['ich_position'] == "BELOW_CLOUD":
            signals['bearish'].append("Ichimoku: Below cloud")
            signals['score'] -= 1
        else:
            signals['neutral'].append("Ichimoku: Inside cloud")
        
        if metrics['mfi'] < 20:
            signals['bullish'].append("MFI: Money flowing in")
        elif metrics['mfi'] > 80:
            signals['bearish'].append("MFI: Money flowing out")
        
        if metrics['willr'] < -80:
            signals['bullish'].append("Williams %R: Oversold")
        elif metrics['willr'] > -20:
            signals['bearish'].append("Williams %R: Overbought")
        
        sma20 = data['SMA_20'].iloc[-1]
        sma50 = data['SMA_50'].iloc[-1]
        if sma20 > sma50 and data['SMA_20'].iloc[-2] <= data['SMA_50'].iloc[-2]:
            signals['bullish'].append("MA: Golden Cross")
            signals['score'] += 2
        elif sma20 < sma50 and data['SMA_20'].iloc[-2] >= data['SMA_50'].iloc[-2]:
            signals['bearish'].append("MA: Death Cross")
            signals['score'] -= 2
        
        if data['ST_DIR'].iloc[-1] == 1:
            signals['bullish'].append("SuperTrend: Long signal")
            signals['score'] += 1
        else:
            signals['bearish'].append("SuperTrend: Short signal")
            signals['score'] -= 1
        
        for pattern in metrics['patterns']:
            if "Bullish" in pattern or "Hammer" in pattern:
                signals['bullish'].append(f"Pattern: {pattern}")
                signals['score'] += 1
            elif "Bearish" in pattern or "Shooting Star" in pattern:
                signals['bearish'].append(f"Pattern: {pattern}")
                signals['score'] -= 1
            else:
                signals['neutral'].append(f"Pattern: {pattern}")
        
        for div in metrics['divergences']:
            if "Bullish" in div:
                signals['bullish'].append(div)
                signals['score'] += 2
            elif "Bearish" in div:
                signals['bearish'].append(div)
                signals['score'] -= 2
        
        return signals
    
    def detect_correlation_break(self, macro_data):
        us = macro_data['sp']
        eu = macro_data['dax']
        jp = macro_data['nikkei']
        return (us > 1.0 and (eu < -1.0 or jp < -1.0)) or (us < -1.0 and (eu > 1.0 or jp > 1.0))
    
    def check_passive_intensity(self, day, rvol):
        passive_window = (1 <= day <= 5) or (15 <= day <= 20)
        if passive_window:
            if rvol > 1.5: return "STRONG"
            elif rvol > 1.0: return "MODERATE"
            else: return "WEAK"
        return "NEUTRAL"
    
    def calculate_position_size(self, capital, risk_per_trade, risk_distance, method="FIXED", 
                               volatility_mult=1.0, beta=1.0, consecutive_losses=0):
        if risk_distance <= 0:
            return 0
        
        drawdown_mult = 1.0
        if consecutive_losses >= 3:
            drawdown_mult = 0.5
        elif consecutive_losses >= 5:
            drawdown_mult = 0.25
        
        beta_mult = 1.0 / max(beta, 0.5) if beta > 1.5 else 1.0
        
        if method == "FIXED":
            shares = int((risk_per_trade * volatility_mult * drawdown_mult * beta_mult) / risk_distance)
        elif method == "VOLATILITY_ADJUSTED":
            adjusted_risk = risk_per_trade * volatility_mult * drawdown_mult * beta_mult
            shares = int(adjusted_risk / risk_distance)
        elif method == "KELLY":
            win_rate = 0.55
            win_loss_ratio = 2.0
            kelly_fraction = ((win_rate * win_loss_ratio) - (1 - win_rate)) / win_loss_ratio
            kelly_fraction = max(0.1, min(kelly_fraction * 0.5 * drawdown_mult, 0.25))
            adjusted_risk = capital * kelly_fraction
            shares = int(adjusted_risk / risk_distance)
        
        return max(0, shares)
    
    def calculate_sharpe_ratio(self, trades):
        if len(trades) < 5:
            return None
        returns = [t['actual_pnl'] / t['entry'] for t in trades if 'actual_pnl' in t and t['entry'] > 0]
        if not returns:
            return None
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        if std_return == 0:
            return None
        sharpe = (mean_return / std_return) * np.sqrt(252)
        return sharpe
    
    def calculate_expectancy(self, trades):
        if not trades:
            return 0
        total_pnl = sum(t['actual_pnl'] for t in trades if 'actual_pnl' in t)
        return total_pnl / len(trades)
    
    def calculate_profit_factor(self, trades):
        wins = [t['actual_pnl'] for t in trades if 'actual_pnl' in t and t['actual_pnl'] > 0]
        losses = [abs(t['actual_pnl']) for t in trades if 'actual_pnl' in t and t['actual_pnl'] < 0]
        if not losses or sum(losses) == 0:
            return None
        return sum(wins) / sum(losses)
    
    def calculate_max_drawdown(self, trades):
        if not trades:
            return 0
        cumulative = 0
        peak = 0
        max_dd = 0
        for t in trades:
            if 'actual_pnl' in t:
                cumulative += t['actual_pnl']
                if cumulative > peak:
                    peak = cumulative
                dd = peak - cumulative
                if dd > max_dd:
                    max_dd = dd
        return max_dd

engine = InstitutionalAnalyst()

# --- 5. SIDEBAR (WITH V12 HELP NOTES) ---
with st.sidebar:
    st.header("1. Portfolio Settings")
    
    capital = st.number_input("Total Capital ($)", value=10000, min_value=100, 
                             help="Your total account size. Example: $10,000 means you have ten thousand dollars.")
    risk_per_trade = st.number_input("Risk per Trade ($)", value=100, min_value=10, 
                                     help="Maximum $ you're willing to lose on a single trade. Recommended: 1-2% of capital.")
    max_portfolio_risk = st.number_input("Max Portfolio Risk (%)", value=6.0, min_value=1.0, max_value=20.0, step=0.5, 
                                         help="Maximum total risk across ALL open positions combined. Recommended: 5-10%.")
    
    daily_goal = capital * 0.01
    st.caption(f"🎯 Daily Goal (1%): **${daily_goal:.2f}**")
    st.caption("💡 Discipline rule: stop when you hit your daily goal.")
    
    portfolio_risk_pct = (st.session_state.total_risk_deployed / capital) * 100
    if portfolio_risk_pct > max_portfolio_risk:
        st.error(f"⚠️ Portfolio Risk: {portfolio_risk_pct:.1f}% (OVER LIMIT)")
    else:
        st.info(f"📊 Portfolio Risk: {portfolio_risk_pct:.1f}% / {max_portfolio_risk:.1f}%")
    
    pnl_pct = (st.session_state.daily_pnl / capital) * 100
    if st.session_state.goal_met:
        st.success(f"✅ Goal Achieved: +${st.session_state.daily_pnl:.2f} ({pnl_pct:.2f}%)")
    else:
        st.info(f"📈 Session P&L: ${st.session_state.daily_pnl:.2f} ({pnl_pct:.2f}%)")
    
    if st.session_state.consecutive_losses > 0:
        st.warning(f"⚠️ Losing Streak: {st.session_state.consecutive_losses} trades")
        st.caption("💡 After 3 losses, position size automatically reduced by 50%.")

    st.divider()
    st.header("2. Asset Selection")
    
    input_mode = st.radio("Input:", ["VIP List", "Manual Search"], 
                         help="VIP List = Pre-vetted high-liquidity stocks (US + Kenya). Manual = Enter any ticker.")
    
    if input_mode == "VIP List":
        ticker = st.selectbox("Ticker", ALL_TICKERS, help="High-liquidity stocks from US (Nasdaq/NYSE) and Kenya (NSE) exchanges.")
    else:
        ticker = st.text_input("Ticker", "NVDA", help="Enter any stock symbol (e.g., AAPL, TSLA, GME).").upper()

    st.divider()
    st.header("3. Strategy & Execution")
    
    strategy = st.selectbox("Mode", ['Long (Buy)', 'Short (Sell)', 'Income (Puts)'],
                           help="Long: Buy stock (bullish). Short: Sell stock (bearish). Income: Sell put options for premium.")
    
    entry_mode = st.radio("Entry", ["Auto-Limit (Zone)", "Market (Now)", "Manual Override"],
                         help="Auto-Limit: Wait for price to reach zone. Market: Enter immediately. Manual: Test custom price.")
    
    manual_price = 0.0
    if entry_mode == "Manual Override":
        manual_price = st.number_input("Entry Price ($)", value=0.0, step=0.01, 
                                       help="Custom entry price for testing scenarios.")
    
    stop_mode = st.selectbox("Stop Width", [1.0, 0.5, 0.2], 
                            format_func=lambda x: f"Wide ({x} ATR)" if x==1.0 else f"Medium ({x} ATR)" if x==0.5 else f"Tight ({x} ATR)",
                            help="Stop distance in ATR multiples. Wide=swing trades, Tight=day trades.")
    
    position_sizing_method = st.selectbox("Position Sizing", ["FIXED", "VOLATILITY_ADJUSTED", "KELLY"],
                                         help="FIXED: Standard fixed risk. VOLATILITY_ADJUSTED: Scales down in high VIX. KELLY: Mathematical optimal sizing.")
    
    premium = 0.0
    if "Income" in strategy:
        premium = st.number_input("Option Premium ($)", value=0.0, step=0.05, 
                                 help="Per-share premium received for selling put options. Example: $2.50 = $250 per contract (100 shares).")

    st.divider()
    st.header("4. IWT Scorecard")
    st.caption("💡 Hover over (?) for definitions of each element")
    
    if st.session_state.metrics:
        m = st.session_state.metrics
        if "Long" in strategy:
            suggested_fresh = 2 if m.get('support_touches', 0) == 0 else 1 if m.get('support_touches', 0) <= 2 else 0
        else:
            suggested_fresh = 2 if m.get('resistance_touches', 0) == 0 else 1 if m.get('resistance_touches', 0) <= 2 else 0
        st.caption(f"💡 Data suggests: Freshness = {suggested_fresh} ({m.get('support_touches' if 'Long' in strategy else 'resistance_touches', 0)} historical touches)")
    
    fresh = st.selectbox("Freshness", [2, 1, 0], 
                        format_func=lambda x: {2:'2-Fresh', 1:'1-Used', 0:'0-Stale'}[x],
                        help="Fresh: Untested level with strong orders waiting. Used: Touched 1-2x, some orders filled. Stale: Weak, tested 3+ times.")
    
    speed = st.selectbox("Speed Out", [2, 1, 0], 
                        format_func=lambda x: {2:'2-Fast', 1:'1-Avg', 0:'0-Slow'}[x],
                        help="How quickly price left the zone. Fast = strong hands (2+ ATRs in 1-5 candles). Slow = weak interest.")
    
    time_z = st.selectbox("Time in Zone", [2, 1, 0], 
                         format_func=lambda x: {2:'2-Short', 1:'1-Med', 0:'0-Long'}[x],
                         help="How long price lingered in zone. Short (1-2 candles) = strong rejection. Long (6+ candles) = indecision/weakness.")
    
    st.divider()
    if st.button("🔄 Reset Session", help="Clears all positions, P&L, and goal status to start fresh."):
        st.session_state.goal_met = False
        st.session_state.daily_pnl = 0.0
        st.session_state.total_risk_deployed = 0.0
        st.session_state.open_positions = []
        st.session_state.consecutive_losses = 0
        st.success("✅ Session reset!")
        st.rerun()

# --- 6. MAIN UI ---
st.subheader("📊 Market Intelligence Dashboard")

col_macro, col_scan = st.columns([1, 1])

with col_macro:
    if st.button("🌍 Macro Audit", use_container_width=True, 
                help="Scan global markets: VIX, yields, dollar strength, international indices."):
        with st.spinner("Scanning global markets..."):
            st.session_state.macro = engine.get_macro()
            if st.session_state.macro:
                st.success("✅ Macro data loaded")
            else:
                st.error("❌ Macro fetch failed")

with col_scan:
    if st.button(f"🔎 Scan {ticker}", type="primary", use_container_width=True,
                help=f"Load 1 year of data for {ticker} with 15+ technical indicators."):
        with st.spinner(f"Analyzing {ticker}..."):
            df, metrics, fname = engine.fetch_data(ticker)
            
            if df is None:
                st.error(f"🚫 **{ticker}:** {metrics}")
                st.session_state.data = None
            else:
                price = df['Close'].iloc[-1]
                gap_pct = ((df['Open'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2]) * 100 if len(df) >= 2 else 0
                
                st.session_state.data = df
                st.session_state.metrics = {
                    "price": price, 
                    "atr": df['ATRr_14'].iloc[-1],
                    "supp": metrics['support'], 
                    "res": metrics['resistance'],
                    "rvol": df['RVOL'].iloc[-1], 
                    "gap": gap_pct, 
                    "name": fname,
                    **metrics
                }
                
                st.session_state.signals = engine.generate_signals(df, metrics, ticker)
                st.success(f"✅ {fname} loaded successfully")

# MACRO DISPLAY
if st.session_state.macro:
    m = st.session_state.macro
    
    if m.get('data_quality') == 'DEGRADED':
        st.warning("⚠️ Data quality degraded. Using cached/estimated values.")
    
    vix_regime = engine.classify_vix_regime(m['vix'])
    regime_guide = engine.get_regime_guidance(vix_regime)
    
    market_phase, phase_desc = engine.get_market_hours_status()
    
    if vix_regime in ["EXTREME", "HIGH"]:
        st.error(f"🚨 **VIX REGIME: {regime_guide['desc']} ({m['vix']:.1f})**")
        st.caption(f"{regime_guide['action']}")
    elif vix_regime == "ELEVATED":
        st.warning(f"⚠️ **VIX REGIME: {regime_guide['desc']} ({m['vix']:.1f})**")
        st.caption(f"{regime_guide['action']}")
    else:
        st.success(f"✅ **VIX REGIME: {regime_guide['desc']} ({m['vix']:.1f})**")
        st.caption(f"{regime_guide['action']}")
    
    st.caption(f"**Market Hours:** {phase_desc}")
    
    if m.get('risk_off'):
        st.error("🚨 **RISK-OFF REGIME DETECTED:** Gold + VIX both rising = Flight to safety. Avoid aggressive longs.")
    
    if m.get('dollar_headwind') and ticker in COMMODITY_TICKERS:
        st.warning("💵 **DOLLAR HEADWIND:** DXY rising hurts commodities (gold, silver, oil).")
    
    # 🇰🇪 KENYA-SPECIFIC MACRO FILTER
    if ticker.endswith('.NR') and m.get('dxy_chg', 0) > 0.5:
        st.warning("💵 🇰🇪 **KENYA ALERT:** Strong Dollar (DXY rising) typically triggers foreign outflows from NSE. Consider defensive sizing.")
        st.caption("💡 Frontier markets are highly sensitive to USD strength. Watch USD/KES exchange rate closely.")
    
    flow_strength = engine.check_passive_intensity(
        datetime.now().day, 
        st.session_state.metrics.get('rvol', 0) if st.session_state.metrics else 0
    )
    
    if flow_strength == "STRONG":
        st.success("🌊 **STRONG PASSIVE INFLOWS** (Calendar window + High volume)")
        st.caption("💡 $48T in index funds rebalancing. Bullish tailwind.")
    elif flow_strength == "MODERATE":
        st.info("🌊 **MODERATE PASSIVE INFLOWS** (Calendar window + Normal volume)")
    elif flow_strength == "WEAK":
        st.warning("🌊 **WEAK PASSIVE INFLOWS** (Calendar window but Low volume)")
    else:
        st.info("⏸️ **PASSIVE FLOWS NEUTRAL**")
    
    if engine.detect_correlation_break(m):
        st.error("🌍 **GLOBAL CORRELATION BREAK:** US/Europe/Asia markets diverging. Elevated volatility risk.")
    
    with st.expander("🌍 Global Macro Dashboard", expanded=False):
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("🇺🇸 S&P 500", f"{m['sp']:.2f}%", 
                   help="S&P 500 futures. Heartbeat of US markets. Green=risk-on, Red=risk-off.")
        col2.metric("🔥 VIX", f"{m['vix']:.1f}",
                   help="Fear Index. <15=calm, 15-20=normal, 20-30=elevated, >30=crisis mode.")
        col3.metric("🇩🇪 DAX", f"{m['dax']:.2f}%",
                   help="German stock index. Represents European equities. Should correlate with US.")
        col4.metric("🇯🇵 Nikkei", f"{m['nikkei']:.2f}%",
                   help="Japanese stock index. Represents Asian markets. Should correlate with US/EU.")
        col5.metric("💵 DXY", f"{m['dxy']:.1f}", delta=f"{m['dxy_chg']:.2f}%", delta_color="inverse",
                   help="US Dollar Index. UP=Strong dollar=Bad for commodities. DOWN=Weak dollar=Good for gold/oil.")
        col6.metric("📈 10Y Yield", f"{m['tnx']:.2f}%", delta=f"{m['tnx_chg']:.2f}%", delta_color="inverse",
                   help="10-Year Treasury yield. UP=Rising rates=Bad for growth stocks. DOWN=Falling rates=Good for growth.")

# ASSET ANALYSIS
if st.session_state.data is not None:
    m = st.session_state.metrics
    df = st.session_state.data
    
    st.divider()
    st.header(f"📈 {m['name']} ({ticker})")
    st.caption("💡 Key metrics showing current state of the asset")
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    col1.metric("Price", f"${m['price']:.2f}", help="Current closing price")
    
    if abs(m['gap']) > 2.0:
        if m['rvol'] > 1.5:
            col2.metric("Gap %", f"{m['gap']:.2f}%", delta="🚀 PRO", 
                       help="PROFESSIONAL GAP: Large gap (>2%) + High volume = Institutional breakout. Gap likely to HOLD.")
        else:
            col2.metric("Gap %", f"{m['gap']:.2f}%", delta="⚠️ NOVICE", 
                       help="NOVICE GAP: Large gap but low volume = Retail FOMO. Gap likely to FILL.")
    else:
        col2.metric("Gap %", f"{m['gap']:.2f}%", help="Gap = (Today's Open - Yesterday's Close) / Yesterday's Close")
    
    vol_status = "🔥 HOT" if m['rvol'] > 1.5 else "✅ NORMAL" if m['rvol'] > 0.8 else "💤 THIN"
    col3.metric("Volume (RVOL)", f"{m['rvol']:.1f}x", delta=vol_status, 
               help="Relative Volume. >1.5x = HIGH interest. <0.8x = LOW interest (thin/dangerous).")
    
    # 🇰🇪 DATA QUALITY WARNING FOR KENYA
    if ticker.endswith('.NR') and m['rvol'] < 0.5:
        st.caption("⚠️ 🇰🇪 **Kenya Data Note:** Yahoo Finance volume for NSE can be delayed. If showing near-zero, ignore RVOL signal.")
    
    trend_emoji = {"STRONG_BULL": "🚀", "BULL": "📈", "NEUTRAL": "➡️", "BEAR": "📉", "STRONG_BEAR": "🔻"}
    col4.metric("Trend", m['trend_strength'], delta=trend_emoji.get(m['trend_strength'], "➡️"), 
               help="Multi-timeframe trend strength (20/50/200 SMAs). Trade WITH the trend.")
    
    rsi_status = "⚠️OB" if m['rsi'] > 70 else "⚠️OS" if m['rsi'] < 30 else "✅"
    col5.metric("RSI", f"{m['rsi']:.0f}", delta=rsi_status, 
               help="Relative Strength Index. >70 = Overbought. <30 = Oversold. 50 = Neutral.")
    
    col6.metric("ADX (Trend Strength)", f"{m['adx']:.0f}", delta="STRONG" if m['adx'] > 25 else "WEAK", 
               help="Average Directional Index. >25 = Strong trend. <20 = Weak/choppy.")
    
    st.caption(f"**Key Levels:** Support ${m['supp']:.2f} ({m.get('support_touches', 0)} touches) | Resistance ${m['res']:.2f} ({m.get('resistance_touches', 0)} touches)")
    
    # SIGNALS
    if st.session_state.signals:
        st.divider()
        st.subheader("🎯 Multi-Algorithm Signal Fusion (15+ Indicators)")
        st.caption("💡 Combines RSI, MACD, Bollinger Bands, Stochastic, ADX, Ichimoku, MFI, Williams %R, Moving Averages, SuperTrend, Patterns, Divergences")
        
        sig = st.session_state.signals
        
        col_bull, col_bear, col_neut = st.columns(3)
        
        with col_bull:
            st.markdown("### 🟢 Bullish Signals")
            if sig['bullish']:
                for s in sig['bullish']:
                    st.markdown(f"<div class='signal-bull'>• {s}</div>", unsafe_allow_html=True)
            else:
                st.caption("No bullish signals detected")
        
        with col_bear:
            st.markdown("### 🔴 Bearish Signals")
            if sig['bearish']:
                for s in sig['bearish']:
                    st.markdown(f"<div class='signal-bear'>• {s}</div>", unsafe_allow_html=True)
            else:
                st.caption("No bearish signals detected")
        
        with col_neut:
            st.markdown("### 🟡 Neutral/Watch Signals")
            if sig['neutral']:
                for s in sig['neutral']:
                    st.markdown(f"<div class='signal-neutral'>• {s}</div>", unsafe_allow_html=True)
            else:
                st.caption("No neutral signals")
        
        score = sig['score']
        if score >= 3:
            st.success(f"### 🟢 OVERALL MULTI-ALGO VERDICT: BULLISH (+{score} points)")
            st.caption("💡 Most indicators agree this is a BUY setup.")
        elif score <= -3:
            st.error(f"### 🔴 OVERALL MULTI-ALGO VERDICT: BEARISH ({score} points)")
            st.caption("💡 Most indicators agree this is a SELL/SHORT setup.")
        else:
            st.warning(f"### 🟡 OVERALL MULTI-ALGO VERDICT: NEUTRAL ({score} points)")
            st.caption("💡 Indicators are mixed. No clear direction.")
    
    # CHART
    st.divider()
    st.subheader("📊 Technical Chart (Last 60 Days)")
    st.caption("💡 Blue=SMA20, Orange=SMA50, Red=SMA200, Gray=Bollinger Bands, Green/Red lines=Support/Resistance")
    
    try:
        chart_data = df.iloc[-60:]
        
        addplots = []
        
        if 'SMA_20' in chart_data.columns:
            addplots.append(mpf.make_addplot(chart_data['SMA_20'], color='blue', width=1.5))
        
        if 'SMA_50' in chart_data.columns:
            addplots.append(mpf.make_addplot(chart_data['SMA_50'], color='orange', width=1.5))
        
        if 'SMA_200' in chart_data.columns:
            addplots.append(mpf.make_addplot(chart_data['SMA_200'], color='red', width=2))
        
        bb_cols = [col for col in chart_data.columns if 'BB' in col]
        bb_upper = None
        bb_lower = None
        
        for col in bb_cols:
            if 'U' in col or 'upper' in col.lower():
                bb_upper = col
            elif 'L' in col or 'lower' in col.lower():
                bb_lower = col
        
        if bb_upper and bb_upper in chart_data.columns:
            addplots.append(mpf.make_addplot(chart_data[bb_upper], color='gray', linestyle='--', width=1))
        
        if bb_lower and bb_lower in chart_data.columns:
            addplots.append(mpf.make_addplot(chart_data[bb_lower], color='gray', linestyle='--', width=1))
        
        fig, axes = mpf.plot(
            chart_data, 
            type='candle', 
            style='yahoo', 
            volume=True,
            addplot=addplots if addplots else None,
            hlines=dict(hlines=[m['supp'], m['res']], colors=['green', 'red'], linestyle='-.', linewidths=2),
            returnfig=True, 
            figsize=(14, 8), 
            title=f"{ticker} - Multi-Indicator Technical Analysis"
        )
        
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"⚠️ Chart rendering error: {str(e)}")
        st.caption("💡 Try AAPL or NVDA (most reliable data).")
    
    with st.expander("📐 Fibonacci Retracement Levels", expanded=False):
        st.caption("💡 Mathematical support/resistance based on golden ratio")
        fib = m['fib_levels']
        for level, price in fib.items():
            st.caption(f"{level}: ${price:.2f}")
    
    # TRADE CALCULATION
    st.divider()
    st.subheader("🎯 Trade Setup Calculator")
    st.caption("💡 Calculates entry, stop, target, position size, and REAL costs")
    
    if entry_mode == "Manual Override":
        entry = manual_price
    elif "Short" in strategy:
        entry = m['res'] if "Auto" in entry_mode else m['price']
    else:
        entry = m['supp'] if "Auto" in entry_mode else m['price']
    
    if st.session_state.macro:
        vix_regime = engine.classify_vix_regime(st.session_state.macro['vix'])
        regime_guide = engine.get_regime_guidance(vix_regime)
        vol_multiplier = regime_guide['size_multiplier']
        stop_multiplier = regime_guide['stop_multiplier']
    else:
        vol_multiplier = 1.0
        stop_multiplier = 1.0
    
    # === INCOME (PUTS) CALCULATION FIX ===
    if "Income" in strategy:
        # For selling puts:
        # - Entry = Strike price (price you'd buy stock at if assigned)
        # - Stop = Technical stop below entry (or entry - 2*ATR as safety)
        # - Target = Entry (stock stays above strike, you keep premium)
        # - Risk per contract = (Strike × 100) - (Premium × 100) = max capital at risk
        # - Reward per contract = Premium × 100
        
        stop = entry - (m['atr'] * 2)  # Technical stop 2 ATRs below strike
        target = entry  # Keep premium if stock stays above strike
        
        # Risk per contract (if assigned and stock drops to stop)
        risk_per_contract = (entry - stop) * 100  # 100 shares per contract
        
        # Reward per contract (premium collected)
        reward_per_contract = premium * 100
        
        # Calculate number of contracts based on capital and risk
        if risk_per_contract > 0:
            contracts = int(risk_per_trade / risk_per_contract)
        else:
            contracts = 0
        
        shares = max(0, contracts)  # "shares" = contracts for Income strategy
        
        risk = risk_per_contract
        reward = reward_per_contract
        
    elif "Short" in strategy:
        stop = entry + (m['atr'] * stop_mode * stop_multiplier)
        target = m['supp']
        risk = stop - entry
        reward = entry - target
    else:  # Long
        stop = entry - (m['atr'] * stop_mode * stop_multiplier)
        target = m['res']
        risk = entry - stop
        reward = target - entry
    
    rr = reward / risk if risk > 0 else 0
    
    # Position sizing (skip for Income since already calculated)
    if "Income" not in strategy:
        shares = engine.calculate_position_size(
            capital, risk_per_trade, risk, position_sizing_method, 
            vol_multiplier, m.get('beta', 1.0), st.session_state.consecutive_losses
        )
    
    total_trade_risk = shares * risk if shares > 0 else 0
    
    # Costs & Net Reward
    if "Income" in strategy:
        slippage = 0  # Options have bid-ask spread, but we'll keep calculation simple
        commissions = shares * 0.65  # Typical options commission per contract
        gross_reward = shares * reward
    else:
        slippage = entry * (SLIPPAGE_BPS / 10000) * shares
        commissions = shares * COMMISSION_PER_SHARE * 2
        gross_reward = shares * reward
    
    net_reward = gross_reward - slippage - commissions
    
    col_setup1, col_setup2, col_setup3 = st.columns(3)
    
    with col_setup1:
        st.markdown("**📍 Price Levels**")
        st.code(f"""
Entry:  ${entry:.2f}
Stop:   ${stop:.2f}
Target: ${target:.2f}
        """)
        st.caption("💡 Entry=where you buy/sell. Stop=exit if wrong. Target=exit if right.")
    
    with col_setup2:
        st.markdown("**💰 Position Sizing**")
        qty_label = "Contracts" if "Income" in strategy else "Shares"
        st.code(f"""
Size:    {shares} {qty_label}
Risk:    ${total_trade_risk:.2f}
Reward:  ${gross_reward:.2f}
R/R:     {rr:.2f}
        """)
        if "Income" in strategy:
            st.caption(f"💡 Each contract = 100 shares. Premium = ${premium:.2f}/share = ${premium*100:.2f}/contract")
        else:
            st.caption(f"💡 Size calculated using {position_sizing_method}. Risk = max loss if stopped out.")
    
    with col_setup3:
        st.markdown("**💸 Real Costs**")
        st.code(f"""
Slippage:     ${slippage:.2f}
Commissions:  ${commissions:.2f}
Net Reward:   ${net_reward:.2f}
Net R/R:      {(net_reward/(total_trade_risk if total_trade_risk>0 else 1)):.2f}
        """)
        st.caption("💡 Real costs reduce your profit. This is why high-frequency trading is hard.")
    
    # VERDICT
    st.divider()
    st.subheader("🚦 The Ultimate Verdict (IWT + Institutional Filters)")
    st.caption("💡 Combines IWT score with 13 institutional penalty filters")
    
    score_rr = 2 if rr >= 3 or ("Income" in strategy and rr > 0.1) else 1 if rr >= 2 else 0
    total_score = fresh + time_z + speed + score_rr
    
    penalties = []
    warsh_penalty = False
    
    if st.session_state.macro and st.session_state.macro['tnx_chg'] > 1.0 and ticker in GROWTH_TICKERS and "Long" in strategy:
        total_score -= 2
        warsh_penalty = True
        penalties.append("Warsh (-2): Yields rising >1% hurt growth stocks")
    
    market_phase, _ = engine.get_market_hours_status()
    if market_phase in ["LUNCH", "PRE_MARKET", "AFTER_HOURS"]:
        total_score -= 1
        penalties.append(f"Market Hours (-1): {market_phase} (low liquidity)")
    
    if "Long" in strategy and m['trend_strength'] in ["BEAR", "STRONG_BEAR"]:
        total_score -= 1
        penalties.append("Trend Misalignment (-1): Long in downtrend")
    elif "Short" in strategy and m['trend_strength'] in ["BULL", "STRONG_BULL"]:
        total_score -= 1
        penalties.append("Trend Misalignment (-1): Short in uptrend")
    
    sector = SECTOR_MAP.get(ticker, "Unknown")
    sector_exposure = sum(1 for p in st.session_state.open_positions if SECTOR_MAP.get(p['ticker'], '') == sector)
    if sector_exposure >= 2:
        total_score -= 1
        penalties.append(f"Concentration Risk (-1): {sector_exposure+1} positions in {sector}")
    
    if st.session_state.macro and st.session_state.macro.get('risk_off') and "Long" in strategy:
        total_score -= 1
        penalties.append("Risk-Off (-1): Gold+VIX rising")
    
    if st.session_state.macro and st.session_state.macro.get('dollar_headwind') and ticker in COMMODITY_TICKERS and "Long" in strategy:
        total_score -= 1
        penalties.append("Dollar Headwind (-1): DXY rising hurts commodities")
    
    if 'ST_DIR' in df.columns:
        if "Long" in strategy and df['ST_DIR'].iloc[-1] == -1:
            total_score -= 1
            penalties.append("SuperTrend Conflict (-1): Indicator is bearish")
        elif "Short" in strategy and df['ST_DIR'].iloc[-1] == 1:
            total_score -= 1
            penalties.append("SuperTrend Conflict (-1): Indicator is bullish")
    
    if "Long" in strategy and m['rsi'] > 70:
        total_score -= 1
        penalties.append("RSI Extreme (-1): Overbought (>70)")
    elif "Short" in strategy and m['rsi'] < 30:
        total_score -= 1
        penalties.append("RSI Extreme (-1): Oversold (<30)")
    
    if st.session_state.signals:
        sig_score = st.session_state.signals['score']
        if "Long" in strategy and sig_score < -2:
            total_score -= 1
            penalties.append("Multi-Algo Conflict (-1): Signals overwhelmingly bearish")
        elif "Short" in strategy and sig_score > 2:
            total_score -= 1
            penalties.append("Multi-Algo Conflict (-1): Signals overwhelmingly bullish")
    
    col_verdict, col_analysis = st.columns([1, 1])
    
    with col_verdict:
        if st.session_state.goal_met:
            st.error("## 🛑 DAILY GOAL MET - STOP TRADING")
            st.markdown("<div class='risk-warning'><strong>CLOSE YOUR TERMINAL.</strong> Protect your gains. Consistency beats intensity.</div>", unsafe_allow_html=True)
            can_trade = False
        elif (st.session_state.total_risk_deployed + total_trade_risk) > (capital * max_portfolio_risk / 100):
            st.error("## 🛑 PORTFOLIO RISK LIMIT EXCEEDED")
            st.markdown(f"<div class='risk-warning'>Adding this trade would exceed your {max_portfolio_risk}% limit.</div>", unsafe_allow_html=True)
            can_trade = False
        else:
            can_trade = True
            if total_score >= 7:
                st.success(f"## 🟢 GREEN LIGHT\n**Final Score: {total_score}/8**")
                st.caption("✅ **Action:** Execute with FULL confidence. All systems GO.")
            elif total_score >= 5:
                st.warning(f"## 🟡 YELLOW LIGHT\n**Final Score: {total_score}/8**")
                st.caption("⚠️ **Action:** Tradeable but NOT ideal. Reduce size 50% OR wait.")
            else:
                st.error(f"## 🔴 RED LIGHT\n**Final Score: {total_score}/8**")
                st.caption("🛑 **Action:** DO NOT TRADE. Setup is flawed.")
        
        if penalties:
            st.markdown("**⚠️ Penalties Applied:**")
            for p in penalties:
                st.caption(f"• {p}")
    
    with col_analysis:
        st.markdown("**📋 Setup Quality Checklist**")
        
        checks = []
        checks.append(("✅" if fresh == 2 else "⚠️" if fresh == 1 else "❌", f"Freshness: {['Stale (Weak)','Used (OK)','Fresh (Strong)'][fresh]}"))
        checks.append(("✅" if score_rr == 2 else "⚠️" if score_rr == 1 else "❌", f"R/R: {rr:.2f} ({['Poor (<2)', 'Acceptable (2-3)', 'Excellent (3+)'][score_rr]})"))
        checks.append(("✅" if abs(m['gap']) > 2 and m['rvol'] > 1.5 else "⚠️" if abs(m['gap']) > 2 else "➖", f"Gap: {m['gap']:.2f}%"))
        checks.append(("✅" if m['rvol'] > 1.2 else "⚠️", f"Volume: {m['rvol']:.1f}x"))
        checks.append(("✅" if m['adx'] > 25 else "⚠️", f"Trend Strength: ADX {m['adx']:.0f}"))
        
        for icon, text in checks:
            st.caption(f"{icon} {text}")
    
    # === WARREN AI EXPORT ===
    st.markdown("---")
    st.caption("**📋 Copy for WarrenAI:**")
    
    if st.session_state.macro:
        flow_strength = engine.check_passive_intensity(
            datetime.now().day,
            st.session_state.metrics.get('rvol', 0)
        )
    else:
        flow_strength = "UNKNOWN"
    
    ai_export = f"""
[ARCHITECT REVIEW - {ticker}]
Strategy: {strategy}
Score: {total_score}/8
Verdict: {'GREEN' if total_score>=7 else 'YELLOW' if total_score>=5 else 'RED'}
Entry: ${entry:.2f} | Stop: ${stop:.2f} | Target: ${target:.2f}
R/R: {rr:.2f} | Size: {shares} {'contracts' if 'Income' in strategy else 'shares'}
VIX Regime: {vix_regime if st.session_state.macro else 'N/A'}
Passive Flow: {flow_strength}
10Y Yield: {'RISING >1%' if warsh_penalty else 'STABLE'}
    """
    
    st.code(ai_export.strip(), language='text')
    
    # EXECUTION
    if can_trade and total_score >= 5:
        st.divider()
        st.subheader("⚡ Trade Execution")
        st.caption("💡 PAPER = practice (no P&L). LIVE = real trade (affects P&L).")
        
        col_exec1, col_exec2 = st.columns(2)
        
        with col_exec1:
            if st.button("📝 Log as PAPER TRADE", type="secondary", use_container_width=True,
                        help="Logs for learning. Does NOT affect P&L or portfolio risk."):
                trade_record = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ticker": ticker, "action": strategy, "entry": entry, "stop": stop, "target": target,
                    "shares": shares, "score": total_score, "risk": total_trade_risk,
                    "expected_reward": gross_reward, "net_reward": net_reward, "rr_ratio": rr,
                    "slippage": slippage, "commissions": commissions, "status": "PAPER"
                }
                st.session_state.journal.append(trade_record)
                st.success("📋 Paper trade logged!")
        
        with col_exec2:
            if st.button("💵 Log as LIVE TRADE", type="primary", use_container_width=True,
                        help="Logs as open position (affects portfolio risk). Only if you actually entered."):
                trade_record = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ticker": ticker, "action": strategy, "entry": entry, "stop": stop, "target": target,
                    "shares": shares, "score": total_score, "risk": total_trade_risk,
                    "expected_reward": gross_reward, "net_reward": net_reward, "rr_ratio": rr,
                    "slippage": slippage, "commissions": commissions, "status": "OPEN"
                }
                st.session_state.journal.append(trade_record)
                st.session_state.open_positions.append(trade_record)
                st.session_state.total_risk_deployed += total_trade_risk
                st.success("✅ Live position logged!")
                st.rerun()

else:
    st.info("👈 **Quick Start:** 1. Scan Macro → 2. Scan Ticker → 3. Review Signals → 4. Check Verdict → 5. Execute")

# POSITION MANAGEMENT
if st.session_state.open_positions:
    st.divider()
    st.subheader("📊 Open Positions (Live Trades)")
    st.caption("💡 These are trades you logged as LIVE. Close them after you exit in your broker.")
    
    positions_df = pd.DataFrame(st.session_state.open_positions)
    positions_df = positions_df[['ticker', 'action', 'entry', 'stop', 'target', 'shares', 'risk', 'score']]
    st.dataframe(positions_df, use_container_width=True)
    
    st.markdown("**Close a Position:**")
    st.caption("💡 Enter the actual exit price from your broker. System calculates real P&L.")
    
    col_close1, col_close2, col_close3 = st.columns(3)
    
    with col_close1:
        position_to_close = st.selectbox("Select Position", [p['ticker'] for p in st.session_state.open_positions])
    
    with col_close2:
        exit_price = st.number_input("Exit Price ($)", value=0.0, step=0.01)
    
    with col_close3:
        if st.button("✅ CLOSE POSITION"):
            if exit_price > 0:
                for i, pos in enumerate(st.session_state.open_positions):
                    if pos['ticker'] == position_to_close:
                        if "Long" in pos['action']:
                            actual_pnl = (exit_price - pos['entry']) * pos['shares']
                        elif "Income" in pos['action']:
                            # For Income: PnL = Premium kept (if expired worthless) OR loss if assigned
                            # Simplified: If closed early, profit/loss based on option price movement
                            actual_pnl = pos['expected_reward']  # Assume full premium kept for simplicity
                        else:  # Short
                            actual_pnl = (pos['entry'] - exit_price) * pos['shares']
                        
                        actual_pnl -= (pos['slippage'] + pos['commissions'])
                        
                        pos['exit_price'] = exit_price
                        pos['actual_pnl'] = actual_pnl
                        pos['status'] = 'CLOSED'
                        
                        st.session_state.closed_trades.append(pos)
                        st.session_state.daily_pnl += actual_pnl
                        st.session_state.total_risk_deployed -= pos['risk']
                        
                        if actual_pnl < 0:
                            st.session_state.consecutive_losses += 1
                        else:
                            st.session_state.consecutive_losses = 0
                        
                        if st.session_state.daily_pnl >= daily_goal:
                            st.session_state.goal_met = True
                            st.balloons()
                        
                        st.session_state.open_positions.pop(i)
                        st.success(f"✅ Closed {position_to_close}: P&L = ${actual_pnl:.2f}")
                        st.rerun()
                        break
            else:
                st.error("❌ Please enter a valid exit price.")

# PERFORMANCE ANALYTICS
if st.session_state.closed_trades:
    st.divider()
    st.subheader("📈 Performance Analytics Dashboard")
    st.caption("💡 These metrics show how good your trading system is")
    
    closed_df = pd.DataFrame(st.session_state.closed_trades)
    
    col_stats1, col_stats2, col_stats3, col_stats4, col_stats5 = st.columns(5)
    
    wins = len(closed_df[closed_df['actual_pnl'] > 0])
    total_trades = len(closed_df)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    col_stats1.metric("Win Rate", f"{win_rate:.1f}%", delta=f"{wins}/{total_trades} trades",
                     help=">50% is good with 2:1 R/R. >60% is excellent.")
    
    avg_rr = closed_df['rr_ratio'].mean()
    col_stats2.metric("Avg R/R", f"{avg_rr:.2f}",
                     help="Average Risk/Reward. >2.0 means you make 2x what you risk.")
    
    total_pnl = closed_df['actual_pnl'].sum()
    col_stats3.metric("Total P&L", f"${total_pnl:.2f}", delta=f"{(total_pnl/capital)*100:.2f}%",
                     help="Total profit/loss. % shows return on capital.")
    
    sharpe = engine.calculate_sharpe_ratio(st.session_state.closed_trades)
    if sharpe:
        sharpe_quality = "Excellent" if sharpe > 2 else "Good" if sharpe > 1 else "Poor"
        col_stats4.metric("Sharpe Ratio", f"{sharpe:.2f}", delta=sharpe_quality,
                         help="Risk-adjusted return. >1 is good, >2 is excellent.")
    else:
        col_stats4.metric("Sharpe Ratio", "N/A", help="Need at least 5 closed trades.")
    
    profit_factor = engine.calculate_profit_factor(st.session_state.closed_trades)
    if profit_factor:
        pf_quality = "Excellent" if profit_factor > 2 else "Good" if profit_factor > 1.5 else "Poor"
        col_stats5.metric("Profit Factor", f"{profit_factor:.2f}", delta=pf_quality,
                         help="Gross Profit / Gross Loss. >1.5 = profitable.")
    else:
        col_stats5.metric("Profit Factor", "N/A")
    
    col_stats6, col_stats7, col_stats8 = st.columns(3)
    
    expectancy = engine.calculate_expectancy(st.session_state.closed_trades)
    col_stats6.metric("Expectancy", f"${expectancy:.2f}",
                     help="Average $/trade. Must be positive to be profitable long-term.")
    
    max_dd = engine.calculate_max_drawdown(st.session_state.closed_trades)
    col_stats7.metric("Max Drawdown", f"${max_dd:.2f}",
                     help="Largest peak-to-trough decline. Keep under 20% of capital.")
    
    consecutive_wins = 0
    consecutive_losses = 0
    max_consec_wins = 0
    max_consec_losses = 0
    
    for t in st.session_state.closed_trades:
        if t['actual_pnl'] > 0:
            consecutive_wins += 1
            consecutive_losses = 0
            max_consec_wins = max(max_consec_wins, consecutive_wins)
        else:
            consecutive_losses += 1
            consecutive_wins = 0
            max_consec_losses = max(max_consec_losses, consecutive_losses)
    
    col_stats8.metric("Max Streak", f"W:{max_consec_wins} / L:{max_consec_losses}",
                     help="Longest winning/losing streaks.")
    
    with st.expander("📋 Complete Trade History", expanded=False):
        history_df = closed_df[['timestamp', 'ticker', 'action', 'entry', 'exit_price', 'actual_pnl', 'score', 'slippage', 'commissions']]
        st.dataframe(history_df, use_container_width=True)

# JOURNAL EXPORT
if st.session_state.journal:
    st.divider()
    st.subheader("📓 Trading Journal (All Trades)")
    st.caption("💡 Contains BOTH paper and live trades. Review regularly to find patterns.")
    
    journal_df = pd.DataFrame(st.session_state.journal)
    st.dataframe(journal_df, use_container_width=True)
    
    col_export1, col_export2 = st.columns(2)
    
    with col_export1:
        csv = journal_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Full Journal (CSV)",
            data=csv,
            file_name=f"journal_{datetime.now().strftime('%Y%m%d')}.csv",
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

st.divider()
st.caption("🏛️ Quantum Maestro [TradingBot] — Ultimate Institutional Edition | Educational Use Only")
st.caption("© 2026 Gabriel Mahia | Consistency beats intensity.")
