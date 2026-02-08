# Copyright (c) 2024 Gabriel Mahia. All Rights Reserved.
# Unauthorized copying of this file, via any medium, is strictly prohibited.
# Proprietary and confidential.
# Written by Gabriel Mahia, 2026
# app.py - ULTIMATE EDITION
import streamlit as st
import yfinance as yf
import pandas_ta as ta
import mplfinance as mpf
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
from datetime import datetime, time, timedelta
import pytz
from collections import deque

# --- 1. CONFIGURATION ---
VIP_TICKERS = ["NVDA", "AAPL", "AMZN", "GOOGL", "TSLA", "MSFT", "META", "AMD", "NFLX", "SPY", "QQQ", "IWM", "GLD", "SLV", "USO"]
GROWTH_TICKERS = ["NVDA", "AAPL", "AMZN", "GOOGL", "TSLA", "MSFT", "META", "AMD", "NFLX", "QQQ", "ARKK", "COIN", "SHOP", "SQ"]
COMMODITY_TICKERS = ["GLD", "SLV", "GDX", "USO", "XLE", "FCX"]
VALUE_TICKERS = ["JPM", "BAC", "XOM", "CVX", "BRK.B", "JNJ", "PG"]

SECTOR_MAP = {
    "NVDA": "Tech", "AMD": "Tech", "MSFT": "Tech", "AAPL": "Tech", "META": "Tech", "GOOGL": "Tech",
    "TSLA": "Auto", "AMZN": "Consumer", "NFLX": "Media", "SPY": "Index", "QQQ": "Tech-Index", "IWM": "Index",
    "GLD": "Commodity", "SLV": "Commodity", "GDX": "Mining", "USO": "Energy", "XLE": "Energy",
    "JPM": "Finance", "BAC": "Finance", "XOM": "Energy", "CVX": "Energy"
}

# Trading costs (realistic assumptions)
COMMISSION_PER_SHARE = 0.005  # $0.005/share (Interactive Brokers)
SLIPPAGE_BPS = 5  # 5 basis points (0.05%)

st.set_page_config(page_title="Quantum Maestro Ultimate", layout="wide", initial_sidebar_state="expanded", page_icon="🏛️")

st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 4px; height: 3em; font-weight: 600; }
    div[data-testid="stMetric"] { background-color: #f0f2f6; border: 1px solid #d6d6d6; border-radius: 6px; padding: 10px 15px; }
    @media (prefers-color-scheme: dark) {
        div[data-testid="stMetric"] { background-color: #1e2127; border: 1px solid #30333d; }
    }
    .risk-warning { background-color: #fff3cd; padding: 15px; border-left: 4px solid #ffc107; margin: 10px 0; }
    .success-box { background-color: #d4edda; padding: 15px; border-left: 4px solid #28a745; margin: 10px 0; }
    .signal-bull { background-color: #d4edda; padding: 8px; border-radius: 4px; margin: 5px 0; }
    .signal-bear { background-color: #f8d7da; padding: 8px; border-radius: 4px; margin: 5px 0; }
    .signal-neutral { background-color: #fff3cd; padding: 8px; border-radius: 4px; margin: 5px 0; }
</style>
""", unsafe_allow_html=True)

# --- 2. LEGAL ---
st.title("🏛️ Quantum Maestro [TradingBot]: Institutional Edition")
st.caption("Multi-Algorithm Fusion | Pattern Recognition | Adaptive Risk | Performance Analytics")

with st.expander("⚠️ LEGAL DISCLAIMER", expanded=True):
    st.markdown("""
    **1. Not Financial Advice:** Educational tool only. No affiliation with Trade and Travel.
    **2. Risk Warning:** Trading involves substantial risk of loss. Past performance ≠ future results.
    **3. Data Disclaimer:** Market data via Yahoo Finance. Delays/inaccuracies may occur.
    **4. Simulated Results:** Backtests/projections are hypothetical. Real trading differs.
    """)
    agree = st.checkbox("I accept this is educational only and involves real financial risk.")

if not agree:
    st.warning("🛑 Accept disclaimer to continue.")
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
if 'alert_thresholds' not in st.session_state: st.session_state.alert_thresholds = {}
if 'consecutive_losses' not in st.session_state: st.session_state.consecutive_losses = 0

# --- 4. ULTIMATE ANALYST ENGINE ---
class UltimateAnalyst:
    
    def __init__(self):
        self.vix_regimes = {
            "EXTREME_LOW": (0, 12), "LOW": (12, 15), "NORMAL": (15, 20),
            "ELEVATED": (20, 30), "HIGH": (30, 40), "EXTREME": (40, 100)
        }
        self.fib_levels = [0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    
    # ========== MARKET CONTEXT ==========
    def get_market_hours_status(self):
        try:
            et = pytz.timezone('US/Eastern')
            now = datetime.now(et)
            current_time = now.time()
            
            if current_time < time(9, 30):
                return "PRE_MARKET", "⏰ Pre-Market"
            elif current_time < time(10, 0):
                return "OPENING", "🔔 Opening Range"
            elif current_time < time(12, 0):
                return "MORNING", "☀️ Morning Session"
            elif current_time < time(14, 0):
                return "LUNCH", "🍴 Lunch Hour"
            elif current_time < time(15, 0):
                return "AFTERNOON", "🌤️ Afternoon"
            elif current_time < time(16, 0):
                return "POWER_HOUR", "⚡ Power Hour"
            else:
                return "AFTER_HOURS", "🌙 After Hours"
        except:
            return "UNKNOWN", "⚠️ Unknown"
    
    def classify_vix_regime(self, vix_level):
        for regime, (low, high) in self.vix_regimes.items():
            if low <= vix_level < high:
                return regime
        return "EXTREME"
    
    def get_regime_guidance(self, regime):
        guidance = {
            "EXTREME_LOW": {"desc": "Complacency", "size_mult": 0.7, "stop_mult": 1.2},
            "LOW": {"desc": "Calm", "size_mult": 1.0, "stop_mult": 1.0},
            "NORMAL": {"desc": "Healthy", "size_mult": 1.0, "stop_mult": 1.0},
            "ELEVATED": {"desc": "Uncertainty", "size_mult": 0.7, "stop_mult": 1.3},
            "HIGH": {"desc": "Crisis", "size_mult": 0.5, "stop_mult": 1.5},
            "EXTREME": {"desc": "Panic", "size_mult": 0.3, "stop_mult": 2.0}
        }
        return guidance.get(regime, guidance["NORMAL"])
    
    # ========== DATA FETCHING WITH FULL INDICATORS ==========
    def fetch_data(self, t):
        try:
            ticker_obj = yf.Ticker(t)
            data = ticker_obj.history(period="1y")
            if data.empty or len(data) < 50:
                return None, None, f"ERROR: Insufficient data"
            
            try: 
                full_name = ticker_obj.info.get('longName', t)
                beta = ticker_obj.info.get('beta', 1.0)
            except: 
                full_name = t
                beta = 1.0
            
            # ===== CORE INDICATORS =====
            data.ta.atr(length=14, append=True)
            data.ta.rsi(length=14, append=True)
            data.ta.macd(fast=12, slow=26, signal=9, append=True)
            data.ta.bbands(length=20, std=2, append=True)
            data.ta.stoch(k=14, d=3, append=True)
            data.ta.adx(length=14, append=True)
            data.ta.obv(append=True)
            data.ta.mfi(length=14, append=True)
            data.ta.willr(length=14, append=True)
            
            # SMAs
            data.ta.sma(length=20, append=True)
            data.ta.sma(length=50, append=True)
            data.ta.sma(length=200, append=True)
            data.ta.ema(length=12, append=True)
            data.ta.ema(length=26, append=True)
            
            # SuperTrend
            try:
                st_data = data.ta.supertrend(length=10, multiplier=3)
                data['ST_VAL'] = st_data.iloc[:, 0]
                data['ST_DIR'] = st_data.iloc[:, 1]
            except:
                data['ST_VAL'] = data['Close']
                data['ST_DIR'] = 1
            
            # Ichimoku Cloud
            try:
                ich = data.ta.ichimoku()[0]
                data['ICH_SPAN_A'] = ich['ISA_9']
                data['ICH_SPAN_B'] = ich['ISB_26']
                data['ICH_BASE'] = ich['ITS_9']
            except:
                data['ICH_SPAN_A'] = data['Close']
                data['ICH_SPAN_B'] = data['Close']
            
            # Volume
            vol_sma = data['Volume'].rolling(20).mean()
            data['RVOL'] = data['Volume'] / vol_sma
            
            # ===== SUPPORT/RESISTANCE WITH FIBONACCI =====
            swing_high = data['High'].rolling(20).max()
            swing_low = data['Low'].rolling(20).min()
            
            recent_high = swing_high.iloc[-1]
            recent_low = swing_low.iloc[-1]
            price_range = recent_high - recent_low
            
            fib_levels = {}
            for level in self.fib_levels:
                fib_levels[f"fib_{level}"] = recent_high - (price_range * level)
            
            # Mathematical support/resistance
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
            
            # Count zone touches (for auto-freshness scoring)
            support_touches = 0
            resistance_touches = 0
            for i in range(max(0, len(data)-60), len(data)):
                if abs(data['Low'].iloc[i] - support_level) / support_level < 0.01:
                    support_touches += 1
                if abs(data['High'].iloc[i] - resistance_level) / resistance_level < 0.01:
                    resistance_touches += 1
            
            # ===== GAP ANALYSIS =====
            if len(data) >= 2:
                prev_close = data['Close'].iloc[-2]
                curr_open = data['Open'].iloc[-1]
                gap_pct = ((curr_open - prev_close) / prev_close) * 100
            else:
                gap_pct = 0.0
            
            # ===== TREND ANALYSIS =====
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
            
            # ===== PATTERN RECOGNITION =====
            patterns = self.detect_patterns(data)
            
            # ===== DIVERGENCE DETECTION =====
            divergences = self.detect_divergences(data)
            
            # ===== COMPILE METRICS =====
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
    
    # ========== PATTERN RECOGNITION ==========
    def detect_patterns(self, data):
        patterns = []
        if len(data) < 3:
            return patterns
        
        last_3 = data.iloc[-3:]
        c0, c1, c2 = last_3.iloc[0], last_3.iloc[1], last_3.iloc[2]
        
        # Bullish Engulfing
        if c1['Close'] < c1['Open'] and c2['Close'] > c2['Open'] and \
           c2['Open'] < c1['Close'] and c2['Close'] > c1['Open']:
            patterns.append("🟢 Bullish Engulfing")
        
        # Bearish Engulfing
        if c1['Close'] > c1['Open'] and c2['Close'] < c2['Open'] and \
           c2['Open'] > c1['Close'] and c2['Close'] < c1['Open']:
            patterns.append("🔴 Bearish Engulfing")
        
        # Hammer
        body = abs(c2['Close'] - c2['Open'])
        lower_wick = min(c2['Open'], c2['Close']) - c2['Low']
        upper_wick = c2['High'] - max(c2['Open'], c2['Close'])
        if lower_wick > 2 * body and upper_wick < body:
            patterns.append("🔨 Hammer (Bullish)")
        
        # Shooting Star
        if upper_wick > 2 * body and lower_wick < body:
            patterns.append("💫 Shooting Star (Bearish)")
        
        # Doji
        if body < (c2['High'] - c2['Low']) * 0.1:
            patterns.append("➕ Doji (Indecision)")
        
        return patterns
    
    # ========== DIVERGENCE DETECTION ==========
    def detect_divergences(self, data):
        divergences = []
        if len(data) < 20:
            return divergences
        
        recent = data.iloc[-20:]
        
        # RSI Divergence
        if 'RSI_14' in recent.columns:
            price_highs = recent['High'].iloc[-10:].max()
            price_lows = recent['Low'].iloc[-10:].min()
            rsi_highs = recent['RSI_14'].iloc[-10:].max()
            rsi_lows = recent['RSI_14'].iloc[-10:].min()
            
            current_price = recent['Close'].iloc[-1]
            current_rsi = recent['RSI_14'].iloc[-1]
            
            # Bullish Divergence: Price makes lower low, RSI makes higher low
            if current_price < price_lows * 1.01 and current_rsi > rsi_lows * 1.05:
                divergences.append("🟢 RSI Bullish Divergence")
            
            # Bearish Divergence: Price makes higher high, RSI makes lower high
            if current_price > price_highs * 0.99 and current_rsi < rsi_highs * 0.95:
                divergences.append("🔴 RSI Bearish Divergence")
        
        # MACD Divergence
        if 'MACD_12_26_9' in recent.columns:
            macd_current = recent['MACD_12_26_9'].iloc[-1]
            macd_prev_high = recent['MACD_12_26_9'].iloc[-10:].max()
            macd_prev_low = recent['MACD_12_26_9'].iloc[-10:].min()
            
            price_current = recent['Close'].iloc[-1]
            price_prev_high = recent['High'].iloc[-10:].max()
            
            if price_current > price_prev_high * 0.99 and macd_current < macd_prev_high * 0.95:
                divergences.append("🔴 MACD Bearish Divergence")
        
        return divergences
    
    # ========== MACRO DATA ==========
    def get_macro(self):
        try:
            tickers = ["ES=F", "^VIX", "GC=F", "^GDAXI", "^N225", "^TNX", "DX-Y.NYB", "^GSPC"]
            df = yf.download(tickers, period="5d", progress=False, timeout=10)['Close']
            
            if df.empty:
                return None
            
            sp = df["ES=F"].dropna()
            vix = df["^VIX"].dropna()
            gold = df["GC=F"].dropna()
            dax = df["^GDAXI"].dropna()
            nikkei = df["^N225"].dropna()
            tnx = df["^TNX"].dropna()
            dxy = df["DX-Y.NYB"].dropna() if "DX-Y.NYB" in df.columns else None
            spy = df["^GSPC"].dropna()
            
            sp_chg = ((sp.iloc[-1]-sp.iloc[-2])/sp.iloc[-2])*100 if len(sp) >= 2 else 0
            dax_chg = ((dax.iloc[-1]-dax.iloc[-2])/dax.iloc[-2])*100 if len(dax) >= 2 else 0
            nikkei_chg = ((nikkei.iloc[-1]-nikkei.iloc[-2])/nikkei.iloc[-2])*100 if len(nikkei) >= 2 else 0
            tnx_chg = ((tnx.iloc[-1]-tnx.iloc[-2])/tnx.iloc[-2])*100 if len(tnx) >= 2 else 0
            dxy_chg = ((dxy.iloc[-1]-dxy.iloc[-2])/dxy.iloc[-2])*100 if dxy is not None and len(dxy) >= 2 else 0
            gold_chg = ((gold.iloc[-1]-gold.iloc[-2])/gold.iloc[-2])*100 if len(gold) >= 2 else 0
            spy_chg = ((spy.iloc[-1]-spy.iloc[-2])/spy.iloc[-2])*100 if len(spy) >= 2 else 0
            
            day = datetime.now().day
            passive_on = (1 <= day <= 5) or (15 <= day <= 20)
            
            # Risk-Off Detection (Gold + VIX both rising)
            risk_off = gold_chg > 1.0 and vix.iloc[-1] > 25
            
            # Dollar Headwind (DXY rising = bad for commodities)
            dollar_headwind = dxy_chg > 0.5 if dxy is not None else False
            
            return {
                "sp": sp_chg, "vix": vix.iloc[-1], "gold": gold.iloc[-1], "gold_chg": gold_chg,
                "dax": dax_chg, "nikkei": nikkei_chg, "tnx": tnx.iloc[-1], "tnx_chg": tnx_chg,
                "dxy": dxy.iloc[-1] if dxy is not None else 100, "dxy_chg": dxy_chg,
                "spy_chg": spy_chg, "passive": passive_on, "risk_off": risk_off,
                "dollar_headwind": dollar_headwind, "data_quality": "GOOD"
            }
        except Exception as e:
            return None
    
    # ========== SIGNAL GENERATION ==========
    def generate_signals(self, data, metrics, ticker):
        signals = {"bullish": [], "bearish": [], "neutral": [], "score": 0}
        
        # MACD Signal
        if metrics['macd'] > metrics['macd_signal']:
            signals['bullish'].append("MACD: Bullish crossover")
            signals['score'] += 1
        elif metrics['macd'] < metrics['macd_signal']:
            signals['bearish'].append("MACD: Bearish crossover")
            signals['score'] -= 1
        
        # RSI Signals
        if metrics['rsi'] < 30:
            signals['bullish'].append("RSI: Oversold (<30)")
            signals['score'] += 1
        elif metrics['rsi'] > 70:
            signals['bearish'].append("RSI: Overbought (>70)")
            signals['score'] -= 1
        
        # Bollinger Bands
        price = data['Close'].iloc[-1]
        if price < metrics['bb_lower']:
            signals['bullish'].append("BB: Below lower band (oversold)")
            signals['score'] += 1
        elif price > metrics['bb_upper']:
            signals['bearish'].append("BB: Above upper band (overbought)")
            signals['score'] -= 1
        
        # BB Squeeze (low volatility → breakout coming)
        avg_bb_width = (data['BBU_20_2.0'] - data['BBL_20_2.0']).mean() if 'BBU_20_2.0' in data.columns else 0
        if metrics['bb_width'] < avg_bb_width * 0.7:
            signals['neutral'].append("BB: Squeeze detected (breakout imminent)")
        
        # Stochastic
        if metrics['stoch_k'] < 20:
            signals['bullish'].append("Stochastic: Oversold")
            signals['score'] += 1
        elif metrics['stoch_k'] > 80:
            signals['bearish'].append("Stochastic: Overbought")
            signals['score'] -= 1
        
        # ADX (trend strength)
        if metrics['adx'] > 25:
            if metrics['trend_strength'] in ["STRONG_BULL", "BULL"]:
                signals['bullish'].append(f"ADX: Strong uptrend ({metrics['adx']:.1f})")
                signals['score'] += 1
            elif metrics['trend_strength'] in ["STRONG_BEAR", "BEAR"]:
                signals['bearish'].append(f"ADX: Strong downtrend ({metrics['adx']:.1f})")
                signals['score'] -= 1
        else:
            signals['neutral'].append(f"ADX: Weak trend ({metrics['adx']:.1f})")
        
        # Ichimoku Cloud
        if metrics['ich_position'] == "ABOVE_CLOUD":
            signals['bullish'].append("Ichimoku: Above cloud (bullish)")
            signals['score'] += 1
        elif metrics['ich_position'] == "BELOW_CLOUD":
            signals['bearish'].append("Ichimoku: Below cloud (bearish)")
            signals['score'] -= 1
        else:
            signals['neutral'].append("Ichimoku: Inside cloud (neutral)")
        
        # Money Flow Index
        if metrics['mfi'] < 20:
            signals['bullish'].append("MFI: Oversold")
        elif metrics['mfi'] > 80:
            signals['bearish'].append("MFI: Overbought")
        
        # Williams %R
        if metrics['willr'] < -80:
            signals['bullish'].append("Williams %R: Oversold")
        elif metrics['willr'] > -20:
            signals['bearish'].append("Williams %R: Overbought")
        
        # Moving Average Crossovers
        sma20 = data['SMA_20'].iloc[-1]
        sma50 = data['SMA_50'].iloc[-1]
        if sma20 > sma50 and data['SMA_20'].iloc[-2] <= data['SMA_50'].iloc[-2]:
            signals['bullish'].append("MA: Golden Cross (20>50)")
            signals['score'] += 2
        elif sma20 < sma50 and data['SMA_20'].iloc[-2] >= data['SMA_50'].iloc[-2]:
            signals['bearish'].append("MA: Death Cross (20<50)")
            signals['score'] -= 2
        
        # SuperTrend
        if data['ST_DIR'].iloc[-1] == 1:
            signals['bullish'].append("SuperTrend: Long signal")
            signals['score'] += 1
        else:
            signals['bearish'].append("SuperTrend: Short signal")
            signals['score'] -= 1
        
        # Pattern Recognition
        for pattern in metrics['patterns']:
            if "Bullish" in pattern or "Hammer" in pattern:
                signals['bullish'].append(f"Pattern: {pattern}")
                signals['score'] += 1
            elif "Bearish" in pattern or "Shooting Star" in pattern:
                signals['bearish'].append(f"Pattern: {pattern}")
                signals['score'] -= 1
            else:
                signals['neutral'].append(f"Pattern: {pattern}")
        
        # Divergences
        for div in metrics['divergences']:
            if "Bullish" in div:
                signals['bullish'].append(div)
                signals['score'] += 2
            elif "Bearish" in div:
                signals['bearish'].append(div)
                signals['score'] -= 2
        
        return signals
    
    # ========== CORRELATION & HELPERS ==========
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
    
    def calculate_relative_strength(self, ticker_data, spy_data):
        """Calculate RS vs SPY"""
        try:
            ticker_ret = (ticker_data['Close'].iloc[-1] / ticker_data['Close'].iloc[-20] - 1) * 100
            spy_ret = (spy_data['Close'].iloc[-1] / spy_data['Close'].iloc[-20] - 1) * 100
            rs = ticker_ret - spy_ret
            return rs, "OUTPERFORMING" if rs > 0 else "UNDERPERFORMING"
        except:
            return 0, "NEUTRAL"
    
    # ========== POSITION SIZING ==========
    def calculate_position_size(self, capital, risk_per_trade, risk_distance, method="FIXED", 
                               vol_mult=1.0, beta=1.0, consecutive_losses=0):
        if risk_distance <= 0:
            return 0
        
        # Drawdown protection
        drawdown_mult = 1.0
        if consecutive_losses >= 3:
            drawdown_mult = 0.5
        elif consecutive_losses >= 5:
            drawdown_mult = 0.25
        
        # Beta adjustment
        beta_mult = 1.0 / max(beta, 0.5) if beta > 1.5 else 1.0
        
        if method == "FIXED":
            shares = int((risk_per_trade * vol_mult * drawdown_mult * beta_mult) / risk_distance)
        elif method == "VOLATILITY_ADJUSTED":
            adjusted_risk = risk_per_trade * vol_mult * drawdown_mult * beta_mult
            shares = int(adjusted_risk / risk_distance)
        elif method == "KELLY":
            win_rate = 0.55
            win_loss_ratio = 2.0
            kelly_fraction = ((win_rate * win_loss_ratio) - (1 - win_rate)) / win_loss_ratio
            kelly_fraction = max(0.1, min(kelly_fraction * 0.5 * drawdown_mult, 0.25))
            adjusted_risk = capital * kelly_fraction
            shares = int(adjusted_risk / risk_distance)
        
        return max(0, shares)
    
    # ========== PERFORMANCE METRICS ==========
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
        """Average $ per trade"""
        if not trades:
            return 0
        total_pnl = sum(t['actual_pnl'] for t in trades if 'actual_pnl' in t)
        return total_pnl / len(trades)
    
    def calculate_profit_factor(self, trades):
        """Gross profit / Gross loss"""
        wins = [t['actual_pnl'] for t in trades if 'actual_pnl' in t and t['actual_pnl'] > 0]
        losses = [abs(t['actual_pnl']) for t in trades if 'actual_pnl' in t and t['actual_pnl'] < 0]
        if not losses or sum(losses) == 0:
            return None
        return sum(wins) / sum(losses)
    
    def calculate_max_drawdown(self, trades):
        """Maximum peak-to-trough decline"""
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
    
    # ========== BACKTEST SIMULATOR ==========
    def simple_backtest(self, data, strategy, entry_level, stop_mult, target_level):
        """Simple historical simulation"""
        trades = []
        for i in range(60, len(data)):
            price = data['Close'].iloc[i]
            atr = data['ATRr_14'].iloc[i]
            
            # Entry trigger
            if strategy == "LONG" and price <= entry_level:
                entry = price
                stop = entry - (atr * stop_mult)
                target = target_level
                
                # Find exit
                for j in range(i+1, min(i+20, len(data))):
                    if data['Low'].iloc[j] <= stop:
                        exit_price = stop
                        pnl = (exit_price - entry)
                        trades.append({"entry": entry, "exit": exit_price, "pnl": pnl, "result": "LOSS"})
                        break
                    elif data['High'].iloc[j] >= target:
                        exit_price = target
                        pnl = (exit_price - entry)
                        trades.append({"entry": entry, "exit": exit_price, "pnl": pnl, "result": "WIN"})
                        break
        
        if not trades:
            return None
        
        wins = [t for t in trades if t['result'] == "WIN"]
        losses = [t for t in trades if t['result'] == "LOSS"]
        win_rate = len(wins) / len(trades) * 100
        avg_win = np.mean([t['pnl'] for t in wins]) if wins else 0
        avg_loss = np.mean([abs(t['pnl']) for t in losses]) if losses else 0
        
        return {
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expectancy": np.mean([t['pnl'] for t in trades])
        }

engine = UltimateAnalyst()

# --- 5. SIDEBAR ---
with st.sidebar:
    st.header("💼 Portfolio Settings")
    
    capital = st.number_input("Total Capital ($)", value=10000, min_value=100)
    risk_per_trade = st.number_input("Risk/Trade ($)", value=100, min_value=10)
    max_portfolio_risk = st.number_input("Max Portfolio Risk (%)", value=6.0, min_value=1.0, max_value=20.0, step=0.5)
    
    daily_goal = capital * 0.01
    st.caption(f"🎯 Daily Goal (1%): **${daily_goal:.2f}**")
    
    portfolio_risk_pct = (st.session_state.total_risk_deployed / capital) * 100
    if portfolio_risk_pct > max_portfolio_risk:
        st.error(f"⚠️ Risk: {portfolio_risk_pct:.1f}% (OVER)")
    else:
        st.info(f"📊 Risk: {portfolio_risk_pct:.1f}% / {max_portfolio_risk:.1f}%")
    
    pnl_pct = (st.session_state.daily_pnl / capital) * 100
    if st.session_state.goal_met:
        st.success(f"✅ Goal: +${st.session_state.daily_pnl:.2f} ({pnl_pct:.2f}%)")
    else:
        st.info(f"📈 P&L: ${st.session_state.daily_pnl:.2f} ({pnl_pct:.2f}%)")
    
    if st.session_state.consecutive_losses > 0:
        st.warning(f"⚠️ Losing Streak: {st.session_state.consecutive_losses} trades")

    st.divider()
    st.header("🎯 Asset Selection")
    input_mode = st.radio("Input:", ["VIP List", "Manual"])
    if input_mode == "VIP List":
        ticker = st.selectbox("Ticker", VIP_TICKERS)
    else:
        ticker = st.text_input("Ticker", "NVDA").upper()

    st.divider()
    st.header("⚙️ Strategy")
    strategy = st.selectbox("Mode", ['Long (Buy)', 'Short (Sell)', 'Income (Puts)'])
    entry_mode = st.radio("Entry", ["Auto-Limit", "Market", "Manual"])
    manual_price = 0.0
    if entry_mode == "Manual":
        manual_price = st.number_input("Price", 0.0, step=0.01)
    
    stop_mode = st.selectbox("Stop", [1.0, 0.5, 0.2], format_func=lambda x: f"{x} ATR")
    position_sizing_method = st.selectbox("Sizing", ["FIXED", "VOLATILITY_ADJUSTED", "KELLY"])
    
    premium = 0.0
    if "Income" in strategy:
        premium = st.number_input("Premium ($)", 0.0, step=0.05)

    st.divider()
    st.header("📋 IWT Scorecard")
    
    # Auto-suggest based on zone touches
    if st.session_state.metrics:
        m = st.session_state.metrics
        if "Long" in strategy:
            suggested_fresh = 2 if m.get('support_touches', 0) == 0 else 1 if m.get('support_touches', 0) <= 2 else 0
        else:
            suggested_fresh = 2 if m.get('resistance_touches', 0) == 0 else 1 if m.get('resistance_touches', 0) <= 2 else 0
        st.caption(f"💡 Suggested Freshness: {suggested_fresh} ({m.get('support_touches', 0)} touches)")
    
    fresh = st.selectbox("Freshness", [2, 1, 0], format_func=lambda x: {2:'2-Fresh', 1:'1-Used', 0:'0-Stale'}[x])
    speed = st.selectbox("Speed Out", [2, 1, 0], format_func=lambda x: {2:'2-Fast', 1:'1-Avg', 0:'0-Slow'}[x])
    time_z = st.selectbox("Time in Zone", [2, 1, 0], format_func=lambda x: {2:'2-Short', 1:'1-Med', 0:'0-Long'}[x])
    
    st.divider()
    if st.button("🔄 Reset Session"):
        st.session_state.goal_met = False
        st.session_state.daily_pnl = 0.0
        st.session_state.total_risk_deployed = 0.0
        st.session_state.open_positions = []
        st.session_state.consecutive_losses = 0
        st.rerun()

# --- 6. MAIN UI ---
st.subheader("🌍 Global Intelligence")

col_macro, col_scan = st.columns([1, 1])
with col_macro:
    if st.button("🌍 Macro Audit", use_container_width=True):
        with st.spinner("Scanning..."):
            st.session_state.macro = engine.get_macro()

with col_scan:
    if st.button(f"🔎 Scan {ticker}", type="primary", use_container_width=True):
        with st.spinner(f"Analyzing {ticker}..."):
            df, metrics, fname = engine.fetch_data(ticker)
            
            if df is None:
                st.error(f"🚫 {ticker}: {metrics}")
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
                
                # Generate signals
                st.session_state.signals = engine.generate_signals(df, metrics, ticker)
                st.success(f"✅ {fname} loaded")

# --- 7. MACRO DISPLAY ---
if st.session_state.macro:
    m = st.session_state.macro
    
    if m.get('data_quality') == 'DEGRADED':
        st.warning("⚠️ Data quality degraded")
    
    vix_regime = engine.classify_vix_regime(m['vix'])
    regime_guide = engine.get_regime_guidance(vix_regime)
    
    market_phase, phase_desc = engine.get_market_hours_status()
    
    if vix_regime in ["EXTREME", "HIGH"]:
        st.error(f"🚨 VIX: {regime_guide['desc']} ({m['vix']:.1f})")
    elif vix_regime == "ELEVATED":
        st.warning(f"⚠️ VIX: {regime_guide['desc']} ({m['vix']:.1f})")
    else:
        st.success(f"✅ VIX: {regime_guide['desc']} ({m['vix']:.1f})")
    
    if market_phase in ["PRE_MARKET", "AFTER_HOURS"]:
        st.warning(f"⏰ {phase_desc}")
    elif market_phase == "LUNCH":
        st.info(f"{phase_desc}")
    else:
        st.caption(f"{phase_desc}")
    
    # Risk-Off Warning
    if m.get('risk_off'):
        st.error("🚨 **RISK-OFF REGIME:** Gold + VIX both rising. Flight to safety. Avoid aggressive longs.")
    
    # Dollar Headwind
    if m.get('dollar_headwind') and ticker in COMMODITY_TICKERS:
        st.warning("💵 **DOLLAR HEADWIND:** DXY rising = pressure on commodities.")
    
    flow_strength = engine.check_passive_intensity(
        datetime.now().day, 
        st.session_state.metrics.get('rvol', 0) if st.session_state.metrics else 0
    )
    
    if flow_strength == "STRONG":
        st.success("🌊 **STRONG PASSIVE INFLOWS**")
    elif flow_strength == "MODERATE":
        st.info("🌊 **MODERATE PASSIVE INFLOWS**")
    elif flow_strength == "WEAK":
        st.warning("🌊 **WEAK PASSIVE INFLOWS**")
    else:
        st.info("⏸️ **PASSIVE FLOWS NEUTRAL**")
    
    if engine.detect_correlation_break(m):
        st.error("🌍 **CORRELATION BREAK:** Global markets diverging. Volatility risk elevated.")
    
    with st.expander("🌍 Macro Dashboard", expanded=True):
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("🇺🇸 S&P", f"{m['sp']:.2f}%")
        col2.metric("🇩🇪 DAX", f"{m['dax']:.2f}%")
        col3.metric("🇯🇵 Nikkei", f"{m['nikkei']:.2f}%")
        col4.metric("💵 DXY", f"{m['dxy']:.1f}", delta=f"{m['dxy_chg']:.2f}%")
        col5.metric("📈 10Y", f"{m['tnx']:.2f}%", delta=f"{m['tnx_chg']:.2f}%")

# --- 8. ASSET ANALYSIS ---
if st.session_state.data is not None:
    m = st.session_state.metrics
    df = st.session_state.data
    
    st.divider()
    st.header(f"📈 {m['name']} ({ticker})")
    
    # Metrics row
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    col1.metric("Price", f"${m['price']:.2f}")
    
    if abs(m['gap']) > 2.0:
        if m['rvol'] > 1.5:
            col2.metric("Gap", f"{m['gap']:.2f}%", delta="🚀 PRO")
        else:
            col2.metric("Gap", f"{m['gap']:.2f}%", delta="⚠️ NOVICE")
    else:
        col2.metric("Gap", f"{m['gap']:.2f}%")
    
    vol_status = "🔥" if m['rvol'] > 1.5 else "✅" if m['rvol'] > 0.8 else "💤"
    col3.metric("Volume", f"{m['rvol']:.1f}x", delta=vol_status)
    
    trend_emoji = {"STRONG_BULL": "🚀", "BULL": "📈", "NEUTRAL": "➡️", "BEAR": "📉", "STRONG_BEAR": "🔻"}
    col4.metric("Trend", m['trend_strength'], delta=trend_emoji.get(m['trend_strength'], "➡️"))
    
    rsi_status = "⚠️OB" if m['rsi'] > 70 else "⚠️OS" if m['rsi'] < 30 else "✅"
    col5.metric("RSI", f"{m['rsi']:.0f}", delta=rsi_status)
    
    col6.metric("ADX", f"{m['adx']:.0f}", delta="STRONG" if m['adx'] > 25 else "WEAK")
    
    # Relative Strength vs SPY
    if st.session_state.macro and ticker != "SPY":
        spy_data = yf.Ticker("SPY").history(period="1mo")
        rs, rs_status = engine.calculate_relative_strength(df, spy_data)
        st.caption(f"**RS vs SPY:** {rs:+.1f}% ({rs_status})")
    
    # Support/Resistance with touches
    st.caption(f"**Levels:** Support ${m['supp']:.2f} ({m.get('support_touches', 0)} touches) | Resistance ${m['res']:.2f} ({m.get('resistance_touches', 0)} touches)")
    
    # ===== MULTI-INDICATOR SIGNAL DASHBOARD =====
    if st.session_state.signals:
        st.divider()
        st.subheader("🎯 Multi-Algorithm Signals")
        
        sig = st.session_state.signals
        
        col_bull, col_bear, col_neut = st.columns(3)
        
        with col_bull:
            st.markdown("### 🟢 Bullish Signals")
            if sig['bullish']:
                for s in sig['bullish']:
                    st.markdown(f"<div class='signal-bull'>• {s}</div>", unsafe_allow_html=True)
            else:
                st.caption("None")
        
        with col_bear:
            st.markdown("### 🔴 Bearish Signals")
            if sig['bearish']:
                for s in sig['bearish']:
                    st.markdown(f"<div class='signal-bear'>• {s}</div>", unsafe_allow_html=True)
            else:
                st.caption("None")
        
        with col_neut:
            st.markdown("### 🟡 Neutral Signals")
            if sig['neutral']:
                for s in sig['neutral']:
                    st.markdown(f"<div class='signal-neutral'>• {s}</div>", unsafe_allow_html=True)
            else:
                st.caption("None")
        
        # Overall Signal Score
        score = sig['score']
        if score >= 3:
            st.success(f"### 🟢 OVERALL SIGNAL: BULLISH (+{score})")
        elif score <= -3:
            st.error(f"### 🔴 OVERALL SIGNAL: BEARISH ({score})")
        else:
            st.warning(f"### 🟡 OVERALL SIGNAL: NEUTRAL ({score})")
    
    # ===== CHART =====
    st.divider()
    st.subheader("📊 Technical Chart")
    
    chart_data = df.iloc[-60:]
    
    addplots = [
        mpf.make_addplot(chart_data['SMA_20'], color='blue', width=1.5),
        mpf.make_addplot(chart_data['SMA_50'], color='orange', width=1.5),
        mpf.make_addplot(chart_data['SMA_200'], color='red', width=2),
        mpf.make_addplot(chart_data['BBU_20_2.0'], color='gray', linestyle='--', width=1),
        mpf.make_addplot(chart_data['BBL_20_2.0'], color='gray', linestyle='--', width=1),
    ]
    
    fig, axes = mpf.plot(
        chart_data, type='candle', style='yahoo', volume=True,
        addplot=addplots,
        hlines=dict(hlines=[m['supp'], m['res']], colors=['green', 'red'], linestyle='-.', linewidths=2),
        returnfig=True, figsize=(14, 8), title=f"{ticker} - Multi-Indicator Analysis"
    )
    
    st.pyplot(fig)
    
    # Fibonacci Levels
    with st.expander("📐 Fibonacci Retracement Levels"):
        fib = m['fib_levels']
        for level, price in fib.items():
            st.caption(f"{level}: ${price:.2f}")
    
    # ===== TRADE CALCULATION =====
    st.divider()
    st.subheader("🎯 Trade Setup")
    
    if entry_mode == "Manual":
        entry = manual_price
    elif "Short" in strategy:
        entry = m['res'] if "Auto" in entry_mode else m['price']
    else:
        entry = m['supp'] if "Auto" in entry_mode else m['price']
    
    if st.session_state.macro:
        vix_regime = engine.classify_vix_regime(st.session_state.macro['vix'])
        regime_guide = engine.get_regime_guidance(vix_regime)
        vol_multiplier = regime_guide['size_mult']
        stop_multiplier = regime_guide['stop_mult']
    else:
        vol_multiplier = 1.0
        stop_multiplier = 1.0
    
    # Calculate with slippage
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
    else:
        stop = entry - (m['atr'] * stop_mode * stop_multiplier)
        target = m['res']
        risk = entry - stop
        reward = target - entry
    
    rr = reward / risk if risk > 0 else 0
    
    # Position sizing with beta and drawdown
    shares = engine.calculate_position_size(
        capital, risk_per_trade, risk, position_sizing_method, 
        vol_multiplier, m.get('beta', 1.0), st.session_state.consecutive_losses
    )
    
    total_trade_risk = shares * risk if shares > 0 else 0
    
    # Add slippage and commissions
    slippage = entry * (SLIPPAGE_BPS / 10000) * shares
    commissions = shares * COMMISSION_PER_SHARE * 2  # Buy + Sell
    gross_reward = shares * reward
    net_reward = gross_reward - slippage - commissions
    
    col_setup1, col_setup2, col_setup3 = st.columns(3)
    
    with col_setup1:
        st.markdown("**📍 Levels**")
        st.code(f"""
Entry:  ${entry:.2f}
Stop:   ${stop:.2f}
Target: ${target:.2f}
        """)
    
    with col_setup2:
        st.markdown("**💰 Position**")
        sizing_note = f" ({vol_multiplier:.1f}x VIX)" if vol_multiplier != 1.0 else ""
        beta_note = f" (β={m.get('beta', 1.0):.2f})" if m.get('beta', 1.0) != 1.0 else ""
        st.code(f"""
Size:    {shares} shares{sizing_note}{beta_note}
Risk:    ${total_trade_risk:.2f}
Reward:  ${gross_reward:.2f}
R/R:     {rr:.2f}
        """)
    
    with col_setup3:
        st.markdown("**💸 Real Costs**")
        st.code(f"""
Slippage:     ${slippage:.2f}
Commissions:  ${commissions:.2f}
Net Reward:   ${net_reward:.2f}
Net R/R:      {(net_reward/(total_trade_risk if total_trade_risk>0 else 1)):.2f}
        """)
    
    # ===== SCORING & VERDICT =====
    st.divider()
    st.subheader("🚦 The Ultimate Verdict")
    
    score_rr = 2 if rr >= 3 or ("Income" in strategy and rr > 0.1) else 1 if rr >= 2 else 0
    total_score = fresh + time_z + speed + score_rr
    
    penalties = []
    
    # Warsh penalty
    warsh_penalty = False
    if st.session_state.macro and st.session_state.macro['tnx_chg'] > 1.0 and ticker in GROWTH_TICKERS and "Long" in strategy:
        total_score -= 2
        warsh_penalty = True
        penalties.append("Warsh (-2): Yields rising >1%")
    
    # Market hours
    market_phase, _ = engine.get_market_hours_status()
    if market_phase in ["LUNCH", "PRE_MARKET", "AFTER_HOURS"]:
        total_score -= 1
        penalties.append(f"Hours (-1): {market_phase}")
    
    # Trend misalignment
    if "Long" in strategy and m['trend_strength'] in ["BEAR", "STRONG_BEAR"]:
        total_score -= 1
        penalties.append("Trend (-1): Long in downtrend")
    elif "Short" in strategy and m['trend_strength'] in ["BULL", "STRONG_BULL"]:
        total_score -= 1
        penalties.append("Trend (-1): Short in uptrend")
    
    # Sector concentration
    sector = SECTOR_MAP.get(ticker, "Unknown")
    sector_exposure = sum(1 for p in st.session_state.open_positions if SECTOR_MAP.get(p['ticker'], '') == sector)
    if sector_exposure >= 2:
        total_score -= 1
        penalties.append(f"Concentration (-1): {sector_exposure+1} in {sector}")
    
    # Risk-off penalty (Gold + VIX rising)
    if st.session_state.macro and st.session_state.macro.get('risk_off') and "Long" in strategy:
        total_score -= 1
        penalties.append("Risk-Off (-1): Gold + VIX rising")
    
    # Dollar headwind on commodities
    if st.session_state.macro and st.session_state.macro.get('dollar_headwind') and ticker in COMMODITY_TICKERS and "Long" in strategy:
        total_score -= 1
        penalties.append("Dollar (-1): DXY rising hurts commodities")
    
    # SuperTrend conflict
    if 'ST_DIR' in df.columns:
        if "Long" in strategy and df['ST_DIR'].iloc[-1] == -1:
            total_score -= 1
            penalties.append("SuperTrend (-1): Bearish signal")
        elif "Short" in strategy and df['ST_DIR'].iloc[-1] == 1:
            total_score -= 1
            penalties.append("SuperTrend (-1): Bullish signal")
    
    # RSI extreme + misalignment
    if "Long" in strategy and m['rsi'] > 70:
        total_score -= 1
        penalties.append("RSI (-1): Overbought")
    elif "Short" in strategy and m['rsi'] < 30:
        total_score -= 1
        penalties.append("RSI (-1): Oversold")
    
    # Multi-indicator signal conflict
    if st.session_state.signals:
        sig_score = st.session_state.signals['score']
        if "Long" in strategy and sig_score < -2:
            total_score -= 1
            penalties.append("Signals (-1): Multi-algo bearish")
        elif "Short" in strategy and sig_score > 2:
            total_score -= 1
            penalties.append("Signals (-1): Multi-algo bullish")
    
    col_verdict, col_analysis = st.columns([1, 1])
    
    with col_verdict:
        if st.session_state.goal_met:
            st.error("## 🛑 DAILY GOAL MET")
            st.markdown("<div class='risk-warning'><strong>STOP TRADING.</strong> Protect your 1% gain.</div>", unsafe_allow_html=True)
            can_trade = False
        elif (st.session_state.total_risk_deployed + total_trade_risk) > (capital * max_portfolio_risk / 100):
            st.error("## 🛑 PORTFOLIO RISK LIMIT")
            st.markdown(f"<div class='risk-warning'>Adding this trade exceeds {max_portfolio_risk}% portfolio risk limit.</div>", unsafe_allow_html=True)
            can_trade = False
        else:
            can_trade = True
            if total_score >= 7:
                st.success(f"## 🟢 GREEN LIGHT\n**Score: {total_score}/8**")
                st.caption("✅ Execute with confidence.")
            elif total_score >= 5:
                st.warning(f"## 🟡 YELLOW LIGHT\n**Score: {total_score}/8**")
                st.caption("⚠️ Reduce size 50% OR wait.")
            else:
                st.error(f"## 🔴 RED LIGHT\n**Score: {total_score}/8**")
                st.caption("🛑 DO NOT TRADE.")
        
        if penalties:
            st.markdown("**Penalties:**")
            for p in penalties:
                st.caption(f"• {p}")
    
    with col_analysis:
        st.markdown("**📋 Multi-Factor Analysis**")
        
        checks = []
        checks.append(("✅" if fresh == 2 else "⚠️" if fresh == 1 else "❌", f"Freshness: {['Stale','Used','Fresh'][fresh]}"))
        checks.append(("✅" if score_rr == 2 else "⚠️" if score_rr == 1 else "❌", f"R/R: {rr:.2f}"))
        checks.append(("✅" if abs(m['gap']) > 2 else "➖", f"Gap: {m['gap']:.2f}%"))
        checks.append(("✅" if m['rvol'] > 1.2 else "⚠️", f"Volume: {m['rvol']:.1f}x"))
        checks.append(("✅" if m['adx'] > 25 else "⚠️", f"ADX: {m['adx']:.0f}"))
        
        for icon, text in checks:
            st.caption(f"{icon} {text}")
    
    # ===== BACKTEST =====
    with st.expander("🔬 Simple Backtest (Last 60 Days)", expanded=False):
        backtest_results = engine.simple_backtest(
            df, 
            "LONG" if "Long" in strategy else "SHORT", 
            entry, 
            stop_mode, 
            target
        )
        
        if backtest_results:
            col_bt1, col_bt2, col_bt3, col_bt4 = st.columns(4)
            col_bt1.metric("Trades", backtest_results['total_trades'])
            col_bt2.metric("Win Rate", f"{backtest_results['win_rate']:.1f}%")
            col_bt3.metric("Avg Win", f"${backtest_results['avg_win']:.2f}")
            col_bt4.metric("Expectancy", f"${backtest_results['expectancy']:.2f}")
            st.caption("⚠️ Past performance ≠ future results. Backtest uses simplified logic.")
        else:
            st.caption("No historical trades found with these parameters.")
    
    # ===== EXECUTION =====
    if can_trade and total_score >= 5:
        st.divider()
        st.subheader("⚡ Execution")
        
        col_exec1, col_exec2 = st.columns(2)
        
        with col_exec1:
            if st.button("📝 Log PAPER", use_container_width=True, type="secondary"):
                trade_record = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ticker": ticker, "action": strategy, "entry": entry, "stop": stop, "target": target,
                    "shares": shares, "score": total_score, "risk": total_trade_risk,
                    "expected_reward": gross_reward, "net_reward": net_reward, "rr_ratio": rr,
                    "slippage": slippage, "commissions": commissions, "status": "PAPER"
                }
                st.session_state.journal.append(trade_record)
                st.success("📋 Paper logged!")
        
        with col_exec2:
            if st.button("💵 LOG LIVE", use_container_width=True, type="primary"):
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
                st.success("✅ Live position OPEN!")
                st.rerun()

# --- 9. POSITION MANAGEMENT ---
if st.session_state.open_positions:
    st.divider()
    st.subheader("📊 Open Positions")
    
    positions_df = pd.DataFrame(st.session_state.open_positions)
    positions_df = positions_df[['ticker', 'action', 'entry', 'stop', 'target', 'shares', 'risk', 'score']]
    st.dataframe(positions_df, use_container_width=True)
    
    st.markdown("**Close Position:**")
    col_close1, col_close2, col_close3 = st.columns(3)
    
    with col_close1:
        position_to_close = st.selectbox("Position", [p['ticker'] for p in st.session_state.open_positions])
    
    with col_close2:
        exit_price = st.number_input("Exit Price", 0.0, step=0.01)
    
    with col_close3:
        if st.button("Close"):
            if exit_price > 0:
                for i, pos in enumerate(st.session_state.open_positions):
                    if pos['ticker'] == position_to_close:
                        if "Long" in pos['action']:
                            actual_pnl = (exit_price - pos['entry']) * pos['shares']
                        else:
                            actual_pnl = (pos['entry'] - exit_price) * pos['shares']
                        
                        # Subtract costs
                        actual_pnl -= (pos['slippage'] + pos['commissions'])
                        
                        pos['exit_price'] = exit_price
                        pos['actual_pnl'] = actual_pnl
                        pos['status'] = 'CLOSED'
                        
                        st.session_state.closed_trades.append(pos)
                        st.session_state.daily_pnl += actual_pnl
                        st.session_state.total_risk_deployed -= pos['risk']
                        
                        # Update consecutive losses
                        if actual_pnl < 0:
                            st.session_state.consecutive_losses += 1
                        else:
                            st.session_state.consecutive_losses = 0
                        
                        if st.session_state.daily_pnl >= daily_goal:
                            st.session_state.goal_met = True
                        
                        st.session_state.open_positions.pop(i)
                        st.success(f"✅ Closed {position_to_close}: P&L = ${actual_pnl:.2f}")
                        st.rerun()
                        break

# --- 10. PERFORMANCE ANALYTICS ---
if st.session_state.closed_trades:
    st.divider()
    st.subheader("📈 Performance Analytics")
    
    closed_df = pd.DataFrame(st.session_state.closed_trades)
    
    col_stats1, col_stats2, col_stats3, col_stats4, col_stats5 = st.columns(5)
    
    wins = len(closed_df[closed_df['actual_pnl'] > 0])
    total_trades = len(closed_df)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    col_stats1.metric("Win Rate", f"{win_rate:.1f}%", delta=f"{wins}/{total_trades}")
    
    avg_rr = closed_df['rr_ratio'].mean()
    col_stats2.metric("Avg R/R", f"{avg_rr:.2f}")
    
    total_pnl = closed_df['actual_pnl'].sum()
    col_stats3.metric("Total P&L", f"${total_pnl:.2f}", delta=f"{(total_pnl/capital)*100:.2f}%")
    
    sharpe = engine.calculate_sharpe_ratio(st.session_state.closed_trades)
    if sharpe:
        sharpe_quality = "Excellent" if sharpe > 2 else "Good" if sharpe > 1 else "Poor"
        col_stats4.metric("Sharpe", f"{sharpe:.2f}", delta=sharpe_quality)
    else:
        col_stats4.metric("Sharpe", "N/A")
    
    profit_factor = engine.calculate_profit_factor(st.session_state.closed_trades)
    if profit_factor:
        col_stats5.metric("Profit Factor", f"{profit_factor:.2f}", 
                         delta="Excellent" if profit_factor > 2 else "Good" if profit_factor > 1.5 else "Poor")
    else:
        col_stats5.metric("Profit Factor", "N/A")
    
    # Additional stats
    col_stats6, col_stats7, col_stats8 = st.columns(3)
    
    expectancy = engine.calculate_expectancy(st.session_state.closed_trades)
    col_stats6.metric("Expectancy", f"${expectancy:.2f}", help="Average $/trade")
    
    max_dd = engine.calculate_max_drawdown(st.session_state.closed_trades)
    col_stats7.metric("Max Drawdown", f"${max_dd:.2f}", help="Peak-to-trough decline")
    
    # Consecutive wins/losses
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
    
    col_stats8.metric("Max Streak", f"W:{max_consec_wins} / L:{max_consec_losses}")
    
    with st.expander("📋 Trade History", expanded=False):
        history_df = closed_df[['timestamp', 'ticker', 'action', 'entry', 'exit_price', 'actual_pnl', 'score', 'slippage', 'commissions']]
        st.dataframe(history_df, use_container_width=True)

# --- 11. JOURNAL EXPORT ---
if st.session_state.journal:
    st.divider()
    st.subheader("📓 Trading Journal")
    
    journal_df = pd.DataFrame(st.session_state.journal)
    st.dataframe(journal_df, use_container_width=True)
    
    col_export1, col_export2 = st.columns(2)
    
    with col_export1:
        csv = journal_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download Full Journal",
            data=csv,
            file_name=f"journal_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col_export2:
        if st.session_state.closed_trades:
            closed_csv = pd.DataFrame(st.session_state.closed_trades).to_csv(index=False).encode('utf-8')
            st.download_button(
                "📊 Download Performance",
                data=closed_csv,
                file_name=f"performance_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True
            )

else:
    st.info("👈 **Quick Start:** 1. Click **'Macro Audit'** 🌍 above to check the trend. 2. Set your **Ticker/Asset**, **Strategy** & **IWT Scores** on the left sidebar. 3. Click the red **'Scan'** button 🔴 to get your verdict.")

# --- FOOTER ---
st.divider()
st.caption("Quantum Maestro Financial Markets TradingBot | Multi-Algorithm Fusion | Educational Use Only")
