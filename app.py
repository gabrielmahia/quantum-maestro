# Copyright (c) 2026 Gabriel Mahia. All Rights Reserved.
# Unauthorized copying of this file, via any medium, is strictly prohibited.
# Proprietary and confidential.
# Written by Gabriel Mahia, 2026
# app.py - ULTIMATE EDITION WITH BEGINNER GUIDANCE
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

st.set_page_config(page_title="Quantum Maestro TradingBot", layout="wide", initial_sidebar_state="expanded", page_icon="🏛️")

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

with st.expander("⚠️ LEGAL DISCLAIMER - READ CAREFULLY", expanded=True):
    st.markdown("""
    **1. Not Financial Advice:** This is an educational tool only. Not affiliated with Trade and Travel or any trading organization.
    **2. Risk Warning:** Trading stocks involves substantial risk of loss. You can lose more than your initial investment. Past performance does not guarantee future results.
    **3. Data Disclaimer:** Market data provided by Yahoo Finance. Data may be delayed, incomplete, or inaccurate.
    **4. Simulated Results:** Backtests and projections are hypothetical. Real trading results will differ due to slippage, commissions, execution issues, and market conditions.
    **5. Your Responsibility:** All trading decisions are your own. Consult a licensed financial advisor before risking real capital.
    """)
    agree = st.checkbox("✅ I understand this is for educational purposes only and involves real financial risk.")

if not agree:
    st.warning("🛑 Please accept the disclaimer above to continue.")
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
                return "PRE_MARKET", "⏰ Pre-Market: Higher volatility, wider spreads, lower liquidity"
            elif current_time < time(10, 0):
                return "OPENING", "🔔 Opening Range: Wait for direction, avoid chasing"
            elif current_time < time(12, 0):
                return "MORNING", "☀️ Morning Session: Prime trading window with high liquidity"
            elif current_time < time(14, 0):
                return "LUNCH", "🍴 Lunch Hour: Reduced volume, avoid new positions"
            elif current_time < time(15, 0):
                return "AFTERNOON", "🌤️ Afternoon Session: Trend continuation phase"
            elif current_time < time(16, 0):
                return "POWER_HOUR", "⚡ Power Hour: Institutional positioning, high volume"
            else:
                return "AFTER_HOURS", "🌙 After Hours: Extended hours carry higher risk"
        except:
            return "UNKNOWN", "⚠️ Unable to determine market hours"
    
    def classify_vix_regime(self, vix_level):
        for regime, (low, high) in self.vix_regimes.items():
            if low <= vix_level < high:
                return regime
        return "EXTREME"
    
    def get_regime_guidance(self, regime):
        guidance = {
            "EXTREME_LOW": {"desc": "Complacency Zone", "size_mult": 0.7, "stop_mult": 1.2, 
                           "note": "Market pricing in zero risk. Potential for sudden reversals."},
            "LOW": {"desc": "Calm Waters", "size_mult": 1.0, "stop_mult": 1.0,
                   "note": "Normal market conditions. Standard position sizing appropriate."},
            "NORMAL": {"desc": "Healthy Volatility", "size_mult": 1.0, "stop_mult": 1.0,
                      "note": "Ideal trading environment. Markets functioning normally."},
            "ELEVATED": {"desc": "Heightened Uncertainty", "size_mult": 0.7, "stop_mult": 1.3,
                        "note": "Reduce size by 30%. Widen stops. Expect intraday swings."},
            "HIGH": {"desc": "Crisis Mode", "size_mult": 0.5, "stop_mult": 1.5,
                    "note": "Reduce size by 50%. Consider cash. Only highest-conviction setups."},
            "EXTREME": {"desc": "Market Panic", "size_mult": 0.3, "stop_mult": 2.0,
                       "note": "EXTREME VOLATILITY. Close non-essential positions. Capital preservation mode."}
        }
        return guidance.get(regime, guidance["NORMAL"])
    
    # ========== DATA FETCHING WITH FULL INDICATORS ==========
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
            
            # Bullish Divergence
            if current_price < price_lows * 1.01 and current_rsi > rsi_lows * 1.05:
                divergences.append("🟢 RSI Bullish Divergence")
            
            # Bearish Divergence
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
            
            risk_off = gold_chg > 1.0 and vix.iloc[-1] > 25
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
            signals['bullish'].append("MACD: Bullish crossover (MACD above signal line)")
            signals['score'] += 1
        elif metrics['macd'] < metrics['macd_signal']:
            signals['bearish'].append("MACD: Bearish crossover (MACD below signal line)")
            signals['score'] -= 1
        
        # RSI Signals
        if metrics['rsi'] < 30:
            signals['bullish'].append("RSI: Oversold (<30) - potential bounce")
            signals['score'] += 1
        elif metrics['rsi'] > 70:
            signals['bearish'].append("RSI: Overbought (>70) - potential pullback")
            signals['score'] -= 1
        
        # Bollinger Bands
        price = data['Close'].iloc[-1]
        if price < metrics['bb_lower']:
            signals['bullish'].append("BB: Below lower band (oversold)")
            signals['score'] += 1
        elif price > metrics['bb_upper']:
            signals['bearish'].append("BB: Above upper band (overbought)")
            signals['score'] -= 1
        
        # BB Squeeze
        avg_bb_width = (data['BBU_20_2.0'] - data['BBL_20_2.0']).mean() if 'BBU_20_2.0' in data.columns else 0
        if metrics['bb_width'] < avg_bb_width * 0.7:
            signals['neutral'].append("BB: Squeeze detected (low volatility, breakout imminent)")
        
        # Stochastic
        if metrics['stoch_k'] < 20:
            signals['bullish'].append("Stochastic: Oversold (<20)")
            signals['score'] += 1
        elif metrics['stoch_k'] > 80:
            signals['bearish'].append("Stochastic: Overbought (>80)")
            signals['score'] -= 1
        
        # ADX
        if metrics['adx'] > 25:
            if metrics['trend_strength'] in ["STRONG_BULL", "BULL"]:
                signals['bullish'].append(f"ADX: Strong uptrend (ADX={metrics['adx']:.1f})")
                signals['score'] += 1
            elif metrics['trend_strength'] in ["STRONG_BEAR", "BEAR"]:
                signals['bearish'].append(f"ADX: Strong downtrend (ADX={metrics['adx']:.1f})")
                signals['score'] -= 1
        else:
            signals['neutral'].append(f"ADX: Weak/choppy trend (ADX={metrics['adx']:.1f})")
        
        # Ichimoku Cloud
        if metrics['ich_position'] == "ABOVE_CLOUD":
            signals['bullish'].append("Ichimoku: Price above cloud (strong bullish)")
            signals['score'] += 1
        elif metrics['ich_position'] == "BELOW_CLOUD":
            signals['bearish'].append("Ichimoku: Price below cloud (strong bearish)")
            signals['score'] -= 1
        else:
            signals['neutral'].append("Ichimoku: Inside cloud (neutral/transitioning)")
        
        # Money Flow Index
        if metrics['mfi'] < 20:
            signals['bullish'].append("MFI: Money flowing in (<20 oversold)")
        elif metrics['mfi'] > 80:
            signals['bearish'].append("MFI: Money flowing out (>80 overbought)")
        
        # Williams %R
        if metrics['willr'] < -80:
            signals['bullish'].append("Williams %R: Oversold (<-80)")
        elif metrics['willr'] > -20:
            signals['bearish'].append("Williams %R: Overbought (>-20)")
        
        # Moving Average Crossovers
        sma20 = data['SMA_20'].iloc[-1]
        sma50 = data['SMA_50'].iloc[-1]
        if sma20 > sma50 and data['SMA_20'].iloc[-2] <= data['SMA_50'].iloc[-2]:
            signals['bullish'].append("MA: Golden Cross (SMA20 crossed above SMA50)")
            signals['score'] += 2
        elif sma20 < sma50 and data['SMA_20'].iloc[-2] >= data['SMA_50'].iloc[-2]:
            signals['bearish'].append("MA: Death Cross (SMA20 crossed below SMA50)")
            signals['score'] -= 2
        
        # SuperTrend
        if data['ST_DIR'].iloc[-1] == 1:
            signals['bullish'].append("SuperTrend: Long signal (price above SuperTrend)")
            signals['score'] += 1
        else:
            signals['bearish'].append("SuperTrend: Short signal (price below SuperTrend)")
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
        
        drawdown_mult = 1.0
        if consecutive_losses >= 3:
            drawdown_mult = 0.5
        elif consecutive_losses >= 5:
            drawdown_mult = 0.25
        
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
    
    # ========== BACKTEST SIMULATOR ==========
    def simple_backtest(self, data, strategy, entry_level, stop_mult, target_level):
        trades = []
        for i in range(60, len(data)):
            price = data['Close'].iloc[i]
            atr = data['ATRr_14'].iloc[i]
            
            if strategy == "LONG" and price <= entry_level:
                entry = price
                stop = entry - (atr * stop_mult)
                target = target_level
                
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

# --- 5. SIDEBAR WITH COMPREHENSIVE HELP NOTES ---
with st.sidebar:
    st.header("💼 Portfolio Settings")
    
    capital = st.number_input(
        "Total Capital ($)", 
        value=10000, 
        min_value=100,
        help="💡 Your total account size. This is the amount you're risking across all positions. Example: $10,000 means you have ten thousand dollars in your trading account."
    )
    
    risk_per_trade = st.number_input(
        "Risk per Trade ($)", 
        value=100, 
        min_value=10,
        help="💡 Maximum dollar amount you're willing to lose on a single trade. Recommended: 1-2% of capital. Example: $10,000 account × 1% = $100 risk per trade. This protects you from blowing up your account."
    )
    
    max_portfolio_risk = st.number_input(
        "Max Portfolio Risk (%)", 
        value=6.0, 
        min_value=1.0, 
        max_value=20.0, 
        step=0.5,
        help="💡 Maximum total risk across ALL open positions combined. Recommended: 5-10%. Example: If you have 3 positions risking $100 each, your portfolio risk is $300 (3% of $10K account). This prevents over-concentration."
    )
    
    daily_goal = capital * 0.01
    st.caption(f"🎯 Daily Goal (1%): **${daily_goal:.2f}**")
    st.caption("💡 The 1% rule: If you make 1% per day for 20 trading days, that's 20% per month. Consistency beats intensity.")
    
    portfolio_risk_pct = (st.session_state.total_risk_deployed / capital) * 100
    if portfolio_risk_pct > max_portfolio_risk:
        st.error(f"⚠️ Risk: {portfolio_risk_pct:.1f}% (OVER LIMIT)")
    else:
        st.info(f"📊 Risk: {portfolio_risk_pct:.1f}% / {max_portfolio_risk:.1f}%")
    
    pnl_pct = (st.session_state.daily_pnl / capital) * 100
    if st.session_state.goal_met:
        st.success(f"✅ Goal: +${st.session_state.daily_pnl:.2f} ({pnl_pct:.2f}%)")
    else:
        st.info(f"📈 P&L: ${st.session_state.daily_pnl:.2f} ({pnl_pct:.2f}%)")
    
    if st.session_state.consecutive_losses > 0:
        st.warning(f"⚠️ Losing Streak: {st.session_state.consecutive_losses} trades")
        st.caption("💡 After 3 losses, position size is automatically reduced by 50% to protect capital.")

    st.divider()
    st.header("🎯 Asset Selection")
    
    input_mode = st.radio(
        "Input Mode:", 
        ["VIP List", "Manual"],
        help="💡 VIP List = Pre-selected high-liquidity stocks that are safe to trade (tight spreads, high volume). Manual = Enter any ticker, but be careful with low-liquidity stocks."
    )
    
    if input_mode == "VIP List":
        ticker = st.selectbox(
            "Ticker", 
            VIP_TICKERS,
            help="💡 These are institutional favorites: high volume (>10M shares/day), tight spreads (<0.1%), safe for day trading and swing trading."
        )
    else:
        ticker = st.text_input(
            "Ticker", 
            "NVDA",
            help="💡 Enter any stock symbol (e.g., TSLA, AAPL, GME). Warning: Low-volume stocks have wide spreads and poor execution."
        ).upper()

    st.divider()
    st.header("⚙️ Strategy & Execution")
    
    strategy = st.selectbox(
        "Trading Mode", 
        ['Long (Buy)', 'Short (Sell)', 'Income (Puts)'],
        help="💡 Long = Buy stock, profit when price goes UP. Short = Sell stock, profit when price goes DOWN. Income = Sell put options, collect premium, willing to buy stock at lower price."
    )
    
    entry_mode = st.radio(
        "Entry Method", 
        ["Auto-Limit (Zone)", "Market (Now)", "Manual Override"],
        help="💡 Auto-Limit = Wait for price to reach your ideal zone (support for longs, resistance for shorts). Market = Enter immediately at current price (risky if chasing). Manual = Test custom prices for 'what if' scenarios."
    )
    
    manual_price = 0.0
    if entry_mode == "Manual Override":
        manual_price = st.number_input(
            "Entry Price ($)", 
            value=0.0, 
            step=0.01,
            help="💡 Custom entry price for testing. Example: Stock is at $500, but you want to see what the risk/reward looks like at $490."
        )
    
    stop_mode = st.selectbox(
        "Stop Width", 
        [1.0, 0.5, 0.2], 
        format_func=lambda x: f"{'Wide' if x==1.0 else 'Medium' if x==0.5 else 'Tight'} ({x} ATR)",
        help="💡 ATR = Average True Range (how much the stock moves per day). Wide stop = less risk of premature exit, but bigger loss if wrong. Tight stop = smaller loss, but more likely to get stopped out by noise. Recommended: 1.0 ATR for swing trades, 0.5 for day trades."
    )
    
    position_sizing_method = st.selectbox(
        "Position Sizing", 
        ["FIXED", "VOLATILITY_ADJUSTED", "KELLY"],
        help="💡 FIXED = Standard size based on your risk amount. VOLATILITY_ADJUSTED = Reduces size when VIX is high (market is wild). KELLY = Mathematical optimal sizing based on win rate (advanced, can be aggressive). Beginners: use FIXED."
    )
    
    premium = 0.0
    if "Income" in strategy:
        premium = st.number_input(
            "Option Premium ($)", 
            value=0.0, 
            step=0.05,
            help="💡 The per-share premium you collect for selling a put option. Example: If you sell 1 contract (100 shares) at $2.50 premium, you collect $250 cash immediately."
        )

    st.divider()
    st.header("📋 IWT Scorecard")
    st.caption("💡 IWT = 'In With Then Out' - Teri Ijeoma's 7-step system for scoring trade quality.")
    
    # Auto-suggest based on zone touches
    if st.session_state.metrics:
        m = st.session_state.metrics
        if "Long" in strategy:
            suggested_fresh = 2 if m.get('support_touches', 0) == 0 else 1 if m.get('support_touches', 0) <= 2 else 0
        else:
            suggested_fresh = 2 if m.get('resistance_touches', 0) == 0 else 1 if m.get('resistance_touches', 0) <= 2 else 0
        st.caption(f"💡 Data-Driven Suggestion: Freshness = {suggested_fresh} ({m.get('support_touches' if 'Long' in strategy else 'resistance_touches', 0)} historical touches detected)")
    
    fresh = st.selectbox(
        "1. Freshness (How Many Times Tested?)", 
        [2, 1, 0], 
        format_func=lambda x: {2:'2 Points - Fresh (Never/Once Tested)', 1:'1 Point - Used (2-3 Times)', 0:'0 Points - Stale (4+ Times)'}[x],
        help="💡 FRESH = Zone is untested or held strongly once. Orders are waiting. USED = Zone tested 2-3x, some orders filled. STALE = Zone tested 4+ times, weak, likely to break."
    )
    
    speed = st.selectbox(
        "2. Speed Out (How Fast Did Price Leave?)", 
        [2, 1, 0], 
        format_func=lambda x: {2:'2 Points - Fast (Explosive Move)', 1:'1 Point - Average', 0:'0 Points - Slow (Drifted)'}[x],
        help="💡 FAST = Price left zone with conviction (2+ ATRs in 1-5 candles). Strong hands. AVERAGE = Normal bounce. SLOW = Price barely moved, weak interest, likely to fail."
    )
    
    time_z = st.selectbox(
        "3. Time in Zone (How Long Did Price Stay?)", 
        [2, 1, 0], 
        format_func=lambda x: {2:'2 Points - Short (1-2 Candles)', 1:'1 Point - Medium (3-5 Candles)', 0:'0 Points - Long (6+ Candles)'}[x],
        help="💡 SHORT = Price touched zone and immediately reversed. Strong rejection. MEDIUM = Some indecision. LONG = Price consolidated for many candles. Weak zone, no conviction."
    )
    
    st.divider()
    
    if st.button("🔄 Reset Session", help="💡 Clears all positions, P&L, and goal status. Use this to start a new trading day."):
        st.session_state.goal_met = False
        st.session_state.daily_pnl = 0.0
        st.session_state.total_risk_deployed = 0.0
        st.session_state.open_positions = []
        st.session_state.consecutive_losses = 0
        st.success("✅ Session reset! Start fresh.")
        st.rerun()

# --- 6. MAIN UI ---
st.subheader("🌍 Global Market Intelligence")
st.caption("💡 Check macro conditions BEFORE scanning individual stocks. The market can override even perfect setups.")

col_macro, col_scan = st.columns([1, 1])

with col_macro:
    if st.button(
        "🌍 1. Scan Macro (Global Markets)", 
        use_container_width=True,
        help="💡 Fetches S&P futures, VIX (fear index), 10Y yields, gold, dollar strength, and global indices. This tells you if institutions are risk-on or risk-off."
    ):
        with st.spinner("Scanning global markets..."):
            st.session_state.macro = engine.get_macro()
            if st.session_state.macro:
                st.success("✅ Macro data loaded")
            else:
                st.error("❌ Macro fetch failed. Check internet connection.")

with col_scan:
    if st.button(
        f"🔎 2. Scan {ticker} (Technical Analysis)", 
        type="primary", 
        use_container_width=True,
        help=f"💡 Loads 1 year of data for {ticker} and calculates 15+ technical indicators: RSI, MACD, Bollinger Bands, SuperTrend, Ichimoku, ADX, patterns, divergences, support/resistance, and more."
    ):
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
                st.success(f"✅ {fname} analyzed with 15+ indicators")

# --- 7. MACRO DISPLAY WITH HELP NOTES ---
if st.session_state.macro:
    m = st.session_state.macro
    
    if m.get('data_quality') == 'DEGRADED':
        st.warning("⚠️ Data quality degraded. Using cached/estimated values.")
    
    vix_regime = engine.classify_vix_regime(m['vix'])
    regime_guide = engine.get_regime_guidance(vix_regime)
    
    market_phase, phase_desc = engine.get_market_hours_status()
    
    # VIX Regime Display
    if vix_regime in ["EXTREME", "HIGH"]:
        st.error(f"🚨 **VIX REGIME: {regime_guide['desc']} ({m['vix']:.1f})**")
        st.caption(f"💡 {regime_guide['note']}")
    elif vix_regime == "ELEVATED":
        st.warning(f"⚠️ **VIX REGIME: {regime_guide['desc']} ({m['vix']:.1f})**")
        st.caption(f"💡 {regime_guide['note']}")
    else:
        st.success(f"✅ **VIX REGIME: {regime_guide['desc']} ({m['vix']:.1f})**")
        st.caption(f"💡 {regime_guide['note']}")
    
    # Market Hours Warning
    if market_phase in ["PRE_MARKET", "AFTER_HOURS"]:
        st.warning(f"⏰ **{phase_desc}**")
    elif market_phase == "LUNCH":
        st.info(f"🍴 **{phase_desc}**")
    else:
        st.caption(f"✅ **{phase_desc}**")
    
    # Risk-Off Warning
    if m.get('risk_off'):
        st.error("🚨 **RISK-OFF REGIME DETECTED:** Gold + VIX both rising = Flight to safety. Institutions selling stocks, buying bonds/gold. Avoid aggressive longs.")
    
    # Dollar Headwind
    if m.get('dollar_headwind') and ticker in COMMODITY_TICKERS:
        st.warning("💵 **DOLLAR HEADWIND:** DXY rising = Strong dollar hurts commodities (gold, silver, oil). Foreign buyers pay more, demand drops.")
    
    # Passive Flow Analysis
    flow_strength = engine.check_passive_intensity(
        datetime.now().day, 
        st.session_state.metrics.get('rvol', 0) if st.session_state.metrics else 0
    )
    
    if flow_strength == "STRONG":
        st.success("🌊 **STRONG PASSIVE INFLOWS** (Calendar window + High volume)")
        st.caption("💡 $48 trillion in index funds rebalance on 1st-5th and 15th-20th. Volume confirms institutions are buying dips. Bullish tailwind.")
    elif flow_strength == "MODERATE":
        st.info("🌊 **MODERATE PASSIVE INFLOWS** (Calendar window + Normal volume)")
        st.caption("💡 Passive flow window is open, but volume is only moderate. Some institutional support.")
    elif flow_strength == "WEAK":
        st.warning("🌊 **WEAK PASSIVE INFLOWS** (Calendar window but Low volume)")
        st.caption("💡 Calendar suggests inflows, but low volume means retail-driven session. No institutional tailwind.")
    else:
        st.info("⏸️ **PASSIVE FLOWS NEUTRAL**")
        st.caption("💡 Not in passive flow window (1st-5th or 15th-20th of month). Market driven by active traders and headlines.")
    
    # Correlation Break
    if engine.detect_correlation_break(m):
        st.error("🌍 **GLOBAL CORRELATION BREAK:** US/Europe/Asia markets are diverging (US up, others down OR vice versa). This signals geopolitical stress, currency issues, or policy divergence. Volatility risk is elevated. Consider tighter stops or smaller size.")
    
    # Macro Dashboard
    with st.expander("🌍 Global Macro Dashboard (Click to Expand)", expanded=True):
        st.caption("💡 These show how global markets are performing TODAY. If everything is green, risk is ON. If everything is red, risk is OFF.")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        col1.metric(
            "🇺🇸 S&P 500", 
            f"{m['sp']:.2f}%",
            help="S&P 500 futures. The heartbeat of US markets. Green = risk-on, Red = risk-off."
        )
        
        col2.metric(
            "🇩🇪 DAX (Europe)", 
            f"{m['dax']:.2f}%",
            help="German stock index. Represents European equities. Should correlate with US."
        )
        
        col3.metric(
            "🇯🇵 Nikkei (Asia)", 
            f"{m['nikkei']:.2f}%",
            help="Japanese stock index. Represents Asian markets. Should correlate with US and EU."
        )
        
        col4.metric(
            "💵 DXY (Dollar Strength)", 
            f"{m['dxy']:.1f}", 
            delta=f"{m['dxy_chg']:.2f}%",
            delta_color="inverse",
            help="US Dollar Index. UP = Strong dollar = Bad for commodities and international stocks. DOWN = Weak dollar = Good for gold, oil, emerging markets."
        )
        
        col5.metric(
            "📈 US 10-Year Yield", 
            f"{m['tnx']:.2f}%", 
            delta=f"{m['tnx_chg']:.2f}%",
            delta_color="inverse",
            help="10-Year Treasury yield. UP = Rising rates = Bad for growth stocks (NVDA, TSLA). DOWN = Falling rates = Good for growth stocks."
        )

# --- 8. ASSET ANALYSIS WITH COMPREHENSIVE HELP NOTES ---
if st.session_state.data is not None:
    m = st.session_state.metrics
    df = st.session_state.data
    
    st.divider()
    st.header(f"📈 {m['name']} ({ticker})")
    st.caption("💡 Below are key metrics that tell you if the stock is in a tradeable state. Green = good, Red = caution.")
    
    # Metrics row
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    col1.metric(
        "Price", 
        f"${m['price']:.2f}",
        help="Current price (last close). This is what you'd pay if you buy right now."
    )
    
    # Enhanced Gap Display
    if abs(m['gap']) > 2.0:
        if m['rvol'] > 1.5:
            col2.metric(
                "Gap %", 
                f"{m['gap']:.2f}%", 
                delta="🚀 PRO",
                help="PROFESSIONAL GAP: Large gap (>2%) + High volume (>1.5x) = Institutional breakout. Gap is likely to HOLD and extend. Bullish if gap up, bearish if gap down."
            )
        else:
            col2.metric(
                "Gap %", 
                f"{m['gap']:.2f}%", 
                delta="⚠️ NOVICE",
                help="NOVICE GAP: Large gap (>2%) but Low volume (<1.5x) = Retail FOMO. Gap is likely to FILL (mean revert). Don't chase. Wait for pullback."
            )
    else:
        col2.metric(
            "Gap %", 
            f"{m['gap']:.2f}%",
            help="Gap = (Today's Open - Yesterday's Close) / Yesterday's Close. >2% = significant. <1% = normal."
        )
    
    vol_status = "🔥 HOT" if m['rvol'] > 1.5 else "✅ NORMAL" if m['rvol'] > 0.8 else "💤 THIN"
    col3.metric(
        "Volume (RVOL)", 
        f"{m['rvol']:.1f}x", 
        delta=vol_status,
        help="Relative Volume = Today's volume / 20-day average. >1.5x = HIGH interest (institutions present). <0.8x = LOW interest (thin, dangerous). You want >1.0x minimum."
    )
    
    trend_emoji = {"STRONG_BULL": "🚀", "BULL": "📈", "NEUTRAL": "➡️", "BEAR": "📉", "STRONG_BEAR": "🔻"}
    col4.metric(
        "Trend", 
        m['trend_strength'], 
        delta=trend_emoji.get(m['trend_strength'], "➡️"),
        help="Multi-timeframe trend (20/50/200 SMAs). STRONG_BULL = All moving averages aligned up. STRONG_BEAR = All aligned down. NEUTRAL = Choppy/sideways. Trade WITH the trend, not against it."
    )
    
    rsi_status = "⚠️OB" if m['rsi'] > 70 else "⚠️OS" if m['rsi'] < 30 else "✅"
    col5.metric(
        "RSI", 
        f"{m['rsi']:.0f}", 
        delta=rsi_status,
        help="Relative Strength Index (0-100). >70 = Overbought (potential pullback). <30 = Oversold (potential bounce). 50 = Neutral. Use with other indicators."
    )
    
    col6.metric(
        "ADX (Trend Strength)", 
        f"{m['adx']:.0f}", 
        delta="STRONG" if m['adx'] > 25 else "WEAK",
        help="Average Directional Index. >25 = Strong trending market (good for trend following). <20 = Weak/choppy (avoid trend strategies, use mean reversion instead)."
    )
    
    # Relative Strength vs SPY
    if st.session_state.macro and ticker != "SPY":
        try:
            spy_data = yf.Ticker("SPY").history(period="1mo")
            rs, rs_status = engine.calculate_relative_strength(df, spy_data)
            st.caption(f"**Relative Strength vs SPY (20-day):** {rs:+.1f}% ({rs_status})")
            st.caption(f"💡 This stock is {rs_status} the overall market. Positive = Leading (strong), Negative = Lagging (weak). Only trade stocks that are outperforming.")
        except:
            pass
    
    # Support/Resistance with touches
    st.caption(f"**Key Levels:** Support ${m['supp']:.2f} ({m.get('support_touches', 0)} touches) | Resistance ${m['res']:.2f} ({m.get('resistance_touches', 0)} touches)")
    st.caption(f"💡 Support = Floor where buyers step in. Resistance = Ceiling where sellers appear. 0 touches = Fresh zone (strong). 4+ touches = Stale zone (weak).")
    
    # ===== MULTI-INDICATOR SIGNAL DASHBOARD =====
    if st.session_state.signals:
        st.divider()
        st.subheader("🎯 Multi-Algorithm Signal Fusion (15+ Indicators)")
        st.caption("💡 This combines RSI, MACD, Bollinger Bands, Stochastic, ADX, Ichimoku, MFI, Williams %R, Moving Averages, SuperTrend, Patterns, and Divergences into ONE verdict. More signals in one direction = Higher confidence.")
        
        sig = st.session_state.signals
        
        col_bull, col_bear, col_neut = st.columns(3)
        
        with col_bull:
            st.markdown("### 🟢 Bullish Signals")
            if sig['bullish']:
                for s in sig['bullish']:
                    st.markdown(f"<div class='signal-bull'>• {s}</div>", unsafe_allow_html=True)
            else:
                st.caption("No bullish signals detected.")
        
        with col_bear:
            st.markdown("### 🔴 Bearish Signals")
            if sig['bearish']:
                for s in sig['bearish']:
                    st.markdown(f"<div class='signal-bear'>• {s}</div>", unsafe_allow_html=True)
            else:
                st.caption("No bearish signals detected.")
        
        with col_neut:
            st.markdown("### 🟡 Neutral/Watch Signals")
            if sig['neutral']:
                for s in sig['neutral']:
                    st.markdown(f"<div class='signal-neutral'>• {s}</div>", unsafe_allow_html=True)
            else:
                st.caption("No neutral signals.")
        
        # Overall Signal Score
        score = sig['score']
        if score >= 3:
            st.success(f"### 🟢 OVERALL MULTI-ALGO VERDICT: BULLISH (+{score} points)")
            st.caption("💡 Most indicators agree this is a BUY setup. However, still check IWT score and macro conditions.")
        elif score <= -3:
            st.error(f"### 🔴 OVERALL MULTI-ALGO VERDICT: BEARISH ({score} points)")
            st.caption("💡 Most indicators agree this is a SELL/SHORT setup. Or avoid if going long.")
        else:
            st.warning(f"### 🟡 OVERALL MULTI-ALGO VERDICT: NEUTRAL ({score} points)")
            st.caption("💡 Indicators are mixed. No clear direction. Wait for better setup or use smaller size.")
    
   # COMPLETE FIX - Replace the entire chart section (around lines 1115-1150)

# ===== CHART WITH HELP NOTE =====
st.divider()
st.subheader("📊 Technical Chart (Last 60 Days)")
st.caption("💡 Blue line = SMA20 (short-term trend). Orange = SMA50 (medium-term). Red = SMA200 (long-term trend). Gray bands = Bollinger Bands (volatility envelope). Green/Red horizontal lines = Support/Resistance zones.")

chart_data = df.iloc[-60:]

# Build addplots list dynamically with error handling
addplots = []

try:
    # SMAs
    if 'SMA_20' in chart_data.columns:
        addplots.append(mpf.make_addplot(chart_data['SMA_20'], color='blue', width=1.5, label='SMA 20'))
    
    if 'SMA_50' in chart_data.columns:
        addplots.append(mpf.make_addplot(chart_data['SMA_50'], color='orange', width=1.5, label='SMA 50'))
    
    if 'SMA_200' in chart_data.columns:
        addplots.append(mpf.make_addplot(chart_data['SMA_200'], color='red', width=2, label='SMA 200'))
    
    # Bollinger Bands - Find columns dynamically
    bb_cols = [col for col in chart_data.columns if 'BB' in col]
    
    bb_upper = None
    bb_lower = None
    
    # Search for upper and lower bands
    for col in bb_cols:
        if 'U' in col or 'upper' in col.lower():
            bb_upper = col
        elif 'L' in col or 'lower' in col.lower():
            bb_lower = col
    
    if bb_upper and bb_upper in chart_data.columns:
        addplots.append(mpf.make_addplot(chart_data[bb_upper], color='gray', linestyle='--', width=1, label='BB Upper'))
    
    if bb_lower and bb_lower in chart_data.columns:
        addplots.append(mpf.make_addplot(chart_data[bb_lower], color='gray', linestyle='--', width=1, label='BB Lower'))
    
    # Create the plot
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
    
except KeyError as ke:
    st.error(f"⚠️ Chart column error: Missing indicator data. Try a different ticker.")
    st.caption(f"Technical detail: {str(ke)}")
    
    # Fallback: Simple chart without indicators
    try:
        fig_simple, axes_simple = mpf.plot(
            chart_data, 
            type='candle', 
            style='yahoo', 
            volume=True,
            hlines=dict(hlines=[m['supp'], m['res']], colors=['green', 'red'], linestyle='-.', linewidths=2),
            returnfig=True, 
            figsize=(14, 8), 
            title=f"{ticker} - Basic Price Chart"
        )
        st.pyplot(fig_simple)
        st.caption("📊 Showing basic chart without technical indicators.")
    except Exception as fallback_error:
        st.error(f"❌ Unable to render chart: {str(fallback_error)}")

except Exception as e:
    st.error(f"⚠️ Chart rendering error: {str(e)}")
    st.caption("💡 This can happen with low-liquidity stocks or data gaps. Try a major ticker like AAPL or NVDA.")
    
    # ===== TRADE CALCULATION WITH COMPREHENSIVE HELP =====
    st.divider()
    st.subheader("🎯 Trade Setup Calculator")
    st.caption("💡 This calculates entry, stop, target, position size, and REAL costs (slippage + commissions). The R/R ratio tells you if the trade is mathematically worth it.")
    
    if entry_mode == "Manual Override":
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
        st.caption("💡 Entry = Where you buy/sell. Stop = Where you exit if wrong (to limit loss). Target = Where you exit if right (to take profit).")
    
    with col_setup2:
        st.markdown("**💰 Position Sizing**")
        sizing_note = f" (VIX-adjusted: {vol_multiplier:.1f}x)" if vol_multiplier != 1.0 else ""
        beta_note = f" (Beta-adjusted: β={m.get('beta', 1.0):.2f})" if m.get('beta', 1.0) != 1.0 else ""
        drawdown_note = f" (Drawdown protection: 50% size)" if st.session_state.consecutive_losses >= 3 else ""
        st.code(f"""
Size:    {shares} {'contracts' if 'Income' in strategy else 'shares'}
Risk:    ${total_trade_risk:.2f}
Reward:  ${gross_reward:.2f}
R/R:     {rr:.2f}
        """)
        st.caption(f"💡 Size calculated using {position_sizing_method} method. Risk = Maximum you lose if stopped out. R/R = Reward/Risk ratio (must be >2.0).{sizing_note}{beta_note}{drawdown_note}")
    
    with col_setup3:
        st.markdown("**💸 Real-World Costs**")
        st.code(f"""
Slippage:     ${slippage:.2f}
Commissions:  ${commissions:.2f}
Net Reward:   ${net_reward:.2f}
Net R/R:      {(net_reward/(total_trade_risk if total_trade_risk>0 else 1)):.2f}
        """)
        st.caption(f"💡 Slippage = Price movement during order (5 bps). Commissions = ${COMMISSION_PER_SHARE}/share × {shares} × 2 = ${commissions:.2f}. These reduce your profit. High-frequency trading fails due to these costs.")
    
    # ===== SCORING & VERDICT WITH COMPREHENSIVE PENALTIES =====
    st.divider()
    st.subheader("🚦 The Ultimate Verdict (IWT + Institutional Filters)")
    st.caption("💡 This combines your IWT score (Freshness + Time + Speed + R/R) with institutional penalty filters (macro, trend, signals, concentration, etc.). Final score determines if you should trade.")
    
    score_rr = 2 if rr >= 3 or ("Income" in strategy and rr > 0.1) else 1 if rr >= 2 else 0
    total_score = fresh + time_z + speed + score_rr
    
    penalties = []
    
    # Warsh penalty
    warsh_penalty = False
    if st.session_state.macro and st.session_state.macro['tnx_chg'] > 1.0 and ticker in GROWTH_TICKERS and "Long" in strategy:
        total_score -= 2
        warsh_penalty = True
        penalties.append("Warsh Penalty (-2): Rising yields >1% hurt growth stocks")
    
    # Market hours
    market_phase, _ = engine.get_market_hours_status()
    if market_phase in ["LUNCH", "PRE_MARKET", "AFTER_HOURS"]:
        total_score -= 1
        penalties.append(f"Market Hours (-1): {market_phase} (low liquidity, poor execution)")
    
    # Trend misalignment
    if "Long" in strategy and m['trend_strength'] in ["BEAR", "STRONG_BEAR"]:
        total_score -= 1
        penalties.append("Trend Misalignment (-1): Long in downtrend (fighting the trend)")
    elif "Short" in strategy and m['trend_strength'] in ["BULL", "STRONG_BULL"]:
        total_score -= 1
        penalties.append("Trend Misalignment (-1): Short in uptrend (fighting the trend)")
    
    # Sector concentration
    sector = SECTOR_MAP.get(ticker, "Unknown")
    sector_exposure = sum(1 for p in st.session_state.open_positions if SECTOR_MAP.get(p['ticker'], '') == sector)
    if sector_exposure >= 2:
        total_score -= 1
        penalties.append(f"Concentration Risk (-1): {sector_exposure+1} positions in {sector} sector (correlation risk)")
    
    # Risk-off penalty
    if st.session_state.macro and st.session_state.macro.get('risk_off') and "Long" in strategy:
        total_score -= 1
        penalties.append("Risk-Off (-1): Gold + VIX rising (flight to safety, avoid longs)")
    
    # Dollar headwind
    if st.session_state.macro and st.session_state.macro.get('dollar_headwind') and ticker in COMMODITY_TICKERS and "Long" in strategy:
        total_score -= 1
        penalties.append("Dollar Headwind (-1): DXY rising hurts commodities")
    
    # SuperTrend conflict
    if 'ST_DIR' in df.columns:
        if "Long" in strategy and df['ST_DIR'].iloc[-1] == -1:
            total_score -= 1
            penalties.append("SuperTrend Conflict (-1): Indicator is bearish")
        elif "Short" in strategy and df['ST_DIR'].iloc[-1] == 1:
            total_score -= 1
            penalties.append("SuperTrend Conflict (-1): Indicator is bullish")
    
    # RSI extreme
    if "Long" in strategy and m['rsi'] > 70:
        total_score -= 1
        penalties.append("RSI Extreme (-1): Overbought (>70), bad timing for longs")
    elif "Short" in strategy and m['rsi'] < 30:
        total_score -= 1
        penalties.append("RSI Extreme (-1): Oversold (<30), bad timing for shorts")
    
    # Multi-indicator signal conflict
    if st.session_state.signals:
        sig_score = st.session_state.signals['score']
        if "Long" in strategy and sig_score < -2:
            total_score -= 1
            penalties.append("Multi-Algo Conflict (-1): 15+ indicators are bearish")
        elif "Short" in strategy and sig_score > 2:
            total_score -= 1
            penalties.append("Multi-Algo Conflict (-1): 15+ indicators are bullish")
    
    col_verdict, col_analysis = st.columns([1, 1])
    
    with col_verdict:
        # Goal override check
        if st.session_state.goal_met:
            st.error("## 🛑 DAILY GOAL MET - STOP TRADING")
            st.markdown("<div class='risk-warning'><strong>CLOSE YOUR TERMINAL.</strong> You've hit your 1% daily target. Protect your gains. Consistency beats intensity. Come back tomorrow.</div>", unsafe_allow_html=True)
            can_trade = False
        
        # Portfolio risk check
        elif (st.session_state.total_risk_deployed + total_trade_risk) > (capital * max_portfolio_risk / 100):
            st.error("## 🛑 PORTFOLIO RISK LIMIT EXCEEDED")
            st.markdown(f"<div class='risk-warning'>Adding this trade would exceed your {max_portfolio_risk}% portfolio risk limit. You already have too much capital at risk. Close existing positions or wait.</div>", unsafe_allow_html=True)
            can_trade = False
        
        else:
            can_trade = True
            if total_score >= 7:
                st.success(f"## 🟢 GREEN LIGHT\n**Final Score: {total_score}/8**")
                st.caption("✅ **Action:** Execute with FULL confidence. All systems GO.")
            elif total_score >= 5:
                st.warning(f"## 🟡 YELLOW LIGHT\n**Final Score: {total_score}/8**")
                st.caption("⚠️ **Action:** This is tradeable, but NOT ideal. Reduce size by 50% OR wait for better confirmation. Your call.")
            else:
                st.error(f"## 🔴 RED LIGHT\n**Final Score: {total_score}/8**")
                st.caption("🛑 **Action:** DO NOT TRADE. Setup is flawed. Cash is a position. Wait for better opportunity.")
        
        if penalties:
            st.markdown("**⚠️ Penalties Applied:**")
            for p in penalties:
                st.caption(f"• {p}")
            st.caption("💡 These penalties are institutional risk filters that protect you from common mistakes.")
    
    with col_analysis:
        st.markdown("**📋 Setup Quality Checklist**")
        
        checks = []
        checks.append(("✅" if fresh == 2 else "⚠️" if fresh == 1 else "❌", f"Freshness: {['Stale (Weak)','Used (OK)','Fresh (Strong)'][fresh]}"))
        checks.append(("✅" if score_rr == 2 else "⚠️" if score_rr == 1 else "❌", f"R/R: {rr:.2f} ({['Poor (<2)', 'Acceptable (2-3)', 'Excellent (3+)'][score_rr]})"))
        checks.append(("✅" if abs(m['gap']) > 2 and m['rvol'] > 1.5 else "⚠️" if abs(m['gap']) > 2 else "➖", f"Gap: {m['gap']:.2f}% ({'Professional' if abs(m['gap']) > 2 and m['rvol'] > 1.5 else 'Novice' if abs(m['gap']) > 2 else 'Normal'})"))
        checks.append(("✅" if m['rvol'] > 1.2 else "⚠️", f"Volume: {m['rvol']:.1f}x ({'Strong' if m['rvol'] > 1.2 else 'Weak'})"))
        checks.append(("✅" if m['adx'] > 25 else "⚠️", f"Trend Strength: ADX {m['adx']:.0f} ({'Strong' if m['adx'] > 25 else 'Weak'})"))
        
        for icon, text in checks:
            st.caption(f"{icon} {text}")
    
    # ===== BACKTEST WITH HELP NOTE =====
    with st.expander("🔬 Simple Backtest (Last 60 Days) - See How This Setup Performed Historically", expanded=False):
        st.caption("💡 This simulates what would have happened if you took this EXACT setup (entry, stop, target) every time it appeared in the last 60 days. Uses simplified logic. Real trading will differ.")
        
        backtest_results = engine.simple_backtest(
            df, 
            "LONG" if "Long" in strategy else "SHORT", 
            entry, 
            stop_mode, 
            target
        )
        
        if backtest_results:
            col_bt1, col_bt2, col_bt3, col_bt4 = st.columns(4)
            col_bt1.metric("Total Trades", backtest_results['total_trades'])
            col_bt2.metric("Win Rate", f"{backtest_results['win_rate']:.1f}%", help="Percentage of trades that hit target before stop.")
            col_bt3.metric("Avg Win", f"${backtest_results['avg_win']:.2f}", help="Average profit per winning trade.")
            col_bt4.metric("Expectancy", f"${backtest_results['expectancy']:.2f}", help="Average $/trade (wins + losses combined). Must be positive to be profitable.")
            
            st.caption("⚠️ **Disclaimer:** Past performance does NOT guarantee future results. Backtests use perfect hindsight and simplified logic. Real trading involves slippage, commissions, emotions, and execution issues.")
        else:
            st.caption("No historical trades found with these exact parameters in the last 60 days.")
    
    # ===== EXECUTION WITH HELP NOTES =====
    if can_trade and total_score >= 5:
        st.divider()
        st.subheader("⚡ Trade Execution")
        st.caption("💡 PAPER = Practice trade (not real money, just logged for analysis). LIVE = Real trade (logged and tracked in your open positions).")
        
        col_exec1, col_exec2 = st.columns(2)
        
        with col_exec1:
            if st.button(
                "📝 Log as PAPER TRADE", 
                use_container_width=True, 
                type="secondary",
                help="💡 Logs this setup to your journal for analysis, but doesn't count toward P&L or portfolio risk. Use this to practice without real money."
            ):
                trade_record = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ticker": ticker, "action": strategy, "entry": entry, "stop": stop, "target": target,
                    "shares": shares, "score": total_score, "risk": total_trade_risk,
                    "expected_reward": gross_reward, "net_reward": net_reward, "rr_ratio": rr,
                    "slippage": slippage, "commissions": commissions, "status": "PAPER"
                }
                st.session_state.journal.append(trade_record)
                st.success("📋 Paper trade logged! Check journal below.")
        
        with col_exec2:
            if st.button(
                "💵 LOG AS LIVE TRADE", 
                use_container_width=True, 
                type="primary",
                help="💡 Logs this as a REAL trade. Adds to open positions, counts toward portfolio risk, and will affect your P&L when you close it. Only click if you're actually entering this trade in your broker."
            ):
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
                st.success("✅ Live position logged! Now showing in Open Positions.")
                st.rerun()

# --- 9. POSITION MANAGEMENT WITH HELP NOTES ---
if st.session_state.open_positions:
    st.divider()
    st.subheader("📊 Open Positions (Live Trades)")
    st.caption("💡 These are trades you've logged as LIVE. You're currently risking real capital on these. Close them when you exit in your broker.")
    
    positions_df = pd.DataFrame(st.session_state.open_positions)
    positions_df = positions_df[['ticker', 'action', 'entry', 'stop', 'target', 'shares', 'risk', 'score']]
    st.dataframe(positions_df, use_container_width=True)
    
    st.markdown("**Close a Position:**")
    st.caption("💡 Enter the actual exit price from your broker. System will calculate real P&L (including slippage + commissions).")
    
    col_close1, col_close2, col_close3 = st.columns(3)
    
    with col_close1:
        position_to_close = st.selectbox(
            "Select Position", 
            [p['ticker'] for p in st.session_state.open_positions],
            help="Choose which position you want to close."
        )
    
    with col_close2:
        exit_price = st.number_input(
            "Exit Price ($)", 
            value=0.0, 
            step=0.01,
            help="The actual price you sold at in your broker. Example: If you bought at $500 and sold at $510, enter 510."
        )
    
    with col_close3:
        if st.button("✅ Close Position", help="Finalizes the trade, calculates P&L, updates your session total, and moves trade to closed history."):
            if exit_price > 0:
                for i, pos in enumerate(st.session_state.open_positions):
                    if pos['ticker'] == position_to_close:
                        if "Long" in pos['action']:
                            actual_pnl = (exit_price - pos['entry']) * pos['shares']
                        else:
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
                st.error("❌ Please enter a valid exit price (greater than $0).")

# --- 10. PERFORMANCE ANALYTICS WITH COMPREHENSIVE HELP NOTES ---
if st.session_state.closed_trades:
    st.divider()
    st.subheader("📈 Performance Analytics Dashboard")
    st.caption("💡 These metrics show you how good your trading system actually is. Track these over time to see if you're improving.")
    
    closed_df = pd.DataFrame(st.session_state.closed_trades)
    
    col_stats1, col_stats2, col_stats3, col_stats4, col_stats5 = st.columns(5)
    
    wins = len(closed_df[closed_df['actual_pnl'] > 0])
    total_trades = len(closed_df)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    col_stats1.metric(
        "Win Rate", 
        f"{win_rate:.1f}%", 
        delta=f"{wins}/{total_trades} trades",
        help="Percentage of trades that were profitable. >50% is good with 2:1 R/R. >60% is excellent."
    )
    
    avg_rr = closed_df['rr_ratio'].mean()
    col_stats2.metric(
        "Avg R/R", 
        f"{avg_rr:.2f}",
        help="Average Risk/Reward ratio of executed trades. >2.0 means you make 2x what you risk. Higher is better."
    )
    
    total_pnl = closed_df['actual_pnl'].sum()
    col_stats3.metric(
        "Total P&L", 
        f"${total_pnl:.2f}", 
        delta=f"{(total_pnl/capital)*100:.2f}%",
        help="Total profit/loss from all closed trades. Positive = making money. Negative = losing money. % shows return on capital."
    )
    
    sharpe = engine.calculate_sharpe_ratio(st.session_state.closed_trades)
    if sharpe:
        sharpe_quality = "Excellent" if sharpe > 2 else "Good" if sharpe > 1 else "Poor"
        col_stats4.metric(
            "Sharpe Ratio", 
            f"{sharpe:.2f}", 
            delta=sharpe_quality,
            help="Risk-adjusted return. Measures return per unit of risk taken. >1 is good, >2 is excellent, >3 is elite (top 1% of traders). Accounts for volatility of returns."
        )
    else:
        col_stats4.metric("Sharpe Ratio", "N/A", help="Need at least 5 closed trades to calculate Sharpe Ratio.")
    
    profit_factor = engine.calculate_profit_factor(st.session_state.closed_trades)
    if profit_factor:
        pf_quality = "Excellent" if profit_factor > 2 else "Good" if profit_factor > 1.5 else "Poor"
        col_stats5.metric(
            "Profit Factor", 
            f"{profit_factor:.2f}", 
            delta=pf_quality,
            help="Gross Profit / Gross Loss. >1.5 = profitable system. >2.0 = very good. <1.0 = losing money. Example: If you made $3000 on winners and lost $1500 on losers, PF = 3000/1500 = 2.0."
        )
    else:
        col_stats5.metric("Profit Factor", "N/A", help="Need winning AND losing trades to calculate.")
    
    # Additional stats
    col_stats6, col_stats7, col_stats8 = st.columns(3)
    
    expectancy = engine.calculate_expectancy(st.session_state.closed_trades)
    col_stats6.metric(
        "Expectancy", 
        f"${expectancy:.2f}",
        help="Average $/trade across all trades (winners + losers). Must be positive to be profitable long-term. Example: $50 expectancy × 100 trades/year = $5,000 annual profit."
    )
    
    max_dd = engine.calculate_max_drawdown(st.session_state.closed_trades)
    col_stats7.metric(
        "Max Drawdown", 
        f"${max_dd:.2f}",
        help="Largest peak-to-trough decline in your equity curve. This is your worst-case scenario. Example: If you were up $1000, then lost $400, your max drawdown is $400. Keep this under 20% of capital."
    )
    
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
    
    col_stats8.metric(
        "Max Streak", 
        f"W:{max_consec_wins} / L:{max_consec_losses}",
        help="Longest winning streak and longest losing streak. Helps you understand your psychological limits. If you can't handle 5 losses in a row, you need to work on mental discipline."
    )
    
    with st.expander("📋 Complete Trade History", expanded=False):
        st.caption("💡 Full record of all closed trades with entry, exit, P&L, costs, and score. Download as CSV for analysis in Excel.")
        history_df = closed_df[['timestamp', 'ticker', 'action', 'entry', 'exit_price', 'actual_pnl', 'score', 'slippage', 'commissions']]
        st.dataframe(history_df, use_container_width=True)

# --- 11. JOURNAL EXPORT WITH HELP NOTES ---
if st.session_state.journal:
    st.divider()
    st.subheader("📓 Trading Journal (All Trades)")
    st.caption("💡 This contains BOTH paper trades (practice) and live trades (real). Review this regularly to find patterns in your mistakes and successes.")
    
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
            use_container_width=True,
            help="💡 Downloads all trades (paper + live) as CSV file. Open in Excel or Google Sheets for deeper analysis."
        )
    
    with col_export2:
        if st.session_state.closed_trades:
            closed_csv = pd.DataFrame(st.session_state.closed_trades).to_csv(index=False).encode('utf-8')
            st.download_button(
                "📊 Download Performance Report (CSV)",
                data=closed_csv,
                file_name=f"performance_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv",
                use_container_width=True,
                help="💡 Downloads only CLOSED trades with actual P&L. Use this for tax records and performance analysis."
            )

else:
    st.info("👈 **Quick Start Guide:** 1. Scan Macro (check global markets) → 2. Scan a Ticker/Analysis (technical analysis) → 3. Review Multi-Algo Signals → 4. Check/Enter IWT Score + Penalties → 5. Log Paper or Live Trade")
    
    with st.expander("❓ New to Trading? Read This First", expanded=False):
        st.markdown("""
        ### 🎓 Beginner's Guide to Using Quantum Maestro
        
        **Step 1: Understand Your Risk**
        - Set "Total Capital" to your ACTUAL account size
        - Set "Risk per Trade" to 1% of capital ($100 on $10,000 account)
        - Never risk more than 2% per trade
        
        **Step 2: Check Macro Conditions FIRST**
        - Click "Scan Macro" before looking at individual stocks
        - If VIX is HIGH or EXTREME → Reduce size or don't trade
        - If Risk-Off (Gold+VIX rising) → Avoid aggressive longs
        
        **Step 3: Scan a Stock**
        - Choose from VIP List (safest) or enter your own ticker
        - Wait for data to load (15+ indicators calculated)
        
        **Step 4: Review Signals**
        - Green signals = Bullish, Red = Bearish, Yellow = Neutral
        - Overall score ≥+3 = Strong buy bias, ≤-3 = Strong sell bias
        
        **Step 5: Score the IWT Setup**
        - Freshness: Is the zone fresh (untested) or stale (weak)?
        - Time in Zone: Did price reject quickly or linger?
        - Speed Out: Did price leave fast or slow?
        - R/R: Must be ≥2.0 (risk $100 to make $200+)
        
        **Step 6: Check Final Verdict**
        - 7-8 points = GREEN LIGHT (trade with confidence)
        - 5-6 points = YELLOW LIGHT (reduce size or wait)
        - 0-4 points = RED LIGHT (don't trade)
        
        **Step 7: Execute (Paper First!)**
        - Start with PAPER trades (practice, no real money)
        - Once you're profitable on paper for 20+ trades, try LIVE
        - Track your performance (win rate, Sharpe, profit factor)
        
        **Golden Rules:**
        1. Stop trading when you hit your 1% daily goal
        2. Never trade more than 3 positions at once
        3. Don't fight the trend (trade WITH moving averages)
        4. High VIX = smaller size or sit out
        5. Journal EVERY trade (winners and losers)
        
        **Resources:**
        - Teri Ijeoma's IWT: www.tradewithteri.com
        - Technical Analysis: "Technical Analysis of Stock Trends" by Edwards & Magee
        - Risk Management: "Trade Your Way to Financial Freedom" by Van Tharp
        """)

st.divider()
st.caption("🏛️ Quantum  Maestro  Financial  Markets TradingBot | Multi-Algorithm Fusion | Educational Use Only")
st.caption("💡 Not financial advice. For educational and simulation purposes. Trading involves substantial risk of loss.")
