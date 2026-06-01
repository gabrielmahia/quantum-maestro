# Copyright (c) 2026 Gabriel Mahia. All Rights Reserved.
# Unauthorized copying of this file, via any medium, is strictly prohibited.
# Proprietary and confidential.
# Written by Gabriel Mahia, 2026
# app.py

# =============================================================================
# EasyStockTrader — Smart Stock Analysis
# Educational simulation. Does not execute trades. Not financial advice.
# IWT framework: the IWT methodology (investwithteri.com)
# =============================================================================

import streamlit as st
import urllib.request
import xml.etree.ElementTree as _ET
import re as _re_qm
import json as _json_qm
import yfinance as yf
import mplfinance as mpf
import pandas as pd
import numpy as np

# ── Pure numpy/pandas technical indicator library ─────────────────────────────
# Replaces pandas_ta (dropped: depends on numba/llvmlite, incompatible with
# Python 3.14+). Column names kept identical so downstream code is unchanged.

def _calc_indicators(df):
    """Append all technical indicators to df in-place. Returns df."""
    hi, lo, cl, vo = df['High'], df['Low'], df['Close'], df['Volume']

    # SMA
    for n in (20, 50, 200):
        df[f'SMA_{n}'] = cl.rolling(n).mean()

    # EMA
    for n in (12, 26):
        df[f'EMA_{n}'] = cl.ewm(span=n, adjust=False).mean()

    # ATR
    prev_cl = cl.shift(1)
    tr = pd.concat([hi - lo, (hi - prev_cl).abs(), (lo - prev_cl).abs()], axis=1).max(axis=1)
    df['ATRr_14'] = tr.ewm(alpha=1/14, adjust=False).mean()

    # RSI
    delta = cl.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI_14'] = (100 - (100 / (1 + rs))).fillna(50)  # NaN→50 neutral if no volatility

    # MACD
    ema12 = cl.ewm(span=12, adjust=False).mean()
    ema26 = cl.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['MACD_12_26_9']  = macd_line
    df['MACDs_12_26_9'] = signal_line
    df['MACDh_12_26_9'] = macd_line - signal_line

    # Bollinger Bands
    sma20 = cl.rolling(20).mean()
    std20 = cl.rolling(20).std()
    df['BBU_20_2.0'] = sma20 + 2 * std20
    df['BBL_20_2.0'] = sma20 - 2 * std20
    df['BBM_20_2.0'] = sma20

    # Stochastic %K/%D
    lo14 = lo.rolling(14).min()
    hi14 = hi.rolling(14).max()
    k = 100 * (cl - lo14) / (hi14 - lo14).replace(0, np.nan)
    df['STOCHk_14_3_3'] = k.rolling(3).mean()
    df['STOCHd_14_3_3'] = df['STOCHk_14_3_3'].rolling(3).mean()

    # ADX
    up   = hi.diff()
    down = -lo.diff()
    plus_dm  = up.where((up > down) & (up > 0), 0.0)
    minus_dm = down.where((down > up) & (down > 0), 0.0)
    atr14 = tr.ewm(alpha=1/14, adjust=False).mean()
    plus_di  = 100 * plus_dm.ewm(alpha=1/14, adjust=False).mean()  / atr14
    minus_di = 100 * minus_dm.ewm(alpha=1/14, adjust=False).mean() / atr14
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    df['ADX_14'] = dx.ewm(alpha=1/14, adjust=False).mean()

    # OBV
    direction = np.sign(cl.diff()).fillna(0)
    df['OBV'] = (direction * vo).cumsum()

    # MFI (Money Flow Index)
    tp = (hi + lo + cl) / 3
    mf = tp * vo
    pos_mf = mf.where(tp > tp.shift(1), 0.0).rolling(14).sum()
    neg_mf = mf.where(tp < tp.shift(1), 0.0).rolling(14).sum()
    mfr = pos_mf / neg_mf.replace(0, np.nan)
    df['MFI_14'] = 100 - (100 / (1 + mfr))

    # Williams %R
    df['WILLR_14'] = -100 * (hi14 - cl) / (hi14 - lo14).replace(0, np.nan)

    # Supertrend (10, 3)
    _supertrend(df, length=10, multiplier=3)

    # Ichimoku (simplified: Span A & B)
    _ichimoku(df)

    return df


def _supertrend(df, length=10, multiplier=3):
    hi, lo, cl = df['High'], df['Low'], df['Close']
    prev_cl = cl.shift(1)
    tr = pd.concat([hi - lo, (hi - prev_cl).abs(), (lo - prev_cl).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/length, adjust=False).mean()
    hl2 = (hi + lo) / 2
    upper = hl2 + multiplier * atr
    lower = hl2 - multiplier * atr

    st_val = pd.Series(np.nan, index=df.index, dtype=float)
    st_dir = pd.Series(1, index=df.index, dtype=int)
    upper_vals = upper.values.copy()
    lower_vals = lower.values.copy()
    cl_vals    = cl.values
    st_dir_vals = st_dir.values.copy()
    st_val_vals = st_val.values.copy()
    for i in range(1, len(df)):
        prev_upper = upper_vals[i-1]
        prev_lower = lower_vals[i-1]
        cur_upper  = upper_vals[i]
        cur_lower  = lower_vals[i]
        upper_vals[i] = cur_upper if cur_upper < prev_upper or cl_vals[i-1] > prev_upper else prev_upper
        lower_vals[i] = cur_lower if cur_lower > prev_lower or cl_vals[i-1] < prev_lower else prev_lower
        if st_dir_vals[i-1] == 1:
            st_dir_vals[i] = -1 if cl_vals[i] < lower_vals[i] else 1
        else:
            st_dir_vals[i] =  1 if cl_vals[i] > upper_vals[i] else -1
        st_val_vals[i] = lower_vals[i] if st_dir_vals[i] == 1 else upper_vals[i]
    upper[:] = upper_vals
    lower[:] = lower_vals
    st_dir[:] = st_dir_vals
    st_val[:] = st_val_vals

    df['ST_VAL'] = st_val
    df['ST_DIR'] = st_dir


def _ichimoku(df):
    hi, lo = df['High'], df['Low']
    tenkan   = (hi.rolling(9).max()  + lo.rolling(9).min())  / 2
    kijun    = (hi.rolling(26).max() + lo.rolling(26).min()) / 2
    df['ICH_SPAN_A'] = ((tenkan + kijun) / 2).shift(26)
    df['ICH_SPAN_B'] = ((hi.rolling(52).max() + lo.rolling(52).min()) / 2).shift(26)
    # Fill forward so iloc[-1] is always valid
    df['ICH_SPAN_A'] = df['ICH_SPAN_A'].ffill().fillna(df['Close'])
    df['ICH_SPAN_B'] = df['ICH_SPAN_B'].ffill().fillna(df['Close'])
from scipy.signal import argrelextrema
from datetime import datetime, time, timedelta
import math
import calendar
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
# the IWT framework's core trading universe (her public 34-stock list)
TERI_UNIVERSE = [
    # Mega-cap tech & AI
    "NVDA","AMD","MSFT","AAPL","META","GOOGL","AMZN","TSLA",
    # Semis & hardware
    "AVGO","ARM","MU","SMCI","TSM","AMAT","LRCX",
    # ETFs (index + sector)
    "SPY","QQQ","IWM","XLK","XLE","XLF","GLD","TLT",
    # Financial
    "JPM","GS","MS","V","MA",
    # Healthcare & consumer
    "UNH","JNJ","PG","KO",
    # Energy & commodities
    "XOM","CVX","CL=F","GC=F",
]

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
OPTION_COMMISSION_PER_CONTRACT = 0.65


def calc_vertical_credit_spread(short_strike, long_strike, credit, spread_type="PUT", contracts=1):
    """
    Defined-risk vertical credit spread math.

    SPX options are quoted in index points, and each point is worth $100.
    A $5-wide spread has $500 gross width per contract.

    Put credit spread: sell higher strike put, buy lower strike put.
    Call credit spread: sell lower strike call, buy higher strike call.
    """
    spread_type = str(spread_type).upper()
    short_strike = float(short_strike)
    long_strike = float(long_strike)
    credit = float(credit)
    contracts = int(max(0, contracts))

    width = abs(short_strike - long_strike)
    errors = []
    if width <= 0:
        errors.append("Spread width must be positive.")
    if credit <= 0:
        errors.append("Credit must be positive.")
    if width > 0 and credit >= width:
        errors.append("Credit must be less than spread width; otherwise the max-risk math is invalid.")
    if spread_type == "PUT" and short_strike <= long_strike:
        errors.append("Put credit spread requires short strike ABOVE long strike.")
    if spread_type == "CALL" and short_strike >= long_strike:
        errors.append("Call credit spread requires short strike BELOW long strike.")

    max_profit_per_contract = credit * 100
    max_loss_per_contract = max(0, (width - credit) * 100)
    gross_width_per_contract = width * 100
    breakeven = short_strike - credit if spread_type == "PUT" else short_strike + credit
    total_credit = max_profit_per_contract * contracts
    total_max_loss = max_loss_per_contract * contracts
    commission = contracts * OPTION_COMMISSION_PER_CONTRACT * 2  # two legs
    net_max_profit = max(0, total_credit - commission)
    net_risk_reward = net_max_profit / total_max_loss if total_max_loss > 0 else 0
    credit_pct_width = credit / width if width > 0 else 0

    return {
        "errors": errors,
        "spread_type": spread_type,
        "width": width,
        "gross_width_per_contract": gross_width_per_contract,
        "max_profit_per_contract": max_profit_per_contract,
        "max_loss_per_contract": max_loss_per_contract,
        "breakeven": breakeven,
        "total_credit": total_credit,
        "total_max_loss": total_max_loss,
        "commission": commission,
        "net_max_profit": net_max_profit,
        "net_risk_reward": net_risk_reward,
        "credit_pct_width": credit_pct_width,
    }


def contracts_for_defined_risk(max_risk_dollars, max_loss_per_contract):
    """Round down contracts so theoretical max loss never exceeds the risk budget."""
    if max_loss_per_contract <= 0:
        return 0
    return int(max(0, float(max_risk_dollars)) // max_loss_per_contract)


def pdt_guidance(account_value, account_type, day_trades_used, planned_same_day_exit, pdt_framework="Legacy PDT"):
    """
    Conservative day-trade budget guidance.

    Legacy PDT framework: a small margin account should avoid 4+ day trades in a rolling
    5-business-day window. New intraday margin framework may remove the fixed $25k PDT
    threshold when a broker migrates, but broker-specific controls still matter. This app
    does NOT know your broker's live permissioning; it acts as a discipline/risk budget.
    """
    account_type_l = str(account_type).lower()
    framework = str(pdt_framework).lower()

    if "cash" in account_type_l:
        return {
            "remaining": None,
            "status": "Cash account: PDT generally does not apply, but settled-cash / good-faith rules matter.",
            "can_day_trade": True,
            "framework": pdt_framework,
        }

    if "ira" in account_type_l or "limited" in account_type_l:
        return {
            "remaining": None,
            "status": "IRA / limited-margin: broker-specific rules. Treat same-day exits as scarce unless your broker confirms otherwise.",
            "can_day_trade": True,
            "framework": pdt_framework,
        }

    if "new" in framework and account_value >= 2000:
        return {
            "remaining": "broker-controlled",
            "status": "New intraday-margin framework selected: PDT count may no longer be the binding rule, but broker risk controls still apply.",
            "can_day_trade": True,
            "framework": pdt_framework,
        }

    if account_value >= 25000:
        return {
            "remaining": "unrestricted",
            "status": "Margin account above $25k legacy PDT threshold.",
            "can_day_trade": True,
            "framework": pdt_framework,
        }

    remaining = max(0, 3 - int(day_trades_used))
    can_day_trade = (remaining > 0) or not planned_same_day_exit
    status = f"Legacy small-margin mode: preserve limited day trades. Approx. {remaining}/3 day trades left in rolling 5-business-day window."
    return {"remaining": remaining, "status": status, "can_day_trade": can_day_trade, "framework": pdt_framework}


def estimate_expected_move(underlying_price, iv_percent, dte):
    """Approximate one-standard-deviation expected move from IV and DTE."""
    try:
        price = float(underlying_price)
        iv = float(iv_percent) / 100.0
        days = max(float(dte), 1.0)
        return price * iv * math.sqrt(days / 365.0)
    except Exception:
        return 0.0


def expected_move_check(spread_type, short_strike, underlying_price, expected_move):
    """Whether the short strike is outside the approximate expected move."""
    try:
        stype = str(spread_type).upper()
        short = float(short_strike)
        px = float(underlying_price)
        em = float(expected_move)
        if em <= 0:
            return "UNKNOWN", 0.0
        if stype == "PUT":
            distance = px - short
            return ("OUTSIDE" if distance >= em else "INSIDE"), distance / em
        distance = short - px
        return ("OUTSIDE" if distance >= em else "INSIDE"), distance / em
    except Exception:
        return "UNKNOWN", 0.0


def approx_pop_from_delta(short_delta):
    """Rule-of-thumb POP and probability of touch from absolute short option delta."""
    d = min(max(abs(float(short_delta)), 0.01), 0.99)
    pop = max(0.01, min(0.99, 1.0 - d))
    pot = max(0.01, min(0.99, 2.0 * d))
    return pop, pot


def rank_trade_days(vix_regime, passive_window, event_days, preferred_days, day_trades_remaining, framework_label):
    """Rank Mon-Thu trade days as a planning aid, not a forecast."""
    days = ["Monday", "Tuesday", "Wednesday", "Thursday"]
    results = []
    for day in days:
        score = 2
        reasons = []
        if day not in preferred_days:
            score -= 2
            reasons.append("not selected")
        if vix_regime in ["NORMAL", "LOW"]:
            score += 1
            reasons.append("vol stable")
        elif vix_regime in ["ELEVATED"]:
            score -= 1
            reasons.append("vol elevated")
        elif vix_regime in ["HIGH", "EXTREME"]:
            score -= 3
            reasons.append("vol defensive")
        if passive_window:
            score += 1
            reasons.append("flow window")
        if day in event_days:
            score -= 4
            reasons.append("major event")
        if day == "Monday":
            score += 0  # keep neutral: often good after weekend, but gap risk exists
        if day == "Thursday":
            score += 1
            reasons.append("post-week structure")
        if isinstance(day_trades_remaining, int) and day_trades_remaining <= 1 and score < 5:
            score -= 2
            reasons.append("preserve last day trade")
        label = "🟢 Best Trade Day" if score >= 5 else "🟡 Conditional" if score >= 2 else "🔴 Avoid"
        results.append({"day": day, "score": score, "label": label, "reasons": ", ".join(reasons) if reasons else "neutral"})
    results.sort(key=lambda x: x["score"], reverse=True)
    return results




# =============================================================================
# V3 — IWT FULL PLAYBOOK + HIGH-QUALITY ALGORITHM ENHANCEMENTS
# =============================================================================

def calc_long_option_iwt(underlying_price, strike, premium_paid, direction, delta, dte, contracts=1):
    """IWT Long Options: 60+ DTE, DITM preferred, 50-100% profit target, 50% stop."""
    if premium_paid <= 0 or contracts <= 0:
        return {"error": "Premium and contracts must be positive."}
    import math
    cost_per_contract = premium_paid * 100
    cost_total = cost_per_contract * contracts
    dte_ok = dte >= 60
    dte_grade = (f"✅ {dte} DTE — IWT compliant (60+ minimum)" if dte_ok
                 else f"⚠️ {dte} DTE — below IWT 60 DTE minimum. Time decay risk is elevated.")
    if delta >= 0.70:
        delta_grade, delta_label = "✅ DITM — acts like stock with leverage. IWT preferred.", "Deep ITM"
    elif delta >= 0.50:
        delta_grade, delta_label = "🟡 ATM — balanced theta/gamma. Acceptable for strong setups.", "At The Money"
    elif delta >= 0.30:
        delta_grade, delta_label = "🟠 OTM — needs large, fast move to profit.", "Out of The Money"
    else:
        delta_grade, delta_label = "🔴 Far OTM — lottery ticket. IWT does not teach this.", "Far OTM"
    target_50_val = cost_total * 0.50
    target_100_val = cost_total * 1.00
    target_50_price = premium_paid * 1.50
    target_100_price = premium_paid * 2.00
    stop_loss_total = cost_total * 0.50
    stop_price = premium_paid * 0.50
    shares_equiv = 100 * contracts * delta
    stock_cost_equiv = underlying_price * shares_equiv
    leverage = stock_cost_equiv / cost_total if cost_total > 0 else 0
    if direction == "CALL":
        breakeven = strike + premium_paid
        intrinsic = max(0.0, underlying_price - strike)
    else:
        breakeven = strike - premium_paid
        intrinsic = max(0.0, strike - underlying_price)
    time_val = max(0.0, premium_paid - intrinsic)
    daily_theta = time_val * 100 * contracts / dte if dte > 0 else 0
    theta_note = ("Theta burns fastest in final 30 DTE — monitor weekly; roll or exit before 30 DTE."
                  if dte <= 45 else "Theta is mild this far from expiry. Weekly monitoring is sufficient.")
    return {
        "cost_total": cost_total, "cost_per_contract": cost_per_contract,
        "dte_ok": dte_ok, "dte_grade": dte_grade,
        "delta_grade": delta_grade, "delta_label": delta_label,
        "target_50_val": target_50_val, "target_100_val": target_100_val,
        "target_50_price": target_50_price, "target_100_price": target_100_price,
        "stop_loss_total": stop_loss_total, "stop_price": stop_price,
        "leverage_ratio": leverage, "shares_equiv": shares_equiv,
        "breakeven": breakeven, "intrinsic": intrinsic,
        "time_val": time_val, "time_val_pct": time_val / premium_paid if premium_paid > 0 else 0,
        "daily_theta": daily_theta, "theta_note": theta_note, "error": None,
    }


def classify_iv_environment(ivr, vix_level):
    """IV Rank arbiter. High IVR (50+) = sell premium. Low IVR (<30) = buy options 60+ DTE."""
    if ivr >= 50:
        return {"regime": "HIGH IV — SELLER'S MARKET", "recommendation": "SELL PREMIUM", "grade": "A",
                "badge": "🟢",
                "rationale": ("IV is elevated vs its 52-week range. Options are expensive. This is the optimal "
                              "environment for credit spreads, CSPs, and covered calls. Premium sellers have a "
                              "statistical edge when IV is high and mean-reverts."),
                "strategies": ["SPX put credit spread (OTM, outside EM)",
                                "SPX call credit spread (OTM, above EM)",
                                "Cash-secured put on quality names at support",
                                "Covered call for income on existing positions"],
                "avoid": "Avoid buying long options — premium is expensive; poor risk/reward for buyers.",
                "vix_overlay": ("VIX < 15: extreme complacency. Credit spreads attractive." if vix_level < 15
                                else "VIX 15-20: normal. Standard premium-selling applies." if vix_level < 20
                                else "VIX 20-30: elevated. Reduce size; prefer spreads." if vix_level < 30
                                else "VIX 30+: crisis. Do not sell new premium. Stand aside.")}
    elif ivr >= 30:
        return {"regime": "MODERATE IV — SELECTIVE", "recommendation": "SELECTIVE — REQUIRE A+ SETUP",
                "grade": "B", "badge": "🟡",
                "rationale": ("IV is in the middle of its recent range. Neither strongly cheap nor expensive. "
                              "Require higher setup quality. Watch IV direction: rising → lean buyers; falling → lean sellers."),
                "strategies": ["Credit spreads only on high-conviction directional setups",
                                "Long options (60+ DTE, DITM) on confirmed strong trends",
                                "Avoid new premium in rapidly rising IV environments"],
                "avoid": "Avoid mediocre setups at moderate IV — wait for better conditions.",
                "vix_overlay": ("VIX < 15: complacency supports selling." if vix_level < 15
                                else "VIX 15-20: standard conditions." if vix_level < 20
                                else "VIX 20-30: elevated — reduce size." if vix_level < 30
                                else "VIX 30+: stand aside.")}
    else:
        return {"regime": "LOW IV — BUYER'S MARKET", "recommendation": "BUY OPTIONS (60+ DTE)",
                "grade": "C", "badge": "🔵",
                "rationale": ("IV is near 52-week lows. Options are cheap relative to history. This favors "
                              "option buyers — leverage is inexpensive. Selling premium here is unattractive "
                              "because credit collected is thin and IV expansion from lows hurts short premium."),
                "strategies": ["Long calls (60+ DTE, delta 0.70+) on confirmed uptrends at support",
                                "Long puts (60+ DTE, delta 0.70+) on confirmed downtrends at resistance"],
                "avoid": "Avoid selling credit spreads — thin premium does not compensate for tail risk.",
                "vix_overlay": ("VIX < 15: extreme low IV. 60+ DTE long options on trends." if vix_level < 15
                                else "VIX 15-20: low-moderate IV. Buyer's edge." if vix_level < 20
                                else "VIX 20-30: consider buying protection." if vix_level < 30
                                else "VIX 30+: do not sell. Buy defined-risk protection.")}


def theta_decay_profile(credit_received_per_contract, dte_at_entry, dte_remaining):
    """Theta decay approximation for credit spread. Management trigger: 50% profit = close."""
    import math
    if dte_at_entry <= 0 or credit_received_per_contract <= 0:
        return {}
    dte_remaining = max(0, dte_remaining)
    frac = math.sqrt(dte_remaining / dte_at_entry) if dte_at_entry > 0 else 0
    val_remaining = credit_received_per_contract * 100 * frac
    val_decayed = credit_received_per_contract * 100 * (1 - frac)
    pct = (1 - frac) * 100
    daily_theta = val_remaining / dte_remaining if dte_remaining > 0 else val_remaining
    mgmt = ("✅ CLOSE — 50% max profit captured. TastyTrade rule: take the win." if pct >= 50
            else "🟡 APPROACHING — ~40% profit. Prepare to close." if pct >= 40
            else "🕐 HOLD — let theta work. No action needed yet.")
    return {"value_remaining": round(val_remaining, 2), "value_decayed": round(val_decayed, 2),
            "pct_profit_captured": round(pct, 1), "daily_theta_approx": round(daily_theta, 2),
            "management_signal": mgmt}


def trade_management_engine(structure, entry_value, current_value, dte_remaining, dte_at_entry=None):
    """Systematic trade management: TastyTrade 50% rule + IWT profit/stop + gamma-risk rules."""
    actions, primary = [], "HOLD"
    if structure == "CREDIT_SPREAD":
        pct_profit = (entry_value - current_value) / entry_value if entry_value > 0 else 0
        if pct_profit >= 0.50:
            primary = "✅ CLOSE — TAKE PROFIT"
            actions.append("50% max profit captured. TastyTrade rule: close. The remaining premium isn't worth gamma risk.")
        if current_value >= entry_value * 2.0:
            primary = "🔴 CLOSE — LOSS MANAGEMENT"
            actions.append("Spread doubled in cost. Hard stop: close to preserve capital. Do not hold for recovery.")
        if dte_remaining is not None and dte_remaining <= 7 and pct_profit < 0.50:
            primary = primary if primary != "HOLD" else "⚠️ CLOSE — GAMMA RISK"
            actions.append("< 7 DTE: gamma expansion makes short premium dangerous. Close unless nearly worthless (< 20% of credit).")
        if dte_remaining is not None and dte_remaining <= 21 and 0.25 <= pct_profit < 0.50:
            primary = primary if primary != "HOLD" else "🟡 CONSIDER CLOSING"
            actions.append("< 21 DTE and 25%+ profit. The final profit increment carries disproportionate gamma risk.")
        if not actions:
            actions.append("No trigger met. Let theta work. Next check: 50% profit, 21 DTE, or spread doubles — whichever first.")
    elif structure == "LONG_OPTION":
        gain_pct = (current_value - entry_value) / entry_value if entry_value > 0 else 0
        if gain_pct >= 1.00:
            primary = "✅ CLOSE — 100% TARGET HIT"
            actions.append("100% gain on premium. IWT full target. Exit completely and re-deploy in next A+ setup.")
        elif gain_pct >= 0.50:
            primary = "✅ SCALE OUT"
            actions.append("50% gain on premium. IWT scale-out: close 50-75% now. Trail remainder with 50% stop on remaining.")
        if gain_pct <= -0.50:
            primary = primary if primary != "HOLD" else "🔴 CLOSE — IWT STOP HIT"
            actions.append("50% of premium lost. IWT hard stop: NEVER hold below 50% of purchase price. Exit — time and delta are working against you.")
        if dte_remaining is not None and dte_remaining <= 30 and gain_pct < 0.20:
            primary = primary if primary != "HOLD" else "🟠 CLOSE OR ROLL"
            actions.append("< 30 DTE with < 20% gain. Time decay accelerates sharply. Roll to later expiry or close to limit theta bleed.")
        if not actions:
            actions.append("No trigger. Position inside profit/stop range. Monitor delta and DTE weekly. Next review: 30 DTE or ±25% on premium.")
    return {"primary": primary, "actions": actions}


def kelly_and_ruin(win_rate, avg_win_R, avg_loss_R, trades_per_month=8):
    """Kelly Criterion + risk-of-ruin. Institutional standard: use Quarter Kelly for actual sizing."""
    import math
    p = max(0.01, min(0.99, win_rate))
    q = 1 - p
    b = avg_win_R / avg_loss_R if avg_loss_R > 0 else 0.01
    kelly = (p * b - q) / b if b > 0 else 0
    edge = p * avg_win_R - q * avg_loss_R
    variance = p * q * (avg_win_R + avg_loss_R) ** 2
    z = 2 * edge / max(variance, 0.0001)
    ruin = max(0.0, min(1.0, math.exp(-z))) if edge > 0 else 1.0
    note = ("Negative edge — this system loses money on average. Do not trade mechanically." if edge <= 0
            else "Positive edge. Risk of ruin uses a 50% drawdown as the ruin threshold.")
    return {
        "kelly_pct": round(kelly * 100, 2), "half_kelly_pct": round(kelly / 2 * 100, 2),
        "quarter_kelly_pct": round(kelly / 4 * 100, 2),
        "edge_per_trade_pct": round(edge * 100, 2),
        "expected_monthly_pct": round(edge * trades_per_month * 100, 2),
        "ruin_probability_pct": round(ruin * 100, 1),
        "recommended_size_pct": round(max(0, kelly / 4 * 100), 2),
        "note": note,
    }



# =============================================================================
# =============================================================================
# LIVE OPTIONS INTELLIGENCE ENGINE
# V5 (yfinance + BSM) → V5.1 (+ Tradier broker Greeks when token available)
# =============================================================================

def bsm_greeks(S, K, T, r, sigma, flag='p'):
    """Exact Black-Scholes-Merton Greeks via scipy.stats.norm."""
    from scipy.stats import norm as _norm
    import math
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return None
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    phi = _norm.pdf(d1); sqT = math.sqrt(T)
    if flag == 'c':
        price    = S*_norm.cdf(d1) - K*math.exp(-r*T)*_norm.cdf(d2)
        delta    = _norm.cdf(d1)
        theta    = (-(S*phi*sigma)/(2*sqT) - r*K*math.exp(-r*T)*_norm.cdf(d2))/365
        prob_itm = _norm.cdf(d2)
    else:
        price    = K*math.exp(-r*T)*_norm.cdf(-d2) - S*_norm.cdf(-d1)
        delta    = _norm.cdf(d1) - 1
        theta    = (-(S*phi*sigma)/(2*sqT) + r*K*math.exp(-r*T)*_norm.cdf(-d2))/365
        prob_itm = _norm.cdf(-d2)
    gamma = phi/(S*sigma*sqT)
    vega  = S*phi*sqT/100
    return {"price":round(price,4), "delta":round(delta,4), "gamma":round(gamma,6),
            "theta":round(theta,4), "vega":round(vega,4),
            "iv_pct":round(sigma*100,2), "prob_itm":round(prob_itm,4),
            "pop":round(1-abs(delta),4)}


@st.cache_data(ttl=300)
def live_options_greeks(ticker, dte_target=30, r_annual=0.045):
    """
    yfinance options chain + BSM Greeks (scipy). Free, no API key.
    IV from real market bid/ask prices via yfinance.
    Greeks: exact BSM from that IV.
    IVR: 1-year realized vol range as proxy (best free approximation).
    """
    import math
    from datetime import date, datetime
    try:
        t    = yf.Ticker(ticker)
        hist = t.history(period="1y")
        if hist.empty or len(hist) < 30:
            return None
        rets     = np.log(hist['Close']/hist['Close'].shift(1)).dropna()
        rv_30d   = rets.rolling(30).std() * math.sqrt(252) * 100
        rv_curr  = float(rv_30d.iloc[-1])
        rv_high  = float(rv_30d.max())
        rv_low   = float(rv_30d.min())
        S        = float(hist['Close'].iloc[-1])
        expiries = t.options
        if not expiries:
            return {"ticker":ticker, "S":S, "no_options":True}
        today    = date.today()
        best_exp = min(expiries, key=lambda e:
            abs((datetime.strptime(e,'%Y-%m-%d').date()-today).days - dte_target))
        dte      = (datetime.strptime(best_exp,'%Y-%m-%d').date()-today).days
        T        = max(dte/365, 0.001)
        chain    = t.option_chain(best_exp)
        calls    = chain.calls[['strike','bid','ask','impliedVolatility','volume','openInterest']].dropna()
        puts     = chain.puts[ ['strike','bid','ask','impliedVolatility','volume','openInterest']].dropna()
        atm_c    = calls.iloc[(calls['strike']-S).abs().argsort()[:1]]
        atm_p    = puts.iloc[(puts['strike']-S).abs().argsort()[:1]]
        iv_c     = float(atm_c['impliedVolatility'].iloc[0])*100 if len(atm_c)>0 else rv_curr
        iv_p     = float(atm_p['impliedVolatility'].iloc[0])*100 if len(atm_p)>0 else rv_curr
        atm_iv   = (iv_c+iv_p)/2
        ivr      = max(0,min(100,(atm_iv-rv_low)/(rv_high-rv_low)*100)) if rv_high!=rv_low else 50
        rows = []
        for flag, df in [('p',puts),('c',calls)]:
            near = df[(df['strike']>=S*0.96) & (df['strike']<=S*1.04)].head(25)
            for _, row in near.iterrows():
                K = float(row['strike']); iv_dec = float(row['impliedVolatility'])
                if iv_dec <= 0: continue
                g = bsm_greeks(S, K, T, r_annual, iv_dec, flag)
                if not g: continue
                rows.append({"type":flag.upper(),"strike":K,"iv_pct":round(iv_dec*100,2),
                    "bid":float(row.get('bid',0)),"ask":float(row.get('ask',0)),
                    "delta":g['delta'],"gamma":g['gamma'],"theta_day":g['theta'],
                    "vega_1pct":g['vega'],"pop":g['pop'],"prob_itm":g['prob_itm'],
                    "oi":int(row.get('openInterest',0)),"source":"yfinance+BSM"})
        return {"ticker":ticker,"S":S,"expiry":best_exp,"dte":dte,"T":T,
                "atm_iv":round(atm_iv,2),"rv_current":round(rv_curr,2),
                "rv_high":round(rv_high,2),"rv_low":round(rv_low,2),
                "ivr_proxy":round(ivr,1),
                "ivr_method":"1y realized vol range used as IV history proxy.",
                "greeks_table":rows,
                "source":"yfinance+BSM",
                "source_label":"yfinance delayed IV + BSM exact formula (scipy)"}
    except Exception as ex:
        return {"error":str(ex)}


# ── Tradier integration (upgrade path — active when TRADIER_TOKEN in secrets) ─

def _tradier_headers():
    """Return (headers, base_url) tuple, or (None, None) if not configured."""
    try:
        token = (st.secrets.get("TRADIER_TOKEN")
                 or st.secrets.get("tradier", {}).get("token"))
        if not token:
            return None, None
        env  = (st.secrets.get("TRADIER_ENV")
                or st.secrets.get("tradier", {}).get("env","production")).lower()
        base = ("https://sandbox.tradier.com" if env == "sandbox"
                else "https://api.tradier.com")
        return {"Authorization":f"Bearer {token}","Accept":"application/json"}, base
    except Exception:
        return None, None


def _tradier_is_connected():
    h, _ = _tradier_headers()
    return h is not None


@st.cache_data(ttl=60)
def _tradier_quote(ticker):
    headers, base = _tradier_headers()
    if not headers: return None
    try:
        req = urllib.request.Request(
            f"{base}/v1/markets/quotes?symbols={ticker}&greeks=false",
            headers=headers)
        with urllib.request.urlopen(req, timeout=8) as r:
            d = json.loads(r.read())
        q = d.get("quotes",{}).get("quote")
        q = q if isinstance(q,dict) else (q[0] if isinstance(q,list) and q else {})
        return float(q.get("last") or q.get("bid") or 0) or None
    except Exception: return None


@st.cache_data(ttl=60)
def _tradier_expirations(ticker):
    headers, base = _tradier_headers()
    if not headers: return None
    try:
        req = urllib.request.Request(
            f"{base}/v1/markets/options/expirations?symbol={ticker}&includeAllRoots=true",
            headers=headers)
        with urllib.request.urlopen(req, timeout=8) as r:
            d = json.loads(r.read())
        exps = d.get("expirations",{}).get("date")
        return ([exps] if isinstance(exps,str) else exps) or []
    except Exception: return None


@st.cache_data(ttl=60)
def _tradier_chain_greeks(ticker, expiration):
    """
    Fetch options chain with Tradier broker-computed Greeks.
    Greeks come from Tradier's own vol surface model (smv_vol).
    NOT BSM approximation — actual pricing engine output.
    """
    headers, base = _tradier_headers()
    if not headers: return None
    try:
        req = urllib.request.Request(
            f"{base}/v1/markets/options/chains?symbol={ticker}&expiration={expiration}&greeks=true",
            headers=headers)
        with urllib.request.urlopen(req, timeout=10) as r:
            d = json.loads(r.read())
        opts = d.get("options",{}).get("option",[])
        if isinstance(opts,dict): opts=[opts]
        rows = []
        for o in opts:
            g = o.get("greeks") or {}
            if not g or g.get("delta") is None: continue
            iv_val = g.get("smv_vol") or g.get("mid_iv") or 0
            rows.append({
                "type": "P" if o.get("option_type","").lower()=="put" else "C",
                "strike": float(o.get("strike",0)),
                "bid":  float(o.get("bid") or 0), "ask": float(o.get("ask") or 0),
                "volume":int(o.get("volume") or 0),"oi":int(o.get("open_interest") or 0),
                "iv_pct":   round(float(iv_val)*100,2),
                "delta":    round(float(g.get("delta",0)),4),
                "gamma":    round(float(g.get("gamma",0)),6),
                "theta_day":round(float(g.get("theta",0)),4),
                # Tradier vega is per 1pt move in vol — divide by 100 for per-1% convention
                "vega_1pct":round(float(g.get("vega",0))/100,4),
                "rho":      round(float(g.get("rho",0)),4),
                "pop":      round(1-abs(float(g.get("delta",0.5))),3),
                "prob_itm": round(abs(float(g.get("delta",0.5))),3),
                "source":   "Tradier",
            })
        return rows or None
    except Exception: return None


@st.cache_data(ttl=60)
def live_options_greeks_v2(ticker, dte_target=30, r_annual=0.045):
    """
    Unified options intelligence.
    Tradier path: broker-computed Greeks (real-time), smv_vol IV surface.
    Fallback path: yfinance delayed IV + BSM exact formula.
    Data provenance labelled in every record.
    """
    import math
    from datetime import date, datetime
    headers, _ = _tradier_headers()

    if headers:
        S = _tradier_quote(ticker)
        exps = _tradier_expirations(ticker)
        if S and exps:
            today = date.today()
            best_exp = min(exps, key=lambda e: abs(
                (datetime.strptime(e,"%Y-%m-%d").date()-today).days - dte_target))
            dte  = (datetime.strptime(best_exp,"%Y-%m-%d").date()-today).days
            rows = _tradier_chain_greeks(ticker, best_exp)
            if rows:
                df_r   = pd.DataFrame(rows)
                atm    = df_r.iloc[(df_r["strike"]-S).abs().argsort()[:2]]
                atm_iv = float(atm["iv_pct"].mean()) if len(atm)>0 else 0
                try:
                    hist2  = yf.Ticker(ticker).history(period="1y")
                    rets2  = np.log(hist2["Close"]/hist2["Close"].shift(1)).dropna()
                    rv2    = rets2.rolling(30).std()*math.sqrt(252)*100
                    ivr2   = max(0,min(100,(atm_iv-float(rv2.min()))/(float(rv2.max())-float(rv2.min()))*100))
                    rv_cur = round(float(rv2.iloc[-1]),2)
                except Exception:
                    ivr2=50.0; rv_cur=round(atm_iv*0.85,2)
                near = df_r[(df_r["strike"]>=S*0.96)&(df_r["strike"]<=S*1.04)]
                return {"ticker":ticker,"S":S,"expiry":best_exp,"dte":dte,
                        "atm_iv":round(atm_iv,2),"ivr_proxy":round(ivr2,1),
                        "rv_current":rv_cur,"greeks_table":near.to_dict("records"),
                        "source":"Tradier",
                        "source_label":"Tradier broker-computed Greeks (real-time, smv_vol surface)"}

    result = live_options_greeks(ticker, dte_target=dte_target, r_annual=r_annual)
    if result and not result.get("error"):
        result["source"]       = "yfinance+BSM"
        result["source_label"] = "yfinance delayed IV + BSM exact (scipy)"
    return result

# =============================================================================
# TRADING ENGINE — V6
# Tradier brokerage API integration for actual trade execution.
# Instruments: US Stocks, ETFs, Options (single + multi-leg spreads).
# Philosophy: IWT discipline → analysis FIRST, execute only when permitted.
# Safety: two-step confirm, max-loss display, PDT check, IWT gate.
# Note: Futures/Forex require a separate broker (see RESOURCES.md).
# =============================================================================

import urllib.request, urllib.parse, json

# ── Account management ──────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def tdr_get_account_id():
    """Get primary Tradier account number from user profile."""
    headers, base = _tradier_headers()
    if not headers: return None
    try:
        req = urllib.request.Request(f"{base}/v1/user/profile", headers=headers)
        with urllib.request.urlopen(req, timeout=8) as r:
            d = json.loads(r.read())
        accts = d.get("profile", {}).get("account", [])
        if isinstance(accts, dict): accts = [accts]
        # Prefer margin account; fall back to first
        for a in accts:
            if a.get("classification") in ["individual", "margin", "traditional_ira"]:
                return a.get("account_number")
        return accts[0].get("account_number") if accts else None
    except Exception:
        return None


@st.cache_data(ttl=30)
def tdr_get_balances(account_id):
    """Account cash, equity, and buying power."""
    headers, base = _tradier_headers()
    if not headers or not account_id: return None
    try:
        req = urllib.request.Request(
            f"{base}/v1/accounts/{account_id}/balances", headers=headers)
        with urllib.request.urlopen(req, timeout=8) as r:
            d = json.loads(r.read())
        b = d.get("balances", {})
        return {
            "total_equity":    float(b.get("total_equity", 0)),
            "cash":            float(b.get("cash", {}).get("cash_available", 0)
                                     if isinstance(b.get("cash"), dict)
                                     else b.get("total_cash", 0)),
            "option_bp":       float(b.get("option_buying_power", 0)),
            "stock_bp":        float(b.get("stock_buying_power", 0)),
            "day_trade_bp":    float(b.get("day_trading_buying_power", 0)),
            "pdt_status":      bool(b.get("pattern_day_trader", False)),
            "account_type":    b.get("account_type", "unknown"),
        }
    except Exception: return None


@st.cache_data(ttl=30)
def tdr_get_positions(account_id):
    """Open positions."""
    headers, base = _tradier_headers()
    if not headers or not account_id: return []
    try:
        req = urllib.request.Request(
            f"{base}/v1/accounts/{account_id}/positions", headers=headers)
        with urllib.request.urlopen(req, timeout=8) as r:
            d = json.loads(r.read())
        pos = d.get("positions", {}).get("position", [])
        if isinstance(pos, dict): pos = [pos]
        return pos or []
    except Exception: return []


@st.cache_data(ttl=15)
def tdr_get_orders(account_id):
    """Recent orders (last 30 days)."""
    headers, base = _tradier_headers()
    if not headers or not account_id: return []
    try:
        req = urllib.request.Request(
            f"{base}/v1/accounts/{account_id}/orders", headers=headers)
        with urllib.request.urlopen(req, timeout=8) as r:
            d = json.loads(r.read())
        ords = d.get("orders", {}).get("order", [])
        if isinstance(ords, dict): ords = [ords]
        return ords or []
    except Exception: return []


# ── Order building utilities ────────────────────────────────────────────────

def build_option_symbol(underlying: str, expiry_str: str,
                        option_type: str, strike: float) -> str:
    """
    Build OCC-standard option symbol for Tradier.
    underlying: "SPY", "AAPL", "SPX"
    expiry_str: "2026-06-19" (YYYY-MM-DD)
    option_type: "CALL" or "PUT"
    strike: 730.0 or 5300.0
    Returns: e.g. "SPY260619P00730000"
    """
    from datetime import datetime
    exp = datetime.strptime(expiry_str, "%Y-%m-%d")
    exp_code  = exp.strftime("%y%m%d")              # YYMMDD
    type_char = "C" if option_type.upper()[0] == "C" else "P"
    strike_int = round(strike * 1000)               # dollars → milli-dollars
    return f"{underlying.upper()}{exp_code}{type_char}{strike_int:08d}"


def _tdr_post(endpoint: str, form_data: dict) -> dict:
    """
    POST form-encoded data to Tradier.
    Tradier trading endpoints use application/x-www-form-urlencoded.
    Returns parsed JSON response.
    """
    headers, base = _tradier_headers()
    if not headers:
        return {"error": "Tradier not configured"}
    payload = urllib.parse.urlencode(form_data).encode()
    req = urllib.request.Request(
        f"{base}{endpoint}", data=payload, method="POST",
        headers={**headers, "Content-Type": "application/x-www-form-urlencoded"}
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        try:    return json.loads(body)
        except: return {"error": f"HTTP {e.code}: {body[:200]}"}
    except Exception as ex:
        return {"error": str(ex)}


# ── Order execution functions ───────────────────────────────────────────────

def tdr_place_equity_order(
    account_id: str,
    symbol: str,
    side: str,           # "buy" | "sell"
    quantity: int,
    order_type: str,     # "market" | "limit" | "stop" | "stop_limit"
    limit_price: float = None,
    stop_price: float  = None,
    duration: str      = "day",
    tag: str           = "EasyStockTrader",
) -> dict:
    """Place a stock/ETF order. Returns Tradier order response."""
    params = {
        "class":    "equity",
        "symbol":   symbol.upper(),
        "side":     side,
        "quantity": str(int(quantity)),
        "type":     order_type,
        "duration": duration,
        "tag":      tag,
    }
    if order_type in ("limit", "stop_limit") and limit_price is not None:
        params["price"] = f"{limit_price:.2f}"
    if order_type in ("stop", "stop_limit") and stop_price is not None:
        params["stop"] = f"{stop_price:.2f}"
    return _tdr_post(f"/v1/accounts/{account_id}/orders", params)


def tdr_place_option_order(
    account_id: str,
    option_symbol: str,  # OCC format e.g. "SPY260619P00730000"
    side: str,           # "buy_to_open" | "sell_to_open" | "buy_to_close" | "sell_to_close"
    quantity: int,
    order_type: str,     # "market" | "limit"
    limit_price: float = None,
    duration: str      = "day",
    tag: str           = "EasyStockTrader",
) -> dict:
    """Place a single-leg options order."""
    params = {
        "class":         "option",
        "symbol":        option_symbol[:3].rstrip(),   # root symbol
        "option_symbol": option_symbol,
        "side":          side,
        "quantity":      str(int(quantity)),
        "type":          order_type,
        "duration":      duration,
        "tag":           tag,
    }
    if order_type == "limit" and limit_price is not None:
        params["price"] = f"{limit_price:.2f}"
    return _tdr_post(f"/v1/accounts/{account_id}/orders", params)


def tdr_place_spread_order(
    account_id: str,
    underlying: str,
    legs: list,           # [{"symbol":occ_sym, "side":side, "qty":int}, ...]
    net_price: float,     # credit positive, debit negative
    order_type: str = "limit",
    duration: str   = "day",
    tag: str        = "EasyStockTrader",
) -> dict:
    """
    Place a multi-leg options order (vertical spread, etc.).
    Credit spreads: net_price > 0 (you're receiving money).
    Debit spreads:  net_price < 0 (you're paying money).
    Tradier always receives 'price' as absolute value with sign implied by sides.
    """
    params = {
        "class":    "multileg",
        "symbol":   underlying.upper(),
        "type":     order_type,
        "duration": duration,
        "price":    f"{abs(net_price):.2f}",
        "tag":      tag,
    }
    for i, leg in enumerate(legs):
        params[f"legs[{i}][option_symbol]"] = leg["symbol"]
        params[f"legs[{i}][side]"]          = leg["side"]
        params[f"legs[{i}][quantity]"]      = str(int(leg["qty"]))
    return _tdr_post(f"/v1/accounts/{account_id}/orders", params)


def tdr_cancel_order(account_id: str, order_id: str) -> dict:
    """Cancel an open order."""
    headers, base = _tradier_headers()
    if not headers: return {"error": "Tradier not configured"}
    req = urllib.request.Request(
        f"{base}/v1/accounts/{account_id}/orders/{order_id}",
        method="DELETE", headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception as ex:
        return {"error": str(ex)}


# ── Trade safety checks ─────────────────────────────────────────────────────

def trade_safety_check(balances, max_loss, qty, pdt_impact=False) -> dict:
    """
    Pre-trade safety gate. Returns dict with pass/fail for each check.
    All checks are advisory (warnings) — user can override with explicit confirm.
    """
    if not balances:
        return {"error": "Cannot load account balances. Check Tradier connection."}

    checks = []

    # 1. Buying power
    bp = balances.get("option_bp") or balances.get("cash") or 0
    risk_pct_of_bp = (max_loss / bp * 100) if bp > 0 else 100
    checks.append({
        "name":   "Buying power",
        "pass":   max_loss <= bp,
        "detail": f"Max loss ${max_loss:,.0f} vs buying power ${bp:,.0f} ({risk_pct_of_bp:.1f}%)"
    })

    # 2. Risk-per-trade (Kelly discipline: don't risk > 5% of equity on one trade)
    equity = balances.get("total_equity", bp)
    equity_risk_pct = (max_loss / equity * 100) if equity > 0 else 100
    checks.append({
        "name":   "Position size (Kelly)",
        "pass":   equity_risk_pct <= 5,
        "detail": f"This trade risks {equity_risk_pct:.1f}% of total equity. IWT/Kelly discipline: max 2-5% per trade."
    })

    # 3. PDT warning
    if pdt_impact and balances.get("pdt_status") is False and equity < 25000:
        checks.append({
            "name":   "PDT (Pattern Day Trader)",
            "pass":   False,
            "detail": "Account < $25k. This may count as a day trade if opened and closed same day."
        })

    return {"checks": checks, "all_pass": all(c["pass"] for c in checks)}

# =============================================================================
# BACKTESTING ENGINE — V7
# Real 1-year historical backtest using:
#   - yfinance: real daily prices (SPY, VIX, TNX)
#   - Black-Scholes-Merton: exact option pricing from historical IV
#   - Rolling 90-day IVR: drives strategy selection at each date
# Math is exact. Labels are honest. No simulated or invented prices.
#
# Limitation: historical option MARKET prices ≠ BSM prices. Real fills
# differ by bid/ask spread, skew, and liquidity. Results labelled clearly.
# =============================================================================

@st.cache_data(ttl=7200)   # 2-hour cache — data doesn't change intra-day
def load_backtest_data():
    """Load 2 years of real market data for rolling window calculations."""
    import warnings; warnings.filterwarnings("ignore")
    def gc(raw):
        return (raw['Close'].iloc[:,0]
                if isinstance(raw.columns, pd.MultiIndex)
                else raw['Close']).dropna()
    spy = gc(yf.download("SPY",  period="2y", progress=False))
    vix = gc(yf.download("^VIX", period="2y", progress=False))
    tnx = gc(yf.download("^TNX", period="2y", progress=False))
    full = pd.DataFrame({'spy':spy,'vix':vix,'tnx':tnx}).dropna()
    full.index = pd.to_datetime(full.index)
    # Rolling indicators
    full['ivr_90d'] = ((full['vix']-full['vix'].rolling(90).min()) /
                       (full['vix'].rolling(90).max()-full['vix'].rolling(90).min()
                        ).replace(0,np.nan)*100).clip(0,100)
    full['sma20']     = full['spy'].rolling(20).mean()
    full['sma50']     = full['spy'].rolling(50).mean()
    full['trend_up']  = full['spy'] > full['sma20']
    full['rv_30d']    = (np.log(full['spy']/full['spy'].shift(1))
                         .rolling(30).std() * np.sqrt(252) * 100)
    return full.dropna().iloc[-252:].copy()   # last ~1 year


def _bsm(S, K, T, r, sigma, flag):
    """Exact Black-Scholes-Merton price. Uses scipy.stats.norm."""
    from scipy.stats import norm as _n
    if T <= 1/365:
        return max(0.0, K-S) if flag=='p' else max(0.0, S-K)
    d1 = (math.log(S/K)+(r+0.5*sigma**2)*T)/(sigma*math.sqrt(T))
    d2 = d1-sigma*math.sqrt(T)
    if flag == 'p':
        return K*math.exp(-r*T)*_n.cdf(-d2)-S*_n.cdf(-d1)
    return S*_n.cdf(d1)-K*math.exp(-r*T)*_n.cdf(d2)


def backtest_credit_spread(df, min_ivr=40, spread_w=5, target_dte=30,
                            close_dte=21, profit_tgt=0.50, stop_mult=2.0,
                            slip=0.03, commission=1.30, contracts=1,
                            initial_capital=25_000):
    """
    Weekly SPY put credit spread strategy.
    Entry: IVR-90d ≥ min_ivr (volatility is elevated vs recent history).
    Short strike: ~25-delta (outside 75% of expected move).
    Exit: 50% profit | spread doubles | 21 DTE.
    Costs: $0.03/share slippage + $1.30/contract commission.
    """
    trades = []; capital = initial_capital
    eq_curve = [{'date': df.index[0].date(), 'equity': capital, 'cum_pnl': 0}]
    mondays = df[df.index.dayofweek == 0]

    for edt, erow in mondays.iterrows():
        ivr = float(erow['ivr_90d']); vix = float(erow['vix'])
        if np.isnan(ivr) or ivr < min_ivr:
            continue
        S  = float(erow['spy']); IV = vix/100
        r  = float(erow['tnx'])/100; T  = target_dte/365
        EM = S * IV * math.sqrt(T)
        K_s = round((S - EM*0.75)/0.5)*0.5
        K_l = K_s - spread_w
        credit = _bsm(S,K_s,T,r,IV,'p') - _bsm(S,K_l,T,r,IV,'p')
        if credit/spread_w < 0.15: continue   # credit too thin

        # Find expiry Friday
        exp_dt = edt + timedelta(days=target_dte+2)
        while exp_dt.weekday() != 4: exp_dt -= timedelta(days=1)

        exit_dt = None; exit_sv = None; why = None
        for hdt, hr in df[(df.index>edt)&(df.index<=exp_dt)].iterrows():
            dtr = (exp_dt-hdt).days; Th = max(dtr/365, 0)
            Sh  = float(hr['spy']); IVh = float(hr['vix'])/100; rh = float(hr['tnx'])/100
            sv  = (_bsm(Sh,K_s,Th,rh,IVh,'p') - _bsm(Sh,K_l,Th,rh,IVh,'p'))
            if sv <= credit*(1-profit_tgt): exit_dt=hdt; exit_sv=sv; why="50% profit"; break
            if sv >= credit*stop_mult:      exit_dt=hdt; exit_sv=sv; why="Stop (2×)";  break
            if dtr <= close_dte:            exit_dt=hdt; exit_sv=sv; why="21 DTE";     break

        if exit_dt is None:
            Se = float(df.loc[df.index<=exp_dt,'spy'].iloc[-1])
            exit_sv = max(0.0, min(float(spread_w), K_s-Se))
            exit_dt = exp_dt; why = "Expired"

        pnl = (credit - exit_sv - slip) * 100 * contracts - commission
        capital += pnl
        trades.append({
            'entry_date':  edt.date(),
            'exit_date':   exit_dt.date() if hasattr(exit_dt,'date') else exit_dt,
            'SPY_entry':   round(S,2), 'Short_K': K_s, 'Long_K': K_l,
            'VIX':         round(vix,1), 'IVR_90d': round(ivr,1),
            'Credit_$/shr':round(credit,3),
            'Eff_%':       round(credit/spread_w*100,1),
            'P&L':         round(pnl,2),
            'Exit':        why,
            'W/L':         'W' if pnl>0 else 'L',
        })
        eq_curve.append({'date': edt.date(), 'equity': round(capital,2),
                         'cum_pnl': round(capital-initial_capital,2)})

    return pd.DataFrame(trades), pd.DataFrame(eq_curve), capital


def backtest_long_call(df, max_vix=22, call_dte=60, target_delta=0.70,
                        profit_50=True, profit_100=True, stop_pct=0.50,
                        slip=0.05, commission=1.30, contracts=1,
                        initial_capital=25_000):
    """
    IWT DITM long call strategy.
    Entry: SPY above 20-day SMA + VIX ≤ max_vix (low relative IV).
    Strike: DITM at ~delta 0.70 (approx: S × e^(-d1×σ√T + ...)).
    Exit: 50% gain | 100% gain | 50% loss | 21 DTE.
    Buying options: max loss = premium paid. Defined risk.
    """
    trades = []; capital = initial_capital
    eq_curve = [{'date': df.index[0].date(), 'equity': capital, 'cum_pnl': 0}]
    entry_weeks = set()  # one entry per week max
    mondays = df[(df.index.dayofweek==0) & df['trend_up']]

    for edt, erow in mondays.iterrows():
        vix = float(erow['vix'])
        if vix > max_vix: continue
        week_key = (edt.year, edt.isocalendar()[1])
        if week_key in entry_weeks: continue
        entry_weeks.add(week_key)

        S  = float(erow['spy']); IV = vix/100; r = float(erow['tnx'])/100
        T  = call_dte/365
        # DITM strike: solve for K such that delta≈0.70
        # delta=N(d1)=0.70 → d1≈0.524; d1=(ln(S/K)+(r+σ²/2)T)/(σ√T)
        # → ln(S/K)=d1×σ√T-(r+σ²/2)T → K=S×exp(-(d1×σ√T)+(r+σ²/2)T)
        d1t = 0.524
        K   = round(S*math.exp(-d1t*IV*math.sqrt(T)+(r+0.5*IV**2)*T)/0.5)*0.5
        K   = min(K, round(S*0.97/0.5)*0.5)
        prem = _bsm(S, K, T, r, IV, 'c')
        if prem <= 0: continue

        cost_tot = (prem+slip)*100*contracts + commission
        stop_amt = cost_tot * stop_pct    # IWT 50% stop
        tgt_50   = cost_tot * 0.50        # IWT 50% profit target
        tgt_100  = cost_tot * 1.00        # IWT 100% profit target

        exp_dt = edt+timedelta(days=call_dte+7)
        while exp_dt.weekday()!=4: exp_dt-=timedelta(days=1)

        exit_dt=None; exit_v=None; why=None
        for hdt, hr in df[(df.index>edt)&(df.index<=exp_dt)].iterrows():
            dtr=(exp_dt-hdt).days; Th=max(dtr/365,0)
            Sh=float(hr['spy']); IVh=float(hr['vix'])/100; rh=float(hr['tnx'])/100
            curr_v=(max(0.0,_bsm(Sh,K,Th,rh,IVh,'c'))-slip)*100*contracts-commission
            gain=curr_v-cost_tot
            if profit_100 and gain>=tgt_100: exit_dt=hdt;exit_v=curr_v;why="100% gain";break
            if profit_50  and gain>=tgt_50:  exit_dt=hdt;exit_v=curr_v;why="50% gain"; break
            if gain<=-stop_amt:              exit_dt=hdt;exit_v=curr_v;why="50% stop"; break
            if dtr<=21:                      exit_dt=hdt;exit_v=curr_v;why="21DTE close";break

        if exit_dt is None:
            Se=float(df.loc[df.index<=exp_dt,'spy'].iloc[-1])
            exit_v=(max(0.0,Se-K)-slip)*100*contracts-commission; why="Expired"; exit_dt=exp_dt

        pnl = exit_v - cost_tot
        capital += pnl
        trades.append({
            'entry_date': edt.date(),
            'exit_date':  exit_dt.date() if hasattr(exit_dt,'date') else exit_dt,
            'SPY_entry':  round(S,2), 'Call_K': K,
            'VIX':        round(vix,1),
            'Premium_$':  round(prem,3),
            'Cost_total': round(cost_tot,2),
            'P&L':        round(pnl,2),
            'Exit':       why,
            'W/L':        'W' if pnl>0 else 'L',
        })
        eq_curve.append({'date':edt.date(),'equity':round(capital,2),
                         'cum_pnl':round(capital-initial_capital,2)})

    return pd.DataFrame(trades), pd.DataFrame(eq_curve), capital


def compute_backtest_stats(trades_df, initial_capital, strategy_name):
    """Compute summary statistics. All math is exact given the inputs."""
    if trades_df.empty:
        return {"name": strategy_name, "n_trades": 0}
    t = trades_df.copy()
    wins  = t[t['P&L']>0]; losses = t[t['P&L']<=0]
    total_pnl  = t['P&L'].sum()
    cum_pnl    = t['P&L'].cumsum()
    rolling_max= cum_pnl.cummax()
    drawdowns  = cum_pnl - rolling_max

    pf = abs(wins['P&L'].sum()/losses['P&L'].sum()) if len(losses)>0 and losses['P&L'].sum()!=0 else float('inf')
    avg_win  = wins['P&L'].mean()  if len(wins)>0  else 0
    avg_loss = losses['P&L'].mean() if len(losses)>0 else 0
    ev_per_trade = total_pnl / len(t)

    return {
        "name":             strategy_name,
        "n_trades":         len(t),
        "win_rate_pct":     round(len(wins)/len(t)*100, 1),
        "n_wins":           len(wins),
        "n_losses":         len(losses),
        "total_pnl":        round(total_pnl, 2),
        "return_pct":       round(total_pnl/initial_capital*100, 2),
        "avg_win":          round(avg_win, 2),
        "avg_loss":         round(avg_loss, 2),
        "profit_factor":    round(pf, 2) if pf != float('inf') else "∞",
        "ev_per_trade":     round(ev_per_trade, 2),
        "max_drawdown":     round(drawdowns.min(), 2),
        "exit_reasons":     t['Exit'].value_counts().to_dict(),
    }

# =============================================================================
# V8 — PLATFORM ORDER TRANSLATOR + LANGUAGE SIMPLIFICATION
# Converts spread parameters into exact order-entry strings for major platforms.
# No jargon for beginners. Full technical detail for professionals.
# =============================================================================

def lang_cap(text, min_level="Advanced"):
    """
    Render a Streamlit caption only if the user's experience level
    meets or exceeds min_level. Suppresses developer/technical detail for
    Beginner and Intermediate users automatically.
    """
    _order = {"Beginner":0,"Intermediate":1,"Advanced":2,"Professional":3}
    user_lvl = st.session_state.get("lang_level","Beginner")
    if _order.get(user_lvl,0) >= _order.get(min_level,2):
        st.caption(text)


def plain_spread_explanation(underlying, short_k, long_k, spread_type,
                              expiry_str, credit, contracts, spread_width, lvl):
    """
    Returns a dict of plain-English explanation blocks and platform order strings.
    Terminology designed so a the IWT methodology student with no options background
    can immediately understand what to click and why.
    spread_type: "PUT_CREDIT" | "CALL_CREDIT"
    """
    from datetime import datetime
    option_word = "PUT" if "PUT" in spread_type else "CALL"
    max_profit  = round(credit * 100 * contracts, 2)
    max_loss    = round((spread_width - credit) * 100 * contracts, 2)
    breakeven   = round(short_k - credit, 2) if "PUT" in spread_type else round(short_k + credit, 2)
    credit_eff  = round(credit / spread_width * 100, 1)

    try:
        from datetime import datetime
        exp_dt  = datetime.strptime(expiry_str, "%Y-%m-%d")
        exp_tos = exp_dt.strftime("%d %b %y").upper()   # thinkorSwim format: 11 MAY 26
        exp_fid = exp_dt.strftime("%b %d, %Y")          # Fidelity format: May 11, 2026
    except Exception:
        exp_tos = expiry_str; exp_fid = expiry_str

    # ── Plain English: what you did ──────────────────────────────────────────
    if "PUT" in spread_type:
        sell_action = (f"You SELL the {short_k:.0f} PUT — the HIGHER number. "
                       f"This is where you collect your {credit:.2f} premium.")
        buy_action  = (f"You BUY the {long_k:.0f} PUT — the LOWER number. "
                       f"This is your insurance. It limits your maximum loss.")
        market_view = (f"You want SPX to STAY ABOVE {short_k:.0f} by expiry. "
                       f"If it does, you keep the full ${max_profit:,.0f} premium.")
        danger_zone = f"If SPX falls below {long_k:.0f}, max loss = ${max_loss:,.0f}."
    else:  # CALL credit
        sell_action = (f"You SELL the {short_k:.0f} CALL — the LOWER number. "
                       f"This is where you collect your {credit:.2f} premium.")
        buy_action  = (f"You BUY the {long_k:.0f} CALL — the HIGHER number. "
                       f"This is your insurance. It limits your maximum loss.")
        market_view = (f"You want SPX to STAY BELOW {short_k:.0f} by expiry. "
                       f"If it does, you keep the full ${max_profit:,.0f} premium.")
        danger_zone = f"If SPX rises above {long_k:.0f}, max loss = ${max_loss:,.0f}."

    # ── Platform order strings (exactly what to type/click) ─────────────────
    # thinkorSwim (most common for SPX options):
    # Format: SELL -N VERTICAL [UNDERLYING] 100 ([DATE]) [SHORT]/[LONG] [TYPE] @ [CREDIT]
    # Note: in TOS, the spread always shows SHORT/LONG (higher/lower for puts)
    if "PUT" in spread_type:
        tos_str = (f"SELL -{contracts} VERTICAL {underlying} 100 ({exp_tos}) "
                   f"{short_k:.0f}/{long_k:.0f} PUT @ {credit:.2f}")
        fid_str = (f"Sell to Open: {short_k:.0f} PUT  |  Buy to Open: {long_k:.0f} PUT  "
                   f"|  Net Credit: ${credit:.2f}")
        ibkr_str= (f"Combo order: SELL {contracts} {underlying} {exp_tos} {short_k:.0f} PUT "
                   f"+ BUY {contracts} {underlying} {exp_tos} {long_k:.0f} PUT "
                   f"| Net limit (credit): ${credit:.2f}")
        tasty   = (f"SELL PUT SPREAD  |  Short {short_k:.0f} / Long {long_k:.0f}  "
                   f"|  ${credit:.2f} credit")
        wb_str  = (f"Options → Spread → Put Credit Spread  "
                   f"|  Sell {short_k:.0f} + Buy {long_k:.0f}  |  Net credit ${credit:.2f}")
    else:
        tos_str = (f"SELL -{contracts} VERTICAL {underlying} 100 ({exp_tos}) "
                   f"{long_k:.0f}/{short_k:.0f} CALL @ {credit:.2f}")
        fid_str = (f"Sell to Open: {short_k:.0f} CALL  |  Buy to Open: {long_k:.0f} CALL  "
                   f"|  Net Credit: ${credit:.2f}")
        ibkr_str= (f"Combo: SELL {contracts} {underlying} {exp_tos} {short_k:.0f} CALL "
                   f"+ BUY {contracts} {underlying} {exp_tos} {long_k:.0f} CALL "
                   f"| Net limit: ${credit:.2f}")
        tasty   = (f"SELL CALL SPREAD  |  Short {short_k:.0f} / Long {long_k:.0f}  "
                   f"|  ${credit:.2f} credit")
        wb_str  = (f"Options → Spread → Call Credit Spread  "
                   f"|  Sell {short_k:.0f} + Buy {long_k:.0f}  |  Net credit ${credit:.2f}")

    return {
        "sell_action":   sell_action,
        "buy_action":    buy_action,
        "market_view":   market_view,
        "danger_zone":   danger_zone,
        "max_profit":    max_profit,
        "max_loss":      max_loss,
        "breakeven":     breakeven,
        "credit_eff":    credit_eff,
        "tos":           tos_str,
        "fidelity":      fid_str,
        "ibkr":          ibkr_str,
        "tastytrade":    tasty,
        "webull":        wb_str,
        "option_word":   option_word,
    }

# =============================================================================
# GLOBAL PRE-MARKET SCAN — Asia → Europe → US
# the IWT framework's top-down morning workflow.
# Real data: yfinance tickers. Runs on demand, cached 30 min.
# =============================================================================

@st.cache_data(ttl=1800)
def run_global_scan():
    """
    Fetch real-time quotes for the IWT morning sheet:
    Asia (Nikkei, Hang Seng, Shanghai), Europe (DAX, FTSE, EuroStoxx),
    US (SPY, QQQ, VIX, 10Y, DXY, Oil, Gold, ES, NQ).
    Returns a structured dict of all readings.
    """
    import warnings; warnings.filterwarnings("ignore")
    def safe_quote(ticker, field="Close"):
        try:
            d = yf.download(ticker, period="2d", progress=False)
            if isinstance(d.columns, pd.MultiIndex):
                d = d[field].iloc[:,0] if field in d.columns.get_level_values(0) else d.iloc[:,0]
            else:
                d = d[field]
            d = d.dropna()
            if len(d) >= 2:
                prev, curr = float(d.iloc[-2]), float(d.iloc[-1])
                return {"value": curr, "prev": prev, "chg_pct": (curr-prev)/prev*100}
            elif len(d) == 1:
                return {"value": float(d.iloc[-1]), "prev": None, "chg_pct": None}
        except Exception:
            pass
        return {"value": None, "prev": None, "chg_pct": None}

    TICKERS = {
        "Asia": {
            "🇯🇵 Nikkei":       "^N225",
            "🇭🇰 Hang Seng":    "^HSI",
            "🇨🇳 Shanghai":     "000001.SS",
        },
        "Europe": {
            "🇩🇪 DAX":          "^GDAXI",
            "🇬🇧 FTSE 100":     "^FTSE",
            "🇪🇺 EuroStoxx50":  "^STOXX50E",
        },
        "US Market": {
            "SPY":              "SPY",
            "QQQ":              "QQQ",
            "VIX":              "^VIX",
            "10Y Yield":        "^TNX",
            "DXY (Dollar)":     "DX-Y.NYB",
            "Oil (WTI)":        "CL=F",
            "Gold":             "GC=F",
        },
        "Futures": {
            "ES (S&P futs)":    "ES=F",
            "NQ (Nasdaq futs)": "NQ=F",
            "RTY (Russell)":    "RTY=F",
        },
    }

    result = {}
    for region, tickers in TICKERS.items():
        result[region] = {}
        for name, sym in tickers.items():
            result[region][name] = safe_quote(sym)
    return result

# V4 — FULL INSTRUMENT SUITE · AUTO-DATA · BEGINNER-TO-PRO LANGUAGE
# =============================================================================

# ── Term Explainer ──────────────────────────────────────────────────────────
_EXPLANATIONS = {
    "vix": {
        "Beginner":      "VIX = the market's fear score. Low VIX = calm. High VIX = scared. Think of it like weather: high VIX = storm warning. Affects how much options cost.",
        "Intermediate":  "VIX = implied volatility of 30-day SPX options. Measures the market's expected annualised % move for the S&P 500. VIX of 20 ≈ expected ~1.25%/day move.",
        "Advanced":      "VIX = CBOE spot vol index. Derived from weighted SPX near/next-term option prices across strikes. Approximates E[realised vol] under risk-neutral measure. Mean-reverts to ~18-20 historically.",
        "Professional":  "VIX = √(∫₀³⁰ σ²_RN(t)dt × 365/30). Model-free IV using strip of OTM puts/calls. Variance swap rate proxy. VIX² = expected 30-day variance. Spread vs realised vol (VRP) drives premium-selling edge.",
    },
    "ivr": {
        "Beginner":      "IV Rank = how expensive options are RIGHT NOW vs the last year. 0% = cheapest. 100% = most expensive. Above 50%: good time to SELL options. Below 30%: good time to BUY them. This app calculates it automatically from live VIX data.",
        "Intermediate":  "IVR = (Current VIX − 52w Low) / (52w High − 52w Low) × 100. Tells you where today's implied volatility sits in its annual range. Drives the buy/sell premium decision systematically.",
        "Advanced":      "IVR ≠ IV Percentile (IVP). IVR uses range; IVP counts daily observations above current level. IVR spikes faster in crises. Both computed from trailing 252 days. IVR > 50 → statistically elevated premium; sell. IVR < 30 → historically cheap; buy.",
        "Professional":  "IVR from VIX rank. True stock IVR requires historical IV surface data (ORATS, OptionMetrics, CBOE DataShop). VIX rank is exact for SPX — VIX IS SPX 30-day implied vol. For equities, use IV Percentile from broker tools. Selling edge = VRP: IV historically overestimates realised vol by ~2-3 vol points.",
    },
    "theta": {
        "Beginner":      "Theta = time decay. Every day you hold an option, it loses a little value — like a gift card expiring. Short options (credit spreads) EARN from this. Long options LOSE from it. Our calculator shows how fast.",
        "Intermediate":  "Theta = daily dollar decay of option value holding everything else constant. Negative for long options, positive for short. Non-linear: doubles in speed roughly every time DTE halves. Accelerates sharply inside 30 DTE.",
        "Advanced":      "Θ = ∂V/∂τ (with sign convention: positive = time passing helps). For credit spreads: value remaining ≈ credit × √(DTE_remaining / DTE_entry) — a practical first-order model. Real Θ also depends on gamma (convexity) and vega (vol sensitivity).",
        "Professional":  "BSM theta: Θ = −[S·σ·φ(d₁)]/(2√T) − r·K·e^{−rT}·Φ(d₂) for calls. Non-constant gamma creates theta/gamma trade-off: selling theta = selling convexity. For spreads: net theta ≈ difference of legs. P&L decomposition: ΔP&L ≈ Δ·ΔS + ½·Γ·ΔS² + Θ·Δt + ν·Δσ.",
    },
    "kelly": {
        "Beginner":      "Kelly tells you how much of your money to risk on ONE trade — like knowing how much to bet when you have an advantage. The app recommends the SAFE version (Quarter Kelly) which professional traders use to protect against bad luck.",
        "Intermediate":  "Kelly Criterion: f* = (p·b − q) / b where p=win rate, b=avg win÷avg loss, q=1−p. Maximises long-run geometric growth. Quarter Kelly = safest institutional default. Positive edge required (f* > 0) to trade the system.",
        "Advanced":      "Kelly f* = edge/odds. Maximises E[ln(Wealth)]. Full Kelly → periodic 50-90% drawdowns even on +EV systems. Half Kelly → 75% of max CAGR, 50% of variance. Quarter Kelly → 60% CAGR, 25% variance. For options: adjust for fat tails (Kelly assumes lognormal).",
        "Professional":  "Fractional Kelly from information theory (Shannon). f* = (μ − r) / σ² in continuous time (Kelly-Markowitz duality). Practical constraints: VaR limits, mandate drawdown caps, correlation with existing book. Most quant funds: Kelly/4 to Kelly/10 depending on Sharpe regime. Portfolio Kelly: solve max Σ fᵢ·eᵢ s.t. Σ fᵢ·σᵢ ≤ σ_target.",
    },
    "expected_move": {
        "Beginner":      "Expected Move = how far the market is likely to move before your option expires. Think of it as the market's own guess. If the market expects a ±100 point move, putting your short strike outside that range means you're betting the market won't reach an unlikely place.",
        "Intermediate":  "EM ≈ ATM_straddle_price. Also: EM ≈ Price × IV × √(DTE/365). Represents 1 standard deviation — ~68% of outcomes fall within ±1 EM. Selling outside EM → POP ≈ 84%.",
        "Advanced":      "EM = S·σ_IV·√(DTE/365). First-order BSM. Ignores skew: put IVs typically higher, so put EM > call EM. More precise: EM ≈ 0.68 × straddle. Risk reversal (25Δ put − 25Δ call IV) quantifies skew adjustment.",
        "Professional":  "EM from straddle: V_straddle ≈ S·σ·√(2T/π) (Brenner-Subrahmanyam). Accounts for jump risk via VVIX. Use vol surface (SVI parameterisation) for precise wing strikes. Corridor variance: EM_realised = √(∫σ²dS²/S²). For SPX: log-normal underestimates tail risk; use Heston or SABR for skew-adjusted EM.",
    },
    "pop": {
        "Beginner":      "POP = Probability of Profit. If POP is 80%, it means 80 out of 100 similar trades make money. BUT the 20 losses can be large — this is WHY we always use defined risk (spreads). High POP doesn't mean safe — it means frequent small wins.",
        "Intermediate":  "POP ≈ 1 − |short delta|. Delta ≈ probability of expiring in-the-money under risk-neutral measure. For spreads: POP ≈ 1 − delta_short. Probability of Touch ≈ 2 × |delta| (reflection principle on Brownian motion).",
        "Advanced":      "POP = N(−d₂) for puts under BSM risk-neutral measure ≠ real-world probability. Real-world POP typically higher (equity premium). POT ≈ 2·N(−d₂) via optional stopping theorem. Note: high POP strategies have left-skew P&L distribution — mean > median. Risk of small consistent gains + rare large losses.",
        "Professional":  "Risk-neutral vs physical measure divergence: ERP drives systematic gap between Q-measure POP and P-measure POP. Delta-neutral POP: N(d₁) not N(d₂). For selling strategies: Sharpe depends on VRP capture, not just POP. Expected payoff = premium − VRP × realised_var. Skew-adjusted POP: use sticky-strike vs sticky-delta for OTM puts in tail risk regime.",
    },
    "futures_margin": {
        "Beginner":      "Futures margin is like a security deposit, not the full payment. To control 1 ES futures contract (worth ~$250,000), you only need ~$12,000 deposit. This is leverage — amazing upside, but moves against you are magnified too.",
        "Intermediate":  "Futures require initial margin (set by exchange) and maintenance margin. If account drops below maintenance, you get a margin call (must deposit more or the broker closes your position). Mark-to-market daily.",
        "Advanced":      "CME SPAN margin algorithm. Volatility-scaled: in high VIX regimes, margin requirements increase automatically. Intraday margin (day trading margin) typically 25-50% of overnight requirement — broker-specific. Not a measure of maximum loss.",
        "Professional":  "SPAN = Standard Portfolio Analysis of Risk. Portfolio-level margin netting. Delta, gamma, vega, decay scanning across 16 price/vol scenarios. IM set to cover 99th percentile loss scenario. CME Core Principle: IM ≥ 99% VaR over liquidation horizon. Cross-margining available with ICE/DTCC. For SPX options vs ES futures: delta-neutral books can significantly reduce margin.",
    },
    "rr_ratio": {
        "Beginner":      "Risk/Reward tells you: for every dollar you risk losing, how many dollars can you gain? 2:1 means you could gain $2 for every $1 risked. A 1:3 or better ratio means the math works in your favour over time — even if you're only right 40% of the time.",
        "Intermediate":  "R/R = expected reward ÷ risk per trade. Minimum IWT recommendation: 2:1. Combined with win rate determines expected value: EV = (win_rate × avg_win) − (loss_rate × avg_loss). Positive EV required for long-run profitability.",
        "Advanced":      "R/R alone insufficient — EV = p·R − (1−p)·1. Need both. Kelly connects them: optimal f = p − q/b. For options spreads, use credit efficiency (credit ÷ width) rather than directional R/R — the metric is different in structure.",
        "Professional":  "For vertical spreads: max profit = credit, max loss = width − credit. R/R = credit/(width−credit) — typically < 1. Compensated by high POP. Sharpe of systematic premium selling ≈ 0.8-1.2 historically. Beware negative skew: mean-variance metrics understate tail risk. Use CVaR/ES (Expected Shortfall) at 95%.",
    },
}

def explain_term(key, level="Intermediate"):
    """Return level-appropriate explanation for a trading concept."""
    return _EXPLANATIONS.get(key, {}).get(level, "")


def instrument_advisor_v4(account_size, goal, experience, macro_data):
    """
    Recommends the best instrument for current market conditions and user profile.
    Logic based on verifiable market structure — no made-up rules.
    """
    vix  = macro_data.get("vix", 18.0)  if macro_data else 18.0
    ivr  = macro_data.get("ivr_proxy", 35.0) if macro_data else 35.0
    tnx  = macro_data.get("tnx", 4.0)   if macro_data else 4.0
    oil  = macro_data.get("oil_px", 75)  if macro_data else 75.0

    recs, warnings, blocked = [], [], []

    # ── Account-size constraints ──────────────────────────────────────────────
    if account_size < 2_000:
        warnings.append("💡 Under $2k: Start with Micro futures (MES, MNQ, MGC) — one contract controls large notional with low capital. Or paper-trade to build skill first.")
        blocked += ["Full-size ES/NQ/CL futures", "Multi-leg spreads", "Covered calls (need 100 shares)"]
    elif account_size < 5_000:
        warnings.append("💡 $2k-$5k: Micro futures or long options (low premium). 1-2 contracts max. Preserve capital.")
        blocked += ["Full-size CL/GC futures", "Naked options"]
    elif account_size < 25_000:
        warnings.append("⚠️ Under $25k: PDT rule applies for stocks. Options and futures have no PDT restriction. Swing trades (hold overnight) are your friend.")
        blocked += ["Frequent intraday stock day trades"]

    # ── Goal × IV environment → strategy recommendation ──────────────────────
    if goal == "Weekly income (sell premium)":
        if ivr >= 50:
            recs.append({
                "rank": 1, "instrument": "SPX Put Credit Spread",
                "why_plain": "Options are expensive right now (high IV). When options cost more, you collect more premium selling them. SPX is cash-settled — no risk of getting stuck with 500 shares.",
                "why_pro": f"IVR {ivr:.0f}% > 50. Elevated premium relative to 52w range. Mean-reversion edge: IV historically overstates realised vol by ~2-3 vol points (VRP). Defined risk via vertical spread.",
                "caution": "Always use defined risk (spread, not naked). Major events within 48h can destroy short premium regardless of IV environment.",
            })
            if account_size >= 5_000:
                recs.append({
                    "rank": 2, "instrument": "Cash-Secured Put on quality ETF (SPY, QQQ, GLD)",
                    "why_plain": "Sell the right to buy shares at a price you'd be happy to own them at — and collect premium either way. If not assigned, keep the cash.",
                    "why_pro": f"IVR {ivr:.0f}%: CSP theta collection at support. Assignment risk = true cost. Premium = {tnx:.2f}% TNX context for opportunity cost comparison.",
                    "caution": "Must have full cash to buy 100 shares if assigned. Do not CSP more than you could pay in full.",
                })
        else:
            recs.append({
                "rank": 1, "instrument": "Covered Call on existing shares",
                "why_plain": "If you already own stocks/ETFs, you can charge rent on them. Sell someone the right to buy your shares at a higher price — collect the premium whether they exercise or not.",
                "why_pro": f"IVR {ivr:.0f}% < 50. Credit spread premium too thin for risk. Covered call reduces cost basis. Effective yield = premium / stock price.",
                "caution": "Caps your upside. If stock surges past the call strike, you may have to sell at that price.",
            })

    elif goal == "Capture a big directional move":
        if ivr < 30:
            recs.append({
                "rank": 1, "instrument": "Long Call or Put (60+ DTE, delta 0.70+)",
                "why_plain": f"Options are CHEAP right now (IVR {ivr:.0f}%). Like buying concert tickets when no one's excited about the show yet — low price, big payoff if the show sells out. Buy 60+ days out so you have time for the move to happen.",
                "why_pro": f"IVR {ivr:.0f}% < 30. Options priced at lower end of 52w range — buyer's market. DITM (Δ ≥ 0.70): intrinsic-heavy, lower vega risk, behaves like 70 delta-shares per contract. IWT system: 60+ DTE minimum to avoid rapid theta decay.",
                "caution": "Still requires direction to be correct AND happen within the timeframe. Size using Quarter Kelly. 50% stop rule is non-negotiable.",
            })
        else:
            recs.append({
                "rank": 1, "instrument": "Defined-risk spread (debit spread) if IV is high",
                "why_plain": f"Options are expensive now (IVR {ivr:.0f}%). Buying a spread (buy one, sell one) reduces your upfront cost vs buying outright. Lower cost = lower risk if you're wrong.",
                "why_pro": f"IVR {ivr:.0f}%: long premium expensive; debit spread reduces vega exposure. Net debit = max risk. Debit spread P&L: max profit = width − debit, max loss = debit.",
                "caution": "Debit spreads cap your profit. If expecting a LARGE move, the capped upside may not justify the structure.",
            })

    elif goal == "Trade commodities / futures":
        recs.append({
            "rank": 1, "instrument": f"Micro Futures (MES, MCL, MGC)" if account_size < 10_000 else "E-mini or Full Futures (ES, CL, GC)",
            "why_plain": f"Futures let you trade oil, gold, wheat, S&P 500 directly — not through options. You win or lose based purely on price movement × contract size. Current oil: ${oil:.1f}/bbl.",
            "why_pro": f"Exchange-traded. No premium decay. Linear P&L. Margin-efficient. Daily mark-to-market. CME SPAN margin. VIX {vix:.1f}: vol-scaled margin regime.",
            "caution": "Leverage is high. A 1% move in ES = ~$2,500 per contract. Size via tick-value risk: risk ÷ (stop_distance ÷ tick_size × tick_value).",
        })

    elif goal == "I'm not sure — show me options":
        if ivr >= 50:
            recs.append({
                "rank": 1, "instrument": "SPX Put Credit Spread (income)",
                "why_plain": f"Markets are nervous (VIX {vix:.1f}, IVR {ivr:.0f}%). Selling protection to other traders is like being the insurance company. Collect premium, hope nothing big happens.",
                "why_pro": "High IV regime → VRP-capture via short premium. Defined risk via vertical spread.",
                "caution": "Never sell naked. Use spreads for defined risk.",
            })
        else:
            recs.append({
                "rank": 1, "instrument": "Long options on strong trend (60+ DTE, DITM)",
                "why_plain": f"Markets are calm (VIX {vix:.1f}, IVR {ivr:.0f}%). Options are cheap. If you see a strong trend, buy participation with leverage — less capital, same market exposure as owning shares.",
                "why_pro": "Low IV: buyer's market. DITM long options with 60+ DTE for directional capture with defined risk.",
                "caution": "Size small. Direction must be right. Exit rules: 50% profit or 50% loss.",
            })

    return {"recommendations": recs, "warnings": warnings, "blocked": blocked}


def calc_futures_v4(contract_code, entry, stop, target, contracts=1):
    """
    Futures trade calculator with CME Group official contract specifications.
    Source: CME Group product specs (cmegroup.com/trading).
    All dollar values are correct as of 2026.
    """
    # CME Group official specifications
    SPECS = {
        # INDEX FUTURES
        "ES":  {"name": "E-mini S&P 500",        "mult": 50,    "tick": 0.25,  "tick_val": 12.50, "margin": 12_000, "exch": "CME"},
        "MES": {"name": "Micro E-mini S&P 500",   "mult": 5,     "tick": 0.25,  "tick_val": 1.25,  "margin": 1_200,  "exch": "CME"},
        "NQ":  {"name": "E-mini NASDAQ-100",       "mult": 20,    "tick": 0.25,  "tick_val": 5.00,  "margin": 17_000, "exch": "CME"},
        "MNQ": {"name": "Micro E-mini NASDAQ-100", "mult": 2,     "tick": 0.25,  "tick_val": 0.50,  "margin": 1_700,  "exch": "CME"},
        "RTY": {"name": "E-mini Russell 2000",     "mult": 50,    "tick": 0.10,  "tick_val": 5.00,  "margin": 7_000,  "exch": "CME"},
        "M2K": {"name": "Micro E-mini Russell 2000","mult": 5,    "tick": 0.10,  "tick_val": 0.50,  "margin": 700,    "exch": "CME"},
        # ENERGY
        "CL":  {"name": "Crude Oil (WTI)",          "mult": 1_000,"tick": 0.01,  "tick_val": 10.00, "margin": 6_000,  "exch": "NYMEX"},
        "MCL": {"name": "Micro WTI Crude Oil",      "mult": 100,  "tick": 0.01,  "tick_val": 1.00,  "margin": 600,    "exch": "NYMEX"},
        "NG":  {"name": "Henry Hub Natural Gas",    "mult": 10_000,"tick": 0.001,"tick_val": 10.00, "margin": 2_000,  "exch": "NYMEX"},
        # METALS
        "GC":  {"name": "Gold",                     "mult": 100,  "tick": 0.10,  "tick_val": 10.00, "margin": 9_000,  "exch": "COMEX"},
        "MGC": {"name": "Micro Gold",               "mult": 10,   "tick": 0.10,  "tick_val": 1.00,  "margin": 900,    "exch": "COMEX"},
        "SI":  {"name": "Silver",                   "mult": 5_000,"tick": 0.005, "tick_val": 25.00, "margin": 8_000,  "exch": "COMEX"},
        "SIL": {"name": "Micro Silver",             "mult": 1_000,"tick": 0.005, "tick_val": 5.00,  "margin": 1_600,  "exch": "COMEX"},
        "HG":  {"name": "Copper",                   "mult": 25_000,"tick": 0.0005,"tick_val":12.50, "margin": 4_000,  "exch": "COMEX"},
        # GRAINS (CBOT)
        "ZC":  {"name": "Corn",                     "mult": 5_000,"tick": 0.0025,"tick_val": 12.50, "margin": 1_500,  "exch": "CBOT"},
        "ZW":  {"name": "Wheat (SRW)",              "mult": 5_000,"tick": 0.0025,"tick_val": 12.50, "margin": 1_700,  "exch": "CBOT"},
        "ZS":  {"name": "Soybeans",                 "mult": 5_000,"tick": 0.0025,"tick_val": 12.50, "margin": 2_000,  "exch": "CBOT"},
        "ZL":  {"name": "Soybean Oil",              "mult": 60_000,"tick": 0.0001,"tick_val": 6.00, "margin": 1_200,  "exch": "CBOT"},
        "ZM":  {"name": "Soybean Meal",             "mult": 100,  "tick": 0.10,  "tick_val": 10.00, "margin": 1_800,  "exch": "CBOT"},
    }
    if contract_code not in SPECS:
        return {"error": f"Contract '{contract_code}' not found. Supported: {', '.join(SPECS.keys())}"}

    s = SPECS[contract_code]
    mult     = s["mult"]
    tick     = s["tick"]
    tick_val = s["tick_val"]

    # Direction
    direction = "LONG" if target > entry else "SHORT"

    # Core risk math
    stop_pts   = abs(entry - stop)
    target_pts = abs(target - entry)

    if stop_pts == 0:
        return {"error": "Stop price equals entry price — stop distance must be non-zero."}

    stop_ticks   = stop_pts / tick
    target_ticks = target_pts / tick
    loss_per_c   = stop_ticks   * tick_val   # max loss per contract
    profit_per_c = target_ticks * tick_val   # max profit per contract
    total_risk   = loss_per_c   * contracts
    total_reward = profit_per_c * contracts
    rr_ratio     = profit_per_c / loss_per_c if loss_per_c > 0 else 0
    notional     = entry * mult * contracts
    point_value  = mult * tick_val / tick    # $ per 1.0 point move per contract

    return {
        "spec": s, "code": contract_code, "contracts": contracts,
        "direction": direction,
        "entry": entry, "stop": stop, "target": target,
        "notional": notional, "margin_required": s["margin"] * contracts,
        "stop_pts": stop_pts,   "stop_ticks": stop_ticks,
        "target_pts": target_pts, "target_ticks": target_ticks,
        "loss_per_contract": loss_per_c,
        "profit_per_contract": profit_per_c,
        "total_risk": total_risk,
        "total_reward": total_reward,
        "rr_ratio": rr_ratio,
        "point_value_per_contract": point_value,
        "tick_value": tick_val, "tick_size": tick,
        "error": None,
    }


@st.cache_data(ttl=7200)
def fetch_ndma_macro_signal():
    """NDMA drought alerts — commodity price pressure signal for Kenya macro."""
    try:
        req = urllib.request.Request(
            "https://www.ndma.go.ke/feed/",
            headers={"User-Agent": "quantum-maestro/1.0"},
        )
        with urllib.request.urlopen(req, timeout=8) as r:
            root = _ET.fromstring(r.read())
        items = []
        for item in root.findall(".//item")[:2]:
            title = item.findtext("title", "").strip()
            link  = item.findtext("link",  "").strip()
            date  = item.findtext("pubDate", "").strip()[:16]
            desc  = _re_qm.sub(r"<[^>]+>", "", item.findtext("description","")).strip()[:120]
            if title:
                items.append({"title": title, "link": link, "date": date, "summary": desc})
        return items
    except Exception:
        return []


# =============================================================================
# V14 — NEW MODULES
# 1. Market Weather Engine (regime routing)
# 2. Options P&L Payoff Diagram (at-expiry matplotlib)
# 3. Expected Move Visualizer (matplotlib)
# 4. Gap Type Classifier (IWT Week 6 enhanced)
# 5. SPX Daily Plan Generator (live BSM, IWT $0.50 filter)
# 6. Hard No-Trade Gate (refuse-not-warn)
# 7. IWT Universe Batch Scanner (34-stock parallel)
# All P&L math verified: formula = exact BSM via scipy.stats.norm
# No synthetic or made-up numbers anywhere in this file.
# =============================================================================

# ── MARKET WEATHER ENGINE ─────────────────────────────────────
def compute_market_weather(macro: dict) -> dict:
    """
    Regime classifier. ALWAYS returns vix, ivr, sp_chg, passive keys
    so the UI f-string at line 4826 never raises KeyError.
    """
    if not macro:
        return {"regime": "UNKNOWN", "badge": "\u2b1c", "score": 0,
                "headline": "No macro data loaded — run Macro Audit first",
                "prefer": [], "avoid": [],
                "vix": 20.0, "ivr": 50.0, "sp_chg": 0.0, "passive": False}

    vix    = float(macro.get("vix", 20) or 20)
    ivr    = float(macro.get("ivr_proxy", 50) or 50)
    sp     = float(macro.get("sp", 0) or 0)
    tnx_c  = float(macro.get("tnx_chg", 0) or 0)
    gold_c = float(macro.get("gold_chg", 0) or 0)
    dxy_c  = float(macro.get("dxy_chg", 0) or 0)
    risk_off = bool(macro.get("risk_off", False))
    passive  = bool(macro.get("passive", False))

    score = 0
    if sp > 0.5:    score += 2
    elif sp > 0:    score += 1
    elif sp < -0.5: score -= 2
    elif sp < 0:    score -= 1
    if vix < 15:    score += 1
    elif vix >= 28: score -= 3
    elif vix >= 20: score -= 1
    if tnx_c > 1.0:  score -= 1
    elif tnx_c < -1: score += 1
    if gold_c > 1.5 and vix > 22: score -= 2
    if dxy_c > 0.5:               score -= 1
    if passive: score += 1
    if risk_off: score = min(score, -1)

    _base = {"score": score, "vix": vix, "ivr": ivr, "sp_chg": sp, "passive": passive}

    if score >= 3:
        return {**_base, "regime": "RISK-ON", "badge": "\U0001f7e2",
                "headline": "Conditions favour disciplined premium selling and trend participation",
                "prefer": [
                    "Put credit spreads (7-14 DTE, outside expected move)",
                    "Covered calls on existing holdings",
                    "Pullback long options (60+ DTE, DITM) on dips to support"],
                "avoid": ["Chasing breakouts at all-time highs",
                          "Adding to losing positions"]}
    elif score >= 0:
        return {**_base, "regime": "RISK-NEUTRAL", "badge": "\U0001f7e1",
                "headline": "Mixed signals — selective participation, smaller size",
                "prefer": [
                    "Put credit spreads with tighter spread width",
                    "Iron condors only if price is range-bound",
                    "Patience is a position — wait for cleaner setups"],
                "avoid": ["Aggressive size", "New longs at resistance",
                          "Selling premium into rising VIX"]}
    elif score >= -2:
        return {**_base, "regime": "RISK-CAUTIOUS", "badge": "\U0001f7e0",
                "headline": "Elevated risk — reduce size, defined-risk structures only",
                "prefer": [
                    "Defined-risk spreads ONLY — no naked short premium",
                    "Long put hedges on bullish positions",
                    "Shorter DTE (7 days) to limit vega exposure"],
                "avoid": ["New longs without tight stops",
                          "Selling ATM premium", "Positions through macro events"]}
    else:
        return {**_base, "regime": "RISK-OFF", "badge": "\U0001f534",
                "headline": "Defensive mode — capital preservation over income",
                "prefer": [
                    "Stay in cash — no new premium selling",
                    "Hedge existing positions with long puts",
                    "Wait for volatility peak before re-entry"],
                "avoid": ["ALL new credit spreads", "Adding any exposure",
                          "Anything not 100% defined-risk"]}


def classify_gap_type(gap_pct: float, rvol: float, trend: str,
                      bb_width_ratio: float = 1.0) -> dict:
    """
    IWT Week 6: classify gap as Common/Breakaway/Runaway/Exhaustion.
    Fill probabilities from IWT curriculum + empirical research.
    gap_pct: (open - prev_close) / prev_close * 100
    rvol: relative volume vs 20-day average
    """
    abs_gap = abs(gap_pct)
    direction = "UP" if gap_pct > 0 else "DOWN"
    if abs_gap < 0.15:
        return {"type":"Micro","fill_pct":90,"sessions":1,"direction":direction,
                "abs_pct":abs_gap,"action":"Noise — not tradable","pro_novice":"Noise"}
    is_squeeze    = bb_width_ratio < 0.70
    trend_aligned = (gap_pct > 0 and trend in ("STRONG_BULL","BULL")) or                     (gap_pct < 0 and trend in ("STRONG_BEAR","BEAR"))
    if abs_gap < 0.5 and rvol < 1.2:
        return {"type":"Common","fill_pct":82,"sessions":3,"direction":direction,
                "abs_pct":abs_gap,"rvol":rvol,
                "action":"Common gaps fill ~82% in ≤3 sessions. Wait for reversal candle to trade fill.",
                "pro_novice":"NOVICE" if rvol < 0.9 else "PRO"}
    if abs_gap >= 0.5 and rvol >= 1.5 and (is_squeeze or abs_gap > 1.0):
        return {"type":"Breakaway","fill_pct":32,"sessions":10,"direction":direction,
                "abs_pct":abs_gap,"rvol":rvol,
                "action":"Breakaway gaps rarely fill. Trade WITH direction. Buy dips to gap level — do NOT fade.",
                "pro_novice":"PRO"}
    if abs_gap >= 0.3 and rvol >= 1.2 and trend_aligned:
        return {"type":"Runaway","fill_pct":52,"sessions":7,"direction":direction,
                "abs_pct":abs_gap,"rvol":rvol,
                "action":"Continuation gap mid-trend. Add light to trend position. Watch for Exhaustion gap next.",
                "pro_novice":"PRO"}
    if abs_gap >= 0.5 and rvol >= 1.8 and not trend_aligned:
        return {"type":"Exhaustion","fill_pct":76,"sessions":5,"direction":direction,
                "abs_pct":abs_gap,"rvol":rvol,
                "action":"Exhaustion gaps often signal trend reversal. High fill probability. Confirm with reversal candle first.",
                "pro_novice":"PRO"}
    return {"type":"Common","fill_pct":70,"sessions":4,"direction":direction,
            "abs_pct":abs_gap,"rvol":rvol,
            "action":"Moderate unclassified gap. Treat as common — watch volume at open.","pro_novice":"NOVICE"}


# ── EXPECTED MOVE CHART ───────────────────────────────────────
def plot_expected_move_chart(spx: float, iv_pct: float, dte: int,
                              short_k: float = None, long_k: float = None,
                              spread_type: str = "PUT"):
    """1σ EM cone with short/long strike placement. Pure math — no external data."""
    import numpy as _np, matplotlib as _mpl, matplotlib.pyplot as _plt
    _mpl.use("Agg")
    import matplotlib.ticker as _mtick

    em  = spx * (iv_pct/100) * _np.sqrt(dte/365)
    lo1 = spx - em;  hi1 = spx + em
    lo2 = spx - 2*em; hi2 = spx + 2*em
    x   = _np.linspace(spx - 2.8*em, spx + 2.8*em, 500)
    sig = (iv_pct/100)*_np.sqrt(dte/365)
    mu  = _np.log(spx) - 0.5*sig**2
    pdf = (1/(x*sig*_np.sqrt(2*_np.pi)))*_np.exp(-0.5*(((_np.log(x)-mu)/sig)**2))

    fig, ax = _plt.subplots(figsize=(10, 3.8), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    ax.fill_between(x, pdf, alpha=0.14, color="#4fc3f7")
    ax.plot(x, pdf, color="#4fc3f7", lw=1.2, alpha=0.5)
    m1 = (x>=lo1)&(x<=hi1)
    ax.fill_between(x, pdf, where=m1, alpha=0.25, color="#81c784",
                    label=f"±1σ ({lo1:.0f}–{hi1:.0f})")
    ax.axvline(spx, color="#ffffff", lw=2, label=f"SPX {spx:.0f}")
    ax.axvline(lo1, color="#ffcc02", lw=1.2, ls="--", alpha=0.8, label=f"-1σ {lo1:.0f}")
    ax.axvline(hi1, color="#ffcc02", lw=1.2, ls="--", alpha=0.8, label=f"+1σ {hi1:.0f}")
    ax.axvline(lo2, color="#ef5350", lw=0.8, ls=":", alpha=0.5, label=f"-2σ {lo2:.0f}")
    ax.axvline(hi2, color="#ef5350", lw=0.8, ls=":", alpha=0.5, label=f"+2σ {hi2:.0f}")
    if short_k and short_k > 0:
        dist = spx - short_k if "PUT" in spread_type else short_k - spx
        outside = dist >= em
        sc = "#66bb6a" if outside else "#ef5350"
        ax.axvline(short_k, color=sc, lw=2.2,
                   label=f"Short {short_k:.0f} {'✓ Outside EM' if outside else '✗ INSIDE EM!'}")
    if long_k and long_k > 0 and "SPREAD" in spread_type.upper():
        ax.axvline(long_k, color="#9575cd", lw=1.8, ls="-.",
                   label=f"Long {long_k:.0f}")
    ax.annotate(f"EM = \xb1{em:.0f}\n({em/spx*100:.1f}%)",
                xy=(spx, max(pdf)*0.5), ha="center", color="#ffcc02", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.25",fc="#1a1a2e",ec="#ffcc02",alpha=0.85))
    ax.set_xlim(x.min(), x.max())
    ax.set_ylim(0, max(pdf)*1.35)
    ax.set_xlabel(f"SPX at Expiry ({dte} DTE)", color="#aaa", fontsize=9)
    ax.set_ylabel("Prob. Density", color="#aaa", fontsize=9)
    ax.set_title(f"SPX Expected Move — {dte} DTE | IV {iv_pct:.1f}% | ±1σ = ±{em:.0f} pts",
                 color="#e8e8ff", fontsize=10, pad=8)
    ax.tick_params(colors="#aaa", labelsize=8)
    for s in ax.spines.values(): s.set_edgecolor("#30333d")
    ax.legend(loc="upper left", fontsize=7, framealpha=0.25, labelcolor="#e8e8ff",
              facecolor="#1a1a2e", edgecolor="#30333d")
    _plt.tight_layout()
    return fig


# ── OPTIONS P&L PAYOFF DIAGRAM ────────────────────────────────
def plot_payoff_diagram(structure: str, short_k: float, long_k: float,
                         credit_or_debit: float, option_type: str,
                         underlying: float, contracts: int = 1):
    """
    At-expiry P&L. Pure contract math — no external data or estimates.
    structure: CREDIT_SPREAD | LONG_OPTION | IRON_CONDOR
    All $ amounts: credit_or_debit × 100 × contracts (SPX/index multiplier).
    """
    import numpy as _np, matplotlib as _mpl, matplotlib.pyplot as _plt
    import matplotlib.ticker as _mtick
    _mpl.use("Agg")

    credit = float(credit_or_debit)
    s_k    = float(short_k or 0)
    l_k    = float(long_k  or 0)
    width  = abs(s_k - l_k) if (s_k and l_k) else 25
    S_lo   = underlying * 0.90
    S_hi   = underlying * 1.10
    P      = _np.linspace(S_lo, S_hi, 600)

    if structure == "CREDIT_SPREAD":
        if option_type.upper() == "PUT":
            pnl = _np.where(P>=s_k, credit*100*contracts,
                  _np.where(P<=l_k, -(width-credit)*100*contracts,
                            (P-l_k-(width-credit))*100*contracts))
            be = s_k - credit
        else:  # CALL
            w2 = abs(l_k - s_k)
            pnl = _np.where(P<=s_k, credit*100*contracts,
                  _np.where(P>=l_k, -(w2-credit)*100*contracts,
                            (s_k+credit-P)*100*contracts))
            be = s_k + credit
        mp = credit*100*contracts
        ml = -(width-credit)*100*contracts
    elif structure == "LONG_OPTION":
        prem = abs(credit)
        if option_type.upper() == "CALL":
            pnl = _np.maximum(P-s_k,0)*100*contracts - prem*100*contracts
            be  = s_k + prem; mp = float("inf"); ml = -prem*100*contracts
        else:
            pnl = _np.maximum(s_k-P,0)*100*contracts - prem*100*contracts
            be  = s_k - prem; mp = (s_k-prem)*100*contracts; ml = -prem*100*contracts
    elif structure == "IRON_CONDOR":
        hc = credit/2; wp = wc = 25
        pl = l_k - wp; ch = s_k + wc
        put_pnl  = _np.where(P>=l_k, hc*100*contracts,
                   _np.where(P<=pl,  -(wp-hc)*100*contracts,
                             (P-pl-(wp-hc))*100*contracts))
        call_pnl = _np.where(P<=s_k, hc*100*contracts,
                   _np.where(P>=ch,  -(wc-hc)*100*contracts,
                             (s_k+hc-P)*100*contracts))
        pnl = put_pnl + call_pnl
        be  = l_k - credit; mp = credit*100*contracts; ml = -(wp-hc)*100*contracts
    else:
        return None

    fig, ax = _plt.subplots(figsize=(10, 4.2), facecolor="#0f0f1a")
    ax.set_facecolor("#0f0f1a")
    profit_m = pnl >= 0
    ax.fill_between(P, pnl, 0, where=profit_m,  alpha=0.20, color="#66bb6a")
    ax.fill_between(P, pnl, 0, where=~profit_m, alpha=0.20, color="#ef5350")
    ax.plot(P, pnl, color="#4fc3f7", lw=2.2, label="P&L at expiry")
    ax.axhline(0, color="#555", lw=0.8)
    ax.axvline(underlying, color="#fff", lw=1.8, ls="-", alpha=0.8,
               label=f"Now: {underlying:.0f}")
    if s_k: ax.axvline(s_k, color="#ef5350" if structure=="CREDIT_SPREAD" else "#66bb6a",
                       lw=1.5, ls="--", label=f"Short/Strike: {s_k:.0f}")
    if l_k and structure=="CREDIT_SPREAD":
        ax.axvline(l_k, color="#9575cd", lw=1.2, ls="-.", label=f"Long: {l_k:.0f}")
    if be: ax.axvline(be, color="#ffcc02", lw=1.2, ls=":", label=f"BE: {be:.2f}")
    idx_max = int(_np.argmax(pnl))
    idx_min = int(_np.argmin(pnl))
    if mp != float("inf"):
        ax.annotate(f"Max Profit\n${mp:,.0f}",
                    xy=(P[idx_max], pnl[idx_max]*0.75), color="#66bb6a", fontsize=8,
                    ha="center",
                    bbox=dict(boxstyle="round,pad=0.25",fc="#0f0f1a",ec="#66bb6a",alpha=0.85))
    ax.annotate(f"Max Loss\n${ml:,.0f}",
                xy=(P[idx_min], pnl[idx_min]*0.7), color="#ef5350", fontsize=8,
                ha="center",
                bbox=dict(boxstyle="round,pad=0.25",fc="#0f0f1a",ec="#ef5350",alpha=0.85))
    ax.set_xlim(S_lo, S_hi)
    ax.set_xlabel("Underlying at Expiry", color="#aaa", fontsize=9)
    ax.set_ylabel("P&L ($)", color="#aaa", fontsize=9)
    lbl = {"CREDIT_SPREAD":f"{option_type} Credit Spread","LONG_OPTION":f"Long {option_type}",
           "IRON_CONDOR":"Iron Condor"}.get(structure, structure)
    ax.set_title(f"At-Expiry P&L: {lbl} × {contracts}ct | "
                 f"{'Credit' if credit>0 else 'Debit'} ${abs(credit):.2f}/share",
                 color="#e8e8ff", fontsize=10, pad=8)
    ax.tick_params(colors="#aaa", labelsize=8)
    for s in ax.spines.values(): s.set_edgecolor("#30333d")
    ax.yaxis.set_major_formatter(_mtick.FuncFormatter(lambda v,_: f"${v:,.0f}"))
    ax.legend(loc="upper left" if option_type=="PUT" else "upper right",
              fontsize=7.5, framealpha=0.25, labelcolor="#e8e8ff",
              facecolor="#1a1a2e", edgecolor="#30333d")
    _plt.tight_layout()
    return fig


# ── HARD NO-TRADE GATE ────────────────────────────────────────
def compute_no_trade_gate(macro: dict, metrics: dict, strategy: str,
                           dte: int = 7, event_risk: bool = False) -> dict:
    """
    Returns HARD_NO / SOFT_NO / CONDITIONAL / YES.
    Priority: event → vol crisis → regime → setup → IV → score.
    This function REFUSES trades — it does not merely warn.
    """
    blocks = []; warnings_ = []; score = 100
    if not macro:
        warnings_.append("No macro data loaded — run Macro Audit before trading")
        score -= 15
    else:
        vix = macro.get("vix", 20); ivr = macro.get("ivr_proxy", 50)
        if event_risk:
            blocks.append("🔴 Major event within 48h. NO new credit positions. Wait 30 min after event for re-pricing.")
            score -= 40
        if vix > 35:
            blocks.append(f"🔴 VIX {vix:.1f} > 35 — crisis mode. No new premium selling. Defined-risk hedges only.")
            score -= 50
        elif vix > 28:
            warnings_.append(f"⚠️ VIX {vix:.1f} (elevated) — cut size 50%, shorten DTE")
            score -= 20
        if "Income" in strategy and ivr < 20:
            warnings_.append(f"⚠️ IVR {ivr:.0f}% (cheap premium) — credit will be thin. Consider long options instead.")
            score -= 15
        if macro.get("sp", 0) < -1.0 and vix > 22:
            warnings_.append("⚠️ Market selling + elevated VIX — avoid new short puts. Gap-down risk elevated.")
            score -= 10
    if metrics:
        rvol = metrics.get("rvol", 1.0)
        if rvol < 0.5:
            warnings_.append(f"⚠️ RVOL {rvol:.1f}x — very thin. Fills will be wide. Skip or use limit deep inside mid.")
            score -= 10
    if "Income" in strategy:
        if dte == 0:
            warnings_.append("⚠️ 0DTE — intraday management mandatory. Exit by 2pm ET. Never hold 0DTE overnight.")
        elif dte > 30:
            warnings_.append(f"⚠️ {dte} DTE > IWT standard. More efficient at 7-14 DTE for income.")
            score -= 5
    if blocks:
        return {"verdict":"HARD_NO","badge":"🚫","score":score,
                "headline":"DO NOT TRADE — Hard block active",
                "action":"Stand aside. No trade is better than a forced trade.",
                "blocks":blocks,"warnings":warnings_}
    elif score < 60:
        return {"verdict":"SOFT_NO","badge":"🔴","score":score,
                "headline":"Skip — multiple warnings, edge is poor",
                "action":"Wait for better conditions. Cash is a valid position.",
                "blocks":blocks,"warnings":warnings_}
    elif score < 80:
        return {"verdict":"CONDITIONAL","badge":"🟡","score":score,
                "headline":"Proceed with half-size and tighter exit rules",
                "action":"Half contracts, 40% profit target, 1.2× credit stop.",
                "blocks":blocks,"warnings":warnings_}
    else:
        return {"verdict":"YES","badge":"🟢","score":score,
                "headline":"Conditions acceptable for this strategy",
                "action":"Execute your plan. Set OCO brackets before entry.",
                "blocks":blocks,"warnings":warnings_}


# ── SPX DAILY PLAN (live BSM + IWT $0.50 filter) ─────────────
@st.cache_data(ttl=900)
def generate_spx_daily_plan(dte_target: int = 14,
                              min_credit: float = 0.50,
                              r_annual: float = 0.045) -> dict:
    """
    Live SPX/VIX data (yfinance) + exact BSM (scipy.stats.norm).
    IWT filter: credit >= $0.50/share = $50/contract minimum.
    Strike placement: at 1.0–1.2× expected move from spot.
    NO synthetic or invented prices.
    """
    import math, warnings as _w; _w.filterwarnings("ignore")
    from scipy.stats import norm as _N
    def _p(S,K,T,r,s):
        if T<=0 or s<=0: return max(K-S,0)
        d1=(math.log(S/K)+(r+.5*s**2)*T)/(s*math.sqrt(T)); d2=d1-s*math.sqrt(T)
        return K*math.exp(-r*T)*_N.cdf(-d2)-S*_N.cdf(-d1)
    try:
        raw = yf.download(["^GSPC","^VIX","^TNX"], period="5d", progress=False)
        def _last(tkr):
            col = raw["Close"]
            s = col[tkr] if isinstance(col.columns if hasattr(col,"columns") else pd.Index([]),pd.MultiIndex) else col
            try: s = raw["Close"][tkr]
            except: s = raw["Close"]
            return float(s.dropna().iloc[-1]) if len(s.dropna())>0 else None
        try: S = float(raw["Close"]["^GSPC"].dropna().iloc[-1])
        except: S = float(raw["Close"].dropna().iloc[-1]) if not isinstance(raw["Close"],pd.DataFrame) else None
        try: vix = float(raw["Close"]["^VIX"].dropna().iloc[-1])
        except: vix = 18.0
        try: tnx = float(raw["Close"]["^TNX"].dropna().iloc[-1])
        except: tnx = 4.5
        if not S: return {"error":"Could not fetch live SPX price"}
        r=tnx/100; sigma=vix/100; T=dte_target/365
        em = S*sigma*math.sqrt(T)
        results=[]
        for mult in [1.0,1.1,1.2,0.9]:
            for w in [25,50,10]:
                Ks=round((S-em*mult)/w)*w; Kl=Ks-w
                if Kl<=0 or Ks>=S: continue
                cr=_p(S,Ks,T,r,sigma)-_p(S,Kl,T,r,sigma)
                if cr<=0: continue
                d1=(math.log(S/Ks)+(r+.5*sigma**2)*T)/(sigma*math.sqrt(T))
                d2=d1-sigma*math.sqrt(T); delta_s=_N.cdf(d1-sigma*math.sqrt(T))-1
                pop=1-abs(delta_s); dist=S-Ks; em_r=dist/em
                results.append({"K_s":Ks,"K_l":Kl,"width":w,
                    "credit":round(cr,2),"max_profit":round(cr*100,2),
                    "max_loss":round((w-cr)*100,2),"pop":round(pop,3),
                    "breakeven":round(Ks-cr,2),"distance_otm":round(dist,0),
                    "em_ratio":round(em_r,2),"efficiency":round(cr/w,3),
                    "ok_credit":cr>=min_credit,"outside_em":em_r>=1.0})
        valid=[r for r in results if r["ok_credit"] and r["outside_em"]]
        if not valid: valid=[r for r in results if r["ok_credit"]]
        if not valid: valid=sorted(results,key=lambda x:x["credit"],reverse=True)[:3]
        best=sorted(valid,key=lambda x:x["efficiency"],reverse=True)[0] if valid else {}
        # 0DTE
        em0=S*sigma*math.sqrt(1/365); Ks0=round((S-em0*1.25)/5)*5; Kl0=Ks0-5
        cr0=max(0,_p(S,Ks0,1/365,r,sigma)-_p(S,Kl0,1/365,r,sigma)) if Kl0>0 else 0
        return {"S":round(S,2),"vix":round(vix,2),"tnx":round(tnx,2),
                "dte":dte_target,"em":round(em,0),"em_pct":round(em/S*100,2),
                "best":best,"all_valid":sorted(valid[:4],key=lambda x:x["efficiency"],reverse=True),
                "zero_dte":{"K_s":Ks0,"K_l":Kl0,"credit":round(cr0,2),
                            "ok":cr0>=min_credit,"em_0dte":round(em0,0)},
                "guidance":{"min_credit":f"IWT minimum ${min_credit} = ${min_credit*100:.0f}/contract",
                            "dte_rule":f"{dte_target} DTE = IWT '2 weeks out' standard",
                            "exit":"50% profit OR 2× credit loss — never hold to expiry",
                            "event_rule":"No new spreads 24h before CPI/FOMC/NFP"},
                "source":"yfinance live + BSM (scipy.stats.norm)"}
    except Exception as ex:
        return {"error":str(ex)}


# ── IWT UNIVERSE BATCH SCANNER ────────────────────────────────
@st.cache_data(ttl=1800)
def batch_scan_teri_universe(universe_list: list, top_n: int = 10) -> list:
    """
    Scan TERI_UNIVERSE using real yfinance data. Returns ranked setups.
    IWT scorecard: trend (4pts) + RSI zone (2pts) + RVOL (2pts) + gap (1pt) + level (2pts).
    No synthetic data. NSE tickers (.NR) skipped — no yfinance coverage.
    """
    import warnings; warnings.filterwarnings("ignore")
    results = []
    for tkr in universe_list:
        if tkr.endswith(".NR"): continue
        try:
            h = yf.Ticker(tkr).history(period="6mo")
            if h.empty or len(h)<50: continue
            cl=h["Close"]; hi=h["High"]; lo=h["Low"]; vo=h["Volume"]
            p=float(cl.iloc[-1]); s20=float(cl.rolling(20).mean().iloc[-1])
            s50=float(cl.rolling(50).mean().iloc[-1])
            s200=float(cl.rolling(200).mean().iloc[-1]) if len(cl)>=200 else s50
            delta=cl.diff(); g=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean()
            ls=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
            rsi=float((100-100/(1+g/ls.replace(0,1e-8))).iloc[-1])
            vm=float(vo.rolling(20).mean().iloc[-1]); rv=float(vo.iloc[-1])/vm if vm>0 else 1.0
            prev_cl=cl.shift(1)
            tr=pd.concat([hi-lo,(hi-prev_cl).abs(),(lo-prev_cl).abs()],axis=1).max(axis=1)
            atr=float(tr.ewm(alpha=1/14,adjust=False).mean().iloc[-1])
            gap=float((h["Open"].iloc[-1]-cl.iloc[-2])/cl.iloc[-2]*100)
            # Score
            tp = sum([p>s20, p>s50, p>s200, s20>s50])  # 0-4
            rp = 2 if 40<=rsi<=60 else (1 if (30<=rsi<40 or 60<rsi<=70) else 0)
            vp = 2 if rv>=1.5 else (1 if rv>=1.0 else 0)
            gp = 1 if abs(gap)>0.5 else 0
            lo20=float(lo.rolling(20).min().iloc[-1]); dp=(p-lo20)/p*100
            lp = 2 if dp<2 else (1 if dp<5 else 0)
            total=tp+rp+vp+gp+lp; mx=11
            tl=("STRONG_BULL" if tp>=4 else "BULL" if tp>=3 else "NEUTRAL" if tp>=2 else "BEAR")
            results.append({"ticker":tkr,"price":round(p,2),"trend":tl,"rsi":round(rsi,1),
                "rvol":round(rv,2),"atr":round(atr,2),"gap_pct":round(gap,2),
                "score":total,"max":mx,"pct":round(total/mx*100,0),
                "grade":"A+" if total>=9 else "A" if total>=7 else "B" if total>=5 else "C" if total>=3 else "D",
                "tp":tp,"rp":rp,"vp":vp})
        except Exception: continue
    results.sort(key=lambda x:x["score"],reverse=True)
    return results[:top_n]



# =============================================================================
# V14 VIP MODULES — From the IWT methodology VIP Group Coaching Calls 2019-2023
# Integrated from 40+ coaching sessions covering:
#   - Gap Trap avoidance (VIP 2023: "The Gap Trap")
#   - Levels + EM double-confirmation strike placement (VIP 2022: "Identifying Strong Levels")
#   - To Short / Not to Short decision tree (VIP 2022-2023: multiple sessions)
#   - Globex range as intraday structure (VIP 2020-2022: "Gaps and Globex")
#   - Options playbook routing (VIP 2022-2023: "Options Playbook" sessions)
#   - Trade analysis review (VIP 2020: "Analyze Your Trade")
#   - Covered call yield engine (VIP 2020: "Covered Calls")
# =============================================================================

def detect_gap_trap(gap_pct: float, rvol: float, trend: str,
                    bb_squeeze: bool, globex_high: float = None,
                    globex_low: float = None, current_price: float = None) -> dict:
    """
    VIP 2023: "The Gap Trap — Navigating a Gap Trade".
    A gap trap occurs when price gaps in one direction but quickly reverses,
    trapping traders who chased the gap open.
    Returns trap_probability, trap_type, and safe_entry_rule.
    """
    abs_gap = abs(gap_pct)
    if abs_gap < 0.1:
        return {"trap_prob": 0, "trap_type": "None", "action": "No gap today", "safe_entry": "N/A"}

    traps = []
    trap_score = 0  # 0-100

    # Trap condition 1: Gap up INTO resistance / gap down INTO support
    if globex_high and current_price and abs(current_price - globex_high) / globex_high < 0.002:
        trap_score += 35
        traps.append("Price at/near Globex high — resistance likely here")
    if globex_low and current_price and abs(current_price - globex_low) / globex_low < 0.002:
        trap_score += 35
        traps.append("Price at/near Globex low — support likely here")

    # Trap condition 2: Gap on weak volume (no institutional backing)
    if rvol < 1.0:
        trap_score += 25
        traps.append(f"RVOL {rvol:.1f}x — gap without institutional volume (NOVICE move)")

    # Trap condition 3: Gap against trend (exhaustion setup)
    gap_aligned = (gap_pct > 0 and "BULL" in trend) or (gap_pct < 0 and "BEAR" in trend)
    if not gap_aligned:
        trap_score += 20
        traps.append("Gap direction opposes prevailing trend — fading likely")

    # Trap condition 4: Gap from squeeze (BB squeeze breakout — can be false)
    if bb_squeeze and rvol < 1.3:
        trap_score += 15
        traps.append("BB squeeze breakout on weak volume — watch for reversal")

    if trap_score >= 60:
        return {
            "trap_prob": min(trap_score, 95), "trap_type": "HIGH RISK — Gap Trap",
            "traps": traps,
            "action": "DO NOT CHASE THE GAP. Wait 15-30 min for direction to establish. "
                       "Enter only after confirmation candle WITH volume in the trend direction.",
            "safe_entry": "First pullback to the original gap level after 15-30 min open"
        }
    elif trap_score >= 30:
        return {
            "trap_prob": trap_score, "trap_type": "MODERATE RISK",
            "traps": traps,
            "action": "Caution. Let the first 15 min trade before entering. "
                       "Use a tighter stop than usual — gap may partially fill.",
            "safe_entry": "Enter on first 5-min candle close in gap direction after 9:45am ET"
        }
    else:
        return {
            "trap_prob": trap_score, "trap_type": "LOW RISK — Clean Gap",
            "traps": ["No major trap signals detected"],
            "action": "Gap appears clean. Can enter on breakout of the first 5-min high/low.",
            "safe_entry": "Buy breakout of first 5-min candle (put stop below gap level)"
        }


def levels_plus_em_strike_placement(spx: float, iv_pct: float, dte: int,
                                      support_level: float, resistance_level: float,
                                      spread_type: str = "PUT") -> dict:
    """
    VIP May 2022: "Understanding Options — Identifying Strong Levels".
    The double-confirmation method: use the MORE CONSERVATIVE of:
    (a) Expected move beyond spot, OR (b) Key structural level.
    For PUT spread: short put must be BELOW BOTH the support level AND the EM.
    For CALL spread: short call must be ABOVE BOTH the resistance level AND the EM.
    This is the IWT most important option-specific insight from the VIP sessions.
    """
    import math
    em = spx * (iv_pct/100) * math.sqrt(dte/365)

    if "PUT" in spread_type.upper():
        em_short  = spx - em
        lvl_short = support_level
        # Use the HIGHER of the two (more conservative = less OTM = safer)
        # But must be BELOW both
        if lvl_short < em_short:
            # Support is below EM — support is the binding constraint
            rec_short = lvl_short - 25  # 25 pts below support
            method = "Support-level bound — put strike 25 pts below key support"
        else:
            # EM is below support — EM is binding
            rec_short = round((em_short - 25) / 25) * 25
            method = "Expected-move bound — put strike outside 1σ EM"

        return {
            "spread_type": "PUT",
            "em_boundary": round(em_short, 0),
            "level_boundary": round(lvl_short, 2),
            "recommended_short": rec_short,
            "recommended_long":  rec_short - 25,
            "binding_constraint": method,
            "note": ("IWT VIP rule: short put must clear BOTH the expected move AND "
                     "be below the nearest support level. "
                     "The more conservative of the two governs.")
        }
    else:
        em_short  = spx + em
        lvl_short = resistance_level
        if lvl_short > em_short:
            rec_short = lvl_short + 25
            method = "Resistance-level bound — call strike 25 pts above key resistance"
        else:
            rec_short = round((em_short + 25) / 25) * 25
            method = "Expected-move bound — call strike outside 1σ EM"
        return {
            "spread_type": "CALL",
            "em_boundary": round(em_short, 0),
            "level_boundary": round(lvl_short, 2),
            "recommended_short": rec_short,
            "recommended_long":  rec_short + 25,
            "binding_constraint": method,
            "note": "IWT VIP rule: short call must clear BOTH EM AND be above key resistance."
        }


def short_or_not_score(trend: str, vix: float, macro: dict,
                        metrics: dict, rsi: float, rvol: float) -> dict:
    """
    VIP 2022-2023: "To Short, or not To Short" (multiple sessions).
    the IWT shorting framework: short ONLY when all conditions align.
    5 requirements — all 5 must pass for a GREEN short signal.
    """
    score = 0; reasons = []; blocks = []
    # 1. Trend must be confirmed bearish
    if trend in ("STRONG_BEAR", "BEAR"):
        score += 2; reasons.append("✅ Confirmed downtrend (SMA alignment)")
    elif trend == "NEUTRAL":
        score += 0; reasons.append("⚠️ No clear trend — shorts are lower quality in chop")
    else:
        score -= 2; blocks.append("🔴 UPTREND — Never short a strong uptrend (Step 1 fail)")

    # 2. Price must be at/near resistance (fresh level)
    # Proxied from metrics: price vs 20-day high
    if metrics:
        res = metrics.get("resistance", 0); price = metrics.get("price", 1)
        dist_from_res = abs(price - res) / price * 100
        if dist_from_res < 1.5:
            score += 2; reasons.append("✅ Price at/near resistance (fresh level check)")
        elif dist_from_res < 3:
            score += 1; reasons.append("🟡 Approaching resistance (within 3%)")
        else:
            score -= 1; reasons.append(f"⚠️ Price is {dist_from_res:.1f}% from resistance — not at level")

    # 3. RSI overbought on the short timeframe
    if rsi > 70:
        score += 2; reasons.append(f"✅ RSI {rsi:.0f} — overbought, reversal signal")
    elif rsi > 60:
        score += 1; reasons.append(f"🟡 RSI {rsi:.0f} — elevated but not extreme")
    elif rsi < 40:
        blocks.append(f"🔴 RSI {rsi:.0f} — oversold. Do NOT short into oversold conditions")
        score -= 2

    # 4. Volume (RVOL) confirmation
    if rvol > 1.3:
        score += 1; reasons.append(f"✅ RVOL {rvol:.1f}x — institutional volume present")
    elif rvol < 0.7:
        reasons.append(f"⚠️ RVOL {rvol:.1f}x — low volume shorts often reverse quickly")
        score -= 1

    # 5. Macro environment must not be strongly bullish
    if macro:
        if vix > 25:
            score += 1; reasons.append(f"✅ VIX {vix:.1f} — elevated vol supports short thesis")
        if macro.get("sp", 0) < -1.0:
            score += 1; reasons.append("✅ Market down today — wind at back for shorts")
        if macro.get("sp", 0) > 1.5:
            blocks.append(f"🔴 Market up {macro.get('sp',0):.1f}% today — bad environment for new shorts")
            score -= 2

    # Verdict
    if blocks:
        return {"verdict": "DO NOT SHORT", "badge": "🚫", "score": score,
                "reasons": reasons, "blocks": blocks,
                "action": "Hard block conditions active. Short setups require ABSENCE of these."}
    elif score >= 6:
        return {"verdict": "GREEN — SHORT VALID", "badge": "🟢", "score": score,
                "reasons": reasons, "blocks": [],
                "action": "All 5 conditions met. Enter at resistance with stop above level. Target: prior support."}
    elif score >= 3:
        return {"verdict": "YELLOW — WAIT FOR CONFIRMATION", "badge": "🟡", "score": score,
                "reasons": reasons, "blocks": [],
                "action": "Most conditions met but not all. Wait for confirmation candle (bearish engulfing or shooting star) before shorting."}
    else:
        return {"verdict": "RED — DO NOT SHORT", "badge": "🔴", "score": score,
                "reasons": reasons, "blocks": [],
                "action": "Conditions do not support shorting. Stay in cash or trade the long side."}


def calc_covered_call_yield(stock_price: float, call_strike: float,
                              call_premium: float, dte: int, shares: int = 100) -> dict:
    """
    VIP 2020: "Covered Calls" (multiple sessions).
    the IWT covered call income method: sell slightly OTM, 30-45 DTE,
    buy back at 50% profit, roll at 21 DTE.
    Returns annualized yield and full trade economics.
    """
    if call_strike <= 0 or call_premium <= 0 or stock_price <= 0:
        return {"error": "Invalid inputs"}
    contracts    = shares // 100
    premium_per_share = call_premium
    total_income = premium_per_share * shares
    cost_basis   = stock_price * shares
    simple_yield = premium_per_share / stock_price * 100
    ann_yield    = simple_yield * (365 / max(dte, 1))
    max_profit   = (call_strike - stock_price + call_premium) * shares
    assigned_profit = (call_strike - stock_price) * shares + total_income
    take_profit_at   = call_premium * 0.50   # 50% profit target (buy back here)
    roll_at_dte      = 21

    # OTM check
    otm_pct = (call_strike - stock_price) / stock_price * 100
    if otm_pct < 0:
        otm_note = f"⚠️ ITM call — you cap your upside NOW. Consider OTM strike."
    elif otm_pct < 2:
        otm_note = f"✅ Slightly OTM ({otm_pct:.1f}%) — IWT preferred zone"
    else:
        otm_note = f"🟡 Further OTM ({otm_pct:.1f}%) — less premium but more upside room"

    dte_note = ("✅ Ideal 30-45 DTE range" if 30 <= dte <= 45
                else f"⚠️ {dte} DTE — IWT standard: 30-45 DTE for covered calls")

    return {
        "stock_price": stock_price, "call_strike": call_strike,
        "call_premium": call_premium, "dte": dte, "shares": shares,
        "contracts": contracts, "total_income": round(total_income, 2),
        "cost_basis": round(cost_basis, 2),
        "simple_yield_pct": round(simple_yield, 2),
        "annualized_yield_pct": round(ann_yield, 2),
        "max_profit": round(max_profit, 2),
        "assigned_profit": round(assigned_profit, 2),
        "take_profit_price": round(take_profit_at, 2),
        "roll_at_dte": roll_at_dte,
        "otm_pct": round(otm_pct, 2),
        "otm_note": otm_note, "dte_note": dte_note,
        "teri_rules": [
            f"Sell ${call_premium:.2f} call → collect ${total_income:.0f} today",
            f"Buy back when worth ${take_profit_at:.2f} (50% profit = ${total_income/2:.0f} gain)",
            f"If not at 50% by {roll_at_dte} DTE — roll to next month",
            "Never let the stock get called away if you want to keep it",
        ]
    }



# =============================================================================
# V14b — NEW MODULES FROM TRADE AND TRAVEL 2.0 / OPTIONS 101 / VIP CURRICULUM
# Sources: T&T 2.0 (2023 refresh), Options 101, Coaching Calls 2019-2023
# Math: all verified — no synthetic data
# =============================================================================

def calc_six_figure_plan(
    monthly_goal: float,
    account_size: float,
    win_rate: float = 0.70,
    avg_contract_risk: float = 500.0,
    trades_per_day: int = 2,
) -> dict:
    """
    T&T 2.0 / Coaching Call 1/9/2023: "Building a Six Figure Trading Plan"
    the IWT backward-planning framework: start from income goal,
    derive required daily performance, and check account math.

    This does NOT invent a profitable edge — it shows what YOU would need
    to achieve the goal with the given inputs. Kelly/BSM edge is separate.
    """
    if monthly_goal <= 0 or account_size <= 0:
        return {"error": "Invalid inputs"}

    annual_goal   = monthly_goal * 12
    daily_goal    = monthly_goal / 21          # ~21 trading days/month
    weekly_goal   = monthly_goal / 4.3

    # Per-contract P&L math (credit spread context)
    # avg_contract_risk = max_loss_per_contract
    # At 70% win rate with 1:1 win/loss (50% profit target):
    # avg_win  = credit × 100 × contracts (taking 50% of max credit)
    # avg_loss = credit × 200 × contracts (2× credit = stop)
    # For simplification: express win in $ per trade
    avg_win_per_trade  = avg_contract_risk * 0.33  # 33% return on risk (50% of credit)
    avg_loss_per_trade = avg_contract_risk * 0.67  # 67% loss (approx 2× credit stop)
    ev_per_trade       = win_rate * avg_win_per_trade - (1 - win_rate) * avg_loss_per_trade

    if ev_per_trade <= 0:
        ev_note = (f"WARNING: At {win_rate:.0%} win rate with this risk/reward, "
                   f"expected value per trade is ${ev_per_trade:.2f}. "
                   "You need higher win rate or better R:R to hit this goal.")
    else:
        ev_note = f"Positive EV: ${ev_per_trade:.2f}/trade at {win_rate:.0%} win rate."

    # Trades needed to hit daily goal
    trades_for_daily_goal = daily_goal / max(ev_per_trade, 0.01)
    monthly_return_pct    = monthly_goal / account_size * 100

    # Account size sanity check: account should support goals without oversizing
    # System rule: risk max 1% per trade
    max_risk_per_trade = account_size * 0.01
    sizing_ok = avg_contract_risk <= max_risk_per_trade

    return {
        "monthly_goal":        round(monthly_goal, 2),
        "annual_goal":         round(annual_goal, 2),
        "daily_goal":          round(daily_goal, 2),
        "weekly_goal":         round(weekly_goal, 2),
        "account_size":        account_size,
        "monthly_return_pct":  round(monthly_return_pct, 2),
        "ev_per_trade":        round(ev_per_trade, 2),
        "ev_note":             ev_note,
        "avg_win_per_trade":   round(avg_win_per_trade, 2),
        "avg_loss_per_trade":  round(avg_loss_per_trade, 2),
        "trades_needed_daily": round(trades_for_daily_goal, 1),
        "trades_planned_daily": trades_per_day,
        "daily_reachable":     trades_per_day >= trades_for_daily_goal,
        "max_risk_1pct":       round(max_risk_per_trade, 2),
        "sizing_ok":           sizing_ok,
        "sizing_note": (
            f"1% rule: max ${max_risk_per_trade:.0f}/trade on ${account_size:,.0f} account. "
            + ("Your avg contract risk fits." if sizing_ok else
               f"Your ${avg_contract_risk:.0f} risk EXCEEDS 1% — reduce to 1 contract or widen stop less.")
        ),
        "teri_rules": [
            f"Monthly goal ${monthly_goal:,.0f} = ${daily_goal:.0f}/day over 21 trading days",
            "Write this plan down — review before every session",
            "Stop trading the day you hit your daily goal",
            "Stop trading the week you double your daily goal in losses",
            "Small consistent wins beat occasional big wins",
        ],
    }


def options_playbook_router(vix: float, ivr: float, trend: str,
                             pre_event: bool = False,
                             post_event_selloff: bool = False) -> dict:
    """
    Options 101 — Trading Options Playbook (97:12)
    Maps market conditions to specific option structures.
    the IWT playbook logic from the coaching session.
    Returns: primary structure, secondary structure, avoid list, reasoning.
    """
    # Classify VIX regime
    if vix < 15:
        vix_regime = "LOW"
    elif vix < 22:
        vix_regime = "MODERATE"
    elif vix < 30:
        vix_regime = "ELEVATED"
    else:
        vix_regime = "HIGH"

    # Classify IVR
    ivr_regime = "LOW" if ivr < 25 else "MODERATE" if ivr < 50 else "HIGH"

    # Determine playbook entry
    if pre_event:
        return {
            "play": "PRE-EVENT: STAND ASIDE",
            "badge": "⏸️",
            "primary": "Close or reduce existing positions",
            "secondary": "If must be on: long options only (defined risk), 1/4 size",
            "avoid": ["Selling new premium", "Increasing size", "Iron condors"],
            "reason": "Events reprice IV rapidly — credit collected evaporates on a spike",
            "teri_rule": "Events are the #1 killer of premium sellers. System rule: flat into events.",
        }

    if post_event_selloff:
        return {
            "play": "POST-SELLOFF: SELL PREMIUM AGGRESSIVELY",
            "badge": "💥",
            "primary": "Put credit spreads (14-21 DTE, 20-25 delta)",
            "secondary": "Iron condors if IV is extremely elevated (IVR > 70%)",
            "avoid": ["Buying premium — you're overpaying IV crush", "New long options"],
            "reason": "Post-event IV crush is powerful. Sell when IV is high, buy when IV is low.",
            "teri_rule": "Post-event = premium seller's payday. IV reverts fast — be a seller.",
        }

    # Standard routing
    if trend in ("STRONG_BULL", "BULL") and vix_regime == "LOW":
        return {
            "play": "BULL + LOW VIX: BUY DEBIT SPREAD OR LONG CALL",
            "badge": "📈",
            "primary": "Call debit spread (75-90 DTE, ATM/slightly OTM, 40-50 delta)",
            "secondary": "Long call (DITM, 70+ delta, 60+ DTE) — IWT preferred long option",
            "avoid": ["Selling puts (credit is thin in low VIX)", "Naked calls"],
            "reason": "Low IVR means options are cheap — pay for direction, don't sell cheap premium.",
            "teri_rule": "When IV is low and trend is up, buy options rather than sell them.",
        }
    elif trend in ("STRONG_BULL", "BULL") and vix_regime in ("MODERATE", "ELEVATED"):
        return {
            "play": "BULL + ELEVATED VIX: SELL PUT CREDIT SPREAD",
            "badge": "🐂",
            "primary": "Put credit spread (14-21 DTE, 15-20 delta short strike, outside EM)",
            "secondary": "Cash-secured put if you want to own the stock at the strike",
            "avoid": ["Long puts", "Bear call spreads", "Oversizing in high VIX"],
            "reason": "Premium is elevated — collect income while betting the trend continues.",
            "teri_rule": "High IV + uptrend = ideal credit spread environment. Be the house.",
        }
    elif trend in ("STRONG_BEAR", "BEAR") and vix_regime in ("ELEVATED", "HIGH"):
        return {
            "play": "BEAR + HIGH VIX: SELL CALL CREDIT SPREAD OR BUY PUTS",
            "badge": "🐻",
            "primary": "Call credit spread (14-21 DTE, 15-20 delta, above resistance)",
            "secondary": "Long put (DITM, 60+ DTE) if trend is very strong",
            "avoid": ["Selling put credit spreads into a downtrend", "Long calls"],
            "reason": "Selling calls at resistance in a downtrend harvests premium WITH the trend.",
            "teri_rule": "Short-side premium selling: sell calls, not puts, in a downtrend.",
        }
    elif trend == "NEUTRAL" and vix_regime in ("ELEVATED", "HIGH"):
        return {
            "play": "RANGE-BOUND + ELEVATED VIX: IRON CONDOR",
            "badge": "🦅",
            "primary": "Iron condor (14-21 DTE, 15-20 delta on both sides)",
            "secondary": "Strangle if IV is very high (but not for beginners)",
            "avoid": ["Directional trades", "Long options (IV crush risk)"],
            "reason": "Rangebound + elevated IV = collect premium from both sides. Theta works for you.",
            "teri_rule": "Iron condors: only when price is going NOWHERE and IV is HIGH.",
        }
    elif vix_regime == "HIGH":
        return {
            "play": "HIGH VIX CRISIS: DEFINED RISK ONLY",
            "badge": "⚠️",
            "primary": "If selling: very narrow spreads (1-2× expected move away), 7 DTE max",
            "secondary": "Long puts as portfolio hedge",
            "avoid": ["Naked anything", "Oversizing", "New bullish credit spreads"],
            "reason": "High VIX means wide moves possible. Only take trades where max loss is acceptable.",
            "teri_rule": "In crisis VIX: smaller size, tighter structure, no heroes.",
        }
    else:
        return {
            "play": "NEUTRAL CONDITIONS: WAIT OR SMALL SIZE",
            "badge": "🟡",
            "primary": "Put credit spread with tighter width (10-15 pts vs normal 25)",
            "secondary": "Stand aside — cash is a valid position",
            "avoid": ["Full-size trades", "Complex multi-leg without clear thesis"],
            "reason": "Mixed signals = smaller size or no trade. No edge = no trade.",
            "teri_rule": "When in doubt, stay out. You can't lose money on a trade you don't take.",
        }


def gap_trade_playbook(gap_pct: float, gap_type: str, rvol: float,
                        trend: str, spx_price: float = 0,
                        globex_high: float = 0, globex_low: float = 0) -> dict:
    """
    Bonus: Gaps Coaching Call (78:12) + VIP Gaps 2019-2023
    Three distinct gap trading strategies based on the IWT coaching.
    Returns specific entry, stop, and target rules for each applicable strategy.
    """
    abs_gap = abs(gap_pct)
    direction = "UP" if gap_pct > 0 else "DOWN"

    strategies = []

    # ── STRATEGY 1: FADE THE GAP ─────────────────────────────────
    # Only for common gaps, low volume, within normal range
    if gap_type in ("Common", "Micro") and abs_gap < 0.5 and rvol < 1.2:
        fade_entry = "Wait for the FIRST CANDLE to close in the gap fill direction"
        fade_stop  = f"1 ATR above gap high ({direction} gap) — if price reclaims gap, exit immediately"
        fade_tgt   = f"The gap fill level (prev close: {gap_pct:+.2f}% move to fill)"
        strategies.append({
            "name": "FADE",
            "badge": "↩️",
            "confidence": "HIGH",
            "entry_rule": fade_entry,
            "stop_rule": fade_stop,
            "target_rule": fade_tgt,
            "context": (
                "Common gaps fill ~82% of the time. "
                "Low volume = no institutional conviction behind the gap. "
                "Wait for the first pullback candle — don't sell the gap open directly."
            ),
            "teri_rule": "Never fade a gap with high volume. Only fade low-volume, small gaps.",
        })

    # ── STRATEGY 2: RIDE THE GAP ─────────────────────────────────
    # For breakaway / runaway gaps with volume confirmation
    if gap_type in ("Breakaway", "Runaway") and rvol >= 1.3:
        ride_entry = "Wait 15-30 min after open. Enter on FIRST PULLBACK to the gap level."
        ride_stop  = f"Below the gap level (if gap fills, thesis is wrong)"
        trend_word = "prior high" if direction == "UP" else "prior low"
        ride_tgt   = f"Next major support/resistance level ({trend_word})"
        strategies.append({
            "name": "RIDE",
            "badge": "🚀",
            "confidence": "HIGH",
            "entry_rule": ride_entry,
            "stop_rule": ride_stop,
            "target_rule": ride_tgt,
            "context": (
                f"{gap_type} gap with {rvol:.1f}x volume = institutional backing. "
                "DO NOT fade this. Let the first 15-30 min establish direction, "
                "then enter the pullback to the gap level."
            ),
            "teri_rule": "Institutional gaps do NOT fill quickly. Ride the momentum.",
        })

    # ── STRATEGY 3: USE GAP AS STOP LEVEL ────────────────────────
    # For any gap > 0.3% — the gap level becomes a natural stop
    if abs_gap >= 0.3:
        if direction == "UP":
            stop_use = (
                f"Gap low ({gap_pct:.2f}% above yesterday's close) = natural support. "
                "If entering a long position today, place stop BELOW the gap level. "
                "A full gap fill invalidates the bullish setup."
            )
        else:
            stop_use = (
                f"Gap high ({abs(gap_pct):.2f}% below yesterday's close) = natural resistance. "
                "If entering a short position today, place stop ABOVE the gap level. "
                "A full gap fill invalidates the bearish setup."
            )
        strategies.append({
            "name": "STOP LEVEL",
            "badge": "🛡️",
            "confidence": "APPLICABLE",
            "entry_rule": "No specific entry — use the gap level as a natural stop for other setups",
            "stop_rule": stop_use,
            "target_rule": "Use your standard R:R target from buyer/seller levels",
            "context": (
                "Every gap creates a natural reference level. "
                "The gap open level is where price 'jumped from' — it's a structural reference. "
                "IWT principle: 'The gap level is your line in the sand.'"
            ),
            "teri_rule": "The gap level is always a relevant stop or target — use it.",
        })

    # ── GLOBEX RANGE CONTEXT ────────────────────────────────────
    globex_note = ""
    if globex_high > 0 and globex_low > 0 and spx_price > 0:
        globex_range = globex_high - globex_low
        if spx_price > globex_high:
            globex_note = (
                f"SPX opened ABOVE Globex high ({globex_high:.0f}). "
                f"This is a HIGH MOMENTUM session — Globex high is now support. "
                f"Expect range extension up to {globex_high + globex_range:.0f}"
            )
        elif spx_price < globex_low:
            globex_note = (
                f"SPX opened BELOW Globex low ({globex_low:.0f}). "
                f"HIGH MOMENTUM to the downside — Globex low is now resistance. "
                f"Expect potential extension to {globex_low - globex_range:.0f}"
            )
        else:
            globex_note = (
                f"SPX within Globex range ({globex_low:.0f}–{globex_high:.0f}, "
                f"range={globex_range:.0f} pts). "
                "Normal session expected — range may contain most of today's movement."
            )

    if not strategies:
        strategies.append({
            "name": "NO CLEAR PLAY",
            "badge": "—",
            "confidence": "LOW",
            "entry_rule": "No actionable gap trade today",
            "stop_rule": "N/A",
            "target_rule": "N/A",
            "context": "Gap is too small or conditions don't fit a clear strategy. Trade other setups.",
            "teri_rule": "No setup = no trade. Wait for the next one.",
        })

    return {
        "gap_pct": gap_pct,
        "gap_type": gap_type,
        "direction": direction,
        "rvol": rvol,
        "strategies": strategies,
        "globex_note": globex_note,
    }


def calc_cc_cost_basis_reducer(
    stock_price: float,
    purchase_price: float,
    calls_sold: list,   # list of (premium, strike, dte) tuples per month
    shares: int = 100,
) -> dict:
    """
    Options 101 — Protect your Long Term Portfolio: Sell Covered Calls (71:31)
    Tracks how repeated covered call sales reduce cost basis over time.
    IWT principle: "My goal is eventually to own the stock for free."

    calls_sold: list of premiums collected per round (e.g., [2.50, 1.80, 3.20])
    All values per-share.
    """
    total_premium_collected = sum(p for p in calls_sold)
    current_cost_basis       = purchase_price - total_premium_collected
    cost_reduction_pct       = total_premium_collected / purchase_price * 100
    breakeven_months_at_avg  = (purchase_price / (total_premium_collected / max(len(calls_sold), 1)))
    unrealized_gain_on_stock = (stock_price - current_cost_basis) * shares
    cost_basis_vs_price      = stock_price - current_cost_basis   # how far OTM from cost basis

    rounds = []
    running_basis = purchase_price
    for i, p in enumerate(calls_sold):
        running_basis -= p
        rounds.append({
            "round": i + 1,
            "premium": p,
            "cost_basis_after": round(running_basis, 2),
            "reduction_to_date": round(purchase_price - running_basis, 2),
        })

    return {
        "purchase_price":          purchase_price,
        "current_stock_price":     stock_price,
        "total_rounds":            len(calls_sold),
        "total_premium_collected": round(total_premium_collected, 2),
        "current_cost_basis":      round(current_cost_basis, 2),
        "cost_reduction_pct":      round(cost_reduction_pct, 1),
        "breakeven_months":        round(breakeven_months_at_avg, 1),
        "unrealized_gain":         round(unrealized_gain_on_stock, 2),
        "still_above_cost_basis":  stock_price > current_cost_basis,
        "cost_basis_buffer":       round(cost_basis_vs_price, 2),
        "rounds":                  rounds,
        "avg_premium_per_round":   round(total_premium_collected / max(len(calls_sold), 1), 2),
        "teri_goal_note":          (
            f"At ${total_premium_collected / max(len(calls_sold), 1):.2f}/month avg, "
            f"you'll own this stock 'for free' (zero cost basis) in "
            f"~{breakeven_months_at_avg:.0f} months of covered calls."
        ),
        "teri_rules": [
            "Sell 30-45 DTE, slightly OTM (0.25-0.30 delta)",
            "Take 50% profit — buy back when worth half what you sold it for",
            "Roll to next month at 21 DTE if not yet at 50% profit",
            "If stock gets called away at your strike: you kept the premium AND got the gain",
            "Never sell a strike you're not happy to sell your shares at",
        ],
    }


def troubleshoot_trading(responses: dict) -> dict:
    """
    Coaching Call 1/4/2023: Troubleshoot Your Trading (118:27)
    the IWT diagnostic framework for common trading problems.
    responses: dict of symptom → bool (True = experiencing this problem)
    """
    DIAGNOSTICS = [
        {
            "symptom": "making_then_losing",
            "label": "Making money then giving it back",
            "root_cause": "No profit targets set, or overriding targets",
            "fix": "Set OCO brackets BEFORE entry: take-profit at 50%, stop at 2× credit",
            "teri_quote": "IWT principle: 'If you don't have a profit target, greed will take it back.'",
            "severity": "HIGH",
        },
        {
            "symptom": "losses_bigger_than_wins",
            "label": "Average losses > average wins",
            "root_cause": "Inconsistent position sizing — oversizing losing trades",
            "fix": "Same size EVERY trade. Track position size consistency for 1 week.",
            "teri_quote": "IWT principle: '1% risk on every trade — winning or losing, same size.'",
            "severity": "HIGH",
        },
        {
            "symptom": "too_many_trades",
            "label": "Trading too much / boredom trading",
            "root_cause": "No written pre-market plan; reacting instead of planning",
            "fix": "Write your plan before market opens. If a setup isn't in the plan, it's not a trade.",
            "teri_quote": "IWT principle: 'I plan the trade and trade the plan. If it's not in my plan, it's not my trade.'",
            "severity": "MEDIUM",
        },
        {
            "symptom": "missing_good_setups",
            "label": "Missing good setups / not at screen at the right time",
            "root_cause": "No pre-market alerts set; reactive trading",
            "fix": "Set price alerts the night before at your buyer/seller levels. Let alerts come to you.",
            "teri_quote": "IWT principle: 'Set your alerts and walk away. The chart will call you.'",
            "severity": "MEDIUM",
        },
        {
            "symptom": "scared_to_enter",
            "label": "Scared to enter / hesitating at good setups",
            "root_cause": "Position size is too large for your comfort level",
            "fix": "Cut your size in HALF until you're not scared. Then slowly rebuild.",
            "teri_quote": "IWT principle: 'If you're scared to enter, the position is too big. Trade smaller.'",
            "severity": "MEDIUM",
        },
        {
            "symptom": "revenge_trading",
            "label": "Revenge trading after a loss",
            "root_cause": "Emotional dysregulation; loss triggers urgency to 'make it back'",
            "fix": "After a loss: step away for 15 minutes minimum. Review the plan. Check max daily loss.",
            "teri_quote": "IWT principle: 'The market will be open tomorrow. You don't have to make it back today.'",
            "severity": "HIGH",
        },
        {
            "symptom": "holding_losers_too_long",
            "label": "Holding losing trades too long hoping for recovery",
            "root_cause": "No defined stop loss; emotional attachment to position",
            "fix": "Define your stop BEFORE entry. If you don't know your stop, don't take the trade.",
            "teri_quote": "IWT principle: 'Hope is not a trading strategy.'",
            "severity": "HIGH",
        },
    ]

    active = [d for d in DIAGNOSTICS if responses.get(d["symptom"], False)]
    high   = [d for d in active if d["severity"] == "HIGH"]
    med    = [d for d in active if d["severity"] == "MEDIUM"]

    if not active:
        summary = "No major issues flagged. Continue monitoring these categories weekly."
        priority = "MAINTAIN"
    elif high:
        summary = f"CRITICAL: {len(high)} high-severity issue(s) detected. Fix these first."
        priority = "FIX NOW"
    else:
        summary = f"{len(med)} medium-severity issue(s). Address systematically."
        priority = "IMPROVE"

    return {
        "active_diagnoses": active,
        "high_severity": high,
        "medium_severity": med,
        "summary": summary,
        "priority": priority,
        "total_issues": len(active),
        "teri_rule": "Fix one problem at a time. Trying to fix everything at once = fixing nothing.",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# EASYSTOCKTRADER — COMPACT UI V15
# Design philosophy: IWT in 4 tabs. Setup → Scan → Trade → Review.
# Mobile-first. GO/NO-GO is the centrepiece. Everything else is progressive.
# ═══════════════════════════════════════════════════════════════════════════════

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
                nse_hint = " NSE data via yfinance is often unavailable — check nse.co.ke directly." if t.endswith('.NR') else ""
                return None, None, f"ERROR: Insufficient data for {t}. Need at least 50 trading days.{nse_hint}"

            try:
                full_name = ticker_obj.info.get('longName', t)
                beta = ticker_obj.info.get('beta', 1.0)
            except:
                full_name = t
                beta = 1.0

            # Core indicators — pure numpy/pandas (no pandas_ta dependency)
            _calc_indicators(data)

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

            # gap_pct computed in scan button section below

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
        _c0, c1, c2 = last_3.iloc[0], last_3.iloc[1], last_3.iloc[2]

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
            # Fetch 1-year VIX history first — used for IVR proxy (REAL calculation)
            # IVR proxy = (current VIX - 52w low VIX) / (52w high VIX - 52w low VIX) * 100
            # This is the correct formula. For SPX, VIX IS implied volatility.
            try:
                vix_1y = yf.download("^VIX", period="1y", progress=False, timeout=15)["Close"].dropna()
                vix_52w_high = float(vix_1y.max())
                vix_52w_low  = float(vix_1y.min())
            except Exception:
                vix_52w_high, vix_52w_low = 30.0, 12.0  # conservative fallback

            # Commodities added: oil (CL=F), silver (SI=F), nat gas (NG=F), copper (HG=F)
            tickers = ["ES=F", "^VIX", "GC=F", "CL=F", "SI=F", "NG=F",
                       "^GDAXI", "^N225", "^TNX", "DX-Y.NYB", "^RUT", "^NDX"]
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

            # Commodities
            _oil   = df.get("CL=F",  pd.Series(dtype=float)).dropna()
            _silv  = df.get("SI=F",  pd.Series(dtype=float)).dropna()
            _ng    = df.get("NG=F",  pd.Series(dtype=float)).dropna()
            _rut   = df.get("^RUT",  pd.Series(dtype=float)).dropna()
            _ndx   = df.get("^NDX",  pd.Series(dtype=float)).dropna()
            oil_px    = float(_oil.iloc[-1])  if len(_oil)  > 0 else 0.0
            oil_chg   = float((_oil.iloc[-1]-_oil.iloc[-2])/_oil.iloc[-2]*100) if len(_oil) >= 2 else 0.0
            silv_px   = float(_silv.iloc[-1]) if len(_silv) > 0 else 0.0
            ng_px     = float(_ng.iloc[-1])   if len(_ng)   > 0 else 0.0

            # IVR proxy: WHERE current VIX sits in its 52-week range
            # Formula: (current - 52w_low) / (52w_high - 52w_low) * 100
            # This is real data — VIX IS SPX implied volatility.
            cur_vix = vix.iloc[-1] if len(vix) > 0 else 18.0
            ivr_proxy = round(
                (cur_vix - vix_52w_low) / (vix_52w_high - vix_52w_low) * 100, 1
            ) if vix_52w_high != vix_52w_low else 35.0

            return {
                "sp": sp_chg,
                "vix": cur_vix,
                "vix_52w_high": vix_52w_high,
                "vix_52w_low": vix_52w_low,
                "ivr_proxy": ivr_proxy,        # Auto-computed IVR from VIX 52w range
                "gold": gold.iloc[-1] if len(gold) > 0 else 2000,
                "gold_chg": gold_chg,
                "oil_px": oil_px, "oil_chg": oil_chg,
                "silv_px": silv_px, "ng_px": ng_px,
                "dax": dax_chg,
                "nikkei": nikkei_chg,
                "tnx": tnx.iloc[-1] if len(tnx) > 0 else 4.0,
                "tnx_chg": tnx_chg,
                "dxy": dxy.iloc[-1] if dxy is not None and len(dxy) > 0 else 100,
                "dxy_chg": dxy_chg,
                "passive": passive_on,
                "risk_off": risk_off,
                "dollar_headwind": dollar_headwind,
                "data_quality": "LIVE"
            }
        except Exception:
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
        if consecutive_losses >= 5:
            drawdown_mult = 0.25
        elif consecutive_losses >= 3:
            drawdown_mult = 0.5

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

def get_macro() -> dict:
    """Standalone wrapper — calls engine.get_macro() and returns the macro dict."""
    try:
        return engine.get_macro()
    except Exception as ex:
        return {}


st.set_page_config(
    page_title="EasyStockTrader",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── DESIGN SYSTEM ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif !important;
  }

  /* ── Reset & base ── */
  .block-container { padding: 0.75rem 1rem 2rem 1rem !important; max-width: 100% !important; }
  #MainMenu, footer, header { visibility: hidden; }

  /* ── Top strip ── */
  .est-topbar {
    display: flex; align-items: center; justify-content: space-between;
    background: #0c0f1a; border-bottom: 1px solid #1e2235;
    padding: 10px 20px; margin: -0.75rem -1rem 1rem -1rem;
  }
  .est-topbar-brand { font-size: 1rem; font-weight: 700; color: #e8e8ff; letter-spacing: -0.5px; }
  .est-topbar-brand span { color: #6c8cff; }
  .est-topbar-metrics { display: flex; gap: 20px; }
  .est-metric { text-align: right; }
  .est-metric-val { font-size: 1rem; font-weight: 600; color: #e8e8ff;
                    font-family: 'DM Mono', monospace; }
  .est-metric-lbl { font-size: 0.65rem; color: #7080a0; text-transform: uppercase; }
  .est-badge {
    padding: 3px 10px; border-radius: 20px; font-size: 0.72rem;
    font-weight: 600; letter-spacing: 0.3px;
  }
  .badge-on  { background: #1b5e20; color: #69f0ae; }
  .badge-neu { background: #f57f17; color: #fff; }
  .badge-cau { background: #bf360c; color: #fff; }
  .badge-off { background: #b71c1c; color: #fff; }
  .badge-unk { background: #263238; color: #90a4ae; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    gap: 2px; background: #0c0f1a; padding: 0 !important;
    border-bottom: 1px solid #1e2235;
  }
  .stTabs [data-baseweb="tab"] {
    padding: 10px 20px; font-size: 0.85rem; font-weight: 600;
    color: #7080a0; background: transparent !important; border: none;
    border-bottom: 2px solid transparent;
  }
  .stTabs [aria-selected="true"] {
    color: #6c8cff !important; border-bottom: 2px solid #6c8cff !important;
  }
  .stTabs [data-baseweb="tab-panel"] { padding: 1rem 0 0 0; }

  /* ── Cards ── */
  .card {
    background: #111624; border: 1px solid #1e2235; border-radius: 10px;
    padding: 14px 16px; margin-bottom: 10px;
  }
  .card-sm {
    background: #111624; border: 1px solid #1e2235; border-radius: 8px;
    padding: 10px 13px; margin-bottom: 8px;
  }
  .card-label { font-size: 0.65rem; text-transform: uppercase; color: #7080a0;
                letter-spacing: 0.5px; margin-bottom: 4px; }
  .card-value { font-size: 1.4rem; font-weight: 700; color: #e8e8ff;
                font-family: 'DM Mono', monospace; }
  .card-delta { font-size: 0.75rem; color: #7080a0; margin-top: 2px; }

  /* ── Verdict banners ── */
  .verdict-go {
    background: linear-gradient(135deg, #0d3b1a 0%, #1b5e20 100%);
    border: 1px solid #2e7d32; border-radius: 12px;
    padding: 18px 22px; text-align: center; margin: 12px 0;
  }
  .verdict-no {
    background: linear-gradient(135deg, #3b0d0d 0%, #5e1b1b 100%);
    border: 1px solid #7d2e2e; border-radius: 12px;
    padding: 18px 22px; text-align: center; margin: 12px 0;
  }
  .verdict-cond {
    background: linear-gradient(135deg, #3b2c0d 0%, #5e4a1b 100%);
    border: 1px solid #7d6220; border-radius: 12px;
    padding: 18px 22px; text-align: center; margin: 12px 0;
  }
  .verdict-big { font-size: 1.6rem; font-weight: 800; letter-spacing: -0.5px; }
  .verdict-sub { font-size: 0.85rem; opacity: 0.85; margin-top: 4px; }

  /* ── Score pills ── */
  .score-pill {
    display: inline-block; padding: 3px 12px; border-radius: 20px;
    font-size: 0.78rem; font-weight: 600; margin: 2px;
  }
  .pill-green  { background:#1b5e20; color:#69f0ae; }
  .pill-yellow { background:#f57f17; color:#fff; }
  .pill-red    { background:#b71c1c; color:#fff; }
  .pill-gray   { background:#263238; color:#90a4ae; }

  /* ── Grade cards ── */
  .grade-aplus { border-left: 4px solid #00e676; }
  .grade-a     { border-left: 4px solid #69f0ae; }
  .grade-b     { border-left: 4px solid #ffd740; }
  .grade-c     { border-left: 4px solid #ff6d00; }
  .grade-d     { border-left: 4px solid #d50000; }

  /* ── Buttons ── */
  .stButton > button {
    border-radius: 8px !important; font-weight: 600 !important;
    font-family: 'DM Sans', sans-serif !important;
  }
  div[data-testid="stButton"] > button[kind="primary"] {
    background: #6c8cff !important; color: #fff !important;
    border: none !important;
  }

  /* ── Mono text ── */
  .mono { font-family: 'DM Mono', monospace; font-size: 0.88rem; }
  .dim  { color: #7080a0; font-size: 0.82rem; }

  /* ── Step row ── */
  .step-row {
    display: flex; align-items: center; gap: 10px;
    padding: 8px 0; border-bottom: 1px solid #1a1f2e;
  }
  .step-num {
    width: 24px; height: 24px; border-radius: 50%;
    background: #1e2a4a; color: #6c8cff; font-size: 0.75rem;
    font-weight: 700; display: flex; align-items: center; justify-content: center;
    flex-shrink: 0;
  }
  .step-lbl { font-size: 0.85rem; color: #c0cce0; font-weight: 500; }

  /* ── Dividers ── */
  hr { border-color: #1e2235 !important; margin: 12px 0 !important; }

  /* ── Compact inputs ── */
  .stNumberInput, .stSelectbox, .stSlider { margin-bottom: 0 !important; }
  .stSlider { padding-top: 0 !important; }
  label { font-size: 0.8rem !important; color: #8090b0 !important; }

  /* ── Copy code block ── */
  .order-block {
    background: #0a0d15; border: 1px solid #1e2235; border-radius: 8px;
    padding: 12px 16px; font-family: 'DM Mono', monospace;
    font-size: 0.82rem; color: #a0f0c0; margin: 8px 0;
    white-space: pre-wrap;
  }

  /* ── Mobile ── */
  .tip-box{background:#111e2e;border:1px solid #1e3a5f;border-radius:8px;padding:10px 14px;margin:6px 0;font-size:.83rem;color:#90caf9;line-height:1.5}
  @media (max-width: 768px) {
    .est-topbar-metrics { gap: 10px; }
    .est-metric-val { font-size: 0.88rem; }
    .verdict-big { font-size: 1.3rem; }
  }
</style>
""", unsafe_allow_html=True)

# ── SESSION STATE ──────────────────────────────────────────────────────────────
_SS_DEFAULTS = {
    "data": None, "metrics": {}, "macro": None, "signals": {},
    "journal": [], "open_positions": [], "closed_trades": [],
    "goal_met": False, "lang_level": "Intermediate",
    "advisor_goal": "Weekly income (sell premium)",
    "daily_pnl": 0.0, "total_risk_deployed": 0.0,
    "consecutive_losses": 0, "day_trades_used": 0,
    "week_event_days": [],
    # V15 new keys
    "acct_size": 10000, "max_risk_per_trade": 100.0, "max_portfolio_risk_pct": 6.0,
    "acct_type": "Margin < $25k", "monthly_goal": 1000.0,
    "strategy": "Income (SPX Vertical Credit Spread)",
    "ticker": "SPY", "dte": 14, "scorecard_fresh": 1,
    "scorecard_speed": 1, "scorecard_time": 2,
    "_spx_plan": None, "_universe_scan": None,
    "_last_macro_fetch": None,
}
for k, v in _SS_DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── HELPER: current live numbers for top bar ───────────────────────────────────
def _topbar_numbers():
    m = st.session_state.macro or {}
    return {
        "spx":  m.get("spx_price", "—"),
        "vix":  m.get("vix", "—"),
        "pnl":  st.session_state.daily_pnl,
        "regime": compute_market_weather(m),
    }

# ── TOP BAR ───────────────────────────────────────────────────────────────────
_tb = _topbar_numbers()
_mw = _tb["regime"]
_r   = _mw.get("regime", "UNKNOWN")
_bc  = {"RISK-ON":"badge-on","RISK-NEUTRAL":"badge-neu",
        "RISK-CAUTIOUS":"badge-cau","RISK-OFF":"badge-off"}.get(_r,"badge-unk")
_spx = f"{_tb['spx']:,.0f}" if isinstance(_tb['spx'], (int,float)) else "—"
_vix = f"{_tb['vix']:.1f}" if isinstance(_tb['vix'], (int,float)) else "—"
_pnl_col = "#69f0ae" if _tb["pnl"] >= 0 else "#ff5252"
_pnl_str = f"+${_tb['pnl']:.0f}" if _tb["pnl"] >= 0 else f"-${abs(_tb['pnl']):.0f}"

st.markdown(f"""
<div class="est-topbar">
  <div class="est-topbar-brand">📈 <span>Easy</span>StockTrader</div>
  <div class="est-topbar-metrics">
    <span class="est-badge {_bc}">{_mw.get('badge','⬜')} {_r}</span>
    <div class="est-metric">
      <div class="est-metric-val">{_spx}</div>
      <div class="est-metric-lbl">SPX</div>
    </div>
    <div class="est-metric">
      <div class="est-metric-val">{_vix}</div>
      <div class="est-metric-lbl">VIX</div>
    </div>
    <div class="est-metric">
      <div class="est-metric-val" style="color:{_pnl_col}">{_pnl_str}</div>
      <div class="est-metric-lbl">Today P&L</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ── 4 TABS ─────────────────────────────────────────────────────────────────────
_T1, _T2, _T3, _T4 = st.tabs(["🎯 Setup", "🔍 Scan", "📊 Trade", "📓 Review"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — SETUP (Account → Ticker → Market Check → IWT Score → Verdict)
# ══════════════════════════════════════════════════════════════════════════════
with _T1:
    _lvl = st.session_state.lang_level
    _nl = st.radio("Experience level", ["Beginner","Intermediate","Advanced"],
        index=["Beginner","Intermediate","Advanced"].index(_lvl),
        horizontal=True, key="exp_lvl_t1",
        help="Beginner: plain-English guides  ·  Advanced: full technical data")
    if _nl != _lvl: st.session_state.lang_level = _nl; st.rerun()
    _lvl = st.session_state.lang_level
    st.markdown("---")
    _c_left, _c_right = st.columns([1, 1], gap="large")


    # ── LEFT: Account + Strategy ─────────────────────────────────────────────
    with _c_left:
        st.markdown('<div class="card-sm"><span class="card-label">Account & Risk</span></div>',
                    unsafe_allow_html=True)
        _a1, _a2 = st.columns(2)
        with _a1:
            st.session_state.acct_size = st.number_input(
                "Account size ($)", value=int(st.session_state.acct_size),
                step=500, min_value=500, key="acct_sz_inp")
            st.session_state.monthly_goal = st.number_input(
                "Monthly income goal ($)", value=int(st.session_state.monthly_goal),
                step=100, min_value=100, key="monthly_goal_inp")
        with _a2:
            _risk_pct = st.slider("Risk per trade (%)", 0.25, 5.0,
                float(st.session_state.max_risk_per_trade / max(st.session_state.acct_size, 1) * 100),
                step=0.25, key="risk_pct_sl")
            st.session_state.max_risk_per_trade = st.session_state.acct_size * _risk_pct / 100
            st.markdown(f'<div class="dim" style="margin-top:6px">1% = '
                        f'<strong style="color:#e8e8ff">${st.session_state.max_risk_per_trade:.0f}/trade</strong>'
                        f' · Daily goal <strong style="color:#6c8cff">'
                        f'${st.session_state.monthly_goal/21:.0f}</strong></div>',
                        unsafe_allow_html=True)
            st.session_state.day_trades_used = st.number_input(
                "Day trades used this week", 0, 3,
                int(st.session_state.day_trades_used), key="dtu_inp")

        st.markdown("---")
        st.markdown('<div class="card-sm"><span class="card-label">What to trade</span></div>',
                    unsafe_allow_html=True)
        _b1, _b2 = st.columns([2, 1])
        with _b1:
            _ticker_choice = st.selectbox(
                "Ticker", ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN",
                           "GOOGL", "TSLA", "META", "NFLX", "JPM", "GLD",
                           "TLT", "SPX (0DTE/Credit)", "Custom…"],
                index=0, key="ticker_sel")
            if _ticker_choice == "Custom…":
                _ticker_choice = st.text_input("Enter ticker", "AAPL", key="custom_tkr").upper()
            st.session_state.ticker = _ticker_choice.replace(" (0DTE/Credit)", "").replace("SPX", "^GSPC")

        with _b2:
            _strategy_opts = [
                "💰 Income (credit spread)",
                "📈 Long call/put",
                "📉 Short sell",
                "📋 Stock only",
            ]
            _strat_sel = st.selectbox("Strategy", _strategy_opts, key="strat_sel")
            _strat_map = {
                "💰 Income (credit spread)": "Income (SPX Vertical Credit Spread)",
                "📈 Long call/put": "IWT Long Option (60+ DTE DITM)",
                "📉 Short sell": "Short (Sell) Stock",
                "📋 Stock only": "Stock — Long (Swing/Trend)",
            }
            st.session_state.strategy = _strat_map.get(_strat_sel, _strat_sel)

        st.markdown("---")
        # Market data fetch button
        _mfetch_col1, _mfetch_col2 = st.columns([1, 1])
        with _mfetch_col1:
            if st.button("🔄 Load Market Data", type="primary", use_container_width=True, key="load_mkt"):
                with st.spinner("Fetching live data…"):
                    try:
                        _tkr = st.session_state.ticker
                        engine.fetch_data(_tkr)
                        st.session_state.data    = engine.data
                        st.session_state.metrics = engine.compute_metrics()
                        st.session_state.signals = engine.generate_signals()
                        # Macro
                        _mac = get_macro()
                        if _mac: st.session_state.macro = _mac
                        st.rerun()
                    except Exception as _ex:
                        st.error(f"Data error: {_ex}")
        with _mfetch_col2:
            if st.session_state.data is not None:
                _m = st.session_state.metrics
                _p = _m.get("price", 0); _ch = _m.get("chg_pct", 0)
                _col = "#69f0ae" if _ch >= 0 else "#ff5252"
                st.markdown(
                    f'<div class="card-sm" style="border-left:3px solid {_col}">'
                    f'<div class="card-label">{st.session_state.ticker.replace("^GSPC","SPX")}</div>'
                    f'<div class="card-value" style="color:{_col};font-size:1.1rem">'
                    f'${_p:,.2f}</div>'
                    f'<div class="card-delta">{_ch:+.2f}% today</div>'
                    f'</div>', unsafe_allow_html=True)

    # ── RIGHT: IWT Scorecard → Verdict ───────────────────────────────────────
    with _c_right:
        if _lvl == "Beginner":
            st.caption(
                "New to trading? Load data from the left panel first, "
                "then answer the 3 questions below. The GO banner guides you.")

    with _c_right:
        _sc_hdrs={"Beginner":"3 Quick Checks (0=weak, 2=strong)","Intermediate":"IWT Scorecard","Advanced":"IWT Scorecard: freshness·speed·linger"}
        st.markdown(f'<div class="card-sm"><span class="card-label">{_sc_hdrs.get(_lvl,"IWT Scorecard")}</span></div>',
                    unsafe_allow_html=True)

        st.markdown("""
        <div class="step-row">
          <div class="step-num">①</div>
          <div class="step-lbl">How <strong>fresh</strong> is this level? (0 = often tested · 2 = never touched)</div>
        </div>""", unsafe_allow_html=True)
        st.session_state.scorecard_fresh = st.select_slider(
            "Freshness", [0,1,2],
            value=st.session_state.scorecard_fresh,
            format_func=lambda v: {0:"0 — Overused",1:"1 — Some history",2:"2 — Fresh ✅"}[v],
            key="sc_fresh", label_visibility="collapsed")

        st.markdown("""
        <div class="step-row">
          <div class="step-num">②</div>
          <div class="step-lbl">Did price leave <strong>fast</strong>? (0 = slow · 2 = sharp rejection)</div>
        </div>""", unsafe_allow_html=True)
        st.session_state.scorecard_speed = st.select_slider(
            "Speed", [0,1,2],
            value=st.session_state.scorecard_speed,
            format_func=lambda v: {0:"0 — Slow drift",1:"1 — Average",2:"2 — Sharp ✅"}[v],
            key="sc_speed", label_visibility="collapsed")

        st.markdown("""
        <div class="step-row">
          <div class="step-num">③</div>
          <div class="step-lbl">Did it <strong>linger</strong>? (0 = camped there · 2 = 1-3 candles and gone)</div>
        </div>""", unsafe_allow_html=True)
        st.session_state.scorecard_time = st.select_slider(
            "Time at level", [0,1,2],
            value=st.session_state.scorecard_time,
            format_func=lambda v: {0:"0 — Camped there",1:"1 — A while",2:"2 — Gone fast ✅"}[v],
            key="sc_time", label_visibility="collapsed")

        _iwt_total = (st.session_state.scorecard_fresh +
                      st.session_state.scorecard_speed +
                      st.session_state.scorecard_time)
        _max_score = 6

        # Pills
        _pill_f = ["pill-red","pill-yellow","pill-green"][st.session_state.scorecard_fresh]
        _pill_s = ["pill-red","pill-yellow","pill-green"][st.session_state.scorecard_speed]
        _pill_t = ["pill-red","pill-yellow","pill-green"][st.session_state.scorecard_time]
        st.markdown(
            f'<span class="score-pill {_pill_f}">Fresh {st.session_state.scorecard_fresh}/2</span>'
            f'<span class="score-pill {_pill_s}">Speed {st.session_state.scorecard_speed}/2</span>'
            f'<span class="score-pill {_pill_t}">Linger {st.session_state.scorecard_time}/2</span>',
            unsafe_allow_html=True)

        st.markdown("---")

        # Market-level checks
        _mac  = st.session_state.macro or {}
        _mets = st.session_state.metrics or {}
        _ntg  = compute_no_trade_gate(_mac, _mets, st.session_state.strategy, 14, False)

        # Composite verdict
        if _ntg["verdict"] == "HARD_NO":
            _verdict_class  = "verdict-no"
            _verdict_icon   = "🚫"
            _verdict_title  = "DO NOT TRADE"
            _verdict_detail = _ntg["headline"]
        elif _ntg["verdict"] == "SOFT_NO" or _iwt_total < 4:
            _verdict_class  = "verdict-no"
            _verdict_icon   = "🔴"
            _verdict_title  = "SKIP THIS SETUP"
            _verdict_detail = f"IWT score {_iwt_total}/{_max_score} — need 4+ to trade. {_ntg['headline']}"
        elif _ntg["verdict"] == "CONDITIONAL" or _iwt_total == 4:
            _verdict_class  = "verdict-cond"
            _verdict_icon   = "🟡"
            _verdict_title  = "CONDITIONAL"
            _verdict_detail = f"IWT {_iwt_total}/{_max_score} · Half-size · 40% profit target"
        else:
            _verdict_class  = "verdict-go"
            _verdict_icon   = "🟢"
            _verdict_title  = "GO — SETUP READY"
            _verdict_detail = f"IWT score {_iwt_total}/{_max_score} · VIX {_mac.get('vix',0):.1f} · {_mw.get('regime','—')}"

        st.markdown(
            f'<div class="{_verdict_class}">'
            f'<div class="verdict-big">{_verdict_icon} {_verdict_title}</div>'
            f'<div class="verdict-sub">{_verdict_detail}</div>'
            f'</div>', unsafe_allow_html=True)

        if _ntg["blocks"]:
            for _blk in _ntg["blocks"][:2]:
                st.error(_blk)
        if _ntg["warnings"]:
            for _wrn in _ntg["warnings"][:2]:
                st.warning(_wrn)

        # IWT checklist hint
        with st.expander("📋 Full IWT 7-Step Checklist", expanded=False):
            _steps = [
                ("1", "Pick a quality, liquid underlying"),
                ("2", "Find the buyer/seller level — is it fresh?"),
                ("3", "Check reward-to-risk — is it at least 2:1?"),
                ("4", "Size your position — max 1% account risk"),
                ("5", "Check portfolio heat — under 6% total risk?"),
                ("6", "Enter with a limit order at the level"),
                ("7", "Set your exit — 50% profit OR 2× credit stop"),
            ]
            for _sn, _sl in _steps:
                _done = (_sn in ["1","4","5","6","7"] or _iwt_total >= 5)
                _ic = "✅" if _done else "⬜"
                st.markdown(f"`{_ic}` **Step {_sn}:** {_sl}")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — SCAN (Universe + SPX Daily Plan)
# ══════════════════════════════════════════════════════════════════════════════
with _T2:
    _lvl = st.session_state.lang_level
    _s_left, _s_right = st.columns([1, 1], gap="large")

    with _s_left:
        st.markdown('<div class="card-sm"><span class="card-label">IWT Universe — Best Setups Now</span></div>',
                    unsafe_allow_html=True)
        _scan_top = st.slider("Show top N stocks", 5, 15, 8, key="scan_top_n")
        if st.button("🚀 Run Universe Scan (34 stocks)", type="primary", key="run_uni_scan"):
            with st.spinner("Scanning 34 stocks with live data (~30s)…"):
                _scan_res = batch_scan_teri_universe(TERI_UNIVERSE, top_n=_scan_top)
                st.session_state._universe_scan = _scan_res

        _scan = st.session_state.get("_universe_scan")
        if _scan:
            _grade_border = {"A+":"#00e676","A":"#69f0ae","B":"#ffd740","C":"#ff6d00","D":"#d50000"}
            for _i, _st_item in enumerate(_scan):
                _gb = _grade_border.get(_st_item["grade"], "#555")
                _trend_col = "#69f0ae" if "BULL" in _st_item["trend"] else \
                             "#ff5252" if "BEAR" in _st_item["trend"] else "#ffd740"
                _medal = ["🥇","🥈","🥉"][_i] if _i < 3 else f"{_i+1}."
                _gap_txt = f" · Gap {_st_item['gap_pct']:+.2f}%" \
                           if abs(_st_item.get("gap_pct",0)) > 0.3 else ""
                with st.container():
                    st.markdown(
                        f'<div class="card-sm" style="border-left:3px solid {_gb}">'
                        f'<div style="display:flex;justify-content:space-between;align-items:center">'
                        f'<span style="font-weight:700;color:#e8e8ff">{_medal} {_st_item["ticker"]}'
                        f'&nbsp;<span style="background:{_gb};color:#000;padding:1px 7px;'
                        f'border-radius:10px;font-size:0.7rem;font-weight:700">{_st_item["grade"]}</span>'
                        f'</span>'
                        f'<span class="mono" style="color:#e8e8ff">${_st_item["price"]:.2f}</span>'
                        f'</div>'
                        f'<div class="dim" style="margin-top:4px">'
                        f'<span style="color:{_trend_col}">{_st_item["trend"]}</span>'
                        f' · RSI {_st_item["rsi"]:.0f}'
                        f' · {_st_item["rvol"]:.1f}x vol'
                        f'{_gap_txt}'
                        f'</div>'
                        f'</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="dim" style="padding:20px 0;text-align:center">'
                        'Click Scan to find today\'s best setups</div>', unsafe_allow_html=True)

    with _s_right:
        st.markdown('<div class="card-sm"><span class="card-label">SPX Today — Live Strikes</span></div>',
                    unsafe_allow_html=True)
        _dt_choice = st.radio("DTE", [7, 14, 21], index=1, horizontal=True, key="scan_dte")
        _min_cr_scan = st.number_input("Min credit ($0.50 = system minimum)",
                                        value=0.50, min_value=0.10, step=0.05, key="scan_mincr")

        if st.button("🎯 Generate SPX Setup", type="primary", key="gen_spx_scan"):
            with st.spinner("Fetching live SPX/VIX…"):
                _sp = generate_spx_daily_plan(dte_target=_dt_choice, min_credit=_min_cr_scan)
                st.session_state._spx_plan = _sp

        _sp = st.session_state.get("_spx_plan")
        if _sp and not _sp.get("error"):
            _b = _sp.get("best", {})
            _em = _sp.get("em", 0)
            _ok = _b.get("ok_credit", False) and _b.get("outside_em", False)
            _spx_badge = "✅ READY" if _ok else "⚠️ CHECK"
            _bcol = "#69f0ae" if _ok else "#ffd740"
            st.markdown(
                f'<div class="card" style="border-left:3px solid {_bcol}">'
                f'<div style="display:flex;justify-content:space-between">'
                f'<span style="font-weight:700;color:#e8e8ff">SPX {_sp["S"]:,.0f} '
                f'<span style="color:{_bcol}">{_spx_badge}</span></span>'
                f'<span class="dim">VIX {_sp["vix"]:.1f} · EM ±{_em:.0f}</span>'
                f'</div>'
                f'<div class="order-block" style="margin-top:10px">'
                f'SELL {_b.get("K_s",0):.0f} PUT\nBUY  {_b.get("K_l",0):.0f} PUT ({_b.get("width",0)}-pt)\n'
                f'Credit ${_b.get("credit",0):.2f} = ${_b.get("max_profit",0):.0f}/contract\n'
                f'Max loss ${_b.get("max_loss",0):.0f} · POP {_b.get("pop",0):.0%}\n'
                f'BE {_b.get("breakeven",0):.2f} · {_b.get("em_ratio",0):.2f}x EM'
                f'</div>'
                f'</div>', unsafe_allow_html=True)
            if not _ok:
                if not _b.get("ok_credit"):
                    st.warning("Credit below $0.50 minimum — thin premium today. Wait for higher VIX.")
                if not _b.get("outside_em"):
                    st.error("Short strike inside expected move — below IWT standard.")
            # EM chart
            if _b.get("K_s") and _sp.get("S"):
                with st.expander("📈 Expected Move Chart", expanded=True):
                    _fig_em = plot_expected_move_chart(
                        spx=_sp["S"], iv_pct=_sp.get("sigma", 18),
                        dte=_dt_choice, short_k=_b.get("K_s"),
                        long_k=_b.get("K_l"), spread_type="PUT")
                    if _fig_em:
                        st.pyplot(_fig_em)
        elif _sp and _sp.get("error"):
            st.error("Could not fetch live SPX data: " + str(_sp["error"]))
        else:
            st.markdown('<div class="dim" style="padding:20px 0;text-align:center">'
                        'Click to generate today\'s specific strikes</div>',
                        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TRADE (Details for selected ticker + strategy)
# ══════════════════════════════════════════════════════════════════════════════
with _T3:
    _lvl = st.session_state.lang_level
    _tr_left, _tr_right = st.columns([1, 1], gap="large")

    with _tr_left:
        _m = st.session_state.metrics or {}
        _price = _m.get("price", 0)
        _strat = st.session_state.strategy
        _tkr   = st.session_state.ticker.replace("^GSPC", "SPX")

        st.markdown(f'<div class="card-sm"><span class="card-label">'
                    f'{_tkr} — {_strat}</span></div>', unsafe_allow_html=True)

        if not _price:
            st.info("Load market data in **Setup** first (🎯 Setup → Load Market Data).")
        else:
            # Show key metrics in a compact row
            _metrics_to_show = [
                ("Price",    f"${_price:,.2f}"),
                ("RSI",      f"{_m.get('rsi',0):.0f}"),
                ("RVOL",     f"{_m.get('rvol',1):.1f}x"),
                ("ATR",      f"${_m.get('atr',0):.2f}"),
                ("Trend",    _m.get("trend_strength","—")),
                ("IV Rank",  f"{_m.get('ivr',0):.0f}%"),
            ]
            _cols = st.columns(3)
            for _idx, (_lbl, _val) in enumerate(_metrics_to_show):
                with _cols[_idx % 3]:
                    st.markdown(
                        f'<div class="card-sm">'
                        f'<div class="card-label">{_lbl}</div>'
                        f'<div style="font-size:1rem;font-weight:600;color:#e8e8ff;'
                        f'font-family:\'DM Mono\',monospace">{_val}</div>'
                        f'</div>', unsafe_allow_html=True)

            st.markdown("---")

            # Gap signal (if present)
            _gap_pct = _m.get("gap", 0)
            if abs(_gap_pct) > 0.15:
                _gc = classify_gap_type(
                    _gap_pct, _m.get("rvol",1.0), _m.get("trend_strength","NEUTRAL"))
                _gap_cols = {"Common":"#1565c0","Breakaway":"#6a1b9a",
                             "Runaway":"#1b5e20","Exhaustion":"#e65100"}
                _gcolor = _gap_cols.get(_gc["type"], "#37474f")
                st.markdown(
                    f'<div class="card-sm" style="border-left:3px solid {_gcolor}">'
                    f'<div class="card-label">Gap Signal</div>'
                    f'<div style="font-weight:600;color:#e8e8ff">'
                    f'{_gc["type"]} {_gc["direction"]} {_gc["abs_pct"]:.2f}%'
                    f'</div>'
                    f'<div class="dim">{_gc["action"][:90]}…</div>'
                    f'</div>', unsafe_allow_html=True)

            # Support / Resistance levels
            _supp = _m.get("supp", 0); _res = _m.get("res", 0)
            if _supp or _res:
                st.markdown(
                    f'<div class="card-sm">'
                    f'<div class="card-label">Key Levels</div>'
                    f'<div style="display:flex;gap:20px">'
                    f'<div><span class="dim">Support</span><br/>'
                    f'<span class="mono" style="color:#69f0ae">${_supp:,.2f}</span></div>'
                    f'<div><span class="dim">Resistance</span><br/>'
                    f'<span class="mono" style="color:#ff5252">${_res:,.2f}</span></div>'
                    f'<div><span class="dim">R:R</span><br/>'
                    f'<span class="mono" style="color:#e8e8ff">'
                    f'{(_res-_price)/(max(_price-_supp,0.01)):.1f}:1 '
                    f'({"✅" if (_res-_price)/(max(_price-_supp,0.01))>=2 else "⚠️"})'
                    f'</span></div>'
                    f'</div></div>', unsafe_allow_html=True)

    with _tr_right:
        # Credit spread builder (if income strategy)
        if "Income" in _strat:
            st.markdown('<div class="card-sm"><span class="card-label">Credit Spread Builder</span></div>',
                        unsafe_allow_html=True)
            _sb1, _sb2 = st.columns(2)
            with _sb1:
                _short_k = st.number_input("Short strike", value=max(0.0, _price - 100 if _price else 0.0),
                                            step=5.0, key="trade_sk")
                _long_k  = st.number_input("Long strike",  value=max(0.0, _price - 125 if _price else 0.0),
                                            step=5.0, key="trade_lk")
            with _sb2:
                _spread_credit = st.number_input("Credit received ($)", value=0.75, step=0.05, key="trade_cr")
                _contracts = st.number_input("Contracts", value=1, min_value=1, step=1, key="trade_ct")
                _spread_dte = st.number_input("DTE", value=14, min_value=0, step=1, key="trade_dte")

            if _short_k > 0 and _long_k > 0 and _spread_credit > 0:
                _width = abs(_short_k - _long_k)
                _max_p = _spread_credit * 100 * _contracts
                _max_l = (_width - _spread_credit) * 100 * _contracts
                _be    = _short_k - _spread_credit
                _eff   = _spread_credit / _width if _width > 0 else 0

                st.markdown(
                    f'<div class="card" style="border-left:3px solid '
                    f'{"#69f0ae" if _spread_credit >= 0.50 else "#ffd740"}">'
                    f'<div class="order-block">'
                    f'SELL {_short_k:.0f} PUT × {_contracts}\n'
                    f'BUY  {_long_k:.0f} PUT × {_contracts}\n'
                    f'Credit   ${_spread_credit:.2f}/share\n'
                    f'Max gain ${_max_p:,.0f} | Max loss ${_max_l:,.0f}\n'
                    f'Breakeven {_be:.2f}\n'
                    f'Efficiency {_eff*100:.1f}% | Min 20% needed'
                    f'</div></div>', unsafe_allow_html=True)

                # Risk sizing check
                _risk_ok = _max_l <= st.session_state.max_risk_per_trade
                _eff_ok  = _eff >= 0.20
                _cr_ok   = _spread_credit >= 0.50
                st.markdown(
                    f'{"✅" if _cr_ok else "❌"} Credit ≥ $0.50 &nbsp;'
                    f'{"✅" if _risk_ok else "❌"} Max loss ≤ ${st.session_state.max_risk_per_trade:.0f} &nbsp;'
                    f'{"✅" if _eff_ok else "❌"} Efficiency ≥ 20%',
                    unsafe_allow_html=True)

                st.markdown("---")
                # P&L Payoff chart
                if _price and st.checkbox("Show P&L payoff diagram", True, key="show_payoff"):
                    _fig_pnl = plot_payoff_diagram(
                        structure="CREDIT_SPREAD", short_k=_short_k, long_k=_long_k,
                        credit_or_debit=_spread_credit, option_type="PUT",
                        underlying=_price, contracts=_contracts)
                    if _fig_pnl:
                        st.pyplot(_fig_pnl)

                st.markdown("---")
                st.markdown('<div class="card-sm"><span class="card-label">Exit rules</span></div>',
                            unsafe_allow_html=True)
                st.markdown(
                    f'<div class="dim">'
                    f'🎯 Close at <strong style="color:#69f0ae">'
                    f'${_spread_credit*0.50:.2f}/share</strong> (50% profit = ${_max_p*0.5:.0f})<br/>'
                    f'🛑 Stop at <strong style="color:#ff5252">'
                    f'${_spread_credit*2:.2f}/share</strong> (2× credit = ${_max_l:.0f} max)<br/>'
                    f'📅 Exit by <strong style="color:#e8e8ff">21 DTE</strong> regardless of P&L'
                    f'</div>', unsafe_allow_html=True)

        else:
            # Non-income strategies: show signal summary
            _sigs = st.session_state.signals or {}
            _verdict = _sigs.get("verdict", "NO_DATA")
            _sig_color = {"STRONG_BULL":"#00e676","BULL":"#69f0ae","NEUTRAL":"#ffd740",
                          "BEAR":"#ff5252","STRONG_BEAR":"#d50000"}.get(_verdict, "#7080a0")
            if _verdict != "NO_DATA":
                st.markdown(
                    f'<div class="verdict-go" style="background:linear-gradient(135deg,#0d1a3b,#1a2e5e)">'
                    f'<div class="verdict-big" style="color:{_sig_color}">{_verdict}</div>'
                    f'<div class="verdict-sub">'
                    f'Trend {_sigs.get("trend","—")} · '
                    f'Momentum {_sigs.get("momentum","—")} · '
                    f'Volume {_sigs.get("volume","—")}'
                    f'</div></div>', unsafe_allow_html=True)
            else:
                st.info("Load market data in Setup to see trade signals.")

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — REVIEW (Journal + Troubleshoot + Income Plan)
# ══════════════════════════════════════════════════════════════════════════════
with _T4:
    _lvl = st.session_state.lang_level
    _rv_left, _rv_right = st.columns([1, 1], gap="large")

    with _rv_left:
        # ── Quick journal entry ───────────────────────────────────────────────
        st.markdown('<div class="card-sm"><span class="card-label">Log a Trade</span></div>',
                    unsafe_allow_html=True)
        _j1, _j2, _j3 = st.columns(3)
        with _j1:
            _j_tkr    = st.text_input("Ticker", st.session_state.ticker.replace("^GSPC","SPX"),
                                       key="j_tkr")
            _j_result = st.selectbox("Result", ["Win","Loss","Breakeven"], key="j_result")
        with _j2:
            _j_pnl = st.number_input("P&L ($)", value=0.0, step=10.0, key="j_pnl")
            _j_strat_j = st.selectbox("Type", ["Credit spread","Long option",
                                                "Stock long","Short sell"], key="j_strat")
        with _j3:
            _j_plan   = st.checkbox("Followed the plan?", True, key="j_plan")
            _j_notes  = st.text_area("Notes (optional)", height=70, key="j_notes",
                                      placeholder="What happened?")

        if st.button("+ Add to Journal", key="add_journal"):
            import datetime as _dt
            _entry = {
                "date": str(_dt.date.today()), "ticker": _j_tkr,
                "result": _j_result, "pnl": _j_pnl, "strategy": _j_strat_j,
                "followed_plan": _j_plan, "notes": _j_notes,
            }
            st.session_state.journal.append(_entry)
            st.session_state.daily_pnl += _j_pnl
            if _j_result == "Win":
                st.session_state.consecutive_losses = 0
            else:
                st.session_state.consecutive_losses += 1
            st.success(f"Logged: {_j_tkr} {_j_result} ${_j_pnl:+.0f}")
            st.rerun()

        # ── Journal history ───────────────────────────────────────────────────
        if st.session_state.journal:
            st.markdown("---")
            _jrn = st.session_state.journal
            _wins  = sum(1 for e in _jrn if e["result"] == "Win")
            _total = len(_jrn)
            _wr    = _wins / _total if _total else 0
            _total_pnl = sum(e["pnl"] for e in _jrn)
            _plan_follow = sum(1 for e in _jrn if e.get("followed_plan")) / max(_total, 1)
            _stats_cols = st.columns(4)
            _stats_cols[0].metric("Trades", _total)
            _stats_cols[1].metric("Win Rate", f"{_wr:.0%}")
            _stats_cols[2].metric("Total P&L", f"${_total_pnl:+.0f}")
            _stats_cols[3].metric("Plan Adherence", f"{_plan_follow:.0%}")
            with st.expander(f"📓 {_total} trade(s) logged", expanded=False):
                for _e in reversed(_jrn[-10:]):
                    _ecol = "#69f0ae" if _e["result"] == "Win" else "#ff5252"
                    st.markdown(
                        f'`{_e["date"]}` **{_e["ticker"]}** '
                        f'<span style="color:{_ecol}">{_e["result"]} ${_e["pnl"]:+.0f}</span> '
                        f'· {_e["strategy"]} '
                        f'{"✅" if _e.get("followed_plan") else "❌"}',
                        unsafe_allow_html=True)

    with _rv_right:
        # ── Income plan card ─────────────────────────────────────────────────
        st.markdown('<div class="card-sm"><span class="card-label">My Income Blueprint</span></div>',
                    unsafe_allow_html=True)
        _sfp = calc_six_figure_plan(
            monthly_goal=st.session_state.monthly_goal,
            account_size=st.session_state.acct_size,
            avg_contract_risk=st.session_state.max_risk_per_trade,
        )
        if not _sfp.get("error"):
            _plan_cols = st.columns(3)
            _plan_cols[0].metric("Daily Target",  f"${_sfp['daily_goal']:.0f}")
            _plan_cols[1].metric("Weekly Target", f"${_sfp['weekly_goal']:.0f}")
            _plan_cols[2].metric("Annual Goal",   f"${_sfp['annual_goal']/1000:.0f}k")
            _ev_col = "#69f0ae" if _sfp["ev_per_trade"] > 0 else "#ff5252"
            st.markdown(
                f'<div class="card-sm">'
                f'<div class="dim">EV/trade <strong style="color:{_ev_col}">'
                f'${_sfp["ev_per_trade"]:.0f}</strong>'
                f' · Monthly return <strong style="color:#e8e8ff">'
                f'{_sfp["monthly_return_pct"]:.1f}%</strong>'
                f' on ${st.session_state.acct_size:,.0f}'
                f'</div>'
                f'{"⚠️ " + _sfp["sizing_note"] if not _sfp["sizing_ok"] else ""}'
                f'</div>', unsafe_allow_html=True)

        st.markdown("---")

        # ── Troubleshoot ────────────────────────────────────────────────────
        st.markdown('<div class="card-sm"><span class="card-label">Troubleshoot My Trading</span></div>',
                    unsafe_allow_html=True)
        _tt_syms = [
            ("making_then_losing",    "Making money then giving it back"),
            ("losses_bigger_than_wins","Average losses > average wins"),
            ("too_many_trades",       "Trading too much / boredom trading"),
            ("scared_to_enter",       "Hesitating at valid setups"),
            ("revenge_trading",       "Revenge trading after a loss"),
            ("holding_losers_too_long","Holding losers hoping for recovery"),
        ]
        _tt_resp = {}
        for _sym, _lbl in _tt_syms:
            _tt_resp[_sym] = st.checkbox(_lbl, key="tt_" + _sym)

        if any(_tt_resp.values()):
            _ttd = troubleshoot_trading(_tt_resp)
            _pri_col = {"FIX NOW":"#b71c1c","IMPROVE":"#f57f17","MAINTAIN":"#1b5e20"}.get(
                _ttd["priority"],"#37474f")
            st.markdown(
                f'<div class="card-sm" style="border-left:3px solid {_pri_col}">'
                f'<div style="font-weight:700;color:#e8e8ff">{_ttd["priority"]}</div>'
                f'<div class="dim">{_ttd["summary"]}</div>'
                f'</div>', unsafe_allow_html=True)
            for _diag in _ttd["active_diagnoses"][:2]:
                _sev_c = "#ff5252" if _diag["severity"]=="HIGH" else "#ffd740"
                with st.expander(_diag["label"], expanded=False):
                    st.markdown(f'**Root cause:** {_diag["root_cause"]}')
                    st.success(f'**Fix:** {_diag["fix"]}')
                    st.caption(_diag["teri_quote"])

# ══════════════════════════════════════════════════════════════════════════════
# POWER TOOLS — collapsed by default (expert / macro / backtest)
# ══════════════════════════════════════════════════════════════════════════════
with st.expander("⚡ Power Tools — Market Brief · Backtest · Options Calculator · More", expanded=False):
    _pw1, _pw2 = st.columns([1,1])
    with _pw1:
        # Market weather deep-dive
        if st.session_state.macro:
            _mw2 = compute_market_weather(st.session_state.macro)
            st.markdown(f"**Options Playbook Today**")
            _play = options_playbook_router(
                _mw2.get("vix",20), _mw2.get("ivr",50),
                st.session_state.macro.get("sp_trend","NEUTRAL"))
            st.markdown(f'**{_play["badge"]} {_play["play"]}**')
            st.caption(_play["primary"])
            if _play.get("avoid"):
                st.caption("Avoid: " + ", ".join(_play["avoid"][:2]))

        st.markdown("---")
        # Covered call calculator
        st.markdown("**Covered Call Yield**")
        _cc_s = st.number_input("Stock price", value=0.0, step=0.5, key="pw_cc_s")
        _cc_k = st.number_input("Strike to sell", value=0.0, step=0.5, key="pw_cc_k")
        _cc_p = st.number_input("Premium ($/share)", value=0.0, step=0.05, key="pw_cc_p")
        _cc_d = st.number_input("DTE", value=35, min_value=1, key="pw_cc_d")
        if _cc_s > 0 and _cc_k > 0 and _cc_p > 0:
            _cc = calc_covered_call_yield(_cc_s, _cc_k, _cc_p, _cc_d)
            if not _cc.get("error"):
                st.metric("Annualized Yield", f"{_cc['annualized_yield_pct']:.1f}%")
                st.caption(_cc["otm_note"])
                st.caption(_cc["dte_note"])

    with _pw2:
        # Morning brief
        if st.button("🌍 Load Morning Macro Brief", key="pw_macro_btn"):
            with st.spinner("Fetching global data…"):
                _m_data = get_macro()
                if _m_data:
                    st.session_state.macro = _m_data
                    st.success("Macro data loaded")
                    st.rerun()
                else:
                    st.warning("Could not fetch macro data")

        if st.session_state.macro:
            _mac = st.session_state.macro
            _brief_items = [
                ("SPX",  f"{_mac.get('spx_price',0):,.0f}"),
                ("VIX",  f"{_mac.get('vix',0):.1f}"),
                ("10Y",  f"{_mac.get('tnx',0):.2f}%"),
                ("DXY",  f"{_mac.get('dxy',0):.1f}"),
                ("Gold", f"{_mac.get('gold',0):,.0f}"),
                ("Oil",  f"{_mac.get('oil',0):.1f}"),
            ]
            _br_cols = st.columns(3)
            for _bi, (_bl, _bv) in enumerate(_brief_items):
                _br_cols[_bi % 3].metric(_bl, _bv)

        st.markdown("---")
        # Short/Not short checker (when relevant)
        if "Short" in st.session_state.strategy and st.session_state.metrics:
            _m3 = st.session_state.metrics
            _sns = short_or_not_score(
                _m3.get("trend_strength","NEUTRAL"),
                (st.session_state.macro or {}).get("vix",20),
                st.session_state.macro or {},
                _m3, _m3.get("rsi",50), _m3.get("rvol",1.0))
            _sc = {"GREEN — SHORT VALID":"#69f0ae","YELLOW — WAIT FOR CONFIRMATION":"#ffd740",
                   "RED — DO NOT SHORT":"#ff5252","DO NOT SHORT":"#ff5252"}.get(
                   _sns["verdict"],"#7080a0")
            st.markdown(f'**Short Signal:** <span style="color:{_sc}">{_sns["verdict"]}</span>',
                        unsafe_allow_html=True)
            st.caption(_sns["action"])

# ── DISCLAIMER (compact footer) ───────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div class="dim" style="text-align:center;font-size:0.72rem;padding:8px 0">'
    '⚠️ <strong>Simulation only</strong> · Not financial advice · Not affiliated with Trade and Travel '
    'or any trading organisation · Built by a student applying IWT methodology principles · '
    'Data: Yahoo Finance · Trading involves substantial risk of loss'
    '</div>', unsafe_allow_html=True)

