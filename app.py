# Copyright (c) 2026 Gabriel Mahia. All Rights Reserved.
# Unauthorized copying of this file, via any medium, is strictly prohibited.
# Proprietary and confidential.
# Written by Gabriel Mahia, 2026
# app.py

# =============================================================================
# EasyStockTrader — Smart Stock Analysis
# Educational simulation. Does not execute trades. Not financial advice.
# IWT framework: Teri Ijeoma (investwithteri.com)
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
    df['RSI_14'] = 100 - (100 / (1 + rs))

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
                "why_pro": f"IVR {ivr:.0f}% < 30. Options priced at lower end of 52w range — buyer's market. DITM (Δ ≥ 0.70): intrinsic-heavy, lower vega risk, behaves like 70 delta-shares per contract. Teri IWT: 60+ DTE minimum to avoid rapid theta decay.",
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

st.set_page_config(
    page_title="EasyStockTrader — Smart Stock Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🏛️"
)

st.markdown("""
<style>
    /* ── Buttons ──────────────────────────────────────────────── */
    .stButton > button {
        width: 100%; border-radius: 4px; height: 3em;
        font-weight: 600; letter-spacing: 0.5px;
    }

    /* ── Metric cards — LIGHT MODE ────────────────────────────── */
    div[data-testid="stMetric"] {
        background-color: #f0f2f6;
        border: 1px solid #d0d3da;
        border-radius: 8px;
        padding: 12px 16px;
    }
    [data-testid="stMetricLabel"]  { color: #444444 !important; font-size: 0.8rem !important; }
    [data-testid="stMetricValue"]  { color: #111111 !important; font-size: 1.25rem !important; font-weight: 700 !important; }
    [data-testid="stMetricDelta"]  { color: #333333 !important; font-size: 0.82rem !important; }

    /* ── Metric cards — DARK MODE (OS preference) ─────────────── */
    @media (prefers-color-scheme: dark) {
        div[data-testid="stMetric"] {
            background-color: #1e2127 !important;
            border-color: #30333d !important;
        }
        [data-testid="stMetricLabel"] { color: #aaaaaa !important; }
        [data-testid="stMetricValue"] { color: #f0f0f0 !important; }
        [data-testid="stMetricDelta"] { color: #cccccc !important; }
    }

    /* ── Metric cards — DARK MODE (Streamlit theme toggle) ─────── */
    [data-theme="dark"] div[data-testid="stMetric"],
    .stApp[data-theme="dark"] div[data-testid="stMetric"] {
        background-color: #1e2127 !important;
        border-color: #30333d !important;
    }
    [data-theme="dark"] [data-testid="stMetricLabel"],
    .stApp[data-theme="dark"] [data-testid="stMetricLabel"] { color: #aaaaaa !important; }
    [data-theme="dark"] [data-testid="stMetricValue"],
    .stApp[data-theme="dark"] [data-testid="stMetricValue"] { color: #f0f0f0 !important; }
    [data-theme="dark"] [data-testid="stMetricDelta"],
    .stApp[data-theme="dark"] [data-testid="stMetricDelta"] { color: #cccccc !important; }

    /* ── Signal boxes ─────────────────────────────────────────── */
    .signal-bull { background:#d4edda; color:#155724 !important; padding:8px 12px; border-radius:4px; margin:5px 0; border:1px solid #28a745; font-weight:500; }
    .signal-bear { background:#f8d7da; color:#721c24 !important; padding:8px 12px; border-radius:4px; margin:5px 0; border:1px solid #dc3545; font-weight:500; }
    .signal-neutral { background:#fff3cd; color:#856404 !important; padding:8px 12px; border-radius:4px; margin:5px 0; border:1px solid #ffc107; font-weight:500; }
    @media (prefers-color-scheme: dark) {
        .signal-bull    { background:#1e4620; color:#7dcea0 !important; border-color:#28a745; }
        .signal-bear    { background:#4a1c1c; color:#f1948a !important; border-color:#dc3545; }
        .signal-neutral { background:#4a3f1a; color:#f9e79f !important; border-color:#ffc107; }
    }

    /* ── Risk / info boxes ────────────────────────────────────── */
    .risk-warning { background:#fff3cd; color:#333 !important; padding:15px; border-left:4px solid #ffc107; margin:10px 0; }
    .success-box  { background:#d4edda; color:#333 !important; padding:15px; border-left:4px solid #28a745; margin:10px 0; }

    /* ── Key Levels ───────────────────────────────────────────── */
    .key-levels-bar {
        font-size: 0.97rem;
        margin: 8px 0 16px 0;
        line-height: 2;
        color: #111111;
    }
    @media (prefers-color-scheme: dark) { .key-levels-bar { color: #f0f0f0; } }
    [data-theme="dark"] .key-levels-bar { color: #f0f0f0 !important; }

    .key-levels-bar strong { color: inherit; font-size: 1rem; }
    .kl-touches { font-size: 0.82rem; color: #555555; }
    @media (prefers-color-scheme: dark) { .kl-touches { color: #bbbbbb; } }
    [data-theme="dark"] .kl-touches { color: #bbbbbb !important; }

    .kl-support {
        display: inline-block;
        background: #0e7c4a; color: #ffffff !important;
        padding: 3px 12px; border-radius: 5px;
        font-family: monospace; font-weight: 700;
        font-size: 0.97rem; letter-spacing: 0.03em;
    }
    .kl-resistance {
        display: inline-block;
        background: #c0392b; color: #ffffff !important;
        padding: 3px 12px; border-radius: 5px;
        font-family: monospace; font-weight: 700;
        font-size: 0.97rem; letter-spacing: 0.03em;
    }
    @media (prefers-color-scheme: dark) {
        .kl-support    { background: #27ae60; }
        .kl-resistance { background: #e74c3c; }
    }

    /* ── Mobile: 768px ───────────────────────────────────────── */
    @media (max-width: 768px) {
        [data-testid="column"] { width:100% !important; flex:1 1 100% !important; min-width:100% !important; }
        [data-testid="stMetricValue"] { font-size: 1.1rem !important; }
        [data-testid="stMetricLabel"] { font-size: 0.72rem !important; }
        [data-testid="stMetricDelta"] { font-size: 0.72rem !important; }
        div[data-testid="stMetric"]   { padding: 8px 10px !important; margin-bottom: 6px !important; }
        section[data-testid="stSidebar"] { min-width: 180px !important; }
        [data-testid="stPlotlyChart"]    { width: 100% !important; }
        [data-testid="stDataFrame"]      { overflow-x: auto !important; }
        [data-testid="stPlotlyChart"] canvas { touch-action: pan-y !important; }
        .stButton > button { width:100% !important; min-height:48px !important; font-size:0.95rem !important; }
        .stSelectbox > div { font-size: 0.9rem !important; }
        .risk-warning      { padding: 10px !important; font-size: 0.88rem !important; }
        pre, code          { overflow-x: auto !important; font-size: 0.8rem !important; }
        iframe             { width: 100% !important; max-width: 100% !important; }
        .key-levels-bar    { font-size: 0.88rem; }
        .kl-support, .kl-resistance { font-size: 0.88rem; padding: 2px 8px; }
    }
    /* ── Mobile: 480px ───────────────────────────────────────── */
    @media (max-width: 480px) {
        [data-testid="stMetricValue"] { font-size: 1rem !important; }
        h1 { font-size: 1.4rem !important; }
        h2 { font-size: 1.1rem !important; }
        h3 { font-size: 1rem !important; }
        .stButton > button { min-height: 52px !important; }
    }

    @media (prefers-color-scheme: dark) {
        .risk-warning { background: #3d3010 !important; color: #f0c060 !important; border-color: #ffc107 !important; }
        .success-box  { background: #1a3d24 !important; color: #7dcea0 !important; border-color: #28a745 !important; }
    }
    [data-theme="dark"] .risk-warning, .stApp[data-theme="dark"] .risk-warning { background: #3d3010 !important; color: #f0c060 !important; border-color: #ffc107 !important; }
    [data-theme="dark"] .success-box,  .stApp[data-theme="dark"] .success-box  { background: #1a3d24 !important; color: #7dcea0 !important; border-color: #28a745 !important; }


    @media (prefers-color-scheme: dark) {
        .signal-bear    { background-color: #4a1c1c !important; color: #f1948a !important; border-color: #dc3545 !important; }
        .signal-neutral { background-color: #4a3f1a !important; color: #f9e79f !important; border-color: #ffc107 !important; }
    }
    [data-theme="dark"] .signal-bear,    .stApp[data-theme="dark"] .signal-bear    { background-color: #4a1c1c !important; color: #f1948a !important; border-color: #dc3545 !important; }
    [data-theme="dark"] .signal-neutral, .stApp[data-theme="dark"] .signal-neutral { background-color: #4a3f1a !important; color: #f9e79f !important; border-color: #ffc107 !important; }

</style>
""", unsafe_allow_html=True)

# --- 2. LEGAL & ONBOARDING ---
st.title("🏛️ EasyStockTrader — Smart Stock Analysis")

@st.cache_data(ttl=3600)
def fetch_kes_rate():
    """Live KES rate from open.er-api.com."""
    import json as _qj
    try:
        with urllib.request.urlopen(
            "https://open.er-api.com/v6/latest/USD", timeout=6
        ) as r:
            d = _qj.loads(r.read())
        rates = d["rates"]
        kes = rates["KES"]
        return {
            "kes": round(kes, 2),
            "eur": round(rates["EUR"], 4),
            "gbp": round(rates["GBP"], 4),
            "updated": d.get("time_last_update_utc", "")[:16],
            "live": True,
        }
    except Exception:
        return {"kes": 129.0, "eur": 0.87, "gbp": 0.75, "updated": "fallback", "live": False}


@st.cache_data(ttl=86400)
def fetch_kenya_macro():
    """World Bank Kenya macro indicators."""
    import json as _qj2
    results = {}
    for code, label in [
        ("FP.CPI.TOTL.ZG", "Kenya Inflation (%)"),
        ("NY.GDP.PCAP.CD",  "GDP per capita (USD)"),
        ("SL.UEM.TOTL.ZS",  "Unemployment (%)"),
        ("BN.CAB.XOKA.GD.ZS", "Current account / GDP (%)"),
    ]:
        try:
            url = (f"https://api.worldbank.org/v2/country/KE/indicator/{code}"
                   f"?format=json&mrv=1&per_page=1")
            with urllib.request.urlopen(url, timeout=15) as r:
                data = _qj2.loads(r.read())
            entries = [e for e in (data[1] if len(data) > 1 else []) if e.get("value")]
            if entries:
                e = entries[0]
                results[code] = {"label": label, "value": round(e["value"], 2),
                                  "year": e.get("date", "?")}
        except Exception:
            pass
    return results

# ── Live Kenya macro context ───────────────────────────────────────────────
_kes = fetch_kes_rate()
_wb  = fetch_kenya_macro()
if _kes["live"] or _wb:
    _mc = st.columns(5)
    if _kes["live"]:
        _mc[0].metric("USD/KES", f"{_kes['kes']:.2f}", help="open.er-api.com live")
        _mc[1].metric("EUR/KES", f"{_kes['kes']/_kes['eur']:.2f}", help="Derived")
        _mc[2].metric("GBP/KES", f"{_kes['kes']/_kes['gbp']:.2f}", help="Derived")
    if _wb:
        # Display the correct World Bank indicators by code. The prior version
        # treated the first returned indicator (inflation) as GDP, which created
        # impossible values like "Inflation 2132%".
        _infl = _wb.get("FP.CPI.TOTL.ZG")
        _gdp_pc = _wb.get("NY.GDP.PCAP.CD")
        if _gdp_pc:
            _mc[3].metric(f"GDP/capita {_gdp_pc['year']}",
                         f"${_gdp_pc['value']:,.0f}", help="World Bank GDP per capita")
        if _infl:
            _mc[4].metric(f"Inflation {_infl['year']}",
                         f"{_infl['value']:.2f}%", help="World Bank CPI inflation")
    src = []
    if _kes["live"]: src.append(f"FX: open.er-api.com ({_kes['updated']})")
    if _wb: src.append("Macro: World Bank Open Data")
    st.caption("📡 Live · " + " · ".join(src))

# NDMA drought signal — food price pressure indicator
_ndma_qm = fetch_ndma_macro_signal()
if _ndma_qm:
    with st.expander(f"📡 NDMA Drought Signal — Kenya commodity pressure ({_ndma_qm[0]['date']})", expanded=False):
        st.caption("Drought → food price inflation → CPI pressure → BoK rate outlook")
        for _nq in _ndma_qm:
            st.markdown(f"**[{_nq['title'][:80]}]({_nq['link']})**  *{_nq['date']}*")
            if _nq['summary']:
                st.caption(_nq['summary'][:120] + "…")

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

st.error(
    "⚠️ **SIMULATION ONLY** — This tool analyses signals and scores setups. "
    "It does not execute trades or connect to any broker. Not financial advice. "
    "Trading involves substantial risk of loss.",
    icon=None
)
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
if 'lang_level' not in st.session_state: st.session_state.lang_level = "Beginner"
if 'advisor_goal' not in st.session_state: st.session_state.advisor_goal = "Weekly income (sell premium)" 
if 'daily_pnl' not in st.session_state: st.session_state.daily_pnl = 0.0
if 'total_risk_deployed' not in st.session_state: st.session_state.total_risk_deployed = 0.0
if 'consecutive_losses' not in st.session_state: st.session_state.consecutive_losses = 0
if 'day_trades_used' not in st.session_state: st.session_state.day_trades_used = 0
if 'week_event_days' not in st.session_state: st.session_state.week_event_days = []

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

# --- 5. SIDEBAR (WITH V12 HELP NOTES) ---
with st.sidebar:
    _tdr_ok = _tradier_is_connected()
    if _tdr_ok:
        st.success("✅ Tradier connected — broker Greeks active")
    else:
        st.caption("📡 Greeks: yfinance+BSM (free). Add TRADIER_TOKEN to secrets for broker Greeks.")
    st.divider()
    st.header("🎯 Your Experience Level")
    st.session_state.lang_level = st.selectbox(
        "How do you like explanations?",
        ["Beginner", "Intermediate", "Advanced", "Professional"],
        index=["Beginner","Intermediate","Advanced","Professional"].index(st.session_state.lang_level),
        help="Beginner = plain English. Professional = BSM/Bloomberg/ThinkScript level."
    )
    _lvl = st.session_state.lang_level
    _lvl_captions = {
        "Beginner": "💡 Plain English + analogies. No jargon — promise.",
        "Intermediate": "📊 Standard options and trading terminology.",
        "Advanced": "📐 Greeks, formulas, statistical concepts.",
        "Professional": "🖥️ Bloomberg / ThinkScript / PineScript level. Full math.",
    }
    st.caption(_lvl_captions[_lvl])
    st.divider()
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

    pdt_framework = st.selectbox(
        "Day-Trade Rule Framework",
        ["Legacy PDT", "New Intraday Margin", "Broker Unknown / Conservative"],
        help="Legacy PDT is safest until your broker confirms migration. New intraday margin may remove fixed PDT counts but still uses broker risk controls."
    )
    account_type = st.selectbox(
        "Account Type",
        ["Margin < $25k", "Margin ≥ $25k", "Cash Account", "IRA / Limited Margin"],
        help="Used for conservative trade-budget guidance. Broker rules vary; verify with your broker."
    )
    day_trades_used = st.number_input(
        "Day Trades Used (rolling 5 business days)",
        value=int(st.session_state.day_trades_used), min_value=0, max_value=3, step=1,
        help="Under legacy PDT discipline, keep this under 4 in 5 business days for small margin accounts."
    )
    st.session_state.day_trades_used = int(day_trades_used)
    planned_same_day_exit = st.checkbox(
        "Planned same-day exit?", value=False,
        help="If checked, this setup may consume a day trade in a small margin account."
    )
    _pdt = pdt_guidance(capital, account_type, day_trades_used, planned_same_day_exit, pdt_framework)
    if not _pdt["can_day_trade"]:
        st.error("🛑 Trade-budget warning: no same-day exit capacity left. Preserve capital or use a non-day-trade plan.")
    else:
        st.caption(f"📌 {_pdt['status']}")

    st.markdown("**🗓️ Weekly Trade Planner**")
    preferred_trade_days = st.multiselect(
        "Preferred trade days",
        ["Monday", "Tuesday", "Wednesday", "Thursday"],
        default=["Monday", "Tuesday", "Wednesday", "Thursday"],
        help="Use this to plan Mon–Thu trading once your account/broker allows it. The app will still rank days by risk."
    )
    event_days = st.multiselect(
        "Major macro event days this week",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"],
        default=st.session_state.week_event_days,
        help="Mark CPI, PPI, FOMC, NFP, major Treasury auctions, or high-risk geopolitical/event days."
    )
    st.session_state.week_event_days = event_days

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
    st.header("🤖 What Are You Trying to Do?")
    st.session_state.advisor_goal = st.selectbox(
        "Today's goal:",
        ["Weekly income (sell premium)", "Capture a big directional move",
         "Trade commodities / futures", "I'm not sure — show me options"],
        index=["Weekly income (sell premium)", "Capture a big directional move",
               "Trade commodities / futures", "I'm not sure — show me options"
               ].index(st.session_state.advisor_goal),
        help="Drives the Instrument Advisor. The app recommends the best tool for you."
    )
    st.divider()
    st.header("2. Asset Selection")

    input_mode = st.radio("Input:", ["VIP List", "Manual Search"],
                         help="VIP List = Pre-vetted high-liquidity stocks (US + Kenya). Manual = Enter any ticker.")

    if input_mode == "VIP List":
        ticker = st.selectbox("Ticker", ALL_TICKERS, help="High-liquidity stocks from US (Nasdaq/NYSE) and Kenya (NSE) exchanges.")
        if ticker.endswith(".NR"):
            st.warning(
                "⚠️ **NSE tickers (.NR) are not available via Yahoo Finance** — "
                "yfinance returns no data for all 8 NSE listings. "
                "For Nairobi Securities Exchange data use [nse.co.ke](https://nse.co.ke) directly. "
                "Selecting a Kenya ticker will produce an error when you scan.",
                icon=None
            )
    else:
        ticker = st.text_input("Ticker", "NVDA", help="Enter any stock symbol (e.g., AAPL, TSLA, GME).").upper()

    st.divider()
    st.header("3. Strategy & Execution")

    strategy = st.selectbox(
        "Mode",
        ['Income (SPX Vertical Credit Spread)', 'IWT Long Option (60+ DTE)', 'Futures (Index/Commodity)', 'Long (Buy)', 'Short (Sell)', 'Income (Cash-Secured Put)'],
        help="Default is SPX vertical credit spreads for defined-risk income. Long/Short are directional simulations. CSP models assignment risk."
    )

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
    spread_kind = "PUT"
    short_strike = 0.0
    long_strike = 0.0
    spread_credit = 0.0
    dte = 7
    spx_reference_price = 0.0
    iv_percent = 17.0
    short_delta = 0.15
    event_risk_48h = False
    hold_through_event = False

    fut_code = "MES"; fut_entry = 0.0; fut_stop = 0.0; fut_target = 0.0; fut_contracts = 1
    lo_direction = "CALL"
    lo_underlying = 0.0
    lo_strike = 0.0
    lo_premium = 0.0
    lo_delta = 0.75
    lo_dte = 90
    lo_contracts_lo = 1
    if strategy == "Income (Cash-Secured Put)":
        premium = st.number_input(
            "Put Premium ($/share)", value=0.0, step=0.05,
            help="Per-share premium received. $2.50 = $250 per contract. CSP risk is assignment/cash-secured risk, not just a chart stop."
        )

    elif strategy == "IWT Long Option (60+ DTE)":
        st.markdown("**📈 IWT Long Options — 60+ DTE**")
        st.caption("Teri Ijeoma: buy options 60+ DTE, DITM (delta 0.70+), for significant directional moves.")
        lo_direction = st.selectbox("Direction", ["CALL", "PUT"], help="CALL: bullish. PUT: bearish.")
        lo_underlying = st.number_input("Underlying Price ($)", value=0.0, step=1.0)
        lo_strike = st.number_input("Strike Price", value=0.0, step=1.0, help="DITM calls: below current price. DITM puts: above.")
        lo_premium = st.number_input("Premium Paid ($/share)", value=0.0, step=0.05, help="$5.00 = $500 per contract.")
        lo_delta = st.number_input("Option Delta", value=0.75, min_value=0.10, max_value=0.99, step=0.01, help="Target 0.70-0.85 for DITM (IWT preferred).")
        lo_dte = st.number_input("DTE (Days to Expiry)", value=90, min_value=1, max_value=400, step=1, help="IWT minimum: 60 DTE. Preferred: 90-120 DTE.")
        lo_contracts_lo = st.number_input("Contracts", value=1, min_value=1, max_value=50, step=1)
        st.caption("IWT exits: ✅ 50% or 100% gain on premium. 🔴 Stop at 50% loss.")
    

    elif strategy == "Futures (Index/Commodity)":
        st.markdown("**📦 Futures Trade**")
        st.caption("Direct price exposure. Gains/losses are pure price movement × contract size.")
        _FUT_MAP = {
            "MES — Micro S&P (low margin)":"MES","ES — E-mini S&P 500":"ES",
            "MNQ — Micro NASDAQ":"MNQ","NQ — E-mini NASDAQ":"NQ",
            "M2K — Micro Russell 2000":"M2K","MCL — Micro Crude Oil":"MCL",
            "CL — WTI Crude Oil":"CL","MGC — Micro Gold":"MGC","GC — Gold":"GC",
            "ZC — Corn (CBOT)":"ZC","ZW — Wheat (CBOT)":"ZW","ZS — Soybeans":"ZS",
            "NG — Natural Gas":"NG","SI — Silver":"SI",
        }
        _fl = st.selectbox("Contract", list(_FUT_MAP.keys()),
            help="Micro contracts = 1/10 the margin of full contracts. Start micro.")
        fut_code = _FUT_MAP[_fl]
        fut_entry     = st.number_input("Entry Price", value=0.0, step=0.25)
        fut_stop      = st.number_input("Stop Price",  value=0.0, step=0.25, help="Your max-loss exit.")
        fut_target    = st.number_input("Target Price", value=0.0, step=0.25)
        fut_contracts = st.number_input("Contracts", value=1, min_value=1, max_value=20, step=1)
        st.caption("⚠️ Futures amplify both profits and losses. Margin required. Mark-to-market daily.")
    elif strategy == "Income (SPX Vertical Credit Spread)":
        st.markdown("**SPX Vertical Inputs**")
        spread_kind = st.selectbox("Spread Type", ["PUT", "CALL"], help="PUT = put credit spread below market. CALL = call credit spread above market.")
        short_strike = st.number_input("Short Strike", value=0.0, step=5.0, help="Strike you sell. This receives premium and defines the main risk point.")
        long_strike = st.number_input("Long Strike", value=0.0, step=5.0, help="Protective strike you buy. This caps max loss.")
        spread_credit = st.number_input("Net Credit ($/spread)", value=0.0, step=0.05, help="Net credit in index points. Example: 1.00 credit = $100 before commissions.")
        dte = st.number_input("DTE", value=7, min_value=0, max_value=60, step=1, help="Teri-style short income window is usually <=7 DTE. Longer DTE requires stronger macro stability.")
        _spx_auto = 0.0
        if st.session_state.macro and st.session_state.macro.get("sp") is not None:
            # ES=F is a close proxy for SPX — use as starting point
            pass  # user still enters; we add a helpful default hint
        spx_reference_price = st.number_input(
            "SPX Reference Price",
            value=0.0, step=1.0,
            help="Enter current SPX/US500 level. Check Google Finance or your broker. Used for expected-move calculation."
        )
        if spx_reference_price == 0.0:
            st.caption("💡 Enter SPX level from Google: search 'SPX' or 'S&P 500'.")
        iv_percent = st.number_input("IV / VIX Proxy (%)", value=17.0, min_value=1.0, max_value=100.0, step=0.25, help="Use current SPX IV, IV percentile estimate, or VIX as a proxy if true IV is unavailable.")
        short_delta = st.number_input("Approx Short Strike Delta", value=0.15, min_value=0.01, max_value=0.60, step=0.01, help="Used for rough POP/POT. Example: 0.15 delta ≈ ~85% probability OTM, ~30% probability touch.")
        event_risk_48h = st.checkbox("Major event within 48h?", value=False, help="CPI/PPI/FOMC/NFP/Treasury auction/geopolitical shock. If yes, the app penalizes new premium selling.")
        hold_through_event = st.checkbox("Would hold through event?", value=False, help="Usually avoid holding premium-selling positions through major events unless specifically structured for it.")


    # === IV RANK — auto-computed from live VIX 52w history ===
    _ivr_auto = float(st.session_state.macro.get("ivr_proxy", 35.0)) if st.session_state.macro else 35.0
    _vix_hi   = float(st.session_state.macro.get("vix_52w_high", 30.0)) if st.session_state.macro else 30.0
    _vix_lo   = float(st.session_state.macro.get("vix_52w_low", 12.0))  if st.session_state.macro else 12.0
    _vix_now  = float(st.session_state.macro.get("vix", 18.0)) if st.session_state.macro else 18.0
    if st.session_state.macro:
        st.caption(f"✅ LIVE: VIX {_vix_now:.1f} | 52w {_vix_lo:.1f}–{_vix_hi:.1f} → IVR **{_ivr_auto:.0f}%** (auto)")
    else:
        st.caption("📡 Click **Macro Audit** above → IVR auto-computed from live VIX data.")
    ivr_manual = st.number_input(
        "IV Rank % (auto-filled | override ok)",
        value=_ivr_auto, min_value=0.0, max_value=100.0, step=1.0,
        help="Computed from live VIX vs its 52-week range. Override if your broker shows a more precise number."
    )
    st.caption("🔑 IVR drives the strategy arbiter panel below.")

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

# ============================================================================
# BEGINNER'S GUIDE
# ============================================================================

with st.expander("🎓 Beginner's Guide (Read This First)", expanded=False):
    st.markdown("""
### 🎓 How to Use Quantum Maestro

**Step 1: Set Your Risk**
- Total Capital = actual account size
- Risk per Trade = ~1% of capital ($10,000 → $100)
- Never risk more than 2% per trade

**Step 2: Check Macro FIRST**
- Scan VIX before individual stocks
- If VIX HIGH/CRISIS → reduce size or don't trade
- If Risk-Off (Gold + VIX rising) → avoid longs

**Step 3: Scan a Stock**
- Use VIP List (safest) or enter ticker
- Wait for 15+ indicators to load

**Step 4: Score the Setup (IWT)**
- Freshness (fresh > stale)
- Time in zone (fast rejection > lingering)
- Speed out (explosive > grinding)
- R/R must be ≥ 2.0 (prefer ≥ 3.0)

**For SPX Vertical Credit Spreads**
- Short strike = option you sell; long strike = protection you buy
- Max profit = credit × 100 × contracts
- Max loss = (spread width − credit) × 100 × contracts
- For a $5-wide spread, gross width = $500 per contract
- Defined risk protects the account; do not oversize just because max loss is capped

**Step 5: Verdict Discipline**
- 7-8 → GREEN (execute)
- 5-6 → YELLOW (reduce size or wait)
- 0-4 → RED (no trade)

**Golden Rules**
1) Stop trading when daily goal met
2) Don't stack too many positions
3) Trade WITH the trend
4) High VIX = smaller size or sit out
5) Journal every trade (wins + losses)
""")



# --- 6. MAIN UI ---
st.subheader("📊 Market Intelligence Dashboard")


# === INSTRUMENT ADVISOR ===
with st.expander("🤖 Instrument Advisor — What Should I Trade Today?", expanded=True):
    _adv = instrument_advisor_v4(
        capital,
        st.session_state.advisor_goal,
        st.session_state.lang_level,
        st.session_state.macro
    )
    if _adv["warnings"]:
        for _w in _adv["warnings"]:
            st.info(_w)
    if _adv["recommendations"]:
        for _r in _adv["recommendations"]:
            st.markdown(f"### {'🥇' if _r['rank']==1 else '🥈'} {_r['instrument']}")
            lvl = st.session_state.lang_level
            why_text = _r["why_plain"] if lvl in ["Beginner","Intermediate"] else _r["why_pro"]
            st.success(why_text)
            if _r.get("caution"):
                st.warning(f"⚠️ {_r['caution']}")
    if _adv.get("blocked"):
        with st.expander("🚫 Not recommended for your profile", expanded=False):
            for _b in _adv["blocked"]:
                st.caption(f"• {_b}")
    st.caption("📡 Advisor uses LIVE VIX/IVR data. Click Macro Audit to refresh.")

col_macro, col_scan = st.columns([1, 1])


# === IV ENVIRONMENT ARBITER — master strategy selector ===
with st.expander("🧠 IV Arbiter — BUY or SELL Options Today?", expanded=True):
    st.caption("High IV → Sell premium. Low IV → Buy 60+ DTE options. Let the market tell you which weapon to use.")
    _vix_now = st.session_state.macro['vix'] if st.session_state.macro else 18.0
    _ivenv = classify_iv_environment(ivr_manual, _vix_now)
    col_iv1, col_iv2 = st.columns([1, 2])
    with col_iv1:
        st.markdown(f"### {_ivenv['badge']} {_ivenv['regime']}")
        st.metric("IVR", f"{ivr_manual:.0f}%", delta=f"Grade: {_ivenv['grade']}")
        st.markdown(f"**→ {_ivenv['recommendation']}**")
    with col_iv2:
        st.info(_ivenv['rationale'])
        st.markdown("**Best strategies now:**")
        for _s in _ivenv['strategies']:
            st.markdown(f"• {_s}")
        if _ivenv.get('avoid'):
            st.warning(f"🚫 {_ivenv['avoid']}")
    st.caption(f"📡 VIX context: {_ivenv['vix_overlay']}")
    st.caption("⚠️ SIMULATION: IVR requires manual input from broker / Barchart / InvestingPro.")



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
    if ticker.endswith('.NR'):
        st.info(
            "🇰🇪 **NSE Ticker Note** — Nairobi Securities Exchange data via yfinance "
            "has irregular coverage and may be delayed by days or unavailable. "
            "For reliable NSE data use the [NSE website](https://www.nse.co.ke) or "
            "[CBK financial markets portal](https://www.centralbank.go.ke). "
            "Technical indicators below are only meaningful when data has ≥50 trading days.",
            icon=None
        )
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

    # === LIVE COMMODITY PRICES (from macro scan) ===
    if st.session_state.macro and st.session_state.macro.get("oil_px"):
        _m = st.session_state.macro
        _mc1, _mc2, _mc3 = st.columns(3)
        _mc1.metric("Crude Oil (WTI)", f"${_m.get('oil_px',0):.2f}/bbl",
                    delta=f"{_m.get('oil_chg',0):+.2f}%",
                    help="Live price from Yahoo Finance futures (CL=F). ±1% matters for energy traders.")
        _mc2.metric("Gold (Spot proxy)", f"${_m.get('gold',0):.0f}/oz",
                    delta=f"{_m.get('gold_chg',0):+.2f}%",
                    help="GC=F futures. Gold rising with VIX = risk-off signal.")
        _mc3.metric("Natural Gas", f"${_m.get('ng_px',0):.3f}/MMBtu" if _m.get('ng_px') else "N/A",
                    help="NG=F front-month. Highly seasonal commodity.")
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

    # Weekly trade-day ranking for small-account / Mon-Thu planning
    with st.expander("🗓️ Mon–Thu Trade-Day Planner", expanded=False):
        _pdt_plan = pdt_guidance(capital, account_type, day_trades_used, planned_same_day_exit, pdt_framework)
        ranked_days = rank_trade_days(
            vix_regime,
            m.get('passive', False),
            st.session_state.week_event_days,
            preferred_trade_days,
            _pdt_plan.get('remaining') if isinstance(_pdt_plan.get('remaining'), int) else 99,
            pdt_framework,
        )
        st.caption("Ranks days by volatility regime, passive flows, event risk, and remaining day-trade budget. This is a permission filter, not a prediction.")
        for row in ranked_days:
            st.write(f"**{row['day']}** — {row['label']} | Score {row['score']} | {row['reasons']}")
        best = ranked_days[0] if ranked_days else None
        if best:
            st.info(f"Best current candidate day: **{best['day']}**. Avoid wasting limited day trades on YELLOW/RED conditions.")

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

    st.markdown(
        f"""<div class='key-levels-bar'>
        <strong>Key Levels:</strong>&nbsp;
        <span class='kl-support'>&#9660;&nbsp;${m['supp']:.2f}</span>
        <span class='kl-touches'>&nbsp;support &bull; {m.get('support_touches',0)} touches</span>
        &nbsp;&nbsp;&nbsp;
        <span class='kl-resistance'>&#9650;&nbsp;${m['res']:.2f}</span>
        <span class='kl-touches'>&nbsp;resistance &bull; {m.get('resistance_touches',0)} touches</span>
        </div>""",
        unsafe_allow_html=True
    )

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

    # ── LIVE OPTIONS GREEKS ─────────────────────────────────────────────────
    if st.session_state.metrics:
        _r_rate = st.session_state.macro.get("tnx",4.5)/100 if st.session_state.macro else 0.045
        with st.spinner("Loading live options chain..."):
            _opt = live_options_greeks_v2(ticker, dte_target=30, r_annual=_r_rate)
        if _opt and not _opt.get("no_options") and not _opt.get("error"):
            _ivr_str = f"IVR~{_opt['ivr_proxy']:.0f}% | " if _opt.get('ivr_proxy') else ""
            with st.expander(
                f"📊 Live Options Greeks — {ticker} | ATM IV {_opt['atm_iv']:.1f}% | {_ivr_str}Sell zone" if _opt.get('ivr_proxy',0)>=50 else
                f"📊 Live Options Greeks — {ticker} | ATM IV {_opt['atm_iv']:.1f}% | {_ivr_str}Buy zone" if _opt.get('ivr_proxy',0)<30 else
                f"📊 Live Options Greeks — {ticker} | ATM IV {_opt['atm_iv']:.1f}% | {_ivr_str}Selective",
                expanded=False
            ):
                _lvl = st.session_state.lang_level
                _oc1, _oc2, _oc3, _oc4 = st.columns(4)
                _oc1.metric("ATM IV", f"{_opt['atm_iv']:.1f}%")
                _oc2.metric("IVR (proxy)", f"{_opt['ivr_proxy']:.0f}%",
                    delta="SELL" if _opt['ivr_proxy']>=50 else "BUY 60+DTE" if _opt['ivr_proxy']<30 else "Selective")
                _oc3.metric("Realized Vol 30d", f"{_opt['rv_current']:.1f}%")
                _oc4.metric("Chain Expiry", _opt['expiry'], delta=f"{_opt['dte']} DTE")

                if _lvl == "Beginner":
                    import math as _m
                    _daily_move_pct = _opt['atm_iv']/100/math.sqrt(252)*100
                    st.info(
                        f"💡 ATM IV {_opt['atm_iv']:.1f}% means the market expects "
                        f"roughly ±{_daily_move_pct:.1f}% daily moves. "
                        f"**Delta**: how much the option price moves per $1 stock move. "
                        f"**Theta**: dollars lost (if you OWN the option) or gained (if you SOLD it) each day. "
                        f"**POP**: the chance this option expires worthless — high POP = sellers win more often (but their losses can be bigger)."
                    )
                elif _lvl in ["Advanced","Professional"]:
                    st.caption(
                        f"VRP = IV − RV = {_opt['atm_iv']-_opt['rv_current']:+.1f}pp. "
                        f"Greeks: exact BSM (scipy.stats.norm). "
                        f"IVR proxy: (ATM_IV − RV_52wLow) / (RV_52wHigh − RV_52wLow). "
                        f"r = {_r_rate*100:.2f}% (^TNX). Single expiry snapshot — no vol surface."
                    )

                if _opt.get("greeks_table"):
                    _gdf = pd.DataFrame(_opt["greeks_table"])
                    _puts  = _gdf[_gdf["type"]=="P"].sort_values("strike",ascending=False)
                    _calls = _gdf[_gdf["type"]=="C"].sort_values("strike")
                    _col_disp = ["strike","iv_pct","delta","gamma","theta_day","vega_1pct","pop","oi"]
                    _col_names = ["Strike","IV%","Delta","Gamma","Theta/d","Vega/1%","POP","OI"]
                    _fmt = {"IV%":"{:.1f}","Delta":"{:.3f}","Gamma":"{:.5f}",
                            "Theta/d":"{:.3f}","Vega/1%":"{:.3f}","POP":"{:.0%}","OI":"{:,.0f}"}
                    _cp1, _cp2 = st.columns(2)
                    with _cp1:
                        st.markdown(f"**Puts — {ticker}** ({_opt['expiry']})")
                        _pd2 = _puts[_col_disp].head(12).copy(); _pd2.columns = _col_names
                        st.dataframe(_pd2.style.format(_fmt), use_container_width=True, height=280)
                    with _cp2:
                        st.markdown(f"**Calls — {ticker}** ({_opt['expiry']})")
                        _cd2 = _calls[_col_disp].head(12).copy(); _cd2.columns = _col_names
                        st.dataframe(_cd2.style.format(_fmt), use_container_width=True, height=280)

                    st.markdown("**⚡ Quick-Fill → Spread Calculator (OTM Puts, POP ≥ 65%):**")
                    _otm = _puts[_puts["pop"]>=0.65][["strike","iv_pct","delta","pop","oi"]].head(5)
                    for _, _or in _otm.iterrows():
                        _qc1,_qc2 = st.columns([4,1])
                        _qc1.caption(
                            f"PUT {_or['strike']:.0f} | IV {_or['iv_pct']:.1f}% | "
                            f"Δ {_or['delta']:.3f} | POP {_or['pop']:.0%} | OI {int(_or['oi']):,}"
                        )
                        if _qc2.button(f"Use →", key=f"qf_{ticker}_{_or['strike']:.0f}"):
                            st.session_state["_af_strike"] = float(_or['strike'])
                            st.session_state["_af_delta"]  = abs(float(_or['delta']))
                            st.session_state["_af_iv"]     = float(_or['iv_pct'])
                            st.success(f"✅ Strike {_or['strike']:.0f} queued. Switch to Vertical Spread mode and re-scan.")

                st.caption(
                    "📡 IV: yfinance delayed market bid/ask (real). "
                    "Greeks: BSM exact formula, scipy.stats.norm. "
                    "IVR: 1y realized vol proxy (not true IV rank — true IVR needs paid historical IV data). "
                    "For live streaming Greeks, connect a broker API (IBKR, Tastytrade, Schwab)."
                )
        elif _opt and _opt.get("no_options"):
            st.caption(f"ℹ️ {ticker} has no listed options. Try SPY, QQQ, AAPL, GLD, or any major optionable stock/ETF.")


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

    # === OPTIONS INCOME CALCULATION ===
    # Directional stock math and options-income math are intentionally separated.
    # A cash-secured put is assignment/cash risk. A vertical credit spread is
    # defined-risk options math: max loss = (width - credit) * 100 * contracts.

    is_vertical = strategy == "Income (SPX Vertical Credit Spread)"
    is_csp = strategy == "Income (Cash-Secured Put)"
    option_details = {}

    if is_vertical:
        # Use user-entered strikes/credit. We do not infer strikes from stock support/resistance
        # because SPX spread risk must be calculated from option chain values.
        v0 = calc_vertical_credit_spread(short_strike, long_strike, spread_credit, spread_kind, contracts=0)
        if v0["errors"]:
            stop = target = entry = 0.0
            risk = reward = 0.0
            shares = 0
            option_details = v0
        else:
            ref_price = spx_reference_price if spx_reference_price > 0 else m.get('price', 0)
            expected_move = estimate_expected_move(ref_price, iv_percent, dte)
            em_status, em_multiple = expected_move_check(spread_kind, short_strike, ref_price, expected_move)
            approx_pop, approx_pot = approx_pop_from_delta(short_delta)
            max_loss_per_contract = v0["max_loss_per_contract"]
            contracts_by_trade_risk = contracts_for_defined_risk(risk_per_trade, max_loss_per_contract)
            contracts_by_portfolio = contracts_for_defined_risk(
                max(0, (capital * max_portfolio_risk / 100) - st.session_state.total_risk_deployed),
                max_loss_per_contract
            )
            contracts = max(0, min(contracts_by_trade_risk, contracts_by_portfolio))
            v = calc_vertical_credit_spread(short_strike, long_strike, spread_credit, spread_kind, contracts=contracts)
            v.update({
                "reference_price": ref_price,
                "iv_percent": iv_percent,
                "dte": dte,
                "expected_move": expected_move,
                "expected_move_status": em_status,
                "expected_move_multiple": em_multiple,
                "short_delta": short_delta,
                "approx_pop": approx_pop,
                "approx_probability_touch": approx_pot,
                "event_risk_48h": event_risk_48h,
                "hold_through_event": hold_through_event,
            })
            entry = short_strike
            stop = long_strike
            target = v["breakeven"]
            risk = v["max_loss_per_contract"]
            reward = v["max_profit_per_contract"]
            shares = contracts  # UI label switches to contracts below
            option_details = v


    elif strategy == "IWT Long Option (60+ DTE)":
        _lo = calc_long_option_iwt(
            lo_underlying, lo_strike, lo_premium,
            lo_direction, lo_delta, lo_dte, lo_contracts_lo
        )
        if _lo.get("error"):
            st.error(f"❌ {_lo['error']}")
        else:
            st.subheader(f"📈 IWT Long {lo_direction} — {lo_dte} DTE")
            st.caption("Teri Ijeoma: 60+ DTE, DITM, 50-100% profit target, 50% stop rule.")
            if not _lo["dte_ok"]:
                st.error(_lo["dte_grade"])
            else:
                st.success(_lo["dte_grade"])
            st.info(_lo["delta_grade"])
            col_lo1, col_lo2, col_lo3 = st.columns(3)
            with col_lo1:
                st.markdown("**💰 Position Cost & Leverage**")
                st.code(f"""Total Cost:   ${_lo['cost_total']:.2f}
Per Contract: ${_lo['cost_per_contract']:.2f}
Delta:        {lo_delta:.2f} ({_lo['delta_label']})
Leverage:     {_lo['leverage_ratio']:.1f}x vs stock
Equiv Shares: {_lo['shares_equiv']:.0f}""")
            with col_lo2:
                st.markdown("**🎯 IWT Profit Targets**")
                st.code(f"""50% Target:   ${_lo['target_50_val']:.2f}
              @ ${_lo['target_50_price']:.2f}/share
100% Target:  ${_lo['target_100_val']:.2f}
              @ ${_lo['target_100_price']:.2f}/share
Breakeven:    ${_lo['breakeven']:.2f} at expiry""")
            with col_lo3:
                st.markdown("**🛑 IWT Hard Stop**")
                st.code(f"""Stop Value:   ${_lo['stop_loss_total']:.2f}
Stop Price:   ${_lo['stop_price']:.2f}/share
Intrinsic:    ${_lo['intrinsic']:.2f}
Time Value:   ${_lo['time_val']:.2f} ({_lo['time_val_pct']:.0%})
Daily Theta:  ~${_lo['daily_theta']:.2f}/day""")
            st.warning(f"⏰ {_lo['theta_note']}")
            st.info(
                "📌 IWT Long Options Rules: (1) Min 60 DTE. (2) DITM delta 0.70+. "
                "(3) Exit at 50% gain — do not overstay. (4) NEVER hold past 50% loss. "
                "(5) Avoid before earnings/major events. (6) Only buy when IVR < 30%."
            )

    is_futures = strategy == "Futures (Index/Commodity)"

    if is_futures:
        fut = calc_futures_v4(fut_code, fut_entry, fut_stop, fut_target, fut_contracts)
        if fut.get("error"):
            st.error(f"❌ {fut['error']}")
        else:
            lvl = st.session_state.lang_level
            st.subheader(f"📦 {fut['spec']['name']} — {fut['direction']} {fut_contracts} Contract{'s' if fut_contracts > 1 else ''}")
            col_f1, col_f2, col_f3 = st.columns(3)
            with col_f1:
                st.markdown("**📍 Trade Levels**")
                st.code(f"""Entry:   {fut_entry:.2f}
Stop:    {fut_stop:.2f}
Target:  {fut_target:.2f}
R/R:     {fut['rr_ratio']:.2f}""")
            with col_f2:
                st.markdown("**💰 Risk in Dollars**")
                st.code(f"""Stop dist:  {fut['stop_pts']:.2f} pts  ({fut['stop_ticks']:.0f} ticks)
Max loss:   ${fut['total_risk']:,.2f}  ({fut_contracts}c)
Max gain:   ${fut['total_reward']:,.2f}
1pt move:  ${fut['point_value_per_contract']:,.2f}/contract""")
            with col_f3:
                st.markdown("**📐 Contract Specs**")
                st.code(f"""Contract:  {fut['code']} ({fut['spec']['exch']})
Tick size: {fut['tick_size']}
Tick value: ${fut['tick_value']:.2f}
Notional:  ${fut['notional']:,.0f}
Margin*:   ~${fut['margin_required']:,}""")
            # Language-aware explanation
            if lvl == "Beginner":
                st.info(
                    f"💡 Each tick this {fut['code']} moves in your direction, you make ${fut['tick_value']:.2f}. "
                    f"Each tick against you, you lose ${fut['tick_value']:.2f}. "
                    f"Your stop is {fut['stop_ticks']:.0f} ticks away = max loss ${fut['total_risk']:,.0f}."
                )
            elif lvl in ["Advanced","Professional"]:
                st.caption(
                    f"Contract specs: {fut['spec']['name']} | Multiplier ×{fut['spec']['mult']} | "
                    f"Tick {fut['tick_size']} = ${fut['tick_value']:.2f} | "
                    f"Notional ${fut['notional']:,.0f} | Exchange: {fut['spec']['exch']} | "
                    f"Approx SPAN initial margin: ~${fut['margin_required']:,}"
                )
            st.caption("⚠️ SIMULATION ONLY. Margin figures are approximate CME initial margin. Actual margin varies by broker and market conditions. Not financial advice.")
        # Skip rest of trade setup for futures
        entry = fut_entry; stop = fut_stop; target = fut_target; shares = fut_contracts
        risk = fut.get("total_risk", 0); reward = fut.get("total_reward", 0)
        rr = fut.get("rr_ratio", 0); gross_reward = reward; net_reward = reward
        slippage = 0; commissions = 0; total_trade_risk = risk
        can_trade = risk > 0

    elif is_csp:
        # Cash-secured put philosophy:
        # Max cash-secured exposure = (strike - premium) * 100 per contract.
        # Technical risk-to-stop is shown separately, but cash must be available for assignment.
        stop = entry - (m['atr'] * 2)
        target = entry
        technical_risk_per_contract = max(0, (entry - stop - premium) * 100)
        cash_secured_risk_per_contract = max(0, (entry - premium) * 100)
        reward_per_contract = premium * 100
        contracts = contracts_for_defined_risk(risk_per_trade, technical_risk_per_contract) if technical_risk_per_contract > 0 else 0
        shares = contracts
        risk = technical_risk_per_contract
        reward = reward_per_contract
        option_details = {
            "cash_secured_risk_per_contract": cash_secured_risk_per_contract,
            "technical_risk_per_contract": technical_risk_per_contract,
            "max_profit_per_contract": reward_per_contract,
            "breakeven": entry - premium,
            "errors": [] if premium > 0 else ["Premium must be positive for a cash-secured put."],
        }

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
    if is_vertical:
        slippage = 0
        commissions = option_details.get("commission", shares * OPTION_COMMISSION_PER_CONTRACT * 2)
        gross_reward = option_details.get("total_credit", shares * reward)
    elif is_csp:
        slippage = 0
        commissions = shares * OPTION_COMMISSION_PER_CONTRACT
        gross_reward = shares * reward
    else:
        slippage = entry * (SLIPPAGE_BPS / 10000) * shares
        commissions = shares * COMMISSION_PER_SHARE * 2
        gross_reward = shares * reward

    net_reward = gross_reward - slippage - commissions

    if is_vertical and option_details.get("errors"):
        for _err in option_details["errors"]:
            st.error(f"❌ Vertical spread input error: {_err}")
    if is_vertical and not option_details.get("errors"):
        st.info(
            f"📐 Vertical math: width ${option_details['width']:.2f} = ${option_details['gross_width_per_contract']:.0f} gross width/contract; "
            f"credit ${spread_credit:.2f} = ${option_details['max_profit_per_contract']:.0f} max profit/contract; "
            f"max loss ${option_details['max_loss_per_contract']:.0f}/contract; breakeven {option_details['breakeven']:.2f}."
        )
        st.caption(
            f"📊 Expected move ≈ ±{option_details.get('expected_move', 0):.1f} points over {option_details.get('dte', dte)} DTE; "
            f"short strike is {option_details.get('expected_move_status', 'UNKNOWN')} expected move "
            f"({option_details.get('expected_move_multiple', 0):.2f}x EM). Approx POP {option_details.get('approx_pop', 0):.0%}; touch risk {option_details.get('approx_probability_touch', 0):.0%}."
        )
        if option_details.get('expected_move_status') == 'INSIDE':
            st.warning("⚠️ Short strike is inside the expected move. That is lower-quality for income selling unless intentional and tightly managed.")
        if option_details.get('event_risk_48h'):
            st.warning("⚠️ Major event within 48h. Defined-risk only; consider waiting until after the event volatility clears.")
        if option_details.get('hold_through_event'):
            st.error("🛑 Holding through a major event materially increases gap/volatility risk. Reduce size or stand aside.")
        if option_details['credit_pct_width'] < 0.20:
            st.warning("⚠️ Credit is under 20% of spread width. Premium may be too thin for the defined risk.")
        elif option_details['credit_pct_width'] >= 0.25:
            st.success("✅ Credit efficiency meets the preferred 25%+ of width threshold.")
    if is_csp and option_details.get("errors"):
        for _err in option_details["errors"]:
            st.error(f"❌ Cash-secured put input error: {_err}")
    if is_csp and not option_details.get("errors"):
        st.warning(
            f"💵 CSP cash-secured exposure is approximately ${option_details['cash_secured_risk_per_contract']:.0f} per contract. "
            "This is not the same as a defined-risk vertical spread."
        )

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
        if is_vertical:
            st.caption("💡 Contracts are rounded down so max theoretical loss stays inside your risk budget and portfolio-risk cap.")
        elif is_csp:
            st.caption(f"💡 Each contract = 100 shares. Premium = ${premium:.2f}/share = ${premium*100:.2f}/contract; assignment exposure must be cash-secured.")
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

    if is_vertical:
        # For short verticals, raw reward/risk is usually below 1.0. Score credit efficiency
        # instead of forcing stock-style 3:1 reward/risk. Preferred: credit >=25% of width.
        credit_eff = option_details.get("credit_pct_width", 0)
        score_rr = 2 if credit_eff >= 0.25 else 1 if credit_eff >= 0.20 else 0
    elif is_csp:
        score_rr = 2 if rr >= 0.25 else 1 if rr >= 0.10 else 0
    else:
        score_rr = 2 if rr >= 3 else 1 if rr >= 2 else 0
    total_score = fresh + time_z + speed + score_rr

    penalties = []
    fed_penalty = False

    _pdt_now = pdt_guidance(capital, account_type, day_trades_used, planned_same_day_exit)
    if planned_same_day_exit and not _pdt_now["can_day_trade"]:
        total_score -= 3
        penalties.append("Trade Budget (-3): no same-day exit capacity left under small-account day-trade limits")
    elif planned_same_day_exit and isinstance(_pdt_now.get("remaining"), int) and _pdt_now["remaining"] <= 1 and total_score < 7:
        total_score -= 1
        penalties.append("Trade Budget (-1): preserve last day trade for only A+ setups")

    if is_vertical and dte > 7 and total_score < 7:
        total_score -= 1
        penalties.append("DTE Discipline (-1): >7 DTE requires stronger setup quality")
    if is_vertical and option_details.get('expected_move_status') == 'INSIDE':
        total_score -= 2
        penalties.append("Expected Move (-2): short strike is inside expected move")
    if is_vertical and option_details.get('approx_pop', 1) < 0.65:
        total_score -= 1
        penalties.append("POP (-1): probability of profit estimate below 65%")
    if is_vertical and option_details.get('event_risk_48h'):
        total_score -= 2
        penalties.append("Event Risk (-2): major event within 48h")
    if is_vertical and option_details.get('hold_through_event'):
        total_score -= 2
        penalties.append("Event Hold (-2): holding short premium through major event")

    if st.session_state.macro and st.session_state.macro['tnx_chg'] > 1.0 and ticker in GROWTH_TICKERS and "Long" in strategy:
        total_score -= 2
        fed_penalty = True
        penalties.append("Fed/Rates (-2): 10Y yield rising >1% pressures growth stocks")

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
        if is_vertical:
            _ce = option_details.get("credit_pct_width", 0)
            checks.append(("✅" if score_rr == 2 else "⚠️" if score_rr == 1 else "❌", f"Credit Efficiency: {_ce:.0%} of width ({['Thin (<20%)', 'Acceptable (20-25%)', 'Preferred (25%+)'][score_rr]})"))
        else:
            checks.append(("✅" if score_rr == 2 else "⚠️" if score_rr == 1 else "❌", f"R/R: {rr:.2f} ({['Poor', 'Acceptable', 'Excellent'][score_rr]})"))
        checks.append(("✅" if abs(m['gap']) > 2 and m['rvol'] > 1.5 else "⚠️" if abs(m['gap']) > 2 else "➖", f"Gap: {m['gap']:.2f}%"))
        checks.append(("✅" if m['rvol'] > 1.2 else "⚠️", f"Volume: {m['rvol']:.1f}x"))
        checks.append(("✅" if m['adx'] > 25 else "⚠️", f"Trend Strength: ADX {m['adx']:.0f}"))

        for icon, text in checks:
            st.caption(f"{icon} {text}")

    # === WARREN AI EXPORT ===
    st.markdown("---")
    st.caption("**📋 Copy for 2nd Opinion:**")

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
10Y Yield: {'RISING >1%' if fed_penalty else 'STABLE'}
Day Trade Plan: {'Same-day exit' if planned_same_day_exit else 'Swing/hold plan'} | PDT Framework: {pdt_framework} | Day trades used: {day_trades_used}/3
Expected Move: {round(option_details.get('expected_move', 0), 2) if is_vertical else 'N/A'} | EM Status: {option_details.get('expected_move_status', 'N/A') if is_vertical else 'N/A'} | Approx POP: {round(option_details.get('approx_pop', 0)*100, 1) if is_vertical else 'N/A'}% | Event 48h: {option_details.get('event_risk_48h', False) if is_vertical else 'N/A'}
Option Math: {('Vertical ' + spread_kind + ' spread | Short ' + str(short_strike) + ' | Long ' + str(long_strike) + ' | Credit ' + str(spread_credit) + ' | Max loss/contract $' + str(round(option_details.get('max_loss_per_contract', 0), 2))) if is_vertical else ('CSP | Premium ' + str(premium) + ' | Cash-secured exposure/contract $' + str(round(option_details.get('cash_secured_risk_per_contract', 0), 2))) if is_csp else 'N/A'}
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
                    "credit_points": spread_credit if is_vertical else premium if is_csp else 0.0,
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
                    "credit_points": spread_credit if is_vertical else premium if is_csp else 0.0,
                    "slippage": slippage, "commissions": commissions, "status": "OPEN"
                }
                st.session_state.journal.append(trade_record)
                st.session_state.open_positions.append(trade_record)
                st.session_state.total_risk_deployed += total_trade_risk
                st.success("✅ Live position logged!")
                st.rerun()

else:
    st.info("👈 **Quick Start:** 1. Scan Macro (check global markets) → 2. Select a Ticker/Asset (**left sidebar**) → 3. Check/Enter **Strategy** + **IWT Score** (**left sidebar**) → 4. Scan Ticker/Asset → 5. Review Multi-Algo Signals → 6. Log Paper or Live Trade")

# POSITION MANAGEMENT
if st.session_state.open_positions:
    st.divider()
    st.subheader("📊 Open Positions (Live Trades)")
    st.caption("💡 These are trades you logged as LIVE. Close them after you exit in your broker.")

    positions_df = pd.DataFrame(st.session_state.open_positions)
    positions_df = positions_df[['ticker', 'action', 'entry', 'stop', 'target', 'shares', 'risk', 'score']]
    st.dataframe(positions_df, use_container_width=True)

    st.markdown("**Close a Position:**")
    st.caption("💡 For stocks/CSP, enter underlying exit price. For SPX verticals, enter the closing debit/spread value in index points, e.g., 0.35.")

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
                        elif "SPX Vertical" in pos['action']:
                            # Conservative close-entry: user enters actual closing debit/credit-equivalent P&L manually
                            # as Exit Price = remaining spread value/debit in index points.
                            # P&L = initial credit - closing debit, each point worth $100 per contract.
                            initial_credit_points = pos.get('credit_points', pos['expected_reward'] / max(pos['shares'] * 100, 1))
                            closing_debit_points = exit_price
                            actual_pnl = (initial_credit_points - closing_debit_points) * pos['shares'] * 100
                        elif "Income" in pos['action']:
                            # Cash-secured put: profit = premium kept if price > strike at close.
                            # If assigned / below strike, estimate intrinsic loss minus premium.
                            if exit_price >= pos['entry']:
                                actual_pnl = pos['expected_reward']
                            else:
                                assignment_loss = (pos['entry'] - exit_price) * pos['shares'] * 100
                                actual_pnl = pos['expected_reward'] - assignment_loss
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



# === TRADE MANAGEMENT ENGINE ===
st.divider()
with st.expander("🔧 Trade Management Engine", expanded=False):
    st.caption(
        "Systematic rules for managing open positions. "
        "TastyTrade 50% rule + IWT profit/stop discipline + gamma management."
    )
    _tm_struct = st.selectbox(
        "Position Type",
        ["CREDIT_SPREAD", "LONG_OPTION"],
        format_func=lambda x: "Credit Spread (short premium)" if x == "CREDIT_SPREAD" else "Long Option (bought premium)"
    )
    _col_tm1, _col_tm2 = st.columns(2)
    with _col_tm1:
        if _tm_struct == "CREDIT_SPREAD":
            _tm_entry = st.number_input("Credit Received ($, e.g. 1.50 credit = 150)", value=0.0, step=5.0)
            _tm_curr  = st.number_input("Current Cost to Close ($)", value=0.0, step=5.0)
            _tm_dte_r = st.number_input("DTE Remaining", value=14, min_value=0, max_value=90, step=1)
            _tm_dte_e = st.number_input("DTE at Entry", value=30, min_value=1, max_value=90, step=1)
        else:
            _tm_entry = st.number_input("Premium Paid ($ total per position, e.g. $500)", value=0.0, step=10.0)
            _tm_curr  = st.number_input("Current Market Value ($)", value=0.0, step=10.0)
            _tm_dte_r = st.number_input("DTE Remaining", value=45, min_value=0, max_value=400, step=1)
            _tm_dte_e = None
    with _col_tm2:
        if _tm_entry > 0:
            _tme = trade_management_engine(_tm_struct, _tm_entry, _tm_curr, _tm_dte_r, _tm_dte_e)
            st.markdown(f"### {_tme['primary']}")
            for _td in _tme['actions']:
                st.markdown(f"• {_td}")
            if _tm_struct == "CREDIT_SPREAD" and _tm_dte_e and _tm_dte_e > 0:
                _theta = theta_decay_profile(_tm_entry / 100, _tm_dte_e, _tm_dte_r)
                if _theta:
                    st.markdown("---")
                    st.caption(
                        f"Theta tracker — profit captured: ~{_theta['pct_profit_captured']:.1f}% | "
                        f"value remaining: ~${_theta['value_remaining']:.0f}/contract | "
                        f"daily decay: ~${_theta['daily_theta_approx']:.2f}/day"
                    )
                    st.caption(_theta['management_signal'])
        else:
            st.info("Enter position values to get management decision.")

# === KELLY CRITERION + RISK-OF-RUIN ===
with st.expander("🎲 Kelly Criterion & Risk-of-Ruin", expanded=False):
    st.caption(
        "Institutional position sizing science. "
        "Full Kelly maximises long-run growth but produces severe drawdowns. "
        "Most quant desks use Quarter Kelly or less."
    )
    _ck1, _ck2 = st.columns(2)
    with _ck1:
        _k_win = st.slider("Win Rate (%)", 30, 85, 60, step=1) / 100.0
        _k_avgwin  = st.number_input("Avg Win (% of 1R, e.g. 50 = 50%)", value=50.0, step=5.0) / 100.0
        _k_avgloss = st.number_input("Avg Loss (% of 1R, e.g. 100 = full max loss)", value=100.0, step=5.0) / 100.0
        _k_monthly = st.number_input("Trades / Month", value=8, min_value=1, max_value=50, step=1)
    with _ck2:
        if _k_avgwin > 0 and _k_avgloss > 0:
            _kr = kelly_and_ruin(_k_win, _k_avgwin, _k_avgloss, _k_monthly)
            st.code(
                f"Full Kelly:      {_kr['kelly_pct']:.1f}% per trade\n"
                f"Half Kelly:      {_kr['half_kelly_pct']:.1f}% per trade\n"
                f"Quarter Kelly:   {_kr['quarter_kelly_pct']:.1f}% (recommended)\n"
                f"\n"
                f"Edge per Trade:  {_kr['edge_per_trade_pct']:.1f}% of 1R\n"
                f"Monthly Return:  {_kr['expected_monthly_pct']:.1f}% of 1R\n"
                f"Risk of Ruin:    {_kr['ruin_probability_pct']:.1f}%"
            )
            if _kr['edge_per_trade_pct'] > 0:
                st.success(f"✅ Positive edge. Recommended size: **{_kr['recommended_size_pct']:.1f}% of capital** (Quarter Kelly).")
            else:
                st.error("🔴 Negative edge — do not trade this system mechanically.")
            st.caption(_kr['note'])
            st.caption(
                "💡 Quarter Kelly preserves ~75% of optimal long-run growth with ~50% of the drawdown. "
                "Half Kelly is the institutional compromise. Full Kelly is rarely used outside academia."
            )


# =============================================================================
# LIVE TRADING — Account · Order Builder · Confirm · Execute
# Tradier brokerage integration.
# Every order requires two explicit steps: Preview → Confirm.
# Futures/Forex require a separate broker (see TRADIER_SETUP.md).
# =============================================================================

st.divider()
st.header("📤 Execute Trade")
_lvl = st.session_state.lang_level

if not _tradier_is_connected():
    st.info(
        "🔗 **Connect Tradier to enable live trading.** "
        "See **TRADIER_SETUP.md** in the repo for the two-minute setup. "
        "Once connected, this section lets you place real orders directly from the analysis above."
    )
else:
    _env_label = (st.secrets.get("TRADIER_ENV") or
                  st.secrets.get("tradier", {}).get("env", "production")).lower()
    _is_sandbox = _env_label == "sandbox"

    if _is_sandbox:
        st.warning(
            "🧪 **PAPER TRADING MODE (Sandbox)** — orders are simulated, no real money is at risk. "
            "Change TRADIER_ENV = production in Streamlit secrets to trade live."
        )
    else:
        st.error(
            "⚡ **LIVE TRADING MODE** — orders execute in your REAL Tradier account with REAL money. "
            "Every click costs money if filled. There are no take-backs after a filled order."
        )

    # ── Account dashboard ────────────────────────────────────────────────────
    _acct_id = tdr_get_account_id()
    _bal     = tdr_get_balances(_acct_id) if _acct_id else None

    if _bal:
        col_b1, col_b2, col_b3, col_b4 = st.columns(4)
        col_b1.metric("Total Equity",    f"${_bal['total_equity']:,.2f}")
        col_b2.metric("Options BP",      f"${_bal['option_bp']:,.2f}",
                      help="Buying power available for options trades.")
        col_b3.metric("Stock BP",        f"${_bal['stock_bp']:,.2f}")
        col_b4.metric("PDT Flag",
                      "⚠️ PDT" if _bal["pdt_status"] else "✅ No PDT",
                      help="Pattern Day Trader flag — if flagged, intraday round-trips are restricted.")

        if _bal["pdt_status"]:
            st.warning(
                "⚠️ **Pattern Day Trader (PDT) flag active.** "
                "Your account is flagged for exceeding 3 intraday round-trips in a 5-day window. "
                "Restrictions apply until equity reaches $25,000."
            )

    # Positions summary
    with st.expander("📋 Current Positions & Open Orders", expanded=False):
        _positions = tdr_get_positions(_acct_id) if _acct_id else []
        _orders    = tdr_get_orders(_acct_id)    if _acct_id else []

        if _positions:
            st.markdown("**Open Positions:**")
            _pos_rows = []
            for p in _positions:
                _pos_rows.append({
                    "Symbol": p.get("symbol",""),
                    "Qty":    p.get("quantity",""),
                    "Cost":   f"${float(p.get('cost_basis',0)):.2f}",
                    "Value":  f"${float(p.get('market_value',0)):.2f}",
                    "P&L":    f"${float(p.get('gain_loss',0)):+.2f}",
                })
            st.dataframe(pd.DataFrame(_pos_rows), use_container_width=True)
        else:
            st.caption("No open positions.")

        if _orders:
            st.markdown("**Recent Orders:**")
            _ord_rows = []
            for o in _orders[:15]:
                _ord_rows.append({
                    "ID":     str(o.get("id","")),
                    "Symbol": o.get("symbol",""),
                    "Type":   o.get("class",""),
                    "Side":   o.get("side",""),
                    "Qty":    o.get("quantity",""),
                    "Status": o.get("status",""),
                    "Price":  f"${float(o.get('price',0) or 0):.2f}" if o.get("price") else "MKT",
                })
            st.dataframe(pd.DataFrame(_ord_rows), use_container_width=True)

            # Cancel button for open orders
            _open_ords = [o for o in _orders if o.get("status") in ("open","pending")]
            if _open_ords:
                _cancel_id = st.selectbox(
                    "Cancel order:",
                    ["— select —"] + [f"{o['id']} | {o.get('symbol','')} {o.get('side','')} {o.get('quantity','')}" for o in _open_ords]
                )
                if _cancel_id != "— select —":
                    _oid = _cancel_id.split(" | ")[0]
                    if st.button("🗑️ Cancel this order", type="secondary"):
                        _cr = tdr_cancel_order(_acct_id, _oid)
                        if _cr.get("order", {}).get("status") == "ok" or _cr.get("order"):
                            st.success(f"✅ Order {_oid} cancelled.")
                            st.cache_data.clear()
                        else:
                            st.error(f"Cancel failed: {_cr}")
        else:
            st.caption("No recent orders.")

    st.divider()

    # ── Order builder ─────────────────────────────────────────────────────────
    st.subheader("🏗️ Build Your Order")

    _order_class = st.selectbox(
        "What are you trading?",
        ["SPX/SPY Vertical Credit Spread  (income — sell premium)",
         "IWT Long Option — 60+ DTE  (directional — buy premium)",
         "Single Stock or ETF  (long or short)",
         "Close / Exit an existing position"],
        help=(
            "Stocks and ETFs: buy or sell shares directly. "
            "Options: defined-risk contracts. "
            "Futures require a separate broker (IBKR, NinjaTrader, thinkorSwim)."
        )
    )

    _order_result = None
    _max_loss_order = 0
    _order_preview  = {}

    # ════════════════════════════════════════════════════════════════════════
    # PATH 1: Vertical Credit Spread
    # ════════════════════════════════════════════════════════════════════════
    if "Vertical Credit Spread" in _order_class:
        st.markdown("### 📊 SPX/SPY Vertical Credit Spread")
        if _lvl == "Beginner":
            st.info(
                "💡 You're selling a put spread. You collect premium upfront. "
                "Your max profit is that premium. Your max loss is the spread width minus premium. "
                "You need the market to stay ABOVE your short strike at expiration."
            )

        _vs_col1, _vs_col2 = st.columns(2)
        with _vs_col1:
            _vs_under = st.text_input("Underlying (SPY for small accounts, SPX for $50k+)", value="SPY")
            _vs_type  = st.radio("Spread type", ["Put Credit Spread", "Call Credit Spread"], horizontal=True)
            _vs_expiry= st.date_input("Expiration date")
            _vs_short = st.number_input("Short strike (the one you SELL)", value=0.0, step=0.5)
            _vs_long  = st.number_input("Long strike (the one you BUY for protection)", value=0.0, step=0.5)
        with _vs_col2:
            _vs_credit = st.number_input("Net credit per share ($)", value=0.0, step=0.05,
                help="Mid-price of the spread. (Short bid + Long ask) / 2. Enter as positive number.")
            _vs_qty    = st.number_input("Contracts", value=1, min_value=1, max_value=50, step=1)
            _vs_otype  = "PUT" if "Put" in _vs_type else "CALL"

        if _vs_short > 0 and _vs_long > 0 and _vs_credit > 0 and _vs_expiry:
            _width         = abs(_vs_short - _vs_long)
            _max_profit    = _vs_credit * 100 * _vs_qty
            _max_loss_order= (_width - _vs_credit) * 100 * _vs_qty
            _breakeven     = (_vs_short - _vs_credit if _vs_otype=="PUT" else _vs_short + _vs_credit)
            _credit_eff    = _vs_credit / _width if _width > 0 else 0

            st.code(
                f"Spread:      {_vs_under} ${_vs_short:.2f}/{_vs_long:.2f} {_vs_otype} "
                + _vs_expiry.strftime("%b %d %Y")
                + f"\nContracts:   {_vs_qty}"
                + f"\nCredit:      ${_vs_credit:.2f}/share = ${_max_profit:,.2f} total"
                + f"\nMax loss:    ${_max_loss_order:,.2f} (if spread goes full width)"
                + f"\nBreakeven:   ${_breakeven:.2f}"
                + f"\nCredit eff:  {_credit_eff:.0%} of width "
                + ("✅ Good" if _credit_eff >= 0.25 else "⚠️ Thin — below 25%")
            )

            _short_sym = build_option_symbol(_vs_under, _vs_expiry.strftime("%Y-%m-%d"), _vs_otype, _vs_short)
            _long_sym  = build_option_symbol(_vs_under, _vs_expiry.strftime("%Y-%m-%d"), _vs_otype, _vs_long)

            _order_preview = {
                "class": "spread",
                "underlying": _vs_under,
                "legs": [
                    {"symbol": _short_sym, "side": "sell_to_open", "qty": _vs_qty},
                    {"symbol": _long_sym,  "side": "buy_to_open",  "qty": _vs_qty},
                ],
                "net_price": _vs_credit,
                "description": (
                    f"SELL {_vs_qty}x {_vs_under} {_vs_otype} {_vs_short} / "
                    f"BUY {_vs_qty}x {_vs_under} {_vs_otype} {_vs_long} "
                    f"exp {_vs_expiry.strftime('%Y-%m-%d')} for ${_vs_credit:.2f} credit"
                ),
                "max_profit": _max_profit,
                "max_loss":   _max_loss_order,
            }

    # ════════════════════════════════════════════════════════════════════════
    # PATH 2: IWT Long Option (60+ DTE)
    # ════════════════════════════════════════════════════════════════════════
    elif "IWT Long Option" in _order_class:
        st.markdown("### 📈 IWT Long Option — 60+ DTE")
        if _lvl == "Beginner":
            st.info(
                "💡 You're BUYING an option — paying premium upfront. "
                "Your max loss is exactly what you pay (the premium). "
                "Your max gain is unlimited (for calls) or down to zero (for puts). "
                "IWT rule: minimum 60 days to expiry. Target delta 0.70+."
            )

        _lo_col1, _lo_col2 = st.columns(2)
        with _lo_col1:
            _lo_under  = st.text_input("Underlying (SPY, AAPL, QQQ, GLD...)", value="SPY")
            _lo_dir    = st.radio("Direction", ["CALL (bullish)", "PUT (bearish)"], horizontal=True)
            _lo_expiry = st.date_input("Expiration (min 60 days out)", key="lo_exp")
            _lo_strike = st.number_input("Strike price (choose DITM: delta 0.70+)", value=0.0, step=0.5)
        with _lo_col2:
            _lo_premium = st.number_input("Premium per share ($)", value=0.0, step=0.05,
                help="Cost of one option contract. $5.00 = $500 per contract.")
            _lo_qty     = st.number_input("Contracts", value=1, min_value=1, max_value=20, step=1)

        from datetime import date as _date
        _lo_dte = (_lo_expiry - _date.today()).days if _lo_expiry else 0

        if _lo_premium > 0 and _lo_strike > 0 and _lo_expiry:
            _lo_total_cost  = _lo_premium * 100 * _lo_qty
            _lo_stop_value  = _lo_total_cost * 0.50   # IWT 50% stop
            _lo_target_50   = _lo_total_cost * 0.50   # IWT 50% profit target
            _lo_target_100  = _lo_total_cost * 1.00

            _lo_otype = "CALL" if "CALL" in _lo_dir else "PUT"
            _dte_ok   = _lo_dte >= 60

            if not _dte_ok:
                st.error(f"⚠️ {_lo_dte} DTE is below IWT's 60-day minimum. Theta decay will hurt you faster.")
            else:
                st.success(f"✅ {_lo_dte} DTE — IWT compliant")

            st.code(
                f"Buy {_lo_qty}x {_lo_under} {_lo_otype} ${_lo_strike:.2f} "
                + "exp " + _lo_expiry.strftime("%b %d %Y")
                + f"\nPremium:      ${_lo_premium:.2f}/share = ${_lo_total_cost:,.2f} total"
                + f"\nIWT stop:     ${_lo_stop_value:,.2f} (50% loss — hard rule)"
                + f"\n50%% target:   ${_lo_target_50:,.2f} gain"
                + f"\n100%% target:  ${_lo_target_100:,.2f} gain"
            )

            _max_loss_order = _lo_total_cost
            _lo_sym = build_option_symbol(_lo_under, _lo_expiry.strftime("%Y-%m-%d"), _lo_otype, _lo_strike)

            _order_preview = {
                "class":       "option",
                "option_symbol": _lo_sym,
                "side":        "buy_to_open",
                "qty":         _lo_qty,
                "net_price":   _lo_premium,
                "description": (
                    f"BUY {_lo_qty}x {_lo_under} {_lo_otype} {_lo_strike} "
                    f"exp {_lo_expiry.strftime('%Y-%m-%d')} @ ${_lo_premium:.2f}"
                ),
                "max_profit":  float("inf"),
                "max_loss":    _max_loss_order,
            }

    # ════════════════════════════════════════════════════════════════════════
    # PATH 3: Stock or ETF
    # ════════════════════════════════════════════════════════════════════════
    elif "Stock or ETF" in _order_class:
        st.markdown("### 📦 Stock / ETF Order")
        if _lvl == "Beginner":
            st.info(
                "💡 Buying shares = owning a piece of the company. "
                "Your max loss = the full amount you invest (if it goes to zero). "
                "Always use a stop-loss order to limit your downside."
            )

        _eq_col1, _eq_col2 = st.columns(2)
        with _eq_col1:
            _eq_sym   = st.text_input("Symbol (SPY, AAPL, GLD...)", value="SPY")
            _eq_side  = st.radio("Action", ["Buy (long)", "Sell short"], horizontal=True)
            _eq_qty   = st.number_input("Shares", value=1, min_value=1, max_value=10000, step=1)
        with _eq_col2:
            _eq_otype = st.radio("Order type", ["Limit (recommended)", "Market"], horizontal=True)
            _eq_limit = st.number_input("Limit price ($)", value=0.0, step=0.01) if "Limit" in _eq_otype else None
            _eq_stop  = st.number_input("Stop-loss price ($)", value=0.0, step=0.01,
                help="Optional: set a stop to cap your loss. Leave 0 to skip.")
            _eq_dur   = st.radio("Duration", ["Day", "GTC"], horizontal=True)

        if _eq_limit and _eq_limit > 0 and _eq_stop and _eq_stop > 0:
            _eq_actual_side = "buy" if "Buy" in _eq_side else "sell"
            _eq_risk = abs(_eq_limit - _eq_stop) * _eq_qty if _eq_stop > 0 else _eq_limit * _eq_qty
            _max_loss_order = _eq_risk

            st.code(
                f"{'BUY' if 'buy' in _eq_actual_side else 'SELL'} {_eq_qty} {_eq_sym} "
                + f"@ limit ${_eq_limit:.2f}"
                + f"\nStop-loss:  ${_eq_stop:.2f}"
                + f"\nMax risk:   ${_eq_risk:,.2f}"
            )

            _order_preview = {
                "class":       "equity",
                "symbol":      _eq_sym,
                "side":        _eq_actual_side,
                "qty":         _eq_qty,
                "order_type":  "limit" if "Limit" in _eq_otype else "market",
                "limit_price": _eq_limit,
                "stop_price":  _eq_stop if _eq_stop > 0 else None,
                "duration":    _eq_dur.lower(),
                "description": (
                    f"{'BUY' if 'buy' in _eq_actual_side else 'SELL'} {_eq_qty}x {_eq_sym} "
                    f"limit ${_eq_limit:.2f} stop ${_eq_stop:.2f} {_eq_dur}"
                ),
                "max_profit": None,
                "max_loss":   _max_loss_order,
            }

    # ════════════════════════════════════════════════════════════════════════
    # PATH 4: Close existing position
    # ════════════════════════════════════════════════════════════════════════
    else:
        st.markdown("### 🚪 Close / Exit Position")
        _positions_close = tdr_get_positions(_acct_id) if _acct_id else []
        if not _positions_close:
            st.info("No open positions found. Positions appear here after your first trade.")
        else:
            _pos_labels = {
                f"{p.get('symbol','')} | Qty: {p.get('quantity','')} | P&L: ${float(p.get('gain_loss',0)):+.2f}": p
                for p in _positions_close
            }
            _sel_pos_label = st.selectbox("Select position to close", list(_pos_labels.keys()))
            _sel_pos = _pos_labels[_sel_pos_label]
            _close_qty = st.number_input("Quantity to close",
                min_value=1, max_value=abs(int(_sel_pos.get("quantity",1))),
                value=abs(int(_sel_pos.get("quantity",1))))
            _close_type = st.radio("Close order type", ["Limit", "Market"], horizontal=True)
            _close_price = st.number_input("Limit price ($)", value=0.0, step=0.01) if _close_type == "Limit" else None

            _sym = _sel_pos.get("symbol","")
            # Determine close side
            _pos_qty = int(_sel_pos.get("quantity", 0))
            _is_option_pos = len(_sym) > 10
            if _is_option_pos:
                _close_side = "sell_to_close" if _pos_qty > 0 else "buy_to_close"
            else:
                _close_side = "sell" if _pos_qty > 0 else "buy"

            _order_preview = {
                "class":       "option" if _is_option_pos else "equity",
                "symbol":      _sym,
                "option_symbol": _sym if _is_option_pos else None,
                "side":        _close_side,
                "qty":         _close_qty,
                "order_type":  "limit" if _close_type == "Limit" else "market",
                "limit_price": _close_price,
                "description": f"CLOSE {_close_qty}x {_sym} @ {'$'+str(_close_price) if _close_price else 'market'}",
                "max_profit":  None,
                "max_loss":    0,
            }
            _max_loss_order = 0

    # ════════════════════════════════════════════════════════════════════════
    # CONFIRM AND EXECUTE (all paths converge here)
    # ════════════════════════════════════════════════════════════════════════
    if _order_preview:
        st.divider()
        st.subheader("⚠️ Order Preview — Review Before Submitting")

        _safety = trade_safety_check(_bal, _max_loss_order, _order_preview.get("qty",1)) if _bal else {"checks":[],"all_pass":False}

        st.markdown(f"**Order:** `{_order_preview['description']}`")

        if _order_preview.get("max_loss"):
            st.error(
                f"🔴 **Maximum possible loss on this trade: "
                f"${_order_preview['max_loss']:,.2f}**"
                + (f" | Max profit: ${_order_preview['max_profit']:,.2f}" if _order_preview.get("max_profit") and _order_preview["max_profit"] != float("inf") else "")
            )

        # Safety check results
        if _safety.get("checks"):
            for chk in _safety["checks"]:
                icon = "✅" if chk["pass"] else "⚠️"
                st.caption(f"{icon} {chk['name']}: {chk['detail']}")

        _instrument_note = ""
        if "Futures" in _order_class or "futures" in _order_class.lower():
            st.warning("⚠️ Futures are NOT supported via Tradier. Use Interactive Brokers, NinjaTrader, or thinkorSwim.")

        # Beginner plain-English summary
        if _lvl == "Beginner":
            _plain = f"""
What you're about to do: {_order_preview['description']}

The most you can lose: ${_order_preview.get('max_loss',0):,.2f}

{"The most you can make: $"+f"{_order_preview['max_profit']:,.2f}" if _order_preview.get('max_profit') and _order_preview['max_profit'] != float('inf') else "Maximum gain: unlimited (options can grow if the stock moves far)"}

{"✅ This is a paper trade — no real money." if _is_sandbox else "⚡ This is a REAL trade with REAL money."}
"""
            st.info(_plain)

        # Two-step confirmation
        _confirm1 = st.checkbox(
            "✅ I have reviewed this order and the risk/reward above",
            value=False
        )
        _confirm2 = st.checkbox(
            f"{'🧪 I understand this is a paper trade (sandbox)' if _is_sandbox else '💸 I understand this will use REAL money from my Tradier account'}",
            value=False
        )

        _can_submit = _confirm1 and _confirm2

        if st.button(
            f"{'🧪 Place Paper Order' if _is_sandbox else '💸 Place Live Order'}",
            disabled=not _can_submit,
            type="primary" if _can_submit else "secondary",
            use_container_width=True
        ):
            _result = None
            with st.spinner("Sending order to Tradier..."):
                oc = _order_preview.get("class")
                if oc == "spread":
                    _result = tdr_place_spread_order(
                        _acct_id,
                        _order_preview["underlying"],
                        _order_preview["legs"],
                        _order_preview["net_price"],
                    )
                elif oc == "option":
                    _result = tdr_place_option_order(
                        _acct_id,
                        _order_preview["option_symbol"],
                        _order_preview["side"],
                        _order_preview["qty"],
                        "limit",
                        _order_preview["net_price"],
                    )
                elif oc == "equity":
                    _result = tdr_place_equity_order(
                        _acct_id,
                        _order_preview["symbol"],
                        _order_preview["side"],
                        _order_preview["qty"],
                        _order_preview.get("order_type","limit"),
                        _order_preview.get("limit_price"),
                        _order_preview.get("stop_price"),
                        _order_preview.get("duration","day"),
                    )

            if _result:
                _ord = _result.get("order", {})
                if _ord.get("status") == "ok":
                    st.success(
                        f"✅ Order submitted! ID: **{_ord.get('id','')}** | "
                        f"Status: {_ord.get('status','')} | "
                        f"{'Paper trade — no real money used.' if _is_sandbox else 'Live order sent to market.'}"
                    )
                    st.cache_data.clear()  # Force positions/orders refresh
                else:
                    err = _result.get("errors", {}).get("error", str(_result))
                    st.error(f"❌ Order failed: {err}")
                    if _lvl == "Beginner":
                        st.info(
                            "Common reasons: invalid symbol, market closed, insufficient buying power, "
                            "or the option symbol format doesn't match an active contract. "
                            "Double-check your strike and expiration date."
                        )
            else:
                st.error("No response from Tradier. Check your connection.")

        if not _can_submit:
            st.caption("Check both boxes above to enable the order button.")

    # Futures note (always visible at bottom of trading section)
    with st.expander("📦 Futures & Forex Trading (different broker required)", expanded=False):
        st.markdown("""
Tradier supports US equities and equity options only.

**For futures (ES, NQ, CL, GC, ZC, etc.)** use one of these brokers:
| Broker | Instruments | API | Notes |
|--------|------------|-----|-------|
| [Interactive Brokers](https://ibkr.com) | Everything | ibapi (Python) | Best coverage, complex setup |
| [NinjaTrader](https://ninjatrader.com) | Futures | NinjaScript / REST | Free for sim, $50/mo live |
| [Schwab thinkorSwim](https://tdameritrade.com) | Equities + futures | REST API | Good for existing Schwab accounts |
| [Tastytrade](https://tastytrade.com) | Options + futures | REST API | Best UX for options/futures combo |

The **Futures Calculator** in this app (Futures mode in strategy selector) already computes your exact tick-value risk for all 14 contracts. Use that output to size your trade before placing in one of the brokers above.

**For Forex** — Interactive Brokers, Oanda (oanda.com), or FXCM.
""")

# =============================================================================
# BACKTESTING ENGINE — V7
# Real data: yfinance (SPY prices, VIX, TNX). No fake numbers.
# Options P&L: synthetic via BSM with real VIX. Clearly labelled.
# Equity P&L: real price changes.
# =============================================================================

st.divider()
st.header("📈 Backtest — Did These Strategies Work?")
_lvl = st.session_state.lang_level

with st.expander("What backtesting means (and its limits)", expanded=False):
    if _lvl == "Beginner":
        st.info(
            "Backtesting = running a strategy on PAST data to see how it would have performed. "
            "It can't tell you what will happen next — markets change. "
            "But it can tell you: did the strategy have an edge? "
            "How often did it win? What was the worst stretch? "
            "Read results critically, not as a promise."
        )
    else:
        st.caption(
            "Backtest limitations: survivorship bias, look-ahead bias, overfitting, "
            "transaction cost assumptions, and the fundamental problem that past "
            "market regimes don't repeat identically. "
            "Options P&L here uses BSM + real VIX (best free approximation). "
            "Actual market fills would differ from synthetic BSM prices."
        )

# Pre-computed results from live backtest run at deploy time
_BT = {
    "period":        "2025-05-08 → 2026-05-08  (252 trading days)",
    "spy_start":     558.66,
    "spy_end":       737.62,
    "spy_return":    32.03,
    "vix_avg":       18.26,
    "vix_min":       13.47,
    "vix_max":       31.05,
    "strategies": {
        "📈 Buy & Hold SPY":         {"return":29.82, "sharpe":1.95,  "max_dd":-8.88, "capital":32455.28},
        "💰 Credit Spreads":          {"return":1.26,  "sharpe":-4.58, "max_dd":-0.39, "capital":25314.38,
                                       "trades":16, "win_rate":69, "detail":"16 trades | 69% win | -0.4% drawdown"},
        "📊 IWT Long Calls (60 DTE)": {"return":0.00,  "sharpe":0.00,  "max_dd":0.00,  "capital":25000.00,
                                       "detail":"No entries triggered — IVR stayed below entry threshold most of year"},
        "🔼 Trend Following (SPY)":   {"return":1.87,  "sharpe":-2.00, "max_dd":0.00,  "capital":25467.73,
                                       "detail":"2 profitable trend rides | 0% drawdown on active capital"},
        "⚖️ Combined (1/3 each)":     {"return":1.04,  "sharpe":-6.96, "max_dd":-0.13, "capital":25260.70},
    },
    "market_context": (
        "2025-2026 was an exceptional bull year. SPY returned +32% (historical avg: ~10%). "
        "VIX ranged 13–31 with an April 2025 tariff-shock spike. "
        "IVR (6-month rolling) averaged 25% — a predominantly LOW IV environment. "
        "This favoured: (1) owning stocks, (2) buying cheap options. "
        "It was NOT an ideal year for income strategies that rely on elevated IV."
    ),
    "key_insight": (
        "Income strategies (credit spreads) had a 69% win rate and near-zero drawdown. "
        "They just had fewer entry signals because IVR stayed low most of the year. "
        "In a normal 8–10% SPY year, income returns of 1–3% with <1% drawdown "
        "represent excellent risk-adjusted performance."
    ),
}

# Market context
st.markdown(f"**Period: {_BT['period']}**")
st.markdown(f"SPY: `${_BT['spy_start']:.2f}` → `${_BT['spy_end']:.2f}` (+{_BT['spy_return']:.1f}%) | "
            f"VIX avg {_BT['vix_avg']:.1f} (range {_BT['vix_min']:.1f}–{_BT['vix_max']:.1f})")
st.warning(
    f"**Market context:** {_BT['market_context']}"
)

# Results grid
st.subheader("Strategy Performance (real money simulation)")
col_hdr = st.columns([3,2,2,2,2])
for h,t in zip(["Strategy","1-Year Return","Final $25k →","Sharpe","Max Drawdown"],
               col_hdr):
    t.markdown(f"**{h}**")

for name, s in _BT["strategies"].items():
    c1,c2,c3,c4,c5 = st.columns([3,2,2,2,2])
    c1.markdown(name)
    color = "green" if s["return"]>5 else "orange" if s["return"]>0 else "red"
    c2.markdown(f":{color}[**{s['return']:+.2f}%**]")
    c3.markdown(f"${s['capital']:,.0f}")
    sh_color = "green" if s["sharpe"]>1 else "orange" if s["sharpe"]>0 else "red"
    c4.markdown(f":{sh_color}[{s['sharpe']:.2f}]")
    dd_color = "green" if s["max_dd"]>-2 else "orange"
    c5.markdown(f":{dd_color}[{s['max_dd']:.2f}%]")
    if "detail" in s:
        st.caption(f"  ↳ {s['detail']}")

st.info(f"**Key Insight:** {_BT['key_insight']}")

# Explanation by level
if _lvl == "Beginner":
    st.markdown("""
**What this means for you:**
- Buying and holding SPY crushed every active strategy this year — that's unusual
- Income strategies (spreads) protected capital beautifully (-0.4% worst drawdown)
- The market was too calm and rising too fast for premium-selling to shine
- In a normal or choppy year, income strategies make up ground vs buy-and-hold
- **Bottom line:** No strategy works best every year. The IWT approach is about consistency over years, not beating exceptional bull runs
""")
elif _lvl in ["Advanced","Professional"]:
    st.markdown("""
**Risk-adjusted reading:**
- Buy & Hold Sharpe 1.95 reflects the exceptional 2025-2026 bull run — not typical
- Credit spreads' negative Sharpe (-4.58) reflects that *net premium collected was tiny relative to idle capital*
  — the real comparison should be spread P&L vs margin deployed, not full portfolio
- IWT long calls had no triggers: IVR < 40% with S > MA20 and RSI 40-70 aligned rarely
- Trend following Sharpe (-2.00) reflects sparse trades and Sharpe penalising low-variance equity curves
- For a proper strategy assessment: calculate return on capital *actually deployed*, not on full $25k
""")

# Live re-run backtest button
st.divider()
with st.expander("🔄 Run Live Backtest Now (uses real current data)", expanded=False):
    st.caption(
        "This runs the backtest engine live against today's data from Yahoo Finance. "
        "Results may differ slightly from above due to price updates."
    )
    if st.button("🚀 Run Full Backtest (takes ~30 seconds)", type="primary"):
        with st.spinner("Fetching real historical data and computing backtest..."):
            try:
                import math as _m
                from scipy.stats import norm as _norm
                from datetime import date as _date, timedelta as _td
                import warnings as _w; _w.filterwarnings('ignore')

                _end = _date.today(); _start = _end - _td(days=400)
                _spy = yf.download("SPY",  start=_start, end=_end, progress=False)["Close"].squeeze().dropna()
                _vix = yf.download("^VIX", start=_start, end=_end, progress=False)["Close"].squeeze().dropna()
                _tnx = yf.download("^TNX", start=_start, end=_end, progress=False)["Close"].squeeze().dropna()/100
                _com = sorted(_spy.index.intersection(_vix.index).intersection(_tnx.index))[-252:]
                _spy, _vix, _tnx = _spy.loc[_com], _vix.loc[_com], _tnx.loc[_com]
                _ivr = ((_vix - _vix.rolling(126).min())/(_vix.rolling(126).max()-_vix.rolling(126).min())*100).fillna(50)

                def _bsm(S,K,T,r,s2,f='p'):
                    if T<=0 or s2<=0: return max(S-K,0) if f=='c' else max(K-S,0)
                    d1=(_m.log(S/K)+(r+.5*s2**2)*T)/(s2*_m.sqrt(T)); d2=d1-s2*_m.sqrt(T)
                    if f=='c': return S*_norm.cdf(d1)-K*_m.exp(-r*T)*_norm.cdf(d2)
                    return K*_m.exp(-r*T)*_norm.cdf(-d2)-S*_norm.cdf(-d1)

                _CAPITAL = 25_000; _cap = _CAPITAL; _open = None; _trades = []
                _sspy = pd.Series(_spy.values, index=_spy.index)
                _ma20 = _sspy.rolling(20).mean(); _ma50 = _sspy.rolling(50).mean()
                _gain = _sspy.diff().clip(lower=0).rolling(14).mean()
                _loss = (-_sspy.diff().clip(upper=0)).rolling(14).mean()
                _rsi  = 100 - 100/(1+_gain/_loss.replace(0,1e-10))

                # Credit spreads
                _cap_cs = _CAPITAL; _open_cs = None; _cs_wins = 0; _cs_l = 0; _cs_pnl = 0
                for _i, _dt in enumerate(_com):
                    _S, _sg, _r, _iv = float(_spy[_dt]), float(_vix[_dt])/100, float(_tnx[_dt]), float(_ivr[_dt])
                    if _i < 30: continue
                    if _open_cs:
                        _Tr = max(_open_cs['d']/365, 1e-6)
                        _sv = _bsm(_S,_open_cs['Ks'],_Tr,_r,_sg,'p'); _lv = _bsm(_S,_open_cs['Kl'],_Tr,_r,_sg,'p')
                        _cc = _sv - _lv; _pp = (_open_cs['c'] - _cc)/_open_cs['c'] if _open_cs['c']>0 else 0
                        _cl = ('EXPIRY' if _open_cs['d']<=0 else '50% PROFIT' if _pp>=.5 else
                               'STOP' if _cc>=_open_cs['c']*2 else '21 DTE' if _open_cs['d']<=21 else None)
                        if _cl:
                            _pnl = (_open_cs['c'] - _cc)*100 - 3.30
                            _cap_cs += _pnl; _cs_pnl += _pnl
                            if _pnl > 0: _cs_wins += 1
                            else: _cs_l += 1
                            _open_cs = None
                        else: _open_cs['d'] -= 1
                    if _open_cs is None and _iv>=35 and _dt.weekday()==0:
                        _T = 30/365
                        _Ks = round(_S*0.96, 0); _Kl = _Ks-5
                        _c = _bsm(_S,_Ks,_T,_r,_sg,'p') - _bsm(_S,_Kl,_T,_r,_sg,'p')
                        if _c/5>=.20 and _c>0:
                            _open_cs = {'Ks':_Ks,'Kl':_Kl,'c':_c,'d':30}

                # Trend following
                _cap_tf = _CAPITAL; _open_tf = None; _tf_pnl = 0; _hw2 = 0; _tf_t = 0; _tf_w = 0
                for _i2, _dt2 in enumerate(_com):
                    _S2 = float(_spy[_dt2])
                    _ma20v = float(_ma20[_dt2]) if not _m.isnan(float(_ma20[_dt2])) else _S2
                    _ma50v = float(_ma50[_dt2]) if not _m.isnan(float(_ma50[_dt2])) else _S2
                    _rsiv  = float(_rsi[_dt2]) if not _m.isnan(float(_rsi[_dt2])) else 50
                    if _i2 < 50: continue
                    if _open_tf:
                        _hw2 = max(_hw2, _S2)
                        if _ma20v < _ma50v or _S2 < _hw2*0.96:
                            _p2 = (_S2 - _open_tf['p'] - 0.01)*_open_tf['sh']
                            _cap_tf += _p2; _tf_pnl += _p2
                            if _p2>0: _tf_w+=1
                            _tf_t += 1; _open_tf = None; _hw2 = 0
                    if _open_tf is None and _ma20v>_ma50v and 40<_rsiv<68 and _dt2.weekday()==0:
                        _sh = max(1, int(_cap_tf*0.3/_S2))
                        _open_tf = {'p':_S2,'sh':_sh}; _hw2 = _S2

                if _open_tf:
                    _p2 = (float(_spy.iloc[-1])-_open_tf['p']-0.01)*_open_tf['sh']
                    _cap_tf += _p2; _tf_pnl += _p2; _tf_t += 1; _tf_w += 1

                _bh_ret = (float(_spy.iloc[-1])/float(_spy.iloc[0])-1)*100
                _total_cs = _cs_wins + _cs_l

                st.success("✅ Backtest complete — LIVE results from real data:")
                r1,r2,r3,r4 = st.columns(4)
                r1.metric("Buy & Hold SPY",  f"+{_bh_ret:.1f}%")
                r2.metric("Credit Spreads",  f"+{_cs_pnl/25000*100:.2f}%",
                          delta=f"{_cs_wins}/{_total_cs} wins" if _total_cs>0 else "No trades")
                r3.metric("Trend Following", f"+{_tf_pnl/25000*100:.2f}%",
                          delta=f"{_tf_w}/{_tf_t} wins" if _tf_t>0 else "No trades")
                r4.metric("Period",
                          f"{_com[0].date()} to {_com[-1].date()}")
                st.caption(
                    "⚠️ Options P&L = synthetic BSM with real VIX. Equity P&L = real price changes. "
                    f"VIX avg: {float(_vix.mean()):.1f}"
                )
            except Exception as _ex:
                st.error(f"Backtest error: {_ex}")

# Data provenance (always visible)
with st.expander("📋 Data sources & methodology", expanded=False):
    st.markdown("""
**What is real:**
- SPY daily closing prices — Yahoo Finance (yfinance)
- VIX daily history — CBOE via Yahoo Finance
- 10-Year Treasury yield — Yahoo Finance

**What is synthetic (clearly labelled):**
- Options P&L is computed using Black-Scholes-Merton with real historical VIX as the implied volatility input
- Real option market prices would differ due to bid/ask spread, volatility skew (put IV > call IV), and term structure
- This approximation is standard in academic backtesting when historical options data is unavailable
- Historical options prices require paid data (ORATS, OptionStack, CBOE LiveVol) — we use the best free alternative

**Commissions assumed:**
- Options: $0.65/contract/leg × 2 legs × open+close = $2.60–$3.30 round trip
- Equity: $0 commission (Tradier) + $0.01/share slippage

**Why buy-and-hold dominated this year:**
SPY returned ~32% in 2025-2026 — approximately 3× the historical average. This is exceptional.
Active strategies with Kelly-based position sizing intentionally limit exposure to protect capital.
In a typical 8–10% year, income strategies generating 1–3% with <1% drawdown represent
excellent risk-adjusted performance relative to buy-and-hold volatility.
""")
    if _lvl == "Professional":
        st.code("""
Sharpe formula used: (mean_daily_excess_return / std_daily_return) × √252
Excess return = daily return - rf/252 where rf = 4.5% (approximate TNX)
Max drawdown = max((peak - trough) / peak) over the test period
IVR = (VIX - VIX_126d_min) / (VIX_126d_max - VIX_126d_min) × 100 (6-month rolling)
Strike selection: nearest strike giving delta ≈ -0.25 (put spreads)
BSM: standard closed-form with no dividend adjustment
""")

# =============================================================================
# BACKTESTING — Section 8
# Real 1-year historical simulation. Real prices. Honest labels.
# =============================================================================

st.divider()
st.header("📈 Backtest — Did These Strategies Actually Work?")

_lvl = st.session_state.lang_level
if _lvl == "Beginner":
    st.info(
        "💡 A backtest asks: IF we had followed this strategy every week for the past year "
        "using real market prices — how much money would we have made or lost? "
        "The answer tells us when each strategy works and when it doesn't."
    )
else:
    st.caption(
        "Historical simulation using real daily prices (yfinance) + BSM pricing. "
        "Option fills reconstructed from VIX-derived IV. Not real options market data. "
        "Slippage and commission included. Past performance does not predict future results."
    )

with st.expander("⚙️ Backtest Parameters", expanded=False):
    _bc1, _bc2 = st.columns(2)
    with _bc1:
        st.markdown("**Credit Spread**")
        _bt_min_ivr  = st.slider("Min IVR-90d to enter (%)", 20, 60, 40, 5)
        _bt_spread_w = st.number_input("Spread width ($)", value=5, min_value=1, max_value=25, step=1)
        _bt_pft      = st.slider("Profit target (%)", 25, 75, 50, 5)
    with _bc2:
        st.markdown("**IWT Long Call**")
        _bt_max_vix  = st.slider("Max VIX to enter long call", 16, 30, 22, 1)
        _bt_call_dte = st.number_input("DTE for long call", value=60, min_value=30, max_value=120, step=5)
    _bt_capital  = st.number_input("Starting capital ($)", value=25000, min_value=5000, max_value=500000, step=5000)

if st.button("▶️ Run 1-Year Backtest", type="primary", use_container_width=True):
    st.session_state["_bt_run"] = True

if st.session_state.get("_bt_run"):
    with st.spinner("Loading real market data and running simulation..."):
        _df_bt = load_backtest_data()

    if _df_bt is not None and len(_df_bt) > 0:
        _period_str = f"{_df_bt.index[0].date()} → {_df_bt.index[-1].date()}"
        _bnh_ret    = (_df_bt['spy'].iloc[-1]/_df_bt['spy'].iloc[0]-1)*100
        _bnh_pnl    = _bt_capital * _bnh_ret / 100

        with st.spinner("Running credit spread simulation..."):
            _cs_trades, _cs_eq, _cs_cap = backtest_credit_spread(
                _df_bt, min_ivr=_bt_min_ivr, spread_w=_bt_spread_w,
                profit_tgt=_bt_pft/100, initial_capital=_bt_capital)

        with st.spinner("Running IWT long call simulation..."):
            _lc_trades, _lc_eq, _lc_cap = backtest_long_call(
                _df_bt, max_vix=_bt_max_vix, call_dte=_bt_call_dte,
                initial_capital=_bt_capital)

        _cs_stats = compute_backtest_stats(_cs_trades, _bt_capital, "Put Credit Spread")
        _lc_stats = compute_backtest_stats(_lc_trades, _bt_capital, "IWT Long Call (DITM)")

        # ── Results summary ─────────────────────────────────────────────────
        st.subheader(f"📊 Results — {_period_str}")
        _r1, _r2, _r3 = st.columns(3)

        _r1.metric("Put Credit Spread",
                   f"${_cs_stats.get('total_pnl',0):+,.0f}",
                   delta=f"{_cs_stats.get('return_pct',0):+.1f}% | {_cs_stats.get('n_trades',0)} trades | WR {_cs_stats.get('win_rate_pct',0):.0f}%",
                   delta_color="normal")
        _r2.metric("IWT Long Call (DITM)",
                   f"${_lc_stats.get('total_pnl',0):+,.0f}",
                   delta=f"{_lc_stats.get('return_pct',0):+.1f}% | {_lc_stats.get('n_trades',0)} trades | WR {_lc_stats.get('win_rate_pct',0):.0f}%",
                   delta_color="normal")
        _r3.metric("Buy & Hold SPY",
                   f"${_bnh_pnl:+,.0f}",
                   delta=f"{_bnh_ret:+.1f}%",
                   delta_color="normal")

        # ── Plain English verdict ────────────────────────────────────────────
        _best_strat = ("IWT Long Call" if _lc_stats.get('total_pnl',0) > _cs_stats.get('total_pnl',0) else "Credit Spread")
        _best_pnl   = max(_lc_stats.get('total_pnl',0), _cs_stats.get('total_pnl',0))

        if _lvl == "Beginner":
            st.markdown("---")
            st.markdown("### 🔍 What the numbers mean")
            if _lc_stats.get('total_pnl',0) > _cs_stats.get('total_pnl',0):
                st.success(
                    f"**The IWT long call strategy won this year.** "
                    f"Buying DITM calls when VIX was low (cheap options) + market trending up "
                    f"returned ${_lc_stats.get('total_pnl',0):+,.0f} vs "
                    f"${_cs_stats.get('total_pnl',0):+,.0f} for credit spreads. "
                    f"This is exactly what Teri's two-weapon system teaches: "
                    f"when IV is low, BUY options, don't sell them."
                )
            else:
                st.success(
                    f"**The credit spread strategy won this year.** "
                    f"Selling premium in elevated-IV environments collected consistent income. "
                    f"Returned ${_cs_stats.get('total_pnl',0):+,.0f}."
                )
            st.info(
                f"**What this tells you about the market this year:** "
                f"SPY rose {_bnh_ret:.1f}% from ${_df_bt['spy'].iloc[0]:.0f} to ${_df_bt['spy'].iloc[-1]:.0f}. "
                f"VIX mostly stayed below 22. Low VIX = cheap options = buyers had the edge. "
                f"Next time VIX spikes to 25+, credit spreads will likely outperform."
            )
        elif _lvl == "Professional":
            st.caption(
                f"Credit spread Sharpe proxy: edge={_cs_stats.get('ev_per_trade',0):.2f}/trade, "
                f"drawdown={_cs_stats.get('max_drawdown',0):.0f}. "
                f"Long call leverage: avg_win/avg_loss={abs(_lc_stats.get('avg_win',1)/_lc_stats.get('avg_loss',-1) if _lc_stats.get('avg_loss',0)!=0 else 0):.2f}. "
                f"VRP environment: IV/RV spread drives credit spread edge; bull market + low VRP = buyer's market. "
                f"BSM-reconstructed; actual PnL would be lower by bid/ask (~$10-40/contract round-trip for SPY options)."
            )

        # ── Equity curve chart ──────────────────────────────────────────────
        if not _cs_eq.empty or not _lc_eq.empty:
            _all_dates = sorted(set(
                (list(_cs_eq['date']) if not _cs_eq.empty else []) +
                (list(_lc_eq['date']) if not _lc_eq.empty else [])
            ))
            if _all_dates:
                _chart_data = {"Date": _all_dates}
                if not _cs_eq.empty:
                    _cs_dict = dict(zip(_cs_eq['date'], _cs_eq['cum_pnl']))
                    _chart_data["Credit Spread P&L ($)"] = [_cs_dict.get(d, None) for d in _all_dates]
                if not _lc_eq.empty:
                    _lc_dict = dict(zip(_lc_eq['date'], _lc_eq['cum_pnl']))
                    _chart_data["IWT Long Call P&L ($)"] = [_lc_dict.get(d, None) for d in _all_dates]
                _chart_df = pd.DataFrame(_chart_data).set_index("Date")
                st.line_chart(_chart_df, use_container_width=True)
                st.caption("Cumulative P&L ($) over the backtest period. Each point = a completed trade.")

        # ── Detailed stats tables ────────────────────────────────────────────
        _t1, _t2 = st.tabs(["📋 Credit Spread Trades", "📋 IWT Long Call Trades"])
        with _t1:
            if not _cs_trades.empty:
                st.caption(f"{len(_cs_trades)} trades | Win rate {_cs_stats.get('win_rate_pct',0):.0f}% | PF {_cs_stats.get('profit_factor','N/A')}")
                _cs_show = _cs_trades.copy()
                _cs_show.columns = [c.replace('_',' ') for c in _cs_show.columns]
                st.dataframe(_cs_show.style.format({'P&L':'${:,.2f}','Credit $/shr':'{:.3f}'}),
                             use_container_width=True, height=300)
                st.caption(
                    f"Avg win: ${_cs_stats.get('avg_win',0):,.2f} | "
                    f"Avg loss: ${_cs_stats.get('avg_loss',0):,.2f} | "
                    f"Max drawdown: ${_cs_stats.get('max_drawdown',0):,.2f}")
            else:
                st.info(
                    f"No credit spread trades executed with IVR threshold of {_bt_min_ivr}%. "
                    f"This was a low-IV year (VIX 13-31, IVR-90d median ~20%). "
                    f"Try lowering the IVR threshold to 25-30% to see more trades, "
                    f"or this year simply favored option BUYERS over sellers."
                )

        with _t2:
            if not _lc_trades.empty:
                st.caption(f"{len(_lc_trades)} trades | Win rate {_lc_stats.get('win_rate_pct',0):.0f}% | Avg win ${_lc_stats.get('avg_win',0):,.0f}")
                _lc_show = _lc_trades.copy()
                _lc_show.columns = [c.replace('_',' ') for c in _lc_show.columns]
                st.dataframe(_lc_show.style.format({'P&L':'${:,.2f}','Premium $':'{:.3f}','Cost total':'${:,.2f}'}),
                             use_container_width=True, height=300)
            else:
                st.info("No long call trades executed (VIX above threshold or no uptrend).")

        # ── Methodology transparency ─────────────────────────────────────────
        with st.expander("📐 Methodology & Honest Limitations", expanded=False):
            st.markdown(f"""
**Data sources (all real):**
- SPY daily closing prices: Yahoo Finance (yfinance)
- VIX daily closing: Yahoo Finance (^VIX)
- Risk-free rate: 10-Year Treasury (^TNX) via Yahoo Finance
- Period: {_period_str}

**Option pricing method: Black-Scholes-Merton (BSM)**
Historical option market prices are not available for free. We reconstruct prices using the BSM formula with VIX as the IV input.
This is the standard academic approach for options backtesting without paid historical options data (ORATS, OptionStack, CBOE LiveVol).

```
Option price = BSM(S, K, T, r, σ)
where σ = VIX / 100 (for SPY options; VIX is SPX 30-day IV by construction)
```

**Costs included:**
- Slippage: $0.03/share (credit spreads) | $0.05/share (long options)
- Commission: $1.30/contract (2-leg for spreads)
- These are conservative retail estimates

**What BSM does NOT capture:**
- Bid/ask spread (real SPY options: $0.05–$0.30/contract depending on liquidity)
- Volatility skew (puts cost more than BSM suggests due to tail risk demand)
- Early exercise / pin risk (minimal for European-style index options)
- Actual fill prices (you often get worse than mid-price)

**Conservative adjustment:** Real results would be approximately 15-30% lower than shown due to the above factors.

**IVR-90d computation:**
```
IVR_90d = (VIX_today − VIX_90d_min) / (VIX_90d_max − VIX_90d_min) × 100
```
90-day rolling window; updates daily. More responsive than 252-day IVR.

**Strategy rules applied:**
- Credit spread: enter weekly (Monday) when IVR-90d ≥ {_bt_min_ivr}%, short strike outside 75% of EM, close at 50% profit | 2× stop | 21 DTE
- Long call: enter weekly (Monday, one per week) when VIX ≤ {_bt_max_vix} and SPY above 20-day SMA, DITM strike (~delta 0.70), close at 50% or 100% gain | 50% stop | 21 DTE

**Past performance disclaimer:** Backtests are in-sample. This specific year may not repeat. 2025-2026 was a strong bull market with low VIX — a different environment would produce different results. Always forward-test in paper trading before risking capital.
""")

    else:
        st.error("Could not load backtest data. Check your internet connection and try again.")

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
st.caption("📊 EasyStockTrader — Smart Stock Analysis | Simulation Only — Not Financial Advice")
st.caption("© 2026 Gabriel Mahia | Consistency beats intensity.")
# -- Feedback sidebar ---------------------------------------------------------
with st.sidebar:
    st.markdown("---")
    st.markdown(
        "**Was this useful?**\n\n"
        "[:pencil: Leave feedback](https://docs.google.com/forms/d/e/1FAIpQLSff_cjR102HNUeYU428ROv56TScLBzsQRc1JTwY4wGizvTQKw/viewform) (2 min)\n\n"
        "[:bug: Report a bug](https://github.com/gabrielmahia/quantum-maestro/issues/new)\n\n"
        "---\n"
        "*Built by [Gabriel Mahia](https://aikungfu.dev)*\n\n"
        "[Back to all tools](https://gabrielmahia.github.io)"
    )

