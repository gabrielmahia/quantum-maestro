"""
Position Sizing
===============
Two engines, one cap:

1. Fixed-risk (Teri Ijeoma): risk is constant, size varies with stop distance.
2. Fractional Kelly (Thorp): edge-proportional, capped at quarter-Kelly.

Final size = min(fixed-risk size, Kelly size) x regime multiplier.
Kelly can only shrink a position relative to the 1% rule, never grow it.
"""

from .config import LIMITS


def fixed_risk_shares(account_equity: float, risk_pct: float, entry: float, stop: float) -> dict:
    risk_dollars = account_equity * risk_pct
    stop_dist = abs(entry - stop)
    if stop_dist <= 0:
        return {"error": "Stop must differ from entry."}
    shares = int(risk_dollars // stop_dist)
    return {
        "risk_dollars": round(risk_dollars, 2),
        "stop_distance": round(stop_dist, 2),
        "shares": shares,
        "notional": round(shares * entry, 2),
        "note": "Risk fixed, size variable (Ijeoma). One R = the risk dollars.",
    }


def fixed_risk_contracts(account_equity: float, risk_pct: float, max_loss_per_contract: float) -> dict:
    risk_dollars = account_equity * risk_pct
    if max_loss_per_contract <= 0:
        return {"error": "Max loss per contract must be positive (defined-risk only)."}
    contracts = int(risk_dollars // max_loss_per_contract)
    return {
        "risk_dollars": round(risk_dollars, 2),
        "contracts": contracts,
        "max_loss_total": round(contracts * max_loss_per_contract, 2),
        "note": "Defined-risk options: max loss per contract IS the stop.",
    }


def kelly_fraction(win_rate: float, avg_win_r: float, avg_loss_r: float = 1.0) -> dict:
    """f* = W - (1-W)/(avg_win/avg_loss). Returns capped fractional Kelly."""
    if avg_loss_r <= 0 or not (0 < win_rate < 1):
        return {"error": "Need 0<W<1 and positive avg loss."}
    b = avg_win_r / avg_loss_r
    f_star = win_rate - (1 - win_rate) / b
    capped = max(0.0, min(f_star * LIMITS.KELLY_FRACTION_CAP / 0.25 * 0.25, f_star)) if f_star > 0 else 0.0
    # quarter-Kelly:
    quarter = max(0.0, f_star * 0.25)
    return {
        "full_kelly": round(f_star, 4),
        "quarter_kelly": round(quarter, 4),
        "edge_exists": f_star > 0,
        "note": ("Negative Kelly => you have NO edge at these stats; correct size is zero. "
                 if f_star <= 0 else
                 "Quarter-Kelly guards against estimation error and fat tails (Mandelbrot). "
                 "Your stats come from the journal, not from feelings."),
    }


def final_size(account_equity: float, regime: str, entry: float = 0.0, stop: float = 0.0,
               max_loss_per_contract: float = 0.0, win_rate: float = None,
               avg_win_r: float = None) -> dict:
    mult = LIMITS.REGIME_MULTIPLIER.get(regime, 0.0)
    risk_pct = LIMITS.MAX_RISK_PCT_PER_TRADE * mult
    out = {"regime": regime, "regime_multiplier": mult, "effective_risk_pct": risk_pct}

    if max_loss_per_contract > 0:
        out["fixed_risk"] = fixed_risk_contracts(account_equity, risk_pct, max_loss_per_contract)
    elif entry and stop:
        out["fixed_risk"] = fixed_risk_shares(account_equity, risk_pct, entry, stop)

    if win_rate is not None and avg_win_r is not None:
        k = kelly_fraction(win_rate, avg_win_r)
        out["kelly"] = k
        if k.get("edge_exists") is False:
            out["kelly_override"] = "Kelly says zero. If journal stats show no edge, the sizer's answer is: don't trade."
    return out
