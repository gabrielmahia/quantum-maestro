"""
Portfolio State Engine (Offensive / Neutral / Defensive / Lockdown)
===================================================================
Inspired by Paul Tudor Jones (defense first), Howard Marks (where are we
in the cycle?), and Dalio (regimes, not predictions).

Deterministic scoring over observable inputs. Auto-fills from market data
when available; every input can be overridden manually so the engine works
offline and its logic is fully inspectable (no black box).

Score design: each factor contributes -2..+2. Total maps to a regime.
"""

from dataclasses import dataclass, asdict


@dataclass
class RegimeInputs:
    vix_level: float = 18.0
    vix_5d_change_pct: float = 0.0      # +12 means VIX up 12% in 5 sessions
    spx_vs_50dma_pct: float = 0.0       # +2.0 means SPX 2% above its 50DMA
    breadth_rsp_spy_20d_pct: float = 0.0  # equal-weight vs cap-weight 20d relative change
    oil_5d_change_pct: float = 0.0
    credit_stress: int = 0              # 0 = calm, 1 = spreads widening, 2 = stress
    major_event_within_5d: bool = False # FOMC / CPI / NFP / heavy earnings cluster
    geopolitical_flag: bool = False     # active shooting-war escalation affecting oil/shipping
    account_in_drawdown: bool = False   # recent >=1R loss or below equity high-water mark by >5%


def score_regime(x: RegimeInputs) -> dict:
    factors = {}

    # Volatility level
    if x.vix_level < 15:        factors["VIX level"] = +2
    elif x.vix_level < 20:      factors["VIX level"] = +1
    elif x.vix_level < 26:      factors["VIX level"] = -1
    else:                        factors["VIX level"] = -2

    # Volatility momentum (a rising VIX from a low base is the dangerous pattern)
    if x.vix_5d_change_pct > 20:    factors["VIX momentum"] = -2
    elif x.vix_5d_change_pct > 8:   factors["VIX momentum"] = -1
    elif x.vix_5d_change_pct < -8:  factors["VIX momentum"] = +1
    else:                            factors["VIX momentum"] = 0

    # Trend
    if x.spx_vs_50dma_pct > 1.5:    factors["Trend (SPX vs 50DMA)"] = +2
    elif x.spx_vs_50dma_pct > 0:    factors["Trend (SPX vs 50DMA)"] = +1
    elif x.spx_vs_50dma_pct > -2:   factors["Trend (SPX vs 50DMA)"] = -1
    else:                            factors["Trend (SPX vs 50DMA)"] = -2

    # Breadth (equal-weight underperforming = narrow, fragile market)
    if x.breadth_rsp_spy_20d_pct > 1:    factors["Breadth (RSP/SPY)"] = +1
    elif x.breadth_rsp_spy_20d_pct < -2: factors["Breadth (RSP/SPY)"] = -2
    elif x.breadth_rsp_spy_20d_pct < -1: factors["Breadth (RSP/SPY)"] = -1
    else:                                 factors["Breadth (RSP/SPY)"] = 0

    # Oil shock (inflation-path risk)
    if x.oil_5d_change_pct > 8:     factors["Oil shock"] = -2
    elif x.oil_5d_change_pct > 4:   factors["Oil shock"] = -1
    else:                            factors["Oil shock"] = 0

    factors["Credit"] = {0: 0, 1: -1, 2: -2}[int(x.credit_stress)]
    factors["Event calendar"] = -1 if x.major_event_within_5d else 0
    factors["Geopolitics"] = -2 if x.geopolitical_flag else 0
    factors["Account state"] = -1 if x.account_in_drawdown else 0

    total = sum(factors.values())

    if total >= 4:      regime = "OFFENSIVE"
    elif total >= 0:    regime = "NEUTRAL"
    elif total >= -5:   regime = "DEFENSIVE"
    else:               regime = "LOCKDOWN"

    guidance = {
        "OFFENSIVE": "Full playbook available. Normal sizing (1.0x). Press only exceptional setups (PTJ).",
        "NEUTRAL": "Selective. 0.6x sizing. A-grade setups only; more cash. Cash is a position.",
        "DEFENSIVE": "0.3x sizing. Defined-risk only, wider stops or no trades. Hedges permitted. "
                     "Priority: capital preservation and journaling, not P&L.",
        "LOCKDOWN": "No new risk. Manage/close existing positions. Study, journal, wait. "
                    "Kostolany: the money is made in the sitting.",
    }

    return {
        "regime": regime,
        "score": total,
        "factors": factors,
        "guidance": guidance[regime],
        "inputs": asdict(x),
    }


def try_autofill_inputs() -> "RegimeInputs | None":
    """Best-effort autofill from yfinance. Returns None if data unavailable
    (offline, rate-limited) — the UI then falls back to manual inputs."""
    try:
        import yfinance as yf
        import pandas as pd

        def hist(t, period="4mo"):
            df = yf.Ticker(t).history(period=period)
            return df["Close"].dropna()

        vix = hist("^VIX", "1mo")
        spx = hist("^GSPC", "4mo")
        rsp = hist("RSP", "2mo")
        spy = hist("SPY", "2mo")
        oil = hist("CL=F", "1mo")

        if len(vix) < 6 or len(spx) < 55:
            return None

        vix_level = float(vix.iloc[-1])
        vix_5d = float((vix.iloc[-1] / vix.iloc[-6] - 1) * 100)
        dma50 = float(spx.rolling(50).mean().iloc[-1])
        spx_vs_dma = float((spx.iloc[-1] / dma50 - 1) * 100)

        ratio = (rsp / spy).dropna()
        breadth = float((ratio.iloc[-1] / ratio.iloc[-21] - 1) * 100) if len(ratio) > 21 else 0.0
        oil_5d = float((oil.iloc[-1] / oil.iloc[-6] - 1) * 100) if len(oil) > 6 else 0.0

        return RegimeInputs(
            vix_level=round(vix_level, 2),
            vix_5d_change_pct=round(vix_5d, 1),
            spx_vs_50dma_pct=round(spx_vs_dma, 2),
            breadth_rsp_spy_20d_pct=round(breadth, 2),
            oil_5d_change_pct=round(oil_5d, 1),
        )
    except Exception:
        return None
