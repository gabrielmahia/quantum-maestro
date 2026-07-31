"""
IWT Canonical Risk Plan & Pre-Trade Worksheet — Quantum Maestro
================================================================
Distilled directly from Teri Ijeoma's Trade & Travel course documents:
the Personal Trading Plan risk table, the RR_Excel_Sheet formula chain,
and the IWT Stock Pick / Chart Analysis worksheet.

Two things the repo was missing, both HARD numbers from the source:

1. The full risk-plan CASCADE. The course specifies not just per-trade
   risk but daily / weekly / monthly loss ceilings AND trade-count caps.
   Teri's Stocks/Options defaults on a $100k account:
       per-trade   $1,000   = 1.0%
       daily loss  $3,000   = 3.0%
       weekly loss $5,000   = 5.0%
       monthly     $10,000  = 10.0%
       max trades  2/day, 8/week, 20/month
   These scale with account size; stored here as PERCENTAGES so they
   apply at any equity. (Quantum Maestro's own doctrine is stricter on
   the daily breaker at 2% — see note in check_risk_plan.)

2. The IWT pre-trade WORKSHEET as a scored checklist. The paper form has
   explicit gates the seven-agent stack only partially covered: volume
   > 1M, uptrend, 3-month range position, room to run, earnings distance,
   52-week range position, "$1/day" mover, best-in-breed (relative
   strength), and news. Formalized here so a pick is scored, not vibed.

Pure functions, no network, unit-tested. Advisory only.
"""

from dataclasses import dataclass, field
from typing import Optional


# ======================================================================
# 1. RISK-PLAN CASCADE (canonical percentages, account-size agnostic)
# ======================================================================

@dataclass(frozen=True)
class IWTRiskPlan:
    """Teri's Personal Trading Plan risk parameters, as fractions of equity.
    Defaults are the course's Stocks/Options column (÷ $100k account)."""
    per_trade_pct: float = 0.010       # $1,000 / $100k
    daily_loss_pct: float = 0.030      # $3,000 / $100k
    weekly_loss_pct: float = 0.050     # $5,000 / $100k
    monthly_loss_pct: float = 0.100    # $10,000 / $100k
    max_trades_per_day: int = 2
    max_trades_per_week: int = 8
    max_trades_per_month: int = 20
    min_reward_risk: float = 3.0       # the worksheet's ">3?" gate


# Quantum Maestro's OWN doctrine is deliberately tighter than Teri's on the
# daily breaker (2% vs 3%): the newborn-period impaired-window rule and the
# "protect downside before chasing upside" priority both argue for a smaller
# daily bleed. When the two disagree, the STRICTER number wins.
QM_DAILY_BREAKER_PCT = 0.020


def check_risk_plan(account_equity: float,
                    realized_pnl_today: float,
                    realized_pnl_week: float,
                    realized_pnl_month: float,
                    trades_today: int,
                    trades_week: int,
                    trades_month: int,
                    proposed_trade_risk: float,
                    plan: IWTRiskPlan = IWTRiskPlan()) -> dict:
    """Evaluate a proposed trade against the full canonical cascade.
    Returns allow/block with EVERY breached rule named. Losses are negative.
    The daily ceiling uses the STRICTER of Teri's 3% and QM's 2%."""
    daily_ceiling = min(plan.daily_loss_pct, QM_DAILY_BREAKER_PCT) * account_equity
    weekly_ceiling = plan.weekly_loss_pct * account_equity
    monthly_ceiling = plan.monthly_loss_pct * account_equity
    per_trade_ceiling = plan.per_trade_pct * account_equity

    breaches = []
    # Loss ceilings (pnl negative = loss; compare magnitude)
    if -realized_pnl_today >= daily_ceiling:
        breaches.append(f"daily loss ceiling hit ({daily_ceiling:.0f}); no new trades today")
    if -realized_pnl_week >= weekly_ceiling:
        breaches.append(f"weekly loss ceiling hit ({weekly_ceiling:.0f})")
    if -realized_pnl_month >= monthly_ceiling:
        breaches.append(f"monthly loss ceiling hit ({monthly_ceiling:.0f})")
    # Trade-count caps
    if trades_today >= plan.max_trades_per_day:
        breaches.append(f"max trades/day reached ({plan.max_trades_per_day})")
    if trades_week >= plan.max_trades_per_week:
        breaches.append(f"max trades/week reached ({plan.max_trades_per_week})")
    if trades_month >= plan.max_trades_per_month:
        breaches.append(f"max trades/month reached ({plan.max_trades_per_month})")
    # Per-trade size
    if proposed_trade_risk > per_trade_ceiling + 1e-9:
        breaches.append(f"trade risk {proposed_trade_risk:.0f} exceeds per-trade cap {per_trade_ceiling:.0f}")

    # Would this trade, if it hit its stop, breach the daily ceiling?
    projected_daily_loss = -realized_pnl_today + proposed_trade_risk
    if projected_daily_loss > daily_ceiling + 1e-9:
        breaches.append(
            f"trade's max loss would push daily loss past ceiling "
            f"({projected_daily_loss:.0f} > {daily_ceiling:.0f})")

    return {
        "allowed": len(breaches) == 0,
        "breaches": breaches,
        "ceilings": {
            "per_trade": round(per_trade_ceiling, 2),
            "daily": round(daily_ceiling, 2),
            "weekly": round(weekly_ceiling, 2),
            "monthly": round(monthly_ceiling, 2),
        },
        "daily_ceiling_source": "QM 2% (stricter)" if QM_DAILY_BREAKER_PCT < plan.daily_loss_pct else "Teri 3%",
    }


# ======================================================================
# 2. RR FORMULA CHAIN (canonical, matches RR_Excel_Sheet exactly)
# ======================================================================

def iwt_long_trade(distal_bz: float, proximal_bz: float, proximal_sz: float,
                   atr: float, atr_buffer_pct: float = 0.20,
                   risk_tolerance_dollars: float = 1000.0) -> dict:
    """Canonical LONG math from the course spreadsheet.
    STOP = distal_BZ - 20%*ATR; REWARD = proximal_SZ - proximal_BZ;
    RISK = proximal_BZ - STOP; shares = risk_$ / RISK."""
    stop = distal_bz - atr_buffer_pct * atr
    entry = proximal_bz
    reward = proximal_sz - proximal_bz
    risk = proximal_bz - stop
    rr = reward / risk if risk > 0 else float("nan")
    shares = risk_tolerance_dollars / risk if risk > 0 else 0.0
    return {
        "direction": "long", "entry": round(entry, 2), "stop": round(stop, 2),
        "target": round(proximal_sz, 2), "reward": round(reward, 4),
        "risk": round(risk, 4), "reward_risk": round(rr, 4),
        "shares": round(shares, 2), "cost": round(shares * entry, 2),
        "max_profit": round(shares * reward, 2), "max_loss": round(shares * risk, 2),
        # Canonical order types (IWT worksheet): entry & profit-exit are LIMIT
        # (price control); the protective stop is STOP-MARKET (fill certainty
        # when the level breaks). Shares round DOWN, never up into more risk.
        "order_types": {"entry": "limit", "exit": "limit", "stop": "stop_market"},
        "shares_rounded_down": int(shares),
    }


def iwt_short_trade(distal_sz: float, proximal_sz: float, proximal_bz: float,
                    atr: float, atr_buffer_pct: float = 0.20,
                    risk_tolerance_dollars: float = 1000.0) -> dict:
    """Canonical SHORT math: STOP = distal_SZ + 20%*ATR; entry at proximal_SZ;
    target at proximal_BZ."""
    stop = distal_sz + atr_buffer_pct * atr
    entry = proximal_sz
    reward = proximal_sz - proximal_bz
    risk = stop - proximal_sz
    rr = reward / risk if risk > 0 else float("nan")
    shares = risk_tolerance_dollars / risk if risk > 0 else 0.0
    return {
        "direction": "short", "entry": round(entry, 2), "stop": round(stop, 2),
        "target": round(proximal_bz, 2), "reward": round(reward, 4),
        "risk": round(risk, 4), "reward_risk": round(rr, 4),
        "shares": round(shares, 2), "cost": round(shares * entry, 2),
        "max_profit": round(shares * reward, 2), "max_loss": round(shares * risk, 2),
        "order_types": {"entry": "limit", "exit": "limit", "stop": "stop_market"},
        "shares_rounded_down": int(shares),
    }


# ======================================================================
# 3. IWT PRE-TRADE WORKSHEET (the paper stock-pick form, scored)
# ======================================================================

@dataclass
class PreTradeWorksheet:
    """The IWT Stock Pick / Chart Analysis worksheet as a scored gate.
    Every field maps to a column on Teri's paper form."""
    symbol: str
    volume: Optional[float] = None                # shares/day; gate > 1M
    up_trend: Optional[bool] = None               # UpTrend? Yes/No
    price: Optional[float] = None
    high_52w: Optional[float] = None
    low_52w: Optional[float] = None
    range_3mo_low: Optional[float] = None
    range_3mo_high: Optional[float] = None
    days_to_earnings: Optional[int] = None        # None = unknown
    dollar_a_day: Optional[bool] = None           # moves ~$1/day (liquid mover)
    best_in_breed: Optional[bool] = None          # relative strength / leader
    news_direction: Optional[str] = None          # "up" | "down" | None

    def evaluate(self, earnings_blackout_days: int = 7) -> dict:
        """Return hard blocks (must-fix) and soft warnings, plus a readiness
        verdict. Mirrors how the paper form gates a pick before charting."""
        blocks, warns, notes = [], [], []

        if self.volume is not None and self.volume < 1_000_000:
            blocks.append(f"volume {self.volume:,.0f} < 1M liquidity minimum")
        elif self.volume is None:
            warns.append("volume unknown — confirm > 1M before trading")

        if self.up_trend is False:
            notes.append("not in an uptrend — long setups are counter-trend here")

        # Earnings proximity is a hard block for short premium
        if self.days_to_earnings is not None and self.days_to_earnings <= earnings_blackout_days:
            blocks.append(f"earnings in {self.days_to_earnings}d — inside blackout")

        # 52-week range position
        range_pos_52 = None
        if self.price and self.high_52w and self.low_52w and self.high_52w > self.low_52w:
            range_pos_52 = 100 * (self.price - self.low_52w) / (self.high_52w - self.low_52w)
            if range_pos_52 > 95:
                warns.append("price in top 5% of 52w range — limited room up")
            elif range_pos_52 < 5:
                warns.append("price in bottom 5% of 52w range — limited room down")

        # 3-month range position (Teri: 'where is price in the 3-month range')
        range_pos_3mo = None
        if self.price and self.range_3mo_high and self.range_3mo_low and self.range_3mo_high > self.range_3mo_low:
            range_pos_3mo = 100 * (self.price - self.range_3mo_low) / (self.range_3mo_high - self.range_3mo_low)

        if self.best_in_breed is False:
            warns.append("not best-in-breed — Teri buys sector leaders")
        if self.dollar_a_day is False:
            notes.append("not a ~$1/day mover — may lack the range to hit targets")
        if self.news_direction:
            notes.append(f"news skew: {self.news_direction} — confirm it supports the thesis")

        ready = len(blocks) == 0
        return {
            "symbol": self.symbol, "ready_to_chart": ready,
            "hard_blocks": blocks, "warnings": warns, "notes": notes,
            "range_pos_52w": round(range_pos_52, 1) if range_pos_52 is not None else None,
            "range_pos_3mo": round(range_pos_3mo, 1) if range_pos_3mo is not None else None,
        }
