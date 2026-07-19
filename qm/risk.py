"""
Deterministic Risk Engine — the veto layer
==========================================
The AI (or the human) PROPOSES a trade. This engine DISPOSES.
Every rule returns PASS / VETO / WARN with a reason. A single VETO kills
the trade. Vetoes are logged to the journal as decisions — a prevented
bad trade is a positive-expectancy event and deserves a record.

No rule here uses judgment, sentiment, or narrative. That is the point.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from .config import LIMITS


@dataclass
class TradeProposal:
    underlying: str
    direction: str                 # LONG / SHORT / NEUTRAL / VOLATILITY / HEDGE
    structure: str                 # e.g. "Put Credit Spread", "Long Call", "Shares"
    is_option: bool = True
    is_short_premium: bool = False
    is_defined_risk: bool = True
    dte: int = 30
    entry: float = 0.0
    stop: float = 0.0              # for shares/directional
    target: float = 0.0
    max_loss_per_unit: float = 0.0   # per contract/share, dollars
    max_gain_per_unit: float = 0.0
    account_equity: float = 10000.0
    proposed_risk_dollars: float = 0.0
    open_positions: int = 0
    open_portfolio_heat_pct: float = 0.0   # existing open risk as fraction of equity
    has_open_losing_position_same_underlying: bool = False
    hours_to_next_major_event: float = 999.0
    todays_pnl_pct: float = 0.0
    last_full_loss_within_cooldown: bool = False
    regime: str = "NEUTRAL"
    thesis: str = ""


@dataclass
class RiskVerdict:
    approved: bool
    vetoes: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    passes: list = field(default_factory=list)
    adjusted_max_risk_dollars: float = 0.0


def evaluate(p: TradeProposal) -> RiskVerdict:
    v = RiskVerdict(approved=True)

    def veto(rule, msg):
        v.vetoes.append(f"[{rule}] {msg}")
        v.approved = False

    def warn(rule, msg):
        v.warnings.append(f"[{rule}] {msg}")

    def ok(rule):
        v.passes.append(rule)

    # 1. Regime gate
    mult = LIMITS.REGIME_MULTIPLIER.get(p.regime, 0.0)
    if mult == 0.0:
        veto("REGIME", f"Regime is {p.regime}: no new risk permitted.")
    else:
        ok(f"REGIME ({p.regime}, sizing x{mult})")

    # 2. DTE — the 0DTE ban
    if p.is_option:
        if p.dte < LIMITS.MIN_DTE:
            veto("DTE", f"{p.dte} DTE < minimum {LIMITS.MIN_DTE}. 0DTE/expiry-week gambling is banned. "
                        "(This rule exists because of the AAPL 325/327.5 trade.)")
        elif p.is_short_premium and p.dte > LIMITS.MAX_DTE_SHORT_PREMIUM:
            warn("DTE", f"Short premium at {p.dte} DTE has poor theta efficiency (>60).")
        else:
            ok("DTE")

    # 3. Defined risk only
    if p.is_option and p.is_short_premium and not p.is_defined_risk and LIMITS.BAN_NAKED_SHORT_OPTIONS:
        veto("DEFINED-RISK", "Naked short options are banned. Use spreads.")
    else:
        ok("DEFINED-RISK")

    # 4. Reward:risk
    if p.max_loss_per_unit > 0:
        rr = (p.max_gain_per_unit / p.max_loss_per_unit) if p.max_loss_per_unit else 0
        if p.is_short_premium:
            if rr < LIMITS.MIN_REWARD_RISK_INCOME:
                veto("R:R", f"Credit is only {rr:.2f}x max loss (< {LIMITS.MIN_REWARD_RISK_INCOME}). "
                            "Selling pennies in front of a steamroller.")
            else:
                ok(f"R:R income ({rr:.2f})")
        else:
            if rr < LIMITS.MIN_REWARD_RISK:
                veto("R:R", f"Reward:risk {rr:.2f} < {LIMITS.MIN_REWARD_RISK}:1 minimum for directional trades.")
            else:
                ok(f"R:R directional ({rr:.2f})")

    # 5. Per-trade risk cap (regime-adjusted)
    max_risk = p.account_equity * LIMITS.MAX_RISK_PCT_PER_TRADE * mult
    v.adjusted_max_risk_dollars = round(max_risk, 2)
    if p.proposed_risk_dollars > max_risk + 0.01:
        veto("RISK-CAP", f"Proposed risk ${p.proposed_risk_dollars:,.0f} exceeds regime-adjusted cap "
                         f"${max_risk:,.0f} ({LIMITS.MAX_RISK_PCT_PER_TRADE:.0%} x {mult}x).")
    else:
        ok("RISK-CAP")

    # 6. Portfolio limits
    if p.open_positions >= LIMITS.MAX_CONCURRENT_POSITIONS:
        veto("CONCURRENCY", f"{p.open_positions} open positions >= max {LIMITS.MAX_CONCURRENT_POSITIONS}.")
    else:
        ok("CONCURRENCY")

    projected_heat = p.open_portfolio_heat_pct + (p.proposed_risk_dollars / max(p.account_equity, 1))
    if projected_heat > LIMITS.MAX_PORTFOLIO_HEAT:
        veto("HEAT", f"Projected portfolio heat {projected_heat:.1%} > {LIMITS.MAX_PORTFOLIO_HEAT:.0%} cap.")
    else:
        ok("HEAT")

    # 7. Anti-repair / anti-averaging (behavioral)
    if p.has_open_losing_position_same_underlying and LIMITS.BAN_REPAIR_TRADES:
        veto("NO-REPAIR", f"Open losing position in {p.underlying}. New trades in the same underlying are "
                          "banned until it is closed. One losing trade must not become a campaign.")
    else:
        ok("NO-REPAIR")

    # 8. Event blackout (short premium only; hedges/long premium exempt)
    if p.is_short_premium and p.hours_to_next_major_event < LIMITS.EVENT_BLACKOUT_HOURS:
        veto("EVENT-BLACKOUT", f"Major event in {p.hours_to_next_major_event:.0f}h "
                               f"(< {LIMITS.EVENT_BLACKOUT_HOURS}h). No new short premium into binary events.")
    else:
        ok("EVENT-BLACKOUT")

    # 9. Daily circuit breaker
    if p.todays_pnl_pct <= -LIMITS.DAILY_LOSS_CIRCUIT_BREAKER * 100:
        veto("CIRCUIT-BREAKER", f"Today's P&L {p.todays_pnl_pct:.1f}% breached the "
                                f"-{LIMITS.DAILY_LOSS_CIRCUIT_BREAKER:.0%} daily stop. Done for the day.")
    else:
        ok("CIRCUIT-BREAKER")

    # 10. Post-loss cooldown (anti-revenge)
    if p.last_full_loss_within_cooldown:
        veto("COOLDOWN", "A >=1R loss occurred within the cooldown window. Revenge trading is the single "
                         "most expensive behavior in retail options. Sit out one session.")
    else:
        ok("COOLDOWN")

    # 11. Thesis required (a trade without a falsifiable thesis is a coin flip)
    if len(p.thesis.strip()) < 20:
        warn("THESIS", "Thesis is thin. If you can't write two sentences on why, you don't have a trade.")

    return v
