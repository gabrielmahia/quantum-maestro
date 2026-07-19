"""
Quantum Maestro — Hard Limits (The Constitution)
================================================
These are deterministic, non-negotiable rules. The AI layer may PROPOSE.
Only this layer DISPOSES. Nothing here is overridable from the UI at
runtime by design: changing a limit requires a code commit, which creates
an audit trail and a cooling-off period.

Philosophy: "Determine whether to trade before determining what to trade."
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class HardLimits:
    # --- Per-trade risk ---
    MAX_RISK_PCT_PER_TRADE: float = 0.01      # 1% of account equity, fixed-risk (Ijeoma)
    MIN_REWARD_RISK: float = 2.0              # directional trades need >= 2:1
    MIN_REWARD_RISK_INCOME: float = 0.25      # credit/income structures: credit >= 25% of max loss

    # --- Options guardrails ---
    MIN_DTE: int = 7                          # 0DTE and weekly-expiry-day trading is banned
    MAX_DTE_SHORT_PREMIUM: int = 60           # short premium beyond 60 DTE = poor theta efficiency
    BAN_NAKED_SHORT_OPTIONS: bool = True      # defined-risk only

    # --- Portfolio ---
    MAX_CONCURRENT_POSITIONS: int = 3
    MAX_PORTFOLIO_HEAT: float = 0.03          # sum of open risk <= 3% of equity
    MAX_SINGLE_UNDERLYING_EXPOSURE: float = 0.20  # notional per underlying

    # --- Behavioral circuit breakers (the real edge) ---
    DAILY_LOSS_CIRCUIT_BREAKER: float = 0.02  # stop trading after -2% day
    COOLDOWN_DAYS_AFTER_FULL_LOSS: int = 1    # >= 1R loss -> next trade no sooner than next session
    BAN_REPAIR_TRADES: bool = True            # no new position "fixing" an open loser, same underlying
    BAN_AVERAGING_DOWN: bool = True

    # --- Event risk ---
    EVENT_BLACKOUT_HOURS: int = 24            # no NEW short premium within 24h of FOMC/CPI/NFP/earnings
    # (long premium / hedges are permitted through events — buying protection is always legal)

    # --- Regime sizing multipliers ---
    REGIME_MULTIPLIER: dict = field(default_factory=lambda: {
        "OFFENSIVE": 1.00,
        "NEUTRAL": 0.60,
        "DEFENSIVE": 0.30,
        "LOCKDOWN": 0.00,   # circuit breaker tripped / promotion gate failed
    })

    # --- Kelly ---
    KELLY_FRACTION_CAP: float = 0.25          # never exceed quarter-Kelly


@dataclass(frozen=True)
class PromotionGate:
    """Criteria to promote SHADOW -> LIVE. All must pass. Evaluated from the journal, not from memory or vibes."""
    MIN_DECISIONS_LOGGED: int = 50            # includes NO-TRADE decisions
    MIN_CLOSED_TRADES: int = 30
    MIN_EXPECTANCY_R: float = 0.10            # net of modeled costs, in R units
    MAX_DRAWDOWN_PCT: float = 0.10
    MAX_RULE_VIOLATIONS: int = 0              # a single hard-rule violation resets the clock
    MIN_CALENDAR_DAYS: int = 60               # no promotion on a hot week


LIMITS = HardLimits()
GATE = PromotionGate()

MODES = ("SHADOW", "LIVE")
DEFAULT_MODE = "SHADOW"

WATCHLIST_CORE = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "IWM"]

EVENT_TYPES = ["FOMC", "CPI", "NFP", "PPI", "Earnings (underlying)", "Treasury Auction", "Geopolitical", "OpEx"]
