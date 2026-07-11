"""Deterministic trade-permission engine.

This is the layer that cannot be argued with. LLMs propose; this code disposes.
Every gate returns (passed, reason). A single hard-gate failure => RED, no trade.
Soft-gate failures => YELLOW (requires human approval even in autonomous mode).

Philosophy (Teri/IWT + household rules):
 - Survival first: max loss per trade, daily/weekly loss cutoffs, portfolio heat.
 - No trading through Tier-1 events; no trading stale data; no trading tired
   (session_flags.human_impaired covers newborn-sleep-deprivation mode).
 - PDT budget is a first-class resource, not an afterthought.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from quantum_maestro.execution.order_models import OrderIntent, ApprovalMode


@dataclass
class AccountState:
    equity: float
    cash: float
    open_positions: int
    day_trades_remaining: Optional[int]      # None => not PDT-constrained
    realized_pnl_today: float = 0.0
    realized_pnl_week: float = 0.0
    open_risk_dollars: float = 0.0           # sum of max_loss across open positions


@dataclass
class MarketState:
    market_open: bool
    minutes_to_tier1_event: Optional[int]    # None => no scheduled Tier-1 event
    data_age_seconds: float = 0.0
    vix: Optional[float] = None
    bid_ask_spread_pct: Optional[float] = None   # for the traded instrument


@dataclass
class SessionFlags:
    kill_switch: bool = False                # global stop — set by human only
    human_impaired: bool = False             # e.g. paternity mode / sleep deprivation
    paper_only: bool = True                  # default posture is paper


@dataclass
class RiskLimits:
    max_risk_per_trade_pct: float = 0.5      # of equity
    max_daily_loss_pct: float = 1.0
    max_weekly_loss_pct: float = 2.5
    max_portfolio_heat_pct: float = 3.0
    max_open_positions: int = 3
    event_blackout_minutes: int = 30
    max_data_age_seconds: float = 90.0
    max_bid_ask_spread_pct: float = 5.0      # options liquidity floor
    min_iwt_score_autonomous: int = 7


@dataclass
class Verdict:
    color: str                               # GREEN / YELLOW / RED
    hard_failures: list[str] = field(default_factory=list)
    soft_warnings: list[str] = field(default_factory=list)

    @property
    def approved(self) -> bool:
        return self.color == "GREEN"


def evaluate(
    intent: OrderIntent,
    account: AccountState,
    market: MarketState,
    flags: SessionFlags,
    limits: RiskLimits = RiskLimits(),
) -> Verdict:
    hard: list[str] = []
    soft: list[str] = []

    # ---- structural validity is gate zero -----------------------------------
    errs = intent.validate()
    if errs:
        hard.extend([f"structure: {e}" for e in errs])

    # ---- session gates -------------------------------------------------------
    if flags.kill_switch:
        hard.append("kill switch engaged")
    if flags.human_impaired and intent.approval_mode != ApprovalMode.ADVISORY:
        hard.append("human_impaired flag set: only advisory mode permitted (no order submission)")
    if flags.paper_only and intent.broker.value != "paper":
        hard.append("session is paper_only: live broker intents are blocked")

    # ---- market gates --------------------------------------------------------
    if not market.market_open:
        hard.append("market closed")
    if market.data_age_seconds > limits.max_data_age_seconds:
        hard.append(f"stale data: {market.data_age_seconds:.0f}s old (max {limits.max_data_age_seconds:.0f}s)")
    if (market.minutes_to_tier1_event is not None
            and market.minutes_to_tier1_event <= limits.event_blackout_minutes):
        hard.append(f"Tier-1 event in {market.minutes_to_tier1_event} min (blackout {limits.event_blackout_minutes} min)")
    if (market.bid_ask_spread_pct is not None
            and market.bid_ask_spread_pct > limits.max_bid_ask_spread_pct):
        hard.append(f"illiquid: bid/ask spread {market.bid_ask_spread_pct:.1f}% > {limits.max_bid_ask_spread_pct:.1f}%")

    # ---- account / risk gates -------------------------------------------------
    per_trade_cap = account.equity * limits.max_risk_per_trade_pct / 100.0
    if intent.max_loss > per_trade_cap:
        hard.append(f"max_loss ${intent.max_loss:.0f} exceeds per-trade cap ${per_trade_cap:.0f} "
                    f"({limits.max_risk_per_trade_pct}% of equity)")

    daily_cap = account.equity * limits.max_daily_loss_pct / 100.0
    if -account.realized_pnl_today >= daily_cap:
        hard.append(f"daily loss limit reached (${-account.realized_pnl_today:.0f} >= ${daily_cap:.0f}) — done for today")

    weekly_cap = account.equity * limits.max_weekly_loss_pct / 100.0
    if -account.realized_pnl_week >= weekly_cap:
        hard.append(f"weekly loss limit reached — done for the week")

    heat_after = account.open_risk_dollars + intent.max_loss
    heat_cap = account.equity * limits.max_portfolio_heat_pct / 100.0
    if heat_after > heat_cap:
        hard.append(f"portfolio heat after trade ${heat_after:.0f} exceeds cap ${heat_cap:.0f}")

    if account.open_positions >= limits.max_open_positions:
        hard.append(f"open positions {account.open_positions} at max {limits.max_open_positions}")

    if account.day_trades_remaining is not None and account.day_trades_remaining <= 0:
        soft.append("no day trades remaining — position must be held overnight; confirm that is intended")

    # ---- quality gates (soft) --------------------------------------------------
    if intent.iwt_score is None:
        soft.append("no IWT score attached — trade quality unassessed")
    elif intent.iwt_score < 5:
        hard.append(f"IWT score {intent.iwt_score} < 5: setup does not qualify")
    elif intent.iwt_score < limits.min_iwt_score_autonomous:
        soft.append(f"IWT score {intent.iwt_score} below autonomous threshold "
                    f"{limits.min_iwt_score_autonomous}: human review required")

    # ---- autonomous mode is held to the strictest standard ---------------------
    if intent.approval_mode == ApprovalMode.AUTONOMOUS and soft:
        hard.append("autonomous mode requires zero soft warnings; downgrade to REVIEW")

    color = "RED" if hard else ("YELLOW" if soft else "GREEN")
    intent.risk_verdict = color
    intent.risk_approved = (color == "GREEN")
    return Verdict(color=color, hard_failures=hard, soft_warnings=soft)
