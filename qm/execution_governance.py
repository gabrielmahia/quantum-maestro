"""
Execution Governance — Quantum Maestro
=======================================
Implements the four "between decision and order" layers from the canonical
system specification (§14 account routing, §18 intent-vs-order, §20 tranche
model, §22 managed-vertical recycling accounting).

These are deliberately DETERMINISTIC. Per the spec's architecture boundary,
an LLM may reason about news and scenarios, but code — not a model — controls
strike relationships, credit/debit orientation, sizing, account permissions,
and order-intent consistency.

Nothing here places an order. It validates, splits, routes, and accounts.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


# ======================================================================
# §14 ACCOUNT-AWARE ROUTING
# The same thesis yields different expressions depending on what the
# account is actually approved for. Thesis is invariant; structure is not.
# ======================================================================

ACCOUNT_CAPABILITIES = {
    # Smaller / lower-approval account: long options and cash only.
    "FIDELITY": {"long_call", "long_put", "cash"},
    # Full defined-risk approval.
    "TOS": {"long_call", "long_put", "bull_call_debit", "bear_put_debit",
            "bull_put_credit", "bear_call_credit", "cash"},
    "TRADIER": {"long_call", "long_put", "bull_call_debit", "bear_put_debit",
                "bull_put_credit", "bear_call_credit", "cash"},
}

# Preference order per thesis, best-structure-first. The router walks this
# list and returns the first structure the account can actually trade.
THESIS_PREFERENCE = {
    "BULLISH": ["bull_call_debit", "bull_put_credit", "long_call", "cash"],
    "BEARISH": ["bear_put_debit", "bear_call_credit", "long_put", "cash"],
    "NEUTRAL": ["cash"],
}

# Credit structures require rich IV AND demonstrated support/rejection
# (spec §13: "Credit spreads should be used only when support/resistance
# has actually demonstrated itself").
CREDIT_STRUCTURES = {"bull_put_credit", "bear_call_credit"}
DEBIT_STRUCTURES = {"bull_call_debit", "bear_put_debit"}


def route_strategy(thesis: str, account: str, iv_rich: bool = False,
                   confirmed: bool = False, event_heavy: bool = False) -> dict:
    """Pick the account-legal expression for a thesis.

    Rules from the spec:
      - Event-heavy sessions prefer DEBIT structures (max loss known without
        relying on stop execution).
      - CREDIT structures require iv_rich AND confirmed.
      - Falls through to long options, then cash.
    """
    thesis = thesis.upper()
    caps = ACCOUNT_CAPABILITIES.get(account.upper())
    if caps is None:
        return {"error": f"unknown account: {account}"}
    if thesis not in THESIS_PREFERENCE:
        return {"error": f"unknown thesis: {thesis}"}

    rejected = []
    for structure in THESIS_PREFERENCE[thesis]:
        if structure not in caps:
            rejected.append((structure, "not permitted in this account"))
            continue
        if structure in CREDIT_STRUCTURES and not (iv_rich and confirmed):
            rejected.append((structure, "credit requires rich IV AND confirmed level"))
            continue
        if event_heavy and structure in CREDIT_STRUCTURES:
            rejected.append((structure, "event-heavy session: prefer defined debit"))
            continue
        return {"thesis": thesis, "account": account.upper(), "strategy": structure,
                "rejected": rejected,
                "note": "thesis is invariant; only the expression is account-specific"}
    return {"thesis": thesis, "account": account.upper(), "strategy": "cash",
            "rejected": rejected, "note": "no permitted structure; cash is an active position"}


# ======================================================================
# §18 INTENT-VERSUS-ORDER CHECK
# An immutable intent is created BEFORE the order. The constructed order
# is then diffed against it. Any mismatch rejects unless explicitly waived.
# ======================================================================

# Which structures are directionally bullish / bearish, and their net type.
STRUCTURE_META = {
    "long_call":         {"exposure": "POSITIVE_DELTA", "net": "DEBIT"},
    "bull_call_debit":   {"exposure": "POSITIVE_DELTA", "net": "DEBIT"},
    "bull_put_credit":   {"exposure": "POSITIVE_DELTA", "net": "CREDIT"},
    "long_put":          {"exposure": "NEGATIVE_DELTA", "net": "DEBIT"},
    "bear_put_debit":    {"exposure": "NEGATIVE_DELTA", "net": "DEBIT"},
    "bear_call_credit":  {"exposure": "NEGATIVE_DELTA", "net": "CREDIT"},
}

THESIS_EXPOSURE = {"BULLISH": "POSITIVE_DELTA", "BEARISH": "NEGATIVE_DELTA"}


@dataclass(frozen=True)
class TradeIntent:
    """Immutable. Created before any order is constructed."""
    thesis: str                 # BULLISH | BEARISH
    instrument: str
    strategy: str               # e.g. bear_put_debit
    expiration: str
    net_type: str               # DEBIT | CREDIT
    max_loss: float
    long_strike: Optional[float] = None
    short_strike: Optional[float] = None
    expected_exposure: Optional[str] = None

    def __post_init__(self):
        if self.expected_exposure is None:
            object.__setattr__(self, "expected_exposure",
                               THESIS_EXPOSURE.get(self.thesis.upper()))


def validate_intent(intent: TradeIntent, order: dict,
                    waive_reasons: Optional[list] = None) -> dict:
    """Diff a constructed order against the immutable intent.

    Checks (spec §18): side/net type, expiration, strikes, max loss, and
    bullish/bearish exposure coherence. The canonical failure this exists to
    catch: thesis=BEARISH but order=bull_put_credit.
    """
    waive_reasons = waive_reasons or []
    mismatches = []

    # 1. Does the strategy's own exposure match the thesis?
    meta = STRUCTURE_META.get(intent.strategy)
    if meta is None:
        mismatches.append(f"unknown strategy in intent: {intent.strategy}")
    else:
        expected = THESIS_EXPOSURE.get(intent.thesis.upper())
        if expected and meta["exposure"] != expected:
            mismatches.append(
                f"thesis {intent.thesis} implies {expected} but strategy "
                f"{intent.strategy} is {meta['exposure']}")
        if meta["net"] != intent.net_type.upper():
            mismatches.append(
                f"strategy {intent.strategy} is {meta['net']} but intent says {intent.net_type}")

    # 2. Does the ORDER match the intent field by field?
    order_strategy = order.get("strategy")
    if order_strategy and order_strategy != intent.strategy:
        mismatches.append(f"order strategy {order_strategy} != intent {intent.strategy}")

    order_meta = STRUCTURE_META.get(order_strategy) if order_strategy else None
    if order_meta:
        expected = THESIS_EXPOSURE.get(intent.thesis.upper())
        if expected and order_meta["exposure"] != expected:
            mismatches.append(
                f"ORDER exposure {order_meta['exposure']} contradicts thesis {intent.thesis}")

    if order.get("net_type") and order["net_type"].upper() != intent.net_type.upper():
        mismatches.append(f"order net {order['net_type']} != intent {intent.net_type}")
    if order.get("expiration") and order["expiration"] != intent.expiration:
        mismatches.append(f"order expiration {order['expiration']} != intent {intent.expiration}")
    for leg in ("long_strike", "short_strike"):
        iv, ov = getattr(intent, leg), order.get(leg)
        if iv is not None and ov is not None and abs(float(iv) - float(ov)) > 1e-9:
            mismatches.append(f"order {leg} {ov} != intent {iv}")
    if order.get("max_loss") is not None and intent.max_loss is not None:
        if float(order["max_loss"]) > float(intent.max_loss) + 1e-9:
            mismatches.append(
                f"order max_loss {order['max_loss']} EXCEEDS intent {intent.max_loss}")

    unwaived = [m for m in mismatches if m not in waive_reasons]
    return {"valid": len(unwaived) == 0, "mismatches": mismatches,
            "unwaived": unwaived, "waived": [m for m in mismatches if m in waive_reasons]}


# ======================================================================
# §20 TRANCHE MODEL
# If risk permits N, open ~0.25N. Add only on NEW information that
# improves the thesis. Never add to rescue a failed thesis.
# ======================================================================

@dataclass
class TranchePlan:
    permitted_total: int
    first_fraction: float = 0.25

    def build(self) -> dict:
        n = self.permitted_total
        if n < 1:
            return {"error": "risk budget cannot support a position"}
        first = max(1, int(n * self.first_fraction))
        remaining = n - first
        # Split the remainder into up to three add-on tranches.
        adds = []
        if remaining > 0:
            per = max(1, remaining // 3)
            left = remaining
            for label in ("after confirmation", "after favorable structure", "reserved / fresh setup"):
                if left <= 0:
                    break
                take = min(per, left)
                adds.append({"size": take, "unlock": label})
                left -= take
            if left > 0 and adds:
                adds[-1]["size"] += left
        return {"permitted_total": n, "initial": first, "add_ons": adds,
                "rule": "Add only on NEW information improving the thesis. "
                        "NEVER add to rescue a failed thesis."}


def may_add_tranche(thesis_intact: bool, new_information: bool,
                    position_underwater: bool) -> dict:
    """Gate an add-on. The dangerous case this blocks: averaging down into
    a losing position and calling it 'scaling in'."""
    if not thesis_intact:
        return {"allowed": False, "reason": "thesis invalidated - exit, do not add"}
    if position_underwater and not new_information:
        return {"allowed": False,
                "reason": "underwater with no new information - this is a rescue add, blocked"}
    if not new_information:
        return {"allowed": False, "reason": "no new information improving the thesis"}
    return {"allowed": True, "reason": "thesis intact and new information supports adding"}


# ======================================================================
# §22 MANAGED VERTICAL / SHORT-LEG RECYCLING ACCOUNTING
# The naive error: calling every realized short-leg gain "extra profit."
# Some of it is just capturing the original decay path early.
# ======================================================================

@dataclass
class ShortLegCycle:
    """One sell-to-open -> buy-to-close cycle on the short leg."""
    sto_price: float
    btc_price: float
    contracts: int = 1
    fees: float = 0.0
    slippage: float = 0.0
    multiplier: int = 100

    def gross_pl(self) -> float:
        return (self.sto_price - self.btc_price) * self.multiplier * self.contracts

    def net_pl(self) -> float:
        return self.gross_pl() - self.fees - self.slippage


@dataclass
class ManagedVerticalLedger:
    """Tracks a long hedge plus a recycled short leg, separating true
    management alpha from decay that would have accrued anyway."""
    long_hedge_cost: float           # premium paid for the protective leg
    contracts: int = 1
    multiplier: int = 100
    cycles: list = field(default_factory=list)

    def add_cycle(self, cycle: ShortLegCycle):
        self.cycles.append(cycle)

    def gross_short_pl(self) -> float:
        return sum(c.gross_pl() for c in self.cycles)

    def net_short_pl(self) -> float:
        return sum(c.net_pl() for c in self.cycles)

    def friction(self) -> float:
        return sum(c.fees + c.slippage for c in self.cycles)

    def management_alpha(self) -> dict:
        """Re-entry improvement = STO_new - BTC_prior, summed across
        re-entries, minus the incremental friction those re-entries cost.
        This is the honest measure of whether recycling beat passive holding."""
        if len(self.cycles) < 2:
            return {"re_entries": 0, "re_entry_improvement": 0.0,
                    "incremental_friction": 0.0, "management_alpha": 0.0,
                    "note": "fewer than 2 cycles - no re-entry to measure"}
        improvement = 0.0
        incremental_friction = 0.0
        for prior, nxt in zip(self.cycles, self.cycles[1:]):
            improvement += (nxt.sto_price - prior.btc_price) * self.multiplier * nxt.contracts
            incremental_friction += nxt.fees + nxt.slippage
        alpha = improvement - incremental_friction
        return {"re_entries": len(self.cycles) - 1,
                "re_entry_improvement": round(improvement, 2),
                "incremental_friction": round(incremental_friction, 2),
                "management_alpha": round(alpha, 2),
                "note": "positive alpha = recycling beat passive holding, net of friction"}

    def hedge_status(self) -> dict:
        """How much of the long hedge has been funded by short-leg income."""
        cost = self.long_hedge_cost * self.multiplier * self.contracts
        funded = self.net_short_pl()
        return {"long_hedge_original_cost": round(cost, 2),
                "cumulative_short_income": round(funded, 2),
                "remaining_hedge_basis": round(cost - funded, 2),
                "hedge_fully_funded": funded >= cost}

    def summary(self) -> dict:
        s = {"cycles": len(self.cycles),
             "gross_short_pl": round(self.gross_short_pl(), 2),
             "net_short_pl": round(self.net_short_pl(), 2),
             "friction": round(self.friction(), 2)}
        s.update(self.management_alpha())
        s.update(self.hedge_status())
        s["discipline_rule"] = ("Do NOT re-sell the short leg merely because premium "
                                "rose. The underlying IWT setup must requalify.")
        return s
