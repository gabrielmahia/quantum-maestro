"""Canonical order models — the single contract between intelligence and execution.

Design rule (non-negotiable): no LLM, prompt, or UI element ever talks to a broker.
Everything becomes an OrderIntent, which must pass the deterministic risk engine
(risk/permission_engine.py) before any adapter may transmit it.

    Intelligence (LLM / signals / UI)
        -> OrderIntent (this file)
        -> PermissionEngine.evaluate()   [deterministic gates]
        -> BrokerRouter.preview()        [broker-side validation]
        -> human approval (modes A/B) or policy approval (mode C)
        -> BrokerRouter.submit()
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import uuid


class Broker(str, Enum):
    TRADIER = "tradier"
    ROBINHOOD_MCP = "robinhood_mcp"   # execution happens via Claude+MCP, not this repo
    PAPER = "paper"


class OrderClass(str, Enum):
    EQUITY = "equity"
    OPTION = "option"
    MULTILEG = "multileg"


class Side(str, Enum):
    BUY = "buy"
    SELL = "sell"
    SELL_SHORT = "sell_short"
    BUY_TO_OPEN = "buy_to_open"
    SELL_TO_OPEN = "sell_to_open"
    BUY_TO_CLOSE = "buy_to_close"
    SELL_TO_CLOSE = "sell_to_close"


class ApprovalMode(str, Enum):
    ADVISORY = "advisory"          # Mode A: human executes manually
    REVIEW = "review"              # Mode B: human clicks approve, system submits
    AUTONOMOUS = "autonomous"      # Mode C: policy approves (gated, see engine)


@dataclass
class OptionLeg:
    side: Side
    quantity: int
    option_symbol: str             # OCC symbol, e.g. SPXW260717P06100000

    def validate(self) -> list[str]:
        errs = []
        if self.quantity <= 0:
            errs.append(f"leg quantity must be > 0, got {self.quantity}")
        if len(self.option_symbol) < 16:
            errs.append(f"option_symbol does not look like OCC format: {self.option_symbol}")
        if self.side in (Side.BUY, Side.SELL, Side.SELL_SHORT):
            errs.append(f"equity side {self.side} used on an option leg")
        return errs


@dataclass
class OrderIntent:
    """Everything the risk engine and adapters need, and nothing broker-specific."""
    strategy_id: str                       # e.g. "SPX_PUT_CREDIT_SPREAD_7DTE"
    broker: Broker
    order_class: OrderClass
    symbol: str                            # underlying
    approval_mode: ApprovalMode

    # economics — REQUIRED before risk evaluation
    max_loss: float                        # dollars, worst case, fees excluded
    limit_price: Optional[float] = None    # net credit/debit or equity limit
    quantity: int = 0                      # shares (equity) or contracts (single option)
    legs: list[OptionLeg] = field(default_factory=list)

    # exit plan — an intent without an exit plan is invalid by policy
    profit_target: Optional[float] = None
    stop_trigger: Optional[float] = None
    time_stop: Optional[str] = None        # ISO datetime — exit by this time regardless

    # provenance — every order must be traceable
    thesis: str = ""                       # one-paragraph human-readable reason
    iwt_score: Optional[int] = None        # 0-10 from IWT engine
    model_version: str = "unversioned"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    intent_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    # state
    risk_approved: bool = False            # set ONLY by PermissionEngine
    risk_verdict: str = "UNEVALUATED"      # GREEN / YELLOW / RED / UNEVALUATED

    def validate(self) -> list[str]:
        """Structural validation. Returns list of errors; empty list == valid."""
        errs: list[str] = []
        if self.max_loss is None or self.max_loss <= 0:
            errs.append("max_loss must be a positive dollar amount — 'unknown risk' is not an order")
        if self.order_class == OrderClass.MULTILEG and len(self.legs) < 2:
            errs.append("multileg order requires >= 2 legs")
        if self.order_class == OrderClass.EQUITY and self.quantity <= 0:
            errs.append("equity order requires quantity > 0")
        if self.order_class != OrderClass.EQUITY and self.limit_price is None:
            errs.append("options orders must be limit orders — market orders on options are prohibited by policy")
        if self.profit_target is None and self.stop_trigger is None and self.time_stop is None:
            errs.append("no exit plan: at least one of profit_target / stop_trigger / time_stop required")
        if not self.thesis.strip():
            errs.append("thesis is required — if you can't say why, you can't trade it")
        for i, leg in enumerate(self.legs):
            errs.extend([f"leg[{i}]: {e}" for e in leg.validate()])
        return errs

    def to_dict(self) -> dict:
        d = asdict(self)
        d["broker"] = self.broker.value
        d["order_class"] = self.order_class.value
        d["approval_mode"] = self.approval_mode.value
        d["legs"] = [{**asdict(l), "side": l.side.value} for l in self.legs]
        return d


def build_vertical_credit_spread(
    underlying: str,
    short_occ: str,
    long_occ: str,
    contracts: int,
    net_credit: float,
    spread_width_points: float,
    strategy_id: str,
    thesis: str,
    iwt_score: Optional[int] = None,
    broker: Broker = Broker.TRADIER,
    approval_mode: ApprovalMode = ApprovalMode.REVIEW,
    multiplier: int = 100,
) -> OrderIntent:
    """Convenience constructor with max-loss math done correctly, once.

    max_loss = (width - credit) * multiplier * contracts
    """
    max_loss = round((spread_width_points - net_credit) * multiplier * contracts, 2)
    return OrderIntent(
        strategy_id=strategy_id,
        broker=broker,
        order_class=OrderClass.MULTILEG,
        symbol=underlying,
        approval_mode=approval_mode,
        max_loss=max_loss,
        limit_price=round(net_credit, 2),
        legs=[
            OptionLeg(Side.SELL_TO_OPEN, contracts, short_occ),
            OptionLeg(Side.BUY_TO_OPEN, contracts, long_occ),
        ],
        profit_target=round(net_credit * 0.5, 2),   # default: take 50% of max profit
        stop_trigger=round(net_credit * 2.0, 2),    # default: cut at 2x credit
        thesis=thesis,
        iwt_score=iwt_score,
    )
