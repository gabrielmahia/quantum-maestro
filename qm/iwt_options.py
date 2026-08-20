"""
IWT Options Methodology — Quantum Maestro
==========================================
Canonical from Teri Ijeoma's "IWT Method With Options" documents (the four
single-leg expressions and the vertical-spread worksheet). Encodes the rule
the codebase was missing: HOW the four options expressions map to the buyer/
seller zones, and WHERE the strike goes relative to the stop-market line.

The four expressions and their zone mapping (from the course PDF):

  BUYER ZONE (bullish — betting price goes UP or stays above strike):
    - BUY A CALL  : debit. Max loss = premium. Profit rises with price.
                    Break-even = strike + premium.
    - SELL A PUT  : credit. You want premium to decay. Max profit = premium;
                    downside is large if assigned. Break-even = strike - premium.
    Strike rule: at/inside the BZ, using the STOP-MARKET line as the strike.

  SELLER ZONE (bearish — betting price goes DOWN or stays below strike):
    - BUY A PUT   : debit. Max loss = premium. Profit rises as price falls.
                    Break-even = strike - premium.
    - SELL A CALL : credit. You want premium to decay. Assignment risk if
                    price rises through strike. Break-even = strike + premium.
    Strike rule: as price exits the SZ, using the stop-market line as the
    strike (or slightly inside the SZ).

Greeks doctrine from the course (plain-English, not a pricing model):
  - Buying options: buy 2+ months out, exit by ~2 weeks to expiry (theta
    works AGAINST you). Buy when IV is LOW (don't buy into earnings), hope
    IV rises after. Buy low / sell high.
  - Selling premium: theta works FOR you; sell into HIGH IV to collect
    inflated premium, buy-to-close when IV/premium is low or let it expire.

DOCTRINE GUARDRAILS (Quantum Maestro overrides where stricter):
  - Naked short options are NOT permitted. "Sell a put" / "sell a call" are
    only expressible as DEFINED-RISK VERTICAL SPREADS here (the course's own
    vertical-spread worksheet is the sanctioned form). A naked short flag
    returns a hard block.
  - No 0DTE; no short premium into an event window (earnings/FOMC).
  - These functions compute break-evens and map expressions to zones. They
    NEVER select a live strike, size a position, or send an order — the app
    engine and the deterministic risk gate own that.

Pure functions, unit-tested, advisory only.
"""

from dataclasses import dataclass
from typing import Literal, Optional


# ----------------------------------------------------------------------
# Expression <-> zone mapping
# ----------------------------------------------------------------------

EXPRESSIONS = {
    "buy_call":  {"zone": "buyer",  "bias": "bullish", "flow": "debit",  "naked_short": False},
    "sell_put":  {"zone": "buyer",  "bias": "bullish", "flow": "credit", "naked_short": True},
    "buy_put":   {"zone": "seller", "bias": "bearish", "flow": "debit",  "naked_short": False},
    "sell_call": {"zone": "seller", "bias": "bearish", "flow": "credit", "naked_short": True},
}


def break_even(expression: str, strike: float, premium: float) -> float:
    """Canonical break-evens from the course:
    buy_call: strike + premium; sell_put: strike - premium;
    buy_put: strike - premium; sell_call: strike + premium."""
    if expression in ("buy_call", "sell_call"):
        return round(strike + premium, 4)
    if expression in ("buy_put", "sell_put"):
        return round(strike - premium, 4)
    raise ValueError(f"unknown expression: {expression}")


def choose_expression(zone_kind: str, iv_position: Optional[float],
                      premium_rich_threshold: float = 50.0,
                      premium_cheap_threshold: float = 25.0) -> dict:
    """Given the zone and IV position, recommend the expression FAMILY.
    Follows the course's buy-low/sell-high IV doctrine:
      - premium CHEAP (low IV)  -> prefer DEBIT (buy the option)
      - premium RICH (high IV)  -> prefer CREDIT (sell premium, defined-risk)
    Returns the debit and credit choice for the zone plus which IV favors."""
    if zone_kind == "buyer":
        debit, credit = "buy_call", "sell_put"
    elif zone_kind == "seller":
        debit, credit = "buy_put", "sell_call"
    else:
        return {"error": "zone_kind must be 'buyer' or 'seller'"}

    if iv_position is None:
        favored, why = None, "IV position unknown - no premium bias"
    elif iv_position >= premium_rich_threshold:
        favored, why = "credit", "IV rich -> sell premium (as a DEFINED-RISK spread)"
    elif iv_position <= premium_cheap_threshold:
        favored, why = "debit", "IV cheap -> buy the option"
    else:
        favored, why = None, "IV mixed - either, lean debit for simplicity"

    return {"zone": zone_kind, "debit_expression": debit, "credit_expression": credit,
            "favored_flow": favored, "rationale": why}


# ----------------------------------------------------------------------
# Defined-risk vertical spread (the sanctioned credit form)
# ----------------------------------------------------------------------

@dataclass
class VerticalSpread:
    """A defined-risk vertical from the course worksheet.
    Credit spread: short the near strike, long the far (protective) strike.
    Max loss is bounded by the width minus credit -> naked risk removed."""
    kind: Literal["put_credit", "call_credit", "call_debit", "put_debit"]
    short_strike: float
    long_strike: float
    net_premium: float          # credit received (spreads) or debit paid
    contracts: int = 1

    @property
    def width(self) -> float:
        return abs(self.short_strike - self.long_strike)

    def economics(self, multiplier: int = 100) -> dict:
        w = self.width
        if self.kind in ("put_credit", "call_credit"):
            max_profit = self.net_premium * multiplier * self.contracts
            max_loss = (w - self.net_premium) * multiplier * self.contracts
        else:  # debit spread
            max_profit = (w - self.net_premium) * multiplier * self.contracts
            max_loss = self.net_premium * multiplier * self.contracts
        rr = (max_profit / max_loss) if max_loss > 0 else float("nan")
        return {
            "kind": self.kind, "width": round(w, 4),
            "max_profit": round(max_profit, 2), "max_loss": round(max_loss, 2),
            "reward_risk": round(rr, 4),
            "credit_pct_of_width": round(100 * self.net_premium / w, 1) if w > 0 else None,
        }


def validate_options_trade(expression: str, defined_risk: bool,
                           dte: int, into_event: bool) -> dict:
    """Doctrine gate for an options expression. Naked short options are
    blocked — credits must be defined-risk verticals. No 0DTE; no short
    premium into an event window."""
    blocks = []
    meta = EXPRESSIONS.get(expression)
    if meta is None:
        return {"allowed": False, "blocks": [f"unknown expression: {expression}"]}

    if meta["naked_short"] and not defined_risk:
        blocks.append(
            f"{expression} as a naked short is not permitted - express it as a "
            f"defined-risk vertical spread (the course's vertical-spread form)")
    if dte < 7:
        blocks.append(f"DTE {dte} < 7 - no 0DTE / near-dated gamma risk")
    if meta["flow"] == "credit" and into_event:
        blocks.append("no short premium into an event window (earnings/FOMC)")

    # Course guidance surfaced as non-blocking notes
    notes = []
    if meta["flow"] == "debit":
        notes.append("buying: 2+ months out, exit ~2 weeks to expiry; buy when IV low")
    else:
        notes.append("selling premium: sell into high IV; theta works for you")

    return {"allowed": len(blocks) == 0, "blocks": blocks, "notes": notes,
            "expression": expression, "flow": meta["flow"], "bias": meta["bias"]}
