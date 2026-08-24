"""
IWT Zone Scoring & Boundary Survival — Quantum Maestro research module
=====================================================================
Distilled from a VIP-archive backtest engine (external work), keeping the
sound parts and hardening the fragile ones. This is RESEARCH tooling — it
feeds the promotion gate's expectancy question, it does not place trades.

The eight-point odds enhancer (the genuinely valuable idea):
  base candles + departure speed + freshness + reward:risk -> 0..8 score.
Score 7-8 = primary cohort; 5-6 = separate half-size cohort; 0-4 = rejected.

What this module deliberately does NOT do (honesty over theater):
  - It does not claim to reproduce Teri's manual zone labels. Zones here are
    a transparent MECHANICAL PROXY. The stated next step — comparing proxy
    zones to hand-marked examples — is the real validation and is not done.
  - It does not produce options P&L. Daily bars cannot price a spread.
    Boundary-survival gives the honest observable: did price breach the
    short strike's neighborhood within N days? That is a probability, not
    a fill.
  - Daily bars cannot resolve intraday stop-vs-target order; ambiguous bars
    are counted as STOP-FIRST (conservative), and counted separately so the
    ambiguity is visible, not hidden.

Every function is pure and unit-testable. No network here; data is passed in.
"""

from dataclasses import dataclass
from typing import Literal, Optional


# --------------------------------------------------------------------------
# Score components (the eight-point odds enhancer)
# --------------------------------------------------------------------------

def base_candle_score(n: int) -> int:
    """Tighter base = stronger zone. 1-2 candles=2, 3-4=1, 5+=0."""
    if n <= 2:
        return 2
    if n <= 4:
        return 1
    return 0


def departure_score(strength_atr: float, fast: float = 1.5, average: float = 0.75) -> int:
    """Departure body as a multiple of ATR. Fast>=1.5=2, avg>=0.75=1, slow=0."""
    if strength_atr >= fast:
        return 2
    if strength_atr >= average:
        return 1
    return 0


def freshness_score(prior_visits: int) -> int:
    """Untested zone is strongest. 0 revisits=2, 1=1, 2+=0.
    NOTE: prior_visits must EXCLUDE the current entry touch (look-ahead control)."""
    if prior_visits == 0:
        return 2
    if prior_visits == 1:
        return 1
    return 0


def reward_risk_score(rr: float) -> int:
    """3.0+=2, 2.0-2.99=1, <2.0=0."""
    if rr >= 3.0:
        return 2
    if rr >= 2.0:
        return 1
    return 0


# ── Cohort bands: Teri's materials state TWO different schemes ─────────────
# SOURCE CONFLICT (documented honestly rather than silently resolved):
#
#   "Deciding Your Entry Strategy" (Odd Enhancers PDF):
#       6-8 -> enter at the proximal line
#       4-6 -> confirmation entry
#       <4  -> skip
#     (note: the PDF itself OVERLAPS at 6, so 6 is ambiguous in the source)
#
#   Key Documents odds table (DOCX):
#       "SCORE 7-8 TAKE THE TRADE"
#
# COURSE  = the PDF's entry-strategy bands, faithful to the published rule.
# STRICT  = 7-8 primary / 5-6 secondary, one notch tighter. Quantum Maestro's
#           default, because a 6 sitting in the PDF's overlap should require
#           confirmation rather than a direct fill, and because tightening a
#           quality gate is the conservative direction to err.
#
# Both are selectable. Whichever you run, log which band scheme produced the
# cohort so backtests and the journal remain comparable.
BAND_SCHEMES = {
    "STRICT": {"primary_min": 7, "secondary_min": 5,
               "source": "Key Documents odds table ('7-8 TAKE THE TRADE'), tightened"},
    "COURSE": {"primary_min": 6, "secondary_min": 4,
               "source": "Odd Enhancers PDF 'Deciding Your Entry Strategy' (6-8 / 4-6 / <4)"},
}
DEFAULT_BAND_SCHEME = "STRICT"


def odds_enhancer(base_candles: int, departure_strength_atr: float,
                  prior_visits: int, reward_risk: float,
                  band_scheme: str = DEFAULT_BAND_SCHEME) -> dict:
    """The full 0-8 score plus its parts and the cohort it belongs to.

    band_scheme: 'STRICT' (default, 7/5) or 'COURSE' (6/4, per the published
    entry-strategy PDF). See BAND_SCHEMES for the source of each.
    """
    bands = BAND_SCHEMES.get(band_scheme)
    if bands is None:
        raise ValueError(f"unknown band_scheme {band_scheme}; use {list(BAND_SCHEMES)}")

    parts = {
        "base": base_candle_score(base_candles),
        "departure": departure_score(departure_strength_atr),
        "freshness": freshness_score(prior_visits),
        "reward_risk": reward_risk_score(reward_risk),
    }
    total = sum(parts.values())
    if total >= bands["primary_min"]:
        cohort, size_mult, action = "PRIMARY", 1.0, "eligible (limit at proximal)"
    elif total >= bands["secondary_min"]:
        cohort, size_mult, action = "SECONDARY", 0.5, "conditional (require confirmation, half size)"
    else:
        cohort, size_mult, action = "REJECTED", 0.0, "skip"
    return {"score": total, "parts": parts, "cohort": cohort,
            "size_multiplier": size_mult, "action": action,
            "band_scheme": band_scheme, "bands": bands,
            "note": "log the band scheme with every decision so cohorts stay comparable"}


# --------------------------------------------------------------------------
# "Bank-like numbers" — the round-number stop rule
# --------------------------------------------------------------------------

def bank_number_adjusted_stop(raw_stop: float, direction: str,
                              increments: tuple = (50.0, 25.0, 10.0, 5.0, 1.0),
                              clearance: float = 0.02) -> dict:
    """The worksheet says the stop goes "a little below the Buyers Level --
    20% of the average daily move / BELOW BANK-LIKE NUMBERS".

    Round numbers are where stop clusters sit. If the ATR-derived stop lands
    AT or just above a round number (for a long), a sweep of that level takes
    you out before the thesis has actually failed. This nudges the stop to the
    far side of the nearest round number it is sitting on top of.

    Returns the adjusted stop plus which round number triggered the move, so
    the adjustment is always visible rather than silent.
    """
    if raw_stop <= 0:
        return {"stop": raw_stop, "adjusted": False, "reason": "invalid stop"}

    for inc in increments:
        # nearest round number at this increment, at or below the raw stop
        below = (int(raw_stop / inc)) * inc
        if below <= 0:
            continue
        # distance as a fraction of the increment
        dist = raw_stop - below
        # "sitting on top of" = within 15% of the increment above the round number
        if 0 <= dist <= 0.15 * inc:
            if direction == "long":
                new_stop = below - clearance * inc
            else:  # short: stop is above; push it above the round number
                above = below + inc
                new_stop = above + clearance * inc
            return {"stop": round(new_stop, 4), "adjusted": True,
                    "bank_number": below if direction == "long" else below + inc,
                    "increment": inc,
                    "reason": f"raw stop {raw_stop} sat on the {below if direction=='long' else below+inc} "
                              f"round level; moved clear so a sweep doesn't stop you out"}
    return {"stop": round(raw_stop, 4), "adjusted": False,
            "reason": "stop not sitting on a bank-like number"}


def haircut_target(target: float, direction: str, haircut_pct: float = 0.02,
                   risk: float = None) -> dict:
    """The worksheet says exit "a little BEFORE the first line of Sellers
    Level" — i.e. don't demand the last cent of the move. Taking a small
    haircut raises fill probability at the cost of a little reward.

    haircut is a fraction of RISK when risk is supplied (so it scales with the
    trade), else a fraction of the target price.
    """
    step = (risk * haircut_pct * 10) if risk else (target * haircut_pct)
    adjusted = target - step if direction == "long" else target + step
    return {"target": round(adjusted, 4), "original_target": round(target, 4),
            "haircut": round(abs(target - adjusted), 4),
            "reason": "exit a little before the opposing level to improve fill probability"}


# --------------------------------------------------------------------------
# Zone geometry & trade framing
# --------------------------------------------------------------------------

@dataclass
class ZoneTrade:
    direction: Literal["long", "short"]
    entry: float
    stop: float
    target: float

    @property
    def risk(self) -> float:
        return abs(self.entry - self.stop)

    @property
    def reward(self) -> float:
        return abs(self.target - self.entry)

    @property
    def reward_risk(self) -> float:
        return self.reward / self.risk if self.risk > 0 else 0.0

    def is_valid(self) -> bool:
        """Directional geometry must be coherent: for a long, stop<entry<target."""
        if self.risk <= 0 or self.reward <= 0:
            return False
        if self.direction == "long":
            return self.stop < self.entry < self.target
        return self.target < self.entry < self.stop


def frame_trade(direction: str, proximal: float, distal: float, target: float,
                atr: float, atr_buffer: float = 0.20) -> ZoneTrade:
    """Frame a zone trade with the course's ATR stop buffer.
    Long: enter at proximal (top of buyer zone), stop below distal.
    Short: enter at proximal (bottom of seller zone), stop above distal."""
    if direction == "long":
        return ZoneTrade("long", entry=proximal, stop=distal - atr_buffer * atr, target=target)
    return ZoneTrade("short", entry=proximal, stop=distal + atr_buffer * atr, target=target)


# --------------------------------------------------------------------------
# Boundary survival — the honest options proxy
# --------------------------------------------------------------------------

@dataclass
class BoundaryResult:
    horizon_days: int
    atr_buffer: float
    boundary: float
    ever_breached: bool      # path touched the boundary at any point
    terminal_breach: bool    # closed beyond boundary at horizon end
    max_adverse_excursion_atr: float


def boundary_survival(zone_kind: str, distal: float, atr: float,
                      forward_lows: list, forward_highs: list, forward_closes: list,
                      horizon_days: int, atr_buffer: float) -> Optional[BoundaryResult]:
    """Did price breach the short-strike neighborhood within the horizon?

    For a short PUT under a buyer zone, the danger is price falling through
    (distal - buffer*ATR). For a short CALL over a seller zone, price rising
    through (distal + buffer*ATR). This is the observable that stands in for
    'would my short spread have been tested' WITHOUT pretending to price it.
    """
    if atr <= 0 or not forward_lows:
        return None
    end = min(horizon_days, len(forward_lows))
    lows = forward_lows[:end]
    highs = forward_highs[:end]
    closes = forward_closes[:end]
    if not closes:
        return None

    if zone_kind == "buyer":  # short put danger = downside breach
        boundary = distal - atr_buffer * atr
        ever = any(lo <= boundary for lo in lows)
        terminal = closes[-1] <= boundary
        mae = min((lo - boundary) / atr for lo in lows)
    else:                      # seller zone, short call danger = upside breach
        boundary = distal + atr_buffer * atr
        ever = any(hi >= boundary for hi in highs)
        terminal = closes[-1] >= boundary
        mae = max((hi - boundary) / atr for hi in highs)

    return BoundaryResult(horizon_days, atr_buffer, boundary, ever, terminal, mae)


# --------------------------------------------------------------------------
# Trade outcome under daily-bar ambiguity (conservative, transparent)
# --------------------------------------------------------------------------

def resolve_daily_outcome(direction: str, stop: float, target: float,
                          bar_low: float, bar_high: float) -> Optional[str]:
    """On a single daily bar, which was hit? Daily bars can't tell intraday
    ORDER, so a bar spanning both is STOP-FIRST (conservative) and labelled
    'stop_ambiguous' so the ambiguity is countable, never hidden.
    Returns 'target' | 'stop' | 'stop_ambiguous' | None (neither)."""
    if direction == "long":
        hit_stop, hit_target = bar_low <= stop, bar_high >= target
    else:
        hit_stop, hit_target = bar_high >= stop, bar_low <= target
    if hit_stop and hit_target:
        return "stop_ambiguous"
    if hit_stop:
        return "stop"
    if hit_target:
        return "target"
    return None
