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


def odds_enhancer(base_candles: int, departure_strength_atr: float,
                  prior_visits: int, reward_risk: float) -> dict:
    """The full 0-8 score plus its parts and the cohort it belongs to."""
    parts = {
        "base": base_candle_score(base_candles),
        "departure": departure_score(departure_strength_atr),
        "freshness": freshness_score(prior_visits),
        "reward_risk": reward_risk_score(reward_risk),
    }
    total = sum(parts.values())
    if total >= 7:
        cohort, size_mult, action = "PRIMARY", 1.0, "eligible (limit at proximal)"
    elif total >= 5:
        cohort, size_mult, action = "SECONDARY", 0.5, "conditional (require confirmation, half size)"
    else:
        cohort, size_mult, action = "REJECTED", 0.0, "skip"
    return {"score": total, "parts": parts, "cohort": cohort,
            "size_multiplier": size_mult, "action": action}


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
