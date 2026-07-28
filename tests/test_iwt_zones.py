# Copyright (c) 2026 Gabriel Mahia. All Rights Reserved.
"""Tests for the IWT zone-scoring / boundary-survival research module."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qm.iwt_zones import (
    base_candle_score, departure_score, freshness_score, reward_risk_score,
    odds_enhancer, ZoneTrade, frame_trade, boundary_survival, resolve_daily_outcome,
)


def test_base_candle_score():
    assert base_candle_score(1) == 2 and base_candle_score(2) == 2
    assert base_candle_score(3) == 1 and base_candle_score(4) == 1
    assert base_candle_score(5) == 0


def test_departure_score():
    assert departure_score(1.6) == 2
    assert departure_score(0.9) == 1
    assert departure_score(0.5) == 0


def test_freshness_excludes_current_touch():
    # prior_visits is BEFORE the entry touch — look-ahead control
    assert freshness_score(0) == 2 and freshness_score(1) == 1 and freshness_score(2) == 0


def test_reward_risk_score():
    assert reward_risk_score(3.5) == 2 and reward_risk_score(2.5) == 1 and reward_risk_score(1.5) == 0


def test_odds_enhancer_cohorts():
    top = odds_enhancer(base_candles=2, departure_strength_atr=1.6, prior_visits=0, reward_risk=3.2)
    assert top["score"] == 8 and top["cohort"] == "PRIMARY" and top["size_multiplier"] == 1.0
    mid = odds_enhancer(base_candles=3, departure_strength_atr=0.8, prior_visits=1, reward_risk=2.2)
    assert mid["score"] == 4  # 1+1+1+1 -> actually rejected boundary; verify:
    # base3=1, dep0.8=1, fresh1=1, rr2.2=1 => 4 => REJECTED
    assert mid["cohort"] == "REJECTED"
    sec = odds_enhancer(base_candles=2, departure_strength_atr=0.8, prior_visits=0, reward_risk=2.2)
    assert sec["score"] == 6 and sec["cohort"] == "SECONDARY" and sec["size_multiplier"] == 0.5


def test_zone_trade_geometry_long():
    t = frame_trade("long", proximal=100, distal=98, target=110, atr=1.0, atr_buffer=0.2)
    assert t.stop == 98 - 0.2  # 97.8
    assert t.is_valid()
    assert abs(t.reward_risk - (10 / 2.2)) < 1e-9


def test_zone_trade_geometry_short():
    t = frame_trade("short", proximal=100, distal=102, target=90, atr=1.0, atr_buffer=0.2)
    assert t.stop == 102 + 0.2
    assert t.is_valid()


def test_invalid_geometry_rejected():
    bad = ZoneTrade("long", entry=100, stop=101, target=110)  # stop above entry on a long
    assert not bad.is_valid()


def test_boundary_survival_downside_breach():
    # buyer zone, distal 98, atr 1, buffer 0.2 -> boundary 97.8
    r = boundary_survival("buyer", distal=98, atr=1.0,
                          forward_lows=[99, 98.5, 97.0], forward_highs=[101, 100, 99],
                          forward_closes=[100, 99, 97.5], horizon_days=90, atr_buffer=0.2)
    assert r.ever_breached is True   # 97.0 <= 97.8
    assert r.terminal_breach is True  # closes at 97.5 <= 97.8


def test_boundary_survival_holds():
    r = boundary_survival("buyer", distal=98, atr=1.0,
                          forward_lows=[99, 98.5, 98.2], forward_highs=[101, 100, 100],
                          forward_closes=[100, 99, 100], horizon_days=90, atr_buffer=0.2)
    assert r.ever_breached is False and r.terminal_breach is False


def test_daily_ambiguity_is_stop_first():
    # bar spans both stop and target -> conservative stop_ambiguous
    assert resolve_daily_outcome("long", stop=98, target=110, bar_low=97, bar_high=111) == "stop_ambiguous"
    assert resolve_daily_outcome("long", stop=98, target=110, bar_low=99, bar_high=111) == "target"
    assert resolve_daily_outcome("long", stop=98, target=110, bar_low=97, bar_high=105) == "stop"
    assert resolve_daily_outcome("long", stop=98, target=110, bar_low=99, bar_high=105) is None
