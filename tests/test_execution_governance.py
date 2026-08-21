# Copyright (c) 2026 Gabriel Mahia. All Rights Reserved.
"""Tests for qm.execution_governance — spec sections 14, 18, 20, 22."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qm.execution_governance import (
    route_strategy, TradeIntent, validate_intent,
    TranchePlan, may_add_tranche,
    ShortLegCycle, ManagedVerticalLedger,
)


# ---------------- §14 account routing ----------------

def test_fidelity_cannot_trade_spreads():
    r = route_strategy("BEARISH", "FIDELITY", iv_rich=True, confirmed=True)
    assert r["strategy"] == "long_put"  # falls through to what's permitted
    assert any("not permitted" in why for _, why in r["rejected"])


def test_tos_prefers_debit_spread():
    r = route_strategy("BEARISH", "TOS")
    assert r["strategy"] == "bear_put_debit"


def test_credit_requires_rich_iv_and_confirmation():
    # not confirmed -> credit rejected, falls to debit
    r = route_strategy("BULLISH", "TOS", iv_rich=True, confirmed=False)
    assert r["strategy"] == "bull_call_debit"


def test_event_heavy_avoids_credit():
    r = route_strategy("BULLISH", "TOS", iv_rich=True, confirmed=True, event_heavy=True)
    assert r["strategy"] in ("bull_call_debit", "long_call")


def test_neutral_routes_to_cash():
    r = route_strategy("NEUTRAL", "TOS")
    assert r["strategy"] == "cash"


# ---------------- §18 intent vs order ----------------

def test_bearish_thesis_with_bullish_order_rejected():
    intent = TradeIntent(thesis="BEARISH", instrument="SPX", strategy="bear_put_debit",
                         expiration="2026-09-04", net_type="DEBIT", max_loss=2100,
                         long_strike=7800, short_strike=7750)
    bad_order = {"strategy": "bull_put_credit", "net_type": "CREDIT",
                 "expiration": "2026-09-04", "max_loss": 2100}
    r = validate_intent(intent, bad_order)
    assert not r["valid"]
    assert any("contradicts thesis" in m or "!= intent" in m for m in r["unwaived"])


def test_matching_order_valid():
    intent = TradeIntent(thesis="BEARISH", instrument="SPX", strategy="bear_put_debit",
                         expiration="2026-09-04", net_type="DEBIT", max_loss=2100,
                         long_strike=7800, short_strike=7750)
    good = {"strategy": "bear_put_debit", "net_type": "DEBIT",
            "expiration": "2026-09-04", "long_strike": 7800,
            "short_strike": 7750, "max_loss": 2100}
    assert validate_intent(intent, good)["valid"]


def test_order_exceeding_max_loss_rejected():
    intent = TradeIntent(thesis="BULLISH", instrument="SPY", strategy="bull_call_debit",
                         expiration="2026-09-04", net_type="DEBIT", max_loss=500)
    over = {"strategy": "bull_call_debit", "net_type": "DEBIT",
            "expiration": "2026-09-04", "max_loss": 900}
    r = validate_intent(intent, over)
    assert not r["valid"] and any("EXCEEDS" in m for m in r["unwaived"])


def test_expiration_mismatch_rejected():
    intent = TradeIntent(thesis="BULLISH", instrument="SPY", strategy="long_call",
                         expiration="2026-09-04", net_type="DEBIT", max_loss=300)
    wrong = {"strategy": "long_call", "net_type": "DEBIT", "expiration": "2026-10-16"}
    assert not validate_intent(intent, wrong)["valid"]


def test_intent_infers_exposure():
    i = TradeIntent(thesis="BEARISH", instrument="X", strategy="bear_put_debit",
                    expiration="e", net_type="DEBIT", max_loss=1)
    assert i.expected_exposure == "NEGATIVE_DELTA"


# ---------------- §20 tranche model ----------------

def test_tranche_starts_at_quarter():
    p = TranchePlan(permitted_total=40).build()
    assert p["initial"] == 10
    assert sum(a["size"] for a in p["add_ons"]) == 30


def test_tranche_minimum_one():
    p = TranchePlan(permitted_total=2).build()
    assert p["initial"] >= 1


def test_tranche_zero_blocked():
    assert "error" in TranchePlan(permitted_total=0).build()


def test_rescue_add_blocked():
    r = may_add_tranche(thesis_intact=True, new_information=False, position_underwater=True)
    assert not r["allowed"] and "rescue" in r["reason"]


def test_add_blocked_when_thesis_dead():
    r = may_add_tranche(thesis_intact=False, new_information=True, position_underwater=False)
    assert not r["allowed"]


def test_legitimate_add_allowed():
    r = may_add_tranche(thesis_intact=True, new_information=True, position_underwater=False)
    assert r["allowed"]


# ---------------- §22 recycling accounting ----------------

def test_cycle_gross_and_net():
    c = ShortLegCycle(sto_price=2.50, btc_price=0.15, contracts=1, fees=2.0, slippage=1.0)
    assert abs(c.gross_pl() - 235.0) < 1e-9   # (2.50-0.15)*100 = 235, matches course example
    assert abs(c.net_pl() - 232.0) < 1e-9


def test_management_alpha_needs_two_cycles():
    led = ManagedVerticalLedger(long_hedge_cost=3.00)
    led.add_cycle(ShortLegCycle(2.50, 0.15))
    assert led.management_alpha()["re_entries"] == 0


def test_management_alpha_measures_reentry():
    led = ManagedVerticalLedger(long_hedge_cost=3.00)
    led.add_cycle(ShortLegCycle(2.50, 0.50, fees=1.0))       # closed at 0.50
    led.add_cycle(ShortLegCycle(1.80, 0.20, fees=1.0))       # resold at 1.80
    ma = led.management_alpha()
    # improvement = (1.80 - 0.50)*100 = 130, minus 1.0 friction
    assert abs(ma["re_entry_improvement"] - 130.0) < 1e-9
    assert abs(ma["management_alpha"] - 129.0) < 1e-9
    assert ma["re_entries"] == 1


def test_hedge_funding_status():
    led = ManagedVerticalLedger(long_hedge_cost=3.00)   # cost 300
    led.add_cycle(ShortLegCycle(2.50, 0.15))            # +235
    h = led.hedge_status()
    assert h["long_hedge_original_cost"] == 300.0
    assert not h["hedge_fully_funded"]
    led.add_cycle(ShortLegCycle(1.00, 0.10))            # +90 -> 325 total
    assert led.hedge_status()["hedge_fully_funded"]


def test_summary_carries_discipline_rule():
    led = ManagedVerticalLedger(long_hedge_cost=1.0)
    led.add_cycle(ShortLegCycle(1.0, 0.1))
    assert "requalify" in led.summary()["discipline_rule"]
