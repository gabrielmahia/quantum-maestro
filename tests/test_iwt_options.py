# Copyright (c) 2026 Gabriel Mahia. All Rights Reserved.
"""Tests for qm.iwt_options — validated against the IWT options course PDF."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qm.iwt_options import (
    EXPRESSIONS, break_even, choose_expression, VerticalSpread, validate_options_trade,
)


def test_expression_zone_mapping():
    assert EXPRESSIONS["buy_call"]["zone"] == "buyer"
    assert EXPRESSIONS["sell_put"]["zone"] == "buyer"
    assert EXPRESSIONS["buy_put"]["zone"] == "seller"
    assert EXPRESSIONS["sell_call"]["zone"] == "seller"


def test_break_even_formulas():
    # From the PDF: buy call BE = strike + premium; sell put BE = strike - premium
    assert break_even("buy_call", 100, 2.5) == 102.5
    assert break_even("sell_put", 100, 2.5) == 97.5
    assert break_even("buy_put", 100, 2.5) == 97.5
    assert break_even("sell_call", 100, 2.5) == 102.5


def test_choose_expression_iv_doctrine():
    # buyer zone, cheap IV -> debit (buy call)
    r = choose_expression("buyer", iv_position=15)
    assert r["debit_expression"] == "buy_call" and r["favored_flow"] == "debit"
    # buyer zone, rich IV -> credit (sell put, defined-risk)
    r2 = choose_expression("buyer", iv_position=75)
    assert r2["credit_expression"] == "sell_put" and r2["favored_flow"] == "credit"
    # seller zone
    r3 = choose_expression("seller", iv_position=80)
    assert r3["debit_expression"] == "buy_put" and r3["credit_expression"] == "sell_call"


def test_vertical_put_credit_economics():
    # width 5, credit 1.50 -> max profit 150, max loss 350, on 1 contract
    v = VerticalSpread("put_credit", short_strike=100, long_strike=95, net_premium=1.50)
    e = v.economics()
    assert e["max_profit"] == 150.0
    assert e["max_loss"] == 350.0
    assert e["width"] == 5.0


def test_vertical_debit_economics():
    v = VerticalSpread("call_debit", short_strike=105, long_strike=100, net_premium=2.0)
    e = v.economics()
    # debit: max loss = premium*100 = 200; max profit = (5-2)*100 = 300
    assert e["max_loss"] == 200.0
    assert e["max_profit"] == 300.0


def test_naked_short_blocked():
    r = validate_options_trade("sell_put", defined_risk=False, dte=30, into_event=False)
    assert not r["allowed"]
    assert any("naked short" in b for b in r["blocks"])


def test_defined_risk_credit_allowed():
    r = validate_options_trade("sell_put", defined_risk=True, dte=30, into_event=False)
    assert r["allowed"]


def test_zero_dte_blocked():
    r = validate_options_trade("buy_call", defined_risk=False, dte=0, into_event=False)
    assert not r["allowed"]
    assert any("0DTE" in b or "DTE" in b for b in r["blocks"])


def test_credit_into_event_blocked():
    r = validate_options_trade("sell_call", defined_risk=True, dte=30, into_event=True)
    assert not r["allowed"]
    assert any("event" in b for b in r["blocks"])


def test_debit_into_event_allowed():
    # buying (debit) into an event is allowed by doctrine (long defined risk)
    r = validate_options_trade("buy_put", defined_risk=False, dte=30, into_event=True)
    assert r["allowed"]
