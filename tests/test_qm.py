# Copyright (c) 2026 Gabriel Mahia. All Rights Reserved.
"""Smoke tests for the Quantum Maestro decision engine (qm package).
Run by CI on every push: the risk engine's vetoes are load-bearing."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["QM_DB_PATH"] = os.path.join(tempfile.mkdtemp(), "journal.db")

from qm.regime import RegimeInputs, score_regime
from qm.risk import TradeProposal, evaluate
from qm.sizing import final_size, kelly_fraction
from qm.agents import aggregate, select_strategy
from qm import journal


def test_regime_defensive_under_stress():
    r = score_regime(RegimeInputs(28.0, 25.0, -3.0, -2.5, 9.0, 1, True, True, True))
    assert r["regime"] in ("DEFENSIVE", "LOCKDOWN")


def test_regime_offensive_when_calm():
    r = score_regime(RegimeInputs(13.0, -10.0, 2.5, 1.5, 0.0, 0, False, False, False))
    assert r["regime"] == "OFFENSIVE"


def test_zero_dte_is_vetoed():
    p = TradeProposal(underlying="AAPL", direction="SHORT", structure="Call Credit Spread",
                      is_short_premium=True, dte=0, max_loss_per_unit=200, max_gain_per_unit=60,
                      account_equity=10000, proposed_risk_dollars=50, regime="NEUTRAL", thesis="x" * 30)
    assert not evaluate(p).approved


def test_repair_trade_is_vetoed():
    p = TradeProposal(underlying="AAPL", direction="LONG", structure="Put Credit Spread",
                      is_short_premium=True, dte=14, max_loss_per_unit=200, max_gain_per_unit=60,
                      account_equity=10000, proposed_risk_dollars=50,
                      has_open_losing_position_same_underlying=True, regime="NEUTRAL", thesis="x" * 30)
    assert not evaluate(p).approved


def test_good_trade_approved():
    p = TradeProposal(underlying="SPY", direction="LONG", structure="Bull Call Spread",
                      is_short_premium=False, dte=30, max_loss_per_unit=150, max_gain_per_unit=350,
                      account_equity=10000, proposed_risk_dollars=60, regime="NEUTRAL",
                      thesis="Held 50DMA on volume; invalidation below prior swing low.")
    v = evaluate(p)
    assert v.approved and v.adjusted_max_risk_dollars == 60.0


def test_circuit_breaker():
    p = TradeProposal(underlying="SPY", direction="LONG", structure="Shares", is_option=False,
                      max_loss_per_unit=1, max_gain_per_unit=3, account_equity=10000,
                      proposed_risk_dollars=50, todays_pnl_pct=-2.5, regime="NEUTRAL", thesis="x" * 30)
    assert not evaluate(p).approved


def test_single_agent_veto_kills_trade():
    a = aggregate({"Macro": "VETO", "Flows": "STRONG", "Technical": "STRONG",
                   "Options": "STRONG", "Psychology": "STRONG", "Portfolio": "STRONG"})
    assert not a["passed"]


def test_iron_condor_banned_in_defensive():
    s = select_strategy("NEUTRAL", "RICH", "DEFENSIVE")
    assert s["structure"] == "No trade"


def test_kelly_negative_means_no_edge():
    k = kelly_fraction(0.40, 1.0)
    assert k["edge_exists"] is False and k["fractional_kelly"] == 0.0


def test_defensive_sizer_refuses_unaffordable_contract():
    fs = final_size(10000, "DEFENSIVE", max_loss_per_contract=150)
    assert fs["fixed_risk"]["contracts"] == 0


def test_journal_roundtrip_and_gate_locked():
    i = journal.log_decision(mode="SHADOW", decision_type="TRADE", underlying="SPY",
                             structure="BCS", direction="LONG", regime="NEUTRAL",
                             regime_score=1, thesis="t", risk_dollars=60, status="OPEN")
    journal.close_trade(i, 5.0, 1.8, "worked")
    s = journal.stats()
    assert s["closed_trades"] >= 1 and s["expectancy_r"] > 0
    assert journal.evaluate_gate()["promotable"] is False  # small sample must never promote


def test_cohort_sizing_combines_multiplicatively():
    from qm.sizing import final_size
    # 1% x regime 0.60 (NEUTRAL) x cohort 0.50 (SECONDARY) = 0.30%
    r = final_size(10000, "NEUTRAL", max_loss_per_contract=150, cohort="SECONDARY")
    assert abs(r["effective_risk_pct"] - 0.003) < 1e-9
    assert r["combined_multiplier"] == 0.3


def test_rejected_cohort_is_zero_size():
    from qm.sizing import final_size
    r = final_size(10000, "OFFENSIVE", max_loss_per_contract=150, cohort="REJECTED")
    assert r["effective_risk_pct"] == 0.0


def test_correlation_multiplier_clamped():
    from qm.sizing import final_size
    r = final_size(10000, "OFFENSIVE", max_loss_per_contract=150,
                   cohort="PRIMARY", correlation_multiplier=1.5)  # >1 clamps to 1
    assert r["correlation_multiplier"] == 1.0


def test_kelly_single_cap_no_dead_var():
    from qm.sizing import kelly_fraction
    k = kelly_fraction(0.55, 2.0)
    assert "fractional_kelly" in k and "quarter_kelly" not in k
    assert k["cap_used"] == 0.25
    assert abs(k["fractional_kelly"] - k["full_kelly"] * 0.25) < 0.001
