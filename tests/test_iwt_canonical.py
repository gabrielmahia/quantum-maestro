# Copyright (c) 2026 Gabriel Mahia. All Rights Reserved.
"""Tests for qm.iwt_canonical — validated against Teri's course documents."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from qm.iwt_canonical import (
    IWTRiskPlan, check_risk_plan, iwt_long_trade, iwt_short_trade,
    PreTradeWorksheet, QM_DAILY_BREAKER_PCT,
)


def test_canonical_long_matches_spreadsheet_V():
    # RR_Excel_Sheet LONG, symbol V: ATR 9.07, distal_BZ 179.23, proximal_BZ 184.87,
    # proximal_SZ 202.82 -> stop 177.416, risk 7.454, reward 17.95, RR 2.408
    r = iwt_long_trade(distal_bz=179.23, proximal_bz=184.87, proximal_sz=202.82, atr=9.07)
    assert abs(r["stop"] - 177.42) < 0.01
    assert abs(r["risk"] - 7.454) < 0.01
    assert abs(r["reward"] - 17.95) < 0.01
    assert abs(r["reward_risk"] - 2.408) < 0.01
    # shares = 1000 / 7.454 = 134.16
    assert abs(r["shares"] - 134.16) < 0.1


def test_canonical_short_geometry():
    r = iwt_short_trade(distal_sz=152.62, proximal_sz=149.74, proximal_bz=140.0, atr=4.58)
    # stop = 152.62 + 0.2*4.58 = 153.536
    assert abs(r["stop"] - 153.536) < 0.01
    assert r["risk"] > 0 and r["reward"] > 0


def test_risk_plan_percentages_are_canonical():
    p = IWTRiskPlan()
    # Teri Stocks/Options on $100k: 1/3/5/10 %
    assert p.per_trade_pct == 0.010
    assert p.daily_loss_pct == 0.030
    assert p.weekly_loss_pct == 0.050
    assert p.monthly_loss_pct == 0.100
    assert (p.max_trades_per_day, p.max_trades_per_week, p.max_trades_per_month) == (2, 8, 20)


def test_daily_ceiling_uses_stricter_qm_number():
    # QM 2% is stricter than Teri 3% -> ceiling should be 2% of equity
    r = check_risk_plan(100_000, realized_pnl_today=-2100, realized_pnl_week=0,
                        realized_pnl_month=0, trades_today=0, trades_week=0,
                        trades_month=0, proposed_trade_risk=500)
    assert r["ceilings"]["daily"] == 2000.0  # 2% not 3%
    assert not r["allowed"]  # -2100 already past the 2000 ceiling
    assert any("daily loss ceiling" in b for b in r["breaches"])


def test_trade_count_caps_block():
    r = check_risk_plan(100_000, 0, 0, 0, trades_today=2, trades_week=3,
                        trades_month=5, proposed_trade_risk=500)
    assert not r["allowed"]
    assert any("max trades/day" in b for b in r["breaches"])


def test_projected_loss_would_breach_daily():
    # already down 1600, proposing a trade risking 500 -> 2100 > 2000 ceiling
    r = check_risk_plan(100_000, realized_pnl_today=-1600, realized_pnl_week=0,
                        realized_pnl_month=0, trades_today=0, trades_week=0,
                        trades_month=0, proposed_trade_risk=500)
    assert not r["allowed"]
    assert any("push daily loss past ceiling" in b for b in r["breaches"])


def test_clean_trade_allowed():
    r = check_risk_plan(100_000, 0, 0, 0, 0, 0, 0, proposed_trade_risk=1000)
    assert r["allowed"] and r["breaches"] == []


def test_worksheet_volume_block():
    w = PreTradeWorksheet("ABC", volume=500_000, up_trend=True)
    out = w.evaluate()
    assert not out["ready_to_chart"]
    assert any("liquidity minimum" in b for b in out["hard_blocks"])


def test_worksheet_earnings_block():
    w = PreTradeWorksheet("ABC", volume=2_000_000, days_to_earnings=3)
    out = w.evaluate()
    assert not out["ready_to_chart"]
    assert any("earnings" in b for b in out["hard_blocks"])


def test_worksheet_ready_with_warnings():
    w = PreTradeWorksheet("ABC", volume=5_000_000, up_trend=True, price=95,
                          high_52w=100, low_52w=40, best_in_breed=False)
    out = w.evaluate()
    assert out["ready_to_chart"]  # no hard blocks
    assert any("best-in-breed" in x for x in out["warnings"])
    assert out["range_pos_52w"] is not None


def test_canonical_order_types_and_round_down():
    r = iwt_long_trade(distal_bz=179.23, proximal_bz=184.87, proximal_sz=202.82, atr=9.07)
    # Worksheet: entry & exit are LIMIT, stop is STOP-MARKET
    assert r["order_types"] == {"entry": "limit", "exit": "limit", "stop": "stop_market"}
    # shares round DOWN (134.16 -> 134)
    assert r["shares_rounded_down"] == 134
    s = iwt_short_trade(distal_sz=152.62, proximal_sz=149.74, proximal_bz=140.0, atr=4.58)
    assert s["order_types"]["stop"] == "stop_market"
