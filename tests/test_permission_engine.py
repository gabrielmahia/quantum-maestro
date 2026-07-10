import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from quantum_maestro.execution.order_models import build_vertical_credit_spread, OrderIntent, OrderClass, Broker, ApprovalMode
from quantum_maestro.risk.permission_engine import evaluate, AccountState, MarketState, SessionFlags


def _intent(iwt=7):
    return build_vertical_credit_spread("SPX", "SPXW260717P06100000", "SPXW260717P06095000",
                                        1, 1.05, 5.0, "T", "test thesis", iwt_score=iwt)

def _acct(equity=100000):
    return AccountState(equity=equity, cash=equity*0.6, open_positions=0,
                        day_trades_remaining=3, open_risk_dollars=0)

def _mkt(**kw):
    d = dict(market_open=True, minutes_to_tier1_event=None, data_age_seconds=5, bid_ask_spread_pct=2.0)
    d.update(kw); return MarketState(**d)


def test_max_loss_math():
    assert _intent().max_loss == 395.0

def test_green_path():
    assert evaluate(_intent(), _acct(), _mkt(), SessionFlags(paper_only=False)).color == "GREEN"

def test_event_blackout_is_hard():
    v = evaluate(_intent(), _acct(), _mkt(minutes_to_tier1_event=15), SessionFlags(paper_only=False))
    assert v.color == "RED"

def test_impaired_blocks_submission():
    v = evaluate(_intent(), _acct(), _mkt(), SessionFlags(paper_only=False, human_impaired=True))
    assert v.color == "RED"

def test_small_account_size_gate():
    v = evaluate(_intent(), _acct(equity=5000), _mkt(), SessionFlags(paper_only=False))
    assert v.color == "RED"

def test_no_exit_plan_invalid():
    i = _intent(); i.profit_target = i.stop_trigger = i.time_stop = None
    assert any("exit plan" in e for e in i.validate())

def test_low_iwt_score_hard_fail():
    v = evaluate(_intent(iwt=4), _acct(), _mkt(), SessionFlags(paper_only=False))
    assert v.color == "RED"

def test_paper_only_blocks_live_broker():
    v = evaluate(_intent(), _acct(), _mkt(), SessionFlags(paper_only=True))
    assert v.color == "RED"

def test_approved_flag_set_only_on_green():
    i = _intent()
    evaluate(i, _acct(equity=5000), _mkt(), SessionFlags(paper_only=False))
    assert i.risk_approved is False
