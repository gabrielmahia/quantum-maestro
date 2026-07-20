# Copyright (c) 2026 Gabriel Mahia. All Rights Reserved.
"""Broker adapter tests. The two load-bearing guarantees:
1. Orders without an approved RiskVerdict are REFUSED (engine in the call path).
2. The endpoint is pinned to sandbox — no production URL exists in the module."""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from qm.broker_tradier import TradierSandbox, SANDBOX_BASE, OrderNotApprovedError
from qm.risk import RiskVerdict
import qm.broker_tradier as bt


def _client():
    return TradierSandbox(token="test-token", account_id="VA000000")


def test_endpoint_is_pinned_to_sandbox():
    assert SANDBOX_BASE == "https://sandbox.tradier.com/v1"
    src = open(bt.__file__).read()
    assert "api.tradier.com" not in src, "production endpoint must not exist in the adapter"


def test_place_order_refused_without_verdict():
    c = _client()
    with pytest.raises(OrderNotApprovedError):
        c.place_order({"class": "equity", "symbol": "SPY", "side": "buy",
                       "quantity": 1, "type": "market", "duration": "day"}, verdict=None)


def test_place_order_refused_on_vetoed_verdict():
    c = _client()
    v = RiskVerdict(approved=False, vetoes=["[DTE] 0 DTE banned"])
    with pytest.raises(OrderNotApprovedError):
        c.place_order({"class": "equity", "symbol": "SPY", "side": "buy",
                       "quantity": 1, "type": "market", "duration": "day"}, verdict=v)


def test_approved_verdict_reaches_http_layer(monkeypatch):
    c = _client()
    sent = {}

    def fake_post(path, data):
        sent["path"], sent["data"] = path, data
        return {"order": {"id": 1, "status": "ok"}}

    monkeypatch.setattr(c, "_post", fake_post)
    v = RiskVerdict(approved=True)
    out = c.place_order({"class": "equity", "symbol": "SPY", "side": "buy",
                         "quantity": 1, "type": "market", "duration": "day"}, verdict=v)
    assert out["order"]["status"] == "ok"
    assert sent["path"] == "/accounts/VA000000/orders"
    assert "preview" not in sent["data"]


def test_preview_sets_preview_flag(monkeypatch):
    c = _client()
    sent = {}
    monkeypatch.setattr(c, "_post", lambda p, d: sent.update(d) or {"order": {}})
    c.preview_order({"class": "equity", "symbol": "SPY", "side": "buy",
                     "quantity": 1, "type": "market", "duration": "day"})
    assert sent.get("preview") == "true"


def test_multileg_payload_flattening(monkeypatch):
    c = _client()
    sent = {}
    monkeypatch.setattr(c, "_post", lambda p, d: sent.update(d) or {"order": {}})
    order = {"class": "multileg", "symbol": "SPY", "type": "debit", "duration": "day", "price": 1.25,
             "legs": [{"option_symbol": "SPY260821C00650000", "side": "buy_to_open", "quantity": 1},
                      {"option_symbol": "SPY260821C00660000", "side": "sell_to_open", "quantity": 1}]}
    c.preview_order(order)
    assert sent["option_symbol[0]"] == "SPY260821C00650000"
    assert sent["side[1]"] == "sell_to_open"
    assert "legs" not in sent


def test_missing_token_raises_helpful_error(monkeypatch):
    monkeypatch.delenv("TRADIER_SANDBOX_TOKEN", raising=False)
    monkeypatch.delenv("TRADIER_TOKEN", raising=False)
    from qm.broker_tradier import TradierAuthError
    with pytest.raises(TradierAuthError):
        TradierSandbox(token=None, account_id=None)


def test_tradier_env_production_is_refused(monkeypatch):
    from qm.broker_tradier import TradierAuthError
    monkeypatch.setenv("TRADIER_ENV", "production")
    with pytest.raises(TradierAuthError):
        TradierSandbox(token="x", account_id="VA000000")
    monkeypatch.setenv("TRADIER_ENV", "sandbox")
    TradierSandbox(token="x", account_id="VA000000")  # sandbox passes
