"""Tradier adapter — preview-first, idempotent, refuses unapproved intents.

Environment variables (never hardcode, never commit):
    TRADIER_ENV        = "sandbox" | "live"          (default sandbox)
    TRADIER_TOKEN      = API access token
    TRADIER_ACCOUNT_ID = account id

The adapter will raise before transmitting any OrderIntent whose
risk_approved is False. That check is duplicated here on purpose:
defense in depth against a caller skipping the permission engine.
"""
from __future__ import annotations

import os
import time
import requests

from quantum_maestro.execution.order_models import OrderIntent, OrderClass

_BASE = {
    "sandbox": "https://sandbox.tradier.com",
    "live": "https://api.tradier.com",
}


class TradierAdapter:
    def __init__(self, env: str | None = None, token: str | None = None,
                 account_id: str | None = None, timeout: int = 15):
        self.env = (env or os.getenv("TRADIER_ENV", "sandbox")).lower()
        self.token = token or os.getenv("TRADIER_TOKEN", "")
        self.account_id = account_id or os.getenv("TRADIER_ACCOUNT_ID", "")
        self.timeout = timeout
        self._submitted_intent_ids: set[str] = set()   # duplicate-order lock (per process)

    # ---------------------------------------------------------------- plumbing
    @property
    def base(self) -> str:
        return _BASE[self.env]

    def _headers(self) -> dict:
        return {"Authorization": f"Bearer {self.token}", "Accept": "application/json"}

    def _get(self, path: str, params: dict | None = None) -> dict:
        r = requests.get(f"{self.base}{path}", headers=self._headers(),
                         params=params or {}, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, data: dict) -> dict:
        r = requests.post(f"{self.base}{path}", headers=self._headers(),
                          data=data, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    # ---------------------------------------------------------------- reads
    def balances(self) -> dict:
        return self._get(f"/v1/accounts/{self.account_id}/balances")

    def positions(self) -> dict:
        return self._get(f"/v1/accounts/{self.account_id}/positions")

    def orders(self) -> dict:
        return self._get(f"/v1/accounts/{self.account_id}/orders")

    def quote(self, symbol: str) -> dict:
        return self._get("/v1/markets/quotes", {"symbols": symbol, "greeks": "false"})

    def option_chain(self, symbol: str, expiration: str) -> dict:
        return self._get("/v1/markets/options/chains",
                         {"symbol": symbol, "expiration": expiration, "greeks": "true"})

    def market_clock(self) -> dict:
        return self._get("/v1/markets/clock")

    # ---------------------------------------------------------------- writes
    def _intent_to_form(self, intent: OrderIntent, preview: bool) -> dict:
        if intent.order_class == OrderClass.MULTILEG:
            form = {
                "class": "multileg",
                "symbol": intent.symbol,
                "type": "credit" if intent.limit_price and intent.limit_price > 0 else "debit",
                "duration": "day",
                "price": f"{abs(intent.limit_price):.2f}",
                "preview": "true" if preview else "false",
            }
            for i, leg in enumerate(intent.legs):
                form[f"side[{i}]"] = leg.side.value
                form[f"quantity[{i}]"] = str(leg.quantity)
                form[f"option_symbol[{i}]"] = leg.option_symbol
            return form
        if intent.order_class == OrderClass.EQUITY:
            return {
                "class": "equity",
                "symbol": intent.symbol,
                "side": "buy",
                "quantity": str(intent.quantity),
                "type": "limit",
                "duration": "day",
                "price": f"{intent.limit_price:.2f}",
                "preview": "true" if preview else "false",
            }
        raise NotImplementedError(f"order_class {intent.order_class} not yet supported")

    def preview(self, intent: OrderIntent) -> dict:
        """Broker-side validation. Always call before submit; never costs anything."""
        return self._post(f"/v1/accounts/{self.account_id}/orders",
                          self._intent_to_form(intent, preview=True))

    def submit(self, intent: OrderIntent) -> dict:
        # Defense in depth: refuse unapproved or duplicate intents.
        if not intent.risk_approved:
            raise PermissionError(
                f"intent {intent.intent_id} not risk-approved (verdict={intent.risk_verdict}); "
                "run quantum_maestro.risk.permission_engine.evaluate() first")
        if intent.intent_id in self._submitted_intent_ids:
            raise PermissionError(f"duplicate submission blocked for intent {intent.intent_id}")
        if self.env == "live" and os.getenv("QM_LIVE_CONFIRM") != "I_UNDERSTAND_LIVE_RISK":
            raise PermissionError(
                "live submission requires env var QM_LIVE_CONFIRM=I_UNDERSTAND_LIVE_RISK "
                "— set it deliberately, per session, never in code")
        result = self._post(f"/v1/accounts/{self.account_id}/orders",
                            self._intent_to_form(intent, preview=False))
        self._submitted_intent_ids.add(intent.intent_id)
        return result

    def cancel(self, order_id: str) -> dict:
        r = requests.delete(f"{self.base}/v1/accounts/{self.account_id}/orders/{order_id}",
                            headers=self._headers(), timeout=self.timeout)
        r.raise_for_status()
        return r.json()
