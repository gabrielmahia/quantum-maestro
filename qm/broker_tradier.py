"""
Tradier SANDBOX Broker Adapter — Paper Desk
===========================================
Hard constraints, by construction:

1. ENDPOINT IS PINNED to https://sandbox.tradier.com — the production host
   does not appear anywhere in this module (CI-enforced). Paper trading only.
2. TOKEN COMES FROM SECRETS ONLY (Streamlit secrets or environment), never
   from code or UI input. Rotating the key never touches the repo.
3. ORDERS REQUIRE AN APPROVED RiskVerdict. place_order() raises if the
   deterministic risk engine has not approved the proposal. The engine is
   not advisory here — it is physically in the call path.

Sandbox notes (per Tradier): $100k virtual funds, 15-minute delayed data,
no streaming, limited account activity endpoints.
"""

import os
import requests

SANDBOX_BASE = "https://sandbox.tradier.com/v1"  # pinned; do not parameterize


class TradierAuthError(RuntimeError):
    pass


class OrderNotApprovedError(RuntimeError):
    pass


def _get_secret(name: str):
    """Streamlit secrets first (several accepted shapes), then environment."""
    try:
        import streamlit as st
        if name in st.secrets:
            return st.secrets[name]
        if "tradier" in st.secrets and name.lower().replace("tradier_", "") in st.secrets["tradier"]:
            return st.secrets["tradier"][name.lower().replace("tradier_", "")]
    except Exception:
        pass
    return os.environ.get(name)


class TradierSandbox:
    def __init__(self, token: str = None, account_id: str = None):
        self.token = token or _get_secret("TRADIER_SANDBOX_TOKEN") or _get_secret("TRADIER_TOKEN")
        if not self.token:
            raise TradierAuthError(
                "No sandbox token found. Add TRADIER_SANDBOX_TOKEN to Streamlit secrets "
                "(App → Settings → Secrets) or environment. Never hardcode it."
            )
        self.account_id = account_id or _get_secret("TRADIER_SANDBOX_ACCOUNT")
        self._s = requests.Session()
        self._s.headers.update({"Authorization": f"Bearer {self.token}",
                                "Accept": "application/json"})

    # ------------------------------------------------------------------ http
    def _get(self, path: str, **params):
        r = self._s.get(f"{SANDBOX_BASE}{path}", params=params, timeout=15)
        if r.status_code == 401:
            raise TradierAuthError("Sandbox rejected the token (401). If you regenerated the key, "
                                   "update the Streamlit secret. Production keys do not work here — "
                                   "this adapter only speaks to sandbox.tradier.com.")
        r.raise_for_status()
        return r.json()

    def _post(self, path: str, data: dict):
        r = self._s.post(f"{SANDBOX_BASE}{path}", data=data, timeout=15)
        if r.status_code == 401:
            raise TradierAuthError("Sandbox rejected the token (401).")
        r.raise_for_status()
        return r.json()

    # ------------------------------------------------------------- read-only
    def profile(self):
        return self._get("/user/profile")

    def resolve_account(self) -> str:
        if self.account_id:
            return self.account_id
        prof = self.profile()
        acct = prof.get("profile", {}).get("account")
        if isinstance(acct, list):
            acct = acct[0]
        self.account_id = acct.get("account_number") if acct else None
        if not self.account_id:
            raise TradierAuthError("Could not resolve sandbox account number from profile.")
        return self.account_id

    def balances(self):
        return self._get(f"/accounts/{self.resolve_account()}/balances").get("balances", {})

    def positions(self):
        out = self._get(f"/accounts/{self.resolve_account()}/positions").get("positions")
        if not out or out == "null":
            return []
        pos = out.get("position", [])
        return pos if isinstance(pos, list) else [pos]

    def orders(self):
        out = self._get(f"/accounts/{self.resolve_account()}/orders").get("orders")
        if not out or out == "null":
            return []
        o = out.get("order", [])
        return o if isinstance(o, list) else [o]

    def quotes(self, symbols):
        if isinstance(symbols, (list, tuple)):
            symbols = ",".join(symbols)
        q = self._get("/markets/quotes", symbols=symbols).get("quotes", {}).get("quote", [])
        return q if isinstance(q, list) else [q]

    def expirations(self, symbol: str):
        e = self._get("/markets/options/expirations", symbol=symbol, includeAllRoots="true")
        ex = e.get("expirations") or {}
        d = ex.get("date", [])
        return d if isinstance(d, list) else [d]

    def option_chain(self, symbol: str, expiration: str):
        c = self._get("/markets/options/chains", symbol=symbol, expiration=expiration, greeks="true")
        opts = (c.get("options") or {}).get("option", [])
        return opts if isinstance(opts, list) else [opts]

    # ------------------------------------------------- gated order path
    def _order_payload(self, order: dict) -> dict:
        """Equity: {class:'equity', symbol, side, quantity, type, duration, [price]}
        Single-leg option: {class:'option', symbol, option_symbol, side, quantity, type, duration, [price]}
        Multileg: {class:'multileg', symbol, type, duration, price, legs:[{option_symbol, side, quantity}...]}"""
        payload = {k: v for k, v in order.items() if k != "legs" and v is not None}
        for i, leg in enumerate(order.get("legs", []) or []):
            payload[f"option_symbol[{i}]"] = leg["option_symbol"]
            payload[f"side[{i}]"] = leg["side"]
            payload[f"quantity[{i}]"] = leg["quantity"]
        return payload

    def preview_order(self, order: dict):
        """Preview is ungated: pricing an order commits nothing."""
        payload = self._order_payload(order)
        payload["preview"] = "true"
        return self._post(f"/accounts/{self.resolve_account()}/orders", payload)

    def place_order(self, order: dict, verdict):
        """SUBMITS a sandbox order. Requires an APPROVED RiskVerdict — no verdict,
        no trade. This is the deterministic engine physically in the call path."""
        if verdict is None or not getattr(verdict, "approved", False):
            vetoes = getattr(verdict, "vetoes", ["no verdict supplied"])
            raise OrderNotApprovedError(
                "Risk engine has not approved this trade; order refused. Vetoes: " + "; ".join(vetoes)
            )
        return self._post(f"/accounts/{self.resolve_account()}/orders", self._order_payload(order))
