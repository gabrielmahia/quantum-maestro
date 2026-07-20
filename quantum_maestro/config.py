"""Unified credential resolution: Streamlit secrets -> environment -> none.
Never hardcode credentials; never commit them. Sandbox is the default env.
"""
from __future__ import annotations
import os


def _from_streamlit(key: str):
    try:
        import streamlit as st
        v = st.secrets.get(key)
        if v:
            return v
        return st.secrets.get("tradier", {}).get(key.replace("TRADIER_", "").lower())
    except Exception:
        return None


def get(key: str, default: str | None = None) -> str | None:
    return _from_streamlit(key) or os.getenv(key) or default


def tradier_credentials() -> dict:
    return {
        "env": (get("TRADIER_ENV", "sandbox") or "sandbox").lower(),
        "token": get("TRADIER_TOKEN"),
        "account_id": get("TRADIER_ACCOUNT_ID"),
    }
