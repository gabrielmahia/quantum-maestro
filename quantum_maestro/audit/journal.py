"""Append-only decision journal (JSONL). Every intent, verdict, and outcome
gets a line — including trades NOT taken. Shadow mode is built on this:
log what the system would have done, then grade it against what happened.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional

from quantum_maestro.execution.order_models import OrderIntent
from quantum_maestro.risk.permission_engine import Verdict

DEFAULT_PATH = os.getenv("QM_JOURNAL_PATH", "journal/decisions.jsonl")


def _write(path: str, record: dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    record["logged_at"] = datetime.now(timezone.utc).isoformat()
    with open(path, "a") as f:
        f.write(json.dumps(record, default=str) + "\n")


def log_decision(intent: OrderIntent, verdict: Verdict,
                 mode: str = "shadow", path: str = DEFAULT_PATH) -> None:
    _write(path, {
        "kind": "decision",
        "mode": mode,                      # shadow | sandbox | live
        "intent": intent.to_dict(),
        "verdict": {"color": verdict.color,
                    "hard_failures": verdict.hard_failures,
                    "soft_warnings": verdict.soft_warnings},
    })


def log_no_trade(reason: str, context: Optional[dict] = None,
                 path: str = DEFAULT_PATH) -> None:
    """'No trade' is a decision too — journal it so discipline is measurable."""
    _write(path, {"kind": "no_trade", "reason": reason, "context": context or {}})


def log_outcome(intent_id: str, outcome: dict, path: str = DEFAULT_PATH) -> None:
    """Fill, exit, P&L, or shadow-mode hypothetical result for a prior intent."""
    _write(path, {"kind": "outcome", "intent_id": intent_id, "outcome": outcome})
