"""
Decision Journal — the actual MVP of Quantum Maestro
====================================================
Logs EVERY decision, including NO-TRADE decisions and risk-engine vetoes.
A prevented bad trade is a positive-expectancy event; if it isn't recorded,
the system can't learn from it and the promotion gate can't count it.

SQLite for durability. Exportable to CSV. The promotion gate reads ONLY
from this table — never from memory, never from vibes.
"""

import sqlite3
import os
from datetime import datetime, timedelta
import pandas as pd
from .config import GATE

DB_PATH = os.environ.get("QM_DB_PATH", os.path.join(os.path.dirname(__file__), "..", "data", "journal.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    mode TEXT NOT NULL,               -- SHADOW / LIVE
    decision_type TEXT NOT NULL,      -- TRADE / NO_TRADE / VETOED
    underlying TEXT,
    structure TEXT,
    direction TEXT,
    regime TEXT,
    regime_score INTEGER,
    thesis TEXT,
    planned_entry REAL,
    planned_stop REAL,
    planned_target REAL,
    risk_dollars REAL,
    veto_reasons TEXT,
    agent_notes TEXT,
    status TEXT DEFAULT 'OPEN',       -- OPEN / CLOSED / N/A
    exit_price REAL,
    realized_r REAL,                  -- P&L in R units, net of modeled costs
    rule_violation INTEGER DEFAULT 0, -- 1 if a hard rule was broken (manual override etc.)
    lesson TEXT
);
"""


def _conn():
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.execute(SCHEMA)
    return c


def log_decision(**kw) -> int:
    kw.setdefault("ts", datetime.utcnow().isoformat())
    cols = ",".join(kw.keys())
    q = ",".join("?" * len(kw))
    with _conn() as c:
        cur = c.execute(f"INSERT INTO decisions ({cols}) VALUES ({q})", list(kw.values()))
        return cur.lastrowid


def close_trade(decision_id: int, exit_price: float, realized_r: float, lesson: str = ""):
    with _conn() as c:
        c.execute("UPDATE decisions SET status='CLOSED', exit_price=?, realized_r=?, lesson=? WHERE id=?",
                  (exit_price, realized_r, lesson, decision_id))


def load(limit: int = 500) -> pd.DataFrame:
    with _conn() as c:
        return pd.read_sql_query("SELECT * FROM decisions ORDER BY id DESC LIMIT ?", c, params=(limit,))


def stats() -> dict:
    df = load(limit=100000)
    closed = df[(df.decision_type == "TRADE") & (df.status == "CLOSED") & df.realized_r.notna()]
    out = {
        "decisions_logged": len(df),
        "trades_taken": int((df.decision_type == "TRADE").sum()),
        "no_trades": int((df.decision_type == "NO_TRADE").sum()),
        "vetoes": int((df.decision_type == "VETOED").sum()),
        "closed_trades": len(closed),
        "rule_violations": int(df.rule_violation.fillna(0).sum()),
    }
    if len(closed):
        wins = closed[closed.realized_r > 0]
        losses = closed[closed.realized_r <= 0]
        w = len(wins) / len(closed)
        avg_win = wins.realized_r.mean() if len(wins) else 0.0
        avg_loss = abs(losses.realized_r.mean()) if len(losses) else 0.0
        expectancy = w * avg_win - (1 - w) * avg_loss
        eq = closed.sort_values("id").realized_r.cumsum()
        dd = float((eq - eq.cummax()).min()) if len(eq) else 0.0
        out.update({
            "win_rate": round(w, 3),
            "avg_win_r": round(float(avg_win), 3),
            "avg_loss_r": round(float(avg_loss), 3),
            "expectancy_r": round(float(expectancy), 3),
            "max_drawdown_r": round(dd, 2),
            "cum_r": round(float(closed.realized_r.sum()), 2),
        })
    if len(df):
        try:
            first = pd.to_datetime(df.ts).min()
            out["calendar_days"] = (datetime.utcnow() - first.to_pydatetime().replace(tzinfo=None)).days
        except Exception:
            out["calendar_days"] = 0
    return out


def evaluate_gate() -> dict:
    """SHADOW -> LIVE promotion. All criteria must pass. Read-only; promotion itself is a human decision."""
    s = stats()
    checks = {
        f"Decisions >= {GATE.MIN_DECISIONS_LOGGED}": s.get("decisions_logged", 0) >= GATE.MIN_DECISIONS_LOGGED,
        f"Closed trades >= {GATE.MIN_CLOSED_TRADES}": s.get("closed_trades", 0) >= GATE.MIN_CLOSED_TRADES,
        f"Expectancy >= {GATE.MIN_EXPECTANCY_R}R (net)": s.get("expectancy_r", -9) >= GATE.MIN_EXPECTANCY_R,
        f"Max DD within limit": abs(s.get("max_drawdown_r", -99)) <= GATE.MAX_DRAWDOWN_PCT * 100,  # R-proxy
        f"Rule violations == {GATE.MAX_RULE_VIOLATIONS}": s.get("rule_violations", 99) <= GATE.MAX_RULE_VIOLATIONS,
        f"Calendar days >= {GATE.MIN_CALENDAR_DAYS}": s.get("calendar_days", 0) >= GATE.MIN_CALENDAR_DAYS,
    }
    return {"stats": s, "checks": checks, "promotable": all(checks.values())}
