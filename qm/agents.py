"""
Seven-Agent Evidence Checklist (v1: structured human reasoning)
===============================================================
In v1 the "agents" are structured checklists the human completes — this is
deliberate. Running the full stack manually for 60-90 decisions produces
the labeled dataset that future LLM agents will be evaluated against.
You cannot grade an AI analyst without first recording what a disciplined
human analyst concluded and what subsequently happened.

Aggregation rule (anti-committee-drift):
  - ANY agent may VETO.  - NO agent may INITIATE.
  - A trade requires: zero vetoes AND >= 5 of 7 agents at NEUTRAL-or-better
    AND the Risk Agent (deterministic engine) approving.
"""

AGENTS = {
    "Macro": {
        "question": "Do rates, inflation path, dollar, oil and credit support risk-taking this week?",
        "prompts": [
            "Fed posture and next meeting distance?",
            "Oil / energy shock in progress?",
            "Credit spreads calm or widening?",
        ],
    },
    "Flows": {
        "question": "Are institutional flows (ETF, buybacks, OpEx, dealer gamma) a tailwind or headwind?",
        "prompts": ["Near OpEx?", "Buyback blackout window?", "Known rebalancing pressure?"],
    },
    "Technical": {
        "question": "Is there a clean institutional level (Ijeoma) with defined invalidation?",
        "prompts": ["Where did institutions accumulate/distribute?", "Is the stop obvious and close?",
                    "Is structure trending, ranging, or exhausted?"],
    },
    "Options": {
        "question": "Does the options market (IV vs HV, skew, expected move) favor the structure?",
        "prompts": ["Is IV rich (sell) or cheap (buy)?", "Expected move vs your target?",
                    "DTE >= 7 and defined-risk?"],
    },
    "Psychology": {
        "question": "Sentiment extremes? And — honestly — your own state?",
        "prompts": ["Retail euphoria or panic?", "Are YOU trading from loss, boredom, or FOMO?",
                    "Would you take this trade if flat on the week?"],
    },
    "Risk": {
        "question": "(Deterministic — completed by the risk engine, not by you.)",
        "prompts": ["Run the engine. Its verdict is final."],
    },
    "Portfolio": {
        "question": "Does this trade fit the book — heat, concentration, correlation?",
        "prompts": ["Total heat after this trade?", "Correlated with existing positions?",
                    "Is cash the better position?"],
    },
}

SCORES = ["VETO", "BEARISH-FOR-TRADE", "NEUTRAL", "SUPPORTIVE", "STRONG"]


def aggregate(scores: dict) -> dict:
    """scores: {agent: score_string}. Returns verdict per aggregation rule."""
    vetoes = [a for a, s in scores.items() if s == "VETO"]
    ok = [a for a, s in scores.items() if s in ("NEUTRAL", "SUPPORTIVE", "STRONG")]
    passed = (len(vetoes) == 0) and (len(ok) >= 5)
    return {
        "passed": passed,
        "vetoes": vetoes,
        "supportive_or_better": ok,
        "rule": "Zero vetoes AND >=5/7 neutral-or-better AND risk engine approval.",
    }


# ---------------------------------------------------------------------------
# Strategy selector: regime x directional view x vol view -> structure
# ---------------------------------------------------------------------------

def select_strategy(view: str, iv_state: str, regime: str) -> dict:
    """view: BULLISH/BEARISH/NEUTRAL/BIG-MOVE-UNSURE-DIRECTION
    iv_state: RICH/CHEAP/NORMAL  |  regime: OFFENSIVE/NEUTRAL/DEFENSIVE/LOCKDOWN"""
    if regime == "LOCKDOWN":
        return {"structure": "NONE", "why": "Lockdown: no new risk. Cash and journaling."}

    table = {
        ("BULLISH", "RICH"): ("Put Credit Spread", "Sell rich IV below institutional support; defined risk."),
        ("BULLISH", "CHEAP"): ("Long Call / Bull Call Spread", "Cheap IV favors buying premium; debit spread if IV mid."),
        ("BULLISH", "NORMAL"): ("Bull Call Spread", "Balanced: directional exposure, capped cost."),
        ("BEARISH", "RICH"): ("Call Credit Spread", "Sell rich IV above resistance; defined risk."),
        ("BEARISH", "CHEAP"): ("Long Put / Bear Put Spread", "Cheap IV favors buying downside."),
        ("BEARISH", "NORMAL"): ("Bear Put Spread", "Defined-risk directional."),
        ("NEUTRAL", "RICH"): ("Iron Condor", "Only when regime is OFFENSIVE/NEUTRAL and no events inside the wings."),
        ("NEUTRAL", "CHEAP"): ("Calendar Spread", "Cheap front IV; harvest term structure."),
        ("NEUTRAL", "NORMAL"): ("No trade / Covered Call on holdings", "Neutral view + normal IV = thin edge."),
        ("BIG-MOVE-UNSURE-DIRECTION", "RICH"): ("No trade", "Everyone already knows: straddles are priced. Skip."),
        ("BIG-MOVE-UNSURE-DIRECTION", "CHEAP"): ("Long Straddle/Strangle", "Cheap vol into catalyst — the only good time."),
        ("BIG-MOVE-UNSURE-DIRECTION", "NORMAL"): ("Small Long Strangle or skip", "Marginal."),
    }
    s, why = table.get((view, iv_state), ("No trade", "Unmapped combination = no edge identified."))
    if regime == "DEFENSIVE" and "Credit" in s:
        why += " DEFENSIVE regime: short premium at 0.3x size only, or prefer skipping."
    if regime == "DEFENSIVE" and s == "Iron Condor":
        s, why = "No trade", "Iron condors in DEFENSIVE regimes are how accounts die. Skip."
    return {"structure": s, "why": why, "regime": regime}
