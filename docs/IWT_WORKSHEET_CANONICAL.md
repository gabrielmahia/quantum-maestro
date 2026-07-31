# IWT Worksheet — Canonical Reference (verbatim from course)

Transcribed directly from Teri Ijeoma's Trade & Travel "Key Documents." This is
the authoritative wording; Quantum Maestro's `qm/iwt_canonical.py` implements it.

## BUY worksheet

```
BUY
  SELLERS EXIT
  ENTRY  BUYERS  STOP

  STOP  = ATR  X 20% = ___
          (SUBTRACT FROM BOTTOM OF BUYER'S LEVEL)

  ENTRY        ___
  - STOP       ___
  RISK  =      ___

  EXIT         ___
  - ENTRY      ___
  REWARD =     ___

  RATIO = REWARD / RISK
  (RATIO = 3+, TAKE THE TRADE)

  RISK TOLERANCE / RISK = # OF SHARES  (ROUND DOWN)

  ORDER STOCK:
  EXIT     ENTRY     STOP
  LIMIT    LIMIT     STOP MARKET
```

**Key confirmations for the codebase:**

- "SUBTRACT FROM BOTTOM OF BUYER'S LEVEL" — the bottom of the buyer's level is
  the **distal** line. So `stop = distal_BZ − 0.20·ATR`, exactly as implemented.
- Entry is at the **top** of the buyer's level (the proximal line).
- Exit/target is the seller's level (the opposing proximal).
- **Order types are not uniform:** EXIT = limit, ENTRY = limit, **STOP = stop-market.**
  Entries and profit exits are limit orders (price control); the protective stop
  is a stop-*market* (fill certainty when the level breaks). Quantum Maestro's
  limit-only entry doctrine matches the ENTRY leg; the STOP leg is legitimately a
  stop-market and the broker adapter's generic `type` field supports it.
- "ROUND DOWN" on share count — never round up into more risk than tolerated.

## Odds enhancer table (verbatim)

```
ODDS      2            1            0
IN     1-2 CANDLES   3-4 CANDLES   4+ CANDLES
OUT    FAST          AVERAGE       SLOW
FRESH  0 VISITS      1 VISIT       1+ VISITS
RATIO  3+            2             <2
SCORE  7-8 TAKE THE TRADE
```

**Confirmations:** freshness is `0 visits → 2, exactly 1 visit → 1, more than 1
→ 0` (the "1+ VISITS" column means *more than one*). `qm/iwt_zones.freshness_score`
and `qm/iwt_canonical` both match this exactly. The score-7-8 "take the trade"
band matches the PRIMARY cohort.

## Steps to find levels (verbatim)

```
BUYERS LEVELS                     SELLERS LEVELS
1. Start at current price.        1. Start at current price.
2. Look DOWN and to the left.     2. Look UP and to the left.
3. Find the formation (U/Chair).  3. Find the formation (Upside-down U/Chair).
4. Use odd enhancers for strength.4. Use odd enhancers for strength.
5. If strong, mark distal/proximal.5. If strong, mark distal/proximal.
```

This is exactly the U/Chair + odds-enhancer + distal/proximal pipeline the
TeriQuantumZones ThinkScript study approximates mechanically.
