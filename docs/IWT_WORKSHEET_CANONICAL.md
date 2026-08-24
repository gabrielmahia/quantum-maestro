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

---

## Source conflict: the cohort bands (documented, not silently resolved)

Teri's materials state **two different band schemes** for turning the 0–8 odds
score into an entry decision:

| Source | Direct entry | Confirmation entry | Skip |
|---|---|---|---|
| "Deciding Your Entry Strategy" (Odd Enhancers PDF) | **6–8** | **4–6** | below 4 |
| Key Documents odds table (DOCX) | "7–8 TAKE THE TRADE" | — | — |

The PDF's own bands **overlap at 6**, so a score of exactly 6 is ambiguous in
the source itself.

`qm/iwt_zones.odds_enhancer` now takes a `band_scheme` argument:

- **`COURSE`** — 6/4, faithful to the published entry-strategy PDF.
- **`STRICT`** — 7/5, one notch tighter. **This is the default**, because a 6
  sitting in the PDF's own overlap should earn confirmation rather than a direct
  fill, and because tightening a quality gate errs in the conservative direction.

Both are selectable, and the scheme is returned with every result so decisions
stay comparable across the journal and any backtest. **Log which scheme produced
a cohort** — mixing them silently would corrupt cohort expectancy.

### Correction of an earlier call

An external v5.0 ThinkScript draft used `directEntryMinimumScore=6` /
`confirmationMinimumScore=4`. In review, this was labelled a bug — "shifted down
by one cohort" — on the strength of the DOCX table alone. That was **overstated**:
those thresholds match the published entry-strategy PDF exactly. The engineer was
reading a legitimate source. The tightening to 7/5 remains defensible as a
deliberate policy choice, but it is a *choice*, not a correction of an error, and
it is now implemented as such.

## "Bank-like numbers" — the round-number stop rule

The Stock Pick worksheet specifies the stop as "a little below the Buyers Level —
20% of the average daily move / **below bank-like numbers**."

Round numbers ($50, $100, $250) are where stop clusters sit. An ATR-derived stop
that lands just *above* a round number (on a long) gets swept before the thesis
has actually failed. `bank_number_adjusted_stop()` detects a stop sitting on a
round level and moves it clear, reporting which level triggered the adjustment so
it is never silent.

## Target haircut

The worksheet says exit "a little **before** the first line of Sellers Level" —
don't demand the last cent. `haircut_target()` applies a small, explicit haircut
(scaled to risk when supplied) to raise fill probability, and reports the amount
given up.
