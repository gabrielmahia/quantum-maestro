Below is the system I would hand to an engineer as the **canonical functional specification**. It is self-contained: they should be able to implement the methodology without seeing Teri’s PDFs, spreadsheets, videos, or anything from this chat.

The first distinction matters: **the core buyer/seller-zone methodology, ATR stop logic, reward/risk logic, and Odd Enhancers come from the IWT course material.** The regime engine, time-of-day rules, defined-risk options router, market crosschecks, earnings controls, managed-vertical recycling analysis, and portfolio controls are the system we have built around it.

---

# IWT / Quantum Maestro Trading System Specification

## 1. System purpose

The system is a **rule-governed discretionary trading decision engine**.

It does not attempt to predict every market move. Its job is to identify locations where the relationship between potential reward and defined risk is attractive, determine whether current market conditions support acting at that location, select the appropriate instrument, size the position so a failed thesis is survivable, and manage the position according to predefined invalidation rules.

The fundamental sequence is:

**Location → Direction → Quality → Context → Confirmation → Reward/Risk → Structure → Size → Execution → Management → Attribution.**

The engine must be capable of returning:

> **NO TRADE**

as a fully valid output.

That is central to the philosophy.

---

# 2. Core market model: buyer zones and seller zones

The original methodology treats price as moving between areas where buyers previously overwhelmed sellers and areas where sellers previously overwhelmed buyers.

A **Buyer Zone (BZ)** is an area below or near current price where demand previously produced a meaningful upward reaction.

A **Seller Zone (SZ)** is an area above or near current price where supply previously produced a meaningful downward reaction.

These are **zones, not single prices**.

Each zone has two boundaries:

| Term          | Meaning                                     |
| ------------- | ------------------------------------------- |
| Proximal line | Boundary nearest current market price       |
| Distal line   | Boundary farthest from current market price |

For a buyer zone below price:

[
BZ_{proximal} > BZ_{distal}
]

For a seller zone above price:

[
SZ_{distal} > SZ_{proximal}
]

The course's shorting material explicitly demonstrates entering from a seller level and covering toward the buyer level. 

The model should visually and logically treat the market as:

```text
           higher price

      Seller Distal
     ┌───────────────┐
     │  SELLER ZONE  │
     └───────────────┘
      Seller Proximal


        current price


      Buyer Proximal
     ┌───────────────┐
     │   BUYER ZONE  │
     └───────────────┘
      Buyer Distal

           lower price
```

A zone itself is **not permission to trade**.

It is only a candidate location.

---

# 3. Canonical long trade

A long setup begins when price approaches a qualified buyer zone.

The default IWT mathematical model is:

[
Entry_L = BZ_{proximal}
]

The stop is beyond the distal boundary with a volatility buffer equal to approximately **20% of ATR / average daily movement**:

[
Buffer=0.20\times ATR
]

[
Stop_L=BZ_{distal}-Buffer
]

The default opposing target is the proximal seller line:

[
Target_L=SZ_{proximal}
]

Therefore:

[
Risk_L=Entry_L-Stop_L
]

[
Reward_L=Target_L-Entry_L
]

and:

[
RR_L=\frac{Reward_L}{Risk_L}
]

The original model strongly prefers approximately **3:1 or greater reward/risk**.

If:

[
RR_L<3
]

the default conclusion is that the trade is not attractive enough unless some explicitly defined variant justifies it.

Position size is determined by dollar risk tolerance:

[
Quantity_L=
\left\lfloor
\frac{DollarRiskBudget}{Risk_L}
\right\rfloor
]

Critically:

> **Desired profit does not determine position size. Maximum acceptable loss does.**

---

# 4. Canonical short trade

The system must implement the short side symmetrically.

At a qualified seller zone:

[
Entry_S=SZ_{proximal}
]

[
Stop_S=SZ_{distal}+0.20ATR
]

[
Target_S=BZ_{proximal}
]

Then:

[
Risk_S=Stop_S-Entry_S
]

[
Reward_S=Entry_S-Target_S
]

[
RR_S=\frac{Reward_S}{Risk_S}
]

and:

[
Quantity_S=
\left\lfloor
\frac{DollarRiskBudget}{Risk_S}
\right\rfloor
]

Teri's shorting example explicitly frames the trade as selling stock near the seller zone and later using **Buy to Cover** after the decline toward the buyer area. 

Thus the system should never be architected as inherently bullish.

It is a **bidirectional location-based system**.

---

# 5. Zone-quality engine: the Odd Enhancers

The course does not treat all buyer/seller zones equally.

A zone receives an **Odd Enhancer score** based on four characteristics.

| Factor             | 2 points            | 1 point     | 0 points          |
| ------------------ | ------------------- | ----------- | ----------------- |
| Time spent in zone | roughly 1–2 candles | ~3–4        | 4+ / prolonged    |
| Speed leaving zone | fast / decisive     | average     | slow / weak       |
| Freshness          | untested            | once tested | repeatedly tested |
| Reward/Risk        | ≥3:1                | around 2:1  | <2:1              |

Conceptually, the reasoning is:

**Short residence time** suggests orders overwhelmed the opposing side quickly.

**Fast departure** suggests meaningful imbalance.

**Fresh zones** are preferred because repeated tests may consume resting orders.

**Good R:R** means being right produces materially more than being wrong costs.

The source workbook explicitly says a **7 or 8 score is a take-the-trade quality zone**. Our implementation is deliberately conservative: 7–8 is strongest, 4–6 requires additional confirmation, and below 4 is generally rejected.

Define:

[
OddScore=T+S+F+R
]

where each component is 0–2.

The engine should interpret:

| Score | Interpretation                              |
| ----: | ------------------------------------------- |
|   7–8 | High-quality candidate                      |
|   4–6 | Conditional; stronger confirmation required |
|   0–3 | Skip                                        |

This is a **quality filter**, not a profitability guarantee.

---

# 6. Price confirmation layer

This is one of the most important additions we developed.

A zone describes **where something could happen**.

Confirmation asks whether it is **actually beginning to happen**.

For longs, confirmation evidence can include:

* price entering/reaching the buyer zone and rejecting lower prices;
* reclaiming VWAP;
* sustaining above VWAP after reclaim;
* bullish reversal candle;
* U-shaped or Chair-like reversal structure;
* higher low after the reaction;
* improving index/breadth participation.

For shorts, use the inverse:

* rejection from seller zone;
* failed VWAP reclaim;
* loss of VWAP after testing it;
* bearish reversal candle;
* inverted U/Chair structure;
* lower high after rejection;
* weakening breadth/leadership.

This produces two independent confirmation scores:

[
C_L
]

and

[
C_S
]

A level should not be acted on merely because price touches it.

The desired transition is:

```text
Zone reached
    ↓
Reaction observed
    ↓
Confirmation develops
    ↓
Risk/reward remains acceptable
    ↓
Trade becomes eligible
```

If price slices directly through the zone, the level has **failed**, not become a better bargain.

---

# 7. Macro/regime overlay

This is our principal extension beyond classical IWT.

The same buyer level does **not** have identical expectancy in every market environment.

Therefore the engine maintains a regime state:

```text
OFFENSIVE
NEUTRAL-OFFENSIVE
NEUTRAL
NEUTRAL-DEFENSIVE
DEFENSIVE
LOCKDOWN
```

Regime is derived from a basket rather than a single indicator.

The engine monitors:

| Domain            | Examples                                          |
| ----------------- | ------------------------------------------------- |
| Trend             | SPX/QQQ/DIA/IWM structure                         |
| Breadth           | advancing participation, leadership concentration |
| Rates             | 2Y/10Y/30Y, real yields, curve                    |
| Credit            | IG/HY spreads                                     |
| Liquidity         | Fed balance sheet, reserves, TGA, RRP             |
| Volatility        | VIX, term structure, VVIX                         |
| Options structure | dealer/gamma positioning when reliable            |
| Commodities       | especially oil as inflation/geopolitical input    |
| USD               | financial-condition impulse                       |
| Earnings          | especially mega-cap technology                    |
| Macro             | CPI/PPI/jobs/retail/housing/PMI/Fed               |
| Flows             | passive, institutional, month/quarter-end         |
| Geopolitics       | wars, trade, energy chokepoints                   |
| Sentiment         | positioning and fear/greed inputs                 |

No single variable controls the state.

A simple implementation could use:

[
RegimeScore=\sum_i w_i x_i
]

with normalized inputs between -1 and +1.

But the production implementation should preserve explainability:

```json
{
  "regime": "NEUTRAL_DEFENSIVE",
  "confidence": 0.74,
  "drivers": [
    "long_yields_rising",
    "oil_rising",
    "semiconductor_relative_weakness",
    "breadth_deteriorating"
  ],
  "counterevidence": [
    "credit_spreads_stable",
    "SPX_major_support_intact"
  ]
}
```

The system must always retain the strongest counterargument.

---

# 8. Cross-index confirmation

A move in one index is less informative than a move confirmed across the market.

The engine therefore separately tracks:

**Dow / DIA**

**S&P / SPX/SPY**

**Nasdaq / QQQ**

**Russell / IWM**

**Semiconductors / SMH**

and market breadth.

The old course resource used constituent crosschecks. We retained the concept but not the old static weights because those holdings were dated.

An engineer should calculate live relationships.

Examples:

```text
SPX ↑
QQQ ↑
SMH ↑
breadth ↑
yields stable
        → strong bullish confirmation
```

versus:

```text
SPX ↑
QQQ ↓
SMH ↓
breadth ↓
        → headline index strength may be masking distribution
```

This is a **divergence detector**.

---

# 9. Direction-selection state machine

The engine should not begin with a trade structure.

It should begin with a hypothesis.

A simplified state machine is:

```text
                           ┌──────────────┐
                           │    WAIT      │
                           └──────┬───────┘
                                  │
                       candidate zone reached
                                  │
                   ┌──────────────┴──────────────┐
                   │                             │
               Buyer Zone                   Seller Zone
                   │                             │
             bullish reaction               bearish reaction
                   │                             │
             confirmation?                  confirmation?
                   │                             │
              YES / NO                       YES / NO
                   │                             │
       context supports long?        context supports short?
                   │                             │
                   ▼                             ▼
                LONG                           SHORT
```

At any failed gate:

[
State \rightarrow WAIT
]

That is intentional.

---

# 10. Instrument-selection philosophy

Only after direction is established does the system decide **how** to express the trade.

This separates:

> **thesis**

from:

> **instrument**.

The strategy router should work approximately as follows:

| Thesis                                     | Location      | Preferred expressions                                                  |
| ------------------------------------------ | ------------- | ---------------------------------------------------------------------- |
| Bullish                                    | buyer zone    | shares, long call, bull-call debit spread, bull-put credit spread      |
| Bearish                                    | seller zone   | short shares, long put, bear-put debit spread, bear-call credit spread |
| Neutral                                    | between zones | usually cash; defined range strategies only under strong conditions    |
| High event risk                            | either        | defined-risk debit structures preferred                                |
| Elevated IV after confirmed hold/rejection | either        | defined-risk credit structures become more attractive                  |

The underlying course material maps buying calls/selling puts to the buyer zone and buying puts/selling calls to the seller zone.  

Our system modifies this by **favoring defined-risk spreads** instead of naked short options.

---

# 11. Long-option philosophy

For purchased calls or puts:

[
MaxLoss = PremiumPaid \times Multiplier \times Contracts
]

The source material emphasizes that theta works against purchased options and generally suggests buying **2+ months of time** and exiting before the final approximately two weeks rather than holding to expiration. 

Our practical default therefore becomes approximately:

[
45\text{–}90\ DTE
]

for ordinary directional long-option exposure.

This is a heuristic, not a law.

The engineer should model:

* delta;
* theta;
* implied volatility;
* DTE;
* break-even;
* premium at risk;
* expected move;
* earnings/event proximity.

Do not choose an option merely because it is inexpensive.

Far-OTM cheap contracts are frequently low-probability lottery tickets.

---

# 12. Short-premium philosophy

For sold premium, theta generally works in the trader's favor.

The source material explicitly describes selling premium when volatility is elevated, then potentially buying it back after premium/volatility falls. 

Our implementation adds stricter controls:

**Never sell premium solely because IV is high.**

High IV often exists because tail risk is genuinely high.

Require:

```text
qualified level
+
confirmation
+
acceptable opposing-zone room
+
acceptable event risk
+
defined maximum loss
+
sufficient premium
```

For a credit spread:

[
MaxProfit=Credit\times100\times N
]

For width (W):

[
MaxLoss=(W-Credit)\times100\times N
]

Sizing is based on the loss, not the premium.

---

# 13. Defined-risk structure preference

Our production system should favor:

**Bullish continuation:** bull-call debit spread.

**Bearish continuation:** bear-put debit spread.

**Confirmed support + rich IV:** bull-put credit spread.

**Confirmed rejection + rich IV:** bear-call credit spread.

Debit spreads are particularly useful on event-heavy or volatile days because maximum loss is known without relying upon successful stop execution.

Credit spreads should be used only when support/resistance has **actually demonstrated itself**.

---

# 14. Account-aware routing

The system must know which account is being used.

For the smaller Fidelity account we discussed, ordinary eligible strategies are essentially:

```text
Long Call
Long Put
Cash
```

For TOS/other approved accounts:

```text
Long Call
Long Put
Bull Call Debit
Bear Put Debit
Bull Put Credit
Bear Call Credit
Other defined-risk structures
Cash
```

Therefore the same market signal may yield two different recommendations.

Example:

```text
Signal: bearish SPX failed reclaim
```

Fidelity expression:

> Long QQQ/SPY/eligible-symbol put.

TOS expression:

> SPX bear-put debit spread.

The underlying thesis remains identical.

---

# 15. Earnings no-trade window

Company-specific earnings are treated as discrete volatility events.

For watched companies, maintain:

```json
{
  "ticker": "NVDA",
  "earnings_date": "...",
  "timing": "AMC|BMO|UNKNOWN",
  "status": "CONFIRMED|ESTIMATED",
  "calendar_days_remaining": 0,
  "trading_sessions_remaining": 0
}
```

Ordinary directional or short-premium trades should generally be prohibited from:

> **the final trading session before earnings through the first full session after earnings**

unless the user explicitly intends an earnings trade and maximum loss is completely defined.

Estimated dates must not be represented as confirmed dates.

---

# 16. Time-of-day model

The system does **not** treat all minutes of the trading session equally.

### 9:30–10:30 ET

Primarily observation and price discovery.

Establish:

* opening range;
* VWAP;
* buyer/seller reaction;
* gap acceptance/rejection;
* actual institutional direction.

Ordinary new trades are generally avoided.

### 10:30–11:45

Primary trading window.

This is where a morning structure has had time to reveal itself.

### 12:00–1:30

Lunch.

Avoid ordinary new trades.

Manage existing positions.

### 1:30–3:00

Secondary window.

Only participate if a genuinely fresh setup develops.

### After 3:00

Increasing caution.

Institutional rebalancing, ETF/mutual-fund flows, options hedging, market-on-close orders and algorithms can dominate.

Prefer management or exit over chasing new positions.

On known **10:00, 2:00 or 2:30 event days**, the event clock overrides the ordinary timetable.

---

# 17. Event gate

Before any order, calculate:

[
MinutesToEvent
]

For a major scheduled market event, if:

[
MinutesToEvent < Threshold
]

ordinary new trades become:

> **WAIT**

A reasonable initial threshold is 30 minutes, adjustable by event type.

The same applies immediately after the event: allow price discovery.

---

# 18. Intent-versus-order check

This is an explicit safety requirement.

Before sending any option order, create an immutable intent object:

```json
{
  "thesis": "BEARISH",
  "instrument": "SPX",
  "strategy": "BEAR_PUT_DEBIT",
  "expiration": "2026-09-04",
  "long_strike": 7800,
  "short_strike": 7750,
  "order_side": "BUY",
  "net_type": "DEBIT",
  "max_loss": 2100,
  "expected_exposure": "NEGATIVE_DELTA"
}
```

Then compare the actual broker order to intent.

Reject if, for example:

```text
thesis = bearish
but
order = bull put credit
```

unless the discrepancy is explicitly intentional.

The system should verify:

* buy/sell side;
* expiration;
* strikes;
* debit versus credit;
* maximum loss;
* bullish/bearish exposure.

---

# 19. Position-sizing engine

There are two risk concepts.

### Planned risk

What we expect to lose if the stop works.

For a credit spread sold for (C) with planned stop price (S):

[
PlannedLoss=(S-C)\times100
]

### Catastrophic risk

Worst defined loss:

[
CatastrophicLoss=(Width-C)\times100
]

Both limits matter.

Suggested system parameters:

[
RiskPerTrade \approx 0.5%-1.0%
]

for ordinary high-quality trades, depending on strategy/account.

Total intraday planned loss should also have a hard ceiling.

The key rule is:

> **Contract count is the result of risk allocation. Never decide to trade “40 contracts” and then reverse-engineer justification.**

---

# 20. Tranche model

Large allowable positions should not generally be entered all at once.

If the risk engine permits (N) contracts, start with approximately:

[
0.25N
]

Then add only if additional market information improves the thesis.

Example:

```text
10 contracts
+
10 after confirmation
+
10 after favorable structure
+
10 reserved for fresh setup
```

not:

```text
40 immediately
```

Never use subsequent tranches to rescue a failed thesis.

---

# 21. 0DTE regime

0DTE is a distinct strategy class because gamma becomes dominant.

The setup must be stronger, not weaker.

For bullish 0DTE short premium:

```text
buyer level
→ hold
→ reclaim
→ confirmation
→ bull-put spread
```

For bearish:

```text
seller level / failed rally
→ rejection
→ lower high or failed reclaim
→ bear-call spread
```

The system must know that a theoretical stop may not fill during a fast 0DTE move.

Therefore both:

[
PlannedRisk
]

and:

[
MaxDefinedRisk
]

must be survivable.

---

# 22. Managed vertical / short-leg recycling

This is a major extension created from our work.

Suppose we hold a defined vertical:

```text
Long protective put
+
Short put
```

The ordinary strategy holds both.

The managed strategy may:

1. sell the short option;
2. buy it back after substantial decay;
3. retain the long hedge;
4. wait;
5. if the setup requalifies, sell the short option again;
6. repeat.

The naive accounting mistake is to call every realized short-option gain “extra profit.”

Some of it is merely **capturing portions of the original decay path early**.

Therefore the system must distinguish:

### Gross short-leg P/L

[
GrossCyclePL=(STO-BTC)\times100\times N
]

### Net short-leg P/L

[
NetCyclePL=GrossCyclePL-Fees-Slippage
]

### Re-entry improvement

If the previous short was bought back for (BTC_{prior}) and later resold at (STO_{new}):

[
ReEntryImprovement=STO_{new}-BTC_{prior}
]

### Management alpha

[
ManagementAlpha=
ReEntryImprovement\times100\times N
-----------------------------------

IncrementalFriction
]

This is closer to measuring whether active recycling actually added value over passive holding.

The ledger should separately track:

```text
gross short-leg realized P/L
net short-leg realized P/L
true re-entry alpha
long hedge original cost
cumulative funding of hedge
remaining hedge basis
```

The key rule:

> **Do not re-sell the short leg merely because premium has risen again. The underlying IWT setup must requalify.**

---

# 23. Probable path, never certainty

The engine may produce forecasts, but they are explicitly labeled:

> **PROBABLE**

not:

> CONFIRMED.

Example:

```json
{
  "probable_path": "SPX consolidates or retests buyer support before another upside attempt",
  "confidence": 0.62,
  "confirmation": [
    "10Y falls below X",
    "QQQ recovers VWAP",
    "breadth > threshold"
  ],
  "invalidation": [
    "SPX closes below buyer distal",
    "credit spreads widen",
    "SMH leadership deteriorates"
  ]
}
```

This prevents narrative from hardening into certainty.

---

# 24. Regime transitions

The system is not simply looking for price changes.

It is watching **causal combinations**.

Examples:

```text
oil ↑
+
10Y/30Y ↑
+
QQQ ↓
+
SMH ↓
+
breadth ↓
```

is much more meaningful than:

```text
SPX -0.5%
```

Likewise:

```text
growth data weak
+
yields fall
+
tech rises
```

may actually be risk-positive.

Whereas:

```text
growth weak
+
long yields remain high
+
oil high
```

suggests a much less benign environment.

The engine should therefore reason about **relationships**, not indicators in isolation.

---

# 25. Position-management hierarchy

Once a position exists, management follows this order:

```text
1. Has the thesis been invalidated?
2. Has maximum permitted risk changed?
3. Has the opposing zone been reached?
4. Has market regime changed?
5. Has an event materially altered expected distribution?
6. Has reward remaining become too small relative to risk?
7. Has time decay changed the economics?
8. Is there a superior use of capital?
```

P/L by itself is not a sufficient decision criterion.

A losing position with intact thesis can differ from a losing position whose original reason no longer exists.

Likewise a profitable trade can still require closing if remaining risk dominates remaining reward.

---

# 26. Journaling and attribution

Every trade must produce a structured record.

Suggested schema:

```json
{
  "trade_id": "...",
  "timestamp_entry": "...",
  "timestamp_exit": "...",

  "symbol": "SPX",
  "account": "TOS",
  "direction": "SHORT",

  "buyer_proximal": 7600,
  "buyer_distal": 7575,
  "seller_proximal": 7790,
  "seller_distal": 7815,
  "atr": 65,

  "odd_score": 7,
  "context_score": -4,
  "confirmation_score": 3,
  "regime": "NEUTRAL_DEFENSIVE",

  "strategy": "CALL_CREDIT_SPREAD",
  "expiration": "...",
  "strikes": [7825, 7830],
  "credit": 0.80,

  "planned_risk": 500,
  "max_loss": 2100,

  "entry_reason": "...",
  "invalidation": "...",
  "exit_reason": "...",

  "gross_pnl": 650,
  "fees": 34,
  "net_pnl": 616,

  "r_multiple": 1.23,
  "rule_adherence": 0.94,

  "lesson": "..."
}
```

The system eventually evaluates:

[
Expectancy=
P(win)\times AvgWin
-------------------

P(loss)\times AvgLoss
]

not just win rate.

---

# 27. Learning engine

The objective is to determine which components actually generate edge.

Segment results by:

* regime;
* Odd Score;
* buyer versus seller setup;
* symbol;
* strategy;
* DTE;
* time of day;
* confirmation type;
* IV regime;
* event/non-event;
* passive versus recycled spread management;
* Fidelity versus TOS expression.

Then ask questions such as:

[
E[R\mid OddScore\ge7]
]

versus:

[
E[R\mid OddScore=4\text{–}6]
]

or:

[
E[R_{\text{managed spread}}]
----------------------------

E[R_{\text{passive spread}}]
]

The methodology therefore becomes **falsifiable**.

If some cherished rule does not improve expectancy, the data should expose that.

---

# 28. Morning decision engine

At approximately 9:00 a.m. ET, the system performs the full macro scan and returns a concise decision sheet.

Its internal output should include:

| Component       | Output                                           |
| --------------- | ------------------------------------------------ |
| Asia            | direction + leadership                           |
| Europe          | direction + breadth                              |
| U.S. futures    | SPX/NDX/Dow/Russell                              |
| Rates           | 2Y/10Y/30Y + real yields                         |
| Dollar          | direction                                        |
| Oil/gold        | direction                                        |
| Volatility      | VIX/term structure/VVIX                          |
| Macro           | releases/events                                  |
| Earnings        | watched companies                                |
| Geopolitics     | active risks                                     |
| Regime          | classification + confidence                      |
| Session type    | normal/event-heavy/earnings/gap/expiry/month-end |
| Top symbols     | exactly three where possible                     |
| SPX             | always assessed                                  |
| Setup           | preferred pattern                                |
| Account routing | Fidelity vs TOS                                  |
| Action          | ranked choices including cash                    |

It should end with something resembling:

```text
SPX confirmed seller rejection >
NVDA relative-weakness continuation >
QQQ failed reclaim >
CASH
```

No trade is preferable to a fabricated fourth choice.

---

# 29. Non-morning monitoring

At later scheduled runs, the system should remain silent unless one of three things changes materially:

```text
market regime
preferred trading posture
earnings risk
```

The system should not spam ordinary market fluctuations.

It is a **change detector**, not a ticker feed.

---

# 30. The deepest philosophy

The methodology rests on several principles.

First:

> **Price location matters more than excitement.**

Buying after a huge rally merely because a stock looks strong often produces poor reward/risk. Selling after collapse can have the same problem.

Second:

> **Risk is decided before reward is pursued.**

The trader first asks how much can be lost if wrong.

Third:

> **Let price reveal buyers and sellers.**

Don't impose an opinion merely because macro news sounds bullish or bearish.

Fourth:

> **Good trade ≠ guaranteed profitable trade.**

A high-quality setup can lose.

A terrible setup can win.

The system evaluates **decision quality independently of outcome**.

Fifth:

> **Time is part of price.**

Options require explicitly accounting for DTE, theta, IV, event timing and gamma.

Sixth:

> **Cash is an active position.**

The system does not have an obligation to deploy capital.

Seventh:

> **The environment modifies the setup.**

A beautiful buyer zone under broad risk-on conditions is not equivalent to the same buyer zone during credit deterioration, surging oil, exploding yields and collapsing breadth.

Eighth:

> **Scale comes after edge.**

Position size increases only after the method demonstrates positive expectancy over enough observations.

---

# 31. Engineering architecture

I would separate the implementation into deterministic modules:

```text
Market Data Layer
       ↓
Zone Engine
       ↓
Odd Enhancer Engine
       ↓
Regime Engine
       ↓
Confirmation Engine
       ↓
Reward/Risk Engine
       ↓
Strategy Router
       ↓
Account Constraint Engine
       ↓
Event/Earnings Gate
       ↓
Position Sizer
       ↓
Intent Validator
       ↓
Broker Preview
       ↓
Human Approval
       ↓
Execution
       ↓
Position Manager
       ↓
Trade Journal
       ↓
Attribution / Learning Engine
```

The LLM or AI component may assist with:

```text
news interpretation
macro synthesis
scenario generation
earnings summaries
geopolitical assessment
competing explanations
```

But **deterministic code**, not an LLM, should control:

```text
max risk
strike relationships
credit/debit orientation
position sizing
earnings blackout
event blackout
account permissions
order intent consistency
maximum daily loss
```

That boundary is essential.

---

# 32. Core decision pseudocode

An engineer could start with this:

```python
def evaluate_trade(symbol, market, zones, account, options=None):

    # 1. Locate market
    location = classify_location(
        price=market.price,
        buyer_zone=zones.buyer,
        seller_zone=zones.seller
    )

    # 2. Score zone quality
    odd_score = score_odd_enhancers(
        time_in_zone=zones.time_in_zone,
        speed_out=zones.speed_out,
        freshness=zones.freshness,
        rr_quality=zones.rr
    )

    # 3. Determine global environment
    regime = evaluate_regime(market)

    # 4. Observe actual reaction
    confirmation = evaluate_confirmation(
        price_action=market.intraday,
        vwap=market.vwap,
        breadth=market.breadth
    )

    # 5. Calculate both directions
    long_math = calculate_long_iwt(zones, market.atr)
    short_math = calculate_short_iwt(zones, market.atr)

    # 6. Direction
    thesis = choose_direction(
        location,
        odd_score,
        regime,
        confirmation,
        long_math,
        short_math
    )

    if thesis == "WAIT":
        return no_trade("No qualified directional edge")

    # 7. Event gate
    if major_event_imminent(market.calendar):
        return no_trade("Event risk")

    # 8. Earnings gate
    if inside_earnings_window(symbol):
        return no_trade("Earnings no-trade window")

    # 9. Choose legal/appropriate expression
    strategy = route_strategy(
        thesis=thesis,
        account=account,
        iv=market.iv,
        dte=options.dte if options else None
    )

    # 10. Compute maximum risk
    risk = calculate_max_loss(strategy)

    # 11. Position size
    qty = size_from_risk(
        account_equity=account.equity,
        risk_budget=account.risk_budget,
        max_loss_per_unit=risk
    )

    if qty < 1:
        return no_trade("Risk budget cannot support position")

    # 12. Validate intent against order
    order = construct_order(strategy, qty)

    if not intent_matches_order(thesis, order):
        raise OrderValidationError()

    return TradePlan(
        thesis=thesis,
        strategy=strategy,
        quantity=qty,
        confirmation=confirmation,
        invalidation=derive_invalidation(zones, thesis),
        regime=regime
    )
```

That captures the core architecture.

---

# 33. One sentence specification

If the engineer remembers nothing else:

> **Identify where institutional buying or selling previously overwhelmed the market, prefer fresh zones that price left quickly, require attractive reward relative to a volatility-adjusted stop, let current price action confirm the reaction, adjust aggressiveness for the broader market regime, express the thesis with the safest account-appropriate structure, size exclusively from defined loss, and record enough information to prove later whether the supposed edge was real.**

That is the system we have been converging toward: **Teri's location-and-risk methodology as the foundation, with regime intelligence, options engineering, execution discipline, and empirical feedback wrapped around it.**
