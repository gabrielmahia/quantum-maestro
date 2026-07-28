# ═══════════════════════════════════════════════════════════════════
# TeriQuantumOsc v4.4 - Regime (index) + Instrument (chart symbol)
#
# v4.3 BUG THIS FIXES: with useMarketData=yes, EVERY layer read the
# index. On an AAP chart the study reported SPX trend, SPX momentum,
# SPX levels, SPX headroom and SPX IV - identical output to an SPX
# chart. Nothing on screen described the instrument being traded.
#
# v4.4 splits the two questions Teri's method actually asks:
#   REGIME     <- INDEX (marketSymbol). Should I be trading at all?
#   LOCATION   <- CHART SYMBOL. Where are this name's buyer/seller
#                 levels, and is there room? (entry quality)
#   PREMIUM    <- CHART SYMBOL IV. Rich/cheap for THIS name's options.
#   PERMISSION <- events, vol stress, earnings, ex-div.
#
# New gates (equity-critical, were entirely missing):
#   - EARNINGS proximity: blocks short premium inside the window.
#   - EX-DIVIDEND proximity: early-assignment warning on short calls.
#   - RELATIVE STRENGTH vs the index: Teri buys leaders, not laggards.
#
# Proxy only. Quantum Maestro app engine remains canonical.
# No 0DTE, defined risk only, no short premium into events. No strikes.
# ═══════════════════════════════════════════════════════════════════

declare lower;

# ── Inputs ──────────────────────────────────────────────────────────
input marketSymbol = "SPX";
input vixSymbol    = "VIX";
input vix9dSymbol  = "VIX9D";

input fastTrendLength   = 20;
input mediumTrendLength = 50;
input slowTrendLength   = 200;

input rsiLength = 14;
input macdFast = 12;
input macdSlow = 26;
input macdSignalLength = 9;

input atrLength = 14;
input levelLookback = 20;
input requiredHeadroomATR = 0.5;
input rsLookback = 60;

input vixCaution = 18;
input vixDanger  = 25;
input stressBackwardation = 1.02;
input calmContango = 0.95;

input ivRankLookback = 252;
input premiumRichThreshold = 50;
input premiumCheapThreshold = 25;

input earningsBlackoutDays = 7;
input dividendWarnDays = 5;

input eventBlackout = no;
input allowLongDefinedRiskDuringEvents = no;
input requireBreakoutConfirmation = yes;
input enableAlerts = yes;

# ── REGIME SOURCE: the index ────────────────────────────────────────
def rClose = close(marketSymbol);
def indexPullOK = !IsNaN(rClose);
def rC = if indexPullOK then rClose else close;   # fail-soft

# ── INSTRUMENT SOURCE: the chart symbol ─────────────────────────────
def iC = close;
def iH = high;
def iL = low;
def instrumentDataOK = !IsNaN(iC) and !IsNaN(iH) and !IsNaN(iL);

# ── Trend (INDEX) ───────────────────────────────────────────────────
def smaFast   = Average(rC, fastTrendLength);
def smaMedium = Average(rC, mediumTrendLength);
def smaSlow   = Average(rC, slowTrendLength);

def strongBull = rC > smaFast and smaFast > smaMedium and smaMedium > smaSlow;
def strongBear = rC < smaFast and smaFast < smaMedium and smaMedium < smaSlow;
def mildBull   = rC > smaMedium and smaMedium > smaSlow;
def mildBear   = rC < smaMedium and smaMedium < smaSlow;

def trendScore =
    if strongBull then 4
    else if strongBear then -4
    else if mildBull then 2
    else if mildBear then -2
    else 0;

# ── Momentum (INDEX) ────────────────────────────────────────────────
def netChange = rC - rC[1];
def gain = WildersAverage(Max(netChange, 0), rsiLength);
def loss = WildersAverage(Max(-netChange, 0), rsiLength);
def rsi =
    if loss == 0 and gain == 0 then 50
    else if loss == 0 then 100
    else 100 - 100 / (1 + gain / loss);

def macdValue  = ExpAverage(rC, macdFast) - ExpAverage(rC, macdSlow);
def macdSignal = ExpAverage(macdValue, macdSignalLength);

def rsiVote  = if rsi >= 55 then 1 else if rsi <= 45 then -1 else 0;
def macdVote = if macdValue > macdSignal then 1 else if macdValue < macdSignal then -1 else 0;
def momentumScore = Max(-2, Min(2, rsiVote + macdVote));

# ── Regime score: SYMMETRIC, index-driven [-6,+6] ───────────────────
def regimeScore = Max(-6, Min(6, trendScore + momentumScore));
def bullishRegime = regimeScore >= 4;
def bearishRegime = regimeScore <= -4;

# ── LOCATION: the CHART SYMBOL's own levels (Teri) ──────────────────
def sellerLevel = Highest(iH[1], levelLookback);
def buyerLevel  = Lowest(iL[1], levelLookback);
def atr = Average(TrueRange(iH, iC, iL), atrLength);

def breakoutUp   = iC > sellerLevel;
def breakoutDown = iC < buyerLevel;

def roomUpATR   = if atr > 0 and !breakoutUp   then (sellerLevel - iC) / atr else Double.NaN;
def roomDownATR = if atr > 0 and !breakoutDown then (iC - buyerLevel) / atr else Double.NaN;

def bullHeadroomOK = !IsNaN(roomUpATR)   and roomUpATR   >= requiredHeadroomATR;
def bearHeadroomOK = !IsNaN(roomDownATR) and roomDownATR >= requiredHeadroomATR;

def bullBreakoutConfirmed = breakoutUp   and iC[1] > sellerLevel[1];
def bearBreakoutConfirmed = breakoutDown and iC[1] < buyerLevel[1];

def bullLocationOK =
    bullHeadroomOK or (breakoutUp and (!requireBreakoutConfirmation or bullBreakoutConfirmed));
def bearLocationOK =
    bearHeadroomOK or (breakoutDown and (!requireBreakoutConfirmation or bearBreakoutConfirmed));

def locationScore =
    if bullLocationOK and !bearLocationOK then 1
    else if bearLocationOK and !bullLocationOK then -1
    else 0;

# ── RELATIVE STRENGTH: instrument vs index (Teri buys leaders) ──────
def instROC  = if iC[rsLookback] > 0 then iC / iC[rsLookback] else Double.NaN;
def indexROC = if rC[rsLookback] > 0 then rC / rC[rsLookback] else Double.NaN;
def rsRatio  = if !IsNaN(instROC) and !IsNaN(indexROC) and indexROC != 0 then instROC / indexROC else Double.NaN;
def rsDataOK = !IsNaN(rsRatio);
def leading  = rsDataOK and rsRatio > 1.02;
def lagging  = rsDataOK and rsRatio < 0.98;

# ── Warm-up: test the COMPUTED SERIES ───────────────────────────────
def trendHistoryOK    = !IsNaN(smaSlow) and !IsNaN(smaMedium) and !IsNaN(smaFast);
def momentumHistoryOK = !IsNaN(macdSignal) and !IsNaN(rsi);
def levelHistoryOK    = !IsNaN(sellerLevel) and !IsNaN(buyerLevel) and !IsNaN(atr);
def regimeDataReady   = instrumentDataOK and trendHistoryOK and momentumHistoryOK and levelHistoryOK;

# ── Volatility: SPOT VIX term structure (NOT VX futures) ────────────
def vix   = close(vixSymbol);
def vix9d = close(vix9dSymbol);
def vixDataOK   = !IsNaN(vix);
def vix9dDataOK = !IsNaN(vix9d);

def stressRatio = if vixDataOK and vix9dDataOK and vix > 0 then vix9d / vix else Double.NaN;
def volStress   = vixDataOK and (vix >= vixDanger or (!IsNaN(stressRatio) and stressRatio > stressBackwardation));
def volCalm     = vixDataOK and vix < vixCaution and (IsNaN(stressRatio) or stressRatio < calmContango);

# ── EARNINGS / EX-DIV proximity (chart symbol; DAY aggregation) ─────
def chartAgg = GetAggregationPeriod();
def dayAgg = chartAgg == AggregationPeriod.DAY;

def erOffset  = GetEventOffset(Events.EARNINGS, 0);
def daysToER  = if IsNaN(erOffset) then Double.NaN else AbsValue(erOffset);
def erKnown   = dayAgg and !IsNaN(daysToER);
def erNear    = erKnown and daysToER <= earningsBlackoutDays;

def divOffset = GetEventOffset(Events.DIVIDEND, 0);
def daysToDiv = if IsNaN(divOffset) then Double.NaN else AbsValue(divOffset);
def divKnown  = dayAgg and !IsNaN(daysToDiv);
def divNear   = divKnown and daysToDiv <= dividendWarnDays;

# ── PERMISSION engine ───────────────────────────────────────────────
def shortPremiumAllowed =
    regimeDataReady and !eventBlackout and !volStress and !erNear;

def longDefinedRiskAllowed =
    regimeDataReady and (!eventBlackout or allowLongDefinedRiskDuringEvents);

def tradePermission =
    if !regimeDataReady then 0
    else if eventBlackout and !allowLongDefinedRiskDuringEvents then 0
    else if volStress then 1
    else if erNear then 1
    else 2;
# 0 = blocked | 1 = defensive only | 2 = selective

# ── Plots ───────────────────────────────────────────────────────────
plot Regime = regimeScore;
Regime.SetPaintingStrategy(PaintingStrategy.HISTOGRAM);
Regime.SetLineWeight(4);
Regime.AssignValueColor(
    if regimeScore >= 6 then Color.GREEN
    else if regimeScore >= 4 then Color.LIGHT_GREEN
    else if regimeScore <= -6 then Color.DARK_RED
    else if regimeScore <= -4 then Color.RED
    else Color.GRAY
);

plot ZeroLine = 0;
ZeroLine.SetDefaultColor(Color.GRAY);
ZeroLine.HideBubble();
ZeroLine.HideTitle();

plot BullThreshold = 4;
BullThreshold.SetDefaultColor(Color.GREEN);
BullThreshold.SetStyle(Curve.SHORT_DASH);
BullThreshold.HideBubble();
BullThreshold.HideTitle();

plot BearThreshold = -4;
BearThreshold.SetDefaultColor(Color.RED);
BearThreshold.SetStyle(Curve.SHORT_DASH);
BearThreshold.HideBubble();
BearThreshold.HideTitle();

# ── Labels ──────────────────────────────────────────────────────────
AddLabel(yes,
    (if chartAgg < AggregationPeriod.DAY then "MODE: INTRADAY"
     else if dayAgg then "MODE: DAILY"
     else "MODE: HIGHER TF") + "   REGIME SRC: " + marketSymbol + "   INSTRUMENT: " + GetSymbol(),
    Color.DARK_GRAY);

AddLabel(yes,
    if bullishRegime then "REGIME: BULLISH" else if bearishRegime then "REGIME: BEARISH" else "REGIME: NEUTRAL",
    if bullishRegime then Color.GREEN else if bearishRegime then Color.RED else Color.GRAY);

AddLabel(yes,
    if tradePermission == 0 then "PERMISSION: BLOCKED"
    else if tradePermission == 1 then "PERMISSION: DEFENSIVE ONLY"
    else "PERMISSION: SELECTIVE",
    if tradePermission == 0 then Color.DARK_ORANGE
    else if tradePermission == 1 then Color.YELLOW
    else Color.GREEN);

AddLabel(yes,
    "Score " + regimeScore + "  (T " + trendScore + " M " + momentumScore + " | L " + locationScore + ")",
    Color.WHITE);

AddLabel(!indexPullOK,
    "NOTE: '" + marketSymbol + "' did not resolve - regime fell back to chart symbol",
    Color.ORANGE);

AddLabel(!regimeDataReady,
    "WARM-UP: insufficient history (need " + slowTrendLength + " bars)",
    Color.ORANGE);

AddLabel(yes,
    if !vixDataOK then "VIX spot: unavailable"
    else "VIX spot " + Round(vix, 2) + "  9D/30D " +
        (if IsNaN(stressRatio) then "N/A" else AsText(Round(stressRatio, 2))) +
        (if volStress then " STRESS" else if volCalm then " calm" else ""),
    if !vixDataOK then Color.GRAY
    else if volStress then Color.RED
    else if volCalm then Color.GREEN
    else Color.YELLOW);

# Relative strength - instrument vs index
AddLabel(yes,
    if !rsDataOK then "RS vs " + marketSymbol + ": N/A"
    else "RS vs " + marketSymbol + " " + AsText(Round(rsRatio, 3)) +
        (if leading then " LEADING" else if lagging then " lagging" else " inline"),
    if leading then Color.GREEN else if lagging then Color.RED else Color.LIGHT_GRAY);

# Earnings / ex-dividend gates
AddLabel(yes,
    if !dayAgg then "ER/DIV: switch to DAILY chart to read"
    else if !erKnown then "ER: none in window"
    else "ER in " + Round(daysToER, 0) + "d" + (if erNear then " - NO SHORT PREMIUM" else ""),
    if erNear then Color.RED else if erKnown then Color.YELLOW else Color.GRAY);

AddLabel(divKnown and divNear,
    "EX-DIV in " + Round(daysToDiv, 0) + "d - early-assignment risk on short calls",
    Color.ORANGE);

# Instrument IV (premium bias must reflect the NAME being traded)
def ivInst = imp_volatility();
def ivIdx  = imp_volatility(marketSymbol);
def iv     = if !IsNaN(ivInst) then ivInst else ivIdx;
def ivFromInstrument = !IsNaN(ivInst);

def ivHigh  = Highest(iv, ivRankLookback);
def ivLow   = Lowest(iv, ivRankLookback);
def ivRange = ivHigh - ivLow;
def ivDataOK = !IsNaN(iv) and !IsNaN(ivHigh) and !IsNaN(ivLow) and ivRange > 0;

def ivPosition   = if ivDataOK then 100 * (iv - ivLow) / ivRange else Double.NaN;
def premiumRich  = ivDataOK and ivPosition >= premiumRichThreshold;
def premiumCheap = ivDataOK and ivPosition <= premiumCheapThreshold;

AddLabel(yes,
    (if !ivDataOK then "IV Position: unavailable - no premium bias"
     else "IV Position " + Round(ivPosition, 0) +
        (if premiumRich then " - rich" else if premiumCheap then " - cheap" else " - mixed")) +
    "  [src: " + (if ivFromInstrument then GetSymbol() else marketSymbol) + "]",
    if !ivDataOK then Color.GRAY
    else if premiumRich then Color.ORANGE
    else if premiumCheap then Color.CYAN
    else Color.LIGHT_GRAY);

AddLabel(yes,
    if breakoutUp then "LOCATION: " + GetSymbol() + " above prior seller level"
    else if breakoutDown then "LOCATION: " + GetSymbol() + " below prior buyer level"
    else GetSymbol() + " headroom: +" + (if IsNaN(roomUpATR) then "N/A" else AsText(Round(roomUpATR, 1))) +
         " / -" + (if IsNaN(roomDownATR) then "N/A" else AsText(Round(roomDownATR, 1))) + " ATR",
    if breakoutUp then Color.LIGHT_GREEN else if breakoutDown then Color.LIGHT_RED else Color.LIGHT_GRAY);


# ═══════════════════════════════════════════════════════════════════
# v4.5 ADDITION — on-chart EIGHT-POINT ODDS ENHANCER (Teri zone quality)
# Scores the MOST RECENT detected zone on the chart symbol:
#   base candles + departure speed + freshness + reward:risk -> 0..8
# 7-8 PRIMARY (full size) | 5-6 SECONDARY (half, confirm) | 0-4 SKIP
# This mirrors qm/iwt_zones.py in the app. Proxy: mechanical zone
# detection is NOT validated against Teri's hand-marked zones.
# ═══════════════════════════════════════════════════════════════════
input enableOddsEnhancer = yes;
input baseMaxCandles = 8;
input smallBodyATR = 0.45;       # a "base" candle: |body| < 0.45 ATR
input avgDepartureATR = 0.75;    # departure body >= 0.75 ATR = average
input fastDepartureATR = 1.50;   # >= 1.50 ATR = fast

# Departure bar = most recent bar whose body exceeds the average threshold
def instBody = AbsValue(iC - open);
def isDeparture = atr > 0 and instBody >= avgDepartureATR * atr;
# bars since the last departure (0 = current bar is the departure)
def barsSinceDep = if isDeparture then 0 else barsSinceDep[1] + 1;

# Departure strength (ATR multiples) at that most-recent departure bar
def depStrengthRaw = if atr > 0 then instBody / atr else 0;
def depStrength = if isDeparture then depStrengthRaw else depStrength[1];
def depIsUp = if isDeparture then (iC > open) else depIsUp[1];

# Count base candles immediately BEFORE the departure: consecutive small
# bodies. Look back from the bar just before the departure.
def baseCount = fold bi = 1 to baseMaxCandles + 1 with cnt = 0 while
    (GetValue(atr, barsSinceDep + bi) > 0 and
     AbsValue(GetValue(iC, barsSinceDep + bi) - GetValue(open, barsSinceDep + bi))
        < smallBodyATR * GetValue(atr, barsSinceDep + bi))
    do cnt + 1;

# -- Score components (mirror qm/iwt_zones.py) --
def sBase = if baseCount <= 2 then 2 else if baseCount <= 4 then 1 else 0;
def sDep  = if depStrength >= fastDepartureATR then 2 else if depStrength >= avgDepartureATR then 1 else 0;
# Freshness: revisits to the departure zone since it formed (exclude current)
def depZoneHi = GetValue(iH, barsSinceDep);
def depZoneLo = GetValue(iL, barsSinceDep);
def revisit = barsSinceDep > 0 and iH >= depZoneLo and iL <= depZoneHi;
def revisitCount = if isDeparture then 0 else revisitCount[1] + (if revisit then 1 else 0);
def sFresh = if revisitCount == 0 then 2 else if revisitCount == 1 then 1 else 0;
# Reward:risk uses the location engine's headroom as proxy (room to opposing level)
def rrProxy = if depIsUp then (if !IsNaN(roomUpATR) then roomUpATR / Max(requiredHeadroomATR, 0.01) else 0)
              else (if !IsNaN(roomDownATR) then roomDownATR / Max(requiredHeadroomATR, 0.01) else 0);
def sRR = if rrProxy >= 3.0 then 2 else if rrProxy >= 2.0 then 1 else 0;

def oddsScore = sBase + sDep + sFresh + sRR;
def oddsCohort = if oddsScore >= 7 then 2 else if oddsScore >= 5 then 1 else 0;  # 2=primary 1=secondary 0=skip

AddLabel(enableOddsEnhancer and !IsNaN(depStrength),
    "ZONE ODDS " + oddsScore + "/8 " +
    (if oddsCohort == 2 then "PRIMARY (full)" else if oddsCohort == 1 then "SECONDARY (half)" else "SKIP") +
    "  [base " + baseCount + " dep " + Round(depStrength, 1) + "ATR fresh " + revisitCount + " rr " + Round(rrProxy, 1) + "]",
    if oddsCohort == 2 then Color.GREEN else if oddsCohort == 1 then Color.ORANGE else Color.GRAY);

# ── Decision support ────────────────────────────────────────────────
AddLabel(yes,
    if !regimeDataReady then "SETUP: DATA WARM-UP - wait"
    else if eventBlackout and !allowLongDefinedRiskDuringEvents then "SETUP: WAIT - event blackout"
    else if volStress then "SETUP: PRESERVE CAPITAL - volatility stress"
    else if erNear then "SETUP: EARNINGS IN " + Round(daysToER, 0) + "d - no short premium; long defined-risk only"
    else if bullishRegime and lagging then "SETUP: WAIT - bull regime but this name LAGS the index"
    else if bearishRegime and leading then "SETUP: WAIT - bear regime but this name LEADS (poor short)"
    else if bullishRegime and !bullLocationOK then "SETUP: WAIT - bull regime, poor entry location on " + GetSymbol()
    else if bearishRegime and !bearLocationOK then "SETUP: WAIT - bear regime, poor entry location on " + GetSymbol()
    else if bullishRegime and premiumRich and shortPremiumAllowed then "BIAS: bullish defined-risk premium candidate"
    else if bullishRegime and premiumCheap and longDefinedRiskAllowed then "BIAS: bullish debit candidate"
    else if bearishRegime and premiumCheap and longDefinedRiskAllowed then "BIAS: bearish debit / hedge candidate"
    else if bearishRegime and premiumRich and shortPremiumAllowed then "BIAS: bearish defined-risk premium candidate"
    else "SETUP: WAIT - no strong asymmetry",
    Color.CYAN);

# ── Edge-triggered alerts ───────────────────────────────────────────
def enteredBullish    = bullishRegime and !bullishRegime[1];
def enteredBearish    = bearishRegime and !bearishRegime[1];
def enteredVolStress  = volStress and !volStress[1];
def enteredERWindow   = erNear and !erNear[1];
def permissionBlocked = tradePermission == 0 and tradePermission[1] != 0;

Alert(enableAlerts and enteredBullish, "TeriQuantumOsc: bullish regime entered", Alert.BAR, Sound.Ding);
Alert(enableAlerts and enteredBearish, "TeriQuantumOsc: bearish regime entered", Alert.BAR, Sound.Ding);
Alert(enableAlerts and enteredVolStress, "TeriQuantumOsc: volatility stress entered", Alert.BAR, Sound.Bell);
Alert(enableAlerts and enteredERWindow, "TeriQuantumOsc: earnings window - short premium blocked", Alert.BAR, Sound.Bell);
Alert(enableAlerts and permissionBlocked, "TeriQuantumOsc: trading permission blocked", Alert.BAR, Sound.Bell);
