# ═══════════════════════════════════════════════════════════════════
# TeriQuantumMaestro v6 - THE ALL-IN-ONE
#
# The culmination: one upper study that unifies every piece built across
# the whole TeriQuantumOsc / TeriQuantumZones line, aligned to Teri
# Ijeoma's canonical Trade & Travel course documents.
#
# WHAT IT DOES, all on the price chart:
#   ZONES      - draws buyer/seller zones (U/Chair), non-repainting,
#                with proximal/distal lines and shaded clouds.
#   REGIME     - index-driven bull/bear/neutral (should I trade at all).
#   LOCATION   - this name's zones + headroom (entry quality).
#   PREMIUM    - this name's IV position (rich/cheap for its options).
#   PERMISSION - event blackout / vol stress / earnings / ex-div gates.
#   ODDS       - the eight-point enhancer per zone: base + departure +
#                freshness + reward:risk -> 0-8, PRIMARY/SECONDARY/SKIP.
#   RELATIVE STRENGTH - instrument vs index (Teri buys leaders).
#   CANONICAL GATES - RR ">3 TAKE" rule; order-type intent (entry LIMIT,
#                exit LIMIT, stop STOP-MARKET) from the IWT BUY worksheet.
#
# WHY UPPER (not a lower oscillator): a ThinkScript study is upper OR
# lower, never both. The zones-on-price ARE Teri's method made visible,
# so v6 lives on the chart; the regime score survives as a label chip.
# For the histogram panel, run TeriQuantumOsc_v4_6 alongside.
#
# CANONICAL ALIGNMENT (verified against course docs):
#   - Freshness: 0 visits=2, 1 visit=1, MORE THAN 1 ("1+ VISITS")=0.
#   - Stop = distal_BZ - 20% ATR ("subtract from bottom of buyer's level").
#   - Cohorts: 7-8 primary / 5-6 secondary / 0-4 skip.
#   - Base >=5 tight candles = strongest (treated as full 4-base).
#
# Mechanical approximation of a discretionary method. Zone detection is
# NOT validated against Teri's hand-marked zones - a decision aid, not a
# signal. Quantum Maestro app engine remains canonical. No 0DTE, defined
# risk only, no short premium into events. Selects no strikes, no orders.
# ═══════════════════════════════════════════════════════════════════

declare upper;

# --------------------------------------------------------------- INPUTS
input marketSymbol = "SPX";
input vixSymbol = "VIX";
input vix9dSymbol = "VIX9D";

input fastTrendLength = 20;
input mediumTrendLength = 50;
input slowTrendLength = 200;

input rsiLength = 14;
input macdFast = 12;
input macdSlow = 26;
input macdSignalLength = 9;

input atrLength = 14;
input atrStopBufferPercent = 20.0;

input rsLookback = 60;
input leadingThreshold = 1.02;
input laggingThreshold = 0.98;

input vixCaution = 18;
input vixDanger = 25;
input stressBackwardation = 1.02;
input calmContango = 0.95;

input ivRankLookback = 252;
input premiumRichThreshold = 50;
input premiumCheapThreshold = 25;

input earningsBlackoutDays = 7;
input dividendWarningDays = 5;

input eventBlackout = no;
input allowLongDefinedRiskDuringEvents = no;

input baseBodyMaxATR = 0.45;
input minimumDepartureATR = 0.75;
input strongDepartureATR = 1.50;
input useTrueRangeForDeparture = no;

input requireTwoClosesToBreakZone = yes;
input maximumZoneWidthATR = 2.00;

# Canonical cohort thresholds (7-8 primary / 5-6 secondary / 0-4 skip)
# SOURCE CONFLICT in Teri's materials, both encoded. STRICT default.
input useCourseBands = no;   # yes = 6/4 per the entry-strategy PDF
def primaryMinScore   = if useCourseBands then 6 else 7;
def secondaryMinScore = if useCourseBands then 4 else 5;
input primaryMinRR = 3.0;
input secondaryMinRR = 2.0;

input showZones = yes;         # zone lines AND their shaded clouds
input showTradeLevels = yes;
input showLabels = yes;
input enableAlerts = yes;

# --------------------------------------------------------- DATA SOURCES
def rCloseRaw = close(marketSymbol);
def indexDataOK = !IsNaN(rCloseRaw);
def rC = if indexDataOK then rCloseRaw else close;

def iO = open;
def iH = high;
def iL = low;
def iC = close;

def tr = TrueRange(iH, iC, iL);
def atr = Average(tr, atrLength);
def atrBuffer = atr * atrStopBufferPercent / 100;

def instrumentDataOK = !IsNaN(iO) and !IsNaN(iH) and !IsNaN(iL) and !IsNaN(iC) and atr > 0;

# --------------------------------------------------------- INDEX REGIME
def smaFast = Average(rC, fastTrendLength);
def smaMedium = Average(rC, mediumTrendLength);
def smaSlow = Average(rC, slowTrendLength);

def strongBull = rC > smaFast and smaFast > smaMedium and smaMedium > smaSlow;
def mildBull = rC > smaMedium and smaMedium > smaSlow and !strongBull;
def strongBear = rC < smaFast and smaFast < smaMedium and smaMedium < smaSlow;
def mildBear = rC < smaMedium and smaMedium < smaSlow and !strongBear;

def trendScore =
    if strongBull then 4 else if mildBull then 2
    else if strongBear then -4 else if mildBear then -2 else 0;

def netChange = rC - rC[1];
def averageGain = WildersAverage(Max(netChange, 0), rsiLength);
def averageLoss = WildersAverage(Max(-netChange, 0), rsiLength);
def regimeRSI =
    if averageGain == 0 and averageLoss == 0 then 50
    else if averageLoss == 0 then 100
    else 100 - 100 / (1 + averageGain / averageLoss);

def macdValue = ExpAverage(rC, macdFast) - ExpAverage(rC, macdSlow);
def macdSignal = ExpAverage(macdValue, macdSignalLength);
def rsiVote = if regimeRSI >= 55 then 1 else if regimeRSI <= 45 then -1 else 0;
def macdVote = if macdValue > macdSignal then 1 else if macdValue < macdSignal then -1 else 0;
def momentumScore = Max(-2, Min(2, rsiVote + macdVote));

def regimeScore = Max(-6, Min(6, trendScore + momentumScore));
def bullishRegime = regimeScore >= 4;
def bearishRegime = regimeScore <= -4;

# ------------------------------------------------------ RELATIVE STRENGTH
def instrumentROC = if iC[rsLookback] > 0 then iC / iC[rsLookback] else Double.NaN;
def indexROC = if rC[rsLookback] > 0 then rC / rC[rsLookback] else Double.NaN;
def rsRatio = if !IsNaN(instrumentROC) and !IsNaN(indexROC) and indexROC != 0 then instrumentROC / indexROC else Double.NaN;
def rsDataOK = !IsNaN(rsRatio);
def leading = rsDataOK and rsRatio > leadingThreshold;
def lagging = rsDataOK and rsRatio < laggingThreshold;

# ---------------------------------------------------------- VOLATILITY
def vix = close(vixSymbol);
def vix9d = close(vix9dSymbol);
def vixDataOK = !IsNaN(vix);
def vix9dDataOK = !IsNaN(vix9d);
def stressRatio = if vixDataOK and vix9dDataOK and vix > 0 then vix9d / vix else Double.NaN;
def volStress = vixDataOK and (vix >= vixDanger or (!IsNaN(stressRatio) and stressRatio > stressBackwardation));
def volCalm = vixDataOK and vix < vixCaution and (IsNaN(stressRatio) or stressRatio < calmContango);

# ------------------------------------------------------ EARNINGS / DIV
def chartAggregation = GetAggregationPeriod();
def dailyAggregation = chartAggregation == AggregationPeriod.DAY;
def earningsOffset = GetEventOffset(Events.EARNINGS, 0);
def daysToEarnings = if IsNaN(earningsOffset) then Double.NaN else AbsValue(earningsOffset);
def earningsKnown = dailyAggregation and !IsNaN(daysToEarnings);
def earningsNear = earningsKnown and daysToEarnings <= earningsBlackoutDays;
def dividendOffset = GetEventOffset(Events.DIVIDEND, 0);
def daysToDividend = if IsNaN(dividendOffset) then Double.NaN else AbsValue(dividendOffset);
def dividendKnown = dailyAggregation and !IsNaN(daysToDividend);
def dividendNear = dividendKnown and daysToDividend <= dividendWarningDays;

# ---------------------------------------------- BASE-CANDLE DETECTION
# FIX 1: count compact candles and CLAMP to [0,4] - a 5+ base is a strong
# 4-base, never discarded. Departure is the current bar; base is behind it.
def body = AbsValue(iC - iO);
def compact1 = atr[1] > 0 and AbsValue(iC[1] - iO[1]) <= baseBodyMaxATR * atr[1];
def compact2 = atr[2] > 0 and AbsValue(iC[2] - iO[2]) <= baseBodyMaxATR * atr[2];
def compact3 = atr[3] > 0 and AbsValue(iC[3] - iO[3]) <= baseBodyMaxATR * atr[3];
def compact4 = atr[4] > 0 and AbsValue(iC[4] - iO[4]) <= baseBodyMaxATR * atr[4];

# consecutive compact candles behind the departure, clamped to 4
def baseCount =
    if !compact1 then 0
    else if !compact2 then 1
    else if !compact3 then 2
    else if !compact4 then 3
    else 4;  # 4 OR MORE compact candles -> treat as a full 4-candle base

# --------------------------------------------------------- BASE BOUNDS
def bodyHigh1 = Max(iO[1], iC[1]);
def bodyHigh2 = Max(iO[2], iC[2]);
def bodyHigh3 = Max(iO[3], iC[3]);
def bodyHigh4 = Max(iO[4], iC[4]);
def bodyLow1 = Min(iO[1], iC[1]);
def bodyLow2 = Min(iO[2], iC[2]);
def bodyLow3 = Min(iO[3], iC[3]);
def bodyLow4 = Min(iO[4], iC[4]);

def baseHighestBody =
    if baseCount == 1 then bodyHigh1
    else if baseCount == 2 then Max(bodyHigh1, bodyHigh2)
    else if baseCount == 3 then Max(Max(bodyHigh1, bodyHigh2), bodyHigh3)
    else if baseCount == 4 then Max(Max(bodyHigh1, bodyHigh2), Max(bodyHigh3, bodyHigh4))
    else Double.NaN;
def baseLowestBody =
    if baseCount == 1 then bodyLow1
    else if baseCount == 2 then Min(bodyLow1, bodyLow2)
    else if baseCount == 3 then Min(Min(bodyLow1, bodyLow2), bodyLow3)
    else if baseCount == 4 then Min(Min(bodyLow1, bodyLow2), Min(bodyLow3, bodyLow4))
    else Double.NaN;
def baseHighestWick =
    if baseCount == 1 then iH[1]
    else if baseCount == 2 then Max(iH[1], iH[2])
    else if baseCount == 3 then Max(Max(iH[1], iH[2]), iH[3])
    else if baseCount == 4 then Max(Max(iH[1], iH[2]), Max(iH[3], iH[4]))
    else Double.NaN;
def baseLowestWick =
    if baseCount == 1 then iL[1]
    else if baseCount == 2 then Min(iL[1], iL[2])
    else if baseCount == 3 then Min(Min(iL[1], iL[2]), iL[3])
    else if baseCount == 4 then Min(Min(iL[1], iL[2]), Min(iL[3], iL[4]))
    else Double.NaN;

def candidateBuyerProximal = baseHighestBody;
def candidateBuyerDistal = baseLowestWick;
def candidateSellerProximal = baseLowestBody;
def candidateSellerDistal = baseHighestWick;

# ------------------------------------------------------ DEPARTURE
# FIX 6: default body; TR mode counts wicks (a rejection bar can look like a
# false departure), so it is opt-in and surfaced in a label.
def departureMeasure = if useTrueRangeForDeparture then tr else body;
def departureATR = if atr > 0 then departureMeasure / atr else 0;
def bullishDeparture = iC > iO and departureATR >= minimumDepartureATR and iC > baseHighestBody;
def bearishDeparture = iC < iO and departureATR >= minimumDepartureATR and iC < baseLowestBody;

# ---------------------------------------------------- U / CHAIR APPROACH
def preBaseClose = if baseCount > 0 then GetValue(iC, baseCount + 1) else Double.NaN;
def oldestBaseClose = if baseCount > 0 then GetValue(iC, baseCount) else Double.NaN;
def buyerU = baseCount > 0 and preBaseClose > oldestBaseClose;
def sellerInvertedU = baseCount > 0 and preBaseClose < oldestBaseClose;

# ------------------------------------------------- ZONE VALIDATION
def buyerWidth = candidateBuyerProximal - candidateBuyerDistal;
def sellerWidth = candidateSellerDistal - candidateSellerProximal;
def buyerWidthOK = baseCount >= 1 and buyerWidth > 0 and buyerWidth <= maximumZoneWidthATR * atr;
def sellerWidthOK = baseCount >= 1 and sellerWidth > 0 and sellerWidth <= maximumZoneWidthATR * atr;
def newBuyerZone = instrumentDataOK and bullishDeparture and buyerWidthOK;
def newSellerZone = instrumentDataOK and bearishDeparture and sellerWidthOK;

# ------------------------------------------- STORE LATEST BUYER ZONE
rec buyerProximal = CompoundValue(1, if newBuyerZone then candidateBuyerProximal else buyerProximal[1], Double.NaN);
rec buyerDistal = CompoundValue(1, if newBuyerZone then candidateBuyerDistal else buyerDistal[1], Double.NaN);
rec buyerBaseCount = CompoundValue(1, if newBuyerZone then baseCount else buyerBaseCount[1], 0);
rec buyerDepartureATR = CompoundValue(1, if newBuyerZone then departureATR else buyerDepartureATR[1], 0);
rec buyerPatternCode = CompoundValue(1, if newBuyerZone then (if buyerU then 1 else 2) else buyerPatternCode[1], 0);

# ------------------------------------------- STORE LATEST SELLER ZONE
rec sellerProximal = CompoundValue(1, if newSellerZone then candidateSellerProximal else sellerProximal[1], Double.NaN);
rec sellerDistal = CompoundValue(1, if newSellerZone then candidateSellerDistal else sellerDistal[1], Double.NaN);
rec sellerBaseCount = CompoundValue(1, if newSellerZone then baseCount else sellerBaseCount[1], 0);
rec sellerDepartureATR = CompoundValue(1, if newSellerZone then departureATR else sellerDepartureATR[1], 0);
rec sellerPatternCode = CompoundValue(1, if newSellerZone then (if sellerInvertedU then 1 else 2) else sellerPatternCode[1], 0);

# --------------------------------------------------------- ZONE BREAKS
def buyerBreakTwo = !IsNaN(buyerDistal) and iC < buyerDistal and iC[1] < buyerDistal[1];
def buyerBreakOne = !IsNaN(buyerDistal) and iC < buyerDistal;
def buyerBroken = if requireTwoClosesToBreakZone then buyerBreakTwo else buyerBreakOne;
def sellerBreakTwo = !IsNaN(sellerDistal) and iC > sellerDistal and iC[1] > sellerDistal[1];
def sellerBreakOne = !IsNaN(sellerDistal) and iC > sellerDistal;
def sellerBroken = if requireTwoClosesToBreakZone then sellerBreakTwo else sellerBreakOne;

rec buyerActive = CompoundValue(1, if newBuyerZone then 1 else if buyerBroken then 0 else buyerActive[1], 0);
rec sellerActive = CompoundValue(1, if newSellerZone then 1 else if sellerBroken then 0 else sellerActive[1], 0);

# ----------------------------------------------- DISTINCT REVISITS
def buyerOverlap = buyerActive and iH >= buyerDistal and iL <= buyerProximal;
def sellerOverlap = sellerActive and iH >= sellerProximal and iL <= sellerDistal;
def buyerEntered = buyerOverlap and !buyerOverlap[1] and !newBuyerZone;
def sellerEntered = sellerOverlap and !sellerOverlap[1] and !newSellerZone;
rec buyerVisits = CompoundValue(1, if newBuyerZone then 0 else if buyerEntered then buyerVisits[1] + 1 else buyerVisits[1], 0);
rec sellerVisits = CompoundValue(1, if newSellerZone then 0 else if sellerEntered then sellerVisits[1] + 1 else sellerVisits[1], 0);

# ------------------------------------------- ENTRY / STOP / TARGET / RR
def longEntry = if buyerActive then buyerProximal else Double.NaN;
def longStop = if buyerActive then buyerDistal - atrBuffer else Double.NaN;
def longTarget = if sellerActive and sellerProximal > longEntry then sellerProximal else Double.NaN;
def longRisk = if !IsNaN(longEntry) and !IsNaN(longStop) then longEntry - longStop else Double.NaN;
def longReward = if !IsNaN(longTarget) and !IsNaN(longEntry) then longTarget - longEntry else Double.NaN;
def longRR = if longRisk > 0 and longReward > 0 then longReward / longRisk else Double.NaN;

def shortEntry = if sellerActive then sellerProximal else Double.NaN;
def shortStop = if sellerActive then sellerDistal + atrBuffer else Double.NaN;
def shortTarget = if buyerActive and buyerProximal < shortEntry then buyerProximal else Double.NaN;
def shortRisk = if !IsNaN(shortEntry) and !IsNaN(shortStop) then shortStop - shortEntry else Double.NaN;
def shortReward = if !IsNaN(shortTarget) and !IsNaN(shortEntry) then shortEntry - shortTarget else Double.NaN;
def shortRR = if shortRisk > 0 and shortReward > 0 then shortReward / shortRisk else Double.NaN;

# ------------------------------------------------ EIGHT-POINT SCORE
def buyerBaseScore = if buyerBaseCount <= 0 then 0 else if buyerBaseCount <= 2 then 2 else if buyerBaseCount <= 4 then 1 else 0;
def sellerBaseScore = if sellerBaseCount <= 0 then 0 else if sellerBaseCount <= 2 then 2 else if sellerBaseCount <= 4 then 1 else 0;
def buyerDepScore = if buyerDepartureATR >= strongDepartureATR then 2 else if buyerDepartureATR >= minimumDepartureATR then 1 else 0;
def sellerDepScore = if sellerDepartureATR >= strongDepartureATR then 2 else if sellerDepartureATR >= minimumDepartureATR then 1 else 0;
def buyerFreshScore = if buyerVisits == 0 then 2 else if buyerVisits == 1 then 1 else 0;
def sellerFreshScore = if sellerVisits == 0 then 2 else if sellerVisits == 1 then 1 else 0;
def buyerRRScore = if IsNaN(longRR) then 0 else if longRR > 3 then 2 else if longRR >= 2 then 1 else 0;
def sellerRRScore = if IsNaN(shortRR) then 0 else if shortRR > 3 then 2 else if shortRR >= 2 then 1 else 0;

def buyerScore = buyerBaseScore + buyerDepScore + buyerFreshScore + buyerRRScore;
def sellerScore = sellerBaseScore + sellerDepScore + sellerFreshScore + sellerRRScore;

# FIX 2+3: canonical cohorts. PRIMARY 7-8 + RR>=3; SECONDARY 5-6 + RR>=2
# + confirmation. The score gate and RR gate now AGREE.
def buyerPrimary = buyerActive and buyerScore >= primaryMinScore and !IsNaN(longRR) and longRR >= primaryMinRR;
def buyerSecondary = buyerActive and buyerScore >= secondaryMinScore and buyerScore < primaryMinScore and !IsNaN(longRR) and longRR >= secondaryMinRR;
def buyerSkip = !(buyerPrimary or buyerSecondary);
def sellerPrimary = sellerActive and sellerScore >= primaryMinScore and !IsNaN(shortRR) and shortRR >= primaryMinRR;
def sellerSecondary = sellerActive and sellerScore >= secondaryMinScore and sellerScore < primaryMinScore and !IsNaN(shortRR) and shortRR >= secondaryMinRR;
def sellerSkip = !(sellerPrimary or sellerSecondary);

# Confirmation: price tested the zone then closed back on the working side
def buyerTurnConfirmed = buyerSecondary and buyerOverlap[1] and iC > buyerProximal and iC > iO;
def sellerTurnConfirmed = sellerSecondary and sellerOverlap[1] and iC < sellerProximal and iC < iO;

# --------------------------------------------------- INSTRUMENT IV
def ivInstrument = imp_volatility();
def ivIndex = imp_volatility(marketSymbol);
def iv = if !IsNaN(ivInstrument) then ivInstrument else ivIndex;
def ivFromInstrument = !IsNaN(ivInstrument);
def ivHigh = Highest(iv, ivRankLookback);
def ivLow = Lowest(iv, ivRankLookback);
def ivRange = ivHigh - ivLow;
def ivDataOK = !IsNaN(iv) and !IsNaN(ivHigh) and !IsNaN(ivLow) and ivRange > 0;
def ivPosition = if ivDataOK then 100 * (iv - ivLow) / ivRange else Double.NaN;
def premiumRich = ivDataOK and ivPosition >= premiumRichThreshold;
def premiumCheap = ivDataOK and ivPosition <= premiumCheapThreshold;

# ------------------------------------------------- PERMISSION ENGINE
def historyReady = !IsNaN(smaSlow) and !IsNaN(macdSignal) and instrumentDataOK;
def shortPremiumAllowed = historyReady and !eventBlackout and !volStress and !earningsNear;
def longDefinedRiskAllowed = historyReady and (!eventBlackout or allowLongDefinedRiskDuringEvents);
def tradePermission =
    if !historyReady then 0
    else if eventBlackout and !allowLongDefinedRiskDuringEvents then 0
    else if volStress then 1
    else if earningsNear then 1
    else 2;

# --------------------------------------------- SETUP QUALIFICATION
# FIX 3+5: primary OR (secondary+confirmed) qualifies. Bearish long puts
# use longDefinedRiskAllowed (debit, allowed defensively); bearish CREDIT
# is only surfaced when shortPremiumAllowed (checked in the label).
def longQualified = buyerPrimary or buyerTurnConfirmed;
def shortQualified = sellerPrimary or sellerTurnConfirmed;
def bullishSetup = bullishRegime and !lagging and longQualified and longDefinedRiskAllowed;
def bearishSetup = bearishRegime and !leading and shortQualified and longDefinedRiskAllowed;

# --------------------------------------------------------- ZONE PLOTS
# FIX 4: plot only where the zone is active -> non-repainting history.
plot BuyerProx = if showZones and buyerActive then buyerProximal else Double.NaN;
BuyerProx.SetDefaultColor(Color.GREEN); BuyerProx.SetLineWeight(2); BuyerProx.SetStyle(Curve.SHORT_DASH);
plot BuyerDist = if showZones and buyerActive then buyerDistal else Double.NaN;
BuyerDist.SetDefaultColor(Color.DARK_GREEN); BuyerDist.SetLineWeight(2);
plot SellerProx = if showZones and sellerActive then sellerProximal else Double.NaN;
SellerProx.SetDefaultColor(Color.RED); SellerProx.SetLineWeight(2); SellerProx.SetStyle(Curve.SHORT_DASH);
plot SellerDist = if showZones and sellerActive then sellerDistal else Double.NaN;
SellerDist.SetDefaultColor(Color.DARK_RED); SellerDist.SetLineWeight(2);

# AddCloud cannot take rec-derived expressions. The plots below already carry
# the showZones+buyerActive gating (they are NaN when the zone is inactive), so
# pass the PLOTS directly - do NOT re-wrap them in an if over buyerActive (a rec),
# which is what caused "recs are not used inside addcloud".
AddCloud(BuyerProx, BuyerDist, Color.LIGHT_GREEN, Color.LIGHT_GREEN);
AddCloud(SellerDist, SellerProx, Color.LIGHT_RED, Color.LIGHT_RED);

plot LongStopLine = if showTradeLevels and buyerActive and bullishRegime then longStop else Double.NaN;
LongStopLine.SetDefaultColor(Color.DARK_ORANGE); LongStopLine.SetStyle(Curve.LONG_DASH);
plot LongTargetLine = if showTradeLevels and buyerActive and bullishRegime and !IsNaN(longTarget) then longTarget else Double.NaN;
LongTargetLine.SetDefaultColor(Color.CYAN); LongTargetLine.SetStyle(Curve.LONG_DASH);
plot ShortStopLine = if showTradeLevels and sellerActive and bearishRegime then shortStop else Double.NaN;
ShortStopLine.SetDefaultColor(Color.DARK_ORANGE); ShortStopLine.SetStyle(Curve.LONG_DASH);
plot ShortTargetLine = if showTradeLevels and sellerActive and bearishRegime and !IsNaN(shortTarget) then shortTarget else Double.NaN;
ShortTargetLine.SetDefaultColor(Color.CYAN); ShortTargetLine.SetStyle(Curve.LONG_DASH);

# ------------------------------------------------------------- LABELS
AddLabel(showLabels,
    "REGIME " + (if bullishRegime then "BULLISH" else if bearishRegime then "BEARISH" else "NEUTRAL") +
    " | Score " + regimeScore + " | Src " + marketSymbol,
    if bullishRegime then Color.GREEN else if bearishRegime then Color.RED else Color.GRAY);

AddLabel(showLabels,
    if tradePermission == 0 then "PERMISSION: BLOCKED"
    else if tradePermission == 1 then "PERMISSION: DEFENSIVE ONLY" else "PERMISSION: SELECTIVE",
    if tradePermission == 0 then Color.DARK_ORANGE else if tradePermission == 1 then Color.YELLOW else Color.GREEN);

AddLabel(showLabels and useTrueRangeForDeparture,
    "CAUTION: TR departure mode - long-wick rejection bars may score as departures", Color.ORANGE);

AddLabel(showLabels,
    if !rsDataOK then "RS vs " + marketSymbol + ": N/A"
    else "RS vs " + marketSymbol + ": " + AsText(Round(rsRatio, 3)) +
         (if leading then " LEADING" else if lagging then " LAGGING" else " INLINE"),
    if leading then Color.GREEN else if lagging then Color.RED else Color.LIGHT_GRAY);

AddLabel(showLabels,
    if !vixDataOK then "VIX: unavailable"
    else "VIX " + AsText(Round(vix, 2)) + " | 9D/30D " +
         (if IsNaN(stressRatio) then "N/A" else AsText(Round(stressRatio, 2))) +
         (if volStress then " STRESS" else if volCalm then " CALM" else ""),
    if !vixDataOK then Color.GRAY else if volStress then Color.RED else if volCalm then Color.GREEN else Color.YELLOW);

AddLabel(showLabels,
    if !dailyAggregation then "ER/DIV: use DAILY chart"
    else if earningsNear then "EARNINGS IN " + AsText(Round(daysToEarnings, 0)) + "d - NO SHORT PREMIUM"
    else if earningsKnown then "EARNINGS IN " + AsText(Round(daysToEarnings, 0)) + "d"
    else "EARNINGS: not detected",
    if earningsNear then Color.RED else if earningsKnown then Color.YELLOW else Color.GRAY);

AddLabel(showLabels and dividendNear,
    "EX-DIV IN " + AsText(Round(daysToDividend, 0)) + "d - short-call assignment risk", Color.ORANGE);

AddLabel(showLabels,
    if !ivDataOK then "IV POSITION: unavailable"
    else "IV POSITION " + AsText(Round(ivPosition, 0)) +
         (if premiumRich then " RICH" else if premiumCheap then " CHEAP" else " MIXED") +
         " [" + (if ivFromInstrument then GetSymbol() else marketSymbol) + "]",
    if !ivDataOK then Color.GRAY else if premiumRich then Color.ORANGE else if premiumCheap then Color.CYAN else Color.LIGHT_GRAY);

AddLabel(showLabels and buyerActive,
    "BUYER " + (if buyerPatternCode == 1 then "U" else "CHAIR") + " | " +
    AsText(Round(buyerDistal, 2)) + "-" + AsText(Round(buyerProximal, 2)) +
    " | Score " + buyerScore + "/8 | Visits " + buyerVisits +
    " | RR " + (if IsNaN(longRR) then "N/A" else AsText(Round(longRR, 2))) +
    " | " + (if buyerPrimary then "PRIMARY (direct)" else if buyerSecondary then "SECONDARY (confirm)" else "SKIP"),
    if buyerPrimary then Color.GREEN else if buyerSecondary then Color.ORANGE else Color.GRAY);

AddLabel(showLabels and sellerActive,
    "SELLER " + (if sellerPatternCode == 1 then "INV U" else "INV CHAIR") + " | " +
    AsText(Round(sellerProximal, 2)) + "-" + AsText(Round(sellerDistal, 2)) +
    " | Score " + sellerScore + "/8 | Visits " + sellerVisits +
    " | RR " + (if IsNaN(shortRR) then "N/A" else AsText(Round(shortRR, 2))) +
    " | " + (if sellerPrimary then "PRIMARY (direct)" else if sellerSecondary then "SECONDARY (confirm)" else "SKIP"),
    if sellerPrimary then Color.RED else if sellerSecondary then Color.ORANGE else Color.GRAY);

# Odds-component breakdown (which of the four factors are carrying the score)
AddLabel(showLabels and buyerActive,
    "BUY ODDS: base " + buyerBaseScore + " + dep " + buyerDepScore +
    " + fresh " + buyerFreshScore + " + rr " + buyerRRScore + " = " + buyerScore + "/8",
    if buyerScore >= 7 then Color.GREEN else if buyerScore >= 5 then Color.ORANGE else Color.GRAY);
AddLabel(showLabels and sellerActive,
    "SELL ODDS: base " + sellerBaseScore + " + dep " + sellerDepScore +
    " + fresh " + sellerFreshScore + " + rr " + sellerRRScore + " = " + sellerScore + "/8",
    if sellerScore >= 7 then Color.GREEN else if sellerScore >= 5 then Color.ORANGE else Color.GRAY);

# Canonical order-type intent (IWT BUY worksheet): entry & profit-exit are
# LIMIT (price control); the protective stop is STOP-MARKET (fill certainty
# when the level breaks). Shares round DOWN. RR ">3 TAKE THE TRADE".
AddLabel(showLabels and (buyerActive or sellerActive),
    "ORDERS: entry LIMIT / exit LIMIT / stop STOP-MKT  |  round shares DOWN",
    Color.DARK_GRAY);

AddLabel(showLabels,
    if !historyReady then "SETUP: WAIT - DATA WARM-UP"
    else if eventBlackout and !allowLongDefinedRiskDuringEvents then "SETUP: WAIT - EVENT BLACKOUT"
    else if volStress then "SETUP: PRESERVE CAPITAL - VOLATILITY STRESS"
    else if earningsNear then "SETUP: EARNINGS WINDOW - LONG DEFINED-RISK ONLY"
    else if bullishRegime and lagging then "SETUP: WAIT - BULL MARKET, LAGGING STOCK"
    else if bearishRegime and leading then "SETUP: WAIT - BEAR MARKET, LEADING STOCK"
    else if bullishSetup and premiumCheap then "SETUP: BULLISH DEBIT CANDIDATE"
    else if bullishSetup and premiumRich and shortPremiumAllowed then "SETUP: BULLISH DEFINED-RISK CREDIT CANDIDATE"
    else if bearishSetup and premiumCheap then "SETUP: BEARISH DEBIT / HEDGE CANDIDATE"
    else if bearishSetup and premiumRich and shortPremiumAllowed then "SETUP: BEARISH DEFINED-RISK CREDIT CANDIDATE"
    else if bullishRegime and buyerActive and !IsNaN(longRR) and longRR < secondaryMinRR then "SETUP: REJECT - BUYER LEVEL BUT RR BELOW 2:1"
    else if bearishRegime and sellerActive and !IsNaN(shortRR) and shortRR < secondaryMinRR then "SETUP: REJECT - SELLER LEVEL BUT RR BELOW 2:1"
    else "SETUP: WAIT - NO STRONG ASYMMETRY",
    Color.CYAN);

# ------------------------------------------------------------ BUBBLES
AddChartBubble(newBuyerZone, candidateBuyerDistal,
    "NEW BUYER " + (if buyerU then "U" else "CHAIR") + " b" + baseCount + " d" + AsText(Round(departureATR, 1)),
    Color.GREEN, no);
AddChartBubble(newSellerZone, candidateSellerDistal,
    "NEW SELLER " + (if sellerInvertedU then "INV U" else "INV CHAIR") + " b" + baseCount + " d" + AsText(Round(departureATR, 1)),
    Color.RED, yes);

# ------------------------------------------------------------ ALERTS
def enteredBull = bullishRegime and !bullishRegime[1];
def enteredBear = bearishRegime and !bearishRegime[1];
def enteredStress = volStress and !volStress[1];
def enteredER = earningsNear and !earningsNear[1];
def bullTrigger = bullishSetup and !bullishSetup[1];
def bearTrigger = bearishSetup and !bearishSetup[1];

Alert(enableAlerts and enteredBull, "TeriQuantumZones: bullish regime entered", Alert.BAR, Sound.Ding);
Alert(enableAlerts and enteredBear, "TeriQuantumZones: bearish regime entered", Alert.BAR, Sound.Ding);
Alert(enableAlerts and enteredStress, "TeriQuantumZones: volatility stress entered", Alert.BAR, Sound.Bell);
Alert(enableAlerts and enteredER, "TeriQuantumZones: earnings window entered", Alert.BAR, Sound.Bell);
Alert(enableAlerts and bullTrigger, "TeriQuantumZones: qualified bullish IWT setup", Alert.BAR, Sound.Ring);
Alert(enableAlerts and bearTrigger, "TeriQuantumZones: qualified bearish IWT setup", Alert.BAR, Sound.Ring);
