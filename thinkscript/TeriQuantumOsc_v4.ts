# ═══════════════════════════════════════════════════════════════════
# TeriQuantumOsc v4.3 - Chart-Adaptive, fail-soft
# REGIME / LOCATION / PERMISSION / STRUCTURE BIAS
#
# v4.3 changes vs v4.2:
#  1. Warm-up detection now tests the COMPUTED SERIES (IsNaN on the
#     moving averages themselves) instead of dynamic bar offsets like
#     mClose[momentumLookback - 1]. ThinkScript wants constant offsets;
#     testing the output is both safer and more direct.
#  2. Regime score is now SYMMETRIC: trend + momentum only, range
#     [-6,+6] against +/-4 thresholds. v4.2 added calmBonus into the
#     directional score, making the scale +7/-6 - easier to read very
#     bullish than very bearish. Volatility is not directional evidence.
#  3. Calm/stress now live ONLY where they belong: the VIX label and
#     the PERMISSION engine. (Symmetric measurement, asymmetric policy.)
#  4. VIX label says "spot" - VX futures (VX[U26] etc.) trade at a
#     premium to spot in contango and are NOT what this reads.
#
# Kept from v4.2 (all good): breakout-aware location, IV fail-soft
# fallback with honest "IV Position" naming, MODE label, IV SOURCE
# label, warm-up warning, edge-triggered alerts.
#
# Proxy only. Quantum Maestro app engine remains canonical.
# No 0DTE, defined risk only, no short premium into events. No strikes.
# ═══════════════════════════════════════════════════════════════════

declare lower;

# ── Inputs ──────────────────────────────────────────────────────────
input marketSymbol = "SPX";
input vixSymbol    = "VIX";
input vix9dSymbol  = "VIX9D";
input useMarketData = yes;

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

input vixCaution = 18;
input vixDanger  = 25;
input stressBackwardation = 1.02;
input calmContango = 0.95;

input ivRankLookback = 252;
input premiumRichThreshold = 50;
input premiumCheapThreshold = 25;

input eventBlackout = no;
input allowLongDefinedRiskDuringEvents = no;
input requireBreakoutConfirmation = yes;
input enableAlerts = yes;

# ── Price source with fail-soft fallback ────────────────────────────
def rawMClose = close(marketSymbol);
def rawMHigh  = high(marketSymbol);
def rawMLow   = low(marketSymbol);

def marketPullOK = !IsNaN(rawMClose) and !IsNaN(rawMHigh) and !IsNaN(rawMLow);
def useMkt = useMarketData and marketPullOK;

def mClose = if useMkt then rawMClose else close;
def mHigh  = if useMkt then rawMHigh  else high;
def mLow   = if useMkt then rawMLow   else low;

def priceDataOK = !IsNaN(mClose) and !IsNaN(mHigh) and !IsNaN(mLow);

# ── Trend ───────────────────────────────────────────────────────────
def smaFast   = Average(mClose, fastTrendLength);
def smaMedium = Average(mClose, mediumTrendLength);
def smaSlow   = Average(mClose, slowTrendLength);

def strongBull = mClose > smaFast and smaFast > smaMedium and smaMedium > smaSlow;
def strongBear = mClose < smaFast and smaFast < smaMedium and smaMedium < smaSlow;
def mildBull   = mClose > smaMedium and smaMedium > smaSlow;
def mildBear   = mClose < smaMedium and smaMedium < smaSlow;

def trendScore =
    if strongBull then 4
    else if strongBear then -4
    else if mildBull then 2
    else if mildBear then -2
    else 0;

# ── Momentum ────────────────────────────────────────────────────────
def netChange = mClose - mClose[1];
def gain = WildersAverage(Max(netChange, 0), rsiLength);
def loss = WildersAverage(Max(-netChange, 0), rsiLength);

def rsi =
    if loss == 0 and gain == 0 then 50
    else if loss == 0 then 100
    else 100 - 100 / (1 + gain / loss);

def macdValue  = ExpAverage(mClose, macdFast) - ExpAverage(mClose, macdSlow);
def macdSignal = ExpAverage(macdValue, macdSignalLength);

def rsiVote  = if rsi >= 55 then 1 else if rsi <= 45 then -1 else 0;
def macdVote = if macdValue > macdSignal then 1 else if macdValue < macdSignal then -1 else 0;
def momentumScore = Max(-2, Min(2, rsiVote + macdVote));

# ── Levels (prior bars only) + ATR ──────────────────────────────────
def sellerLevel = Highest(mHigh[1], levelLookback);
def buyerLevel  = Lowest(mLow[1], levelLookback);
def atr = Average(TrueRange(mHigh, mClose, mLow), atrLength);

# ── Warm-up: test the COMPUTED SERIES, not dynamic bar offsets ──────
def trendHistoryOK    = !IsNaN(smaSlow) and !IsNaN(smaMedium) and !IsNaN(smaFast);
def momentumHistoryOK = !IsNaN(macdSignal) and !IsNaN(rsi);
def levelHistoryOK    = !IsNaN(sellerLevel) and !IsNaN(buyerLevel) and !IsNaN(atr);
def regimeDataReady   = priceDataOK and trendHistoryOK and momentumHistoryOK and levelHistoryOK;

# ── Volatility: SPOT VIX term structure (NOT VX futures) ────────────
def vix   = close(vixSymbol);
def vix9d = close(vix9dSymbol);
def vixDataOK   = !IsNaN(vix);
def vix9dDataOK = !IsNaN(vix9d);

def stressRatio =
    if vixDataOK and vix9dDataOK and vix > 0 then vix9d / vix else Double.NaN;

def volStress =
    vixDataOK and (vix >= vixDanger or (!IsNaN(stressRatio) and stressRatio > stressBackwardation));

def volCalm =
    vixDataOK and vix < vixCaution and (IsNaN(stressRatio) or stressRatio < calmContango);

# ── Regime score: SYMMETRIC, directional evidence only [-6,+6] ──────
def rawRegime   = trendScore + momentumScore;
def regimeScore = Max(-6, Min(6, rawRegime));
def bullishRegime = regimeScore >= 4;
def bearishRegime = regimeScore <= -4;

# ── Location (breakout-aware) ───────────────────────────────────────
def breakoutUp   = mClose > sellerLevel;
def breakoutDown = mClose < buyerLevel;

def roomUpATR   = if atr > 0 and !breakoutUp   then (sellerLevel - mClose) / atr else Double.NaN;
def roomDownATR = if atr > 0 and !breakoutDown then (mClose - buyerLevel) / atr else Double.NaN;

def bullHeadroomOK = !IsNaN(roomUpATR)   and roomUpATR   >= requiredHeadroomATR;
def bearHeadroomOK = !IsNaN(roomDownATR) and roomDownATR >= requiredHeadroomATR;

def bullBreakoutConfirmed = breakoutUp   and mClose[1] > sellerLevel[1];
def bearBreakoutConfirmed = breakoutDown and mClose[1] < buyerLevel[1];

def bullLocationOK =
    bullHeadroomOK or (breakoutUp and (!requireBreakoutConfirmation or bullBreakoutConfirmed));
def bearLocationOK =
    bearHeadroomOK or (breakoutDown and (!requireBreakoutConfirmation or bearBreakoutConfirmed));

def locationScore =
    if bullLocationOK and !bearLocationOK then 1
    else if bearLocationOK and !bullLocationOK then -1
    else 0;

# ── IV with fail-soft fallback (chart-window position, not IV rank) ─
def ivMarket = imp_volatility(marketSymbol);
def ivChart  = imp_volatility();
def iv = if useMkt and !IsNaN(ivMarket) then ivMarket else ivChart;

def ivHigh  = Highest(iv, ivRankLookback);
def ivLow   = Lowest(iv, ivRankLookback);
def ivRange = ivHigh - ivLow;
def ivDataOK = !IsNaN(iv) and !IsNaN(ivHigh) and !IsNaN(ivLow) and ivRange > 0;

def ivPosition   = if ivDataOK then 100 * (iv - ivLow) / ivRange else Double.NaN;
def premiumRich  = ivDataOK and ivPosition >= premiumRichThreshold;
def premiumCheap = ivDataOK and ivPosition <= premiumCheapThreshold;

# ── Permission engine (where volatility actually belongs) ───────────
def shortPremiumAllowed    = regimeDataReady and !eventBlackout and !volStress;
def longDefinedRiskAllowed = regimeDataReady and (!eventBlackout or allowLongDefinedRiskDuringEvents);

def tradePermission =
    if !regimeDataReady then 0
    else if eventBlackout and !allowLongDefinedRiskDuringEvents then 0
    else if volStress then 1
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
def chartAgg = GetAggregationPeriod();

AddLabel(yes,
    if chartAgg < AggregationPeriod.DAY then "MODE: INTRADAY"
    else if chartAgg == AggregationPeriod.DAY then "MODE: DAILY"
    else "MODE: HIGHER TIMEFRAME",
    Color.DARK_GRAY);

AddLabel(yes,
    if bullishRegime then "REGIME: BULLISH"
    else if bearishRegime then "REGIME: BEARISH"
    else "REGIME: NEUTRAL",
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

AddLabel(!useMkt and useMarketData,
    "NOTE: '" + marketSymbol + "' did not resolve - using chart symbol",
    Color.ORANGE);

AddLabel(!regimeDataReady,
    "WARM-UP: insufficient history for configured lengths (need " + slowTrendLength + " bars)",
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

AddLabel(yes,
    if !ivDataOK then "IV Position: unavailable - no premium bias"
    else "IV Position " + Round(ivPosition, 0) +
        (if premiumRich then " - premium rich" else if premiumCheap then " - premium cheap" else " - mixed"),
    if !ivDataOK then Color.GRAY
    else if premiumRich then Color.ORANGE
    else if premiumCheap then Color.CYAN
    else Color.LIGHT_GRAY);

AddLabel(yes,
    if useMkt and !IsNaN(ivMarket) then "IV SOURCE: " + marketSymbol else "IV SOURCE: chart symbol",
    Color.DARK_GRAY);

AddLabel(yes,
    if breakoutUp then "LOCATION: above prior seller level"
    else if breakoutDown then "LOCATION: below prior buyer level"
    else "Headroom: +" + (if IsNaN(roomUpATR) then "N/A" else AsText(Round(roomUpATR, 1))) +
         " / -" + (if IsNaN(roomDownATR) then "N/A" else AsText(Round(roomDownATR, 1))) + " ATR",
    if breakoutUp then Color.LIGHT_GREEN else if breakoutDown then Color.LIGHT_RED else Color.LIGHT_GRAY);

# ── Decision support ────────────────────────────────────────────────
AddLabel(yes,
    if !regimeDataReady then "SETUP: DATA WARM-UP - wait"
    else if eventBlackout and !allowLongDefinedRiskDuringEvents then "SETUP: WAIT - event blackout; new trades blocked"
    else if volStress then "SETUP: PRESERVE CAPITAL - volatility stress"
    else if eventBlackout and allowLongDefinedRiskDuringEvents and bullishRegime and premiumCheap and bullLocationOK
        then "BIAS: event-risk bullish debit candidate - minimum size"
    else if eventBlackout and allowLongDefinedRiskDuringEvents and bearishRegime and premiumCheap and bearLocationOK
        then "BIAS: event-risk bearish debit / hedge candidate - minimum size"
    else if eventBlackout and allowLongDefinedRiskDuringEvents then "SETUP: EVENT MODE - long defined-risk only"
    else if bullishRegime and !bullLocationOK then "SETUP: WAIT - bullish regime but poor entry location"
    else if bearishRegime and !bearLocationOK then "SETUP: WAIT - bearish regime but poor entry location"
    else if bullishRegime and premiumRich and shortPremiumAllowed then "BIAS: bullish defined-risk premium candidate"
    else if bullishRegime and premiumCheap and longDefinedRiskAllowed then "BIAS: bullish debit candidate"
    else if bearishRegime and premiumCheap and longDefinedRiskAllowed then "BIAS: bearish debit / hedge candidate"
    else if bearishRegime and premiumRich and shortPremiumAllowed then "BIAS: bearish defined-risk premium candidate"
    else "SETUP: WAIT - no strong asymmetry",
    Color.CYAN);

# ── Edge-triggered alerts ───────────────────────────────────────────
def enteredBullish     = bullishRegime and !bullishRegime[1];
def enteredBearish     = bearishRegime and !bearishRegime[1];
def enteredVolStress   = volStress and !volStress[1];
def permissionBlocked  = tradePermission == 0 and tradePermission[1] != 0;

Alert(enableAlerts and enteredBullish, "TeriQuantumOsc: bullish regime entered", Alert.BAR, Sound.Ding);
Alert(enableAlerts and enteredBearish, "TeriQuantumOsc: bearish regime entered", Alert.BAR, Sound.Ding);
Alert(enableAlerts and enteredVolStress, "TeriQuantumOsc: volatility stress entered", Alert.BAR, Sound.Bell);
Alert(enableAlerts and permissionBlocked, "TeriQuantumOsc: trading permission blocked", Alert.BAR, Sound.Bell);
