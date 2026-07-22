# ═══════════════════════════════════════════════════════════════════
# TeriQuantumOsc v4.1 — Chart-Adaptive, fail-soft
#
# FIX vs v4 (the all-N/A bug on equity charts):
#  - Default symbols are BARE strings ("SPX","VIX","VIX9D"), which resolve
#    reliably as secondary symbols. The "$XXX.X" form failed to align on
#    a NASDAQ equity chart (AAPL), returning NaN and - because every label
#    was gated on *DataOK - cascading the WHOLE study to N/A.
#  - GRACEFUL DEGRADATION: trend + momentum come from PRICE and always
#    compute. If VIX/VIX9D/IV feeds are missing, only those components
#    drop to 0 and their labels say "unavailable" - the regime score still
#    works off price. A missing vol feed no longer blanks everything.
#  - marketDataOK falls back to the CHART symbol if the market symbol
#    can't be pulled, so the study still runs.
#
# Three outputs (symmetric measurement, asymmetric policy):
#   REGIME     - what the market is doing (price-driven, always available)
#   LOCATION   - price near a buyer/seller level with room (Teri)
#   PERMISSION - may you act (blackout / vol-stress restrict THIS only)
#
# DOCTRINE: proxy only. App engine (breadth/oil/credit) is canonical.
# No 0DTE, defined risk only, no short premium into events. No strikes.
# ═══════════════════════════════════════════════════════════════════
declare lower;

# -- Symbols (BARE strings resolve most reliably as secondaries) -----
input marketSymbol = "SPX";
input vixSymbol    = "VIX";
input vix9dSymbol  = "VIX9D";
input useMarketData = yes;   # yes = regime on marketSymbol; no = chart symbol

# -- Trend / momentum ------------------------------------------------
input fastTrendLength   = 20;
input mediumTrendLength = 50;
input slowTrendLength   = 200;
input rsiLength = 14;
input macdFast = 12;
input macdSlow = 26;
input macdSignalLength = 9;

# -- Levels / volatility / IV ----------------------------------------
input atrLength = 14;
input levelLookback = 20;
input requiredHeadroomATR = 0.5;   # raise to 1.0 for stricter index-only premium
input vixCaution = 18;
input vixDanger  = 25;
input stressBackwardation = 1.02;
input calmContango = 0.92;
input ivRankLookback = 252;
input premiumRichThreshold = 50;
input premiumCheapThreshold = 25;

# -- Permission inputs -----------------------------------------------
input eventBlackout = no;                     # FOMC/CPI/NFP/earnings
input allowLongDefinedRiskDuringEvents = no;  # hedges/long premium exception

# -- Market data with fallback to the chart symbol -------------------
def rawMClose = close(marketSymbol);
def rawMHigh  = high(marketSymbol);
def rawMLow   = low(marketSymbol);
def marketPullOK = !IsNaN(rawMClose);

# If the market-symbol pull fails, degrade to the chart symbol so trend/
# momentum/levels still compute rather than blanking the whole study.
def useMkt = useMarketData and marketPullOK;
def mClose = if useMkt then rawMClose else close;
def mHigh  = if useMkt then rawMHigh  else high;
def mLow   = if useMkt then rawMLow   else low;
def marketDataOK = !IsNaN(mClose);   # true as long as SOMETHING resolved

def vix   = close(vixSymbol);
def vix9d = close(vix9dSymbol);
def vixDataOK   = !IsNaN(vix);
def vix9dDataOK = !IsNaN(vix9d);

# -- Trend (symmetric; PRICE-DRIVEN, always available) ---------------
def smaFast   = Average(mClose, fastTrendLength);
def smaMedium = Average(mClose, mediumTrendLength);
def smaSlow   = Average(mClose, slowTrendLength);
def strongBull = mClose > smaFast and smaFast > smaMedium and smaMedium > smaSlow;
def strongBear = mClose < smaFast and smaFast < smaMedium and smaMedium < smaSlow;
def mildBull   = mClose > smaMedium and smaMedium > smaSlow;
def mildBear   = mClose < smaMedium and smaMedium < smaSlow;
def trendScore =
    if strongBull then 4 else if strongBear then -4
    else if mildBull then 2 else if mildBear then -2 else 0;

# -- Momentum (RSI + MACD, PRICE-DRIVEN, bounded +/-2) ---------------
def netChange = mClose - mClose[1];
def gain = WildersAverage(Max(netChange, 0), rsiLength);
def loss = WildersAverage(Max(-netChange, 0), rsiLength);
def rsi = if loss == 0 then 100 else 100 - 100 / (1 + gain / loss);
def macdValue = ExpAverage(mClose, macdFast) - ExpAverage(mClose, macdSlow);
def macdSignal = ExpAverage(macdValue, macdSignalLength);
def rsiVote  = if rsi >= 55 then 1 else if rsi <= 45 then -1 else 0;
def macdVote = if macdValue > macdSignal then 1 else if macdValue < macdSignal then -1 else 0;
def momentumScore = Max(-2, Min(2, rsiVote + macdVote));

# -- Volatility (degrades to 0 contribution if feed missing) ---------
def stressRatio = if vixDataOK and vix9dDataOK and vix > 0 then vix9d / vix else Double.NaN;
def volStress = vixDataOK and (vix >= vixDanger or (!IsNaN(stressRatio) and stressRatio > stressBackwardation));
def volCalm   = vixDataOK and vix < vixCaution and (IsNaN(stressRatio) or stressRatio < calmContango);
def volatilityScore = if volCalm then 1 else 0;

# -- Teri levels (exclude current bar) + headroom (PRICE-DRIVEN) -----
def sellerLevel = Highest(mHigh[1], levelLookback);
def buyerLevel  = Lowest(mLow[1], levelLookback);
def atr = Average(TrueRange(mHigh, mClose, mLow), atrLength);
def roomUpATR   = if atr > 0 then (sellerLevel - mClose) / atr else Double.NaN;
def roomDownATR = if atr > 0 then (mClose - buyerLevel) / atr else Double.NaN;
def bullHeadroomOK = !IsNaN(roomUpATR) and roomUpATR >= requiredHeadroomATR;
def bearHeadroomOK = !IsNaN(roomDownATR) and roomDownATR >= requiredHeadroomATR;
def locationScore =
    if bullHeadroomOK and !bearHeadroomOK then 1
    else if bearHeadroomOK and !bullHeadroomOK then -1 else 0;

# -- IV rank (degrades cleanly; never fakes a value) -----------------
def iv = imp_volatility(marketSymbol);
def ivHigh = Highest(iv, ivRankLookback);
def ivLow  = Lowest(iv, ivRankLookback);
def ivRange = ivHigh - ivLow;
def ivDataOK = !IsNaN(iv) and !IsNaN(ivHigh) and !IsNaN(ivLow) and ivRange > 0;
def ivRank = if ivDataOK then 100 * (iv - ivLow) / ivRange else Double.NaN;
def premiumRich  = ivDataOK and ivRank >= premiumRichThreshold;
def premiumCheap = ivDataOK and ivRank <= premiumCheapThreshold;

# -- Regime score (price-driven core; vol only adds +1 when calm) ----
def rawRegime = trendScore + momentumScore + volatilityScore + locationScore;
def regimeScore = Max(-9, Min(9, rawRegime));
def bullishRegime = regimeScore >= 4;
def bearishRegime = regimeScore <= -4;

# -- Permission (policy layer, separate from regime) -----------------
def shortPremiumAllowed = !eventBlackout and !volStress and marketDataOK;
def longDefinedRiskAllowed = marketDataOK and (!eventBlackout or allowLongDefinedRiskDuringEvents);
def tradePermission =
    if !marketDataOK then 0
    else if eventBlackout and !allowLongDefinedRiskDuringEvents then 0
    else if volStress then 1
    else 2;   # 0 blocked | 1 defensive only | 2 selective

# -- Plots -----------------------------------------------------------
plot Regime = regimeScore;
Regime.SetPaintingStrategy(PaintingStrategy.HISTOGRAM);
Regime.SetLineWeight(4);
Regime.AssignValueColor(
    if regimeScore >= 6 then Color.GREEN
    else if regimeScore >= 4 then Color.LIGHT_GREEN
    else if regimeScore <= -6 then Color.DARK_RED
    else if regimeScore <= -4 then Color.RED
    else Color.GRAY);
plot ZeroLine = 0;       ZeroLine.SetDefaultColor(Color.GRAY);        ZeroLine.HideBubble();
plot BullThreshold = 4;  BullThreshold.SetDefaultColor(Color.GREEN);  BullThreshold.SetStyle(Curve.SHORT_DASH); BullThreshold.HideBubble();
plot BearThreshold = -4; BearThreshold.SetDefaultColor(Color.RED);    BearThreshold.SetStyle(Curve.SHORT_DASH); BearThreshold.HideBubble();

# -- Labels ----------------------------------------------------------
AddLabel(yes,
    if bullishRegime then "REGIME: BULLISH" else if bearishRegime then "REGIME: BEARISH" else "REGIME: NEUTRAL",
    if bullishRegime then Color.GREEN else if bearishRegime then Color.RED else Color.GRAY);
AddLabel(yes,
    if tradePermission == 0 then "PERMISSION: BLOCKED"
    else if tradePermission == 1 then "PERMISSION: DEFENSIVE ONLY"
    else "PERMISSION: SELECTIVE",
    if tradePermission == 0 then Color.DARK_ORANGE else if tradePermission == 1 then Color.YELLOW else Color.GREEN);
AddLabel(yes,
    "Score " + regimeScore + "  (T " + trendScore + " M " + momentumScore + " V " + volatilityScore + " L " + locationScore + ")",
    Color.WHITE);
AddLabel(!useMkt and useMarketData,
    "NOTE: '" + marketSymbol + "' did not resolve - regime computed on CHART symbol instead. Check the symbol input.",
    Color.ORANGE);
AddLabel(yes,
    if !vixDataOK then "VIX: unavailable (vol component = 0)"
    else "VIX " + Round(vix, 2) + "  9D/30D " + (if IsNaN(stressRatio) then "N/A" else AsText(Round(stressRatio, 2))),
    if !vixDataOK then Color.GRAY else if volStress then Color.RED else if volCalm then Color.GREEN else Color.YELLOW);
AddLabel(yes,
    if !ivDataOK then "IV Rank: unavailable - no premium bias"
    else "IV Rank " + Round(ivRank, 0) + (if premiumRich then " - premium rich" else if premiumCheap then " - premium cheap" else " - mixed"),
    if !ivDataOK then Color.GRAY else if premiumRich then Color.ORANGE else if premiumCheap then Color.CYAN else Color.LIGHT_GRAY);
AddLabel(yes,
    "Headroom: +" + (if IsNaN(roomUpATR) then "N/A" else AsText(Round(roomUpATR, 1))) +
    " / -" + (if IsNaN(roomDownATR) then "N/A" else AsText(Round(roomDownATR, 1))) + " ATR",
    Color.LIGHT_GRAY);

# -- Decision support: BIAS family only, never strikes ---------------
AddLabel(yes,
    if !marketDataOK then "SETUP: DATA ERROR - verify symbols"
    else if eventBlackout then "SETUP: WAIT - event blackout (regime unchanged; you may not act)"
    else if volStress then "SETUP: PRESERVE CAPITAL - volatility stress"
    else if bullishRegime and !bullHeadroomOK then "SETUP: WAIT - bullish but poor upside headroom (Teri: no chase)"
    else if bearishRegime and !bearHeadroomOK then "SETUP: WAIT - bearish but poor downside headroom"
    else if bullishRegime and premiumRich and shortPremiumAllowed then "BIAS: bullish defined-risk premium candidate"
    else if bullishRegime and premiumCheap and longDefinedRiskAllowed then "BIAS: bullish debit candidate"
    else if bearishRegime and premiumCheap and longDefinedRiskAllowed then "BIAS: bearish debit / hedge candidate"
    else if bearishRegime and premiumRich and shortPremiumAllowed then "BIAS: bearish defined-risk premium candidate"
    else "SETUP: WAIT - no strong asymmetry",
    Color.CYAN);
