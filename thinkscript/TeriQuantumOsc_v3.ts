# ═══════════════════════════════════════════════════════════════════
# TeriQuantumOsc v3 — Quantum Maestro Regime / Location / Permission
#
# v3 merges an external engineering review into v2. Adopted:
#  - Regime computed on an explicit aggregation (default DAY) so the
#    study is stable across chart timeframes (v2's biggest flaw).
#  - Buyer/seller levels exclude the current (unfinished) bar via [1].
#  - REGIME, LOCATION and PERMISSION are separate outputs. Blackout and
#    vol-stress restrict PERMISSION; they no longer corrupt the regime
#    score. (Symmetric measurement, asymmetric policy — the same rule
#    the Quantum Maestro handoff doc states.)
#  - Momentum de-duplicated: RSI + MACD (dropped stochastic; it re-measured
#    the same thing a third time). Bounded to +/-2.
#  - Missing IV/VIX data suppresses the recommendation instead of faking
#    a neutral value.
#  - Bare index symbols (SPX / VIX / VIX9D) resolve reliably as secondaries;
#    the $XXX.X form could return NaN on equity charts and blank the study.
#
# Added beyond the review (neither v2 nor the review handled this):
#  - AGGREGATION GUARD: on weekly/monthly charts a DAY secondary
#    aggregation is illegal in thinkScript. v3 detects chart agg >=
#    regime agg and falls back to the CHART timeframe, with a visible
#    warning label, instead of throwing a script error. Your 3Y-weekly
#    chart is exactly this case.
#
# DOCTRINE: proxy only. The app's engine (breadth/oil/credit) is
# canonical; on disagreement, the app wins. No 0DTE, defined-risk only,
# no short premium into events. This study never selects strikes/orders.
# ═══════════════════════════════════════════════════════════════════
declare lower;

# ── Symbols ─────────────────────────────────────────────────────────
input marketSymbol = "SPX";
input vixSymbol    = "VIX";
input vix9dSymbol  = "VIX9D";

# ── Regime timeframe (independent of chart timeframe) ───────────────
input regimeAggregation = AggregationPeriod.DAY;
input useMarketMomentum = yes;   # momentum on regime symbol vs chart symbol

# ── Trend / momentum ────────────────────────────────────────────────
input fastTrendLength   = 20;
input mediumTrendLength = 50;
input slowTrendLength   = 200;
input rsiLength = 14;
input macdFast = 12;
input macdSlow = 26;
input macdSignalLength = 9;

# ── Levels / volatility / IV ────────────────────────────────────────
input atrLength = 14;
input levelLookback = 20;
input requiredHeadroomATR = 0.5;   # NOTE: review suggested 1.0. 0.5 keeps
                                   # equity-name setups tradable; raise to
                                   # 1.0 for stricter index-only premium.
input vixCaution = 18;
input vixDanger  = 25;
input stressBackwardation = 1.02;
input calmContango = 0.92;
input ivRankLookback = 252;
input premiumRichThreshold = 50;
input premiumCheapThreshold = 25;

# ── Permission inputs ───────────────────────────────────────────────
input eventBlackout = no;                       # FOMC/CPI/NFP/earnings
input allowLongDefinedRiskDuringEvents = no;    # hedges/long premium exception

# ═══════════════════════════════════════════════════════════════════
# AGGREGATION GUARD — the honest version
# thinkScript rule: a secondary period cannot be LOWER than the chart's
# primary, AND this is enforced at COMPILE time — a conditional cannot
# rescue an illegal close(period=DAY) call on a weekly chart; ToS rejects
# the whole study. So we cannot "fall back" inside the script.
#
# Therefore: regime data is pulled at `regimeAggregation` unconditionally
# (works on intraday + daily charts — the intended use). We DETECT a
# chart >= regime agg and show a loud warning so you know the study is
# only valid at/below its regime timeframe. On weekly/monthly charts,
# set regimeAggregation to WEEK/MONTH, or use the simpler v2 there.
# ═══════════════════════════════════════════════════════════════════
def chartAgg = GetAggregationPeriod();
def regimeAggValid = chartAgg <= regimeAggregation;   # drives the warning label

# ── Regime-timeframe data (pulled at regimeAggregation) ─────────────
def mClose = close(symbol = marketSymbol, period = regimeAggregation, priceType = PriceType.LAST);
def mHigh  = high(symbol = marketSymbol, period = regimeAggregation, priceType = PriceType.LAST);
def mLow   = low(symbol = marketSymbol, period = regimeAggregation, priceType = PriceType.LAST);
def vix    = close(symbol = vixSymbol, period = regimeAggregation, priceType = PriceType.LAST);
def vix9d  = close(symbol = vix9dSymbol, period = regimeAggregation, priceType = PriceType.LAST);

def marketDataOK = !IsNaN(mClose);
def vixDataOK    = !IsNaN(vix);
def vix9dDataOK  = !IsNaN(vix9d);

def src     = if useMarketMomentum then mClose else close;
def srcHigh = if useMarketMomentum then mHigh  else high;
def srcLow  = if useMarketMomentum then mLow   else low;

# ── Trend (symmetric, ±4/±2/0) ──────────────────────────────────────
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

# ── Momentum (de-duplicated: RSI + MACD only, bounded ±2) ───────────
def netChange = src - src[1];
def gain = WildersAverage(Max(netChange, 0), rsiLength);
def loss = WildersAverage(Max(-netChange, 0), rsiLength);
def rsi = if loss == 0 then 100 else 100 - 100 / (1 + gain / loss);
def macdValue = ExpAverage(src, macdFast) - ExpAverage(src, macdSlow);
def macdSignal = ExpAverage(macdValue, macdSignalLength);
def rsiVote  = if rsi >= 55 then 1 else if rsi <= 45 then -1 else 0;
def macdVote = if macdValue > macdSignal then 1 else if macdValue < macdSignal then -1 else 0;
def momentumScore = Max(-2, Min(2, rsiVote + macdVote));

# ── Volatility: small regime contribution; stress routes to permission ─
def stressRatio = if vixDataOK and vix9dDataOK and vix > 0 then vix9d / vix else Double.NaN;
def volStress = vixDataOK and (vix >= vixDanger or (!IsNaN(stressRatio) and stressRatio > stressBackwardation));
def volCalm   = vixDataOK and vix < vixCaution and (IsNaN(stressRatio) or stressRatio < calmContango);
# Only calm adds a small +1; stress is handled in PERMISSION, not as a
# large negative direction (review point #7 — done properly this time).
def volatilityScore = if volCalm then 1 else 0;

# ── Teri levels (exclude current bar) + headroom ────────────────────
def sellerLevel = Highest(srcHigh[1], levelLookback);
def buyerLevel  = Lowest(srcLow[1], levelLookback);
def atr = Average(TrueRange(srcHigh, src, srcLow), atrLength);
def roomUpATR   = if atr > 0 then (sellerLevel - src) / atr else Double.NaN;
def roomDownATR = if atr > 0 then (src - buyerLevel) / atr else Double.NaN;
def bullHeadroomOK = !IsNaN(roomUpATR) and roomUpATR >= requiredHeadroomATR;
def bearHeadroomOK = !IsNaN(roomDownATR) and roomDownATR >= requiredHeadroomATR;
def locationScore =
    if bullHeadroomOK and !bearHeadroomOK then 1
    else if bearHeadroomOK and !bullHeadroomOK then -1 else 0;

# ── IV rank (missing data suppresses bias, never fakes 50) ──────────
def iv = imp_volatility(marketSymbol, AggregationPeriod.DAY, PriceType.LAST);
def ivHigh = Highest(iv, ivRankLookback);
def ivLow  = Lowest(iv, ivRankLookback);
def ivRange = ivHigh - ivLow;
def ivDataOK = !IsNaN(iv) and !IsNaN(ivHigh) and !IsNaN(ivLow) and ivRange > 0;
def ivRank = if ivDataOK then 100 * (iv - ivLow) / ivRange else Double.NaN;
def premiumRich  = ivDataOK and ivRank >= premiumRichThreshold;
def premiumCheap = ivDataOK and ivRank <= premiumCheapThreshold;

# ── Regime score (measurement only) ─────────────────────────────────
def rawRegime = trendScore + momentumScore + volatilityScore + locationScore;
def regimeScore = Max(-9, Min(9, rawRegime));
def bullishRegime = regimeScore >= 4;
def bearishRegime = regimeScore <= -4;

# ── Permission (policy layer — separate from regime) ────────────────
def shortPremiumAllowed = !eventBlackout and !volStress and marketDataOK;
def longDefinedRiskAllowed = marketDataOK and (!eventBlackout or allowLongDefinedRiskDuringEvents);
def tradePermission =
    if !marketDataOK then 0
    else if eventBlackout and !allowLongDefinedRiskDuringEvents then 0
    else if volStress then 1
    else 2;   # 0 blocked · 1 defensive only · 2 selective

# ── Plots ───────────────────────────────────────────────────────────
plot Regime = regimeScore;
Regime.SetPaintingStrategy(PaintingStrategy.HISTOGRAM);
Regime.SetLineWeight(4);
Regime.AssignValueColor(
    if regimeScore >= 6 then Color.GREEN
    else if regimeScore >= 4 then Color.LIGHT_GREEN
    else if regimeScore <= -6 then Color.DARK_RED
    else if regimeScore <= -4 then Color.RED
    else Color.GRAY);
plot ZeroLine = 0;      ZeroLine.SetDefaultColor(Color.GRAY);       ZeroLine.HideBubble();
plot BullThreshold = 4; BullThreshold.SetDefaultColor(Color.GREEN); BullThreshold.SetStyle(Curve.SHORT_DASH); BullThreshold.HideBubble();
plot BearThreshold = -4; BearThreshold.SetDefaultColor(Color.RED);  BearThreshold.SetStyle(Curve.SHORT_DASH); BearThreshold.HideBubble();

# ── Aggregation-guard warning (visible when it matters) ─────────────
AddLabel(!regimeAggValid,
    "WARNING: chart timeframe >= regime aggregation. Regime data may be unreliable here. "
    + "Use a daily/intraday chart, or set regimeAggregation to match this chart.",
    Color.ORANGE);

# ── Labels: REGIME / PERMISSION / components ────────────────────────
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
AddLabel(yes,
    if !vixDataOK then "VIX: unavailable"
    else "VIX " + Round(vix, 2) + "  9D/30D " + (if IsNaN(stressRatio) then "N/A" else AsText(Round(stressRatio, 2))),
    if volStress then Color.RED else if volCalm then Color.GREEN else Color.YELLOW);
AddLabel(yes,
    if !ivDataOK then "IV Rank: unavailable - no premium bias"
    else "IV Rank " + Round(ivRank, 0) + (if premiumRich then " - premium rich" else if premiumCheap then " - premium cheap" else " - mixed"),
    if !ivDataOK then Color.GRAY else if premiumRich then Color.ORANGE else if premiumCheap then Color.CYAN else Color.LIGHT_GRAY);
AddLabel(yes,
    "Headroom: +" + (if IsNaN(roomUpATR) then "N/A" else AsText(Round(roomUpATR, 1))) +
    " / -" + (if IsNaN(roomDownATR) then "N/A" else AsText(Round(roomDownATR, 1))) + " ATR",
    Color.LIGHT_GRAY);

# ── Decision-support: BIAS family only, never strikes ───────────────
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
