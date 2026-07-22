# ═══════════════════════════════════════════════════════════════════
# TQO_RegimeColumn_v4 — chart-adaptive per-name regime (watchlist column)
# Install: Watchlist gear -> Customize -> Scripts -> new -> paste
#
# ADAPTIVE: computes on the ROW's own symbol at the column's aggregation.
# Set the column's aggregation (gear -> aggregation period) to Daily for
# a stable regime, or Weekly to scan weekly-swing regime across the list.
# Matches TeriQuantumOsc_v4 logic, compressed to a single colored cell.
#
# The cell shows the REGIME SCORE (-9..+9) for THIS name. Color encodes
# permission-aware regime: green shades = bullish, red = bearish, plus a
# distinct amber when volatility stress is present (act defensively).
#
# DOCTRINE: proxy. App engine (breadth/oil/credit) is canonical. This is
# a scan aid to find WHERE to look, never a trade trigger.
# ═══════════════════════════════════════════════════════════════════
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
input vixSymbol = "VIX";
input vix9dSymbol = "VIX9D";
input vixCaution = 18;
input vixDanger = 25;
input stressBackwardation = 1.02;
input calmContango = 0.92;

# Row symbol data (native aggregation = adaptive)
def c = close;
def h = high;
def l = low;

# Trend (symmetric)
def smaFast   = Average(c, fastTrendLength);
def smaMedium = Average(c, mediumTrendLength);
def smaSlow   = Average(c, slowTrendLength);
def strongBull = c > smaFast and smaFast > smaMedium and smaMedium > smaSlow;
def strongBear = c < smaFast and smaFast < smaMedium and smaMedium < smaSlow;
def mildBull   = c > smaMedium and smaMedium > smaSlow;
def mildBear   = c < smaMedium and smaMedium < smaSlow;
def trendScore =
    if strongBull then 4 else if strongBear then -4
    else if mildBull then 2 else if mildBear then -2 else 0;

# Momentum (RSI + MACD, bounded +/-2)
def netChange = c - c[1];
def gain = WildersAverage(Max(netChange, 0), rsiLength);
def loss = WildersAverage(Max(-netChange, 0), rsiLength);
def rsi = if loss == 0 then 100 else 100 - 100 / (1 + gain / loss);
def macdValue = ExpAverage(c, macdFast) - ExpAverage(c, macdSlow);
def macdSignal = ExpAverage(macdValue, macdSignalLength);
def rsiVote  = if rsi >= 55 then 1 else if rsi <= 45 then -1 else 0;
def macdVote = if macdValue > macdSignal then 1 else if macdValue < macdSignal then -1 else 0;
def momentumScore = Max(-2, Min(2, rsiVote + macdVote));

# Volatility (shared market context; small +1 for calm only)
def vix = close(vixSymbol);
def vix9d = close(vix9dSymbol);
def vixDataOK = !IsNaN(vix);
def vix9dDataOK = !IsNaN(vix9d);
def stressRatio = if vixDataOK and vix9dDataOK and vix > 0 then vix9d / vix else Double.NaN;
def volStress = vixDataOK and (vix >= vixDanger or (!IsNaN(stressRatio) and stressRatio > stressBackwardation));
def volCalm = vixDataOK and vix < vixCaution and (IsNaN(stressRatio) or stressRatio < calmContango);
def volatilityScore = if volCalm then 1 else 0;

# Location (prior-bar levels + headroom, on the row symbol)
def sellerLevel = Highest(h[1], levelLookback);
def buyerLevel  = Lowest(l[1], levelLookback);
def atr = Average(TrueRange(h, c, l), atrLength);
def roomUpATR   = if atr > 0 then (sellerLevel - c) / atr else Double.NaN;
def roomDownATR = if atr > 0 then (c - buyerLevel) / atr else Double.NaN;
def bullHeadroomOK = !IsNaN(roomUpATR) and roomUpATR >= requiredHeadroomATR;
def bearHeadroomOK = !IsNaN(roomDownATR) and roomDownATR >= requiredHeadroomATR;
def locationScore =
    if bullHeadroomOK and !bearHeadroomOK then 1
    else if bearHeadroomOK and !bullHeadroomOK then -1 else 0;

def regimeScore = Max(-9, Min(9, trendScore + momentumScore + volatilityScore + locationScore));

plot Score = regimeScore;

# Color: bullish greens / bearish reds / amber overlay when vol-stressed
Score.AssignValueColor(
    if volStress then Color.ORANGE
    else if regimeScore >= 6 then Color.GREEN
    else if regimeScore >= 4 then Color.LIGHT_GREEN
    else if regimeScore <= -6 then Color.DARK_RED
    else if regimeScore <= -4 then Color.RED
    else Color.GRAY);
AssignBackgroundColor(
    if volStress then CreateColor(90, 55, 0)
    else if regimeScore >= 6 then CreateColor(0, 90, 0)
    else if regimeScore >= 4 then CreateColor(0, 60, 0)
    else if regimeScore <= -6 then CreateColor(120, 0, 0)
    else if regimeScore <= -4 then CreateColor(80, 0, 0)
    else CreateColor(40, 40, 40));
