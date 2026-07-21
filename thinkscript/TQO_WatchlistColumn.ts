# ═══════════════════════════════════════════════════════════════════
# TQO_WatchlistColumn — Quantum Maestro regime score as a watchlist column
# Add via: Watchlist gear -> Customize -> Scripts -> new -> paste
# Shows the SAME composite score as TeriQuantumOsc, compressed to a cell.
#
# DOCTRINE: proxy only. The app's engine (breadth/oil/credit) is canonical.
# Column shows DIRECTION + PERMISSION at a glance across the whole list.
# ═══════════════════════════════════════════════════════════════════
# Compact clone of the oscillator score for tabular scanning.
# Momentum computes on the row's own symbol; trend/vol reference the index
# only when the row IS an index — otherwise trend uses the row symbol so the
# column means "this name's regime," not "SPX's regime" pasted on every row.

def src = close;
def h = high;
def l = low;

# Trend (symmetric ladder, on the row symbol)
def sma20 = Average(src, 20);
def sma50 = Average(src, 50);
def sma200 = Average(src, 200);
def trendScore =
    if src > sma20 and sma20 > sma50 and sma50 > sma200 then 3
    else if src < sma20 and sma20 < sma50 and sma50 < sma200 then -3
    else if src > sma50 and src > sma200 then 1
    else if src < sma50 and src < sma200 then -1
    else 0;

# Momentum (RSI + stoch + MACD votes)
def netChg = src - src[1];
def rsiUp = WildersAverage(Max(netChg, 0), 14);
def rsiDn = WildersAverage(Max(-netChg, 0), 14);
def rsiV = if rsiDn == 0 then 100 else 100 - 100 / (1 + rsiUp / rsiDn);
def lowK = Lowest(l, 14);
def highK = Highest(h, 14);
def rawK = if highK != lowK then 100 * (src - lowK) / (highK - lowK) else 50;
def stochK = Average(rawK, 3);
def macdV = ExpAverage(src, 12) - ExpAverage(src, 26);
def macdSig = ExpAverage(macdV, 9);
def momentumScore = (if rsiV > 50 then 1 else -1)
                  + (if stochK > 50 then 1 else -1)
                  + (if macdV > macdSig then 1 else -1);

# Volatility (VIX level bands — shared regime context)
def vix = close("VIX");
def vixScore = if vix < 18 then 2 else if vix < 22 then 0 else -3;

def totalScore = trendScore + momentumScore + vixScore;

plot Score = totalScore;
Score.AssignValueColor(
    if totalScore >= 6 then Color.GREEN
    else if totalScore >= 3 then Color.LIGHT_GREEN
    else if totalScore <= -6 then Color.DARK_RED
    else if totalScore <= -3 then Color.RED
    else Color.GRAY
);
# Background mirrors the color so the whole cell reads at a glance
AssignBackgroundColor(
    if totalScore >= 6 then CreateColor(0, 90, 0)
    else if totalScore >= 3 then CreateColor(0, 60, 0)
    else if totalScore <= -6 then CreateColor(120, 0, 0)
    else if totalScore <= -3 then CreateColor(80, 0, 0)
    else CreateColor(40, 40, 40)
);
