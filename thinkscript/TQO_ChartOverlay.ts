# ═══════════════════════════════════════════════════════════════════
# TQO_ChartOverlay — buyer/seller zones + setup card (UPPER chart study)
# The §13.2 price-chart overlay from the system design doc.
# Pairs with TeriQuantumOsc_v2 (lower panel). This one draws ON the price.
#
# Plots nearest daily + weekly buyer/seller levels as clouds, an ATR
# extension band, and an on-chart setup card (entry / stop / targets / R:R).
# DOCTRINE: display only. Never sizes from balances, never places orders.
# ═══════════════════════════════════════════════════════════════════
input dailyLookback = 20;
input weeklyLookback = 20;      # in the chart's own aggregation
input atrLength = 14;
input minRR = 2.0;
input showSetupCard = yes;

def atr = Average(TrueRange(high, close, low), atrLength);

# Daily-horizon levels (chart timeframe)
def dSeller = Highest(high, dailyLookback);
def dBuyer  = Lowest(low, dailyLookback);

# Weekly-horizon levels (wider lookback as a proxy for higher timeframe)
def wSeller = Highest(high, weeklyLookback * 5);
def wBuyer  = Lowest(low, weeklyLookback * 5);

plot DailySeller = dSeller;
plot DailyBuyer  = dBuyer;
plot WeeklySeller = wSeller;
plot WeeklyBuyer  = wBuyer;

DailySeller.SetDefaultColor(Color.LIGHT_RED);
DailyBuyer.SetDefaultColor(Color.LIGHT_GREEN);
DailySeller.SetStyle(Curve.SHORT_DASH);
DailyBuyer.SetStyle(Curve.SHORT_DASH);
WeeklySeller.SetDefaultColor(Color.RED);
WeeklyBuyer.SetDefaultColor(Color.GREEN);
WeeklySeller.SetLineWeight(2);
WeeklyBuyer.SetLineWeight(2);

# ATR extension band — "don't chase" visual (Teri: penalize extended entries)
plot ExtHigh = Average(close, 20) + 2 * atr;
plot ExtLow  = Average(close, 20) - 2 * atr;
ExtHigh.SetDefaultColor(Color.DARK_GRAY);
ExtLow.SetDefaultColor(Color.DARK_GRAY);
ExtHigh.SetStyle(Curve.POINTS);
ExtLow.SetStyle(Curve.POINTS);

# Location logic for the setup card
def distToSeller = (dSeller - close) / atr;
def distToBuyer  = (close - dBuyer) / atr;
def atSupport    = distToBuyer <= 0.5;
def atResistance = distToSeller <= 0.5;
def rrLong  = distToSeller / Max(distToBuyer, 0.01);
def rrShort = distToBuyer / Max(distToSeller, 0.01);

# Setup card
AddLabel(showSetupCard,
    "TQO SETUP — " + GetSymbol(), Color.WHITE);
AddLabel(showSetupCard,
    "Room: up " + Round(distToSeller, 1) + " ATR / down " + Round(distToBuyer, 1) + " ATR",
    Color.LIGHT_GRAY);
AddLabel(showSetupCard,
    if atSupport and rrLong >= minRR then
        "LONG candidate: entry ~" + Round(close, 2) + " stop <" + Round(dBuyer, 2) +
        " T1 " + Round(dSeller, 2) + "  R:R " + Round(rrLong, 1)
    else if atResistance and rrShort >= minRR then
        "SHORT candidate: entry ~" + Round(close, 2) + " stop >" + Round(dSeller, 2) +
        " T1 " + Round(dBuyer, 2) + "  R:R " + Round(rrShort, 1)
    else if atSupport or atResistance then
        "AT LEVEL but R:R < " + minRR + " — no chase (Teri)"
    else "MID-RANGE — wait for a buyer/seller zone",
    if (atSupport and rrLong >= minRR) then Color.GREEN
    else if (atResistance and rrShort >= minRR) then Color.RED
    else if (atSupport or atResistance) then Color.YELLOW
    else Color.GRAY);
AddLabel(showSetupCard,
    "Extended? " + (if close > ExtHigh then "YES — overbought, don't chase"
                    else if close < ExtLow then "YES — oversold extension"
                    else "no — within bands"),
    if close > ExtHigh or close < ExtLow then Color.ORANGE else Color.GRAY);
