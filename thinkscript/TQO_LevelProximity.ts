# ═══════════════════════════════════════════════════════════════════
# TQO_LevelProximity — "is this name AT a level with room?" watchlist column
# The Teri layer as a scannable cell: distance to nearest buyer/seller
# level in ATRs, colored by whether a favorable-R:R entry is even possible.
#
# Green cell  = price near a level WITH room to the opposing level (setup候)
# Gray cell   = mid-range, no location edge (Teri: "don't chase")
# ═══════════════════════════════════════════════════════════════════
input levelLookback = 20;
input proximityATR = 0.5;   # "at a level" if within this many ATR
input minRoomRR = 2.0;      # room to opposing level must be >= this x risk

def src = close;
def atr = Average(TrueRange(high, close, low), 14);

def sellerLevel = Highest(high, levelLookback);   # resistance proxy
def buyerLevel  = Lowest(low, levelLookback);     # support proxy

def distToSeller = (sellerLevel - src) / atr;     # ATRs up to resistance
def distToBuyer  = (src - buyerLevel) / atr;      # ATRs down to support

# Are we AT a level (within proximity), and is the opposing level far enough
# to give favorable reward:risk if we enter here?
def atSupport    = distToBuyer <= proximityATR;
def atResistance = distToSeller <= proximityATR;
def roomFromSupport = distToSeller / Max(distToBuyer, 0.01);   # reward:risk long
def roomFromResist  = distToBuyer / Max(distToSeller, 0.01);   # reward:risk short

def longSetup  = atSupport and roomFromSupport >= minRoomRR;
def shortSetup = atResistance and roomFromResist >= minRoomRR;

# Cell value = ATRs to the NEAREST level (smaller = closer = more actionable)
def nearest = Min(distToSeller, distToBuyer);
plot Proximity = Round(nearest, 1);

Proximity.AssignValueColor(
    if longSetup then Color.GREEN
    else if shortSetup then Color.RED
    else if atSupport or atResistance then Color.YELLOW   # at level but poor R:R
    else Color.GRAY                                       # mid-range
);
AssignBackgroundColor(
    if longSetup then CreateColor(0, 80, 0)
    else if shortSetup then CreateColor(80, 0, 0)
    else if atSupport or atResistance then CreateColor(70, 70, 0)
    else CreateColor(35, 35, 35)
);
