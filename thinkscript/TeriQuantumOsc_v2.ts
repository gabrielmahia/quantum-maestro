# ═══════════════════════════════════════════════════════════════════
# TeriQuantumOsc v2 — Quantum Maestro Regime Proxy (lower study)
# Teri Ijeoma levels + symmetric regime score + IV-rank premium bias
#
# DOCTRINE NOTES
# - This is a fast visual PROXY. The canonical regime engine lives in
#   the Quantum Maestro app (qm/regime.py) — it sees breadth, oil and
#   credit, which thinkScript cannot. When they disagree, the app wins.
# - Suggestions here never override the hard rules: no 0DTE, defined
#   risk only, 1% x regime multiplier, no short premium into events.
# ═══════════════════════════════════════════════════════════════════
declare lower;

input marketSymbol = "SPX";
input vixSymbol = "VIX";
input vix9dSymbol = "VIX9D";       # short-dated vol for stress ratio
input momentumOnMarket = yes;      # yes = pure regime study; no = chart momentum
input rsiLength = 14;              # 7 is noisy; 14 default, tune deliberately
input stochLength = 14;
input stochSmooth = 3;
input macdFast = 12;
input macdSlow = 26;
input macdSignalLength = 9;
input vixCaution = 18;
input vixDanger = 22;
input ivRankLookback = 252;        # for premium buy-vs-sell bias
input levelLookback = 20;          # buyer/seller level proxy (Teri)
input levelBufferATR = 0.5;        # min ATRs of headroom to a level
input eventBlackout = no;          # SET MANUALLY on FOMC/CPI/earnings weeks

# ── Reference frames ────────────────────────────────────────────────
def market = close(marketSymbol);
def vix    = close(vixSymbol);
def vix9d  = close(vix9dSymbol);
def src    = if momentumOnMarket then market else close;
def srcH   = if momentumOnMarket then high(marketSymbol) else high;
def srcL   = if momentumOnMarket then low(marketSymbol)  else low;

# ── Momentum (each component ±1) ───────────────────────────────────
def netChg   = src - src[1];
def rsiUp    = WildersAverage(Max(netChg, 0), rsiLength);
def rsiDn    = WildersAverage(Max(-netChg, 0), rsiLength);
def rsiV     = if rsiDn == 0 then 100 else 100 - 100 / (1 + rsiUp / rsiDn);
def lowK     = Lowest(srcL, stochLength);
def highK    = Highest(srcH, stochLength);
def rawK     = if highK != lowK then 100 * (src - lowK) / (highK - lowK) else 50;
def stochK   = Average(rawK, stochSmooth);
def macdV    = ExpAverage(src, macdFast) - ExpAverage(src, macdSlow);
def macdSig  = ExpAverage(macdV, macdSignalLength);
def momentumScore = (if rsiV > 50 then 1 else -1)
                  + (if stochK > 50 then 1 else -1)
                  + (if macdV > macdSig then 1 else -1);

# ── Trend: SYMMETRIC ladder (v1 flaw: bullish +3 vs bearish floor -1) ─
def sma20  = Average(market, 20);
def sma50  = Average(market, 50);
def sma200 = Average(market, 200);
def trendScore =
    if market > sma20 and sma20 > sma50 and sma50 > sma200 then  3
    else if market < sma20 and sma20 < sma50 and sma50 < sma200 then -3
    else if market > sma50 and market > sma200 then  1
    else if market < sma50 and market < sma200 then -1
    else 0;

# ── Volatility: level AND acceleration ─────────────────────────────
def vixLevelScore =
    if vix < vixCaution then 2
    else if vix < vixDanger then 0
    else -3;
def stressRatio = if vix > 0 then vix9d / vix else 1;   # >1 = backwardation
def vixAccelScore =
    if stressRatio > 1.02 then -2                        # stress accelerating
    else if stressRatio < 0.90 then 1                    # term structure calm
    else 0;
def vixScore = vixLevelScore + vixAccelScore;

# ── Composite (range now symmetric ~[-11..+9]) ─────────────────────
def rawTotal = trendScore + momentumScore + vixScore;
def totalScore = if eventBlackout then Min(rawTotal, 0) else rawTotal;

# ── Teri layer: buyer/seller level proxy + headroom gate ───────────
def sellerLevel = Highest(srcH, levelLookback);          # resistance proxy
def buyerLevel  = Lowest(srcL, levelLookback);           # support proxy
def atr = Average(TrueRange(srcH, src, srcL), 14);
def roomUp   = (sellerLevel - src) / atr;                # ATRs to resistance
def roomDown = (src - buyerLevel) / atr;                 # ATRs to support
def bullEntryOK = roomUp   >= levelBufferATR;            # don't buy into the ceiling
def bearEntryOK = roomDown >= levelBufferATR;            # don't short into the floor

# ── IV rank: premium buy-vs-sell bias (v1 flaw: used VIX level) ────
def iv = imp_volatility(marketSymbol);
def ivRank = if IsNaN(iv) then 50 else
    100 * (iv - Lowest(iv, ivRankLookback)) /
    Max(Highest(iv, ivRankLookback) - Lowest(iv, ivRankLookback), 0.0001);
def premiumRich = ivRank >= 50;                          # sell spreads
def premiumCheap = ivRank <= 25;                         # buy debit structures

# ── Plots ──────────────────────────────────────────────────────────
plot RegimeScore = totalScore;
RegimeScore.SetPaintingStrategy(PaintingStrategy.HISTOGRAM);
RegimeScore.SetLineWeight(4);
RegimeScore.AssignValueColor(
    if totalScore >= 6 then Color.GREEN
    else if totalScore >= 3 then Color.LIGHT_GREEN
    else if totalScore <= -6 then Color.DARK_RED
    else if totalScore <= -3 then Color.RED
    else Color.GRAY
);
plot Zero = 0;
Zero.SetDefaultColor(Color.GRAY);
Zero.HideBubble();
plot BullThreshold = 6;
BullThreshold.SetDefaultColor(Color.GREEN);
BullThreshold.SetStyle(Curve.SHORT_DASH);
BullThreshold.HideBubble();
plot BearThreshold = -6;
BearThreshold.SetDefaultColor(Color.RED);
BearThreshold.SetStyle(Curve.SHORT_DASH);
BearThreshold.HideBubble();

# ── Labels (aligned with app vocabulary) ───────────────────────────
AddLabel(yes,
    if eventBlackout then "QM: LOCKDOWN — event blackout set"
    else if totalScore >= 6 then "QM: OFFENSIVE (1.0x)"
    else if totalScore >= 3 then "QM: NEUTRAL+ (0.6x)"
    else if totalScore <= -6 then "QM: LOCKDOWN (0x) — cash"
    else if totalScore <= -3 then "QM: DEFENSIVE (0.3x)"
    else "QM: NEUTRAL (0.6x) — A-grade setups only",
    if eventBlackout then Color.DARK_ORANGE
    else if totalScore >= 6 then Color.GREEN
    else if totalScore >= 3 then Color.LIGHT_GREEN
    else if totalScore <= -6 then Color.DARK_RED
    else if totalScore <= -3 then Color.RED
    else Color.GRAY);
AddLabel(yes, "Score " + totalScore + "  (T " + trendScore + " / M " + momentumScore + " / V " + vixScore + ")", Color.WHITE);
AddLabel(yes, "VIX " + Round(vix, 2) + "  9D/30D " + Round(stressRatio, 2),
    if stressRatio > 1.02 then Color.RED else if vix < vixCaution then Color.GREEN else Color.YELLOW);
AddLabel(yes, "IV Rank " + Round(ivRank, 0) + (if premiumRich then " — SELL premium" else if premiumCheap then " — BUY premium" else " — mixed"),
    if premiumRich then Color.ORANGE else if premiumCheap then Color.CYAN else Color.GRAY);
AddLabel(yes, "Room: up " + Round(roomUp, 1) + " ATR / down " + Round(roomDown, 1) + " ATR", Color.LIGHT_GRAY);

# ── Structure suggestion: regime x IV rank x level headroom ────────
AddLabel(yes,
    if eventBlackout then "Play: NONE — no new short premium into events"
    else if totalScore >= 3 and !bullEntryOK then "Play: WAIT — bullish but <" + levelBufferATR + " ATR under seller level (Teri: bad R:R)"
    else if totalScore >= 3 and premiumRich then "Play: Put Credit Spread below buyer level (7-45 DTE)"
    else if totalScore >= 3 and premiumCheap then "Play: Bull Call Spread / 60+DTE Calls (IV cheap)"
    else if totalScore >= 3 then "Play: Bull Call Spread (defined risk)"
    else if totalScore <= -3 and !bearEntryOK then "Play: WAIT — bearish but <" + levelBufferATR + " ATR above buyer level"
    else if totalScore <= -6 then "Play: CASH / long hedges only — no short premium in stress"
    else if totalScore <= -3 and premiumCheap then "Play: Bear Put Spread (0.3x) / hedges"
    else if totalScore <= -3 then "Play: reduce size, prefer cash; CCS only at 0.3x with headroom"
    else "Play: WAIT — no edge (cash is a position)",
    Color.CYAN);
