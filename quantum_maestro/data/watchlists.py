"""Canonical IWT watchlists — single source of truth.

TERI_OCT_2024: Teri Ijeoma's actual published watchlist, October 2024,
preserved verbatim as historical record (includes $VIX.X and SQ as she
listed them).

IWT_WATCHLIST_CURRENT: the tradeable, corrected version for live use:
  - $VIX.X excluded (index — regime input, not a position)
  - SQ -> XYZ (Block ticker change, Jan 2025)
  - SOXL flagged leveraged (3x) — excluded from agent allowlists by policy

A watchlist is a snapshot, not doctrine: names rotate as leadership and
liquidity rotate. Re-audit quarterly.
"""

TERI_OCT_2024 = {
    "$VIX.X": "CBOE Volatility Index",
    "AAP": "Advance Auto Parts", "AAPL": "Apple", "ADBE": "Adobe",
    "AMD": "Advanced Micro Devices", "AMZN": "Amazon.com", "AZO": "AutoZone",
    "CMG": "Chipotle Mexican Grill", "CRM": "Salesforce",
    "CRWD": "CrowdStrike Holdings", "DIA": "SPDR Dow Jones ETF",
    "GOOGL": "Alphabet", "GS": "Goldman Sachs", "HD": "Home Depot",
    "KBH": "KB Home", "META": "Meta Platforms", "MSFT": "Microsoft",
    "NFLX": "Netflix", "NVDA": "Nvidia", "PYPL": "PayPal",
    "SBUX": "Starbucks", "SHOP": "Shopify", "SMH": "VanEck Semiconductor ETF",
    "SOXL": "Direxion Daily Semiconductor 3x", "SPY": "SPDR S&P 500 ETF",
    "SQ": "Block (ticker now XYZ)", "TGT": "Target", "TOL": "Toll Brothers",
    "TSLA": "Tesla", "TTD": "The Trade Desk", "V": "Visa", "WMT": "Walmart",
    "XLK": "Technology Select Sector ETF", "XLY": "Consumer Discretionary ETF",
}

REGIME_INPUTS = ["$VIX.X"]                 # watch, never trade
LEVERAGED = ["SOXL"]                       # tradeable but excluded from agents
TICKER_RENAMES = {"SQ": "XYZ"}

IWT_WATCHLIST_CURRENT = sorted(
    (TICKER_RENAMES.get(t, t))
    for t in TERI_OCT_2024
    if t not in REGIME_INPUTS
)

IWT_WATCHLIST_AGENT_SAFE = [t for t in IWT_WATCHLIST_CURRENT if t not in LEVERAGED]


def on_watchlist(ticker: str, include_leveraged: bool = True) -> bool:
    t = ticker.upper().strip()
    wl = IWT_WATCHLIST_CURRENT if include_leveraged else IWT_WATCHLIST_AGENT_SAFE
    return t in wl
