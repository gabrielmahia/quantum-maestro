# ═══════════════════════════════════════════════════════════════════
# TQO_GuardRails — settlement + event + DTE alert study (lower or upper)
# Encodes the §17 operational guardrails as visible warnings + alerts.
# Especially: the "close physically-settled spreads before the pin-risk
# window" rule, and the 0DTE ban surfaced as a chart warning.
# ═══════════════════════════════════════════════════════════════════
input eventLockout = no;          # flip on FOMC/CPI/NFP/earnings
input isPhysicallySettled = no;   # yes for AAPL/equity options; no for SPX/XSP
input lateDayCloseHourET = 1500;  # 3:00pm ET close-spreads reminder for 0-1 DTE

# Time-of-day (ToS SecondsFromTime uses regular session)
def now = SecondsFromTime(0930) / 3600;   # hours since open (approx)
def marketMinutesLeft = SecondsTillTime(1600) / 60;

# --- 0DTE ban surfaced on the chart (mirrors MIN_DTE=7 in the app) ---
AddLabel(yes,
    "0DTE/expiry-day trading is BANNED by risk doctrine (MIN_DTE=7). Observation only.",
    Color.DARK_ORANGE);

# --- Event lockout ---
AddLabel(eventLockout,
    "EVENT LOCKOUT ACTIVE — no new short premium. Long hedges only.",
    Color.RED);
Alert(eventLockout and !eventLockout[1], "TQO: event lockout activated", Alert.BAR, Sound.Ring);

# --- Settlement warning ---
AddLabel(yes,
    if isPhysicallySettled then
        "PHYSICAL SETTLEMENT — assignment/exercise risk. Close spreads before expiry."
    else "CASH SETTLED (SPX-style) — no assignment, but a.m. settlement risk on expiry day.",
    if isPhysicallySettled then Color.YELLOW else Color.GRAY);

# --- Late-day close reminder for expiring physically-settled spreads ---
def lateDay = SecondsFromTime(lateDayCloseHourET) >= 0;
AddLabel(isPhysicallySettled and lateDay,
    "LATE DAY: if holding an expiring physical spread, CLOSE THE FULL SPREAD and confirm fill.",
    Color.RED);
Alert(isPhysicallySettled and lateDay and !lateDay[1],
    "TQO: late-day close-the-spread reminder (physical settlement)", Alert.BAR, Sound.Ring);

# --- Anti-repair reminder (surfacing the NO-REPAIR hard rule) ---
AddLabel(yes,
    "NO-REPAIR: a second spread on a loser is a NEW trade with its own tail. Don't campaign.",
    Color.GRAY);

plot MinutesLeft = marketMinutesLeft;
MinutesLeft.SetDefaultColor(Color.DARK_GRAY);
MinutesLeft.Hide();
