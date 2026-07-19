"""
Global Masters Library
======================
Principles, not quotes-for-decoration. Each entry maps a practitioner to the
specific Quantum Maestro module their insight governs. Deliberately global:
the game is not American, and never was.

Entries are paraphrased principles in our own words.
"""

MASTERS = [
    {
        "name": "Teri Ijeoma", "origin": "USA (Trade & Travel)",
        "principle": "Trade liquid institutional names at institutional levels. Fix the risk, vary the size. "
                     "Predefine entry, stop, target — then execute without emotion.",
        "governs": "Technical Agent, Position Sizer, execution discipline",
        "caution": "Her method assumes you actually honor the stop. The system's job is to make dishonoring it impossible.",
    },
    {
        "name": "Paul Tudor Jones", "origin": "USA",
        "principle": "Defense first. Never average a loser. Press only when the opportunity is exceptional. "
                     "The 5:1 mindset: be wrong four times and still come out ahead.",
        "governs": "Regime Engine, R:R minimums, anti-averaging rule",
        "caution": "PTJ's edge included macro information networks retail cannot replicate. Copy the defense, not the swagger.",
    },
    {
        "name": "Warren Buffett", "origin": "USA",
        "principle": "Rule 1: don't lose money. Circle of competence: the size of the circle matters less than "
                     "knowing its boundary. Temperament beats intellect. Inactivity is a rational strategy.",
        "governs": "Watchlist discipline, NO-TRADE logging, Lockdown mode",
        "caution": "Buffett is not a trader — his lesson here is that most days the right position is the one you already have.",
    },
    {
        "name": "Ray Dalio", "origin": "USA",
        "principle": "Markets move in regimes driven by growth and inflation surprises. Diversify across "
                     "environments, not tickers. Pain + reflection = progress: the journal IS the strategy. "
                     "Write principles down so decisions outlive moods.",
        "governs": "Regime Engine design, Journal, this codebase's config-as-constitution pattern",
        "caution": "All-weather thinking suits allocation more than short-dated options. Import the epistemology, not the portfolio.",
    },
    {
        "name": "George Soros", "origin": "Hungary → UK/USA",
        "principle": "Reflexivity: prices change the fundamentals they supposedly reflect. It's not whether "
                     "you're right or wrong, but how much you make when right and lose when wrong.",
        "governs": "Psychology Agent (narrative feedback loops), asymmetric R:R rules",
        "caution": "Soros sized up on conviction in ways the Kelly cap here deliberately forbids for a system still proving its edge.",
    },
    {
        "name": "Jim Simons", "origin": "USA (Renaissance)",
        "principle": "Never override the model. If the system's edge is real it shows in the data; if it isn't, "
                     "no narrative saves it. Hire for science, trade the statistics.",
        "governs": "Promotion Gate, shadow-mode requirement, expectancy-from-journal-only rule",
        "caution": "Medallion's edge came from scale, data and execution infrastructure. The transferable lesson is the discipline, not the returns.",
    },
    {
        "name": "Ed Thorp", "origin": "USA",
        "principle": "Quantify the edge before betting. Kelly sizing — but fractional, because your estimate of "
                     "your own edge is itself uncertain.",
        "governs": "Kelly module, quarter-Kelly cap",
        "caution": "Thorp counted cards before he sized bets. Sequence: edge first, sizing second. Never reverse it.",
    },
    {
        "name": "André Kostolany", "origin": "Hungary → France",
        "principle": "The market runs on money + psychology. Distinguish strong hands (conviction + patience + "
                     "capital) from weak hands (leverage + hope). The profits are made in the sitting, "
                     "and 'sitting' includes sitting OUT.",
        "governs": "Lockdown guidance, cash-is-a-position doctrine, Market Ecology volume",
        "caution": "His four horsemen — money, patience, ideas, luck. The system can supply the first three; budget for the fourth being absent.",
    },
    {
        "name": "Nicolas Darvas", "origin": "Hungary",
        "principle": "Trade what IS, not what should be. Price consolidations (boxes) define entries and stops "
                     "mechanically. He made his fortune trading by telegram, weekly — distance from the "
                     "ticker was the edge, not a handicap.",
        "governs": "Anti-overtrading design; why this app has no live streaming P&L on purpose",
        "caution": "Survivorship applies; his box breakouts also failed. The keeper is mechanical invalidation.",
    },
    {
        "name": "Rakesh Jhunjhunwala", "origin": "India",
        "principle": "The market is supreme — respect price over opinion. Take losses fast, ride winners, and "
                     "keep trading capital strictly separate from investment capital.",
        "governs": "Account separation (Maestro vs. lab account), stop discipline",
        "caution": "His concentrated bets rode a structural India bull market — beta wearing an alpha costume is a global phenomenon.",
    },
    {
        "name": "Li Lu", "origin": "China → USA",
        "principle": "Intellectual honesty about the boundary of your knowledge is the whole game. Position size "
                     "should reflect how well you actually understand, not how excited you are.",
        "governs": "Thesis-required rule, agent evidence standards",
        "caution": "Value patience and options theta run on different clocks; import the honesty, not the holding period.",
    },
    {
        "name": "Takashi Kotegawa (BNF)", "origin": "Japan",
        "principle": "Mean-reversion on quantified extremes, near-zero leverage, and total indifference to "
                     "lifestyle inflation. Turned ~¥1.6M into billions of yen without ever blowing up — "
                     "the not-blowing-up being the actual achievement.",
        "governs": "Circuit breakers, heat cap, leverage bans",
        "caution": "His deviation-rate signals were tuned to 2000s JP microstructure; the transferable asset is the risk posture.",
    },
    {
        "name": "Jesse Livermore", "origin": "USA (cautionary volume)",
        "principle": "Everything right: the big money is in the waiting; markets repeat because humans repeat; "
                     "never argue with the tape.",
        "governs": "Patience doctrine",
        "caution": "Everything wrong: no external risk system, so he blew up repeatedly and died broke. Livermore "
                   "is why the risk engine is code and not a promise. He is the null hypothesis this project exists to reject.",
    },
]


def render_table():
    import pandas as pd
    return pd.DataFrame(MASTERS)
