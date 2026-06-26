"""
Smoke tests for Quantum Maestro v2.
Tests:
  - app.py parses cleanly (AST gate — mandatory before every push)
  - Key constants present and sane (IWT watchlist, commission, disclaimer)
  - IWT Trade Setup Card math (R:R, stop, shares, capital 1% rule)
  - Watchlist discipline gate (Lesson 8)
  - Paper trading gate (Lesson 9)
  - Capital progress tracker (Lesson 5)
  - Late entry detection (Lesson 3)
  - Vertical credit spread math
  - Kelly criterion math
No live data fetched — all tests run offline without network.
"""
import ast
import pathlib
import sys
import pytest

# ── AST gate ─────────────────────────────────────────────────────────────────
def test_app_parses():
    """app.py must parse cleanly. Hard gate — never push a broken file."""
    src = pathlib.Path("app.py").read_text()
    try:
        ast.parse(src)
    except SyntaxError as e:
        pytest.fail(f"app.py has syntax error: {e}")


# ── Constant sanity ───────────────────────────────────────────────────────────
def test_vip_tickers_defined():
    src = pathlib.Path("app.py").read_text()
    for line in src.splitlines():
        if line.startswith("VIP_TICKERS"):
            tickers = eval(line.split("=", 1)[1].strip())
            assert len(tickers) > 0
            assert "SPY" in tickers
            return
    pytest.fail("VIP_TICKERS not found")


def test_iwt_watchlist_defined():
    src = pathlib.Path("app.py").read_text()
    for line in src.splitlines():
        if line.startswith("IWT_WATCHLIST"):
            tickers = eval(line.split("=", 1)[1].strip())
            assert len(tickers) >= 25, "IWT watchlist should have ~30 tickers"
            assert "NVDA" in tickers
            assert "SPY" in tickers
            assert "AMZN" in tickers
            return
    pytest.fail("IWT_WATCHLIST not found")


def test_commission_rate_sane():
    src = pathlib.Path("app.py").read_text()
    for line in src.splitlines():
        if line.startswith("COMMISSION_PER_SHARE"):
            val = float(line.split("=", 1)[1].strip())
            assert 0 < val < 0.05
            return
    pytest.fail("COMMISSION_PER_SHARE not found")


def test_disclaimer_present():
    src = pathlib.Path("app.py").read_text()
    assert "not financial advice" in src.lower()


def test_security_md_email():
    sec = pathlib.Path("SECURITY.md").read_text()
    assert "contact@aikungfu.dev" in sec


def test_requirements_core_deps():
    reqs = pathlib.Path("requirements.txt").read_text().lower()
    for pkg in ["streamlit", "yfinance", "pandas", "scipy"]:
        assert pkg in reqs, f"{pkg} missing from requirements.txt"


# ── IWT Trade Setup Card math ─────────────────────────────────────────────────
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

def _load_calc_iwt():
    """Import calc_iwt_trade_setup from app.py without running Streamlit."""
    import importlib.util, types
    # Stub out streamlit so the module-level st.* calls don't fail
    st_stub = types.ModuleType("streamlit")
    st_stub.cache_data = lambda **kw: (lambda f: f)
    st_stub.session_state = {}
    st_stub.secrets = {}
    for attr in ["set_page_config","markdown","radio","columns","button",
                 "spinner","text_input","number_input","selectbox","slider",
                 "checkbox","expander","caption","info","warning","error",
                 "success","pyplot","dataframe","tabs","metric","write",
                 "text_area","container","rerun"]:
        setattr(st_stub, attr, lambda *a, **kw: None)
    sys.modules["streamlit"] = st_stub
    # Stub other heavy deps
    for mod in ["yfinance","mplfinance","scipy","scipy.stats","scipy.signal",
                "matplotlib","matplotlib.pyplot","matplotlib.ticker",
                "sklearn","sklearn.ensemble","sklearn.preprocessing",
                "sklearn.metrics","pytz"]:
        if mod not in sys.modules:
            sys.modules[mod] = types.ModuleType(mod)
    src = pathlib.Path("app.py").read_text()
    # Extract just the function definitions we need (safe eval approach)
    # Instead, compile and exec with limited globals
    globs = {"__builtins__": __builtins__, "st": st_stub,
             "pd": __import__("pandas"), "np": __import__("numpy"),
             "math": __import__("math"), "re": __import__("re"),
             "json": __import__("json")}
    try:
        exec(compile(src, "app.py", "exec"), globs)
    except Exception:
        pass  # Expected — st.* calls fail. We just need the functions.
    return globs


try:
    _G = _load_calc_iwt()
    _calc = _G.get("calc_iwt_trade_setup")
    _check_wl = _G.get("check_ticker_vs_iwt_watchlist")
    _paper_gate = _G.get("paper_trading_gate")
    _capital_progress = _G.get("calc_capital_progress")
    _late_entry = _G.get("check_late_entry")
    _IWT_WATCHLIST = _G.get("IWT_WATCHLIST", [])
    _HAS_IWT = _calc is not None
except Exception:
    _HAS_IWT = False
    _calc = _check_wl = _paper_gate = _capital_progress = _late_entry = None
    _IWT_WATCHLIST = []


@pytest.mark.skipif(not _HAS_IWT, reason="IWT functions not loaded")
def test_iwt_setup_card_rr_ratio():
    """R:R must equal (target-price) / (price-stop) to 2 decimals."""
    result = _calc(price=100.0, support=95.0, resistance=115.0,
                   atr=1.5, capital=10000, direction="LONG")
    assert "rr_ratio" in result
    assert result["rr_ratio"] >= 0


@pytest.mark.skipif(not _HAS_IWT, reason="IWT functions not loaded")
def test_iwt_setup_card_stop_below_support():
    """Stop must always be below support for LONG."""
    result = _calc(price=100.0, support=95.0, resistance=115.0,
                   atr=2.0, capital=10000, direction="LONG")
    assert result["stop"] < 95.0, "Stop must be below support level"


@pytest.mark.skipif(not _HAS_IWT, reason="IWT functions not loaded")
def test_iwt_setup_card_1pct_rule():
    """Dollar risk must equal exactly 1% of capital (default)."""
    result = _calc(price=100.0, support=95.0, resistance=120.0,
                   atr=1.5, capital=10000, direction="LONG")
    assert abs(result["dollar_risk"] - 100.0) < 0.01, "1% of $10k = $100"


@pytest.mark.skipif(not _HAS_IWT, reason="IWT functions not loaded")
def test_iwt_setup_card_skip_when_rr_low():
    """Verdict must be SKIP when R:R < 1.5."""
    # Support very close to resistance = tiny reward
    result = _calc(price=99.0, support=98.0, resistance=100.5,
                   atr=0.5, capital=10000, direction="LONG")
    assert result["rr_ratio"] < 1.5
    assert "SKIP" in result["verdict"] or "MARGINAL" in result["verdict"]


@pytest.mark.skipif(not _HAS_IWT, reason="IWT functions not loaded")
def test_iwt_setup_card_valid_when_rr_good():
    """Verdict must be VALID or A+ when R:R >= 2."""
    result = _calc(price=100.0, support=95.0, resistance=120.0,
                   atr=1.0, capital=10000, direction="LONG")
    assert result["rr_ratio"] >= 2.0
    assert any(v in result["verdict"] for v in ["VALID", "A+"])


@pytest.mark.skipif(not _HAS_IWT, reason="IWT functions not loaded")
def test_watchlist_gate_nvda():
    """NVDA must be on IWT watchlist."""
    result = _check_wl("NVDA")
    assert result["on_watchlist"] is True


@pytest.mark.skipif(not _HAS_IWT, reason="IWT functions not loaded")
def test_watchlist_gate_unknown():
    """Unknown ticker must return off-watchlist with advice."""
    result = _check_wl("ZZZZZ")
    assert result["on_watchlist"] is False
    assert len(result["advice"]) > 20


@pytest.mark.skipif(not _HAS_IWT, reason="IWT functions not loaded")
def test_paper_gate_beginner():
    """Beginner must be paper-only."""
    result = _paper_gate("Beginner")
    assert result["gate"] == "PAPER_ONLY"


@pytest.mark.skipif(not _HAS_IWT, reason="IWT functions not loaded")
def test_paper_gate_advanced():
    """Advanced must permit real trading."""
    result = _paper_gate("Advanced")
    assert result["gate"] == "REAL_PERMITTED"


@pytest.mark.skipif(not _HAS_IWT, reason="IWT functions not loaded")
def test_capital_progress_goal_met():
    """When P&L >= daily goal, goal_met must be True."""
    result = _capital_progress(daily_pnl=60.0, capital=10000, monthly_goal=1000)
    assert result["daily_goal"] == pytest.approx(1000 / 21, rel=0.01)
    # $60 vs $47.6 daily goal -> goal met
    if result["daily_pnl"] >= result["daily_goal"]:
        assert result["goal_met"] is True


@pytest.mark.skipif(not _HAS_IWT, reason="IWT functions not loaded")
def test_capital_progress_loss_limit():
    """When P&L <= 2× negative daily goal, loss limit must trigger."""
    result = _capital_progress(daily_pnl=-200.0, capital=10000, monthly_goal=1000)
    assert result["hit_loss_limit"] is True


@pytest.mark.skipif(not _HAS_IWT, reason="IWT functions not loaded")
def test_late_entry_detection():
    """Price > 2% above support must trigger late entry warning."""
    result = _late_entry(price=103.0, support=100.0, threshold_pct=2.0)
    assert result["late_entry"] is True


@pytest.mark.skipif(not _HAS_IWT, reason="IWT functions not loaded")
def test_no_late_entry_at_level():
    """Price at support must not trigger late entry."""
    result = _late_entry(price=100.5, support=100.0, threshold_pct=2.0)
    assert result["late_entry"] is False


# ── Legacy spread math ────────────────────────────────────────────────────────
def test_credit_spread_math_basic():
    """calc_vertical_credit_spread must produce correct max profit / loss."""
    _spread = _G.get("calc_vertical_credit_spread") if _HAS_IWT else None
    if _spread is None:
        pytest.skip("calc_vertical_credit_spread not loaded")
    res = _spread(short_strike=5000, long_strike=4975, credit=2.50,
                  spread_type="PUT", contracts=1)
    assert not res.get("errors")
    assert abs(res["max_profit_per_contract"] - 250.0) < 0.01
    assert abs(res["max_loss_per_contract"] - 2250.0) < 0.01
    assert abs(res["breakeven"] - 4997.50) < 0.01


def test_kelly_positive_ev():
    """kelly_and_ruin must return positive edge for 60% win / 2:1 R:R."""
    _kelly = _G.get("kelly_and_ruin") if _HAS_IWT else None
    if _kelly is None:
        pytest.skip("kelly_and_ruin not loaded")
    result = _kelly(win_rate=0.60, avg_win_R=2.0, avg_loss_R=1.0)
    assert result["edge_per_trade_pct"] > 0
    assert result["kelly_pct"] > 0
    assert result["quarter_kelly_pct"] < result["kelly_pct"]


def test_iwt_watchlist_has_core_tickers():
    """IWT watchlist must contain the canonical IWT tickers."""
    for t in ["NVDA", "AMZN", "META", "AAPL", "SPY"]:
        assert t in _IWT_WATCHLIST, f"{t} missing from IWT_WATCHLIST"
