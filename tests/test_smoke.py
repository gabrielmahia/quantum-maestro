"""
Smoke tests for Quantum Maestro.
Validates app.py parses and key constants are sane.
No live data fetched — runs without network.
"""
import ast
import pathlib
import pytest


def test_app_parses():
    src = pathlib.Path("app.py").read_text()
    try:
        ast.parse(src)
    except SyntaxError as e:
        pytest.fail(f"app.py has syntax error: {e}")


def test_vip_tickers_defined():
    src = pathlib.Path("app.py").read_text()
    for line in src.splitlines():
        if line.startswith("VIP_TICKERS"):
            tickers = eval(line.split("=", 1)[1].strip())
            assert len(tickers) > 0
            assert "SPY" in tickers
            return
    pytest.fail("VIP_TICKERS not found")


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
    for pkg in ["streamlit", "yfinance", "pandas"]:
        assert pkg in reqs, f"{pkg} missing from requirements.txt"
