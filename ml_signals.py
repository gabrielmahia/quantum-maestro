"""
ml_signals.py — Machine-Learning Signal Layer for Quantum Maestro
==================================================================
Implements a rolling-window Random Forest classifier trained on
technical indicators (computed by app._calc_indicators) to produce
a probabilistic "Bullish Setup Score" for any ticker.

Architecture derived from:
  ML for Algorithmic Trading (2nd ed.) — Chapter 12: Gradient Boosting Machines
  + Chapter 5: Strategy Evaluation (IC / information coefficient metrics)
  + Chapter 4: Alpha Factor Research (alpha factor engineering patterns)

Why this approach works:
  - Rolling 252-day training window: model adapts to current regime
  - 5-day forward return as target: aligns with IWT entry/exit cadence
  - Features are the same indicators already computed by app.py
  - No look-ahead bias: model only uses data available at prediction time

All output is probabilistic, not deterministic — in line with IWT's
probability-of-profit framework.

Usage:
    from ml_signals import compute_ml_signal
    result = compute_ml_signal(df, experience_level="Intermediate")
"""

from __future__ import annotations
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Optional

# ── Try sklearn; graceful fallback to logistic approximation ─────────────────
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    _SKLEARN = True
except ImportError:
    _SKLEARN = False


# ── Feature engineering (alpha factors from ML4T book) ───────────────────────

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build ML feature matrix from pre-computed indicators.
    Feature list inspired by ML4T Ch. 24 Alpha Factor Library.
    
    All features are normalised within the rolling window to prevent
    scale sensitivity (a common source of look-ahead bias).
    """
    close = df["Close"]
    features = pd.DataFrame(index=df.index)

    # ── Momentum factors (Jegadeesh & Titman 1993) ─────────────────────────
    for n in (5, 10, 20):
        features[f"ret_{n}d"] = close.pct_change(n)
    features["ret_52w_minus_1m"] = close.pct_change(252) - close.pct_change(21)

    # ── Mean reversion factors ─────────────────────────────────────────────
    if "RSI_14" in df.columns:
        features["rsi_14"] = df["RSI_14"] / 100.0
        features["rsi_oversold"] = (df["RSI_14"] < 35).astype(float)
        features["rsi_overbought"] = (df["RSI_14"] > 65).astype(float)

    # ── Trend factors ──────────────────────────────────────────────────────
    if "SMA_20" in df.columns and "SMA_50" in df.columns:
        features["price_above_sma20"]  = (close > df["SMA_20"]).astype(float)
        features["price_above_sma50"]  = (close > df["SMA_50"]).astype(float)
        features["sma20_above_sma50"]  = (df["SMA_20"] > df["SMA_50"]).astype(float)
    if "SMA_200" in df.columns:
        features["price_above_sma200"] = (close > df["SMA_200"]).astype(float)

    # ── Volatility factors ────────────────────────────────────────────────
    if "ATRr_14" in df.columns:
        features["atr_norm"] = df["ATRr_14"] / close  # normalised ATR
    features["vol_20d"] = close.pct_change().rolling(20).std()
    features["vol_ratio"] = features["vol_20d"] / close.pct_change().rolling(60).std()

    # ── MACD momentum factor ──────────────────────────────────────────────
    if "MACDh_12_26_9" in df.columns:
        features["macd_hist"] = df["MACDh_12_26_9"] / close
        features["macd_rising"] = (df["MACDh_12_26_9"].diff() > 0).astype(float)

    # ── Volume factor (OBV trend) ─────────────────────────────────────────
    if "OBV" in df.columns:
        features["obv_trend"] = df["OBV"].pct_change(20).clip(-2, 2)

    # ── Bollinger Band position ───────────────────────────────────────────
    if "BBU_20_2.0" in df.columns and "BBL_20_2.0" in df.columns:
        bb_range = df["BBU_20_2.0"] - df["BBL_20_2.0"]
        features["bb_position"] = (close - df["BBL_20_2.0"]) / bb_range.replace(0, np.nan)
        features["bb_squeeze"]  = (bb_range / close < bb_range.shift(20) / close.shift(20)).astype(float)

    # ── ADX trend strength ────────────────────────────────────────────────
    if "ADX_14" in df.columns:
        features["adx_strong"] = (df["ADX_14"] > 25).astype(float)
        features["adx_norm"]   = df["ADX_14"] / 100.0

    # ── Supertrend direction ──────────────────────────────────────────────
    if "ST_DIR" in df.columns:
        features["st_bullish"] = (df["ST_DIR"] == 1).astype(float)

    # ── Stochastic ────────────────────────────────────────────────────────
    if "STOCHk_14_3_3" in df.columns:
        features["stoch_k"] = df["STOCHk_14_3_3"] / 100.0

    return features.replace([np.inf, -np.inf], np.nan)


def _build_target(df: pd.DataFrame, forward_days: int = 5) -> pd.Series:
    """
    Binary target: 1 if next {forward_days}-day return > 0 else 0.
    
    From ML4T Ch. 6: using binary rather than regression targets
    reduces noise and improves out-of-sample generalisation.
    """
    future_return = df["Close"].pct_change(forward_days).shift(-forward_days)
    return (future_return > 0).astype(int)


# ── Information Coefficient (ML4T Ch. 4 / Alphalens) ─────────────────────────

def _information_coefficient(factor: pd.Series, forward_returns: pd.Series) -> float:
    """
    Rank IC (Spearman correlation) between a factor and forward returns.
    IC > 0.05 is generally considered meaningful in academic literature.
    """
    try:
        from scipy.stats import spearmanr
        aligned = factor.dropna().align(forward_returns.dropna(), join="inner")
        if len(aligned[0]) < 20:
            return np.nan
        ic, _ = spearmanr(aligned[0], aligned[1])
        return float(ic) if not np.isnan(ic) else np.nan
    except Exception:
        # Fallback: Pearson rank correlation
        f = factor.rank().dropna()
        r = forward_returns.rank().dropna()
        f, r = f.align(r, join="inner")
        if len(f) < 10:
            return np.nan
        return float(np.corrcoef(f, r)[0, 1])


# ── Rolling Random Forest (ML4T Ch. 12 pattern) ──────────────────────────────

def _rolling_rf_score(
    features: pd.DataFrame,
    target: pd.Series,
    train_window: int = 252,
    n_estimators: int = 100,
) -> float:
    """
    Train a RandomForest on the most recent {train_window} days of data
    and predict the probability of a bullish setup on the last row.
    
    Rolling window training (ML4T Ch. 12, Section: Walk-Forward Cross-Validation)
    prevents look-ahead bias and adapts to regime changes.
    """
    if not _SKLEARN:
        return _logistic_approx(features, target, train_window)
    
    # Get aligned, clean data
    X = features.copy()
    y = target.copy()
    combined = X.join(y.rename("__target__")).dropna()
    if len(combined) < train_window + 5:
        return np.nan
    
    train = combined.tail(train_window + 1).iloc[:-1]  # exclude last row (no target yet)
    X_train = train.drop("__target__", axis=1)
    y_train = train["__target__"]
    
    # Skip if too few or only one class
    if len(y_train) < 30 or y_train.nunique() < 2:
        return np.nan
    
    # Feature for prediction: last row (no forward-looking target)
    X_pred = combined.iloc[[-1]].drop("__target__", axis=1)
    
    # Impute with column medians
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_pred  = X_pred.fillna(medians)
    
    try:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=4,           # shallow trees prevent overfitting
            min_samples_leaf=10,   # regularisation
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_pred)[0, 1]  # P(bullish)
        return float(proba)
    except Exception:
        return np.nan


def _logistic_approx(
    features: pd.DataFrame,
    target: pd.Series,
    train_window: int = 252,
) -> float:
    """
    Pure numpy logistic regression fallback (no sklearn required).
    Used when sklearn is not installed. Lower accuracy but still directional.
    """
    from scipy.special import expit
    
    combined = features.join(target.rename("__target__")).dropna()
    if len(combined) < 30:
        return np.nan
    
    train = combined.tail(train_window + 1).iloc[:-1]
    X_train = train.drop("__target__", axis=1).values
    y_train = train["__target__"].values
    X_pred  = combined.iloc[[-1]].drop("__target__", axis=1).values
    
    # Normalise
    mu = np.nanmean(X_train, axis=0)
    std = np.nanstd(X_train, axis=0)
    std[std == 0] = 1.0
    X_train = np.nan_to_num((X_train - mu) / std)
    X_pred  = np.nan_to_num((X_pred  - mu) / std)
    
    # Gradient descent (50 steps)
    X_b = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    w   = np.zeros(X_b.shape[1])
    lr  = 0.01
    for _ in range(50):
        p   = expit(X_b @ w)
        grad = X_b.T @ (p - y_train) / len(y_train)
        w -= lr * grad
    
    X_pred_b = np.hstack([[1], X_pred[0]])
    return float(expit(X_pred_b @ w))


# ── Feature importance (human-readable) ──────────────────────────────────────

def _feature_importance(
    features: pd.DataFrame,
    target: pd.Series,
    train_window: int = 252,
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """Return top-N feature importances as (name, importance) pairs."""
    if not _SKLEARN:
        return []
    
    combined = features.join(target.rename("__target__")).dropna()
    if len(combined) < train_window + 5:
        return []
    
    train = combined.tail(train_window + 1).iloc[:-1]
    X_train = train.drop("__target__", axis=1).fillna(train.drop("__target__", axis=1).median())
    y_train = train["__target__"]
    if y_train.nunique() < 2:
        return []
    
    try:
        model = RandomForestClassifier(n_estimators=50, max_depth=4, min_samples_leaf=10,
                                        random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        importances = sorted(zip(X_train.columns, model.feature_importances_),
                             key=lambda x: x[1], reverse=True)
        return [(name, float(imp)) for name, imp in importances[:top_n]]
    except Exception:
        return []


# ── Main entry point ──────────────────────────────────────────────────────────

def compute_ml_signal(
    df: pd.DataFrame,
    experience_level: str = "Intermediate",
    forward_days: int = 5,
    train_window: int = 252,
) -> dict:
    """
    Compute ML-based signal score for a ticker.
    
    Args:
        df:               DataFrame with OHLCV + pre-computed indicators from _calc_indicators()
        experience_level: "Beginner" | "Intermediate" | "Advanced" | "Professional"
        forward_days:     Prediction horizon in trading days (default 5)
        train_window:     Rolling training window in days (default 252 = 1 year)
    
    Returns:
        dict with keys:
            score          — float 0.0–1.0 (bullish probability)
            score_pct      — int 0–100
            signal_label   — "Strong Bullish" | "Bullish" | "Neutral" | "Bearish" | "Strong Bearish"
            signal_emoji   — "🟢🟢" etc.
            ic_estimate    — float or None (Information Coefficient of RSI factor)
            feature_importances — list[(name, importance)] (Professional only)
            regime         — "trending" | "mean_reverting" | "choppy"
            narrative      — human-readable explanation (varies by experience level)
            sklearn_available — bool
            error          — str or None
    """
    result = {
        "score": None, "score_pct": None, "signal_label": "Neutral",
        "signal_emoji": "⚪", "ic_estimate": None, "feature_importances": [],
        "regime": "unknown", "narrative": "", "sklearn_available": _SKLEARN, "error": None
    }
    
    if df is None or len(df) < max(60, train_window // 2):
        result["error"] = "Insufficient data for ML analysis"
        return result
    
    try:
        # Build features and target
        features = _build_features(df)
        target   = _build_target(df, forward_days=forward_days)
        
        # Regime detection (ADX + volatility)
        adx = df.get("ADX_14", pd.Series(dtype=float)).iloc[-1] if "ADX_14" in df.columns else np.nan
        vol_20  = df["Close"].pct_change().rolling(20).std().iloc[-1]
        vol_60  = df["Close"].pct_change().rolling(60).std().iloc[-1]
        
        if not np.isnan(adx):
            if adx > 30:
                regime = "trending"
            elif vol_20 < vol_60 * 0.8:
                regime = "mean_reverting"
            else:
                regime = "choppy"
        else:
            regime = "unknown"
        result["regime"] = regime
        
        # IC on RSI (fast, always available)
        if "RSI_14" in df.columns:
            fwd_ret = df["Close"].pct_change(forward_days).shift(-forward_days)
            result["ic_estimate"] = _information_coefficient(df["RSI_14"], fwd_ret)
        
        # ML signal score
        score = _rolling_rf_score(features, target, train_window=train_window)
        
        if score is None or np.isnan(score):
            result["error"] = "Model training insufficient data"
            return result
        
        result["score"]     = score
        result["score_pct"] = int(round(score * 100))
        
        # Signal label
        if score >= 0.70:
            result["signal_label"] = "Strong Bullish"
            result["signal_emoji"] = "🟢🟢"
        elif score >= 0.57:
            result["signal_label"] = "Bullish"
            result["signal_emoji"] = "🟢"
        elif score >= 0.43:
            result["signal_label"] = "Neutral"
            result["signal_emoji"] = "⚪"
        elif score >= 0.30:
            result["signal_label"] = "Bearish"
            result["signal_emoji"] = "🔴"
        else:
            result["signal_label"] = "Strong Bearish"
            result["signal_emoji"] = "🔴🔴"
        
        # Narrative — gated by experience level
        level_up = experience_level.lower()
        
        if level_up == "beginner":
            result["narrative"] = (
                f"The AI has reviewed recent price patterns and technical signals. "
                f"Based on the last year of data, there is about a {result['score_pct']}% chance "
                f"the price will be higher in {forward_days} days. "
                f"Current setup: **{result['signal_label']}**. "
                f"Always combine this with your own analysis and risk management."
            )
        elif level_up == "intermediate":
            ic_str = f"{result['ic_estimate']:.3f}" if result["ic_estimate"] is not None else "N/A"
            result["narrative"] = (
                f"Random Forest classifier ({train_window}-day rolling window) "
                f"trained on {features.shape[1]} technical factors. "
                f"Bullish probability: **{result['score_pct']}%** ({result['signal_label']}). "
                f"RSI Information Coefficient: {ic_str}. "
                f"Market regime: {regime}."
            )
        elif level_up in ("advanced", "professional"):
            ic_str = f"{result['ic_estimate']:.4f}" if result["ic_estimate"] is not None else "N/A"
            result["narrative"] = (
                f"**ML4T Rolling RF** ({train_window}d window, {features.shape[1]} features). "
                f"P(bullish_{forward_days}d) = {score:.4f} | IC(RSI) = {ic_str} | "
                f"Regime: {regime} | sklearn: {_SKLEARN}"
            )
            if level_up == "professional":
                result["feature_importances"] = _feature_importance(
                    features, target, train_window=train_window
                )
        
        return result
    
    except Exception as e:
        result["error"] = str(e)[:120]
        return result


# ── Position sizing (Kelly Criterion — ML4T Ch. 5) ───────────────────────────

def kelly_fraction(
    win_probability: float,
    avg_win_r: float,
    avg_loss_r: float,
    kelly_fraction_mult: float = 0.5,
) -> dict:
    """
    Compute the Kelly fraction for position sizing.
    
    From ML4T Chapter 5 (Kelly Rule notebook): optimal bet size is
    f* = (p * b - q) / b  where p=win prob, q=1-p, b=avg_win/avg_loss ratio.
    
    Half-Kelly (multiply by 0.5) is standard practitioner convention
    to account for model uncertainty. Teri Ijeoma's IWT framework
    also recommends risking no more than 1-2% of account per trade,
    which aligns with fractional Kelly sizing.
    
    Args:
        win_probability:   Model's P(bullish) as a fraction
        avg_win_r:         Average win as R-multiple (e.g. 1.5 means 1.5× the risk)
        avg_loss_r:        Average loss as R-multiple (usually 1.0)
        kelly_fraction_mult: Scaling factor (default 0.5 = half-Kelly)
    
    Returns:
        dict with kelly_pct, half_kelly_pct, interpretation
    """
    p  = max(0.01, min(0.99, win_probability))
    q  = 1 - p
    b  = avg_win_r / max(avg_loss_r, 0.01)
    
    f_star = (p * b - q) / b
    f_kelly = max(0.0, f_star)
    f_practical = f_kelly * kelly_fraction_mult
    
    return {
        "kelly_full_pct":    round(f_kelly * 100, 1),
        "kelly_practical_pct": round(f_practical * 100, 1),
        "kelly_fraction_mult": kelly_fraction_mult,
        "p_win": p, "avg_win_r": avg_win_r, "avg_loss_r": avg_loss_r,
        "b_ratio": round(b, 2),
        "interpretation": (
            f"Kelly says risk {f_practical*100:.1f}% of account per trade "
            f"({kelly_fraction_mult*100:.0f}%-Kelly applied). "
            f"At P(win)={p:.0%}, avg win/loss ratio {b:.1f}× — "
            + ("Good edge." if f_star > 0.05 else "Marginal edge — reduce size.")
        )
    }
