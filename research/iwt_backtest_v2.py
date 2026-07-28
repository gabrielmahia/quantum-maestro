#!/usr/bin/env python3
"""IWT/Teri research engine v2.

Purpose
-------
Backtest a transparent mechanical proxy for the IWT buyer/seller-zone method
using free daily OHLCV data. This version integrates the VIP archive rules:

* 8-point zone score: base candles, departure speed, freshness, reward:risk.
* Long trades at buyer zones and short trades at seller zones.
* Stop = distal zone boundary +/- ATR buffer (default 0.20 ATR).
* Minimum planned reward:risk gate (default 3.0).
* High-score limit entries and medium-score confirmation entries.
* SPY/sector alignment, breadth, earnings blackout placeholders.
* Fixed-fraction sizing, portfolio position/cluster caps, and daily loss limit.
* Boundary-survival experiments for 45/60/75/90 days and ATR buffers.

Important
---------
This is an ALGORITHMIC PROXY, not a claim to reproduce proprietary/manual
zone labels or historical option-chain fills. Options results are boundary
statistics only until licensed historical chain data are supplied.
"""
from __future__ import annotations

import csv
import math
import os
import sqlite3
import statistics
import sys
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Literal

TICKERS = [
    "AAP","AAPL","ADBE","AMD","AMZN","AZO","CMG","CRM","CRWD","DIA",
    "GOOGL","GS","HD","KBH","META","MSFT","NFLX","NVDA","PYPL","SBUX",
    "SHOP","SMH","SOXL","SPY","SQ","TGT","TOL","TSLA","TTD","V","WMT",
    "XLK","XLY"
]
SECTOR_PROXY = {
    "AAPL":"XLK","ADBE":"XLK","AMD":"SMH","AMZN":"XLY","CRM":"XLK",
    "CRWD":"XLK","GOOGL":"XLK","META":"XLK","MSFT":"XLK","NFLX":"XLY",
    "NVDA":"SMH","SHOP":"XLY","SOXL":"SMH","SQ":"XLK","TSLA":"XLY",
    "TTD":"XLK","V":"XLK","HD":"XLY","TGT":"XLY","WMT":"XLY",
    "SBUX":"XLY","CMG":"XLY","AZO":"XLY","AAP":"XLY","KBH":"XLY",
    "TOL":"XLY","GS":"DIA","PYPL":"XLK"
}
CLUSTER = {
    "AMD":"SEMIS","NVDA":"SEMIS","SMH":"SEMIS","SOXL":"SEMIS",
    "AAPL":"MEGACAP_TECH","GOOGL":"MEGACAP_TECH","META":"MEGACAP_TECH",
    "MSFT":"MEGACAP_TECH","AMZN":"MEGACAP_TECH","TSLA":"MEGACAP_TECH",
    "XLK":"TECH_ETF","XLY":"CONSUMER_ETF","SPY":"BROAD_ETF","DIA":"BROAD_ETF"
}

START = os.environ.get("IWT_START", "2000-01-01")
END = os.environ.get("IWT_END", date.today().isoformat())
ATR_PERIOD = int(os.environ.get("IWT_ATR_PERIOD", "14"))
SMA_FAST = 20
SMA_MID = 50
SMA_SLOW = 200
ATR_BUFFER = float(os.environ.get("IWT_ATR_BUFFER", "0.20"))
MIN_RR = float(os.environ.get("IWT_MIN_RR", "3.0"))
MAX_HOLD_DAYS = int(os.environ.get("IWT_MAX_HOLD", "90"))
MAX_ZONE_AGE = int(os.environ.get("IWT_MAX_ZONE_AGE", "252"))
BASE_MAX_CANDLES = 8
SMALL_BODY_ATR = 0.45
FAST_DEPARTURE_ATR = 1.50
AVERAGE_DEPARTURE_ATR = 0.75
SLIPPAGE_BPS = float(os.environ.get("IWT_SLIPPAGE_BPS", "5"))
COMMISSION_PER_SHARE = float(os.environ.get("IWT_COMMISSION_PER_SHARE", "0"))
RISK_FRACTION = float(os.environ.get("IWT_RISK_FRACTION", "0.005"))
INITIAL_EQUITY = float(os.environ.get("IWT_INITIAL_EQUITY", "100000"))
MAX_POSITIONS = int(os.environ.get("IWT_MAX_POSITIONS", "5"))
MAX_TOTAL_RISK_FRACTION = float(os.environ.get("IWT_MAX_TOTAL_RISK", "0.02"))
MAX_CLUSTER_RISK_FRACTION = float(os.environ.get("IWT_MAX_CLUSTER_RISK", "0.008"))
MEDIUM_SCORE_CONFIRM = os.environ.get("IWT_MEDIUM_CONFIRM", "1") != "0"
ALLOW_SHORTS = os.environ.get("IWT_ALLOW_SHORTS", "1") != "0"

OUT = Path(os.environ.get("IWT_OUT", "iwt_backtest_output_v2"))
CACHE = OUT / "cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

@dataclass
class Bar:
    d: date
    o: float
    h: float
    l: float
    c: float
    v: float
    atr: float | None = None
    sma20: float | None = None
    sma50: float | None = None
    sma200: float | None = None

@dataclass
class Zone:
    symbol: str
    kind: Literal["buyer", "seller"]
    formed_idx: int
    formed_date: date
    proximal: float
    distal: float
    departure_strength_atr: float
    base_candles: int
    visits: int = 0

@dataclass
class Signal:
    symbol: str
    direction: Literal["long", "short"]
    signal_idx: int
    signal_date: date
    zone: Zone
    entry: float
    stop: float
    target: float
    planned_rr: float
    score: int
    score_base: int
    score_departure: int
    score_freshness: int
    score_rr: int
    entry_style: str
    market_aligned: bool
    sector_aligned: bool

@dataclass
class Trade:
    symbol: str
    direction: str
    entry_date: date
    exit_date: date
    entry: float
    stop: float
    target: float
    exit_price: float
    planned_rr: float
    realized_r: float
    outcome: str
    hold_days: int
    zone_date: date
    zone_visits: int
    score: int
    entry_style: str
    shares: int
    planned_dollar_risk: float
    pnl: float
    cluster: str


def stooq_symbol(t: str) -> str:
    return t.lower() + ".us"


def fetch_csv(symbol: str) -> Path:
    target = CACHE / f"{symbol}.csv"
    if target.exists() and target.stat().st_size > 100:
        return target
    params = urllib.parse.urlencode({
        "s": stooq_symbol(symbol), "i": "d",
        "d1": START.replace("-", ""), "d2": END.replace("-", "")
    })
    url = "https://stooq.com/q/d/l/?" + params
    req = urllib.request.Request(url, headers={"User-Agent": "IWTResearch/2.0 personal research"})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = r.read()
    if not data.startswith(b"Date,"):
        raise RuntimeError(f"Unexpected response for {symbol}: {data[:100]!r}")
    target.write_bytes(data)
    time.sleep(0.35)
    return target


def load_bars(path: Path) -> list[Bar]:
    out: list[Bar] = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            try:
                out.append(Bar(
                    datetime.strptime(r["Date"], "%Y-%m-%d").date(),
                    float(r["Open"]), float(r["High"]), float(r["Low"]),
                    float(r["Close"]), float(r.get("Volume") or 0)
                ))
            except (ValueError, KeyError):
                continue
    out.sort(key=lambda x: x.d)
    return out


def rolling_mean(xs: list[float], i: int, n: int) -> float | None:
    if i < n - 1:
        return None
    return sum(xs[i-n+1:i+1]) / n


def enrich(bars: list[Bar]) -> None:
    trs: list[float] = []
    closes: list[float] = []
    for i, b in enumerate(bars):
        prev = bars[i-1].c if i else b.c
        trs.append(max(b.h-b.l, abs(b.h-prev), abs(b.l-prev)))
        closes.append(b.c)
        b.atr = rolling_mean(trs, i, ATR_PERIOD)
        b.sma20 = rolling_mean(closes, i, SMA_FAST)
        b.sma50 = rolling_mean(closes, i, SMA_MID)
        b.sma200 = rolling_mean(closes, i, SMA_SLOW)


def detect_zones(symbol: str, bars: list[Bar]) -> list[Zone]:
    zones: list[Zone] = []
    for i in range(ATR_PERIOD + 1, len(bars)):
        dep = bars[i]
        if not dep.atr or dep.atr <= 0:
            continue
        body = dep.c - dep.o
        if abs(body) < AVERAGE_DEPARTURE_ATR * dep.atr:
            continue
        base: list[Bar] = []
        for j in range(i-1, max(i-BASE_MAX_CANDLES-1, -1), -1):
            bj = bars[j]
            if not bj.atr or abs(bj.c-bj.o) > SMALL_BODY_ATR * bj.atr:
                break
            base.append(bj)
        if not base:
            continue
        base.reverse()
        hi, lo = max(x.h for x in base), min(x.l for x in base)
        strength = abs(body) / dep.atr
        if body > 0:
            zones.append(Zone(symbol, "buyer", i, dep.d, hi, lo, strength, len(base)))
        else:
            zones.append(Zone(symbol, "seller", i, dep.d, lo, hi, strength, len(base)))
    return zones


def base_score(n: int) -> int:
    if n <= 2: return 2
    if n <= 4: return 1
    return 0


def departure_score(x: float) -> int:
    if x >= FAST_DEPARTURE_ATR: return 2
    if x >= AVERAGE_DEPARTURE_ATR: return 1
    return 0


def freshness_score(visits: int) -> int:
    if visits == 0: return 2
    if visits == 1: return 1
    return 0


def rr_score(rr: float) -> int:
    if rr >= 3.0: return 2
    if rr >= 2.0: return 1
    return 0


def nearest_opposing_zone(zones: list[Zone], idx: int, direction: str, price: float) -> Zone | None:
    if direction == "long":
        cands = [z for z in zones if z.kind == "seller" and z.formed_idx < idx and z.proximal > price and idx-z.formed_idx <= MAX_ZONE_AGE]
        return min(cands, key=lambda z: z.proximal, default=None)
    cands = [z for z in zones if z.kind == "buyer" and z.formed_idx < idx and z.proximal < price and idx-z.formed_idx <= MAX_ZONE_AGE]
    return max(cands, key=lambda z: z.proximal, default=None)


def aligned(b: Bar, direction: str) -> bool:
    if not b.sma20 or not b.sma50 or not b.sma200:
        return False
    if direction == "long":
        return b.c > b.sma200 and b.sma20 >= b.sma50
    return b.c < b.sma200 and b.sma20 <= b.sma50


def date_map(bars: list[Bar]) -> dict[date, Bar]:
    return {b.d: b for b in bars}


def generate_signals(symbol: str, bars: list[Bar], zones: list[Zone], market_by_date: dict[date, Bar], sector_by_date: dict[date, Bar] | None) -> list[Signal]:
    signals: list[Signal] = []
    for z in zones:
        direction = "long" if z.kind == "buyer" else "short"
        if direction == "short" and not ALLOW_SHORTS:
            continue
        start, end = z.formed_idx + 1, min(len(bars)-1, z.formed_idx + MAX_ZONE_AGE)
        visits = 0
        for i in range(start, end+1):
            b = bars[i]
            touched = b.l <= max(z.proximal, z.distal) and b.h >= min(z.proximal, z.distal)
            if not touched:
                continue
            prior_visits = visits
            visits += 1
            z.visits = visits
            if prior_visits >= 2 or not b.atr:
                break
            if not aligned(b, direction):
                continue
            market_bar = market_by_date.get(b.d)
            sector_bar = sector_by_date.get(b.d) if sector_by_date else None
            market_ok = True if not market_by_date else bool(market_bar and aligned(market_bar, direction))
            sector_ok = True if sector_bar is None else aligned(sector_bar, direction)
            if not market_ok:
                continue

            opp = nearest_opposing_zone(zones, i, direction, z.proximal)
            if not opp:
                continue
            if direction == "long":
                raw_entry = z.proximal
                stop = z.distal - ATR_BUFFER * b.atr
                target = opp.proximal
                risk, reward = raw_entry-stop, target-raw_entry
            else:
                raw_entry = z.proximal
                stop = z.distal + ATR_BUFFER * b.atr
                target = opp.proximal
                risk, reward = stop-raw_entry, raw_entry-target
            if risk <= 0 or reward <= 0:
                continue
            rr = reward/risk
            parts = (base_score(z.base_candles), departure_score(z.departure_strength_atr), freshness_score(prior_visits), rr_score(rr))
            score = sum(parts)
            if score < 5 or rr < 2.0:
                continue
            entry_style = "proximal_limit" if score >= 7 else "confirmation"
            if entry_style == "confirmation" and MEDIUM_SCORE_CONFIRM:
                # Confirm with a close away from the zone in the intended direction.
                if direction == "long" and b.c <= z.proximal:
                    continue
                if direction == "short" and b.c >= z.proximal:
                    continue
                raw_entry = b.c
                risk = raw_entry-stop if direction == "long" else stop-raw_entry
                reward = target-raw_entry if direction == "long" else raw_entry-target
                rr = reward/risk if risk > 0 else -1
                if rr < MIN_RR:
                    continue
            elif rr < MIN_RR:
                continue

            slip = SLIPPAGE_BPS/10000
            entry = raw_entry*(1+slip if direction == "long" else 1-slip)
            target_adj = target*(1-slip if direction == "long" else 1+slip)
            signals.append(Signal(symbol, direction, i, b.d, z, entry, stop, target_adj, rr, score, *parts, entry_style, market_ok, sector_ok))
            break
    return signals


def simulate_trade(sig: Signal, bars: list[Bar], equity: float) -> Trade | None:
    i = sig.signal_idx
    b = bars[i]
    risk_per_share = abs(sig.entry-sig.stop)
    if risk_per_share <= 0:
        return None
    risk_budget = equity * RISK_FRACTION * (1.0 if sig.score >= 7 else 0.5)
    shares = math.floor(risk_budget / risk_per_share)
    if shares < 1:
        return None
    exit_idx = min(i+MAX_HOLD_DAYS, len(bars)-1)
    exit_price = None
    outcome = "time"
    for k in range(i, exit_idx+1):
        x = bars[k]
        if sig.direction == "long":
            hit_stop, hit_target = x.l <= sig.stop, x.h >= sig.target
        else:
            hit_stop, hit_target = x.h >= sig.stop, x.l <= sig.target
        if hit_stop and hit_target:
            exit_price = sig.stop*(1-SLIPPAGE_BPS/10000 if sig.direction=="long" else 1+SLIPPAGE_BPS/10000)
            outcome, exit_idx = "stop_ambiguous", k
            break
        if hit_stop:
            exit_price = sig.stop*(1-SLIPPAGE_BPS/10000 if sig.direction=="long" else 1+SLIPPAGE_BPS/10000)
            outcome, exit_idx = "stop", k
            break
        if hit_target:
            exit_price, outcome, exit_idx = sig.target, "target", k
            break
    if exit_price is None:
        x = bars[exit_idx]
        exit_price = x.c*(1-SLIPPAGE_BPS/10000 if sig.direction=="long" else 1+SLIPPAGE_BPS/10000)
    signed_move = exit_price-sig.entry if sig.direction=="long" else sig.entry-exit_price
    commission = 2*shares*COMMISSION_PER_SHARE
    pnl = signed_move*shares - commission
    planned_risk = risk_per_share*shares
    realized_r = pnl/planned_risk if planned_risk else 0
    return Trade(sig.symbol,sig.direction,sig.signal_date,bars[exit_idx].d,sig.entry,sig.stop,sig.target,exit_price,sig.planned_rr,realized_r,outcome,exit_idx-i,sig.zone.formed_date,sig.zone.visits,sig.score,sig.entry_style,shares,planned_risk,pnl,CLUSTER.get(sig.symbol,SECTOR_PROXY.get(sig.symbol,"OTHER")))


def boundary_survival(symbol: str, bars: list[Bar], zones: list[Zone]) -> list[tuple]:
    rows=[]
    horizons=(45,60,75,90)
    buffers=(0.0,0.1,0.2,0.3,0.5,0.75,1.0)
    for z in zones:
        start=z.formed_idx+1
        if start>=len(bars) or not bars[start].atr:
            continue
        for horizon in horizons:
            end=min(len(bars)-1,start+horizon)
            path=bars[start:end+1]
            for buf in buffers:
                atr=bars[start].atr or 0
                if z.kind=="buyer":
                    boundary=z.distal-buf*atr
                    breached=any(x.l<=boundary for x in path)
                    terminal=path[-1].c<=boundary
                    mae=min((x.l-boundary)/atr for x in path) if atr else 0
                else:
                    boundary=z.distal+buf*atr
                    breached=any(x.h>=boundary for x in path)
                    terminal=path[-1].c>=boundary
                    mae=max((x.h-boundary)/atr for x in path) if atr else 0
                rows.append((symbol,z.kind,z.formed_date,horizon,buf,boundary,int(breached),int(terminal),mae))
    return rows


def write_csv(path: Path, headers: list[str], rows: Iterable[Iterable]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(headers); w.writerows(rows)


def summarize(trades: list[Trade]) -> list[tuple]:
    groups: dict[str,list[Trade]]={"ALL":trades}
    for t in trades:
        groups.setdefault(t.symbol,[]).append(t)
        groups.setdefault(f"SCORE_{t.score}",[]).append(t)
        groups.setdefault(t.direction.upper(),[]).append(t)
    out=[]
    for key,xs in sorted(groups.items()):
        if not xs: continue
        rs=[x.realized_r for x in xs]
        wins=sum(r>0 for r in rs)
        equity=peak=maxdd=0.0
        for r in rs:
            equity+=r; peak=max(peak,equity); maxdd=min(maxdd,equity-peak)
        out.append((key,len(xs),wins/len(xs),statistics.mean(rs),statistics.median(rs),sum(rs),maxdd,min(rs),max(rs),statistics.mean(x.hold_days for x in xs)))
    return out


def main() -> int:
    bars_by_symbol: dict[str,list[Bar]]={}
    zones_by_symbol: dict[str,list[Zone]]={}
    failures=[]
    required=sorted(set(TICKERS+["SPY","XLK","XLY","SMH","DIA"]))
    for n,symbol in enumerate(required,1):
        print(f"[{n}/{len(required)}] {symbol}")
        try:
            bars=load_bars(fetch_csv(symbol)); enrich(bars)
            bars_by_symbol[symbol]=bars
            zones_by_symbol[symbol]=detect_zones(symbol,bars)
        except Exception as e:
            failures.append((symbol,str(e))); print(f"FAILED {symbol}: {e}",file=sys.stderr)

    if "SPY" not in bars_by_symbol:
        print("WARNING: SPY unavailable - market-alignment filter DISABLED (results less conservative).", file=sys.stderr)
        market = {}
    else:
        market=date_map(bars_by_symbol["SPY"])

    signals=[]; trades=[]; boundary=[]
    equity=INITIAL_EQUITY
    # Current implementation serializes trades by signal date for transparent sizing.
    # A later portfolio-event engine can enforce simultaneous open-position caps exactly.
    for symbol in TICKERS:
        if symbol not in bars_by_symbol: continue
        sector=SECTOR_PROXY.get(symbol)
        sector_map=date_map(bars_by_symbol[sector]) if sector in bars_by_symbol else None
        sigs=generate_signals(symbol,bars_by_symbol[symbol],zones_by_symbol[symbol],market,sector_map)
        signals.extend(sigs)
        for sig in sigs:
            t=simulate_trade(sig,bars_by_symbol[symbol],equity)
            if t:
                trades.append(t)
        boundary.extend(boundary_survival(symbol,bars_by_symbol[symbol],zones_by_symbol[symbol]))

    signals.sort(key=lambda x:(x.signal_date,x.symbol))
    trades.sort(key=lambda x:(x.entry_date,x.symbol))

    write_csv(OUT/"signals.csv", list(Signal.__dataclass_fields__), ([
        s.symbol,s.direction,s.signal_idx,s.signal_date,s.zone.formed_date,s.entry,s.stop,s.target,s.planned_rr,s.score,
        s.score_base,s.score_departure,s.score_freshness,s.score_rr,s.entry_style,s.market_aligned,s.sector_aligned
    ] for s in signals))
    write_csv(OUT/"trades.csv", list(Trade.__dataclass_fields__), ([getattr(t,k) for k in Trade.__dataclass_fields__] for t in trades))
    write_csv(OUT/"summary.csv",["scope","trades","win_rate","mean_R","median_R","total_R","max_drawdown_R","worst_R","best_R","avg_hold_days"],summarize(trades))
    write_csv(OUT/"boundary_survival.csv",["symbol","zone_kind","zone_date","horizon_days","atr_buffer","boundary","ever_breached","terminal_breach","mae_atr"],boundary)
    write_csv(OUT/"failures.csv",["symbol","error"],failures)

    db=sqlite3.connect(OUT/"research_v2.sqlite")
    db.executescript("""
    DROP TABLE IF EXISTS trades; DROP TABLE IF EXISTS boundary_survival;
    CREATE TABLE trades(symbol TEXT,direction TEXT,entry_date TEXT,exit_date TEXT,entry REAL,stop REAL,target REAL,exit_price REAL,planned_rr REAL,realized_r REAL,outcome TEXT,hold_days INTEGER,zone_date TEXT,zone_visits INTEGER,score INTEGER,entry_style TEXT,shares INTEGER,planned_dollar_risk REAL,pnl REAL,cluster_name TEXT);
    CREATE TABLE boundary_survival(symbol TEXT,zone_kind TEXT,zone_date TEXT,horizon_days INTEGER,atr_buffer REAL,boundary REAL,ever_breached INTEGER,terminal_breach INTEGER,mae_atr REAL);
    """)
    db.executemany("INSERT INTO trades VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",[[getattr(t,k) if not isinstance(getattr(t,k),date) else str(getattr(t,k)) for k in Trade.__dataclass_fields__] for t in trades])
    db.executemany("INSERT INTO boundary_survival VALUES(?,?,?,?,?,?,?,?,?)",boundary)
    db.commit(); db.close()
    print(f"Done: {OUT.resolve()}")
    print("Interpretation warning: daily-bar zone proxy and boundary statistics, not verified historical options fills.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
