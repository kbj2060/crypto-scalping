#!/usr/bin/env python3
"""**CBOE 지연 옵션 체인 -> 대시보드 Option 카드 요약** (2026-11-01, SNDK).

Deribit 요약기(collect_deribit_option_gex_20260815.options_summary)와 **같은 모양**의 dict 를 만든다 -- 그래야 app.js 의 Option 카드가
그대로 읽는다(예상 폭·사다리·만기 막대·기간 구조·RR/BF). 계산 도구(_skew_interp·_const_maturity·_surface_stats·_atm_spread_iv)는 Deribit 쪽을 가져다 쓴다.
차이: 계약 승수 100주 · 선도가 = 현물(금리·배당 0, 상한 있는 근사) · 만기는 미 동부 16:00 · 실현변동성 rv7 은 없음(None -> 화면 «-») ·
기준 시각 = 마지막 체결 시각(장 마감 후엔 «N시간 전 값»으로 정직하게 보인다). 신호 아님.
쓰기: data/live/cboe_option_state.json (대시보드가 /api/gex 에 합쳐 싣는다).   python scripts/cboe_option_summary_20261101.py [SNDK] | --selftest
"""
from __future__ import annotations

import gzip
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_deribit_option_gex_20260815 as D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CHAIN_DIR = ROOT / "data" / "live" / "options_equity" / "cboe"
STATE = ROOT / "data" / "live" / "cboe_option_state.json"
ET = ZoneInfo("America/New_York")
MULT = 100                       # 계약당 주식 수
OPT_RE = re.compile(r"^[A-Z.]+(\d{6})([CP])(\d{8})$")


def chain_frame(options: list[dict], now: float) -> pd.DataFrame:
    rows = []
    for o in options:
        m = OPT_RE.match(o["option"])
        if not m:
            continue
        exp = datetime.strptime(m[1], "%y%m%d").replace(hour=16, tzinfo=ET)
        days = (exp.timestamp() - now) / 86400
        if days <= 0:
            continue
        rows.append({"expiration_ts": pd.Timestamp(exp).tz_convert("UTC"), "option_type": "call" if m[2] == "C" else "put",
                     "strike": int(m[3]) / 1000, "mark_iv": float(o["iv"]) * 100, "bid_price": float(o["bid"]),
                     "ask_price": float(o["ask"]), "open_interest": float(o["open_interest"]), "days_to_expiry": days})
    return pd.DataFrame(rows)


def summarize(data: dict, now: float | None = None) -> dict:
    now = now if now is not None else time.time()
    S = float(data["current_price"])
    ch = chain_frame(data["options"], now)
    last = datetime.strptime(data["last_trade_time"], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=ET)
    out: dict = {"recorded_at_utc": last.astimezone(timezone.utc).isoformat(), "currency": data["symbol"], "src": "CBOE 지연",
                 "dvol": None, "index": S, "rv7": None, "iv30": float(data["iv30"]) or None, "hedge": []}
    exps = []
    for exp, g in sorted(ch.groupby("expiration_ts"), key=lambda kv: kv[0]):
        gi = g[g["mark_iv"] > 0]                                  # IV 계산용 -- 호가 없는 깊은 내가격은 iv 0
        ci, pi = gi[gi["option_type"] == "call"], gi[gi["option_type"] == "put"]
        if ci.empty or pi.empty:
            continue
        yrs = float(g["days_to_expiry"].iloc[0]) / 365.0
        calls, puts = g[g["option_type"] == "call"], g[g["option_type"] == "put"]
        atm_k = float(gi.iloc[(gi["strike"] - S).abs().argsort()[:1]]["strike"].iloc[0])
        coi, poi = float(calls["open_interest"].sum()), float(puts["open_interest"].sum())
        pain = min(sorted(g["strike"].unique()), key=lambda P: float(((P - calls["strike"]).clip(lower=0) * calls["open_interest"]).sum()
                                                                     + ((puts["strike"] - P).clip(lower=0) * puts["open_interest"]).sum()))
        usd = MULT * S
        exps.append({"exp_ms": int(exp.timestamp() * 1000), "atm_iv": float(gi[gi["strike"] == atm_k]["mark_iv"].mean()),
                     "call_oi_usd": coi * usd, "put_oi_usd": poi * usd, "pc": (poi / coi) if coi else None, "pain": float(pain),
                     **D._skew_interp(gi, S, yrs), "fwd": S,       # ponytail: 선도가 = 현물(금리·배당 0) -- 장기 만기에서 어긋난다, 필요하면 put-call parity 로 역산
                     "atm_oi_usd": float(g.loc[(g["strike"] / S - 1).abs() <= 0.02, "open_interest"].sum()) * usd, "_g": gi, "_yrs": yrs})
    tot = sum(e["call_oi_usd"] + e["put_oi_usd"] for e in exps)
    for e in exps:
        e["oi_share"] = (e["call_oi_usd"] + e["put_oi_usd"]) / tot if tot else None
    live = [e for e in exps if e["hours"] >= 3]
    near = lambda d: min(live, key=lambda e: abs(e["hours"] - d * 24), default=None)   # noqa: E731
    pick = {"front": live[0] if live else None, "7": near(7), "30": near(30)}
    out["surface"] = {k: ({"exp_ms": e["exp_ms"], "hours": round(e["hours"], 2), "fwd": S, **D._surface_stats(e["_g"], S, e["_yrs"])}
                          if e else None) for k, e in pick.items()}
    out["atm_spread_iv"] = D._atm_spread_iv(pick["7"]["_g"], S, pick["7"]["_yrs"], True) if pick["7"] else None
    out["front_atm_oi_usd"] = exps[0]["atm_oi_usd"] if exps else None
    for e in exps:
        e.pop("_g"); e.pop("_yrs")
    out["expiries"] = exps[:8]
    out["cm"] = {str(d): D._const_maturity(exps, d) for d in (1, 7, 30, 60)}
    band = ch[(ch["strike"] >= S * 0.92) & (ch["strike"] <= S * 1.08)]
    usd_oi = band["open_interest"] * MULT * S
    band = band.assign(c=usd_oi.where(band["option_type"] == "call", 0.0), p=usd_oi.where(band["option_type"] == "put", 0.0))
    first = ch["expiration_ts"].min() if len(ch) else None

    def ladder(sel):
        agg = band[sel].groupby("strike")[["c", "p"]].sum().reset_index()
        return [[float(r.strike), round(float(r.c)), round(float(r.p))] for r in agg.itertuples()]
    out["strikes"] = {"front": ladder(band["expiration_ts"] == first) if first is not None else [], "week": ladder(band["days_to_expiry"] <= 7),
                      "all": ladder(band["days_to_expiry"] > 0), "front_exp_ms": int(first.timestamp() * 1000) if first is not None else None}
    return out


def _clean(x):
    """JSON 은 NaN 을 못 읽는다(브라우저 파싱 실패) -- 유한하지 않은 수는 None."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_clean(v) for v in x]
    return None if isinstance(x, float) and x != x or x in (float("inf"), float("-inf")) else x


def write_state(snapshot: Path, state: Path = STATE) -> None:
    """종목 하나의 요약을 상태 파일에 **합친다** -- 다른 종목 항목은 그대로 둔다(예전엔 파일을 통째로 써서 둘째 종목이 첫째를 덮었다)."""
    data = json.loads(gzip.decompress(snapshot.read_bytes()))["data"]
    opt = _clean(summarize(data))
    try:
        cur = json.loads(state.read_text(encoding="utf-8"))["currencies"]
    except (OSError, ValueError, KeyError):
        cur = {}                  # 파일이 없거나 깨졌으면 이 종목부터 다시 쌓는다
    cur[data["symbol"]] = {"opt_hist": {}, "options": opt, "recorded_at_utc": opt["recorded_at_utc"], "spot_price": opt["index"]}
    body = {"generated_at": datetime.now(timezone.utc).isoformat(), "currencies": cur}
    tmp = state.with_name(f"{state.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(body, ensure_ascii=False), encoding="utf-8")
    tmp.replace(state)


def latest_snapshot(sym: str) -> Path:
    return max((CHAIN_DIR / sym).glob("*/*.json.gz"), key=lambda p: (p.parent.name, p.name))


def _selftest() -> None:
    now = datetime(2026, 10, 30, 20, 0, tzinfo=timezone.utc).timestamp()
    opt = lambda e, t, k, iv, oi: {"option": f"SNDK{e}{t}{int(k * 1000):08d}", "iv": iv, "bid": 1.0, "ask": 1.2, "open_interest": oi}   # noqa: E731
    ops = [opt(e, t, k, 0.7 + 0.001 * abs(k - 100), 10) for e in ("261106", "261120", "261218") for t in "CP" for k in range(40, 200, 5)]
    out = summarize({"symbol": "SNDK", "current_price": 100.0, "iv30": 70.0, "last_trade_time": "2026-10-30T15:59:59", "options": ops}, now)
    assert out["src"] == "CBOE 지연" and out["index"] == 100.0 and len(out["expiries"]) == 3
    e0 = out["expiries"][0]
    assert e0["call_oi_usd"] == 10 * 32 * 100 * 100.0 and e0["pc"] == 1.0 and abs(e0["atm_iv"] - 70.0) < 1e-6, e0
    assert out["strikes"]["front"] and out["strikes"]["all"] and out["cm"]["30"] is not None
    assert out["recorded_at_utc"].startswith("2026-10-30T19:59:59"), out["recorded_at_utc"]   # EDT 15:59:59 = 19:59:59Z
    assert _clean({"a": [float("nan"), 1.0]}) == {"a": [None, 1.0]}
    import tempfile
    from gzip import compress
    with tempfile.TemporaryDirectory() as td:       # 두 종목을 차례로 써도 서로 덮지 않는다
        st = Path(td) / "s.json"
        for sym in ("SNDK", "SOXL"):
            snap = Path(td) / f"{sym}.json.gz"
            snap.write_bytes(compress(json.dumps({"data": {"symbol": sym, "current_price": 100.0, "iv30": 70.0, "last_trade_time": "2026-10-30T15:59:59", "options": ops}}).encode()))
            write_state(snap, st)
        assert set(json.loads(st.read_text())["currencies"]) == {"SNDK", "SOXL"}
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        sym = (sys.argv[1:] or ["SNDK"])[0].upper()
        write_state(latest_snapshot(sym))
        print("state ->", STATE)
