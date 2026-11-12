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
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
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


def flow_today(options: list[dict]) -> dict:
    """오늘 체결 요약 -- CBOE 는 체결 한 건씩은 안 주고 계약별 «오늘 거래량 · 마지막 체결가»만 준다(블록 거래 대신 이걸 보인다).
    명목 = 거래량 × 마지막 체결가(없으면 호가 중간) × 100 -- 근사다(체결가가 하루 중 달라졌다). 방향(매수/매도)은 모른다."""
    rows, tot = [], {"C": [0.0, 0.0], "P": [0.0, 0.0]}
    for o in options:
        m, vol = OPT_RE.match(o["option"]), float(o.get("volume") or 0)
        if not m or vol <= 0:
            continue
        px = float(o.get("last_trade_price") or 0) or (float(o["bid"]) + float(o["ask"])) / 2
        usd = vol * px * MULT
        tot[m[2]][0] += vol
        tot[m[2]][1] += usd
        exp = datetime.strptime(m[1], "%y%m%d").replace(hour=16, tzinfo=ET)
        t = o.get("last_trade_time")
        rows.append([int(datetime.strptime(t, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=ET).timestamp() * 1000) if t else None, round(usd),
                     int(exp.timestamp() * 1000), m[2], int(m[3]) / 1000, vol, px])
    rows.sort(key=lambda r: -r[1])
    return {"call": [round(tot["C"][0]), round(tot["C"][1])], "put": [round(tot["P"][0]), round(tot["P"][1])], "top": rows[:6]}


def gex_levels(options: list[dict], S: float, now: float) -> dict:
    """2026-11-13 사용자 «saveticker(Unusual Whales) 옵션 칸을 CBOE 로» -- 감마 노출 주요 레벨. 행사가별 순 GEX($/1% 움직임) =
    Σ 감마 × 미결제 × 100 × S² × 0.01, 콜 +·풋 −(딜러가 콜은 사고 풋은 판다는 흔한 관례 -- 실제 딜러 포지션은 모른다). 전 만기 · 순 합계는 전 행사가 · 레벨 목록은 지금가 ±30%.
    돌려주기 = {net: 합계, pos: 큰 + 행사가 3개, neg: 큰 − 행사가 3개} -- 각 [행사가, $]. 서술용."""
    by: dict[float, float] = {}
    for o in options:
        m = OPT_RE.match(o["option"])
        if not m or datetime.strptime(m[1], "%y%m%d").replace(hour=16, tzinfo=ET).timestamp() <= now:
            continue
        k = int(m[3]) / 1000
        g = float(o.get("gamma") or 0) * float(o.get("open_interest") or 0) * MULT * S * S * 0.01
        by[k] = by.get(k, 0.0) + (g if m[2] == "C" else -g)
    # 순 합계는 전 행사가(감마 국면 gamma_flip 과 같은 계약 집합 -- ±30% 로 자르면 MU 처럼 먼 행사가 콜 몫이 빠져 부호가 뒤집힘) · 레벨 목록만 ±30%
    lv = sorted(((k, v) for k, v in by.items() if abs(k / S - 1) <= 0.30), key=lambda kv: -abs(kv[1]))
    return {"net": round(sum(by.values())), "pos": [[k, round(v)] for k, v in lv if v > 0][:3], "neg": [[k, round(v)] for k, v in lv if v < 0][:3]}


def gamma_flip(options: list[dict], S: float, now: float, span: float = 0.30, n: int = 241) -> dict | None:
    """2026-11-13 사용자 «주식에서 롱감마·숏감마·감마 플립 데이터»: 가격을 지금가 ±30% 로 옮겨 가며 계약마다 블랙-숄즈 감마를 다시 계산
    (그 계약의 CBOE IV · 남은 기간 · 금리·배당 0) → 순 GEX(S') = Σ 부호 × 감마 × 미결제 × 100 × S'² × 0.01 (gex_levels 와 같은 관례: 콜 +·풋 −).
    국면 = 지금가의 순 GEX 부호(+ 롱감마 = 딜러 헤지가 움직임을 누름 · − 숏감마 = 키움) · 플립 = 순 GEX 가 0 을 지나는 가격 중 지금가에 가장 가까운 곳.
    🔴관례 가정(딜러가 콜 매수·풋 매도) -- 실제 딜러 포지션은 모른다 · 검증 안 됨(서술용). IV 0(호가 없는 깊은 내가격)·미결제 0 계약은 뺀다(감마 ≈ 0).
    돌려주기 = {net, regime: long|short, flip, flip_pct, n} 또는 None(계약 없음)."""
    rows = []
    for o in options:
        m = OPT_RE.match(o["option"])
        if not m:
            continue
        t = (datetime.strptime(m[1], "%y%m%d").replace(hour=16, tzinfo=ET).timestamp() - now) / (365 * 86400)
        iv, oi = float(o.get("iv") or 0), float(o.get("open_interest") or 0)
        if t > 0 and iv > 0 and oi > 0:
            rows.append((int(m[3]) / 1000, iv, t, oi, 1.0 if m[2] == "C" else -1.0))
    if not rows:
        return None
    K, iv, T, oi, sg = (np.array(c) for c in zip(*rows))

    def net(x):   # x = 가격 배열 → 순 GEX 배열
        x = np.asarray(x, float)[:, None]
        sd = iv * np.sqrt(T)
        d1 = (np.log(x / K) + 0.5 * sd * sd) / sd
        g = np.exp(-0.5 * d1 * d1) / (np.sqrt(2 * np.pi) * x * sd)
        return (sg * oi * g).sum(1) * MULT * x[:, 0] ** 2 * 0.01
    grid = S * np.linspace(1 - span, 1 + span, n)
    v = net(grid)
    now_v = float(net([S])[0])
    cross = np.nonzero(np.sign(v[:-1]) * np.sign(v[1:]) < 0)[0]
    flips = [grid[i] - v[i] * (grid[i + 1] - grid[i]) / (v[i + 1] - v[i]) for i in cross]   # 선형 보간
    flip = min(flips, key=lambda p: abs(p - S)) if flips else None
    return {"net": round(now_v), "regime": "long" if now_v > 0 else "short", "flip": round(float(flip), 2) if flip else None,
            "flip_pct": round((float(flip) / S - 1) * 100, 2) if flip else None, "n": len(rows)}


def volume_vs_normal(sym_dir: Path, now_et: datetime, today_n: float, days: int = 10) -> dict | None:
    """오늘 지금까지 옵션 거래량(전 만기, 계약) ÷ 지난 거래일들의 «같은 시각까지» 거래량 평균. 저장된 10분 스냅샷이 곧 이력이다
    (CBOE 는 하루 이력을 안 준다). 날마다 같은 ET 시각 이전의 마지막 스냅샷 하나를 읽는다. 지난날이 없으면 None.
    ponytail: 스냅샷을 매번 다시 읽는다(최대 10개 × ~0.7MB) -- 느려지면 날짜·시각별 합계를 캐시."""
    clock, pick = now_et.hour * 60 + now_et.minute, {}
    for p in sym_dir.glob("*/*.json.gz"):
        try:
            t = datetime.strptime(p.parent.name + p.name[:6], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).astimezone(ET)
        except ValueError:
            continue
        if t.date() >= now_et.date() or t.hour * 60 + t.minute > clock:
            continue
        if t.date() not in pick or t > pick[t.date()][0]:
            pick[t.date()] = (t, p)
    vols = []
    for d in sorted(pick)[-days:]:
        try:
            ops = json.loads(gzip.decompress(pick[d][1].read_bytes()))["data"]["options"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        vols.append(sum(float(o.get("volume") or 0) for o in ops))
    vols = [v for v in vols if v > 0]
    if not vols:
        return None
    avg = sum(vols) / len(vols)
    return {"today": round(today_n), "avg": round(avg), "ratio": round(today_n / avg, 3), "days": len(vols)}


RV_CACHE = ROOT / "data" / "live" / "cboe_rv7_cache.json"


def realized_vol_7d(sym: str, cache: Path = RV_CACHE, now: float | None = None) -> float | None:
    """실현 7일 변동성(%) = 바이낸스 TradFi 선물 1시간봉 수익 제곱평균 × √(24·365). 주식 정규장이 아니라 선물(주말 포함) 기준이다.
    6시간 재사용 · 실패하면 24시간 안의 직전 값 · 그래도 없으면 None(화면 «-»). 서버 cron 에서만 부른다(로컬 바이낸스 REST 금지).
    ponytail: 정규장만의 실현변동성이 아니다 -- 필요하면 CBOE 일봉으로 바꾼다."""
    now = now if now is not None else time.time()
    try:
        hit = json.loads(cache.read_text(encoding="utf-8")).get(sym)
    except (OSError, ValueError):
        hit = None
    if hit and now - hit[0] < 6 * 3600:
        return hit[1]
    try:
        url = f"https://fapi.binance.com/fapi/v1/klines?symbol={sym}USDT&interval=1h&limit=169"
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "crypto-scalping-collector/1.0"}), timeout=20) as r:
            cl = [float(k[4]) for k in json.loads(r.read())]
        rets = [b / a - 1 for a, b in zip(cl, cl[1:])]      # ponytail: 단순수익 ≈ 로그수익(1시간 폭에서 차이 무시)
        v = (sum(x * x for x in rets) / len(rets) * 24 * 365) ** 0.5 * 100 if len(rets) > 24 else None
    except Exception:  # noqa: BLE001 -- 부가 값: 실패해도 요약은 계속
        return hit[1] if hit and now - hit[0] < 24 * 3600 else None
    if v is not None:
        try:
            cur = json.loads(cache.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            cur = {}
        cur[sym] = [now, v]
        cache.write_text(json.dumps(cur), encoding="utf-8")
    return v


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
    out["flow_today"] = flow_today(data["options"])
    out["gex"] = gex_levels(data["options"], S, now)
    out["gamma"] = gamma_flip(data["options"], S, now)   # 2026-11-13 롱/숏 감마 국면 · 감마 플립
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


def write_state(snapshot: Path, state: Path = STATE, rv7: bool = True) -> None:
    """종목 하나의 요약을 상태 파일에 **합친다** -- 다른 종목 항목은 그대로 둔다(예전엔 파일을 통째로 써서 둘째 종목이 첫째를 덮었다)."""
    data = json.loads(gzip.decompress(snapshot.read_bytes()))["data"]
    opt = summarize(data)
    if rv7:
        opt["rv7"] = realized_vol_7d(data["symbol"])      # 2026-11-03 «IV−실현» 칸 -- 전엔 늘 «-» 였다
    ft0 = opt["flow_today"]   # 2026-11-13 «평소 대비 거래량» -- 스냅샷이 쌓인 폴더(<종목>/<UTC날짜>/<시각>.json.gz)의 지난 거래일 같은 시각
    # 검증(서브 에이전트): 시계는 **수집 시각**(파일 이름)으로 -- 지난날도 수집 시각으로 고르는데 오늘만 데이터 시각(15분 지연·장 뒤 15:59:59 고정)을 쓰면
    #   장 마감 뒤 지난날 마지막 15분이 빠져 비율이 부풀었다(1.0 → 1.11). 파일 이름이 시각이 아니면(시험) 데이터 시각.
    try:
        clock = datetime.strptime(snapshot.parent.name + snapshot.name[:6], "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).astimezone(ET)
    except ValueError:
        clock = datetime.strptime(data["last_trade_time"], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=ET)
    opt["vol_norm"] = volume_vs_normal(snapshot.parent.parent, clock, ft0["call"][0] + ft0["put"][0])
    opt = _clean(opt)
    try:
        cur = json.loads(state.read_text(encoding="utf-8"))["currencies"]
    except (OSError, ValueError, KeyError):
        cur = {}                  # 파일이 없거나 깨졌으면 이 종목부터 다시 쌓는다
    # 오늘 체결 누적 곡선 -- 10분마다 «지금까지 콜·풋 명목»을 한 점씩 쌓는다(CBOE 는 이력을 안 준다 -- 이 점들이 곧 이력이다). 24시간만 · 하루 거래량은 장 시작에 0 으로 돌아간다.
    prev = ((cur.get(data["symbol"]) or {}).get("options") or {}).get("flow_series") or []
    ft, tm = opt["flow_today"], int(datetime.strptime(data["last_trade_time"], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=ET).timestamp() * 1000)
    pts = [p for p in prev if p[0] > tm - 24 * 3600e3 and p[0] != tm] + [[tm, ft["call"][1], ft["put"][1]]]
    opt["flow_series"] = sorted(pts)
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
    assert out["flow_today"]["call"] == [0, 0] and out["flow_today"]["top"] == []      # 거래량 0 이면 비어 있다
    ops2 = [dict(o, volume=(5.0 if o["option"].endswith("C00100000") else 2.0 if o["option"].endswith("P00090000") else 0.0),
                 last_trade_price=3.0, last_trade_time="2026-10-30T15:00:00") for o in ops]
    ft = summarize({"symbol": "SNDK", "current_price": 100.0, "iv30": 70.0, "last_trade_time": "2026-10-30T15:59:59", "options": ops2}, now)["flow_today"]
    assert ft["call"][0] == 15 and ft["put"][0] == 6 and ft["top"][0][1] == 5 * 3.0 * 100 and ft["top"][0][3] == "C", ft   # 만기 셋 × 5 · 셋 × 2
    # 감마 플립: 콜 110 · 풋 90 같은 미결제 → 위쪽 + · 아래쪽 − → 0 을 지나는 가격이 둘 사이 · 콜만 있으면 플립 없음·롱감마 · 풋만 → 숏감마
    gf = gamma_flip([{"option": "SNDK261120C00110000", "iv": 0.5, "open_interest": 100}, {"option": "SNDK261120P00090000", "iv": 0.5, "open_interest": 100}], 100.0, now)
    assert gf and 90 < gf["flip"] < 110 and gf["n"] == 2, gf
    assert gamma_flip([{"option": "SNDK261120C00110000", "iv": 0.5, "open_interest": 100}], 112.0, now)["regime"] == "long"
    g2 = gamma_flip([{"option": "SNDK261120P00090000", "iv": 0.5, "open_interest": 100}], 95.0, now)
    assert g2["regime"] == "short" and g2["flip"] is None, g2
    assert gamma_flip([{"option": "SNDK261023C00110000", "iv": 0.5, "open_interest": 100}], 100.0, now) is None   # 지난 만기만 = 계약 없음
    gx = gex_levels([{"option": "SNDK261106C00100000", "gamma": 0.05, "open_interest": 10}, {"option": "SNDK261106P00090000", "gamma": 0.02, "open_interest": 30},
                     {"option": "SNDK261023C00100000", "gamma": 1.0, "open_interest": 999}], 100.0, now)   # 지난 만기(10-23)는 뺀다
    assert gx["pos"] == [[100.0, 5000]] and gx["neg"] == [[90.0, -6000]] and gx["net"] == -1000, gx   # 0.05×10×100×100²×0.01 = 5000
    gx2 = gex_levels([{"option": "SNDK261106P00090000", "gamma": 0.02, "open_interest": 30}, {"option": "SNDK261106C00200000", "gamma": 0.02, "open_interest": 40}], 100.0, now)
    assert gx2["net"] == 2000 and gx2["pos"] == [], gx2   # 먼 행사가(200)는 순 합계엔 들고 레벨 목록엔 안 든다
    import tempfile as _t
    with _t.TemporaryDirectory() as td:       # 평소 대비: 지난 거래일의 같은 시각 이전 마지막 스냅샷만 · 오늘·더 늦은 시각은 안 센다
        sd = Path(td)
        for day, hhmmss, vol in (("20261028", "150000", 50.0), ("20261028", "193000", 999.0), ("20261029", "145000", 150.0), ("20261030", "140000", 7.0)):
            (sd / day).mkdir(exist_ok=True)
            (sd / day / f"{hhmmss}.json.gz").write_bytes(gzip.compress(json.dumps({"data": {"options": [{"volume": vol}]}}).encode()))
        vn = volume_vs_normal(sd, datetime(2026, 10, 30, 11, 30, tzinfo=ET), 200.0)   # ET 11:30 = UTC 15:30 → 10-28 15:00Z(50) · 10-29 14:50Z(150) · 10-30 은 오늘
        assert vn == {"today": 200, "avg": 100, "ratio": 2.0, "days": 2}, vn
    with _t.TemporaryDirectory() as td:       # 6시간 안의 캐시는 네트워크 없이 그대로
        c = Path(td) / "rv.json"
        c.write_text(json.dumps({"SNDK": [now, 77.7]}))
        assert realized_vol_7d("SNDK", c, now + 3600) == 77.7
    import tempfile
    from gzip import compress
    with tempfile.TemporaryDirectory() as td:       # 두 종목을 차례로 써도 서로 덮지 않는다
        st = Path(td) / "s.json"
        for sym in ("SNDK", "SOXL"):
            (Path(td) / sym / "20261030").mkdir(parents=True, exist_ok=True)
            snap = Path(td) / sym / "20261030" / "195959.json.gz"   # 수집 시각 꼴(<종목>/<UTC날짜>/<시각>) -- 평소 대비가 시험 폴더 밖(/tmp)을 훑지 않게
            snap.write_bytes(compress(json.dumps({"data": {"symbol": sym, "current_price": 100.0, "iv30": 70.0, "last_trade_time": "2026-10-30T15:59:59", "options": ops}}).encode()))
            write_state(snap, st, rv7=False)
        assert set(json.loads(st.read_text())["currencies"]) == {"SNDK", "SOXL"}
        write_state(snap, st, rv7=False)         # 같은 스냅샷을 또 써도 점이 겹쳐 쌓이지 않는다
        assert len(json.loads(st.read_text())["currencies"]["SOXL"]["options"]["flow_series"]) == 1
    with tempfile.TemporaryDirectory() as td:       # 2026-11-13 장 마감 뒤: 오늘·지난날 모두 수집 시각으로 -- 지난날 16:20(ET) 최종 21 을 골라 비율 1.0
        sd, st = Path(td) / "SNDK", Path(td) / "s.json"
        for hhmmss, vol in (("195000", 19.0), ("202000", 21.0)):   # 10-29 15:50·16:20 ET(EDT)
            (sd / "20261029").mkdir(parents=True, exist_ok=True)
            (sd / "20261029" / f"{hhmmss}.json.gz").write_bytes(compress(json.dumps({"data": {"options": [{"volume": vol}]}}).encode()))
        (sd / "20261030").mkdir(parents=True, exist_ok=True)
        snap = sd / "20261030" / "203000.json.gz"                  # 오늘 16:30 ET 수집 · 데이터 시각은 15:59:59 에 멈춤
        snap.write_bytes(compress(json.dumps({"data": {"symbol": "SNDK", "current_price": 100.0, "iv30": 70.0, "last_trade_time": "2026-10-30T15:59:59", "options": ops2}}).encode()))
        write_state(snap, st, rv7=False)
        vn = json.loads(st.read_text())["currencies"]["SNDK"]["options"]["vol_norm"]
        assert vn["today"] == 21 and vn["ratio"] == 1.0, vn
    print("selftest OK")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        _selftest()
    else:
        sym = (sys.argv[1:] or ["SNDK"])[0].upper()
        write_state(latest_snapshot(sym))
        print("state ->", STATE)
