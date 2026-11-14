#!/usr/bin/env python3
"""볼륨 프로파일이 «뜻하는 대로» 움직이는가 (2026-11-15). 사전등록 docs/prereg/volume_profile_meaning_20261115.md (3f6ec0d9, 계산 전).

  selftest   손계산 1분봉으로 프로파일·V1~V5 사건 단언
  run        4코인 실제 + 기하 위약 20벌 → tmp/volume_profile_meaning_20261115/{events.parquet,placebo_agg.parquet,result.json}
가치영역·노드 식은 dashboard/market_ctx.py 그대로, 위약·경주·판정은 research_smc_ict_meaning_20261115 그대로 쓴다.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "dashboard"))
import research_smc_ict_meaning_20261115 as SM  # noqa: E402
from market_ctx import PROFILE_BIN_FRAC, value_area  # noqa: E402

OUT = ROOT / "tmp/volume_profile_meaning_20261115"
END = int(pd.Timestamp("2026-10-30", tz="UTC").value // 10**6)      # 거래량 열이 있는 마지막 날 다음
DAY = 86_400_000
CELLS = ["V1", "V2", "V3", "V4", "V5"]


def profiles(k: pd.DataFrame) -> dict:
    """UTC 일 → (bw, k0, hist, VA dict, 그날 저가, 고가). 대표가 (h+l+c)/3 에 그 분 거래량(09-30 규약)."""
    tp = ((k.h + k.l + k.c) / 3).to_numpy(); v = k.v.to_numpy(); day = k.t.to_numpy() // DAY
    out = {}
    cut = np.flatnonzero(np.diff(day)) + 1
    for a, b in zip(np.r_[0, cut], np.r_[cut, len(day)]):
        if b - a < 1380 or not np.isfinite(v[a:b]).all():                # 거의 다 찬 날만(1,440분 중 1,380+)
            continue
        p, w = tp[a:b], v[a:b]
        bw = float(np.median(p)) * PROFILE_BIN_FRAC
        kk = np.floor(p / bw).astype(np.int64)
        va = value_area(kk, w, bw)
        k0 = int(kk.min()); hist = np.bincount(kk - k0, weights=w)
        out[int(day[a])] = (bw, k0, hist, va, float(k.l.iloc[a:b].min()), float(k.h.iloc[a:b].max()))
    return out


def events(k: pd.DataFrame, rng, with_p2: bool) -> tuple[list[dict], list[tuple]]:
    P = profiles(k)
    B = SM.bars(k, 15); B30 = SM.bars(k, 30)
    h, l, c, atr, t = B["h"], B["l"], B["c"], B["atr"], B["t"]
    bday = t // DAY; cut = np.flatnonzero(np.diff(bday)) + 1
    rng_d = {int(bday[a]): (a, b) for a, b in zip(np.r_[0, cut], np.r_[cut, len(t)]) if b - a == 96}
    d30 = B30["t"] // DAY
    R, V5 = [], []
    add = lambda cell, tk, y: R.append(dict(cell=cell, tk=int(tk), y=float(y)))
    days = sorted(rng_d)
    for d in days:
        if d - 1 not in P or d - 1 not in rng_d:
            continue
        bw, k0, hist, va, plo, phi = P[d - 1]
        a, b = rng_d[d]; op = c[a - 1]; lo, hi = l[a:b].min(), h[a:b].max(); tk = t[a]
        poc, vah, val = va["poc"], va["vah"], va["val"]
        # V1 전일 POC 끌림 (+ 거울)
        if abs(op - poc) >= bw:
            add("V1", tk, lo <= poc <= hi)
            if with_p2:
                m = 2 * op - poc; add("P2V1", tk, lo <= m <= hi)
        # V2 VAH·VAL 첫 터치 되밀림 (+ ±1 ATR 옮긴 수준)
        levs = [(vah, "V2"), (val, "V2")]
        if with_p2 and atr[a] > 0:
            levs += [(vah + rng.choice([-1, 1]) * atr[a], "P2V2"), (val + rng.choice([-1, 1]) * atr[a], "P2V2")]
        for lev, cell in levs:
            sgn = np.sign(op - lev)
            if sgn == 0:
                continue
            hit = (l[a:b] <= lev) & (h[a:b] >= lev)
            if hit.any():
                s = a + int(np.argmax(hit)); y = SM.first_hit_race(h, l, c, atr, s, int(sgn), 100)
                if y >= 0:
                    add(cell, t[s], y)
        # V3 80% 규칙
        if op > vah or op < val:
            j30 = np.flatnonzero(d30 == d)
            if len(j30) == 48:
                cc = B30["c"][j30]; ins = (cc >= val) & (cc <= vah)
                two = np.flatnonzero(ins[:-1] & ins[1:])
                if len(two):
                    q = two[0] + 2
                    rest = j30[q:]
                    reach = (B30["l"][rest] <= val).any() if op > vah else (B30["h"][rest] >= vah).any()
                    add("V3", B30["t"][j30[q - 1]], reach)
        # V4 손 안 탄 POC: 날 d 에 안 닿음 → d+1..d+5 에 닿나 (+ 거울)
        if not (lo <= poc <= hi) and all(d + i in rng_d for i in range(1, 6)):
            a5, b5 = rng_d[d + 1][0], rng_d[d + 5][1]; cl = c[b - 1]
            lo5, hi5 = l[a5:b5].min(), h[a5:b5].max()
            add("V4", t[b - 1], (hi5 >= poc) if poc > cl else (lo5 <= poc))
            if with_p2:
                m = 2 * cl - poc; add("P2V4", t[b - 1], (hi5 >= m) if m > cl else (lo5 <= m))
        # V5 LVN 대 vs HVN 대: 다음 4봉 |이동|/ATR
        inr = hist > 0
        pct = np.empty(len(hist)); order = np.argsort(hist, kind="stable"); pct[order] = np.arange(len(hist)) / max(len(hist) - 1, 1)
        for tt in range(a, b):
            if tt + 4 >= len(c) or not atr[tt] > 0 or not (plo <= c[tt] <= phi):
                continue
            kb = int(math.floor(c[tt] / bw)) - k0
            if not (0 <= kb < len(hist)):
                continue
            g = "L" if pct[kb] <= 0.3 else ("H" if pct[kb] >= 0.7 else None)
            if g:
                V5.append((int(t[tt]), g, abs(c[tt + 4] - c[tt]) / atr[tt]))
    return R, V5


def v5_stats(rows: pd.DataFrame) -> float:
    return float(rows[rows.g == "L"].z.mean() - rows[rows.g == "H"].z.mean())


def run() -> None:
    from scipy.stats import norm
    OUT.mkdir(parents=True, exist_ok=True)
    rows, aggs, v5r, v5p = [], [], [], []
    for ci, coin in enumerate(SM.COINS):
        k0 = SM.load_1m(f"{coin}USDT"); k0 = k0[k0.t < END].reset_index(drop=True)
        for draw in range(-1, SM.NPLACEBO):
            kk = k0 if draw < 0 else SM.placebo(k0, SM.SEED + 1000 * ci + draw)
            rng = np.random.default_rng(SM.SEED + 7 * ci + 13 * (draw + 1))
            R, V5 = events(kk, rng, with_p2=draw < 0)
            ev = pd.DataFrame(R).assign(coin=coin, tf="15m", draw=draw)
            v5 = pd.DataFrame(V5, columns=["tk", "g", "z"]).assign(coin=coin, draw=draw)
            if draw < 0:
                rows.append(ev); v5r.append(v5)
            else:
                aggs.append(ev.assign(hh=ev.tk >= SM.SPLIT).groupby(["coin", "tf", "draw", "cell", "hh"]).y.agg(["sum", "count"]).reset_index())
                v5p.append(v5.assign(hh=v5.tk >= SM.SPLIT).groupby(["coin", "draw", "g", "hh"]).z.agg(["sum", "count"]).reset_index())
            print(coin, draw, flush=True)
    E, PA = pd.concat(rows, ignore_index=True), pd.concat(aggs, ignore_index=True)
    V5R, V5P = pd.concat(v5r, ignore_index=True), pd.concat(v5p, ignore_index=True)
    E.to_parquet(OUT / "events.parquet"); PA.to_parquet(OUT / "placebo_agg.parquet"); V5R.to_parquet(OUT / "v5_real.parquet"); V5P.to_parquet(OUT / "v5_placebo.parquet")
    # V1~V4: SMC 판정 함수를 칸 이름만 바꿔 그대로
    SM.TFS, SM.MAIN_TFS, SM.CELLS = {"15m": 15}, ("15m",), ["V1", "V2", "V3", "V4"]
    SM.P2OF = {"V1": "P2V1", "V2": "P2V2", "V4": "P2V4"}
    res = {k.split("|")[1]: v for k, v in SM.judge(E, PA).items()}
    # V5: 차이(LVN 대 − HVN 대) — 실제는 날 군집 부트스트랩, 위약은 벌마다 같은 차이
    rng = np.random.default_rng(SM.SEED)
    day = (V5R.tk // DAY).to_numpy(); ud, inv = np.unique(day, return_inverse=True)
    L, H = (V5R.g == "L").to_numpy(), (V5R.g == "H").to_numpy(); z = V5R.z.to_numpy()
    sL, nL, sH, nH = np.bincount(inv, z * L), np.bincount(inv, L), np.bincount(inv, z * H), np.bincount(inv, H)
    bs = [sL[ix].sum() / nL[ix].sum() - sH[ix].sum() / nH[ix].sum() for ix in rng.integers(0, len(ud), (2000, len(ud)))]
    pdiff = lambda g: g[g.g == "L"]["sum"].sum() / g[g.g == "L"]["count"].sum() - g[g.g == "H"]["sum"].sum() / g[g.g == "H"]["count"].sum()
    pk = V5P.groupby("draw").apply(pdiff)
    Rv = v5_stats(V5R); se = math.sqrt(np.var(bs, ddof=1) + pk.var(ddof=1) / len(pk)); diff = Rv - pk.mean()
    res["V5"] = dict(n=int(len(V5R)), real=round(Rv, 4), p1=round(float(pk.mean()), 4), diff_pp=round(diff, 4), lo90_pp=round(diff - 1.2816 * se, 4),
                     p_one=float(1 - norm.cdf(diff / se)), p_rev=float(norm.cdf(diff / se)), unit="ATR (LVN대 − HVN대 다음 4봉 |이동|)",
                     coin_diff_pp={c: round(v5_stats(V5R[V5R.coin == c]) - pdiff(V5P[V5P.coin == c]), 4) for c in SM.COINS},
                     half_diff_pp={h_: round(v5_stats(V5R[(V5R.tk >= SM.SPLIT) == h_]) - pdiff(V5P[V5P.hh == h_]), 4) for h_ in (False, True)},
                     zL=round(float(V5R[V5R.g == "L"].z.mean()), 4), zH=round(float(V5R[V5R.g == "H"].z.mean()), 4))
    # Holm 5칸 다시(SM.judge 는 4칸으로 했으므로 덮어씀)
    order = sorted(CELLS, key=lambda c_: res[c_]["p_one"]); alive = True
    for i, c_ in enumerate(order):
        alive = alive and res[c_]["p_one"] <= .10 / (5 - i); res[c_]["holm"] = alive
    for c_ in CELLS:
        o = res[c_]; consist = sum(v > 0 for v in o["coin_diff_pp"].values()) >= 3 and all(v > 0 for v in o["half_diff_pp"].values())
        o["pass"] = bool(o["holm"] and o["lo90_pp"] > 0 and o.get("lo90_2_pp", 1) > 0 and consist)
        o["candidate"] = bool(o["diff_pp"] > 0 and o.get("diff2_pp", 1) > 0 and consist)
        o["reversed"] = bool(o["p_rev"] <= .10 / 5 and sum(v < 0 for v in o["coin_diff_pp"].values()) >= 3 and all(v < 0 for v in o["half_diff_pp"].values()))
    json.dump(res, open(OUT / "result.json", "w"), ensure_ascii=False, indent=1, default=float)
    for c_ in CELLS:
        print(c_, {x: res[c_].get(x) for x in ("n", "real", "p1", "p2", "diff_pp", "lo90_pp", "diff2_pp", "lo90_2_pp", "coin_diff_pp", "half_diff_pp", "pass", "candidate", "reversed")})


def selftest() -> None:
    # 프로파일: 하루 1,440분 · 100 에 거래량 몰림 → POC ≈ 100, 가치영역이 100 을 감쌈
    t = np.arange(3 * 1440) * 60_000
    c = np.r_[np.full(1440, 100.0), np.linspace(100, 104, 1440), np.full(1440, 104.0)]
    v = np.ones(len(t)); v[:1440] = 10
    k = pd.DataFrame(dict(t=t, h=c + 0.01, l=c - 0.01, c=c, v=v))
    P = profiles(k)
    bw, k0, hist, va, lo, hi = P[0]
    assert va["val"] <= 100 <= va["vah"] and abs(va["poc"] - 100) <= bw
    # V1: 날 1 시가 100 = POC 와 1칸 미만 → 사건 없음 · 날 2 시가 104, 날 1 POC(분포 고름) 는 범위 안 → 날 2 범위 104 고정이면 안 닿음
    R, V5 = events(k, np.random.default_rng(0), True)
    v1 = [r for r in R if r["cell"] == "V1"]
    assert all(r["tk"] != 1440 * 60_000 for r in v1)          # 날 1(시가 = 전일 POC) 은 제외
    # V5 그룹은 L/H 만
    assert {g for _, g, _ in V5} <= {"L", "H"}
    # 80% 규칙: 전일 대부분 100(가치영역 ≈ 100) · 마지막 1시간 106 으로 마감(시가 > VAH) → 다음 날 첫 1시간 100(30분 종가 둘이 안) → 뒤에 98(VAL 도달)
    c2 = np.r_[np.tile(np.linspace(99, 101, 60), 23), np.linspace(105, 107, 60), np.full(1380, 100.5), np.full(60, 98.0)]   # 전일 99~101 고르게(가치영역 ≈ 99.3~100.7)
    k2 = pd.DataFrame(dict(t=t[:2880], h=c2 + 0.01, l=c2 - 0.01, c=c2, v=np.ones(2880)))
    R2, _ = events(k2, np.random.default_rng(0), False)
    v3 = [r for r in R2 if r["cell"] == "V3"]
    assert len(v3) == 1 and v3[0]["y"] == 1.0
    c3 = c2.copy(); c3[-60:] = 100.5                           # 반대 끝까지 안 감
    R3, _ = events(pd.DataFrame(dict(t=t[:2880], h=c3 + 0.01, l=c3 - 0.01, c=c3, v=np.ones(2880))), np.random.default_rng(0), False)
    assert [r["y"] for r in R3 if r["cell"] == "V3"] == [0.0]
    print("selftest ok")


if __name__ == "__main__":
    {"selftest": selftest, "run": run}[sys.argv[1]]()
