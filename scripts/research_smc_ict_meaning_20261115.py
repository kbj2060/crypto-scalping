#!/usr/bin/env python3
"""SMC·ICT 지표가 «뜻하는 대로» 움직이는가 (2026-11-15). 사전등록 docs/prereg/smc_ict_meaning_20261115.md (b4a4755b, 계산 전).

  selftest     손계산 가짜 봉으로 스윙·BOS/CHoCH·OB·FVG·EQH·경주 판정 단언
  run          4코인 × 15분·1시간(+4시간 보조) 실제 + 기하 위약 20벌 → tmp/smc_ict_meaning_20261115/{events.parquet,result.json}
입력: data/binance_vision/klines1m{,_daily} (메인 체크아웃, 읽기만) + SCRATCH_K 의 Vision 일 파일(최근분).
수익은 보지 않는다. 각 지표의 주장을 «비율»로 재고, 같은 변동성·방향 기억 없는 가짜 가격(P1)과 바로 옆 가격대(P2)에 견준다.
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
BV = Path("/home/kbj20/crypto-scalping/data/binance_vision")
SK = Path(os.environ.get("SCRATCH_K", "/nonexistent"))
OUT = ROOT / "tmp/smc_ict_meaning_20261115"
COINS = ["ETH", "BTC", "SOL", "XRP"]
TFS = {"15m": 15, "1h": 60, "4h": 240}
MAIN_TFS = ("15m", "1h")
NPLACEBO, SEED = 20, 20261115
W_TOUCH, W_RACE, W_FILL, W_SWEEP = 100, 100, 50, 100
END = int(pd.Timestamp("2026-11-14", tz="UTC").value // 10**6)
SPLIT = int(pd.Timestamp("2025-01-01", tz="UTC").value // 10**6)


# ── 자료 ─────────────────────────────────────────────────────────────────────
def load_1m(sym: str) -> pd.DataFrame:
    ps = sorted((BV / "klines1m").glob(f"{sym}-1m-*.parquet")) + sorted((BV / "klines1m_daily").glob(f"{sym}-1m-*.parquet"))
    ps += sorted(SK.glob(f"{sym}-*.parquet"))
    fs = []
    for p in ps:
        d = pd.read_parquet(p)
        fs.append(d[["t", "h", "l", "c"] + (["v"] if "v" in d else [])])
    k = pd.concat(fs).drop_duplicates("t").sort_values("t").reset_index(drop=True)
    return k[(k.t >= int(pd.Timestamp("2022-01-01", tz="UTC").value // 10**6)) & (k.t < END)].reset_index(drop=True)


def placebo(k: pd.DataFrame, seed: int) -> pd.DataFrame:
    """1분 로그수익 부호를 1분마다 무작위로 뒤집는다. 1분 안 위·아래 폭도 같이 뒤집어 변동성 경로는 그대로."""
    rng = np.random.default_rng(seed)
    c = k.c.to_numpy(); cp = np.r_[c[0], c[:-1]]
    r, up, dn = np.log(c / cp), np.log(k.h.to_numpy() / cp), np.log(k.l.to_numpy() / cp)
    s = rng.choice([-1.0, 1.0], len(c))
    r2 = s * r
    up2, dn2 = np.where(s > 0, up, -dn), np.where(s > 0, dn, -up)
    c2 = c[0] * np.exp(np.cumsum(r2)); cp2 = np.r_[c2[0], c2[:-1]]
    out = pd.DataFrame({"t": k.t, "h": np.maximum(cp2 * np.exp(up2), c2), "l": np.minimum(cp2 * np.exp(dn2), c2), "c": c2})
    if "v" in k:
        out["v"] = k.v.to_numpy()
    return out


def bars(k: pd.DataFrame, minutes: int) -> dict:
    """1분이 다 찬 봉만 남긴다(빠진 봉은 건너뜀). 시가 = 직전 봉 종가."""
    ms = minutes * 60_000
    g = k.groupby(k.t // ms * ms)
    b = pd.DataFrame({"h": g.h.max(), "l": g.l.min(), "c": g.c.last(), "n": g.t.size(), "v": g.v.sum(min_count=minutes) if "v" in k else np.nan})
    b = b[b.n == minutes].drop(columns="n").rename_axis("t").reset_index()
    # ponytail: 빠진 봉은 건너뛰어 이어 붙임(구멍 0.1% 미만) -- 구멍이 늘면 NaN 격자로
    h, l, c = (b[x].to_numpy(float) for x in "hlc")
    cp = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.c_[h - l, np.abs(h - cp), np.abs(l - cp)], axis=1)
    return dict(t=b.t.to_numpy(), h=h, l=l, c=c, o=np.r_[c[0], c[:-1]], v=b.v.to_numpy(float),
                atr=pd.Series(tr).rolling(14).mean().to_numpy(), atr200=pd.Series(tr).rolling(200).mean().to_numpy(), ms=ms)


# ── 지표 ─────────────────────────────────────────────────────────────────────
def pivots(x: np.ndarray, N: int, high: bool) -> np.ndarray:
    """봉 p 가 앞뒤 N봉 안 최고(최저) → True. 알려지는 시각은 p+N(호출부가 처리)."""
    s = pd.Series(x)
    w = s.rolling(2 * N + 1, center=True)
    ext = w.max() if high else w.min()
    return (s == ext).to_numpy() & ext.notna().to_numpy()


@njit(cache=True)
def structure(h, l, c, ph, pl, N):
    """LuxAlgo 식 인과 구조: 확정(p+N)된 최근 스윙을 종가가 넘으면 돌파. 직전 추세 반대면 CHoCH(2), 아니면 BOS(1). OB 도 같이."""
    n = len(c)
    ev = np.zeros((n, 6))      # t, dir, typ, level, ob_btm, ob_top
    m = 0
    top = np.nan; top_i = -1; top_x = True
    btm = np.nan; btm_i = -1; btm_x = True
    trend = 0
    for t in range(n):
        p = t - N
        if p >= 0:
            if ph[p]:
                top = h[p]; top_i = p; top_x = False
            if pl[p]:
                btm = l[p]; btm_i = p; btm_x = False
        if not top_x and c[t] > top:
            idx = t - 1
            if t - top_i > 1:
                idx = top_i + 1
                for j in range(top_i + 1, t):
                    if l[j] <= l[idx]:
                        idx = j
            ev[m, 0] = t; ev[m, 1] = 1.0; ev[m, 2] = 2.0 if trend == -1 else 1.0; ev[m, 3] = top; ev[m, 4] = l[idx]; ev[m, 5] = h[idx]; m += 1
            top_x = True; trend = 1
        if not btm_x and c[t] < btm:
            idx = t - 1
            if t - btm_i > 1:
                idx = btm_i + 1
                for j in range(btm_i + 1, t):
                    if h[j] >= h[idx]:
                        idx = j
            ev[m, 0] = t; ev[m, 1] = -1.0; ev[m, 2] = 2.0 if trend == 1 else 1.0; ev[m, 3] = btm; ev[m, 4] = l[idx]; ev[m, 5] = h[idx]; m += 1
            btm_x = True; trend = -1
    return ev[:m]


@njit(cache=True)
def hold_race(h, l, c, atr, t0, d, btm, top, wt, wr):
    """존 [btm, top] 이 t0 에 알려짐. d=+1 지지(위에서 내려와 닿음)·−1 저항.
    첫 터치(t0+1..t0+wt) → 이탈(존 먼 쪽 너머 종가, 터치 봉부터) vs 목표(가장자리 ± 1 ATR, 터치 다음 봉부터). −1 = 터치 없음, 1 = 버팀, 0 = 못 버팀."""
    n = len(c)
    s = -1
    for j in range(t0 + 1, min(t0 + wt, n - 1) + 1):
        if (d == 1 and l[j] <= top) or (d == -1 and h[j] >= btm):
            s = j
            break
    if s < 0:
        return -1
    a = atr[s]
    if not a > 0:
        return -1
    for j in range(s, min(s + wr, n - 1) + 1):
        if (d == 1 and c[j] < btm) or (d == -1 and c[j] > top):
            return 0
        if j > s and ((d == 1 and h[j] >= top + a) or (d == -1 and l[j] <= btm - a)):
            return 1
    return 0


@njit(cache=True)
def first_hit_race(h, l, c, atr, s, d, wr):
    """봉 s 종가 기준, s+1.. 에서 d 방향 1 ATR 이 반대 1 ATR 보다 먼저면 1. 같은 봉 동시·미결은 0(주장에 불리하게)."""
    n = len(c)
    ref = c[s]; a = atr[s]
    if not a > 0:
        return -1
    for j in range(s + 1, min(s + wr, n - 1) + 1):
        good = (d == 1 and h[j] >= ref + a) or (d == -1 and l[j] <= ref - a)
        bad = (d == 1 and l[j] <= ref - a) or (d == -1 and h[j] >= ref + a)
        if bad:
            return 0
        if good:
            return 1
    return 0


def fvgs(B: dict) -> np.ndarray:
    """패키지 식: 봉 i 양봉 & high[i−1] < low[i+1] → 상방 [high[i−1], low[i+1]]. 알려지는 시각 i+1. → (k, dir, btm, top)."""
    h, l, c, o = B["h"], B["l"], B["c"], B["o"]
    i = np.arange(1, len(c) - 1)
    up = (h[i - 1] < l[i + 1]) & (c[i] > o[i])
    dn = (l[i - 1] > h[i + 1]) & (c[i] < o[i])
    r = [np.c_[i[up] + 1, np.ones(up.sum()), h[i - 1][up], l[i + 1][up]],
         np.c_[i[dn] + 1, -np.ones(dn.sum()), h[i + 1][dn], l[i - 1][dn]]]
    return np.vstack(r)


def eq_levels(B: dict, N: int = 3):
    """LuxAlgo EQH/EQL: 길이 3 연이은 스윙 고점 차 < 0.1·ATR200. 알려진 시각까지 안 넘은 것만. → eq, plain: (k, dir, level)."""
    h, l, a2 = B["h"], B["l"], B["atr200"]
    eq, plain = [], []
    for hi, x, sgn in ((True, h, 1), (False, l, -1)):
        P = np.flatnonzero(pivots(x, N, hi))
        used = np.zeros(len(P), bool)
        for q in range(1, len(P)):
            p1, p2 = P[q - 1], P[q]
            k = p2 + N
            if k >= len(h) or not a2[k] > 0:
                continue
            if abs(x[p2] - x[p1]) < 0.1 * a2[k]:
                lev = max(x[p1], x[p2]) if hi else min(x[p1], x[p2])
                seg = x[p1 + 1: k + 1]
                if (seg.max() <= lev) if hi else (seg.min() >= lev):
                    eq.append((k, sgn, lev)); used[q - 1] = used[q] = True
        for q in np.flatnonzero(~used):
            p = P[q]; k = p + N
            if k < len(h):
                seg = x[p + 1: k + 1]
                if (seg.max() <= x[p]) if hi else (seg.min() >= x[p]):
                    plain.append((k, sgn, x[p]))
    return np.array(eq).reshape(-1, 3), np.array(plain).reshape(-1, 3)


def sweep_info(B: dict, lv: np.ndarray, w: int):
    """수준마다 (w 봉 안 스윕 여부, 스윕 봉 색인 또는 −1, 거리 ATR)."""
    h, l, c, atr = B["h"], B["l"], B["c"], B["atr"]
    out = np.full((len(lv), 3), np.nan)
    for r, (k, sgn, lev) in enumerate(lv):
        k = int(k)
        seg = h[k + 1: k + 1 + w] > lev if sgn > 0 else l[k + 1: k + 1 + w] < lev
        s = k + 1 + int(np.argmax(seg)) if seg.any() else -1
        out[r] = (float(seg.any()), s, sgn * (lev - c[k]) / atr[k] if atr[k] > 0 else np.nan)
    return out


# ── 한 계열(실제 또는 위약) → 사건 행 ──────────────────────────────────────────
def events(B: dict, rng, with_p2: bool) -> list[dict]:
    h, l, c, atr, t = B["h"], B["l"], B["c"], B["atr"], B["t"]
    R = []
    add = lambda cell, k, y, **kw: R.append(dict(cell=cell, tk=int(t[int(k)]) + B["ms"], y=float(y), **kw))
    for N in (5, 50):
        ev = structure(h, l, c, pivots(h, N, True), pivots(l, N, False), N)
        for e in range(len(ev) - 1):
            tt, d, typ = int(ev[e, 0]), ev[e, 1], ev[e, 2]
            add(f"S{1 if typ == 2 else 2}_N{N}", tt, ev[e + 1, 1] == d)
            if tt + 24 < len(c):
                add(f"X{1 if typ == 2 else 2}_N{N}_dir24", tt, np.sign(c[tt + 24] - c[tt]) == d)
        for tt, d, typ, lev, zb, zt in ev:
            y = hold_race(h, l, c, atr, int(tt), int(d), zb, zt, W_TOUCH, W_RACE)
            if y >= 0:
                add(f"S3_N{N}", tt, y)
            if with_p2:
                hgt = zt - zb; sh = rng.choice([-1, 1]) * hgt
                b2, t2 = zb + sh, zt + sh
                if hgt > 0 and ((d == 1 and t2 < c[int(tt)]) or (d == -1 and b2 > c[int(tt)])):
                    y2 = hold_race(h, l, c, atr, int(tt), int(d), b2, t2, W_TOUCH, W_RACE)
                    if y2 >= 0:
                        add(f"P2S3_N{N}", tt, y2)
    F = fvgs(B)
    for k, d, zb, zt in F:
        k = int(k)
        seg = l[k + 1: k + 1 + W_FILL] <= zt if d > 0 else h[k + 1: k + 1 + W_FILL] >= zb
        if k + 1 < len(c):
            add("S4", k, seg.any())
        y = hold_race(h, l, c, atr, k, int(d), zb, zt, W_TOUCH, W_RACE)
        if y >= 0:
            add("S5", k, y)
        if with_p2:
            hgt = zt - zb; sh = rng.choice([-1, 1]) * hgt
            b2, t2 = zb + sh, zt + sh
            if (d > 0 and t2 < c[k]) or (d < 0 and b2 > c[k]):
                y2 = hold_race(h, l, c, atr, k, int(d), b2, t2, W_TOUCH, W_RACE)
                if y2 >= 0:
                    add("P2S5", k, y2)
    eq, plain = eq_levels(B)
    for name, lv in (("", eq), ("P2", plain)) if with_p2 else (("", eq),):
        si = sweep_info(B, lv, W_SWEEP)
        for (k, sgn, lev), (sw, s, dist) in zip(lv, si):
            add(f"{name}S6", k, sw, dist=dist)
            if sw:
                y = first_hit_race(h, l, c, atr, int(s), -int(sgn), W_RACE)
                if y >= 0:
                    vr = B["v"][int(s)] / np.nanmedian(B["v"][max(0, int(s) - 100): int(s)]) if int(s) > 0 else np.nan
                    add(f"{name}S7", s, y, vol=vr)
    day = t // 86_400_000
    ud, first = np.unique(day, return_index=True)
    for q in range(1, len(ud)):
        if ud[q] - ud[q - 1] != 1:
            continue
        a, b0 = first[q - 1], first[q]
        e = first[q + 1] if q + 1 < len(ud) else len(c)
        pdh, pdl = h[a:b0].max(), l[a:b0].min()
        levels = [(pdh, 1, "S8"), (pdl, -1, "S8")]
        if with_p2 and atr[b0] > 0:
            levels += [(pdh + rng.choice([-1, 1]) * atr[b0], 1, "P2S8"), (pdl + rng.choice([-1, 1]) * atr[b0], -1, "P2S8")]
        for lev, sgn, cell in levels:
            hit = h[b0:e] >= lev if sgn > 0 else l[b0:e] <= lev
            if hit.any():
                s = b0 + int(np.argmax(hit))
                y = first_hit_race(h, l, c, atr, s, -sgn, W_RACE)
                if y >= 0:
                    add(cell, s, y)
    return R


def coverage(B: dict) -> dict:
    """보조: FVG·OB(N=5) 존이 «살아 있는» 봉 비율(알려진 뒤 메워지거나 100봉까지)."""
    n = len(B["c"]); live = np.zeros(n, bool)
    for k, d, zb, zt in fvgs(B):
        k = int(k); e = min(k + W_FILL, n - 1)
        seg = B["l"][k + 1: e + 1] <= zt if d > 0 else B["h"][k + 1: e + 1] >= zb
        live[k: (k + 1 + int(np.argmax(seg))) if seg.any() else e + 1] = True
    days = (B["t"][-1] - B["t"][0]) / 86_400_000
    return dict(fvg_per_day=round(len(fvgs(B)) / days, 1), fvg_live_share=round(live.mean(), 3))


# ── 판정 ─────────────────────────────────────────────────────────────────────
CELLS = [f"S{s}_N{N}" for s in (1, 2, 3) for N in (5, 50)] + ["S4", "S5", "S6", "S7", "S8"]
P2OF = {"S3_N5": "P2S3_N5", "S3_N50": "P2S3_N50", "S5": "P2S5", "S6": "P2S6", "S7": "P2S7", "S8": "P2S8"}


def p2_rate(df: pd.DataFrame, cell: str, eqd: np.ndarray | None) -> float:
    if cell != "S6":
        return df.y.mean()
    edges = np.nanquantile(eqd, np.linspace(0, 1, 11))
    bi = np.clip(np.searchsorted(edges, df.dist.to_numpy(), "right") - 1, 0, 9)
    ok = np.isfinite(df.dist.to_numpy()) & (df.dist.to_numpy() >= edges[0]) & (df.dist.to_numpy() <= edges[-1])
    r = pd.Series(df.y.to_numpy()[ok]).groupby(bi[ok]).mean()
    return float(r.reindex(range(10)).mean())


def judge(real: pd.DataFrame, plc: pd.DataFrame) -> dict:
    from scipy.stats import norm
    rng = np.random.default_rng(SEED)
    out = {}
    for tf in TFS:
        for cell in CELLS + ["X1_N5_dir24", "X1_N50_dir24", "X2_N5_dir24", "X2_N50_dir24"]:
            r = real[(real.tf == tf) & (real.cell == cell)]
            if r.empty:
                continue
            p = plc[(plc.tf == tf) & (plc.cell == cell)]
            rate = lambda g: g["sum"].sum() / g["count"].sum()
            pk = p.groupby("draw").apply(rate)
            R = r.y.mean(); Pbar = pk.mean()
            day = (r.tk // 86_400_000).to_numpy(); ud = np.unique(day); ix = {u: np.flatnonzero(day == u) for u in ud}
            y = r.y.to_numpy()
            p2c = P2OF.get(cell); r2 = real[(real.tf == tf) & (real.cell == p2c)] if p2c else None
            d2day = (r2.tk // 86_400_000).to_numpy() if p2c else None
            ix2 = {u: np.flatnonzero(d2day == u) for u in np.unique(d2day)} if p2c else None
            eqd = r.dist.to_numpy() if cell == "S6" else None
            bs, bs2 = [], []
            for _ in range(2000):
                dd = rng.choice(ud, len(ud))
                sel = np.concatenate([ix[u] for u in dd])
                bs.append(y[sel].mean())
            if p2c:                                    # 짝 비교는 두 쪽 날짜의 합집합에서 함께 뽑는다(11-15 채점 뒤 수정: 실제 날만 뽑으면 P2 만 있는 날이 빠져 치우침)
                uu = np.union1d(ud, np.unique(d2day))
                for _ in range(2000):
                    dd = rng.choice(uu, len(uu))
                    sel = np.concatenate([ix[u] for u in dd if u in ix] or [np.array([], int)])
                    s2 = np.concatenate([ix2[u] for u in dd if u in ix2] or [np.array([], int)])
                    if len(sel) and len(s2):
                        bs2.append(y[sel].mean() - p2_rate(r2.iloc[s2], cell, eqd[sel] if eqd is not None else None))
            se = math.sqrt(np.var(bs, ddof=1) + pk.var(ddof=1) / len(pk))
            diff = R - Pbar
            row = dict(n=int(len(r)), per_day=round(len(r) / len(ud), 2) if len(ud) else 0, real=round(R, 4), p1=round(Pbar, 4), p1_n=int(p["count"].sum() / max(len(pk), 1)),
                       diff_pp=round(100 * diff, 2), lo90_pp=round(100 * (diff - 1.2816 * se), 2), p_one=float(1 - norm.cdf(diff / se)) if se > 0 else 1.0,
                       p_rev=float(norm.cdf(diff / se)) if se > 0 else 1.0)
            if p2c:
                P2 = p2_rate(r2, cell, eqd)
                row.update(p2=round(P2, 4), p2_n=int(len(r2)), diff2_pp=round(100 * (R - P2), 2), lo90_2_pp=round(100 * float(np.quantile(bs2, .10)), 2), hi90_2_pp=round(100 * float(np.quantile(bs2, .90)), 2))
            pc = p.groupby("coin").apply(rate)
            row["coin_diff_pp"] = {c: round(100 * (r[r.coin == c].y.mean() - pc.get(c, np.nan)), 2) for c in COINS}
            ph = p.groupby("hh").apply(rate)
            row["half_diff_pp"] = {h_: round(100 * (r[(r.tk >= SPLIT) == h_].y.mean() - ph.get(h_, np.nan)), 2) for h_ in (False, True)}
            out[f"{tf}|{cell}"] = row
    # Holm(α .10) — 주 판정 22칸(15분·1시간)
    main = sorted([k for k in out if k.split("|")[0] in MAIN_TFS and k.split("|")[1] in CELLS], key=lambda k: out[k]["p_one"])
    m = len(main); alive = True
    for i, k in enumerate(main):
        alive = alive and out[k]["p_one"] <= .10 / (m - i)
        out[k]["holm"] = alive
    for k in main:
        o = out[k]; signs = [v > 0 for v in o["coin_diff_pp"].values()]; halves = [v > 0 for v in o["half_diff_pp"].values()]
        consist = sum(signs) >= 3 and all(halves)
        p2ok = o.get("lo90_2_pp", 1) > 0
        o["pass"] = bool(o["holm"] and o["lo90_pp"] > 0 and p2ok and consist)
        o["candidate"] = bool(o["diff_pp"] > 0 and o.get("diff2_pp", 1) > 0 and consist)
        rs = [v < 0 for v in o["coin_diff_pp"].values()]
        o["reversed"] = bool(o["p_rev"] <= .10 / m and sum(rs) >= 3 and all(v < 0 for v in o["half_diff_pp"].values()))
    return out


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows, aggs, cov = [], [], {}
    for ci, coin in enumerate(COINS):
        k = load_1m(f"{coin}USDT")
        for draw in range(-1, NPLACEBO):
            kk = k if draw < 0 else placebo(k, SEED + 1000 * ci + draw)
            for tf, mnt in TFS.items():
                B = bars(kk, mnt)
                rng = np.random.default_rng(SEED + 7 * ci + 13 * (draw + 1) + mnt)
                ev = pd.DataFrame(events(B, rng, with_p2=draw < 0)).assign(coin=coin, tf=tf, draw=draw)
                if draw < 0:
                    rows.append(ev); cov[f"{coin}|{tf}"] = coverage(B)
                else:                                  # 위약은 칸·구간별 합·건수만 (메모리)
                    aggs.append(ev.assign(hh=ev.tk >= SPLIT).groupby(["coin", "tf", "draw", "cell", "hh"]).y.agg(["sum", "count"]).reset_index())
            print(coin, draw, flush=True)
    E = pd.concat(rows, ignore_index=True); PA = pd.concat(aggs, ignore_index=True)
    E.to_parquet(OUT / "events.parquet"); PA.to_parquet(OUT / "placebo_agg.parquet")
    res = judge(E, PA)
    aux = {}
    s7 = E[E.cell.isin(["S7", "P2S7"])]
    aux["sweep_vol_ratio_median"] = {f"{tf}|{c}": round(float(g.vol.median()), 3) for (tf, c), g in s7.groupby(["tf", "cell"])}
    json.dump(dict(result=res, coverage=cov, aux=aux), open(OUT / "result.json", "w"), ensure_ascii=False, indent=1, default=float)
    for kk_, v in res.items():
        print(kk_, {x: v[x] for x in ("n", "real", "p1", "diff_pp", "lo90_pp", "p2", "diff2_pp", "pass", "candidate", "reversed") if x in v})


def rejudge() -> None:
    """저장된 사건으로 판정만 다시(정의·자료 그대로)."""
    E, PA = pd.read_parquet(OUT / "events.parquet"), pd.read_parquet(OUT / "placebo_agg.parquet")
    d = json.load(open(OUT / "result.json"))
    d["result"] = judge(E, PA)
    json.dump(d, open(OUT / "result.json", "w"), ensure_ascii=False, indent=1, default=float)
    for k, v in d["result"].items():
        if "p2" in v or k.split("|")[0] in MAIN_TFS:
            print(k, {x: v.get(x) for x in ("n", "real", "p1", "p2", "diff_pp", "lo90_pp", "diff2_pp", "lo90_2_pp", "pass", "candidate", "reversed")})


# ── 자체 점검 ────────────────────────────────────────────────────────────────
def selftest() -> None:
    # 스윙: 가운데 봉만
    x = np.array([1, 2, 3, 9, 3, 2, 1, 2, 3.])
    assert np.flatnonzero(pivots(x, 2, True)).tolist() == [3]
    # 구조: 고점 10(봉 2) → 확정 봉 4 → 봉 6 종가 11 상방 돌파(추세 0 → BOS) · OB = 봉 3~5 중 저가 최저 봉
    h = np.array([5, 8, 10, 7, 6, 7, 12, 11, 9, 4, 3, 5.]); l = h - 1; c = h - 0.5
    l[4] = 4.0
    ev = structure(h, l, c, pivots(h, 2, True), pivots(l, 2, False), 2)
    up = ev[ev[:, 1] == 1][0]
    assert up[0] == 6 and up[2] == 1 and up[3] == 10 and up[4] == 4.0 and up[5] == 6
    dn = ev[ev[:, 1] == -1]
    assert len(dn) and dn[0][2] == 2                     # 상승 추세 뒤 하방 돌파 = CHoCH
    # 경주: 지지 존 [9,10], 터치 봉 2 → 봉 3 고가 11.5 ≥ 10+1 → 버팀
    H = np.array([12, 11, 10.5, 11.5, 12.]); L = np.array([11, 10.5, 9.5, 10, 11.]); C = np.array([11.5, 11, 10, 11, 11.5]); A = np.ones(5)
    assert hold_race(H, L, C, A, 0, 1, 9.0, 10.0, 10, 10) == 1
    C2 = C.copy(); C2[2] = 8.9                          # 터치 봉 종가가 존 아래 → 못 버팀
    assert hold_race(H, L, C2, A, 0, 1, 9.0, 10.0, 10, 10) == 0
    assert hold_race(H + 5, L + 5, C + 5, A, 0, 1, 9.0, 10.0, 10, 10) == -1
    # 먼저 닿기: 같은 봉에 둘 다면 0
    assert first_hit_race(np.array([10, 11.5, 12.]), np.array([9, 9.5, 8.]), np.array([10, 10, 10.]), np.ones(3), 0, -1, 5) == 0
    assert first_hit_race(np.array([10, 10.5, 12.]), np.array([9, 8.9, 8.]), np.array([10, 10, 10.]), np.ones(3), 0, -1, 5) == 1
    # FVG: 봉 1 양봉, high[0]=10 < low[2]=11 → 알려진 시각 2, [10, 11]
    B = dict(h=np.array([10, 12, 13, 13.]), l=np.array([9, 10, 11, 12.]), c=np.array([9.5, 11.8, 12.5, 12.6]))
    B["o"] = np.r_[B["c"][0], B["c"][:-1]]
    assert fvgs(B).tolist() == [[2, 1, 10, 11]]
    # 위약: 변동 크기 보존
    k = pd.DataFrame({"t": np.arange(6) * 60000, "c": [100, 101, 100.5, 102, 101, 103.]})
    k["h"], k["l"] = k.c + 0.3, k.c - 0.4
    p = placebo(k, 1)
    assert np.allclose(np.abs(np.diff(np.log(p.c))), np.abs(np.diff(np.log(k.c))))
    print("selftest ok")


if __name__ == "__main__":
    {"selftest": selftest, "run": run, "judge": rejudge}[sys.argv[1]]()
