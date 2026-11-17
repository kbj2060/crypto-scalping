#!/usr/bin/env python3
"""풋프린트 «지표 방향 표식·유동성 구역» 규칙의 SOXL 적용 (갈래 C, 2026-11-17). 사전등록 docs/prereg/soxl_dir_marks_20261117.md (91eab1d6, 계산 전).

  build ZIPDIR_SOXL ZIPDIR_ETH   Vision 일 zip → parquet (SOXL 11-11~16 · ETH 원 연구에 있던 최근 일) · SOXL 11-09·10 겹침 대조
  selftest                       튜닝용 변형 함수(매개변수 = 원값)가 원 함수와 같은 사건을 내는지 가짜 경로로 단언
  posctrl                        양성 대조: ETH 로 원 연구 사건(실제·대조·가짜 한 벌) 재현 → tmp/soxl_dir_marks_20261117/posctrl.json
  run                            SOXL E 팔 + 튜닝 격자 실제·가짜 20벌 → events.parquet·placebo.parquet → 판정 result.json
정의·경주·위약·판정 = 원 연구 모듈(복사본, 고치지 않음). 이 파일은 묶기·구간 나누기·판정 트리·변형 격자만.
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_smc_ict_meaning_20261115 as SM  # noqa: E402
import research_a11_a13_newcoins_r2_20261116 as A13  # noqa: E402
import research_orb_meaning_20261115 as OR  # noqa: E402
import research_classic_indicator_bundle_20261115 as CB  # noqa: E402
import research_anchored_vwap_meaning_20261115 as AV  # noqa: E402
import research_volume_osc_bundle_20261116 as VO  # noqa: E402
import research_connors_aroon_soup_bundle_20261116 as CA  # noqa: E402
import research_rsi_divergence_meaning_20261115 as RD  # noqa: E402
from research_soxl_session_vol_20261112 import clock, SESS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/soxl_dir_marks_20261117"
SOXL_K = Path("/home/kbj20/crypto-scalping/.claude/worktrees/support-resistance-breakout-signals-ae2e43/tmp/soxl_vol_20261112/klines")
WT = Path("/home/kbj20/crypto-scalping/.claude/worktrees")
ORIG_F, ORIG_C = WT / "fujimoto-126-trading-analysis-724dd5/tmp", WT / "coinglass-funding-rate-indicators-5206a7/tmp"
ms = lambda s: int(pd.Timestamp(s, tz="UTC").value // 10**6)
WIN = {"train": ("2026-05-22", "2026-09-01", "2026-07-12"), "val": ("2026-09-01", "2026-10-01", "2026-09-16"),
       "test": ("2026-10-01", "2026-11-17", "2026-10-24"), "full": ("2026-05-22", "2026-11-17", "2026-08-19")}   # [시작, 끝) · 반쪽 경계
SOXL_CI = 50                                                   # SOXL 씨앗 자리(원 연구 코인 색인과 겹치지 않게)
MIN_N = 30
# 주 판정 11칸: (묶음, tf, 칸, 대조 칸, 기준 효과 vs 가짜, vs 대조(None = 기준 없음))
MAIN = [("A11", "15m", "S3_N5", "P2S3_N5", 2.19, 2.04), ("A11", "15m", "S3_N50", "P2S3_N50", 4.96, None), ("A11", "15m", "S5", "P2S5", 1.02, 1.54),
        ("OB5", "5m", "S3_N50", "P2S3_N50", 6.76, 2.09), ("A13", "30m", "K", "P3K", 2.66, 1.78), ("A13", "1h", "K", "P3K", 3.78, 2.63),
        ("A14", "or30", "NY", "P3", 2.12, 2.38), ("A14", "or60", "NY", "P3", 4.01, 4.29), ("A15", "1h", "E1", "P3E1", 1.88, 0.99),
        ("A16", "15m", "G1", "P3G1", 1.91, 0.99), ("A12", "15m", "D1", "P2D1", 1.35, 1.14)]
MIN_OF = {"5m": 5, "15m": 15, "30m": 30, "1h": 60, "or15": 5, "or30": 5, "or60": 5}


# ── 자료 ─────────────────────────────────────────────────────────────────────
def vision_csv(raw: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        d = pd.read_csv(z.open(z.namelist()[0]), header=None)
    if not str(d.iloc[0, 0]).isdigit():
        d = d.iloc[1:]
    d = d.astype(float)
    return pd.DataFrame({"t": d[0].astype("int64"), "h": d[2], "l": d[3], "c": d[4], "v": d[5], "tb": d[9]})


def build(zs: str, ze: str) -> None:
    for sub, src, sym in (("klines", zs, "SOXLUSDT"), ("eth_k", ze, "ETHUSDT")):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
        for z in sorted(Path(src).glob(f"{sym}-1m-*.zip")):
            d = vision_csv(z.read_bytes()); day = z.stem.split("-1m-")[1]
            assert len(d) == 1440 and (np.diff(d.t) == 60000).all(), (z, len(d))
            old = SOXL_K / f"{sym}-1m-{day}.parquet"
            if sym == "SOXLUSDT" and old.exists():                 # 겹치는 날 = 대조만(쓰지 않음)
                o = pd.read_parquet(old)
                assert np.allclose(o[["t", "h", "l", "c", "v"]].to_numpy(), d[["t", "h", "l", "c", "v"]].to_numpy()), day
                print("겹침 일치", day); continue
            d.to_parquet(OUT / sub / f"{sym}-1m-{day}.parquet")
        print(sub, len(list((OUT / sub).glob("*.parquet"))))


def load_soxl() -> pd.DataFrame:
    ps = sorted(SOXL_K.glob("SOXLUSDT-1m-*.parquet")) + sorted((OUT / "klines").glob("SOXLUSDT-1m-*.parquet"))
    k = pd.concat([pd.read_parquet(p)[["t", "h", "l", "c", "v"]] for p in ps]).drop_duplicates("t").sort_values("t").reset_index(drop=True)
    k = k[k.t < ms("2026-11-17")].reset_index(drop=True)
    assert (np.diff(k.t) == 60000).all() and k.t.iloc[0] == ms("2026-05-15 14:00")
    return k


# ── 튜닝용 변형 (매개변수 = 원값이면 원 함수와 같은 사건 · selftest 단언) ─────────────
def race_add(R, cell, tf, B, i, d, k=1.0):
    h, l, c, atr, t = B["h"], B["l"], B["c"], B["atr"], B["t"]
    if i < CB.WARM or i + 1 >= len(c) or d == 0 or not atr[i] > 0:
        return
    y = SM.first_hit_race(h, l, c, atr * k, int(i), int(d), 100)
    if y >= 0:
        R.append(dict(cell=cell, tf=tf, tk=int(t[i]) + B["ms"], y=float(y)))


def ob_var(B, tf, N, rng, ctrl, cell):
    h, l, c, atr, t = B["h"], B["l"], B["c"], B["atr"], B["t"]; R = []
    for tt, d, typ, lev, zb, zt in SM.structure(h, l, c, SM.pivots(h, N, True), SM.pivots(l, N, False), N):
        y = SM.hold_race(h, l, c, atr, int(tt), int(d), zb, zt, SM.W_TOUCH, SM.W_RACE)
        if y >= 0:
            R.append(dict(cell=cell, tf=tf, tk=int(t[int(tt)]) + B["ms"], y=float(y)))
        if ctrl:
            hgt = zt - zb; sh = rng.choice([-1, 1]) * hgt; b2, t2 = zb + sh, zt + sh
            if hgt > 0 and ((d == 1 and t2 < c[int(tt)]) or (d == -1 and b2 > c[int(tt)])):
                y2 = SM.hold_race(h, l, c, atr, int(tt), int(d), b2, t2, SM.W_TOUCH, SM.W_RACE)
                if y2 >= 0:
                    R.append(dict(cell=cell + "|C", tf=tf, tk=int(t[int(tt)]) + B["ms"], y=float(y2)))
    return R


def a13_var(B, tf, N, k, ctrl, cell):
    h, l, c, atr, t = B["h"], B["l"], B["c"], B["atr"], B["t"]; R = []
    for s, d in SM.structure(h, l, c, SM.pivots(h, N, True), SM.pivots(l, N, False), N)[:, :2]:
        s = int(s)
        if s + 1 < len(c) and atr[s] > 0:
            y = SM.first_hit_race(h, l, c, k * atr, s, int(d), 100)
            if y >= 0:
                R.append(dict(cell=cell, tf=tf, tk=int(t[s]) + B["ms"], y=float(y)))
    if ctrl:
        for i in range(250, len(c) - 1, A13.P3STEP[tf]):
            d = np.sign(c[i] - c[i - 20])
            if d != 0 and atr[i] > 0:
                y = SM.first_hit_race(h, l, c, k * atr, i, int(d), 100)
                if y >= 0:
                    R.append(dict(cell=cell + "|C", tf=tf, tk=int(t[i]) + B["ms"], y=float(y)))
    return R


def orb_var(B, L, K, ctrl, cell):
    h, l, c, t = B["h"], B["l"], B["c"], B["t"]; R = []
    S = OR.starts(int(t[0]), int(t[-1]))
    sets = {cell: S["NY"]}
    if ctrl:
        sets[cell + "|C"] = S["P3"][pd.to_datetime(S["P3"], unit="ms").dayofweek < 5]
    for kind, ss in sets.items():
        ix = np.searchsorted(t, ss); ok = (ix < len(t)) & (t[np.clip(ix, 0, len(t) - 1)] == ss)
        for s0, i0 in zip(ss[ok], ix[ok]):
            o = OR.orb(B, int(s0), L, int(i0))
            if o is not None:
                b, d, W = o
                R.append(dict(cell=kind, tf=f"or{L}", tk=int(t[b]) + OR.BAR, y=float(OR.race3(h, l, c, b, d, K * W))))
    return R


def cci_var(B, tf, thr, ctrl, cell):
    h, l, c = B["h"], B["l"], B["c"]; n = len(c); R = []
    C = VO.cci(h, l, c)
    x = np.where(CB.cross(C, np.full(n, float(thr))) == 1, 1, np.where(CB.cross(C, np.full(n, -float(thr))) == -1, -1, 0))
    for i in np.flatnonzero(x):
        race_add(R, cell, tf, B, i, x[i])
    if ctrl:
        for i in range(CB.WARM, n - 1, CB.STEP[tf]):
            race_add(R, cell + "|C", tf, B, i, np.sign(c[i] - c[i - 20]))
    return R


STEP5 = 12                                                     # 5분봉 대조 간격 = 60분마다(15분 4·1시간 1 과 같은 시계 간격 · 원 CB.STEP 에 5분 없음)


def rsi2_var(B, tf, lo, hi, sma_n, ctrl, cell):
    c = B["c"]; n = len(c); R = []
    r2, tr = RD.rsi(c, 2), np.sign(c - CB.sma(c, sma_n))
    with np.errstate(invalid="ignore"):
        G = np.where((tr == 1) & (CB.cross(r2, np.full(n, float(lo))) == -1), 1, np.where((tr == -1) & (CB.cross(r2, np.full(n, float(hi))) == 1), -1, 0))
    for i in np.flatnonzero(G):
        race_add(R, cell, tf, B, i, G[i])
    if ctrl:
        for i in range(CB.WARM, n - 1, CB.STEP.get(tf, STEP5)):
            if 30 <= r2[i] <= 70:
                race_add(R, cell + "|C", tf, B, i, tr[i])
    return R


def rdiv_var(k1, N, GAP, ctrl, cell):
    RD.N, RD.GAP = N, GAP
    try:
        ev = RD.events(k1, "15m", 15)
    finally:
        RD.N, RD.GAP = 5, 60
    return [dict(e, cell=cell if e["cell"] == "D1" else cell + "|C") for e in ev if e["cell"] == "D1" or (ctrl and e["cell"] == "P2D1")]


GRID = ([("OB15", f"N{N}", ("ob", "15m", N)) for N in (5, 10, 20, 50)] + [("OB5", f"N{N}", ("ob", "5m", N)) for N in (5, 10, 20, 50)]
        + [("FVG", tf, ("fvg", tf)) for tf in ("5m", "15m", "30m")]
        + [("A13", f"{tf}_N{N}_{k}atr", ("a13", tf, N, k)) for tf in ("30m", "1h") for N in (10, 20, 50) for k in (1, 2)]
        + [("A14", f"L{L}_{K}W", ("orb", L, K)) for L in (15, 30, 60) for K in (1, 2)]
        + [("A15", f"{tf}_{thr}", ("cci", tf, thr)) for tf in ("15m", "1h") for thr in (100, 150, 200)]
        + [("A16", f"{tf}_{lo}_{sm}", ("rsi2", tf, lo, 100 - lo, sm)) for tf in ("5m", "15m") for lo in (5, 10, 20) for sm in (100, 200)]
        + [("A12", f"N{N}_G{G}", ("rdiv", N, G)) for N in (3, 5, 8) for G in (30, 60)])


# ── 한 계열(실제 또는 가짜) → 사건 ───────────────────────────────────────────────
def e_events(k1, ci, draw, ctrl, av_end=True) -> list[dict]:
    """E 팔 11칸(+대조) — 원 함수 그대로. ci·draw = 원 연구 씨앗 규약(옆 띠 무작위 부호)."""
    R = []
    for tf, mnt in (("15m", 15), ("5m", 5)):                   # SMC: OB·FVG (원 run 의 rng 씨앗)
        B = SM.bars(k1, mnt); rng = np.random.default_rng(SM.SEED + 7 * ci + 13 * (draw + 1) + mnt)
        keep = {"S3_N5", "S3_N50", "S5", "P2S3_N5", "P2S3_N50", "P2S5"} if tf == "15m" else {"S3_N50", "P2S3_N50"}
        R += [dict(cell=e["cell"], tf=tf, tk=e["tk"], y=e["y"]) for e in SM.events(B, rng, with_p2=ctrl) if e["cell"] in keep]
    for tf, mnt in (("30m", 30), ("1h", 60)):                  # A13
        B = SM.bars(k1, mnt); B["tf"] = tf
        R += [dict(cell=e["cell"], tf=tf, tk=e["tk"], y=e["y"]) for e in A13.a13_events(B, with_p3=ctrl) if e["cell"] in ("K", "P3K")]
    R += [e for e in OR.events2(SM.bars(k1, 5), ctrl) if e["cell"] in ("NY", "P3", "P3b")]   # A14 (y −1 = 미결)
    old_end, VO.ONLY = AV.END, ("E1",)
    if not av_end:
        AV.END = 2**62                                          # 거래량 열 끝(원 4코인 자료 한계) -- SOXL 은 끝까지 있음
    try:
        B = SM.bars(k1, 60); ev = VO.events(B, "1h", ctrl)
    finally:
        AV.END, VO.ONLY = old_end, None
    R += [dict(cell=c_, tf="1h", tk=t_, y=y_) for c_, t_, y_ in zip(ev["cell"], ev["tk"], ev["y"]) if c_ in ("E1", "P3E1")]
    ev = CA.events(SM.bars(k1, 15), "15m", ctrl)
    R += [dict(cell=c_, tf="15m", tk=t_, y=y_) for c_, t_, y_ in zip(ev["cell"], ev["tk"], ev["y"]) if c_ in ("G1", "P3G1")]
    R += [dict(cell=e["cell"], tf="15m", tk=e["tk"], y=e["y"]) for e in RD.events(k1, "15m", 15) if e["cell"] == "D1" or (ctrl and e["cell"] == "P2D1")]
    return R


def grid_events(k1, draw, ctrl) -> list[dict]:
    R, bars = [], {}
    getB = lambda tf: bars.setdefault(tf, dict(SM.bars(k1, MIN_OF[tf]), tf=tf))
    rng = np.random.default_rng(SM.SEED + 77_000 + 13 * (draw + 1))
    AV.END = 2**62
    for rule, name, p in GRID:
        cell = f"{rule}:{name}"
        if p[0] == "ob":
            R += ob_var(getB(p[1]), p[1], p[2], rng, ctrl, cell)
        elif p[0] == "fvg":
            R += [dict(cell=cell if e["cell"] == "S5" else cell + "|C", tf=p[1], tk=e["tk"], y=e["y"])
                  for e in SM.events(getB(p[1]), rng, with_p2=ctrl) if e["cell"] in ("S5", "P2S5")]
        elif p[0] == "a13":
            R += a13_var(getB(p[1]), p[1], p[2], p[3], ctrl, cell)
        elif p[0] == "orb":
            R += orb_var(getB("or15"), p[1], p[2], ctrl, cell)
        elif p[0] == "cci":
            R += cci_var(getB(p[1]), p[1], p[2], ctrl, cell)
        elif p[0] == "rsi2":
            R += rsi2_var(getB(p[1]), p[1], p[2], p[3], p[4], ctrl, cell)
        else:
            R += rdiv_var(k1, p[1], p[2], ctrl, cell)
    for r in R:                                                 # 변형은 tf 를 칸 이름에 넣어 묶음 판정용 tf 하나로
        r["tf"] = "v"
    return R


# ── 판정 ─────────────────────────────────────────────────────────────────────
def window(df: pd.DataFrame, w: str) -> pd.DataFrame:
    a, b, _ = WIN[w]
    return df[(df.tk >= ms(a)) & (df.tk < ms(b))]


def judge_family(E, P, w, tfs, cells, ctrl_of, minutes):
    """SM.judge 그대로(한 종목 · 창 안 사건 · 반쪽 = 창 가운데) → 칸별 행 + 트리 판정 재료."""
    SM.TFS, SM.MAIN_TFS, SM.CELLS, SM.P2OF, SM.COINS = {tf: minutes(tf) for tf in tfs}, tuple(tfs), list(cells), dict(ctrl_of), ["SOXL"]
    SM.SPLIT = ms(WIN[w][2])
    e, p = window(E, w), window(P, w)
    pa = p.assign(hh=p.tk >= SM.SPLIT).groupby(["coin", "tf", "draw", "cell", "hh"]).y.agg(["sum", "count"]).reset_index()
    return {k: v for k, v in SM.judge(e, pa).items() if k.split("|")[1] in cells}


def tree(v: dict, ref1: float, ref2) -> dict:
    hi1 = 2 * v["diff_pp"] - v["lo90_pp"]; hv = list(v["half_diff_pp"].values())
    cand = bool(v["n"] >= MIN_N and v["diff_pp"] > 0 and v.get("diff2_pp", -1) > 0 and all(x > 0 for x in hv))
    if v["n"] < MIN_N:
        g = "판정 불가(건수 미달)"
    elif v.get("holm") and v["lo90_pp"] > 0 and v.get("lo90_2_pp", -1) > 0 and all(x > 0 for x in hv):
        g = "엄격 통과"
    elif v["p_rev"] <= .10 / v.get("m", 1) and all(x < 0 for x in hv) and v.get("diff2_pp", 0) < 0:
        g = "거꾸로"
    elif hi1 < ref1 or (ref2 is not None and v.get("hi90_2_pp", 1e9) < ref2):
        g = "불통과(원 효과보다 작음)"
    else:
        g = "판정 불가(검정력 부족)"
    return dict(n=v["n"], real=v["real"], p1=v["p1"], ctrl=v.get("p2"), ctrl_n=v.get("p2_n"), d1=v["diff_pp"], ci1=[v["lo90_pp"], round(hi1, 2)],
                d2=v.get("diff2_pp"), ci2=[v.get("lo90_2_pp"), v.get("hi90_2_pp")], halves=v["half_diff_pp"], holm=v.get("holm"),
                grade=g, candidate=cand, ref=[ref1, ref2])


def score_main(E, P, w) -> dict:
    E = E[(E.tf != "v")]; P = P[(P.tf != "v")]
    dec = lambda d: d[~d.tf.str.startswith("or") | (d.y >= 0)]  # A14 = 결판 난 경주만
    E, P = dec(E), dec(P)
    out = {}
    for fam in dict.fromkeys(m[0] for m in MAIN):
        rows = [m for m in MAIN if m[0] == fam]
        tfs = list(dict.fromkeys(m[1] for m in rows)); cells = list(dict.fromkeys(m[2] for m in rows))
        J = judge_family(E, P, w, tfs, cells, {m[2]: m[3] for m in rows}, MIN_OF.get)
        for _, tf, cell, _, r1, r2 in rows:
            v = J.get(f"{tf}|{cell}")
            if v is None:
                out[f"{fam}|{tf}|{cell}"] = dict(n=0, grade="판정 불가(건수 미달)", candidate=False); continue
            v["m"] = len(rows)
            out[f"{fam}|{tf}|{cell}"] = tree(v, r1, r2)
    return out


def grid_stats(E, P, w) -> dict:
    """변형별 창 안 n·실제·대조·가짜(점추정) — 고르기·거르기용."""
    e, p = window(E[E.tf == "v"], w), window(P[P.tf == "v"], w)
    out = {}
    for rule, name, prm in GRID:
        c_ = f"{rule}:{name}"
        r, cc, pp = e[e.cell == c_], e[e.cell == c_ + "|C"], p[p.cell == c_]
        if prm[0] == "orb":
            r, cc, pp = r[r.y >= 0], cc[cc.y >= 0], pp[pp.y >= 0]
        if not len(r):
            out[c_] = dict(n=0); continue
        m = r.y.mean(); pm = pp.groupby("draw").y.mean().mean() if len(pp) else np.nan
        out[c_] = dict(n=int(len(r)), real=round(m, 4), ctrl=round(cc.y.mean(), 4) if len(cc) else None, p1=round(float(pm), 4),
                       d1=round(100 * (m - pm), 2), d2=round(100 * (m - cc.y.mean()), 2) if len(cc) else None)
    return out


def tune(E, P) -> dict:
    tr, va = grid_stats(E, P, "train"), grid_stats(E, P, "val")
    res = {}
    for rule in dict.fromkeys(g[0] for g in GRID):
        ok = {c: v for c, v in tr.items() if c.startswith(rule + ":") and v.get("n", 0) >= MIN_N and v.get("d2") is not None}
        if not ok:
            res[rule] = dict(verdict="튜닝 불가(학습 건수 미달)"); continue
        pick = max(ok, key=lambda c: ok[c]["d2"]); vv = va.get(pick, {})
        r = dict(pick=pick, train=ok[pick], val=vv)
        if not (vv.get("n", 0) and vv.get("d1", -1) > 0 and (vv.get("d2") or -1) > 0):
            r["verdict"] = "검증 탈락"
        else:
            rows = E[(E.tf == "v") & (E.cell.isin([pick, pick + "|C"]))].assign(tf="v"); pr = P[(P.tf == "v") & (P.cell == pick)]
            if pick.startswith("A14"):
                rows, pr = rows[rows.y >= 0], pr[pr.y >= 0]
            v = judge_family(rows, pr, "test", ["v"], [pick], {pick: pick + "|C"}, lambda _: 15)[f"v|{pick}"]
            hv = list(v["half_diff_pp"].values())
            ok_c = bool(v["n"] >= MIN_N and v["diff_pp"] > 0 and v.get("diff2_pp", -1) > 0 and all(x > 0 for x in hv))
            r["test"] = dict(n=v["n"], real=v["real"], p1=v["p1"], ctrl=v.get("p2"), d1=v["diff_pp"], lo1=v["lo90_pp"], d2=v.get("diff2_pp"),
                             ci2=[v.get("lo90_2_pp"), v.get("hi90_2_pp")], halves=v["half_diff_pp"])
            r["verdict"] = "통과 후보" if ok_c else "불통과"
        res[rule] = r
    return dict(train=tr, val=va, picks=res)


def sessions(E, P) -> dict:
    """보조: 전 기간 주 칸 세션별 n·실제·가짜·대조 (사건 tk = 알려진 시각의 세션)."""
    sess = lambda d: d.assign(s=clock(d.tk.to_numpy() // 1000, 60).sess.to_numpy()) if len(d) else d.assign(s=[])
    e, p = window(E[E.tf != "v"], "full"), window(P[P.tf != "v"], "full")
    e = e[~e.tf.str.startswith("or") | (e.y >= 0)]; p = p[~p.tf.str.startswith("or") | (p.y >= 0)]
    e, p = sess(e), sess(p); out = {}
    for fam, tf, cell, ctrl, *_ in MAIN:
        r, cc, pp = e[(e.tf == tf) & (e.cell == cell)], e[(e.tf == tf) & (e.cell == ctrl)], p[(p.tf == tf) & (p.cell == cell)]
        row = {}
        for s in range(6):
            rs, cs, ps = r[r.s == s], cc[cc.s == s], pp[pp.s == s]
            if len(rs):
                row[SESS[s]] = dict(n=int(len(rs)), real=round(rs.y.mean(), 3), p1=round(ps.y.mean(), 3) if len(ps) else None,
                                    ctrl=round(cs.y.mean(), 3) if len(cs) else None,
                                    d1=round(100 * (rs.y.mean() - ps.y.mean()), 1) if len(ps) else None,
                                    d2=round(100 * (rs.y.mean() - cs.y.mean()), 1) if len(cs) else None)
        out[f"{fam}|{tf}|{cell}"] = row
    return out


def aux(E, P) -> dict:
    """보조: A14 휴장 뺀 값·미국 오후 대조·결판 비율 · 2 ATR 판은 생략(사전등록 보조 중 A14 만 계산)."""
    from research_soxl_session_vol_20261112 import HOLIDAYS
    out = {}
    for w in ("test", "full"):
        e = window(E[E.tf.isin(["or30", "or60"])], w)
        for tf in ("or30", "or60"):
            ny = e[(e.tf == tf) & (e.cell == "NY")]
            et = pd.to_datetime(ny.tk, unit="ms", utc=True).dt.tz_convert("America/New_York").dt.strftime("%Y-%m-%d")
            d = ny[(ny.y >= 0) & ~et.isin(HOLIDAYS).to_numpy()]
            p3b = e[(e.tf == tf) & (e.cell == "P3b") & (e.y >= 0)]
            out[f"{w}|{tf}"] = dict(n_all=int(len(ny)), decided_share=round(float((ny.y >= 0).mean()), 3) if len(ny) else None,
                                    n_dec_no_holiday=int(len(d)), real_no_holiday=round(d.y.mean(), 3) if len(d) else None,
                                    us_afternoon_ctrl=round(p3b.y.mean(), 3) if len(p3b) else None, us_afternoon_n=int(len(p3b)))
    return out


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    k = load_soxl(); rows, plc = [], []
    for draw in range(-1, SM.NPLACEBO):
        kk = k if draw < 0 else SM.placebo(k, SM.SEED + 50_000 + draw)
        ev = pd.DataFrame(e_events(kk, SOXL_CI, draw, draw < 0, av_end=False) + grid_events(kk, draw, draw < 0)).assign(coin="SOXL", draw=draw)
        (rows if draw < 0 else plc).append(ev)
        print("draw", draw, len(ev), flush=True)
    E, P = pd.concat(rows, ignore_index=True), pd.concat(plc, ignore_index=True)
    E.to_parquet(OUT / "events.parquet"); P.to_parquet(OUT / "placebo.parquet")
    score(E, P)


def score(E=None, P=None) -> None:
    if E is None:
        E, P = pd.read_parquet(OUT / "events.parquet"), pd.read_parquet(OUT / "placebo.parquet")
    res = dict(prereg="docs/prereg/soxl_dir_marks_20261117.md@91eab1d6", windows=WIN,
               test=score_main(E, P, "test"), full=score_main(E, P, "full"), val=score_main(E, P, "val"),
               sessions=sessions(E, P), a14_aux=aux(E, P), tuning=tune(E, P))
    json.dump(res, open(OUT / "result.json", "w"), ensure_ascii=False, indent=1, default=float)
    for w in ("test", "full"):
        for c_, v in res[w].items():
            print(w, c_, {x: v.get(x) for x in ("n", "real", "p1", "ctrl", "d1", "ci1", "d2", "ci2", "halves", "grade", "candidate")})
    for r_, v in res["tuning"]["picks"].items():
        print("tune", r_, v.get("pick"), v.get("verdict"), v.get("val"), v.get("test"))


# ── 양성 대조 ────────────────────────────────────────────────────────────────
def posctrl() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SM.SK = OUT / "eth_k"
    k = SM.load_1m("ETHUSDT")
    E = pd.DataFrame(e_events(k, 0, -1, True, av_end=True))
    O = {"15m": pd.read_parquet(ORIG_F / "smc_ict_meaning_20261115/events.parquet"), "5m": pd.read_parquet(ORIG_F / "smc_ict_meaning_20261115_r2/events.parquet")}
    orig = []
    for tf, cells in (("15m", ["S3_N5", "S3_N50", "S5", "P2S3_N5", "P2S3_N50", "P2S5"]), ("5m", ["S3_N50", "P2S3_N50"])):
        o = O[tf]; orig.append(o[(o.coin == "ETH") & (o.tf == tf) & o.cell.isin(cells)][["cell", "tf", "tk", "y"]])
    o = pd.read_parquet(ORIG_F / "orb_r2_ny_open_20261115/events_orig.parquet"); orig.append(o[(o.coin == "ETH") & o.cell.isin(["NY", "P3", "P3b"])][["cell", "tf", "tk", "y"]])
    o = pd.read_parquet(ORIG_F / "volume_osc_bundle_20261116/events.parquet"); orig.append(o[(o.coin == "ETH") & (o.tf == "1h") & o.cell.isin(["E1", "P3E1"])][["cell", "tf", "tk", "y"]])
    o = pd.read_parquet(ORIG_F / "connors_aroon_soup_bundle_20261116/events.parquet"); orig.append(o[(o.coin == "ETH") & (o.tf == "15m") & o.cell.isin(["G1", "P3G1"])][["cell", "tf", "tk", "y"]])
    o = pd.read_parquet(ORIG_F / "rsi_divergence_meaning_20261115/events.parquet"); orig.append(o[(o.coin == "ETH") & (o.tf == "15m") & o.cell.isin(["D1", "P2D1"])][["cell", "tf", "tk", "y"]])
    Og = pd.concat(orig)
    out, ok_all = {}, True
    agg = lambda d: d.groupby(["tf", "cell"]).y.agg(["count", "sum"])
    mine = E[~E.cell.isin(["K", "P3K"])]
    cmp = agg(Og).join(agg(mine), lsuffix="_orig", rsuffix="_mine", how="outer")
    for (tf, cell), r in cmp.iterrows():
        dn = r.count_mine - r.count_orig; dr = 100 * (r.sum_mine / r.count_mine - r.sum_orig / r.count_orig)
        ok = bool(abs(dn) <= 0.005 * r.count_orig and abs(dr) <= 0.3); exact = bool(dn == 0 and r.sum_mine == r.sum_orig)
        ok_all &= ok
        out[f"{tf}|{cell}"] = dict(n_orig=int(r.count_orig), n_mine=int(r.count_mine), rate_orig=round(r.sum_orig / r.count_orig, 5),
                                   rate_mine=round(r.sum_mine / r.count_mine, 5), exact=exact, ok=ok)
    a = json.load(open(ORIG_C / "a11_a13_r2_20261116/result.json"))["old4_p3"]
    for tf in ("30m", "1h"):
        r, p = E[(E.tf == tf) & (E.cell == "K")], E[(E.tf == tf) & (E.cell == "P3K")]
        ref = a[f"ETH|{tf}"]; ok = bool(len(r) == ref["n"] and abs(r.y.mean() - ref["real"]) < 1e-4 and abs(p.y.mean() - ref["p3"]) < 1e-4)
        ok_all &= ok
        out[f"{tf}|K(old4_p3)"] = dict(n_orig=ref["n"], n_mine=int(len(r)), real_orig=ref["real"], real_mine=round(r.y.mean(), 4), p3_orig=ref["p3"], p3_mine=round(p.y.mean(), 4), ok=ok)
    # 가짜 한 벌(ETH draw 0, 원 run 씨앗) · SMC 15분 2025 이전
    kp = SM.placebo(k, SM.SEED + 0 + 0)
    ev = pd.DataFrame(SM.events(SM.bars(kp, 15), np.random.default_rng(SM.SEED + 0 + 13 + 15), with_p2=False))
    ev = ev[ev.cell.isin(["S3_N5", "S3_N50", "S5"]) & (ev.tk < SM.SPLIT)].groupby("cell").y.agg(["count", "sum"])
    pa = pd.read_parquet(ORIG_F / "smc_ict_meaning_20261115/placebo_agg.parquet")
    pa = pa[(pa.coin == "ETH") & (pa.draw == 0) & (pa.tf == "15m") & ~pa.hh].set_index("cell")
    for c_ in ev.index:
        ok = bool(ev.loc[c_, "count"] == pa.loc[c_, "count"] and ev.loc[c_, "sum"] == pa.loc[c_, "sum"])
        ok_all &= ok
        out[f"placebo0_15m_pre2025|{c_}"] = dict(n_orig=int(pa.loc[c_, "count"]), n_mine=int(ev.loc[c_, "count"]), sum_orig=float(pa.loc[c_, "sum"]), sum_mine=float(ev.loc[c_, "sum"]), ok=ok)
    json.dump(dict(pass_=bool(ok_all), cells=out), open(OUT / "posctrl.json", "w"), ensure_ascii=False, indent=1)
    for k_, v in out.items():
        print(k_, v)
    print("양성 대조", "통과" if ok_all else "실패")


# ── 자체 점검 ────────────────────────────────────────────────────────────────
def selftest() -> None:
    rng = np.random.default_rng(1)
    n = 60 * 24 * 40                                           # 40일 가짜 1분봉(월요일 시작) -- 실자료 아님
    t = ms("2026-01-05") + np.arange(n, dtype=np.int64) * 60_000
    c = 100 * np.exp(np.cumsum(rng.normal(0, 1e-3, n)))
    k = pd.DataFrame({"t": t, "h": c * (1 + np.abs(rng.normal(0, 5e-4, n))), "l": c * (1 - np.abs(rng.normal(0, 5e-4, n))), "c": c, "v": 1.0})
    AV.END = 2**62
    for tf, mnt in (("15m", 15), ("5m", 5)):
        B = SM.bars(k, mnt)
        ref = pd.DataFrame(SM.events(B, np.random.default_rng(0), with_p2=False))
        for N in (5, 50):
            mine = pd.DataFrame(ob_var(B, tf, N, None, False, "x"))
            r = ref[ref.cell == f"S3_N{N}"]
            assert len(mine) == len(r) > 0 and (mine.y.to_numpy() == r.y.to_numpy()).all() and (mine.tk.to_numpy() == r.tk.to_numpy()).all(), (tf, N)
    for tf, mnt in (("30m", 30), ("1h", 60)):
        B = dict(SM.bars(k, mnt), tf=tf)
        r = pd.DataFrame(A13.a13_events(B, True)); m = pd.DataFrame(a13_var(B, tf, 50, 2, True, "x"))
        for a_, b_ in (("K", "x"), ("P3K", "x|C")):
            assert (r[r.cell == a_].y.to_numpy() == m[m.cell == b_].y.to_numpy()).all() and len(m[m.cell == b_]) > 0, (tf, a_)
    B5 = SM.bars(k, 5)
    r = pd.DataFrame(OR.events2(B5, True))
    for L in (30, 60):
        m = pd.DataFrame(orb_var(B5, L, 1, True, "x"))
        for a_, b_ in (("NY", "x"), ("P3", "x|C")):
            ra, mb = r[(r.cell == a_) & (r.tf == f"or{L}")], m[m.cell == b_]
            assert len(ra) == len(mb) > 0 and (ra.y.to_numpy() == mb.y.to_numpy()).all(), (L, a_)
    VO.ONLY = ("E1",); B = SM.bars(k, 60); ev = VO.events(B, "1h", True); VO.ONLY = None
    m = pd.DataFrame(cci_var(B, "1h", 100, True, "x"))
    for a_, b_ in (("E1", "x"), ("P3E1", "x|C")):
        ya = np.array([y for c_, y in zip(ev["cell"], ev["y"]) if c_ == a_]); assert len(ya) > 0 and (ya == m[m.cell == b_].y.to_numpy()).all(), a_
    B = SM.bars(k, 15); ev = CA.events(B, "15m", True); m = pd.DataFrame(rsi2_var(B, "15m", 10, 90, 200, True, "x"))
    for a_, b_ in (("G1", "x"), ("P3G1", "x|C")):
        ya = np.array([y for c_, y in zip(ev["cell"], ev["y"]) if c_ == a_]); assert len(ya) > 0 and (ya == m[m.cell == b_].y.to_numpy()).all(), a_
    r = pd.DataFrame(RD.events(k, "15m", 15)); m = pd.DataFrame(rdiv_var(k, 5, 60, True, "x"))
    assert (r[r.cell == "D1"].y.to_numpy() == m[m.cell == "x"].y.to_numpy()).all() and (r[r.cell == "P2D1"].y.to_numpy() == m[m.cell == "x|C"].y.to_numpy()).all()
    assert RD.N == 5 and RD.GAP == 60
    # 판정 트리: 손으로 만든 judge 행
    base = dict(n=100, real=.6, p1=.5, p2=.55, p2_n=100, diff_pp=10.0, lo90_pp=4.0, diff2_pp=5.0, lo90_2_pp=1.0, hi90_2_pp=9.0, half_diff_pp={False: 3, True: 2}, holm=True, p_one=.01, p_rev=.99, m=1)
    assert tree(base, 2, 2)["grade"] == "엄격 통과"
    assert tree(dict(base, n=20), 2, 2)["grade"] == "판정 불가(건수 미달)"
    assert tree(dict(base, diff_pp=0.5, lo90_pp=-0.5, holm=False), 2, 2)["grade"] == "불통과(원 효과보다 작음)"      # hi1 = 1.5 < 원 효과 2
    assert tree(dict(base, diff_pp=1.0, lo90_pp=-6.0, holm=False, diff2_pp=1.0, lo90_2_pp=-5, hi90_2_pp=7), 2, 2)["grade"] == "판정 불가(검정력 부족)"
    assert tree(dict(base, diff_pp=-5.0, lo90_pp=-9.0, diff2_pp=-3, p_rev=.01, half_diff_pp={False: -1, True: -2}, holm=False), 2, 2)["grade"] == "거꾸로"
    print("selftest ok")


if __name__ == "__main__":
    {"build": lambda: build(sys.argv[2], sys.argv[3]), "selftest": selftest, "posctrl": posctrl, "run": run, "score": score}[sys.argv[1]]()
