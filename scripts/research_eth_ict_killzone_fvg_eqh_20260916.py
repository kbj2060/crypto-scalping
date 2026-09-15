#!/usr/bin/env python3
"""인스타 차트의 ICT 3종 — 킬존+피벗 · FVG · Equal Highs/Lows (2026-09-16).

사용자가 `velotradesnq` 릴스 스샷을 주며 *"이 그림에 나오는 3개 전략에 대해 연구해서 테스트해줘"*.

선행 영수증(이미 닫힌 것들, docs/eth_project_wide_idea_map_20260824.md:100):
  · 킬존류 세션 타이밍 CLOSED(비용게이트) · Po3/Judas OOS 붕괴 · FVG 터치 0.88x · iFVG 0.48x
  · liquidity_sweep 은 2026-09-14 에 1,741일로 CLOSED(메커니즘 필터 0/14, 그 중 **터치수 축**이
    Equal Highs/Lows 개념과 같다)
그럼에도 재검정하는 이유: 위 숫자는 전부 **1년 창 · lift 지표**였고, 09-14 가 "1년 창의 성질"을
실증했다. 여기서는 **4.8년 · gross bp · 같은측면 무작위 진입 귀무 · 비겹침 블록 t** 로 다시 건다.
또 (a) 플로어 피벗 S1/R1, (b) EQH/EQL 을 plain sweep 의 **진부분집합**으로 만든 controlled 비교,
(c) 세 전략의 **교집합**(릴스가 실제로 주장하는 형태)은 이 저장소에서 미측정이다.

재구현 금지: 검정 하네스와 plain sweep 정의는 09-14 스크립트 / 라이브 compute_signals 에서 임포트.
새로 쓰는 것은 킬존·피벗·FVG·EQH 발동식뿐.

사전등록 판정(실행 전 고정):
  PASS = 블록초과 > 5.52bp(양다리 peg) AND 초과 CI95 가 0 배제 AND |t_블록초과| >= 2
         AND 전·후반 부호 일치 AND BH-FDR q < 0.10
  그 외 CLOSED.

출력: tmp/eth_ict3_20260916/*.csv
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))

from research_eth_liquidity_sweep_block_independence_20260914 import (  # noqa: E402
    gross_bp, thin_nonoverlap, same_side_null, day_cluster_boot, load, CSV, BTC)

OUT = ROOT / "tmp/eth_ict3_20260916"
HORIZONS = [12, 24, 48, 144]
WARMUP = 900
FVG_LIFE = 48          # 존 수명 — 기존 FVG/OB 연구 관례(eth_ict2022 문서)와 동일
EQ_TOL_ATR = 0.10      # "같은 값" 허용오차 = 0.1 x atr_pct
COST_PEG = 5.52        # 양다리 peg 실측
RNG = np.random.default_rng(20260916)

# 킬존 — 스샷의 NY 현지시각 그대로(America/New_York, DST 자동)
KZ = {"london": (2 * 60, 5 * 60), "ny_am": (9 * 60 + 30, 11 * 60), "ny_pm": (13 * 60 + 30, 16 * 60)}


# ---------------------------------------------------------------- 통계 (빠른 판, 자체점검으로 동치 확인)
def day_boot(vals: np.ndarray, days: np.ndarray, B: int = 600) -> tuple[float, float, float]:
    """일 군집 부트스트랩 — (CI lo, CI hi, 양측 p). day_cluster_boot 과 같은 재표집을
    일별 합/개수로 벡터화한 것(평균은 sum(합)/sum(개수) 로 동일)."""
    d, inv = np.unique(days, return_inverse=True)
    s = np.bincount(inv, weights=vals, minlength=len(d))
    c = np.bincount(inv, minlength=len(d)).astype(float)
    pick = RNG.integers(0, len(d), size=(B, len(d)))
    means = s[pick].sum(1) / c[pick].sum(1)
    p = 2.0 * min((means <= 0).mean(), (means >= 0).mean())
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float(min(p, 1.0))


def null_hit(op, cl, lo: int, hi: int, H: int, long: bool) -> float:
    """같은측면 무작위 진입의 적중률 — "맞추기는 하는가" 를 볼 때의 기저선."""
    return float((gross_bp(op, cl, np.arange(lo, hi + 1, dtype=np.int64), H, long) > 0).mean())


def null_mean(op, cl, lo: int, hi: int, H: int, long: bool) -> float:
    """같은측면 무작위 진입 귀무의 평균 = 창 전체 봉의 gross 평균(표집의 기댓값 그 자체).
    same_side_null 을 400회 돌린 것과 같은 값을 O(n) 에 정확히 준다 — selftest 로 확인."""
    return float(gross_bp(op, cl, np.arange(lo, hi + 1, dtype=np.int64), H, long).mean())


def bh_fdr(p: np.ndarray) -> np.ndarray:
    o = np.argsort(p); q = np.empty_like(p)
    n = len(p); r = np.arange(1, n + 1)
    q[o] = np.minimum.accumulate((p[o] * n / r)[::-1])[::-1]
    return np.minimum(q, 1.0)


# ---------------------------------------------------------------- 신호 (사전등록, 변형 탐색 금지)
def killzone_mask(ts: pd.Series) -> dict[str, np.ndarray]:
    ny = ts.dt.tz_localize("UTC").dt.tz_convert("America/New_York")
    mins = (ny.dt.hour * 60 + ny.dt.minute).to_numpy()
    out = {k: (mins >= a) & (mins < b) for k, (a, b) in KZ.items()}
    out["any"] = out["london"] | out["ny_am"] | out["ny_pm"]
    return out


def prior_day_levels(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """직전 UTC 일의 OHLC 로 만든 플로어 피벗 + PDH/PDL. shift(1) 로 인과 보장."""
    d = df["timestamp"].dt.floor("D")
    g = df.groupby(d).agg(h=("high", "max"), l=("low", "min"), c=("close", "last"))
    pp = (g.h + g.l + g.c) / 3.0
    lv = pd.DataFrame({"pp": pp, "r1": 2 * pp - g.l, "s1": 2 * pp - g.h, "pdh": g.h, "pdl": g.l}).shift(1)
    m = d.map(lv.to_dict("index"))
    return {k: np.array([np.nan if not isinstance(x, dict) else x[k] for x in m], float)
            for k in ("pp", "r1", "s1", "pdh", "pdl")}


def level_reclaim(high, low, close, lvl, *, below: bool) -> np.ndarray:
    """레벨을 관통했다가 종가로 되찾는 봉. below=True 면 지지(S1/PDL) 되찾기 = 롱 발동."""
    prev = np.roll(close, 1); prev[0] = np.nan
    if below:
        return (low <= lvl) & (close > lvl) & (prev > lvl)
    return (high >= lvl) & (close < lvl) & (prev < lvl)


def fvg_touch(high, low, close, atr_pct, min_gap_atr: float) -> tuple[np.ndarray, np.ndarray]:
    """3봉 Fair Value Gap 의 미티게이션 터치. (강세=롱, 약세=숏) 불리언."""
    n = len(high)
    bull = np.zeros(n, bool); bear = np.zeros(n, bool)
    for sign in (+1, -1):
        if sign > 0:                                   # 강세 갭: high[g-2] < low[g], 존=[high[g-2], low[g]]
            bot, top = high[:-2], low[2:]
        else:                                          # 약세 갭: low[g-2] > high[g], 존=[high[g], low[g-2]]
            bot, top = high[2:], low[:-2]
        gidx = np.flatnonzero(top > bot) + 2
        if len(gidx) == 0: continue
        gb, gt = bot[gidx - 2], top[gidx - 2]
        keep = (gt - gb) / np.maximum(close[gidx], 1e-12) >= min_gap_atr * atr_pct[gidx]
        gidx, gb, gt = gidx[keep], gb[keep], gt[keep]
        fire = bull if sign > 0 else bear
        for g, b, t in zip(gidx, gb, gt):
            j0, j1 = g + 1, min(g + 1 + FVG_LIFE, n)
            if j0 >= j1: continue
            c = close[j0:j1]
            bad = np.flatnonzero(c < b) if sign > 0 else np.flatnonzero(c > t)  # 존 완전 무효
            end = j0 + (bad[0] if len(bad) else (j1 - j0))
            if end <= j0: continue
            sl = slice(j0, end)
            fire[sl] |= (low[sl] <= t) & (high[sl] >= b)
    return bull, bear


def equal_level_count(arr: np.ndarray, lvl: np.ndarray, tol: np.ndarray, win: int, *, upper: bool) -> np.ndarray:
    """직전 win 봉 중 레벨을 tol 안에서 태그한 봉 수. 2 이상이면 Equal Highs/Lows."""
    from numpy.lib.stride_tricks import sliding_window_view
    n = len(arr)
    cnt = np.zeros(n, np.int32)
    w = sliding_window_view(arr, win)              # w[k] = arr[k:k+win]  -> 봉 j 의 직전창은 w[j-win]
    thr = lvl * (1.0 - tol) if upper else lvl * (1.0 + tol)
    for a in range(win, n, 50_000):                 # 청크 — 전체 복사 시 ~190MB
        v = np.arange(a, min(a + 50_000, n))
        sub = w[v - win]
        cnt[v] = ((sub >= thr[v, None]) if upper else (sub <= thr[v, None])).sum(1)
    return cnt


# ---------------------------------------------------------------- 자체점검
def selftest() -> None:
    # day_boot ≡ day_cluster_boot (같은 재표집의 벡터화) — 평균 일치, CI 폭 유사
    v = RNG.normal(3, 10, 400); dd = np.repeat(np.arange(40), 10)
    a = day_cluster_boot(v, dd, B=2000); b = day_boot(v, dd, B=2000)
    assert abs((a[0] + a[1]) / 2 - (b[0] + b[1]) / 2) < 1.0, (a, b)
    assert abs((a[1] - a[0]) - (b[1] - b[0])) < 1.5, (a, b)
    # level_reclaim: 아래로 뚫었다 되찾으면 롱 발동, 그냥 뚫고 마감하면 미발동
    hi = np.array([10., 10., 10.]); lo = np.array([9., 8., 8.]); cl = np.array([9.5, 9.2, 8.5])
    lv = np.full(3, 9.0)
    assert list(level_reclaim(hi, lo, cl, lv, below=True)) == [False, True, False]
    # fvg_touch: 강세갭 [10,12] 형성 후 되돌림 터치 1건
    h = np.array([10., 11., 13., 13., 13., 12.5]); l = np.array([9., 10.5, 12., 12.5, 12.4, 9.9])
    c = np.array([9.5, 11., 12.5, 12.8, 12.6, 10.5]); at = np.zeros(6)
    bull, bear = fvg_touch(h, l, c, at, 0.0)
    assert bull[5] and not bull[3], (bull, bear)
    # 약세갭 미러: low[0]=12 > high[2]=10 -> 존 [10,12], bar5 에서 되돌림 터치
    h2 = np.array([13., 12., 10., 9.8, 9.9, 10.5]); l2 = np.array([12., 10.5, 9., 9., 9.2, 9.8])
    c2_ = np.array([12.5, 11., 9.5, 9.4, 9.5, 10.2])
    b2, br2 = fvg_touch(h2, l2, c2_, at, 0.0)
    assert br2[5] and not br2[3] and not b2.any(), (b2, br2)
    # null_mean ≡ same_side_null 의 기댓값 (MC 오차 안)
    rn = RNG.normal(100, 1, 3000).cumsum() / 100 + 100
    o2, c2 = rn, rn * 1.0001
    nm = null_mean(o2, c2, 10, 2800, 12, True)
    ss = same_side_null(o2, c2, 300, 12, True, 10, 2800)
    assert abs(nm - ss.mean()) < 3 * ss.std(ddof=1), (nm, ss.mean(), ss.std(ddof=1))
    # equal_level_count: 직전 4봉 중 레벨 10 을 tol 0 으로 태그한 봉 = 2
    a2 = np.array([10., 9., 10., 9., 0.])
    cnt = equal_level_count(a2, np.full(5, 10.0), np.zeros(5), 4, upper=True)
    assert cnt[4] == 2, cnt
    # bh_fdr 단조 + 상한
    q = bh_fdr(np.array([0.001, 0.02, 0.5, 0.9]))
    assert np.all(np.diff(q) >= -1e-12) and q.max() <= 1.0
    print("selftest OK")


# ---------------------------------------------------------------- 실행
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        selftest(); return 0
    selftest()
    OUT.mkdir(parents=True, exist_ok=True)
    import live_evidence_signal_dashboard_20260823 as EV  # noqa: E402

    print("[1/4] CSV …", flush=True)
    kl = load(CSV); btc = load(BTC) if BTC.exists() else None
    sig = EV.compute_signals(kl, btc_df=btc, funding_df=None)
    ts = sig["timestamp"]
    op, cl = sig["open"].to_numpy(float), sig["close"].to_numpy(float)
    hi, lo_ = sig["high"].to_numpy(float), sig["low"].to_numpy(float)
    atr = sig["atr_pct"].to_numpy(float)
    day = ts.dt.floor("D").astype("int64").to_numpy()
    half = ts.iloc[len(sig) // 2]
    n = len(sig); lo, hi_i = WARMUP, n - max(HORIZONS) - 2
    span = (ts.iloc[hi_i] - ts.iloc[lo]).total_seconds() / 86400
    print(f"  {n:,}봉  {ts.iloc[0]} ~ {ts.iloc[-1]}  평가 {span:.0f}일", flush=True)

    print("[2/4] 신호 …", flush=True)
    kz = killzone_mask(ts)
    lv = prior_day_levels(sig)
    piv_s1 = level_reclaim(hi, lo_, cl, lv["s1"], below=True)
    piv_r1 = level_reclaim(hi, lo_, cl, lv["r1"], below=False)
    piv_pdl = level_reclaim(hi, lo_, cl, lv["pdl"], below=True)
    piv_pdh = level_reclaim(hi, lo_, cl, lv["pdh"], below=False)
    fvg = {m: fvg_touch(hi, lo_, cl, atr, m) for m in (0.0, 0.5, 1.0)}
    kz_open = {s: kz[s] & ~np.roll(kz[s], 1) for s in ("london", "ny_am", "ny_pm")}
    kz_first = kz_open["london"] | kz_open["ny_am"] | kz_open["ny_pm"]

    # plain sweep(라이브 정의) 와 그 진부분집합인 EQH/EQL
    sw_top = sig["top_liquidity_sweep"].fillna(False).to_numpy(bool)
    sw_bot = sig["bottom_liquidity_sweep"].fillna(False).to_numpy(bool)
    W = EV.SWEEP_LOOKBACK
    lvl_hi = sig["high"].rolling(W, min_periods=W).max().shift(1).to_numpy(float)
    lvl_lo = sig["low"].rolling(W, min_periods=W).min().shift(1).to_numpy(float)
    tol = np.nan_to_num(atr) * EQ_TOL_ATR
    cnt_hi = equal_level_count(hi, lvl_hi, tol, W, upper=True)
    cnt_lo = equal_level_count(lo_, lvl_lo, tol, W, upper=False)

    ARMS: list[tuple[str, np.ndarray, np.ndarray]] = [
        # (이름, 롱(바닥) 발동, 숏(천장) 발동)
        ("1_kz_open_only", kz_first, kz_first),                       # 킬존 단독(순수 세션 타이밍)
        ("1_pivot_s1r1", piv_s1, piv_r1),                             # 피벗 단독
        ("1_pivot_pdhl", piv_pdl, piv_pdh),                           # 전일고저 단독
        ("1_kz_x_pivot_s1r1", piv_s1 & kz["any"], piv_r1 & kz["any"]),
        ("1_kz_x_pivot_pdhl", piv_pdl & kz["any"], piv_pdh & kz["any"]),
        ("2_fvg_all", fvg[0.0][0], fvg[0.0][1]),
        ("2_fvg_ge05atr", fvg[0.5][0], fvg[0.5][1]),
        ("2_fvg_ge10atr", fvg[1.0][0], fvg[1.0][1]),
        ("2_kz_x_fvg05", fvg[0.5][0] & kz["any"], fvg[0.5][1] & kz["any"]),
        ("3_sweep_plain(대조)", sw_bot, sw_top),
        ("3_eqhl_2touch", sw_bot & (cnt_lo >= 2), sw_top & (cnt_hi >= 2)),
        ("3_eqhl_3touch", sw_bot & (cnt_lo >= 3), sw_top & (cnt_hi >= 3)),
        ("3_kz_x_eqhl", sw_bot & (cnt_lo >= 2) & kz["any"], sw_top & (cnt_hi >= 2) & kz["any"]),
        ("4_fvg_x_eqhl", sw_bot & (cnt_lo >= 2) & fvg[0.0][0], sw_top & (cnt_hi >= 2) & fvg[0.0][1]),
        ("4_all3_kz_fvg_eqhl", sw_bot & (cnt_lo >= 2) & fvg[0.0][0] & kz["any"],
                               sw_top & (cnt_hi >= 2) & fvg[0.0][1] & kz["any"]),
        ("4_all3_plus_pivot", sw_bot & (cnt_lo >= 2) & fvg[0.0][0] & kz["any"] & (piv_s1 | piv_pdl),
                              sw_top & (cnt_hi >= 2) & fvg[0.0][1] & kz["any"] & (piv_r1 | piv_pdh)),
    ]

    print("[3/4] 검정 …", flush=True)
    NULL = {(H, lg): null_mean(op, cl, lo, hi_i, H, lg) for H in HORIZONS for lg in (True, False)}
    NHIT = {(H, lg): null_hit(op, cl, lo, hi_i, H, lg) for H in HORIZONS for lg in (True, False)}
    rows = []
    for name, f_long, f_short in ARMS:
        for side, fire, long in (("bottom", f_long, True), ("top", f_short, False)):
            idx = np.flatnonzero(np.nan_to_num(fire).astype(bool))
            idx = idx[(idx >= lo) & (idx <= hi_i)]
            if len(idx) < 30:
                rows.append(dict(arm=name, side=side, H=0, n_all=len(idx), note="표본부족")); continue
            for H in HORIZONS:
                g = gross_bp(op, cl, idx, H, long)
                nul = NULL[(H, long)]
                kept = thin_nonoverlap(idx, H)
                ex_k = gross_bp(op, cl, kept, H, long) - nul
                t_ex = float(ex_k.mean() / (ex_k.std(ddof=1) / np.sqrt(len(ex_k)))) if len(ex_k) > 2 else np.nan
                clo, chi, p = day_boot(g - nul, day[idx])
                h1 = g[ts.iloc[idx].to_numpy() < np.datetime64(half)]
                h2 = g[ts.iloc[idx].to_numpy() >= np.datetime64(half)]
                rows.append(dict(arm=name, side=side, H=H, n_all=len(idx), n_block=len(kept),
                                 per_day=round(len(idx) / span, 2),
                                 gross=round(float(g.mean()), 2), null=round(float(nul), 2),
                                 excess=round(float(g.mean() - nul), 2),
                                 ci_lo=round(clo, 2), ci_hi=round(chi, 2), p=round(p, 4),
                                 excess_block=round(float(ex_k.mean()), 2), t_block=round(t_ex, 2),
                                 h1=round(float(h1.mean()), 2) if len(h1) else np.nan,
                                 h2=round(float(h2.mean()), 2) if len(h2) else np.nan,
                                 hit=round(float((g > 0).mean()), 4),
                                 null_hit=round(NHIT[(H, long)], 4),
                                 hit_edge_pp=round(100 * (float((g > 0).mean()) - NHIT[(H, long)]), 2),
                                 # b 분해 — 적중률이 높아도 이긴 판이 작으면 비용을 못 넘는다
                                 win=round(float(g[g > 0].mean()), 2) if (g > 0).any() else np.nan,
                                 loss=round(float(-g[g <= 0].mean()), 2) if (g <= 0).any() else np.nan,
                                 acc_needed=round(float((-g[g <= 0].mean() + COST_PEG) /
                                                        (g[g > 0].mean() - g[g <= 0].mean())), 4)
                                 if (g > 0).any() and (g <= 0).any() else np.nan))
        print(f"  · {name}", flush=True)

    D = pd.DataFrame(rows)
    ok = D.H > 0
    D.loc[ok, "q"] = bh_fdr(D.loc[ok, "p"].to_numpy())
    D["PASS"] = (ok & (D.excess_block > COST_PEG) & (D.ci_lo > 0) & (D.t_block.abs() >= 2)
                 & (np.sign(D.h1) == np.sign(D.h2)) & (D.q < 0.10))
    D.to_csv(OUT / "cells.csv", index=False)

    print("\n[4/4] 결과 (gross = 비용 차감 전 bp · 초과 = 같은측면 무작위 진입 대비 · 비용선 5.52bp)")
    print("=" * 140)
    hdr = (f"{'팔':<22}{'측':>4}{'H':>5}{'건수':>8}{'건/일':>7}{'블록':>7}{'gross':>8}{'귀무':>8}"
           f"{'초과':>8}{'초과CI95':>18}{'블록초과':>9}{'t':>6}{'q':>7}{'전반':>8}{'후반':>8}{'':>5}")
    print(hdr)
    for r in D.itertuples():
        if r.H == 0:
            print(f"{r.arm:<22}{'바닥' if r.side=='bottom' else '천장':>4}  — 건수 {r.n_all} (표본부족)"); continue
        print(f"{r.arm:<22}{'바닥' if r.side=='bottom' else '천장':>4}{r.H:>5}{r.n_all:>8}{r.per_day:>7.2f}"
              f"{r.n_block:>7}{r.gross:>8.2f}{r.null:>8.2f}{r.excess:>+8.2f}"
              f"{f'[{r.ci_lo:+.2f},{r.ci_hi:+.2f}]':>18}{r.excess_block:>+9.2f}{r.t_block:>6.2f}"
              f"{r.q:>7.3f}{r.h1:>+8.2f}{r.h2:>+8.2f}{'  ★PASS' if r.PASS else ''}")
    print("=" * 140)
    print(json.dumps({"cells": int(ok.sum()), "pass": int(D.PASS.sum()), "days": round(span, 1)},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
