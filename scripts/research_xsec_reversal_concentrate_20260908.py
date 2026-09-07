#!/usr/bin/env python3
"""횡단면 단기 되돌림 **집중** -- 효과를 비용 위로 끌어올릴 수 있는가 (2026-09-08).

## 1차 스크린이 찾은 것 (`research_xsec_perp_reversal_screen_20260908.py`, 840셀)
짧은 L(3~12봉) · 짧은 H(3~12봉)에서 **횡단면 되돌림이 실재**한다:
`L=3,H=6,k=5,NU=59` 총수익 TRAIN **+1.24bp** [+0.88,+1.58] (n=29,202) ·
VAL +0.94 · OOS +2.82 · HOLDOUT +1.18 -- **네 창 전부 같은 부호**.
210셀 중 TRAIN CI 가 0 을 배제한 셀 35, 네 창 부호 동일 55(우연 26).
**그러나 1.24bp 는 비용 7.8bp 의 1/6 이다.** 통계적으로 실재하지만 테이커로는 못 먹는다.

## 이 스크립트의 질문
효과를 **집중**하면 비용을 넘는가? 세 가지 손잡이:
1. `k` 를 5 -> 1~3 으로 (극단만)
2. 진입 문턱: 극단 종목의 |과거수익| 이 THR 이상일 때만 (큰 과잉반응만)
3. 분산 조건: 그 시점 횡단면 표준편차 상위 분위일 때만
비용은 세 가지로 동시 보고: 5.5(양다리 전부 메이커) · 7.8(배포 가정) · 12.0(소형주 현실).
⚠️1차 스크린에서 효과는 NU 가 **클수록**(=소형주 포함) 컸다 -- 비용이 가장 비싼 곳이다.
   따라서 NU=20(상위 유동성)에서도 살아남는지를 별도로 본다.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/xsec_perp_screen_20260908"
SPLITS = {"TRAIN": ("2024-01-01", "2025-08-31"), "VAL": ("2025-09-01", "2025-12-31"),
          "OOS": ("2026-01-01", "2026-03-31"), "HOLDOUT_SPENT": ("2026-04-01", "2026-07-31")}
L_GRID = (1, 2, 3, 6, 12)
H_GRID = (1, 2, 3, 6, 12)
K_GRID = (1, 2, 3, 5)
THR = (0.0, 0.01, 0.02, 0.03)
NU_GRID = (20, 59)
LIQW = 288
BOOT = 2000
SEED = 20260908
COSTS = (5.5, 7.8, 12.0)


def day_ci(v, day, rng, B=BOOT):
    u, inv = np.unique(day, return_inverse=True)
    if len(u) < 8 or len(v) < 8: return (np.nan, np.nan)
    s = np.bincount(inv, weights=v, minlength=len(u)); c = np.bincount(inv, minlength=len(u)).astype(float)
    pick = rng.integers(0, len(u), (B, len(u)))
    o = s[pick].sum(1) / np.maximum(c[pick].sum(1), 1.0)
    return tuple(np.percentile(o, [2.5, 97.5]))


def main() -> int:
    rng = np.random.default_rng(SEED)
    z = np.load(OUT / "panel.npz", allow_pickle=True)
    ts = pd.to_datetime(z["ts"]).to_numpy(); Om = z["O"]; Cm = z["C"]; Qm = z["Q"]
    print(f"[1/2] 패널 {Om.shape[0]:,}봉 × {Om.shape[1]}종목", flush=True)
    Qr = pd.DataFrame(Qm).rolling(LIQW, min_periods=LIQW // 2).median().to_numpy()
    liq = np.argsort(np.argsort(-np.nan_to_num(Qr, nan=-1.0), axis=1), axis=1)
    day_all = pd.Series(ts).dt.floor("D").to_numpy()
    win_of = np.full(len(ts), "", object)
    for w, (a, b) in SPLITS.items():
        win_of[(ts >= np.datetime64(a)) & (ts <= np.datetime64(b + "T23:59:59"))] = w

    print("[2/2] 격자 ...", flush=True)
    rows = []
    for L in L_GRID:
        past = np.full_like(Cm, np.nan); past[L:] = Cm[L:] / Cm[:-L] - 1.0
        for H in H_GRID:
            fwd = np.full_like(Om, np.nan)
            fwd[:-(H + 1)] = Om[H + 1:] / Om[1:-H] - 1.0
            tidx = np.arange(max(L, LIQW) + 1, len(ts) - H - 2, H)
            for NU in NU_GRID:
                elig = (liq[tidx] < NU) & np.isfinite(past[tidx]) & np.isfinite(fwd[tidx])
                pa = np.where(elig, past[tidx], np.nan); fw = np.where(elig, fwd[tidx], np.nan)
                nval = np.isfinite(pa).sum(1)
                disp = np.nanstd(pa, axis=1)
                order = np.argsort(np.where(np.isfinite(pa), pa, np.inf), axis=1)
                for k in K_GRID:
                    base = nval >= 2 * k + 2
                    if base.sum() < 200: continue
                    rr = np.flatnonzero(base)
                    lo_i = order[rr][:, :k]
                    hi_i = order[rr][np.arange(len(rr))[:, None],
                                     (nval[rr][:, None] - 1 - np.arange(k)[None, :])]
                    pl = np.take_along_axis(pa[rr], lo_i, 1).mean(1)   # 하위 k 의 과거수익(음수)
                    ph = np.take_along_axis(pa[rr], hi_i, 1).mean(1)
                    fl = np.take_along_axis(fw[rr], lo_i, 1).mean(1)
                    fh = np.take_along_axis(fw[rr], hi_i, 1).mean(1)
                    port = (fl - fh) / 2.0 * 1e4
                    spread = (ph - pl)                                  # 극단 간 과거수익 격차
                    tt = tidx[rr]; dd = day_all[tt]; ww = win_of[tt]
                    dq = disp[rr]
                    for th in THR:
                        sel = spread >= 2 * th if th > 0 else np.ones(len(rr), bool)
                        for dcond, dname in ((np.ones(len(rr), bool), "all"),
                                             (dq >= np.nanquantile(dq, 0.7), "disp70")):
                            s2 = sel & dcond
                            if s2.sum() < 400: continue
                            rec = dict(L=L, H=H, k=k, NU=NU, thr=th, disp=dname)
                            okall = True
                            for w in SPLITS:
                                m = s2 & (ww == w)
                                if m.sum() < 60: okall = False; break
                                v = port[m]; lo, hi = day_ci(v, dd[m], rng)
                                rec[f"{w}_g"] = float(v.mean()); rec[f"{w}_lo"] = lo
                                rec[f"{w}_hi"] = hi; rec[f"{w}_n"] = int(m.sum())
                            if okall:
                                rec["rebal_day"] = 288.0 / H
                                rows.append(rec)
    R = pd.DataFrame(rows); R.to_csv(OUT / "concentrate.csv", index=False)
    print(f"      셀 {len(R):,}\n", flush=True)
    for C in COSTS:
        ok = R[[all(R.loc[i, f"{w}_lo"] > C for w in SPLITS) for i in R.index]]
        print(f"⭐비용 {C}bp: 네 창 모두 CI 하한 > 비용 인 셀 {len(ok)}/{len(R)}")
        if len(ok):
            o = ok.copy(); o["net_day"] = (o["TRAIN_g"] - C) * o["rebal_day"]
            print(o.sort_values("net_day", ascending=False).head(10)
                  [["L", "H", "k", "NU", "thr", "disp", "TRAIN_g", "VAL_g", "OOS_g",
                    "HOLDOUT_SPENT_g", "net_day"]].round(2).to_string(index=False))
    print("\n=== 총수익 최대 셀 (비용 무관) ===")
    b = R.reindex(R[["TRAIN_g", "VAL_g", "OOS_g", "HOLDOUT_SPENT_g"]].min(1).sort_values(ascending=False).index)
    print(b.head(12)[["L", "H", "k", "NU", "thr", "disp", "TRAIN_g", "TRAIN_lo", "VAL_g",
                      "OOS_g", "HOLDOUT_SPENT_g", "TRAIN_n"]].round(2).to_string(index=False))
    print(json.dumps({"cells": len(R)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
