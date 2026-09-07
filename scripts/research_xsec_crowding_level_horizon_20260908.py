#!/usr/bin/env python3
"""횡단면 **쏠림 수준(level)** × 보유지평 확장 -- 마지막 후보 (2026-09-08).

## 왜
576셀 스크린에서 **수준 신호가 H 에 대해 단조 증가**했다(TRAIN 평균 bp/재조정):

| 신호 | H=3 | H=6 | H=12 | H=48 |
|---|---|---|---|---|
| retail_level | 0.23 | 0.43 | 0.84 | **2.91** |
| tt_count_level | 0.17 | 0.33 | 0.72 | **3.03** |
| tt_pos_level | 0.05 | 0.13 | 0.28 | 0.84 |

수준 신호는 **되돌아보기 창 L 이 없다** -> 격자 차원이 하나 줄어 다중비교가 작다.
방향: 롱숏비가 **낮은**(=숏 쏠림) 종목 롱 / **높은**(롱 쏠림) 종목 숏 = 쏠림 페이드.
H 를 48 -> 864봉(4h -> 3일)까지 늘려 단조성이 계속되는지, 비용 5.5/12bp 를 넘는지 본다.

⚠️`d_retail`(변화량)은 같은 형태의 단조성을 보였다가 겹침 표본 재검정에서 우연 수준으로 붕괴했다
(부록 AF). 그래서 여기서는 **처음부터 겹침 재조정(step=H/4)** 과 **상위 1% 제거·일 집중도 감사**를
같이 낸다.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "tmp/xsec_perp_screen_20260908"
SPLITS = {"TRAIN": ("2024-01-01", "2025-08-31"), "VAL": ("2025-09-01", "2025-12-31"),
          "OOS": ("2026-01-01", "2026-03-31"), "HOLDOUT_SPENT": ("2026-04-01", "2026-07-31")}
H_GRID = (48, 144, 288, 576, 864)
K_GRID = (3, 5, 8)
NU_GRID = (20, 40)
LIQW = 288
BOOT = 3000
SEED = 20260908
COSTS = (5.5, 12.0)


def day_ci(v, day, rng, B=BOOT):
    u, inv = np.unique(day, return_inverse=True)
    if len(u) < 8 or len(v) < 8: return (np.nan, np.nan)
    s = np.bincount(inv, weights=v, minlength=len(u)); c = np.bincount(inv, minlength=len(u)).astype(float)
    pick = rng.integers(0, len(u), (B, len(u)))
    return tuple(np.percentile(s[pick].sum(1) / np.maximum(c[pick].sum(1), 1.0), [2.5, 97.5]))


def main() -> int:
    rng = np.random.default_rng(SEED)
    z = np.load(DIR / "panel.npz", allow_pickle=True)
    ts = pd.to_datetime(z["ts"]); Om = z["O"]; Qm = z["Q"]; syms = list(z["syms"])
    mz = np.load(DIR / "metrics_panel.npz", allow_pickle=True)
    lg = lambda X: np.where(X > 0, np.log(np.maximum(X, 1e-9)), np.nan)
    SIG = {"retail_level": lg(mz["count_long_short_ratio"]),
           "tt_count_level": lg(mz["count_toptrader_long_short_ratio"]),
           "tt_pos_level": lg(mz["sum_toptrader_long_short_ratio"])}
    Qr = pd.DataFrame(Qm).rolling(LIQW, min_periods=LIQW // 2).median().to_numpy()
    liq = np.argsort(np.argsort(-np.nan_to_num(Qr, nan=-1.0), axis=1), axis=1)
    day_all = ts.floor("D").to_numpy()
    win_of = np.full(len(ts), "", object)
    for w, (a, b) in SPLITS.items():
        win_of[(ts >= a) & (ts <= b + " 23:59:59")] = w
    print(f"패널 {Om.shape[0]:,}봉 × {len(syms)}종목", flush=True)

    rows = []; store = {}
    for H in H_GRID:
        fwd = np.full_like(Om, np.nan); fwd[:-(H + 1)] = Om[H + 1:] / Om[1:-H] - 1.0
        tid = np.arange(LIQW + 1, len(ts) - H - 2, max(H // 4, 12))
        for sname, S in SIG.items():
            for NU in NU_GRID:
                el = (liq[tid] < NU) & np.isfinite(S[tid]) & np.isfinite(fwd[tid])
                sa = np.where(el, S[tid], np.nan); fw = np.where(el, fwd[tid], np.nan)
                nval = np.isfinite(sa).sum(1)
                order = np.argsort(np.where(np.isfinite(sa), sa, np.inf), axis=1)
                for k in K_GRID:
                    gd = nval >= 2 * k + 2
                    if gd.sum() < 200: continue
                    rr = np.flatnonzero(gd)
                    lo_i = order[rr][:, :k]
                    hi_i = order[rr][np.arange(len(rr))[:, None],
                                     (nval[rr][:, None] - 1 - np.arange(k)[None, :])]
                    port = (np.take_along_axis(fw[rr], lo_i, 1).mean(1)
                            - np.take_along_axis(fw[rr], hi_i, 1).mean(1)) / 2.0 * 1e4
                    ww = win_of[tid][rr]; dd = day_all[tid][rr]
                    rec = dict(sig=sname, H=H, k=k, NU=NU)
                    ok = True
                    for w in SPLITS:
                        m = (ww == w) & np.isfinite(port)
                        if m.sum() < 60: ok = False; break
                        lo, hi = day_ci(port[m], dd[m], rng)
                        rec[f"{w}_g"] = float(port[m].mean()); rec[f"{w}_lo"] = lo
                        rec[f"{w}_hi"] = hi; rec[f"{w}_n"] = int(m.sum())
                    if ok:
                        rows.append(rec); store[(sname, H, k, NU)] = (port, ww, dd, lo_i, tid[rr])
        print(f"  H={H} 완료 (셀 {len(rows)})", flush=True)
    R = pd.DataFrame(rows); R.to_csv(DIR / "crowding_level.csv", index=False)
    print(f"\n셀 {len(R)}\n", flush=True)
    print("=== 단조성: 신호 × H 의 TRAIN 평균 총수익 ===")
    print(R.pivot_table(index="sig", columns="H", values="TRAIN_g").round(2).to_string())
    print("\n=== 네 창 평균 (신호 × H) ===")
    R["mean4"] = R[[f"{w}_g" for w in SPLITS]].mean(1)
    print(R.pivot_table(index="sig", columns="H", values="mean4").round(2).to_string())
    for C in COSTS:
        ok = R[[all(R.loc[i, f"{w}_lo"] > C for w in SPLITS) for i in R.index]]
        print(f"\n⭐비용 {C}bp 네 창 CI 통과: {len(ok)}/{len(R)}")
        if len(ok):
            print(ok[["sig", "H", "k", "NU"] + [f"{w}_g" for w in SPLITS]
                    + [f"{w}_lo" for w in SPLITS]].round(1).to_string(index=False))
    pos4 = R[[all(R.loc[i, f"{w}_g"] > 0 for w in SPLITS) for i in R.index]]
    print(f"\n네 창 점추정 양수 {len(pos4)}/{len(R)} (우연 기대 {len(R)/16:.1f})")
    best = R.reindex(R[[f"{w}_g" for w in SPLITS]].min(1).sort_values(ascending=False).index)
    print("\n=== 네 창 최소 상위 10 ===")
    print(best.head(10)[["sig", "H", "k", "NU", "TRAIN_g", "TRAIN_lo", "VAL_g", "VAL_lo",
                         "OOS_g", "OOS_lo", "HOLDOUT_SPENT_g", "TRAIN_n"]].round(2).to_string(index=False))
    bk = (best.iloc[0]["sig"], int(best.iloc[0]["H"]), int(best.iloc[0]["k"]), int(best.iloc[0]["NU"]))
    port, ww, dd, lo_i, tt = store[bk]
    fin = np.isfinite(port); v = port[fin]; d2 = dd[fin]
    print(f"\n=== 대표 셀 감사 {bk} ===")
    print(f"평균 {v.mean():+.2f} · 중앙값 {np.median(v):+.2f} · 승률 {(v>0).mean():.1%} · "
          f"상위1% 제거 평균 {v[v<np.percentile(v,99)].mean():+.2f}")
    Dg = pd.Series(v).groupby(pd.Series(d2)).sum().sort_values(ascending=False)
    print(f"거래일 {len(Dg)} · 상위5일 {Dg.head(5).sum()/v.sum():.1%} · 상위20일 {Dg.head(20).sum()/v.sum():.1%}")
    cnt = pd.Series([syms[i] for i in lo_i[fin].ravel()]).value_counts()
    print(f"롱 종목 {len(cnt)} · 상위5 비중 {cnt.head(5).sum()/cnt.sum():.1%} · {list(cnt.head(5).items())}")
    print(json.dumps({"cells": len(R)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
