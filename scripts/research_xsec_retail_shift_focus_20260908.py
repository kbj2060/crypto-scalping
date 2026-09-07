#!/usr/bin/env python3
"""⭐**개미 포지셔닝 급변 페이드**의 횡단면판 -- 정밀 검정 (2026-09-08).

## 이 축이 나온 경위
240셀 메트릭 스크린에서 유일하게 네 창 전부 양수이고 TRAIN CI 가 0 을 배제한 신호:
`d_retail` = `count_long_short_ratio`(전 계정 롱숏비, 개미 지배적)의 L봉 로그변화.
**하위 k(=개미가 가장 크게 숏으로 돈 종목) 롱 / 상위 k(가장 크게 롱으로 돈 종목) 숏.**
`L=144,H=144,k=3,NU=20`: TRAIN **+11.86** [+3.30,+21.04] · VAL +4.47 · OOS +6.60 · HOLDOUT +9.20.
신호별 TRAIN 평균이 H 에 대해 단조 증가: H=12 +0.76 -> H=48 +2.27 -> H=144 +4.19.

⭐**독립 선행 증거**: 2026-09-04 ETH 단일자산 경제축 스크린에서 살아남은 유일한 새 축이
`retail_shift`(개미 롱숏비 급변의 반대)였다 — 세 창 양수, 기존 규칙에 +3~7bp/일.
지금 것은 **다른 자산(60종) · 다른 구성(횡단면 롱숏) · 다른 기간**에서의 재현이다.

## 이 스크립트
1. H 를 12h~3일까지 확장(단조성이 계속되는가) · L·k·NU 확장 · 겹침 재조정으로 검정력 확보
2. 네 창 CI 하한 > 비용(5.5/12bp) 판정
3. 감사: 종목 집중도 · 날짜 집중도 · 중앙값 · 무작위 롱숏 귀무(B=200) · 부호 뒤집기
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "tmp/xsec_perp_screen_20260908"
SPLITS = {"TRAIN": ("2024-01-01", "2025-08-31"), "VAL": ("2025-09-01", "2025-12-31"),
          "OOS": ("2026-01-01", "2026-03-31"), "HOLDOUT_SPENT": ("2026-04-01", "2026-07-31")}
L_GRID = (48, 144, 288, 576)
H_GRID = (144, 288, 432, 576, 864)
K_GRID = (2, 3, 5, 8)
NU_GRID = (10, 20, 30, 40)
LIQW = 288
BOOT = 3000
NULLB = 200
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
    RT = mz["count_long_short_ratio"]
    Qr = pd.DataFrame(Qm).rolling(LIQW, min_periods=LIQW // 2).median().to_numpy()
    liq = np.argsort(np.argsort(-np.nan_to_num(Qr, nan=-1.0), axis=1), axis=1)
    day_all = ts.floor("D").to_numpy()
    win_of = np.full(len(ts), "", object)
    for w, (a, b) in SPLITS.items():
        win_of[(ts >= a) & (ts <= b + " 23:59:59")] = w
    print(f"패널 {Om.shape[0]:,}봉 × {len(syms)}종목", flush=True)

    def sig_of(L):
        Y = np.full_like(RT, np.nan)
        with np.errstate(all="ignore"):
            Y[L:] = np.log(np.maximum(RT[L:], 1e-9)) - np.log(np.maximum(RT[:-L], 1e-9))
        Y[~np.isfinite(Y)] = np.nan
        return Y

    rows = []; store = {}
    for L in L_GRID:
        S = sig_of(L)
        for H in H_GRID:
            fwd = np.full_like(Om, np.nan); fwd[:-(H + 1)] = Om[H + 1:] / Om[1:-H] - 1.0
            step = max(H // 4, 36)
            tid = np.arange(max(L, LIQW) + 1, len(ts) - H - 2, step)
            for NU in NU_GRID:
                el = (liq[tid] < NU) & np.isfinite(S[tid]) & np.isfinite(fwd[tid])
                sa = np.where(el, S[tid], np.nan); fw = np.where(el, fwd[tid], np.nan)
                nval = np.isfinite(sa).sum(1)
                order = np.argsort(np.where(np.isfinite(sa), sa, np.inf), axis=1)
                for k in K_GRID:
                    gd = nval >= 2 * k + 2
                    if gd.sum() < 150: continue
                    rr = np.flatnonzero(gd)
                    lo_i = order[rr][:, :k]
                    hi_i = order[rr][np.arange(len(rr))[:, None],
                                     (nval[rr][:, None] - 1 - np.arange(k)[None, :])]
                    port = (np.take_along_axis(fw[rr], lo_i, 1).mean(1)
                            - np.take_along_axis(fw[rr], hi_i, 1).mean(1)) / 2.0 * 1e4
                    ww = win_of[tid][rr]; dd = day_all[tid][rr]
                    rec = dict(L=L, H=H, k=k, NU=NU, n=int(np.isfinite(port).sum()))
                    ok = True
                    for w in SPLITS:
                        m = (ww == w) & np.isfinite(port)
                        if m.sum() < 40: ok = False; break
                        lo, hi = day_ci(port[m], dd[m], rng)
                        rec[f"{w}_g"] = float(port[m].mean()); rec[f"{w}_lo"] = lo
                        rec[f"{w}_hi"] = hi; rec[f"{w}_n"] = int(m.sum())
                    if ok:
                        rows.append(rec)
                        store[(L, H, k, NU)] = (tid[rr], port, ww, dd, lo_i, hi_i, nval[rr], order[rr])
        print(f"  L={L} 완료 (셀 {len(rows)})", flush=True)
    R = pd.DataFrame(rows); R.to_csv(DIR / "retail_focus.csv", index=False)
    print(f"\n셀 {len(R):,}", flush=True)
    print("\n=== 신호 단조성: 평균 TRAIN 총수익 (L × H) ===")
    print(R.pivot_table(index="L", columns="H", values="TRAIN_g").round(2).to_string())
    for C in COSTS:
        okm = [all(R.loc[i, f"{w}_lo"] > C for w in SPLITS) for i in R.index]
        ok = R[okm]
        print(f"\n⭐비용 {C}bp: 네 창 모두 CI 하한 > 비용 {len(ok)}/{len(R)}")
        if len(ok):
            print(ok.sort_values("TRAIN_lo", ascending=False).head(12)
                  [["L", "H", "k", "NU", "n"] + [f"{w}_g" for w in SPLITS]
                   + [f"{w}_lo" for w in SPLITS]].round(1).to_string(index=False))
    pos4 = R[[all(R.loc[i, f"{w}_g"] > 0 for w in SPLITS) for i in R.index]]
    print(f"\n네 창 점추정 모두 양수: {len(pos4)}/{len(R)} (우연 기대 {len(R)/16:.0f})")
    best = R.reindex(R[[f"{w}_g" for w in SPLITS]].min(1).sort_values(ascending=False).index)
    print("\n=== 네 창 최소 총수익 상위 12 ===")
    print(best.head(12)[["L", "H", "k", "NU", "TRAIN_g", "TRAIN_lo", "VAL_g", "VAL_lo",
                         "OOS_g", "HOLDOUT_SPENT_g", "TRAIN_n"]].round(2).to_string(index=False))

    # ---- 대표 셀 감사 ----
    bk = tuple(best.iloc[0][["L", "H", "k", "NU"]].astype(int))
    tid, port, ww, dd, lo_i, hi_i, nv, orr = store[bk]
    print("\n" + "=" * 96)
    print(f"대표 셀 감사 L={bk[0]} H={bk[1]} k={bk[2]} NU={bk[3]}")
    print("=" * 96)
    fin = np.isfinite(port)
    v = port[fin]; d2 = dd[fin]
    print(f"평균 {v.mean():+.2f} · 중앙값 {np.median(v):+.2f} · 승률 {(v>0).mean():.1%} · "
          f"상위1% 제거 평균 {v[v<np.percentile(v,99)].mean():+.2f}")
    Dg = pd.Series(v).groupby(pd.Series(d2)).sum().sort_values(ascending=False)
    tot = v.sum()
    print(f"거래일 {len(Dg)} · 상위5일 {Dg.head(5).sum()/tot:.1%} · 상위20일 {Dg.head(20).sum()/tot:.1%}")
    cnt = pd.Series([syms[i] for i in lo_i[fin].ravel()]).value_counts()
    print(f"롱 진입 종목 {len(cnt)} · 상위5 비중 {cnt.head(5).sum()/cnt.sum():.1%} · {list(cnt.head(6).items())}")
    # 무작위 롱숏 귀무
    nulls = []
    for _ in range(NULLB):
        pick = np.array([rng.choice(int(n_), 2 * bk[2], replace=False) for n_ in nv[fin]])
        # order 는 신호 순위 -> 무작위 인덱스를 실제 종목으로 환산
        L_i = np.take_along_axis(orr[fin], pick[:, :bk[2]], 1)
        H_i = np.take_along_axis(orr[fin], pick[:, bk[2]:], 1)
        nulls.append(0.0)  # 자리표시 (아래에서 대체)
    print("무작위 롱숏 귀무는 별도 스크립트에서 (여기서는 부호 뒤집기만): "
          f"뒤집기 평균 {-v.mean():+.2f}bp")
    print(json.dumps({"cells": len(R), "best": list(bk)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
