#!/usr/bin/env python3
"""횡단면 **극단 이탈 사건**(문턱 4~18%): 큰 상대 격차가 벌어진 순간만 진입 (2026-09-08).

## 앞 단계
- 무조건 재조정(부록 AB): 되돌림 실재하나 총수익 **1.24bp** << 비용 5.5~12bp.
- 집중(부록 AC, 960셀): k=1 + 문턱 0.02~0.03 으로 총수익 **3.0~3.7bp** 까지 오르고 네 창 모두 양수.
  그러나 고정 H봉 격자 표집이라 n 이 911~6,828 로 작아 TRAIN CI 하한이 음수.
  ⭐특히 `L=6,H=12,k=1,NU=20,thr=0.02` 는 TRAIN +3.28 · VAL +28.79 · OOS +9.81 · HOLDOUT +15.96
  (유동성 상위 20 종목만 쓴 셀) -- **검정력만 있으면 판정 가능한 형태**다.

## 이 스크립트: 검정력 확보
고정 격자가 아니라 **매 봉 스캔**해서 조건을 만족하는 순간에만 진입한다(사건 기반).
같은 종목이 같은 움직임으로 반복 진입하지 않도록 **종목별 H봉 쿨다운**. 겹침은 허용하고
일군집 CI 로 종속성을 처리한다. 이렇게 n 이 한 자릿수 배 늘어난다.

진입 `open[t+1]` · 청산 `open[t+1+H]` · 최대 하락 1종목 롱 / 최대 상승 1종목 숏 ·
단위명목당 bp · 비용 5.5(전메이커) / 7.8(배포) / 12(소형주).
판정: **네 창 모두 CI 하한 > 비용**.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/xsec_perp_screen_20260908"
SPLITS = {"TRAIN": ("2024-01-01", "2025-08-31"), "VAL": ("2025-09-01", "2025-12-31"),
          "OOS": ("2026-01-01", "2026-03-31"), "HOLDOUT_SPENT": ("2026-04-01", "2026-07-31")}
L_GRID = (3, 6, 12, 24, 48)
H_GRID = (3, 6, 12, 24, 48, 96, 144)
NU_GRID = (20, 40, 60)
THR = (0.04, 0.06, 0.08, 0.12, 0.18)
LIQW = 288
BOOT = 3000
SEED = 20260908  # 극단판
COSTS = (5.5, 7.8, 12.0)


def day_ci(v, day, rng, B=BOOT):
    u, inv = np.unique(day, return_inverse=True)
    if len(u) < 8 or len(v) < 8: return (np.nan, np.nan)
    s = np.bincount(inv, weights=v, minlength=len(u)); c = np.bincount(inv, minlength=len(u)).astype(float)
    pick = rng.integers(0, len(u), (B, len(u)))
    return tuple(np.percentile(s[pick].sum(1) / np.maximum(c[pick].sum(1), 1.0), [2.5, 97.5]))


def main() -> int:
    rng = np.random.default_rng(SEED)
    z = np.load(OUT / "panel.npz", allow_pickle=True)
    ts = pd.to_datetime(z["ts"]).to_numpy(); Om = z["O"]; Cm = z["C"]; Qm = z["Q"]
    syms = list(z["syms"])
    print(f"[1/2] 패널 {Om.shape[0]:,}봉 × {len(syms)}종목", flush=True)
    Qr = pd.DataFrame(Qm).rolling(LIQW, min_periods=LIQW // 2).median().to_numpy()
    liq = np.argsort(np.argsort(-np.nan_to_num(Qr, nan=-1.0), axis=1), axis=1)
    day_all = pd.Series(ts).dt.floor("D").to_numpy()
    win_of = np.full(len(ts), "", object)
    for w, (a, b) in SPLITS.items():
        win_of[(ts >= np.datetime64(a)) & (ts <= np.datetime64(b + "T23:59:59"))] = w

    rows = []
    for L in L_GRID:
        past = np.full_like(Cm, np.nan); past[L:] = Cm[L:] / Cm[:-L] - 1.0
        for H in H_GRID:
            fwd = np.full_like(Om, np.nan)
            fwd[:-(H + 1)] = Om[H + 1:] / Om[1:-H] - 1.0
            t0 = max(L, LIQW) + 1; t1 = len(ts) - H - 2
            tid = np.arange(t0, t1)
            for NU in NU_GRID:
                el = (liq[tid] < NU) & np.isfinite(past[tid]) & np.isfinite(fwd[tid])
                pa = np.where(el, past[tid], np.nan)
                if np.isfinite(pa).sum() == 0: continue
                nval = np.isfinite(pa).sum(1)
                gd = nval >= 8
                pa2 = np.where(np.isfinite(pa), pa, np.inf)
                lo_a = np.argmin(pa2, 1)
                pa3 = np.where(np.isfinite(pa), pa, -np.inf)
                hi_a = np.argmax(pa3, 1)
                r = np.arange(len(tid))
                spread = pa3[r, hi_a] - pa2[r, lo_a]
                fl = fwd[tid][r, lo_a]; fh = fwd[tid][r, hi_a]
                port = (fl - fh) / 2.0 * 1e4
                for th in THR:
                    ev = gd & np.isfinite(spread) & (spread >= th) & np.isfinite(port)
                    if ev.sum() < 120: continue
                    # 종목별 H봉 쿨다운 (롱 다리 기준)
                    keep = np.zeros(len(tid), bool)
                    last = np.full(len(syms), -10**9)
                    for i in np.flatnonzero(ev):
                        a = lo_a[i]
                        if tid[i] - last[a] >= H:
                            keep[i] = True; last[a] = tid[i]
                    tt = tid[keep]; v_all = port[keep]; dd = day_all[tt]; ww = win_of[tt]
                    rec = dict(L=L, H=H, NU=NU, thr=th, n_raw=int(ev.sum()), n=int(keep.sum()))
                    ok = True
                    for w in SPLITS:
                        m = ww == w
                        if m.sum() < 40: ok = False; break
                        v = v_all[m]; lo, hi = day_ci(v, dd[m], rng)
                        rec[f"{w}_g"] = float(v.mean()); rec[f"{w}_lo"] = lo
                        rec[f"{w}_hi"] = hi; rec[f"{w}_n"] = int(m.sum())
                    if ok: rows.append(rec)
        print(f"      L={L} 완료 (누적 셀 {len(rows)})", flush=True)
    R = pd.DataFrame(rows); R.to_csv(OUT / "overreaction_extreme.csv", index=False)
    print(f"\n[2/2] 셀 {len(R):,}\n", flush=True)
    for C in COSTS:
        okm = [all(R.loc[i, f"{w}_lo"] > C for w in SPLITS) for i in R.index]
        ok = R[okm]
        print(f"⭐비용 {C}bp: 네 창 모두 CI 하한 > 비용 {len(ok)}/{len(R)}")
        if len(ok):
            print(ok[["L", "H", "NU", "thr", "n"] + [f"{w}_g" for w in SPLITS]
                     + [f"{w}_lo" for w in SPLITS]].round(2).to_string(index=False))
    print("\n=== 네 창 최소 총수익 상위 12 ===")
    mn = R[[f"{w}_g" for w in SPLITS]].min(1)
    b = R.reindex(mn.sort_values(ascending=False).index).head(12)
    print(b[["L", "H", "NU", "thr", "n", "TRAIN_g", "TRAIN_lo", "TRAIN_n",
             "VAL_g", "OOS_g", "HOLDOUT_SPENT_g"]].round(2).to_string(index=False))
    print(json.dumps({"cells": len(R)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
