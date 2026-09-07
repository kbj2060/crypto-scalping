#!/usr/bin/env python3
"""횡단면 **다일(multi-day)** 롱숏 -- 비용이 구속하지 않는 지평 (2026-09-08).

## 왜 이 지평인가
09-08 밤 전수 결과의 일관된 형태: **찾아지는 엣지는 1~5bp, 비용은 5.5~12bp.**
극단 이탈(6~8%)만 수십 bp 였지만 감사 결과 **비유동 꼬리에만 존재**했다
(NU=10/20 에서 TRAIN −37.5/−21.7, NU=60 에서만 +25~46; 상위10종목이 손익 70.5%,
상위20일이 86.3%; 왕복 30bp 면 전 창 붕괴).
⇒ 구속조건은 신호가 아니라 **비용**이다. 보유를 1~14일로 늘리면 건당 비용 5.5~12bp 는
   200~500bp 움직임에 비해 무시할 수준이 된다. **비용 구속을 설계로 제거**하는 첫 시도.

⚠️**생존편향**: 이 패널 60종은 현재 상장 중인 종목만이다. 상장폐지된 종목이 없으므로
   다일 모멘텀/되돌림 추정치는 낙관적으로 치우친다. 롱숏이라 부분 상쇄되지만 완전하지 않다.
   승격 판단에 반드시 병기한다.

L,H 단위는 5분봉: 288=1일 · 864=3일 · 2016=7일 · 4032=14일 · 8640=30일.
되돌림 부호로 보고(음수면 모멘텀). 판정: 네 창 CI 하한 > 비용.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/xsec_perp_screen_20260908"
SPLITS = {"TRAIN": ("2024-01-01", "2025-08-31"), "VAL": ("2025-09-01", "2025-12-31"),
          "OOS": ("2026-01-01", "2026-03-31"), "HOLDOUT_SPENT": ("2026-04-01", "2026-07-31")}
D = 288
L_GRID = (D, 3 * D, 7 * D, 14 * D, 30 * D)
H_GRID = (D, 3 * D, 7 * D, 14 * D)
K_GRID = (3, 5, 8, 10)
NU_GRID = (20, 40, 60)
LIQW = 288
BOOT = 3000
SEED = 20260908
COSTS = (5.5, 12.0)


def day_ci(v, day, rng, B=BOOT):
    u, inv = np.unique(day, return_inverse=True)
    if len(u) < 6 or len(v) < 6: return (np.nan, np.nan)
    s = np.bincount(inv, weights=v, minlength=len(u)); c = np.bincount(inv, minlength=len(u)).astype(float)
    pick = rng.integers(0, len(u), (B, len(u)))
    return tuple(np.percentile(s[pick].sum(1) / np.maximum(c[pick].sum(1), 1.0), [2.5, 97.5]))


def main() -> int:
    rng = np.random.default_rng(SEED)
    z = np.load(OUT / "panel.npz", allow_pickle=True)
    ts = pd.to_datetime(z["ts"]); Om = z["O"]; Cm = z["C"]; Qm = z["Q"]; syms = list(z["syms"])
    Qr = pd.DataFrame(Qm).rolling(LIQW, min_periods=LIQW // 2).median().to_numpy()
    liq = np.argsort(np.argsort(-np.nan_to_num(Qr, nan=-1.0), axis=1), axis=1)
    day_all = ts.floor("D").to_numpy()
    win_of = np.full(len(ts), "", object)
    for w, (a, b) in SPLITS.items():
        win_of[(ts >= a) & (ts <= b + " 23:59:59")] = w
    print(f"패널 {Om.shape[0]:,}봉 × {len(syms)}종목", flush=True)

    rows = []
    for L in L_GRID:
        past = np.full_like(Cm, np.nan); past[L:] = Cm[L:] / Cm[:-L] - 1.0
        for H in H_GRID:
            fwd = np.full_like(Om, np.nan); fwd[:-(H + 1)] = Om[H + 1:] / Om[1:-H] - 1.0
            step = max(H // 4, D // 4)                      # 겹침 허용, 일군집 CI 로 처리
            tid = np.arange(max(L, LIQW) + 1, len(ts) - H - 2, step)
            for NU in NU_GRID:
                el = (liq[tid] < NU) & np.isfinite(past[tid]) & np.isfinite(fwd[tid])
                pa = np.where(el, past[tid], np.nan); fw = np.where(el, fwd[tid], np.nan)
                nval = np.isfinite(pa).sum(1)
                order = np.argsort(np.where(np.isfinite(pa), pa, np.inf), axis=1)
                for k in K_GRID:
                    gd = nval >= 2 * k + 2
                    if gd.sum() < 60: continue
                    rr = np.flatnonzero(gd)
                    lo_i = order[rr][:, :k]
                    hi_i = order[rr][np.arange(len(rr))[:, None],
                                     (nval[rr][:, None] - 1 - np.arange(k)[None, :])]
                    port = (np.take_along_axis(fw[rr], lo_i, 1).mean(1)
                            - np.take_along_axis(fw[rr], hi_i, 1).mean(1)) / 2.0 * 1e4
                    ww = win_of[tid][rr]; dd = day_all[tid][rr]
                    rec = dict(Ld=L // D, Hd=H // D, k=k, NU=NU, n=int(len(rr)))
                    ok = True
                    for w in SPLITS:
                        m = (ww == w) & np.isfinite(port)
                        if m.sum() < 30: ok = False; break
                        lo, hi = day_ci(port[m], dd[m], rng)
                        rec[f"{w}_g"] = float(port[m].mean()); rec[f"{w}_lo"] = lo
                        rec[f"{w}_hi"] = hi; rec[f"{w}_n"] = int(m.sum())
                    if ok: rows.append(rec)
        print(f"  L={L//D}일 완료 (셀 {len(rows)})", flush=True)
    R = pd.DataFrame(rows); R.to_csv(OUT / "multiday.csv", index=False)
    print(f"\n셀 {len(R):,}\n", flush=True)
    for C in COSTS:
        rev = R[[all(R.loc[i, f"{w}_lo"] > C for w in SPLITS) for i in R.index]]
        mom = R[[all(R.loc[i, f"{w}_hi"] < -C for w in SPLITS) for i in R.index]]
        print(f"⭐비용 {C}bp -- 되돌림 통과 {len(rev)}/{len(R)} · 모멘텀 통과 {len(mom)}/{len(R)}")
        for nm, t in (("되돌림", rev), ("모멘텀", mom)):
            if len(t):
                o = t.copy(); o["day_net"] = (o["TRAIN_g"].abs() - C) / o["Hd"]
                print(f"  [{nm}]")
                print(o.sort_values("day_net", ascending=False).head(10)
                      [["Ld", "Hd", "k", "NU", "n"] + [f"{w}_g" for w in SPLITS]
                       + [f"{w}_lo" for w in SPLITS] + ["day_net"]].round(1).to_string(index=False))
    print("\n=== |네 창 최소| 상위 12 (부호 일관 셀만) ===")
    sg = np.sign(R[[f"{w}_g" for w in SPLITS]])
    cons = (sg.nunique(axis=1) == 1)
    Rc = R[cons].copy()
    Rc["m"] = Rc[[f"{w}_g" for w in SPLITS]].abs().min(1)
    print(f"네 창 부호 일관 {int(cons.sum())}/{len(R)} (우연 기대 {len(R)/8:.0f})")
    print(Rc.sort_values("m", ascending=False).head(12)
          [["Ld", "Hd", "k", "NU", "TRAIN_g", "TRAIN_lo", "TRAIN_hi", "VAL_g", "OOS_g",
            "HOLDOUT_SPENT_g", "TRAIN_n"]].round(1).to_string(index=False))
    print(json.dumps({"cells": len(R)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
