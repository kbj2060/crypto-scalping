#!/usr/bin/env python3
"""쏠림 페이드 후보의 **통합 표본외 검정** (2026-09-08).

## 왜 통합하나
`tt_count_level` 후보는 창별로 무작위배정 귀무 p 가 TRAIN 0.020 · VAL 0.0025 · **OOS 0.3875 ·
HOLDOUT 0.3350** 이었다. 그런데 H=864(3일) 비겹침이면 OOS 는 **n=30**, HOLDOUT n=40 이다 --
효과가 없어서가 아니라 **잴 수 없어서** 유의하지 않을 수 있다.
VAL+OOS+HOLDOUT 을 하나의 표본외 구간(2025-09-01~2026-07-31, 11개월)으로 합치면
H=288(1일) 기준 비겹침 n≈330 이 된다.

⚠️**선택 편향 고지**: 이 셀(신호·H·k)은 90셀 스크린에서 네 창을 다 본 뒤 골랐다.
따라서 통합 표본외도 완전한 선택-밖이 아니다. 무작위배정 귀무 p 를 셀 수로 보수 보정해 병기한다.

## 검정
- 신호 3종(retail/tt_count/tt_pos) + **3종 평균 합성**
- H ∈ {288, 576} 비겹침 · k ∈ {3,5,8} · NU=40 · 원가중 및 1/vol 가중
- 무작위 배정 귀무 B=1000 (같은 시각·같은 유니버스에서 k개씩 무작위)
- 비용 5.5/12bp 차감 후 순bp 와 부트스트랩 CI
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "tmp/xsec_perp_screen_20260908"
OOS_A, OOS_B = "2025-09-01", "2026-07-31"
H_GRID = (288, 576)
K_GRID = (3, 5, 8)
NU = 40
LIQW, VOLW = 288, 576
BOOT, NULLB = 4000, 1000
SEED = 20260908
NCELL = 90        # 후보를 고른 스크린의 셀 수 (보수 보정용)


def boot_ci(v, rng, B=BOOT):
    if len(v) < 10: return (np.nan, np.nan)
    return tuple(np.percentile(v[rng.integers(0, len(v), (B, len(v)))].mean(1), [2.5, 97.5]))


def main() -> int:
    rng = np.random.default_rng(SEED)
    z = np.load(DIR / "panel.npz", allow_pickle=True)
    ts = pd.to_datetime(z["ts"]); Om = z["O"]; Cm = z["C"]; Qm = z["Q"]; syms = list(z["syms"])
    mz = np.load(DIR / "metrics_panel.npz", allow_pickle=True)
    lg = lambda X: np.where(X > 0, np.log(np.maximum(X, 1e-9)), np.nan)
    RAW = {"retail_level": lg(mz["count_long_short_ratio"]),
           "tt_count_level": lg(mz["count_toptrader_long_short_ratio"]),
           "tt_pos_level": lg(mz["sum_toptrader_long_short_ratio"])}
    # 합성 = 세 신호의 횡단면 순위 평균 (스케일 다름 -> 순위로)
    def csrank(X):
        R = np.full_like(X, np.nan)
        fin = np.isfinite(X)
        n = fin.sum(1)
        o = np.argsort(np.where(fin, X, np.inf), 1)
        rk = np.argsort(o, 1).astype(np.float32)
        R = np.where(fin, rk / np.maximum(n[:, None] - 1, 1), np.nan)
        return R
    RAW["composite"] = np.nanmean(np.stack([csrank(v) for v in RAW.values()]), 0)
    lr = np.full_like(Cm, np.nan); lr[1:] = np.log(Cm[1:] / Cm[:-1])
    vol = pd.DataFrame(lr).rolling(VOLW, min_periods=VOLW // 2).std().to_numpy()
    Qr = pd.DataFrame(Qm).rolling(LIQW, min_periods=LIQW // 2).median().to_numpy()
    liq = np.argsort(np.argsort(-np.nan_to_num(Qr, nan=-1.0), axis=1), axis=1)
    oos = (ts >= OOS_A) & (ts <= OOS_B + " 23:59:59")
    ins = ts < OOS_A
    print(f"패널 {len(ts):,}봉 · 통합 표본외 {OOS_A}~{OOS_B}", flush=True)

    print("\n" + "=" * 118)
    print(f"{'신호':>15}{'H':>5}{'k':>3}{'구간':>8} {'n':>4} {'총bp':>9} {'[부트 CI95]':>20} "
          f"{'귀무평균':>8}{'sd':>7}{'p':>8} {'순@5.5':>8}{'순@12':>8}")
    print("=" * 118)
    res = []
    for H in H_GRID:
        fwd = np.full_like(Om, np.nan); fwd[:-(H + 1)] = Om[H + 1:] / Om[1:-H] - 1.0
        tid = np.arange(max(LIQW, VOLW) + 1, len(ts) - H - 2, H)
        F = fwd[tid]
        for sname, S in RAW.items():
            el = (liq[tid] < NU) & np.isfinite(S[tid]) & np.isfinite(F) & np.isfinite(vol[tid])
            sa = np.where(el, S[tid], np.nan)
            nval = np.isfinite(sa).sum(1)
            order = np.argsort(np.where(np.isfinite(sa), sa, np.inf), 1)
            for k in K_GRID:
                gd = nval >= 2 * k + 2
                rr = np.flatnonzero(gd)
                if len(rr) < 60: continue
                lo_i = order[rr][:, :k]
                hi_i = order[rr][np.arange(len(rr))[:, None],
                                 (nval[rr][:, None] - 1 - np.arange(k)[None, :])]
                Fr = np.where(el, F, np.nan)[rr]
                port = (np.take_along_axis(Fr, lo_i, 1).mean(1)
                        - np.take_along_axis(Fr, hi_i, 1).mean(1)) / 2 * 1e4
                nvr = nval[rr]; orr = order[rr]
                for seg, mask in (("표본내", ins[tid][rr]), ("표본외", oos[tid][rr])):
                    m = mask & np.isfinite(port)
                    if m.sum() < 25: continue
                    v = port[m]
                    lo, hi = boot_ci(v, rng)
                    nl = np.empty(NULLB)
                    for b in range(NULLB):
                        pick = np.stack([rng.permutation(int(x))[:2 * k] for x in nvr[m]])
                        Li = np.take_along_axis(orr[m], pick[:, :k], 1)
                        Hi = np.take_along_axis(orr[m], pick[:, k:], 1)
                        nl[b] = np.nanmean((np.take_along_axis(Fr[m], Li, 1).mean(1)
                                            - np.take_along_axis(Fr[m], Hi, 1).mean(1)) / 2 * 1e4)
                    p = max((nl >= v.mean()).mean(), 1.0 / NULLB)
                    print(f"{sname:>15}{H:>5}{k:>3}{seg:>8} {m.sum():>4} {v.mean():>+9.1f} "
                          f"[{lo:>+8.1f},{hi:>+8.1f}] {nl.mean():>+8.1f}{nl.std():>7.1f}"
                          f"{p:>8.4f} {v.mean()-5.5:>+8.1f}{v.mean()-12:>+8.1f}", flush=True)
                    res.append(dict(sig=sname, H=H, k=k, seg=seg, n=int(m.sum()),
                                    g=float(v.mean()), lo=lo, hi=hi, p=float(p)))
    R = pd.DataFrame(res); R.to_csv(DIR / "crowding_pooled.csv", index=False)
    o = R[R.seg == "표본외"]
    print("\n" + "=" * 118)
    print(f"⭐통합 표본외 셀 {len(o)} · p<0.05 {int((o.p < 0.05).sum())} · "
          f"보수 보정(p < 0.05/{NCELL} = {0.05/NCELL:.2e}) 통과 {int((o.p < 0.05/NCELL).sum())}")
    print(f"⭐표본외에서 CI 하한 > 12bp: {int((o.lo > 12).sum())}/{len(o)} · "
          f"CI 하한 > 5.5bp: {int((o.lo > 5.5).sum())}/{len(o)}")
    print(o.sort_values("p").head(10).round(3).to_string(index=False))
    print(json.dumps({"oos_cells": len(o), "p05": int((o.p < 0.05).sum())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
