#!/usr/bin/env python3
"""쏠림 페이드 -- **절대 유동성 문턱과 수용량(capacity)** (2026-09-08).

## 왜 이걸 재는가
지금까지 유니버스를 **순위**(상위 NU종)로 잘랐다. 순위는 체결 가능성을 말해 주지 않는다 --
바이낸스 상위 60종의 40위도 일거래대금이 수천만 달러일 수 있다.
쏠림 페이드의 유일한 남은 질문이 "꼬리에서 실제로 채울 수 있는가"이므로,
**절대 일거래대금 문턱**으로 자르고 **수용량(다리당 최대 명목)** 을 같이 낸다.

- 유니버스 = 직전 288봉(1일) quote_volume 합 ≥ 문턱 (2천만~50억 달러)
- 수용량 = 그날 선택된 종목들의 일거래대금 × 참여율(0.5% / 1% / 2%) 의 **최솟값**
  (가장 얇은 다리가 병목이다)
- 신호 가중(|신호 z| 비례) 변형도 같이 본다
- 판정: 표본외 무작위배정 귀무 p 와 순@12bp CI, 그리고 그때의 수용량(달러)
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "tmp/xsec_perp_screen_20260908"
OOS_A, OOS_B = "2025-09-01", "2026-07-31"
H = 288
K_GRID = (3, 5)
DV_GRID = (2e7, 5e7, 2e8, 5e8, 1e9, 5e9)     # 일거래대금 문턱 (USDT)
PARTIC = (0.005, 0.01, 0.02)
VOLW = 576
BOOT, NULLB = 4000, 600
SEED = 20260908


def boot_ci(v, rng, B=BOOT):
    if len(v) < 10: return (np.nan, np.nan)
    return tuple(np.percentile(v[rng.integers(0, len(v), (B, len(v)))].mean(1), [2.5, 97.5]))


def main() -> int:
    rng = np.random.default_rng(SEED)
    z = np.load(DIR / "panel.npz", allow_pickle=True)
    ts = pd.to_datetime(z["ts"]); Om = z["O"]; Cm = z["C"]; Qm = z["Q"]; syms = list(z["syms"])
    RAW = np.load(DIR / "metrics_panel.npz")["count_toptrader_long_short_ratio"]
    S = np.where(RAW > 0, np.log(np.maximum(RAW, 1e-9)), np.nan)
    DV = pd.DataFrame(Qm).rolling(288, min_periods=200).sum().to_numpy()      # 직전 1일 거래대금
    lr = np.full_like(Cm, np.nan); lr[1:] = np.log(Cm[1:] / Cm[:-1])
    vol = pd.DataFrame(lr).rolling(VOLW, min_periods=VOLW // 2).std().to_numpy()
    fwd = np.full_like(Om, np.nan); fwd[:-(H + 1)] = Om[H + 1:] / Om[1:-H] - 1.0
    tid = np.arange(VOLW + 1, len(ts) - H - 2, H)
    tt = ts[tid]
    seg = np.where(tt < OOS_A, "IN", np.where(tt <= OOS_B + " 23:59:59", "OUT", "X"))
    print(f"재조정 {len(tid)}회 (1일) · 일거래대금 중앙 전체 ${np.nanmedian(DV)/1e6:.0f}M", flush=True)

    print("\n" + "=" * 126)
    print(f"{'문턱':>7}{'k':>3}{'가중':>7}{'구간':>5} {'종목/일':>7} {'n':>4} {'총bp':>8} "
          f"{'순@12 [CI95]':>23} {'p':>7} {'수용량@1% (다리당)':>19} {'상5%제거':>9}")
    print("=" * 126)
    best = []
    for DVt in DV_GRID:
        el = (DV[tid] >= DVt) & np.isfinite(S[tid]) & np.isfinite(fwd[tid]) & np.isfinite(vol[tid])
        nn = el.sum(1)
        sa = np.where(el, S[tid], np.nan); F = np.where(el, fwd[tid], np.nan)
        Dv = np.where(el, DV[tid], np.nan)
        order = np.argsort(np.where(np.isfinite(sa), sa, np.inf), 1)
        for k in K_GRID:
            gd = nn >= 2 * k + 2
            rr = np.flatnonzero(gd)
            if len(rr) < 50: continue
            lo_i = order[rr][:, :k]
            hi_i = order[rr][np.arange(len(rr))[:, None], (nn[rr][:, None] - 1 - np.arange(k)[None, :])]
            fl = np.take_along_axis(F[rr], lo_i, 1); fh = np.take_along_axis(F[rr], hi_i, 1)
            sl = np.take_along_axis(sa[rr], lo_i, 1); sh = np.take_along_axis(sa[rr], hi_i, 1)
            dl = np.take_along_axis(Dv[rr], lo_i, 1); dh = np.take_along_axis(Dv[rr], hi_i, 1)
            # 신호 가중: 횡단면 중앙값에서의 거리
            med = np.nanmedian(sa[rr], 1, keepdims=True)
            wl = np.maximum(med - sl, 1e-6); wh = np.maximum(sh - med, 1e-6)
            arms = {"동일": (fl.mean(1) - fh.mean(1)) / 2 * 1e4,
                    "신호가중": ((fl * wl).sum(1) / wl.sum(1) - (fh * wh).sum(1) / wh.sum(1)) / 2 * 1e4}
            cap = np.minimum(dl.min(1), dh.min(1))          # 가장 얇은 다리
            sg = seg[rr]
            for an, port in arms.items():
                for s_ in ("IN", "OUT"):
                    m = (sg == s_) & np.isfinite(port)
                    if m.sum() < 25: continue
                    v = port[m]; net = v - 12.0
                    lo, hi = boot_ci(net, rng)
                    nl = np.empty(NULLB)
                    for b in range(NULLB):
                        pick = np.stack([rng.permutation(int(x))[:2 * k] for x in nn[rr][m]])
                        Li = np.take_along_axis(order[rr][m], pick[:, :k], 1)
                        Hi = np.take_along_axis(order[rr][m], pick[:, k:], 1)
                        nl[b] = np.nanmean((np.take_along_axis(F[rr][m], Li, 1).mean(1)
                                            - np.take_along_axis(F[rr][m], Hi, 1).mean(1)) / 2 * 1e4)
                    p = max((nl >= v.mean()).mean(), 1 / NULLB)
                    c1 = np.nanmedian(cap[m]) * 0.01
                    t5 = v[v < np.percentile(v, 95)].mean()
                    print(f"${DVt/1e6:>6.0f}M{k:>3}{an:>7}{s_:>5} {nn[rr][m].mean():>7.1f} {m.sum():>4} "
                          f"{v.mean():>+8.1f} {net.mean():>+7.1f}[{lo:>+6.1f},{hi:>+6.1f}] {p:>7.3f} "
                          f"${c1/1e6:>16.2f}M {t5:>+9.1f}", flush=True)
                    if s_ == "OUT":
                        best.append(dict(dv=DVt, k=k, arm=an, g=float(v.mean()), lo=lo, p=float(p),
                                         cap1=float(c1), t5=float(t5), names=float(nn[rr][m].mean())))
    B = pd.DataFrame(best)
    B.to_csv(DIR / "capacity.csv", index=False)
    print("\n" + "=" * 126)
    print(f"⭐표본외 순@12bp CI 하한 > 0: {int((B.lo > 0).sum())}/{len(B)} · "
          f"귀무 p<0.01: {int((B.p < 0.01).sum())}/{len(B)}")
    print("\n수용량 사다리 (참여율별, 최고 p 셀 기준):")
    b0 = B.loc[B.p.idxmin()]
    for pr in PARTIC:
        print(f"   참여율 {pr:.1%} -> 다리당 ${b0.cap1/0.01*pr/1e6:,.1f}M · "
              f"총 명목 ${2*b0.k*b0.cap1/0.01*pr/1e6:,.1f}M")
    print(f"   (문턱 ${b0.dv/1e6:.0f}M · k={int(b0.k)} · {b0.arm} · 총 {b0.g:+.1f}bp · p={b0.p:.3f})")
    print(json.dumps({"cells": len(B), "pass": int((B.lo > 0).sum())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
