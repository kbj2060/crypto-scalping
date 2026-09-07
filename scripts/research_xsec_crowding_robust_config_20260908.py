#!/usr/bin/env python3
"""쏠림 페이드 -- **꼬리 의존을 견디는 구성 찾기** (2026-09-08).

## 남은 두 결함
후보 `tt_count_level` H=1일 NU=40 k=3 은 표본외 +37.0bp(p=0.001) 이지만
(1) **왕복 12bp 에서 CI 하한 −0.9** 로 아슬하게 미달, (2) **상위 5% 날 제거하면 평균 −5.6bp**.
꼬리 의존을 줄이는 구성(더 넓은 k · 1/vol 가중 · 수익률 윈저라이즈)이 두 결함을 동시에
해소하는지 본다. 판정: **표본외 순@12bp CI 하한 > 0 이면서 상위5% 제거 후에도 양수**.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "tmp/xsec_perp_screen_20260908"
OOS_A, OOS_B = "2025-09-01", "2026-07-31"
H = 288
K_GRID = (3, 5, 8, 12)
NU_GRID = (20, 30, 40)
WINSOR = (0.0, 0.10, 0.05)      # 종목 수익률 상하 절단 비율 (0=없음)
LIQW, VOLW = 288, 576
BOOT, NULLB = 4000, 600
SEED = 20260908


def boot_ci(v, rng, B=BOOT):
    if len(v) < 10: return (np.nan, np.nan)
    return tuple(np.percentile(v[rng.integers(0, len(v), (B, len(v)))].mean(1), [2.5, 97.5]))


def main() -> int:
    rng = np.random.default_rng(SEED)
    z = np.load(DIR / "panel.npz", allow_pickle=True)
    ts = pd.to_datetime(z["ts"]); Om = z["O"]; Cm = z["C"]; Qm = z["Q"]
    mz = np.load(DIR / "metrics_panel.npz", allow_pickle=True)
    S = np.where(mz["count_toptrader_long_short_ratio"] > 0,
                 np.log(np.maximum(mz["count_toptrader_long_short_ratio"], 1e-9)), np.nan)
    lr = np.full_like(Cm, np.nan); lr[1:] = np.log(Cm[1:] / Cm[:-1])
    vol = pd.DataFrame(lr).rolling(VOLW, min_periods=VOLW // 2).std().to_numpy()
    Qr = pd.DataFrame(Qm).rolling(LIQW, min_periods=LIQW // 2).median().to_numpy()
    liq = np.argsort(np.argsort(-np.nan_to_num(Qr, nan=-1.0), axis=1), axis=1)
    fwd = np.full_like(Om, np.nan); fwd[:-(H + 1)] = Om[H + 1:] / Om[1:-H] - 1.0
    tid = np.arange(max(LIQW, VOLW) + 1, len(ts) - H - 2, H)
    tt = ts[tid]
    seg = np.where(tt < OOS_A, "IN", np.where(tt <= OOS_B + " 23:59:59", "OUT", "X"))

    print("=" * 122)
    print(f"{'NU':>4}{'k':>4}{'윈저':>6}{'가중':>7}{'구간':>5} {'n':>4} {'총bp':>9} {'순@12 [CI95]':>24} "
          f"{'p':>7} {'상5%제거':>9} {'상1%제거':>9} {'승률':>6} {'일샤프':>7}")
    print("=" * 122)
    ok_rows = []
    for NU in NU_GRID:
        el = (liq[tid] < NU) & np.isfinite(S[tid]) & np.isfinite(fwd[tid]) & np.isfinite(vol[tid])
        sa = np.where(el, S[tid], np.nan); V = np.where(el, vol[tid], np.nan)
        nval = np.isfinite(sa).sum(1); order = np.argsort(np.where(np.isfinite(sa), sa, np.inf), 1)
        for wz in WINSOR:
            F0 = np.where(el, fwd[tid], np.nan)
            if wz > 0:
                lo_q = np.nanquantile(F0, wz, axis=1, keepdims=True)
                hi_q = np.nanquantile(F0, 1 - wz, axis=1, keepdims=True)
                F = np.clip(F0, lo_q, hi_q)
            else:
                F = F0
            for k in K_GRID:
                gd = nval >= 2 * k + 2
                rr = np.flatnonzero(gd)
                if len(rr) < 50: continue
                lo_i = order[rr][:, :k]
                hi_i = order[rr][np.arange(len(rr))[:, None],
                                 (nval[rr][:, None] - 1 - np.arange(k)[None, :])]
                fl = np.take_along_axis(F[rr], lo_i, 1); fh = np.take_along_axis(F[rr], hi_i, 1)
                wl = 1 / np.maximum(np.take_along_axis(V[rr], lo_i, 1), 1e-9)
                wh = 1 / np.maximum(np.take_along_axis(V[rr], hi_i, 1), 1e-9)
                arms = {"동일": (fl.mean(1) - fh.mean(1)) / 2 * 1e4,
                        "1/vol": ((fl * wl).sum(1) / wl.sum(1) - (fh * wh).sum(1) / wh.sum(1)) / 2 * 1e4}
                sg = seg[rr]
                for aname, port in arms.items():
                    for s_ in ("IN", "OUT"):
                        m = (sg == s_) & np.isfinite(port)
                        if m.sum() < 25: continue
                        v = port[m]; net = v - 12.0
                        lo, hi = boot_ci(net, rng)
                        nl = np.empty(NULLB)
                        for b in range(NULLB):
                            pick = np.stack([rng.permutation(int(x))[:2 * k] for x in nval[rr][m]])
                            Li = np.take_along_axis(order[rr][m], pick[:, :k], 1)
                            Hi = np.take_along_axis(order[rr][m], pick[:, k:], 1)
                            nl[b] = np.nanmean((np.take_along_axis(F[rr][m], Li, 1).mean(1)
                                                - np.take_along_axis(F[rr][m], Hi, 1).mean(1)) / 2 * 1e4)
                        p = max((nl >= v.mean()).mean(), 1 / NULLB)
                        t5 = v[v < np.percentile(v, 95)].mean(); t1 = v[v < np.percentile(v, 99)].mean()
                        print(f"{NU:>4}{k:>4}{wz:>6.2f}{aname:>7}{s_:>5} {m.sum():>4} {v.mean():>+9.1f} "
                              f"{net.mean():>+8.1f}[{lo:>+6.1f},{hi:>+6.1f}] {p:>7.3f} "
                              f"{t5:>+9.1f} {t1:>+9.1f} {(v>0).mean():>6.1%} "
                              f"{v.mean()/v.std():>7.3f}", flush=True)
                        if s_ == "OUT" and lo > 0 and t5 > 0:
                            ok_rows.append(dict(NU=NU, k=k, wz=wz, arm=aname, g=float(v.mean()),
                                                net_lo=lo, t5=float(t5), p=float(p)))
    print("\n" + "=" * 122)
    print(f"⭐표본외에서 **순@12bp CI 하한>0 이면서 상위5% 제거 후에도 양수**: {len(ok_rows)}건")
    for r in ok_rows: print("   ", r)
    print(json.dumps({"pass": len(ok_rows)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
