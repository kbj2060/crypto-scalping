#!/usr/bin/env python3
"""쏠림 페이드 -- **분산 감소 공학** (2026-09-08).

## 왜 분산인가
부록 AI 결론: 평균은 충분하다(표본외 +25~34bp/일, 귀무 p 0.002~0.005). 막힌 것은 **분산**이다 --
일 sd ~200bp 라 n=334 에서 순@12bp 의 CI 하한이 0 을 넘지 못한다.
평균을 건드리지 않으면서 분산을 줄이는 네 가지를 각각·조합으로 측정한다.

1. **합성 신호** -- 세 쏠림 지표(개미·상위트레이더 계정수·상위트레이더 포지션)의 횡단면 순위 평균.
   신호 잡음이 평균화된다.
2. **겹침 트랜치** -- 자본을 4등분해 6시간마다 하나씩 재조정(각 트랜치 24시간 보유).
   진입 시점 잡음이 평균화된다. **회전율은 하루 1회로 동일하므로 비용은 그대로 12bp/일.**
3. **변동성 타깃팅** -- 직전 30일 실현 변동성으로 노출을 조절(상한 3배). 평균이 변동성에
   비례하지 않는 만큼만 샤프가 오른다.
4. **신호가중 + k 확대** -- 극단 의존을 줄이면서 신호 세기를 반영.

판정: 표본외 **순@12bp 의 CI 하한 > 0** 과 일샤프. 표본내도 같이 낸다.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "tmp/xsec_perp_screen_20260908"
OOS_A, OOS_B = "2025-09-01", "2026-07-31"
H = 288                       # 보유 1일
DV_GRID = (5e7, 2e8)
K_GRID = (3, 5, 8)
TRANCHES = (1, 4)
VOLW = 576
COST = 12.0
BOOT = 5000
SEED = 20260908


def boot_ci(v, rng, B=BOOT):
    v = v[np.isfinite(v)]
    if len(v) < 10: return (np.nan, np.nan)
    return tuple(np.percentile(v[rng.integers(0, len(v), (B, len(v)))].mean(1), [2.5, 97.5]))


def csrank(X):
    fin = np.isfinite(X); n = fin.sum(1)
    o = np.argsort(np.where(fin, X, np.inf), 1)
    rk = np.argsort(o, 1).astype(np.float32)
    return np.where(fin, rk / np.maximum(n[:, None] - 1, 1), np.nan)


def main() -> int:
    rng = np.random.default_rng(SEED)
    z = np.load(DIR / "panel.npz", allow_pickle=True)
    ts = pd.to_datetime(z["ts"]); Om = z["O"]; Cm = z["C"]; Qm = z["Q"]
    mz = np.load(DIR / "metrics_panel.npz")
    lg = lambda X: np.where(X > 0, np.log(np.maximum(X, 1e-9)), np.nan)
    SIGS = {"tt_count": lg(mz["count_toptrader_long_short_ratio"]),
            "composite": np.nanmean(np.stack([csrank(lg(mz["count_long_short_ratio"])),
                                              csrank(lg(mz["count_toptrader_long_short_ratio"])),
                                              csrank(lg(mz["sum_toptrader_long_short_ratio"]))]), 0)}
    DV = pd.DataFrame(Qm).rolling(288, min_periods=200).sum().to_numpy()
    fwd = np.full_like(Om, np.nan); fwd[:-(H + 1)] = Om[H + 1:] / Om[1:-H] - 1.0
    print(f"패널 {len(ts):,}봉", flush=True)

    def one(sig, DVt, k, sw, offset, step):
        S = SIGS[sig]
        tid = np.arange(VOLW + 1 + offset, len(ts) - H - 2, step)
        el = (DV[tid] >= DVt) & np.isfinite(S[tid]) & np.isfinite(fwd[tid])
        nn = el.sum(1)
        sa = np.where(el, S[tid], np.nan); F = np.where(el, fwd[tid], np.nan)
        order = np.argsort(np.where(np.isfinite(sa), sa, np.inf), 1)
        gd = nn >= 2 * k + 2
        rr = np.flatnonzero(gd)
        if len(rr) < 40: return None, None
        lo_i = order[rr][:, :k]
        hi_i = order[rr][np.arange(len(rr))[:, None], (nn[rr][:, None] - 1 - np.arange(k)[None, :])]
        fl = np.take_along_axis(F[rr], lo_i, 1); fh = np.take_along_axis(F[rr], hi_i, 1)
        if sw:
            sl = np.take_along_axis(sa[rr], lo_i, 1); sh = np.take_along_axis(sa[rr], hi_i, 1)
            med = np.nanmedian(sa[rr], 1, keepdims=True)
            wl = np.maximum(med - sl, 1e-6); wh = np.maximum(sh - med, 1e-6)
            p = ((fl * wl).sum(1) / wl.sum(1) - (fh * wh).sum(1) / wh.sum(1)) / 2 * 1e4
        else:
            p = (fl.mean(1) - fh.mean(1)) / 2 * 1e4
        return ts[tid][rr], p

    print("\n" + "=" * 122)
    print(f"{'신호':>10}{'문턱':>7}{'k':>3}{'가중':>7}{'트랜치':>7}{'volTgt':>7}{'구간':>5} {'n':>4} "
          f"{'총bp':>8} {'순@12 [CI95]':>23} {'일sd':>7}{'일샤프':>7}{'연샤프':>7}")
    print("=" * 122)
    out = []
    for sig in SIGS:
        for DVt in DV_GRID:
            for k in K_GRID:
                for sw in (False, True):
                    for T in TRANCHES:
                        step = H // T
                        parts = [one(sig, DVt, k, sw, o * step, H) for o in range(T)]
                        if any(p is None for _, p in parts): continue
                        ser = pd.concat([pd.Series(p, index=t) for t, p in parts]).sort_index()
                        # 일 단위로 합쳐 자본 1단위 기준 수익 (T개 트랜치 = 각 1/T)
                        d = ser.groupby(ser.index.floor("D")).mean()
                        for vt in (False, True):
                            v = d.to_numpy().astype(float)
                            if vt:
                                rv = pd.Series(v).rolling(30, min_periods=15).std().shift(1).to_numpy()
                                sc = np.clip(np.nanmedian(rv) / np.maximum(rv, 1e-9), 0.25, 3.0)
                                v = v * sc
                            idx = d.index
                            for seg, m in (("IN", idx < OOS_A),
                                           ("OUT", (idx >= OOS_A) & (idx <= OOS_B))):
                                vv = v[m.to_numpy() if hasattr(m, "to_numpy") else m]
                                vv = vv[np.isfinite(vv)]
                                if len(vv) < 60: continue
                                net = vv - COST
                                lo, hi = boot_ci(net, rng)
                                sh_ = vv.mean() / vv.std() if vv.std() > 0 else np.nan
                                print(f"{sig:>10}${DVt/1e6:>6.0f}M{k:>3}"
                                      f"{'신호' if sw else '동일':>7}{T:>7}{'Y' if vt else '-':>7}{seg:>5} "
                                      f"{len(vv):>4} {vv.mean():>+8.1f} "
                                      f"{net.mean():>+7.1f}[{lo:>+6.1f},{hi:>+6.1f}] "
                                      f"{vv.std():>7.0f}{sh_:>7.3f}{sh_*np.sqrt(365):>7.2f}", flush=True)
                                out.append(dict(sig=sig, dv=DVt, k=k, sw=sw, T=T, vt=vt, seg=seg,
                                                n=len(vv), g=float(vv.mean()), lo=lo, hi=hi,
                                                sd=float(vv.std()), sharpe=float(sh_)))
    R = pd.DataFrame(out); R.to_csv(DIR / "portfolio_eng.csv", index=False)
    o = R[R.seg == "OUT"]
    ok = o[o.lo > 0]
    print("\n" + "=" * 122)
    print(f"⭐표본외 순@12bp CI 하한 > 0: {len(ok)}/{len(o)}")
    if len(ok):
        print(ok.sort_values("lo", ascending=False).head(12).round(2).to_string(index=False))
    # 표본내·표본외 둘 다 통과
    both = []
    for _, r in ok.iterrows():
        q = R[(R.sig == r.sig) & (R.dv == r.dv) & (R.k == r.k) & (R.sw == r.sw)
              & (R["T"] == r["T"]) & (R.vt == r.vt) & (R.seg == "IN")]
        if len(q) and q.iloc[0]["lo"] > 0: both.append((r.sig, r.dv, r.k, r.sw, r["T"], r.vt))
    print(f"⭐표본내·표본외 **둘 다** 순@12bp CI 하한 > 0: {len(both)}건 {both}")
    print(json.dumps({"out_cells": len(o), "pass_out": len(ok), "pass_both": len(both)},
                     ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
