#!/usr/bin/env python3
"""⭐쏠림 페이드 -- **승격 심사 감사** (2026-09-08).

## 후보 (통합 표본외 2025-09-01~2026-07-31, 비겹침 n=334)
`tt_count_level`(상위트레이더 롱숏 **계정수** 비율의 수준) H=288봉(1일) k=3 NU=40:
**+37.0bp** [+11.9, +63.3] · 무작위배정 귀무 **p=0.001** · 순@12bp **+25.0bp/일**.
24 표본외 셀 중 **23개가 p<0.05**, 신호 3종(retail/tt_count/tt_pos)이 모두 같은 방향,
표본내(+6~+30)와 표본외(+10~+68)가 모두 양수.
샤프 추정 ~3.0 (일 sd ~239bp) -- **너무 좋아서** 편향을 더 파야 한다.

## 이 감사가 확인하는 것
1. **유동성 사다리** NU 10/20/30/40 -- 상위 유동성만으로도 남는가(비용 현실성)
2. **1/vol 가중** -- 롱 다리가 34% 더 변동성 큼. 리스크 정합 후에도 남는가
3. **연도·월별 분해** -- 특정 국면 의존인가
4. **누적곡선·MDD·승률·상위1% 제거**
5. **비용 사다리** 5.5/12/20/30bp
6. **한 다리씩** -- 롱만/숏만 수익 분해(어느 쪽이 실제 알파인가)
⚠️생존편향: 패널 60종은 현재 상장 종목만. 롱 다리(쏠림숏 종목)의 상장폐지 사례가 빠져 낙관 편향 가능.
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
NU_GRID = (10, 20, 30, 40)
LIQW, VOLW = 288, 576
BOOT, NULLB = 4000, 1000
SEED = 20260908


def boot_ci(v, rng, B=BOOT):
    if len(v) < 10: return (np.nan, np.nan)
    return tuple(np.percentile(v[rng.integers(0, len(v), (B, len(v)))].mean(1), [2.5, 97.5]))


def main() -> int:
    rng = np.random.default_rng(SEED)
    z = np.load(DIR / "panel.npz", allow_pickle=True)
    ts = pd.to_datetime(z["ts"]); Om = z["O"]; Cm = z["C"]; Qm = z["Q"]; syms = list(z["syms"])
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
    seg = np.where(tt < OOS_A, "표본내", np.where(tt <= OOS_B + " 23:59:59", "표본외", "제외"))

    print("=" * 112)
    print("1) 유동성 사다리 (H=1일, 비겹침) · 2) 1/vol 가중")
    print("=" * 112)
    print(f"{'NU':>4}{'k':>3}{'구간':>7} {'n':>4} {'원가중 총bp':>22} {'p':>7} {'1/vol 총bp':>22} {'p':>7}")
    store = {}
    for NU in NU_GRID:
        el = (liq[tid] < NU) & np.isfinite(S[tid]) & np.isfinite(fwd[tid]) & np.isfinite(vol[tid])
        sa = np.where(el, S[tid], np.nan); F = np.where(el, fwd[tid], np.nan)
        V = np.where(el, vol[tid], np.nan)
        nval = np.isfinite(sa).sum(1); order = np.argsort(np.where(np.isfinite(sa), sa, np.inf), 1)
        for k in K_GRID:
            gd = nval >= 2 * k + 2
            rr = np.flatnonzero(gd)
            if len(rr) < 50: continue
            lo_i = order[rr][:, :k]
            hi_i = order[rr][np.arange(len(rr))[:, None], (nval[rr][:, None] - 1 - np.arange(k)[None, :])]
            fl = np.take_along_axis(F[rr], lo_i, 1); fh = np.take_along_axis(F[rr], hi_i, 1)
            port = (fl.mean(1) - fh.mean(1)) / 2 * 1e4
            wl = 1 / np.maximum(np.take_along_axis(V[rr], lo_i, 1), 1e-9)
            wh = 1 / np.maximum(np.take_along_axis(V[rr], hi_i, 1), 1e-9)
            pv = ((fl * wl).sum(1) / wl.sum(1) - (fh * wh).sum(1) / wh.sum(1)) / 2 * 1e4
            sg = seg[rr]
            for sname in ("표본내", "표본외"):
                m = (sg == sname) & np.isfinite(port)
                if m.sum() < 25: continue
                out = []
                for arr in (port, pv):
                    v = arr[m]; lo, hi = boot_ci(v, rng)
                    nl = np.empty(NULLB)
                    for b in range(NULLB):
                        pick = np.stack([rng.permutation(int(x))[:2 * k] for x in nval[rr][m]])
                        Li = np.take_along_axis(order[rr][m], pick[:, :k], 1)
                        Hi = np.take_along_axis(order[rr][m], pick[:, k:], 1)
                        nl[b] = np.nanmean((np.take_along_axis(F[rr][m], Li, 1).mean(1)
                                            - np.take_along_axis(F[rr][m], Hi, 1).mean(1)) / 2 * 1e4)
                    out.append((v.mean(), lo, hi, max((nl >= v.mean()).mean(), 1 / NULLB)))
                print(f"{NU:>4}{k:>3}{sname:>7} {m.sum():>4} "
                      f"{out[0][0]:>+8.1f}[{out[0][1]:>+6.1f},{out[0][2]:>+6.1f}] {out[0][3]:>7.3f} "
                      f"{out[1][0]:>+8.1f}[{out[1][1]:>+6.1f},{out[1][2]:>+6.1f}] {out[1][3]:>7.3f}",
                      flush=True)
            if NU == 40 and k == 3:
                store["main"] = (tid[rr], port, sg, fl, fh, lo_i, hi_i)

    tid_, port, sg, fl, fh, lo_i, hi_i = store["main"]
    fin = np.isfinite(port)
    t2 = ts[tid_][fin]; v = port[fin]
    print("\n" + "=" * 112)
    print("3) 연·월별 · 4) 누적곡선 · 5) 비용 · 6) 다리 분해   [NU=40, k=3, H=1일]")
    print("=" * 112)
    df = pd.DataFrame({"t": t2, "bp": v,
                       "long_bp": np.nanmean(fl[fin], 1) * 1e4,
                       "short_bp": -np.nanmean(fh[fin], 1) * 1e4})
    df["ym"] = df.t.dt.to_period("M")
    yr = df.groupby(df.t.dt.year)["bp"].agg(["count", "mean", "sum"])
    print("연도별:"); print(yr.round(1).to_string())
    mo = df.groupby("ym")["bp"].mean()
    print(f"월별 양수 {int((mo>0).sum())}/{len(mo)}개월 · 최악 {mo.min():+.0f}bp({mo.idxmin()}) · "
          f"최고 {mo.max():+.0f}bp({mo.idxmax()})")
    eq = np.cumsum(v); dd = eq - np.maximum.accumulate(eq)
    print(f"\n누적 {eq[-1]:+.0f}bp / {len(v)}일 · 평균 {v.mean():+.1f} · 중앙 {np.median(v):+.1f} · "
          f"sd {v.std():.0f} · 일샤프 {v.mean()/v.std():.3f} (연 {v.mean()/v.std()*np.sqrt(365):.2f}) · "
          f"승률 {(v>0).mean():.1%} · MDD {dd.min():+.0f}bp")
    print(f"상위1% 제거 평균 {v[v<np.percentile(v,99)].mean():+.1f} · "
          f"상위5% 제거 {v[v<np.percentile(v,95)].mean():+.1f}")
    print("\n비용 사다리 (표본외만):")
    vo = df[df.t >= OOS_A]["bp"].to_numpy()
    for C in (5.5, 12, 20, 30):
        lo, hi = boot_ci(vo - C, rng)
        print(f"  왕복 {C:>4}bp -> 순 {vo.mean()-C:>+7.1f}bp/일 [{lo:>+6.1f},{hi:>+6.1f}] "
              f"{'통과' if lo > 0 else '미달'}")
    print(f"\n다리 분해(단위명목 아님, 각 다리 평균 수익):")
    for s_ in ("표본내", "표본외"):
        d2 = df[(df.t < OOS_A) if s_ == "표본내" else (df.t >= OOS_A)]
        print(f"  {s_}: 롱 {d2.long_bp.mean():>+7.1f}bp · 숏 {d2.short_bp.mean():>+7.1f}bp · "
              f"합/2 {(d2.long_bp.mean()+d2.short_bp.mean())/2:>+7.1f}")
    print(json.dumps({"n": int(fin.sum())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
