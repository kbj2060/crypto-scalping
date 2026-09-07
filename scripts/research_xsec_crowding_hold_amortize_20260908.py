#!/usr/bin/env python3
"""쏠림 페이드 -- **보유기간 연장으로 비용 상각** (2026-09-08).

## 왜 이게 남은 지렛대인가
분산 감소 공학(부록 AJ)은 샤프를 0.110→0.133 으로 올렸지만 판정을 못 바꿨다. 필요 표본일수
중앙값이 1,285 → 1,091일로 줄었을 뿐이다(보유 표본외 334일).
⚠️게다가 **변동성 타깃팅은 오히려 해롭다** — 순bp 평균이 +11.0 → +7.3 으로, sd 감소분보다
평균 감소분이 크다(필요일수 1,091 → 1,335). 설계에서 뺀다.

**아직 안 건드린 축은 보유기간이다.** 비용은 **회전율**에 붙는다:
    보유 H봉 · 6시간마다 트랜치 하나씩 재조정 -> 하루 회전율 = 288/H
    ⇒ **하루 비용 = 12bp × (288/H)**  (1일 보유 12bp · 2일 6bp · 3일 4bp · 5일 2.4bp)
그런데 총수익은 H 에 대해 단조 증가했다(부록 AH: H=1일 +37 · 2일 +68bp/재조정).
일 환산 총수익은 +37 vs +34 로 비슷한데 **비용만 절반**이 된다.

## 격자
신호 2종 × H{1,2,3,5일} × k{3,5} × 신호가중{Y,N} × 일거래대금{$50M,$200M}
트랜치 = H/72 (6시간 간격 스태거) · 일별 손익으로 합산 · 하루 비용 = 12 × 288/H
판정: **표본외 순 CI 하한 > 0**, 표본내도 같이. 필요 표본일수도 병기.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "tmp/xsec_perp_screen_20260908"
OOS_A, OOS_B = "2025-09-01", "2026-07-31"
H_GRID = (288, 576, 864, 1440)          # 1 · 2 · 3 · 5일
STEP = 72                                # 6시간 간격 트랜치
K_GRID = (3, 5)
DV_GRID = (5e7, 2e8)
COST_1D = 12.0
BOOT = 5000
SEED = 20260908


def boot_ci(v, rng, B=BOOT):
    v = v[np.isfinite(v)]
    if len(v) < 10: return (np.nan, np.nan)
    return tuple(np.percentile(v[rng.integers(0, len(v), (B, len(v)))].mean(1), [2.5, 97.5]))


def main() -> int:
    rng = np.random.default_rng(SEED)
    z = np.load(DIR / "panel.npz", allow_pickle=True)
    ts = pd.to_datetime(z["ts"]); Om = z["O"]; Qm = z["Q"]
    mz = np.load(DIR / "metrics_panel.npz")
    lg = lambda X: np.where(X > 0, np.log(np.maximum(X, 1e-9)), np.nan)
    SIGS = {"tt_count": lg(mz["count_toptrader_long_short_ratio"]),
            "retail": lg(mz["count_long_short_ratio"])}
    DV = pd.DataFrame(Qm).rolling(288, min_periods=200).sum().to_numpy()
    print(f"패널 {len(ts):,}봉", flush=True)

    print("\n" + "=" * 128)
    print(f"{'신호':>9}{'보유':>6}{'트랜치':>6}{'k':>3}{'가중':>6}{'문턱':>7}{'구간':>5} {'일수':>5} "
          f"{'일총bp':>8}{'일비용':>7} {'일순bp [CI95]':>24} {'일sd':>7}{'연샤프':>7}{'필요일수':>8}")
    print("=" * 128)
    rows = []
    for sig, S in SIGS.items():
        for H in H_GRID:
            fwd = np.full_like(Om, np.nan)
            fwd[:-(H + 1)] = Om[H + 1:] / Om[1:-H] - 1.0
            T = H // STEP
            cost_day = COST_1D * 288.0 / H
            for DVt in DV_GRID:
                for k in K_GRID:
                    for sw in (False, True):
                        parts = []
                        for off in range(T):
                            tid = np.arange(600 + off * STEP, len(ts) - H - 2, H)
                            el = (DV[tid] >= DVt) & np.isfinite(S[tid]) & np.isfinite(fwd[tid])
                            nn = el.sum(1)
                            sa = np.where(el, S[tid], np.nan); F = np.where(el, fwd[tid], np.nan)
                            order = np.argsort(np.where(np.isfinite(sa), sa, np.inf), 1)
                            gd = nn >= 2 * k + 2
                            rr = np.flatnonzero(gd)
                            if len(rr) < 20: parts = []; break
                            lo_i = order[rr][:, :k]
                            hi_i = order[rr][np.arange(len(rr))[:, None],
                                             (nn[rr][:, None] - 1 - np.arange(k)[None, :])]
                            fl = np.take_along_axis(F[rr], lo_i, 1)
                            fh = np.take_along_axis(F[rr], hi_i, 1)
                            if sw:
                                sl = np.take_along_axis(sa[rr], lo_i, 1)
                                sh = np.take_along_axis(sa[rr], hi_i, 1)
                                med = np.nanmedian(sa[rr], 1, keepdims=True)
                                wl = np.maximum(med - sl, 1e-6); wh = np.maximum(sh - med, 1e-6)
                                p = ((fl * wl).sum(1) / wl.sum(1)
                                     - (fh * wh).sum(1) / wh.sum(1)) / 2 * 1e4
                            else:
                                p = (fl.mean(1) - fh.mean(1)) / 2 * 1e4
                            # H봉 보유 수익 -> 일 환산
                            parts.append(pd.Series(p * 288.0 / H, index=ts[tid][rr]))
                        if not parts: continue
                        ser = pd.concat(parts).sort_index()
                        d = ser.groupby(ser.index.floor("D")).mean()
                        idx = d.index; v_all = d.to_numpy().astype(float)
                        for seg, m in (("IN", idx < OOS_A),
                                       ("OUT", (idx >= OOS_A) & (idx <= OOS_B))):
                            v = v_all[np.asarray(m)]; v = v[np.isfinite(v)]
                            if len(v) < 60: continue
                            net = v - cost_day
                            lo, hi = boot_ci(net, rng)
                            sd = v.std(); sh_ = net.mean() / sd if sd > 0 else np.nan
                            need = (1.96 * sd / net.mean()) ** 2 if net.mean() > 0 else np.nan
                            print(f"{sig:>9}{H//288:>5}일{T:>6}{k:>3}{'신호' if sw else '동일':>6}"
                                  f"${DVt/1e6:>5.0f}M{seg:>5} {len(v):>5} {v.mean():>+8.1f}"
                                  f"{cost_day:>7.1f} {net.mean():>+7.1f}[{lo:>+6.1f},{hi:>+6.1f}] "
                                  f"{sd:>7.0f}{sh_*np.sqrt(365):>7.2f}"
                                  f"{(f'{need:.0f}' if np.isfinite(need) else '--'):>8}", flush=True)
                            rows.append(dict(sig=sig, Hd=H // 288, T=T, k=k, sw=sw, dv=DVt, seg=seg,
                                             n=len(v), g=float(v.mean()), cost=cost_day,
                                             net=float(net.mean()), lo=lo, hi=hi, sd=float(sd),
                                             ann_sharpe=float(sh_ * np.sqrt(365)),
                                             need=float(need) if np.isfinite(need) else np.nan))
    R = pd.DataFrame(rows); R.to_csv(DIR / "hold_amortize.csv", index=False)
    o = R[R.seg == "OUT"]; ok = o[o.lo > 0]
    print("\n" + "=" * 128)
    print(f"⭐표본외 일순bp CI 하한 > 0: {len(ok)}/{len(o)}")
    if len(ok):
        print(ok.sort_values("ann_sharpe", ascending=False).head(12)
              [["sig", "Hd", "k", "sw", "dv", "n", "g", "cost", "net", "lo", "ann_sharpe"]]
              .round(2).to_string(index=False))
    both = []
    for _, r in ok.iterrows():
        q = R[(R.sig == r.sig) & (R.Hd == r.Hd) & (R.k == r.k) & (R.sw == r.sw)
              & (R.dv == r.dv) & (R.seg == "IN")]
        if len(q) and q.iloc[0]["lo"] > 0:
            both.append((r.sig, int(r.Hd), int(r.k), bool(r.sw), r.dv/1e6,
                         round(q.iloc[0]["net"], 1), round(r.net, 1)))
    print(f"\n⭐⭐**표본내·표본외 둘 다** CI 하한 > 0: {len(both)}건")
    for b in both: print("   ", b)
    print("\n=== 보유기간별 표본외 평균 (신호 tt_count) ===")
    t = R[(R.seg == "OUT") & (R.sig == "tt_count")].groupby("Hd")[["g", "cost", "net", "sd", "ann_sharpe", "need"]].mean()
    print(t.round(2).to_string())
    print(json.dumps({"out": len(o), "pass_out": len(ok), "pass_both": len(both)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
