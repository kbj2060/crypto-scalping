#!/usr/bin/env python3
"""⚠️쏠림 페이드 보유연장 결과의 **겹침 보정 재검정** (2026-09-08).

## 왜 다시 재나
`research_xsec_crowding_hold_amortize_20260908.py` 가 표본내·표본외 둘 다 CI 하한>0 인 셀을
**6개** 냈다(5일 보유·6시간 스태거 20트랜치·연샤프 2.4~3.4). 그런데 그 CI 는 **일 단위 iid
부트스트랩**이었다.

5일 보유를 진입일에 귀속시켜 1/5씩 배분하면, 인접한 날의 값은 **달력상 겹치는 5일 구간의
수익**을 담는다 -> 일 계열이 강하게 자기상관될 수밖에 없고, iid 부트스트랩 CI 는 **너무 좁다**.
[[eth_trailing_stop_infeasible_fill_bug_20260907]] 류의 회계 착시와 같은 부류라 반드시 잡아야 한다.

## 이 스크립트
같은 셀들에 대해 세 가지로 다시 잰다.
1. **자기상관 함수** ACF(1..10) -- 겹침이 실제로 얼마나 상관을 만드는지
2. **이동블록 부트스트랩** 블록 길이 1/5/10/20일 (H 이상이 정론)
3. **비겹침 부분표본** -- H일 간격으로 하나씩만 취해 iid 로 (검정력은 낮지만 편향 없음)
4. Newey-West t 통계량 (lag = H일)
판정: **블록 길이 ≥ H 에서도 CI 하한 > 0 인가.**
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "tmp/xsec_perp_screen_20260908"
OOS_A, OOS_B = "2025-09-01", "2026-07-31"
STEP = 72
COST_1D = 12.0
BOOT = 5000
SEED = 20260908
# hold_amortize 에서 두 창 모두 통과한 6셀 + 대표 1일 셀
CELLS = [("tt_count", 1440, 5, True, 2e8), ("tt_count", 1440, 3, True, 5e7),
         ("tt_count", 1440, 5, True, 5e7), ("tt_count", 1440, 3, True, 2e8),
         ("tt_count", 864, 5, True, 2e8), ("retail", 1440, 5, True, 2e8),
         ("tt_count", 288, 3, True, 5e7)]


def mbb_ci(v, L, rng, B=BOOT):
    """이동블록 부트스트랩 (블록 길이 L)."""
    n = len(v)
    if n < 3 * L: return (np.nan, np.nan)
    nb = int(np.ceil(n / L))
    starts = rng.integers(0, n - L + 1, (B, nb))
    idx = (starts[:, :, None] + np.arange(L)[None, None, :]).reshape(B, -1)[:, :n]
    return tuple(np.percentile(v[idx].mean(1), [2.5, 97.5]))


def nw_t(v, lag):
    x = v - v.mean(); n = len(v); g0 = (x * x).mean()
    s = g0
    for l in range(1, lag + 1):
        gl = (x[l:] * x[:-l]).mean()
        s += 2 * (1 - l / (lag + 1)) * gl
    se = np.sqrt(max(s, 1e-12) / n)
    return v.mean() / se


def build(sig, H, k, sw, DVt, ts, Om, DV, S):
    fwd = np.full_like(Om, np.nan); fwd[:-(H + 1)] = Om[H + 1:] / Om[1:-H] - 1.0
    T = H // STEP
    parts = []
    for off in range(T):
        tid = np.arange(600 + off * STEP, len(ts) - H - 2, H)
        el = (DV[tid] >= DVt) & np.isfinite(S[tid]) & np.isfinite(fwd[tid])
        nn = el.sum(1); sa = np.where(el, S[tid], np.nan); F = np.where(el, fwd[tid], np.nan)
        order = np.argsort(np.where(np.isfinite(sa), sa, np.inf), 1)
        rr = np.flatnonzero(nn >= 2 * k + 2)
        if len(rr) < 20: return None
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
        parts.append(pd.Series(p * 288.0 / H, index=ts[tid][rr]))
    ser = pd.concat(parts).sort_index()
    return ser.groupby(ser.index.floor("D")).mean()


def main() -> int:
    rng = np.random.default_rng(SEED)
    z = np.load(DIR / "panel.npz", allow_pickle=True)
    ts = pd.to_datetime(z["ts"]); Om = z["O"]; Qm = z["Q"]
    mz = np.load(DIR / "metrics_panel.npz")
    lg = lambda X: np.where(X > 0, np.log(np.maximum(X, 1e-9)), np.nan)
    SIGS = {"tt_count": lg(mz["count_toptrader_long_short_ratio"]),
            "retail": lg(mz["count_long_short_ratio"])}
    DV = pd.DataFrame(Qm).rolling(288, min_periods=200).sum().to_numpy()

    print("=" * 124)
    npass = 0
    for sig, H, k, sw, DVt in CELLS:
        d = build(sig, H, k, sw, DVt, ts, Om, DV, SIGS[sig])
        if d is None: continue
        cost = COST_1D * 288.0 / H; Hd = H // 288
        idx = d.index
        print(f"\n■ {sig} · {Hd}일 보유({H//STEP}트랜치) · k={k} · {'신호가중' if sw else '동일'} "
              f"· ≥${DVt/1e6:.0f}M · 하루비용 {cost:.1f}bp")
        for seg, m in (("표본내", idx < OOS_A), ("표본외", (idx >= OOS_A) & (idx <= OOS_B))):
            v = d.to_numpy().astype(float)[np.asarray(m)]
            v = v[np.isfinite(v)]
            if len(v) < 60: continue
            net = v - cost
            ac = [np.corrcoef(net[:-l], net[l:])[0, 1] for l in range(1, 11)]
            line = f"   {seg} n={len(net):>3} 순 {net.mean():>+6.1f}bp · ACF1..5 " \
                   f"{' '.join(f'{a:+.2f}' for a in ac[:5])}"
            print(line)
            for L in (1, 5, 10, 20):
                lo, hi = mbb_ci(net, L, rng)
                mark = "통과" if lo > 0 else "미달"
                print(f"      블록 {L:>2}일: [{lo:>+7.2f}, {hi:>+7.2f}] {mark}"
                      f"{'  <= H 이상 블록' if L >= Hd else ''}")
                if seg == "표본외" and L >= max(Hd, 5) and lo > 0: npass += 1
            # 비겹침 부분표본
            sub = net[::max(Hd, 1)]
            lo2, hi2 = mbb_ci(sub, 1, rng)
            print(f"      비겹침 {Hd}일 간격 (n={len(sub)}): 평균 {sub.mean():>+6.2f} "
                  f"[{lo2:>+7.2f}, {hi2:>+7.2f}] {'통과' if lo2 > 0 else '미달'}")
            print(f"      Newey-West t (lag={max(Hd,1)}) = {nw_t(net, max(Hd, 1)):+.2f}")
    print("\n" + "=" * 124)
    print(f"⭐표본외에서 **블록길이 ≥ max(H,5)일** 부트스트랩 CI 하한 > 0 인 (셀,블록) 조합: {npass}")
    print(json.dumps({"pass": npass}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
