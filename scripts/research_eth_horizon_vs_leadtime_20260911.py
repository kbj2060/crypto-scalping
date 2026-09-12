"""지평 곡선과 실제 리드타임 — "짧을수록 쉬운가" 와 "몇 분 먼저 아는가" 를 분리한다.

혼동 정리:
  지평 H   = 발동 후 H 안에 확장이 오는가 (타깃 창)
  리드타임 = 실제 발동 시각 - 전환 시각 (음수면 먼저)
둘은 다른 양이다. 하루 전에 안다는 주장은 어디에도 없다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
COMPRESS, EXPAND, BACK, TOPQ = 0.7, 1.8, 72, 0.95
HS = [(1, "5분"), (2, "10분"), (3, "15분"), (4, "20분"), (6, "30분"), (9, "45분"),
      (12, "1시간"), (24, "2시간"), (48, "4시간"), (96, "8시간"), (144, "12시간"), (288, "1일")]


def main() -> int:
    d = pd.read_csv(D / "eth_5m_2026_tradecount.csv", parse_dates=["timestamp"])
    n = len(d)
    c = d.c.to_numpy(float)
    lr = np.diff(np.log(c), prepend=np.log(c[0]))
    volexp = pd.Series(lr).rolling(12).std().to_numpy() / pd.Series(lr).rolling(288).std().to_numpy()
    comp = (volexp < COMPRESS) & np.isfinite(volexp)
    zf = lambda s, W: ((s - s.rolling(W).mean()) / s.rolling(W).std()).to_numpy()
    FE = {"체결속도 n (z864)": zf(d.n, 864),
          "체결속도 3봉지속 (z2016)": pd.Series(zf(d.n, 2016)).rolling(3).min().to_numpy(),
          "거래대금 qv (z2016)": zf(d.qv, 2016)}

    print("=== 지평 곡선 (타깃=앞 H봉 실현변동성 상위 5%, 기저 5% 고정, 발동 q99) ===")
    print(f"{'피쳐':24s}" + "".join(f"{hn:>7s}" for _, hn in HS))
    for nm, x in FE.items():
        line = f"{nm:24s}"
        for H, _ in HS:
            fwd = (pd.Series(lr).rolling(max(H, 2)).std().shift(-H).to_numpy() if H > 1
                   else np.abs(np.r_[lr[1:], np.nan]))
            v = comp & np.isfinite(fwd) & np.isfinite(x)
            tgt = fwd >= np.nanquantile(fwd[v], TOPQ)
            thr = float(np.nanquantile(x[v], 0.99))
            m = v & (x >= thr)
            lift = float(np.mean(tgt[m])) / max(float(np.mean(tgt[v])), 1e-9) if m.sum() >= 30 else np.nan
            line += f"{lift:7.2f}"
        print(line, flush=True)

    # ── 실제 리드타임
    cross = (volexp >= EXPAND) & np.r_[False, volexp[:-1] < EXPAND]
    was = pd.Series(volexp < COMPRESS).rolling(BACK).max().shift(12).to_numpy() == 1
    ev = np.flatnonzero(cross & was)
    ev = ev[(ev > 2100) & (ev < n - 300)]
    print(f"\n=== 실제 리드타임 (전환 {len(ev)}건 · 창 [-6시간, +2시간]) ===")
    print(f"{'피쳐':24s} {'컷':>6s} {'포착률':>7s} {'q10':>7s} {'중앙':>7s} {'q90':>7s} {'먼저울린 비율':>12s}")
    for nm, x in FE.items():
        for q in (0.99, 0.95, 0.90):
            thr = float(np.nanquantile(x[comp & np.isfinite(x)], q))
            fire = np.flatnonzero(comp & np.isfinite(x) & (x >= thr))
            leads = []
            for e in ev:
                k = fire[(fire >= e - 72) & (fire <= e + 24)]
                if len(k):
                    leads.append((k[0] - e) * 5)
            if len(leads) < 30:
                continue
            L = np.asarray(leads)
            print(f"{nm:24s} {f'q{q:.2f}':>6s} {len(L)/len(ev)*100:6.1f}% "
                  f"{np.quantile(L,.1):+6.0f}분 {np.median(L):+6.0f}분 {np.quantile(L,.9):+6.0f}분 "
                  f"{float((L<0).mean())*100:11.1f}%")
    print("\n지평 ≠ 리드타임. 지평은 '앞으로 얼마 안에', 리드타임은 '몇 분 먼저'다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
