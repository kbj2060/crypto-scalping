"""탐지 AND 성분 제거 시험 — volexp 를 빼야 하는가.

volexp 는 전환의 **정의**(volexp>=1.8 교차)에 쓰인 양이다. 탐지기에 넣으면 자기 정답의
일부를 들고 있는 셈이라, 기여가 정의상 보장된 것이지 정보가 아닐 수 있다.
단독 성적도 가장 나쁘다(진행률 6.44%, 오탐 113회/일).
"""
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import live_eth_breakout_detector_20260911 as M  # noqa: E402
from backtest_eth_breakout_detector_20260911 import causal_thr  # noqa: E402

D = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
EXPAND, BACK, FULL = 1.8, 72, 144


def main() -> int:
    d = pd.read_csv(D / "eth_5m_2026_tradecount.csv", parse_dates=["timestamp"])
    n = len(d)
    c = d.c.to_numpy(float)
    lr = np.diff(np.log(c), prepend=np.log(c[0]))
    volexp = (pd.Series(lr).rolling(12).std() / pd.Series(lr).rolling(288).std()).to_numpy()
    comp = volexp < M.COMPRESS
    watch = pd.Series(comp).rolling(12, min_periods=1).max().to_numpy() == 1
    zf = lambda col, w: ((d[col] - d[col].rolling(w).mean()) / d[col].rolling(w).std()).to_numpy()
    F = {}
    for label, col, w, q in M.DETECT:
        x = volexp if col == "volexp" else zf(col, w)
        F[label] = (x, w, q)

    cross = (volexp >= EXPAND) & np.r_[False, volexp[:-1] < EXPAND]
    was = pd.Series(comp).rolling(BACK).max().shift(12).to_numpy() == 1
    ev = np.flatnonzero(cross & was)
    ev = ev[(ev > 2100) & (ev < n - FULL - 2)]
    starts, fulls = [], []
    for e in ev:
        w0 = np.flatnonzero(comp[max(e - BACK, 0):e])
        s = (max(e - BACK, 0) + w0[-1]) if len(w0) else e - 12
        starts.append(s)
        seg = c[s:min(e + FULL, n)]
        fulls.append(float(np.max(np.abs(seg - c[s])) / c[s] * 100))
    starts, fulls = np.asarray(starts), np.asarray(fulls)
    k = fulls > 0.2
    ev, starts, fulls = ev[k], starts[k], fulls[k]
    inwin = np.zeros(n, bool)
    for e, s in zip(ev, starts):
        inwin[s:min(e + 24, n)] = True
    off_h = float((watch & ~inwin).sum()) * 5 / 60
    days = (d.timestamp.iloc[-1] - d.timestamp.iloc[0]).total_seconds() / 86400

    def sc(f):
        dly, prog, hit = [], [], 0
        for e, s, fu in zip(ev, starts, fulls):
            kk = np.flatnonzero(f[s:min(e + 24, n)])
            if len(kk):
                t = s + kk[0]
                hit += 1
                dly.append((t - s) * 5)
                prog.append(abs(c[t] - c[s]) / c[s] * 100 / fu * 100)
        return dict(recall=hit / len(ev), dly=float(np.median(dly)) if dly else np.nan,
                    prog=float(np.median(prog)) if prog else np.nan,
                    fa=int((f & watch & ~inwin).sum()) / max(off_h, 1))

    print(f"[전환 {len(ev)}건 · 인과 임계 · q 는 각 피쳐 q90]")
    print(f"{'조합':34s} {'포착률':>7s} {'지연':>6s} {'진행률':>7s} {'헛발동/일':>9s}")
    rows = []
    names = list(F)
    for r in (1, 2, 3):
        for cb in itertools.combinations(names, r):
            fired = np.ones(n, bool)
            for nm in cb:
                x, w, q = F[nm]
                fired &= watch & np.isfinite(x) & (x >= causal_thr(x, comp, q))
            s = sc(fired)
            lab = " + ".join(cb) if r > 1 else cb[0] + " (단독)"
            rows.append({"조합": lab, "k": r, **s})
            print(f"{lab:34s} {s['recall']*100:6.1f}% {s['dly']:4.0f}분 {s['prog']:6.2f}% "
                  f"{s['fa']*24:8.1f}회", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(D / "detect_and_ablation.csv", index=False)
    a3 = df[df.k == 3].iloc[0]
    a2 = df[(df.k == 2) & (~df.조합.str.contains("변동성확장비"))].iloc[0]
    print(f"\n[핵심 비교] volexp 를 빼면:")
    print(f"  3종 AND  포착 {a3.recall*100:.1f}% · 지연 {a3.dly:.0f}분 · 진행률 {a3.prog:.2f}% · "
          f"헛발동 {a3.fa*24:.1f}회/일")
    print(f"  2종 AND  포착 {a2.recall*100:.1f}% · 지연 {a2.dly:.0f}분 · 진행률 {a2.prog:.2f}% · "
          f"헛발동 {a2.fa*24:.1f}회/일")
    print(f"  → 진행률 {a2.prog - a3.prog:+.2f}%p · 헛발동 {(a2.fa - a3.fa)*24:+.1f}회/일")
    print("\nvolexp 는 전환 정의에 쓰인 양이다 — 기여가 정의상 보장된 것인지 정보인지 여기서 갈린다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
