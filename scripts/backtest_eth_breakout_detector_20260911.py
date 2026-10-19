"""라이브 모듈 그대로의 예상 성과 — 후행 창 임계(인과)로 2026 전 구간 재생.

연구 수치는 전역 분위수라 미세한 미래참조가 있다. 여기서는 라이브 모듈과 같은 규칙
(후행 2016개 **압축 봉**의 분위수, 1시간마다 갱신)으로 봉별 재생해 실제로 나올 값을 낸다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import live_eth_breakout_detector_20260911 as M  # noqa: E402

D = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
EXPAND, BACK, FULL, TOPQ, HPRED = 1.8, 72, 144, 0.95, 24
REFRESH = 12          # 임계 갱신 주기(1시간) — 라이브도 이 정도면 충분


def causal_thr(x, comp, q, win=M.QWIN, refresh=REFRESH):
    """봉 i 기준 직전 win 개 압축 봉의 분위수. refresh 봉마다 갱신하고 사이엔 유지."""
    n = len(x)
    out = np.full(n, np.inf)
    ci = np.flatnonzero(comp & np.isfinite(x))
    cur, ptr = np.inf, 0
    for i in range(n):
        while ptr < len(ci) and ci[ptr] < i:
            ptr += 1
        if i % refresh == 0:
            hist = ci[max(ptr - win, 0):ptr]
            cur = float(np.nanquantile(x[hist], q)) if len(hist) >= 200 else np.inf
        out[i] = cur
    return out


def main() -> int:
    d = pd.read_csv(D / "eth_5m_2026_tradecount.csv", parse_dates=["timestamp"])
    n = len(d)
    c = d.c.to_numpy(float)
    lr = np.diff(np.log(c), prepend=np.log(c[0]))
    volexp = (pd.Series(lr).rolling(12).std() / pd.Series(lr).rolling(288).std()).to_numpy()
    comp = volexp < M.COMPRESS
    watch = pd.Series(comp).rolling(12, min_periods=1).max().to_numpy() == 1
    zf = lambda col, w: ((d[col] - d[col].rolling(w).mean()) / d[col].rolling(w).std()).to_numpy()

    A = {}
    for label, col, w, sm, q, hz, lift in M.ALERT:
        x = zf(col, w)
        if sm > 1:
            x = pd.Series(x).rolling(sm).min().to_numpy()
        A[label] = comp & np.isfinite(x) & (x >= causal_thr(x, comp, q))
    T = {}
    for label, col, w, q in M.DETECT:
        x = volexp if col == "volexp" else zf(col, w)
        T[label] = watch & np.isfinite(x) & (x >= causal_thr(x, comp, q))
    alert = np.logical_or.reduce(list(A.values()))
    detect = np.logical_and.reduce(list(T.values()))

    days = (d.timestamp.iloc[-1] - d.timestamp.iloc[0]).total_seconds() / 86400
    fwd = pd.Series(lr).rolling(HPRED).std().shift(-HPRED).to_numpy()
    v = comp & np.isfinite(fwd)
    tgt = fwd >= np.nanquantile(fwd[v], TOPQ)
    base = float(np.mean(tgt[v]))
    print(f"[구간] {d.timestamp.iloc[0]:%Y-%m-%d} ~ {d.timestamp.iloc[-1]:%Y-%m-%d} "
          f"({days:.0f}일 · {n:,}봉) · 압축 {int(comp.sum()):,}봉 ({comp.mean():.0%})")
    print(f"\n=== 경보 신호기 (앞 2시간 상위 5% 타깃, 기저 {base*100:.2f}%) ===")
    print(f"{'신호':22s} {'발동':>7s} {'하루':>6s} {'적중률':>7s} {'lift':>6s}")
    for label, m in list(A.items()) + [("── 합집합(경보 ON)", alert)]:
        mm = m & v
        hit = float(np.mean(tgt[mm])) if mm.sum() >= 20 else np.nan
        print(f"{label:22s} {int(m.sum()):7,d} {m.sum()/days:5.1f}회 {hit*100:6.2f}% "
              f"{hit/base:5.2f}x")

    cross = (volexp >= EXPAND) & np.r_[False, volexp[:-1] < EXPAND]
    was = pd.Series(comp).rolling(BACK).max().shift(12).to_numpy() == 1
    ev = np.flatnonzero(cross & was)
    ev = ev[(ev > 2100) & (ev < n - FULL - 2)]
    starts, fulls = [], []
    for e in ev:
        w = np.flatnonzero(comp[max(e - BACK, 0):e])
        s = (max(e - BACK, 0) + w[-1]) if len(w) else e - 12
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

    def sc(f, nm):
        dly, prog, mv, hit = [], [], [], 0
        for e, s, fu in zip(ev, starts, fulls):
            kk = np.flatnonzero(f[s:min(e + 24, n)])
            if len(kk):
                t = s + kk[0]
                hit += 1
                dly.append(t - s)
                m2 = abs(c[t] - c[s]) / c[s] * 100
                mv.append(m2); prog.append(m2 / fu * 100)
        fa = int((f & watch & ~inwin).sum()) / max(off_h, 1)
        print(f"{nm:22s} {int(f.sum()):7,d} {hit/len(ev)*100:6.1f}% {np.median(dly):4.0f}봉 "
              f"{np.median(mv):8.3f}% {np.median(prog):6.2f}% {fa:7.3f} {fa*24:6.1f}회")

    print(f"\n=== 탐지 신호기 (전환 {len(ev)}건) ===")
    print(f"{'신호':22s} {'발동':>7s} {'포착률':>7s} {'지연':>5s} {'감지시이동':>9s} {'진행률':>7s} "
          f"{'오탐/h':>7s} {'하루':>6s}")
    for label, m in T.items():
        sc(m, label)
    sc(detect, "── 3종 AND (탐지 ON)")
    print(f"\n전체 이동폭 중앙 {np.median(fulls):.2f}% · q90 {np.quantile(fulls,.9):.2f}%")
    fu0 = 1.92
    pr = None
    dly, prog = [], []
    for e, s, fu in zip(ev, starts, fulls):
        kk = np.flatnonzero(detect[s:min(e + 24, n)])
        if len(kk):
            t = s + kk[0]
            prog.append(abs(c[t] - c[s]) / c[s] * 100 / fu * 100)
    pr = float(np.median(prog))
    print(f"\n[사용자 사고 환산] 전체 -1.92% 짜리였다면, 진행률 {pr:.1f}% 시점 "
          f"= 손실 {fu0*pr/100:.3f}% 에서 감지 (실제로는 15% = -0.288% 에서 알아챘다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
