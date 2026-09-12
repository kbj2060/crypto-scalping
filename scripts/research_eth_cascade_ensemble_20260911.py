"""캐스케이드 앙상블 — 느린 게이트 + 빠른 트리거.

3개 피쳐의 속도-강도가 다르다: 3봉지속 z2016 은 lift 7.71x(2시간)지만 15분 평활이라 느리고,
체결속도 n z864 는 5~15분에서 3.85x 로 빠르지만 약하다. 대칭 조합(평균·다수결)은 이 비대칭을
못 살린다.

구조: 게이트(느림)가 위험 구간을 열고, 그 안에서만 트리거(빠름)가 시점을 찍는다.
대조군: 각 단일 · 2/3 다수결 · 3개 AND · 게이트 없이 트리거만.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
COMPRESS, EXPAND, BACK, FULL, TOPQ, HPRED = 0.7, 1.8, 72, 144, 0.95, 24


def main() -> int:
    d = pd.read_csv(D / "eth_5m_2026_tradecount.csv", parse_dates=["timestamp"])
    n = len(d)
    c = d.c.to_numpy(float)
    lr = np.diff(np.log(c), prepend=np.log(c[0]))
    volexp = pd.Series(lr).rolling(12).std().to_numpy() / pd.Series(lr).rolling(288).std().to_numpy()
    comp = (volexp < COMPRESS) & np.isfinite(volexp)
    zf = lambda s, W: ((s - s.rolling(W).mean()) / s.rolling(W).std()).to_numpy()
    GATE = pd.Series(zf(d.n, 2016)).rolling(3).min().to_numpy()      # 느림·강함
    TRIG_N = zf(d.n, 864)                                            # 빠름
    TRIG_Q = zf(d.qv, 2016)                                          # 중간

    fwd = pd.Series(lr).rolling(HPRED).std().shift(-HPRED).to_numpy()
    vv = comp & np.isfinite(fwd)
    tgt = fwd >= np.nanquantile(fwd[vv], TOPQ)
    base_p = float(np.mean(tgt[vv]))

    cross = (volexp >= EXPAND) & np.r_[False, volexp[:-1] < EXPAND]
    was = pd.Series(volexp < COMPRESS).rolling(BACK).max().shift(12).to_numpy() == 1
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
    hours_off = float((comp & ~inwin).sum()) * 5 / 60

    def ev_score(fired):
        dly, prog, hit = [], [], 0
        for e, s, fu in zip(ev, starts, fulls):
            kk = np.flatnonzero(fired[s:min(e + 24, n)])
            if len(kk):
                tt = s + kk[0]
                hit += 1
                dly.append(tt - s)
                prog.append(abs(c[tt] - c[s]) / c[s] * 100 / fu * 100)
        m = comp & fired & np.isfinite(fwd)
        lift = float(np.mean(tgt[m])) / base_p if m.sum() >= 30 else np.nan
        return dict(recall=hit / len(ev), delay=float(np.median(dly)) if dly else np.nan,
                    prog=float(np.median(prog)) if prog else np.nan,
                    fa=int((fired & comp & ~inwin).sum()) / max(hours_off, 1),
                    lift=lift, fire=int((fired & comp).sum()))

    def q(x, p):
        return float(np.nanquantile(x[comp & np.isfinite(x)], p))

    print(f"전환 {len(ev)}건 · 예측 타깃 앞 {HPRED*5}분 상위 5% (기저 {base_p*100:.2f}%) "
          f"· 오탐 분모 {hours_off:,.0f}시간\n")
    print(f"{'구성':40s} {'발동':>7s} {'포착률':>7s} {'지연':>5s} {'진행률':>7s} {'오탐/h':>7s} {'lift':>6s}")
    rows = []

    def show(nm, fired):
        s = ev_score(fired)
        rows.append({"구성": nm, **s})
        print(f"{nm:40s} {s['fire']:7,d} {s['recall']*100:6.1f}% "
              f"{s['delay']:4.0f}봉 {s['prog']:6.2f}% {s['fa']:7.3f} {s['lift']:5.2f}x", flush=True)

    print("── 단일 (대조군)")
    for nm, x, p in (("게이트 3봉지속 z2016", GATE, 0.99), ("트리거 체결속도 n z864", TRIG_N, 0.90),
                     ("트리거 거래대금 qv z2016", TRIG_Q, 0.90)):
        show(f"{nm} q{p:.2f}", np.isfinite(x) & (x >= q(x, p)))
    print("\n── 대칭 조합 (대조군)")
    A = np.isfinite(GATE) & (GATE >= q(GATE, 0.90))
    B = np.isfinite(TRIG_N) & (TRIG_N >= q(TRIG_N, 0.90))
    C = np.isfinite(TRIG_Q) & (TRIG_Q >= q(TRIG_Q, 0.90))
    show("2/3 다수결 (q90)", (A.astype(int) + B.astype(int) + C.astype(int)) >= 2)
    show("3개 AND (q90)", A & B & C)
    show("트리거만 OR (게이트 없음)", B | C)

    print("\n── 캐스케이드 (게이트 열림 유지 W봉 → 트리거)")
    for gq in (0.90, 0.95, 0.99):
        gopen = np.isfinite(GATE) & (GATE >= q(GATE, gq))
        for W in (24, 48, 72):
            live = pd.Series(gopen).rolling(W, min_periods=1).max().to_numpy() == 1
            for tq in (0.85, 0.90, 0.95):
                trig = ((np.isfinite(TRIG_N) & (TRIG_N >= q(TRIG_N, tq)))
                        | (np.isfinite(TRIG_Q) & (TRIG_Q >= q(TRIG_Q, tq))))
                show(f"게이트 q{gq:.2f}·{W*5//60}h → 트리거 OR q{tq:.2f}", live & trig)
    df = pd.DataFrame(rows)
    df.to_csv(D / "cascade_ensemble.csv", index=False)
    ok = df[df.recall >= 0.95]
    print(f"\n=== 포착률 95%+ 중 오탐 최저 5 ===")
    print(ok.nsmallest(5, "fa")[["구성", "recall", "delay", "prog", "fa", "lift"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n=== lift 최고 5 (예측력) ===")
    print(df.nlargest(5, "lift")[["구성", "recall", "prog", "fa", "lift"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
