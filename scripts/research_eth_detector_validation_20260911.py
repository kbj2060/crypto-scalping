"""탐지·예측 확정 검증 — 오탐 재정의 + 예측 셀의 시간분할 전이.

① 오탐 버그: (fired & comp).sum() - hit 은 사건 창이 압축 밖일 때 음수가 된다.
   정정: **사건 창 합집합 밖에서 압축 중 발동한 수**를 센다.
② 선택 편향: 63셀 중 최대(7.71x)를 골랐다. 전반에서 고른 셀을 후반에서 재고,
   반대로도 잰다. 전이가 없으면 과적합이다. 63셀 분포도 같이 본다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
COMPRESS, EXPAND, BACK, FULL, TOPQ = 0.7, 1.8, 72, 144, 0.95
HS = [(2, "10분"), (3, "15분"), (4, "20분"), (6, "30분"), (9, "45분"), (12, "1시간"), (24, "2시간")]
WS = [288, 864, 2016]
SPLIT = pd.Timestamp("2026-05-01")


def main() -> int:
    d = pd.read_csv(D / "eth_5m_2026_tradecount.csv", parse_dates=["timestamp"])
    n = len(d)
    c = d.c.to_numpy(float)
    t = d.timestamp
    lr = np.diff(np.log(c), prepend=np.log(c[0]))
    volexp = pd.Series(lr).rolling(12).std().to_numpy() / pd.Series(lr).rolling(288).std().to_numpy()
    comp = (volexp < COMPRESS) & np.isfinite(volexp)
    zf = lambda s, W: ((s - s.rolling(W).mean()) / s.rolling(W).std()).to_numpy()
    H1 = (t < SPLIT).to_numpy()
    H2 = ~H1

    # ── ② 예측: 63셀 전수 + 시간분할 전이
    def cell(x, H, mask):
        fwd = pd.Series(lr).rolling(H).std().shift(-H).to_numpy()
        v = comp & mask & np.isfinite(fwd) & np.isfinite(x)
        if v.sum() < 1500:
            return None
        tgt = fwd >= np.nanquantile(fwd[v], TOPQ)
        thr = float(np.nanquantile(x[v], 0.99))
        m = v & (x >= thr)
        if m.sum() < 30:
            return None
        return float(np.mean(tgt[m])) / max(float(np.mean(tgt[v])), 1e-9)

    rows = []
    for nm in ("체결속도 n", "체결속도 3봉지속", "거래대금 qv"):
        for W in WS:
            b = zf(d.n, W) if nm.startswith("체결속도") else zf(d.qv, W)
            x = pd.Series(b).rolling(3).min().to_numpy() if "3봉" in nm else b
            for H, hn in HS:
                rows.append({"feat": nm, "W": W, "H": H, "hn": hn,
                             "all": cell(x, H, np.ones(n, bool)),
                             "h1": cell(x, H, H1), "h2": cell(x, H, H2)})
    g = pd.DataFrame(rows).dropna()
    print(f"=== 예측 63셀 분포 (lift) ===")
    print(f"  전체  중앙 {g['all'].median():.2f}x  q75 {g['all'].quantile(.75):.2f}x  "
          f"최대 {g['all'].max():.2f}x  ·  2x 이상 {int((g['all']>=2).sum())}/{len(g)}셀")
    print(f"  전반  중앙 {g.h1.median():.2f}x   후반  중앙 {g.h2.median():.2f}x")
    print(f"  전반↔후반 셀 순위 상관 (스피어만) {g.h1.corr(g.h2, method='spearman'):+.3f}")
    b1 = g.nlargest(1, "h1").iloc[0]
    b2 = g.nlargest(1, "h2").iloc[0]
    ba = g.nlargest(1, "all").iloc[0]
    print(f"\n  전반 최고셀  {b1.feat}/z{int(b1.W)}/{b1.hn}  전반 {b1.h1:.2f}x → **후반 {b1.h2:.2f}x**")
    print(f"  후반 최고셀  {b2.feat}/z{int(b2.W)}/{b2.hn}  후반 {b2.h2:.2f}x → **전반 {b2.h1:.2f}x**")
    print(f"  전체 최고셀  {ba.feat}/z{int(ba.W)}/{ba.hn}  전체 {ba['all']:.2f}x "
          f"(전반 {ba.h1:.2f}x / 후반 {ba.h2:.2f}x)")
    print(f"  전반 상위 5셀의 후반 중앙 {g.nlargest(5,'h1').h2.median():.2f}x "
          f"(전체 중앙 {g.h2.median():.2f}x)")
    g.to_csv(D / "prediction_cell_split.csv", index=False)

    # ── ① 탐지: 오탐 재정의
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
    inwin = np.zeros(n, bool)                      # 사건 창 합집합
    for e, s in zip(ev, starts):
        inwin[s:min(e + 24, n)] = True
    hours_off = float((comp & ~inwin).sum()) * 5 / 60
    print(f"\n=== 탐지 ({len(ev)}건 · 오탐 = 사건창 밖 압축 발동, 분모 {hours_off:,.0f}시간) ===")

    def score(fired):
        dly, prog, hit = [], [], 0
        for e, s, fu in zip(ev, starts, fulls):
            kk = np.flatnonzero(fired[s:min(e + 24, n)])
            if len(kk):
                tt = s + kk[0]
                hit += 1
                dly.append(tt - s)
                prog.append(abs(c[tt] - c[s]) / c[s] * 100 / fu * 100)
        fa = int((fired & comp & ~inwin).sum()) / max(hours_off, 1)
        return dict(recall=hit / len(ev), delay=float(np.median(dly)) if dly else np.nan,
                    prog=float(np.median(prog)) if prog else np.nan, fa=fa)

    print(f"{'탐지기':24s} {'포착률':>7s} {'지연':>5s} {'진행률':>7s} {'오탐/h':>7s}")
    FIRED = {}
    for nm in ("거래대금 qv", "체결속도 n", "volexp 자체"):
        x = volexp if nm == "volexp 자체" else zf(d.n if nm.startswith("체결") else d.qv, 288)
        thr = float(np.nanquantile(x[comp & np.isfinite(x)], 0.90))
        FIRED[nm] = np.isfinite(x) & (x >= thr)
        s = score(FIRED[nm])
        print(f"{nm+' z288 q90':24s} {s['recall']*100:6.1f}% {s['delay']:4.0f}봉 "
              f"{s['prog']:6.2f}% {s['fa']:7.3f}")
    ks = list(FIRED)
    A, B, C = FIRED[ks[0]], FIRED[ks[1]], FIRED[ks[2]]
    for nm2, f in (("OR (셋 중 하나)", A | B | C), ("AND (셋 모두)", A & B & C),
                   ("2/3 다수결", (A.astype(int) + B.astype(int) + C.astype(int)) >= 2)):
        s = score(f)
        print(f"{nm2:24s} {s['recall']*100:6.1f}% {s['delay']:4.0f}봉 {s['prog']:6.2f}% {s['fa']:7.3f}")
    print("\n오탐이 이제 음수가 될 수 없다. OR 의 오탐이 크면 축퇴(항상 켜짐)다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
