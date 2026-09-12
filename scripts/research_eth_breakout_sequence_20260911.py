"""경보 → 탐지 → 전환 순서가 실제로 일어나는가.

세 사건이 그 순서로 나야 신호기 두 개가 한 구조로 작동한다. 순서가 안 지켜지면
따로 노는 두 지표일 뿐이다. 우연 대비도 잰다(순환이동으로 경보 시계열만 이동).
실제 사례를 날짜·가격과 함께 뽑아 눈으로 확인할 수 있게 한다.
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
from backtest_eth_breakout_detector_20260911 import causal_thr  # noqa: E402

D = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
EXPAND, BACK, FULL = 1.8, 72, 144
LOOK = 72          # 전환 전 6시간 안에서 경보/탐지를 찾는다
B_NULL, SEED = 400, 615372041


def main() -> int:
    d = pd.read_csv(D / "eth_5m_2026_tradecount.csv", parse_dates=["timestamp"])
    n = len(d)
    c = d.c.to_numpy(float)
    ts = d.timestamp
    lr = np.diff(np.log(c), prepend=np.log(c[0]))
    volexp = (pd.Series(lr).rolling(12).std() / pd.Series(lr).rolling(288).std()).to_numpy()
    comp = volexp < M.COMPRESS
    watch = pd.Series(comp).rolling(12, min_periods=1).max().to_numpy() == 1
    zf = lambda col, w: ((d[col] - d[col].rolling(w).mean()) / d[col].rolling(w).std()).to_numpy()
    A = []
    for label, col, w, sm, q, hz, lf in M.ALERT:
        x = zf(col, w)
        if sm > 1:
            x = pd.Series(x).rolling(sm).min().to_numpy()
        A.append(comp & np.isfinite(x) & (x >= causal_thr(x, comp, q)))
    alert = np.logical_or.reduce(A)
    T = []
    for label, col, w, q in M.DETECT:
        x = volexp if col == "volexp" else zf(col, w)
        T.append(watch & np.isfinite(x) & (x >= causal_thr(x, comp, q)))
    detect = np.logical_and.reduce(T)

    cross = (volexp >= EXPAND) & np.r_[False, volexp[:-1] < EXPAND]
    was = pd.Series(comp).rolling(BACK).max().shift(12).to_numpy() == 1
    ev = np.flatnonzero(cross & was)
    ev = ev[(ev > 2100) & (ev < n - FULL - 2)]
    rows = []
    for e in ev:
        w = np.flatnonzero(comp[max(e - BACK, 0):e])          # 기준점 = 전환 직전 마지막 압축 봉
        st = (max(e - BACK, 0) + w[-1]) if len(w) else e - 12
        ai = np.flatnonzero(alert[max(e - LOOK, 0):st + 1])   # 경보는 기준점까지(사전)
        di = np.flatnonzero(detect[st:e + 24])                # 탐지는 기준점 이후(즉시)
        a0 = (max(e - LOOK, 0) + ai[-1]) if len(ai) else None  # 기준점에 **가장 가까운** 경보
        d0 = (st + di[0]) if len(di) else None
        seg = c[(a0 if a0 is not None else e):min(e + FULL, n)]
        mv = float((np.max(np.abs(seg - seg[0])) / seg[0]) * 100) if len(seg) > 2 else np.nan
        rows.append({"e": e, "a": a0, "d": d0, "move": mv,
                     "s": st,
                     "seq": (a0 is not None and d0 is not None and a0 <= d0 <= e + 24)})
    r = pd.DataFrame(rows)
    na, nd = int(r.a.notna().sum()), int(r.d.notna().sum())
    ns = int(r.seq.sum())
    print(f"[전환 {len(r)}건 · 전환 전 {LOOK*5//60}시간 안에서 탐색]")
    print(f"  경보가 먼저 있었다        {na:4d}건 ({na/len(r)*100:5.1f}%)")
    print(f"  탐지가 있었다             {nd:4d}건 ({nd/len(r)*100:5.1f}%)")
    print(f"  ⭐경보 → 탐지 → 전환 순서  {ns:4d}건 ({ns/len(r)*100:5.1f}%)")

    rng = np.random.default_rng(SEED)
    nulls = []
    for s in rng.integers(900, n - 900, size=B_NULL):
        al = np.roll(alert, int(s))
        k = 0
        for e in ev:
            w = np.flatnonzero(comp[max(e - BACK, 0):e])
            st = (max(e - BACK, 0) + w[-1]) if len(w) else e - 12
            ai = np.flatnonzero(al[max(e - LOOK, 0):st + 1])
            di = np.flatnonzero(detect[st:e + 24])
            if len(ai) and len(di):
                k += 1
        nulls.append(k / len(ev))
    nulls = np.asarray(nulls)
    print(f"  순환이동 귀무(경보만 이동)  중앙 {np.median(nulls)*100:5.1f}% · "
          f"q95 {np.quantile(nulls,.95)*100:5.1f}% · p={float((nulls>=ns/len(r)).mean()):.3f}")

    ok = r[r.seq].copy()
    ok["a_pre"] = (ok.s - ok.a) * 5      # 기준점 기준 경보 선행(분)
    ok["d_dly"] = (ok.d - ok.s) * 5      # 기준점 기준 탐지 지연(분)
    print(f"\n  [기준점 = 전환 직전 마지막 압축 봉. 창 폭 산물을 피하려면 이 기준으로 잰다]")
    print(f"  경보 선행  중앙 {ok.a_pre.median():.0f}분 (q25 {ok.a_pre.quantile(.25):.0f} / "
          f"q75 {ok.a_pre.quantile(.75):.0f})")
    print(f"  탐지 지연  중앙 {ok.d_dly.median():.0f}분 (q75 {ok.d_dly.quantile(.75):.0f} / "
          f"q90 {ok.d_dly.quantile(.90):.0f})")
    print(f"  기준점→전환(volexp 확정) 중앙 {((r.e-r.s)*5).median():.0f}분 — 확정은 원래 느리다")
    print(f"\n=== 실제 사례 상위 10건 (경보 시점 이후 최대 이동폭 순) ===")
    print(f"{'경보':17s} {'탐지':>6s} {'전환':>6s} {'가격':>9s} {'이동폭':>7s}")
    for x in ok.nlargest(10, "move").itertuples():
        print(f"{str(ts.iloc[int(x.a)]):17s} {str(ts.iloc[int(x.d)])[11:16]:>6s} "
              f"{str(ts.iloc[int(x.e)])[11:16]:>6s} {c[int(x.a)]:9.2f} {x.move:6.2f}%")
    r.to_csv(D / "breakout_sequence.csv", index=False)
    print(f"\n순서 비율이 귀무를 넘지 못하면 두 신호기는 한 구조가 아니라 따로 노는 지표다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
