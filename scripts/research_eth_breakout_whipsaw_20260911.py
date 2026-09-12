"""돌파 후 반전(휩쏘) 빈도 — 3봉 가다가 반대로 뒤집히는가.

사용자 질문: 추세가 바뀌었을 때 한 방향으로 3봉쯤 가다가 갑자기 반대로 가는 경우가 있나.
탐지 후 행동을 가른다 — 휩쏘가 흔하면 "반대로 뒤집기"는 위험하고 "청산해서 관망"이 맞다.

기준점 s = 전환 직전 마지막 압축 봉. 초기 방향 d = sign(c[s+3] - c[s]).
반전 = 이후 H봉 안에 가격이 s 를 지나 반대편으로 **초기 폭보다 크게** 간 경우.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
COMPRESS, EXPAND, BACK = 0.7, 1.8, 72
K = 3                       # 초기 방향을 재는 봉 수
HS = [(12, "1시간"), (48, "4시간"), (144, "12시간"), (288, "1일")]


def main() -> int:
    d = pd.read_csv(D / "eth_5m_2026_tradecount.csv", parse_dates=["timestamp"])
    n = len(d)
    c = d.c.to_numpy(float)
    ts = d.timestamp
    lr = np.diff(np.log(c), prepend=np.log(c[0]))
    volexp = (pd.Series(lr).rolling(12).std() / pd.Series(lr).rolling(288).std()).to_numpy()
    comp = volexp < COMPRESS
    cross = (volexp >= EXPAND) & np.r_[False, volexp[:-1] < EXPAND]
    was = pd.Series(comp).rolling(BACK).max().shift(12).to_numpy() == 1
    ev = np.flatnonzero(cross & was)
    ev = ev[(ev > 2100) & (ev < n - 300)]
    S = []
    for e in ev:
        w = np.flatnonzero(comp[max(e - BACK, 0):e])
        S.append((max(e - BACK, 0) + w[-1]) if len(w) else e - 12)
    S = np.asarray(S)
    ok = (S + K + 288) < n
    ev, S = ev[ok], S[ok]
    p0, p3 = c[S], c[S + K]
    init = (p3 - p0) / p0 * 100
    d0 = np.sign(init)
    live = np.abs(init) > 0.05                       # 초기 움직임이 있어야 방향이 의미 있다
    ev, S, p0, p3, init, d0 = ev[live], S[live], p0[live], p3[live], init[live], d0[live]
    print(f"[전환 {len(ev)}건 · 초기 {K}봉({K*5}분) 방향 기준 · 초기폭 |{np.median(np.abs(init)):.3f}%| 중앙]")
    print(f"  초기 방향 상승 {int((d0>0).sum())} / 하락 {int((d0<0).sum())}")
    print(f"\n{'지평':>7s} {'지속':>7s} {'반전':>7s} {'큰반전':>7s} {'중앙 순행':>9s} {'중앙 역행':>9s} "
          f"{'반전시 폭':>9s}")
    rows = []
    for H, hn in HS:
        cont, rev, big = 0, 0, 0
        fav, adv, revsz = [], [], []
        for i in range(len(ev)):
            s3 = S[i] + K
            seg = c[s3:min(s3 + H, n)]
            r = (seg - p0[i]) / p0[i] * 100 * d0[i]   # 초기 방향 기준 부호
            mx, mn = float(np.max(r)), float(np.min(r))
            fav.append(mx); adv.append(mn)
            end = float(r[-1])
            if end > 0:
                cont += 1
            else:
                rev += 1
                revsz.append(-end)
                if -end > abs(init[i]):               # 초기 폭보다 크게 반대로
                    big += 1
        rows.append({"H": hn, "cont": cont / len(ev), "rev": rev / len(ev), "big": big / len(ev),
                     "fav": float(np.median(fav)), "adv": float(np.median(adv)),
                     "revsz": float(np.median(revsz)) if revsz else np.nan})
        r0 = rows[-1]
        print(f"{hn:>7s} {r0['cont']*100:6.1f}% {r0['rev']*100:6.1f}% {r0['big']*100:6.1f}% "
              f"{r0['fav']:+8.3f}% {r0['adv']:+8.3f}% {r0['revsz']:+8.3f}%")
    pd.DataFrame(rows).to_csv(D / "breakout_whipsaw.csv", index=False)

    # 큰 반전 사례
    H = 144
    cases = []
    for i in range(len(ev)):
        s3 = S[i] + K
        seg = c[s3:min(s3 + H, n)]
        r = (seg - p0[i]) / p0[i] * 100 * d0[i]
        if float(r[-1]) < -abs(init[i]):
            cases.append((S[i], init[i], float(np.max(r)), float(r[-1])))
    cases.sort(key=lambda x: x[3])
    print(f"\n=== 12시간 안에 초기 폭보다 크게 뒤집힌 사례 상위 8 ({len(cases)}건 중) ===")
    print(f"{'기준 시각':17s} {'초기3봉':>8s} {'최대순행':>8s} {'최종':>8s}")
    for s, ini, mx, end in cases[:8]:
        print(f"{str(ts.iloc[int(s)]):17s} {ini:+7.3f}% {mx:+7.3f}% {end:+7.3f}%")
    print("\n'지속'은 H 시점에 초기 방향으로 남아 있는 비율. '큰반전'은 초기 폭보다 크게 반대로 간 비율.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
