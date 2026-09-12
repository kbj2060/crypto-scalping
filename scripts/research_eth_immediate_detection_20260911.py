"""즉시 감지 평가 — 예측이 아니라 "이미 시작된 돌파를 얼마나 빨리 알아채는가".

사용자 용도: 횡보 페이드 중 돌파가 시작되면, 반대 포지션이면 즉시 청산. 예측 불필요.
따라서 앞만 보는 타깃으로 동시 지표를 떨어뜨린 앞 검정은 이 용도에 잘못된 잣대였다.

핵심 지표 = **감지 시점 진행률** = (감지 시점까지 움직인 폭) / (전체 움직인 폭).
사용자 실제 사고에서 -0.28% / -1.92% = 15% 였다. 낮을수록 좋다.
기준점은 확장 직전 마지막 압축 봉의 종가다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/live_gap"
COMPRESS, EXPAND, BACK, FULL = 0.7, 1.8, 72, 144    # FULL=12시간 뒤까지 전체 이동폭


def main() -> int:
    d = pd.read_csv(D / "eth_5m_2026_tradecount.csv", parse_dates=["timestamp"])
    n = len(d)
    c, h, l = d.c.to_numpy(float), d.h.to_numpy(float), d.l.to_numpy(float)
    lr = np.diff(np.log(c), prepend=np.log(c[0]))
    volexp = (pd.Series(lr).rolling(12).std() / pd.Series(lr).rolling(288).std()).to_numpy()
    atr = pd.Series(np.maximum.reduce([h - l, np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))])
                    ).rolling(96).mean().to_numpy() / np.maximum(c, 1e-9)
    bbw = pd.Series(lr).rolling(48).std().rolling(288).rank(pct=True).to_numpy()
    atrr = pd.Series(atr).rolling(288).rank(pct=True).to_numpy()
    cs = np.clip(1.0 - np.maximum(bbw, atrr), 0, 1)
    imp = np.r_[0, np.diff(c) / c[:-1]] / np.maximum(atr, 1e-9)
    rel = np.clip(np.r_[0, cs[:-1]] * np.abs(imp), 0, 3) / 3
    z = lambda s, w=288: ((s - s.rolling(w).mean()) / s.rolling(w).std()).to_numpy()
    tb = d.taker_buy_ratio
    F = {"체결속도 n": z(d.n), "거래대금 qv": z(d.qv), "평균체결크기": z(d.avg_trade_size),
         "테이커 |쏠림|": z((tb - 0.5).abs()), "compression_release": rel,
         "|수익률|/ATR": np.abs(imp), "volexp 상승률": np.r_[0, np.diff(volexp)],
         "volexp 자체": volexp,
         "체결속도 3봉지속": pd.Series(z(d.n)).rolling(3).min().to_numpy()}

    cross = (volexp >= EXPAND) & np.r_[False, volexp[:-1] < EXPAND]
    was = pd.Series(volexp < COMPRESS).rolling(BACK).max().shift(12).to_numpy() == 1
    ev = np.flatnonzero(cross & was)
    ev = ev[(ev > 400) & (ev < n - FULL - 2)]
    comp = (volexp < COMPRESS)
    # 각 사건의 기준점 = 확장 직전 마지막 압축 봉
    starts, fulls = [], []
    for e in ev:
        w = np.flatnonzero(comp[max(e - BACK, 0):e]) 
        s = (max(e - BACK, 0) + w[-1]) if len(w) else e - 12
        starts.append(s)
        seg = c[s:min(e + FULL, n)]
        fulls.append(float(np.max(np.abs(seg - c[s])) / c[s] * 100))
    starts, fulls = np.asarray(starts), np.asarray(fulls)
    ok = fulls > 0.2                       # 전체 이동폭이 0.2% 미만이면 방어 의미 없음
    ev, starts, fulls = ev[ok], starts[ok], fulls[ok]
    print(f"[사건] {len(ev)}건 · 기준점=확장 직전 마지막 압축 봉")
    print(f"  전체 이동폭 중앙 {np.median(fulls):.2f}%  q75 {np.quantile(fulls,.75):.2f}%  "
          f"최대 {fulls.max():.2f}%")
    print(f"  확장 교차까지 지연 중앙 {np.median(ev-starts):.0f}봉 ({np.median(ev-starts)*5:.0f}분)")
    hours = float(comp.sum()) * 5 / 60
    print(f"\n{'피쳐':18s} {'임계':>7s} {'포착률':>7s} {'중앙지연':>8s} {'감지시 이동':>10s} "
          f"{'진행률':>7s} {'오탐/h':>7s}")
    rows = []
    for nm, x in F.items():
        for q in (0.90, 0.95, 0.99):
            thr = float(np.nanquantile(x[comp & np.isfinite(x)], q))
            fired = np.isfinite(x) & (x >= thr)
            dly, mvs, prog, hit = [], [], [], 0
            for e, s, fu in zip(ev, starts, fulls):
                k = np.flatnonzero(fired[s:min(e + 24, n)])
                if len(k):
                    t = s + k[0]
                    hit += 1
                    dly.append(t - s)
                    mv = abs(c[t] - c[s]) / c[s] * 100
                    mvs.append(mv); prog.append(mv / fu * 100)
            if hit < 20:
                continue
            fa = int((fired & comp).sum() - hit) / max(hours, 1)
            rows.append({"피쳐": nm, "q": q, "thr": thr, "recall": hit / len(ev),
                         "delay": float(np.median(dly)), "mv": float(np.median(mvs)),
                         "prog": float(np.median(prog)), "fa_h": fa})
            r = rows[-1]
            print(f"{nm:18s} {thr:7.2f} {r['recall']*100:6.1f}% {r['delay']:6.0f}봉 "
                  f"{r['mv']:9.3f}% {r['prog']:6.1f}% {fa:7.3f}", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(D / "immediate_detection.csv", index=False)
    print(f"\n=== 포착률 90% 이상 중 진행률 최저 ===")
    q = df[df.recall >= 0.90].nsmallest(6, "prog")
    print(q[["피쳐", "q", "recall", "delay", "mv", "prog", "fa_h"]].to_string(index=False,
          float_format=lambda v: f"{v:.3f}"))
    print("\n진행률 = 감지 시점 이동폭 / 전체 이동폭. 사용자 실제 사고는 15% 였다.")
    print("예측이 아니라 지연 최소화 문제다 — 동시 지표가 여기선 유리하다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
