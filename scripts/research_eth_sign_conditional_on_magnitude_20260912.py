#!/usr/bin/env python3
"""부호 ⟂ 크기 분해 — «크기에 조건부인 부호» (2026-09-12).

근거 문헌: arXiv:2606.04153 = **Journal of Banking & Finance** 게재확정
(*A new decomposition approach to modeling financial returns: Conditioning sign on magnitude*).
수익을 부호×크기로 쪼개고 **크기에 조건부인 부호 분포**를 모델링하면 선형 예측회귀보다 낫다는 주장.
    E[r] = ∫ m · (2·P(부호=+ | m) − 1) · f(m) dm
무조건부 P(+)=0.5 여도 **P(+|m) 이 m 에 따라 변하면** 기대수익이 0 이 아니다.

## 왜 지금 이걸 보나 — 우리 데이터가 이미 그 구조를 보여줬다
15분 되돌림 검정에서 **이긴 봉 36.66bp vs 진 봉 47.89bp**(ETH 95~99% 대역)였다.
「되돌림 베팅이 이길 때는 작고 질 때는 크다」 = **큰 움직임은 되돌리지 않고 이어진다**
= 부호가 크기에 의존한다. 이 저장소는 크기 예측을 잘한다(변동성전망 AUC .836 · ATR .82).
⇒ 검정할 것: **작으면 되돌림 · 크면 지속** 이 실제로 성립하고 **거래 시점에 알 수 있는가**.

## 세 층
  A 서술   P(되돌림 | **실현** |r_next| 십분위) — 단조 하강이면 구조가 있다(거래 불가, 진단용)
  B 거래   **결정 시점에 아는** 크기 예측 m̂(직전 |r| EWMA)의 십분위별로
           되돌림/지속 각각의 적중·건당bp. ⚠️건당은 **실제 부호수익 평균**(비대칭 반영)
  C 비용   m̂ 임계 스윕 — 「예측 크기가 비용을 넘을 때만 거래」(arXiv:2606.00060 처방)
규약·비용은 15분 검정과 동일(봉 i 닫힘 → i+1 시가 진입 → 종가 청산 · peg 5.52 / 테이커 10bp).

자체점검 --selftest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import research_eth_15m_reversal_replication_20260912 as R15  # noqa: E402

PEG, TAKER = 5.52, 10.0
SYMBOLS = ("ETHUSDT", "BTCUSDT", "SOLUSDT", "XRPUSDT")


def mhat(r: np.ndarray, hl: int = 24) -> np.ndarray:
    """결정 시점에 아는 크기 예측 — |r| 의 EWMA. 봉 i 까지만 쓴다(i 포함, i+1 미포함)."""
    return pd.Series(np.abs(r)).ewm(halflife=hl, min_periods=hl).mean().to_numpy()


def signed_bp(bet: np.ndarray, nxt: np.ndarray) -> np.ndarray:
    """건당 실제 부호수익 bp. bet=+1 이면 다음 봉 수익 그대로, −1 이면 뒤집는다."""
    return bet * nxt * 1e4


def cell(bet: np.ndarray, nxt: np.ndarray, m: np.ndarray) -> tuple[int, float, float]:
    if m.sum() == 0:
        return 0, float("nan"), float("nan")
    v = signed_bp(bet[m], nxt[m])
    return int(m.sum()), float((v > 0).mean()), float(v.mean())


def pooled(start: str, end: str) -> None:
    """4심볼 풀링 — 상위 대역에서 «지속» 베팅.

    층 C 에서 상위 1% 만 부호가 뒤집혔는데(3/4 심볼) 심볼당 n=345 라 각각으로는 판정 불가다.
    풀링하면 n=1,370 인데 🔴**독립일은 19개뿐**이다 — 예측 변동성 상위 1% 봉은 며칠에 몰린다.
    그래서 일블록 부트와 독립일 수를 같이 낸다. n 을 세면 속는다.
    """
    rng = np.random.default_rng(20260912)
    rows = []
    for sym in SYMBOLS:
        s = R15.series(R15.fetch_15m(sym, start, end))
        r, ts = s["r"], s["ts"][:-1]
        prev, nxt, mh = np.sign(r[:-1]), r[1:], mhat(r)[:-1]
        ok = np.isfinite(mh) & (prev != 0)
        for q in (0.99, 0.95, 0.90):
            m = ok & (mh >= np.nanquantile(mh[ok], q))
            rows.append(pd.DataFrame({"q": q, "ts": ts[m], "cont_bp": prev[m] * nxt[m] * 1e4}))
    P = pd.concat(rows, ignore_index=True)
    print("\nD. 풀링(4심볼) — «지속» 베팅 · 예측크기 상위 대역")
    print(f"{'대역':>8}{'n':>7}{'🔴독립일':>9}{'적중':>8}{'건당bp':>9}{'peg후':>8}{'부트2.5%':>10}"
          f"{'순환p':>8}{'전반':>8}{'후반':>8}")
    for q in (0.99, 0.95, 0.90):
        S = P[P.q == q]
        v = S.cont_bp.to_numpy()
        day = pd.to_datetime(S.ts).dt.floor("D").to_numpy()
        ud = np.unique(day)
        by = {u: v[day == u] for u in ud}
        bs = np.array([np.concatenate([by[u] for u in rng.choice(ud, len(ud))]).mean() - PEG
                       for _ in range(2000)])
        nul = np.array([(np.roll(np.sign(v), int(rng.integers(50, len(v) - 50))) * np.abs(v)).mean()
                        for _ in range(400)])
        mid = len(S) // 2
        print(f"{q:>8.0%}{len(S):>7,}{len(ud):>9}{(v > 0).mean():>8.4f}{v.mean():>9.2f}"
              f"{v.mean() - PEG:>8.2f}{np.percentile(bs, 2.5):>10.2f}"
              f"{float((nul >= v.mean()).mean()):>8.3f}{v[:mid].mean():>8.2f}{v[mid:].mean():>8.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2025-09-01")
    ap.add_argument("--end", default="2026-08-20")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        r = np.array([0.01, -0.02, 0.03, -0.04, 0.05, 0.06])
        mh = mhat(r, hl=2)
        assert np.isnan(mh[0]), "워밍업 전에는 값이 없어야 한다"
        assert mh[-1] > 0
        bet = np.array([1.0, -1.0]); nxt = np.array([0.001, 0.002])
        v = signed_bp(bet, nxt)
        assert abs(v[0] - 10.0) < 1e-9 and abs(v[1] + 20.0) < 1e-9, v
        n_, w_, g_ = cell(bet, nxt, np.array([True, True]))
        assert n_ == 2 and w_ == 0.5 and abs(g_ + 5.0) < 1e-9, (n_, w_, g_)
        print("selftest OK — EWMA 워밍업 · 부호수익 · 셀 집계")
        return 0

    print("A. 서술 — P(되돌림 | **실현** |r_next| 십분위)   ⚠️거래 불가, 구조 확인용")
    print(f"{'심볼':<9}" + "".join(f"{'D'+str(i+1):>7}" for i in range(10)) + f"{'단조ρ':>8}")
    store = {}
    for sym in SYMBOLS:
        d = R15.fetch_15m(sym, a.start, a.end)
        s = R15.series(d)
        r = s["r"]
        prev, nxt = np.sign(r[:-1]), r[1:]
        rev = np.sign(nxt) != prev                      # 되돌림 발생?
        q = pd.qcut(np.abs(nxt), 10, labels=False, duplicates="drop")
        v = [float(rev[q == k].mean()) for k in range(10)]
        rho = float(np.corrcoef(np.arange(10), v)[0, 1])
        print(f"{sym:<9}" + "".join(f"{x:>7.3f}" for x in v) + f"{rho:>8.2f}")
        store[sym] = (r, prev, nxt)

    print("\nB. 거래 — 결정 시점 크기 예측 m̂(|r| EWMA hl=24) 십분위별")
    print("   되돌림 = 직전 봉 반대 · 지속 = 직전 봉 같은 방향 · 건당은 **실제 부호수익 평균**")
    for sym in SYMBOLS:
        r, prev, nxt = store[sym]
        mh = mhat(r)[:-1]
        ok = np.isfinite(mh) & (prev != 0)
        q = np.full(len(mh), -1)
        q[ok] = pd.qcut(mh[ok], 10, labels=False, duplicates="drop")
        print(f"\n  --- {sym} ---")
        print(f"{'십분위':>7}{'m̂ bp':>8}{'n':>7}" + f"{'되돌림 적중':>12}{'건당bp':>9}" + f"{'지속 적중':>11}{'건당bp':>9}")
        for k in range(10):
            m = q == k
            nr, wr, gr = cell(-prev, nxt, m)
            nc, wc, gc = cell(prev, nxt, m)
            print(f"{k+1:>7}{np.nanmean(mh[m])*1e4:>8.1f}{nr:>7,}{wr:>12.4f}{gr:>9.2f}{wc:>11.4f}{gc:>9.2f}")

    print("\nC. 비용 필터 — 「예측 크기 m̂ 이 임계를 넘을 때만 거래」 (arXiv:2606.00060 처방)")
    print(f"{'심볼':<9}{'임계(m̂ bp)':>12}{'거래비율':>9}{'n':>7}{'되돌림 건당':>12}{'peg후':>8}"
          f"{'지속 건당':>11}{'peg후':>8}")
    for sym in SYMBOLS:
        r, prev, nxt = store[sym]
        mh = mhat(r)[:-1]
        ok = np.isfinite(mh) & (prev != 0)
        for thr_q in (0.0, 0.5, 0.8, 0.9, 0.95, 0.99):
            thr = np.nanquantile(mh[ok], thr_q)
            m = ok & (mh >= thr)
            nr, wr, gr = cell(-prev, nxt, m)
            _, _, gc = cell(prev, nxt, m)
            print(f"{sym:<9}{thr*1e4:>12.1f}{m.sum()/ok.sum():>9.1%}{nr:>7,}"
                  f"{gr:>12.2f}{gr-PEG:>8.2f}{gc:>11.2f}{gc-PEG:>8.2f}")
        print()
    pooled(a.start, a.end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
