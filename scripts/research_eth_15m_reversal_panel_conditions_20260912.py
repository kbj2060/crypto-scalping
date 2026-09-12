#!/usr/bin/env python3
"""15분 되돌림 × 재료 패널 조건 (2026-09-12).

선행: `research_eth_15m_reversal_replication_20260912.py` — 직전 봉 크기 95~99% 대역에서
ETH 적중 .5754 · peg(5.52bp) 차감 평균 +0.72bp 인데 **일블록 부트 하한 −7.82** 로 갈렸다.
여기서는 그 대역에 재료 패널(증거신호 8 · 트리거 7 · 극점확률 · 문맥)의 조건을 더해
**부트 하한을 0 위로 올릴 수 있는가**를 본다.

## 🔴구조적 긴장 — 조건은 공짜가 아니다
조건을 붙이면 n 이 줄어 **부트 구간이 넓어진다**. 적중률 상승이 그 손실을 넘어야 한다.
그래서 «적중률이 올랐다» 로는 부족하고 **부트 하한 > 0** 을 기준으로 본다.

## 경계 계약 (15분 ↔ 5분 정렬)
15분 봉 i(시가 t)는 5분 봉 t, t+5, t+10 을 덮고 **t+15 에 닫힌다**. 그 봉이 닫힌 시점까지의
정보는 패널의 **t+10 행**(그 5분 봉이 t+15 에 닫힌다)이다. 진입은 15분 봉 i+1 시가 = t+15.
⇒ 피쳐 t+10 · 라벨 t+15 이후 → 「사건 라벨 경계 계약」 충족.

## 판정
기준선 = 조건 없는 대역. 조건이 **더하는 값**만 본다.
  선택  부트(일블록, B=2000) 2.5% 하한 > 0 · 두 반기 적중 > 0.5 · n ≥ 150
  가족  결과(다음 봉 수익)를 순환이동시켜 **화면 전체를 B_FAM 번 재실행** → 귀무 하 기대 통과 수
비용은 양다리 peg **5.52bp**(가장 싼 값). 여기서 안 되면 더 비싼 비용에서는 볼 것도 없다.

산출 tmp/eth_15m_reversal_20260912/panel_conditions.csv
자체점검 --selftest
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import research_eth_15m_reversal_replication_20260912 as R15  # noqa: E402
import research_eth_rule_direction_probability_20260912 as RD  # noqa: E402

PEG_BP = 5.52
LO_Q, HI_Q = 0.95, 0.99
MIN_N, NBOOT, B_FAM = 150, 2000, 12


def align(panel: pd.DataFrame, bar_open: np.ndarray) -> np.ndarray:
    """15분 봉 시가시각 → 그 봉이 닫힐 때까지의 정보를 담은 **5분 패널 행 인덱스**.

    15분 봉 t 는 t+15 에 닫히고, 그 시각에 닫히는 5분 봉의 **시가**는 t+10 이다.
    (패널의 timestamp 는 5분 봉 시가다.)
    """
    pos = pd.Series(np.arange(len(panel)), index=pd.DatetimeIndex(panel["timestamp"]))
    key = pd.DatetimeIndex(bar_open) + pd.Timedelta(minutes=10)
    return pos.reindex(key).to_numpy()


def block_lo(v: np.ndarray, day: np.ndarray, nboot: int = NBOOT) -> float:
    ud = np.unique(day)
    if len(ud) < 5:
        return float("nan")
    by = {u: v[day == u] for u in ud}
    rng = np.random.default_rng(20260912)
    bs = np.array([np.concatenate([by[u] for u in rng.choice(ud, len(ud))]).mean() for _ in range(nboot)])
    return float(np.percentile(bs, 2.5))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="ETHUSDT")
    ap.add_argument("--start", default="2025-09-01")
    ap.add_argument("--end", default="2026-08-20")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        pan = pd.DataFrame({"timestamp": pd.date_range("2026-01-01", periods=6, freq="5min")})
        idx = align(pan, np.array([np.datetime64("2026-01-01T00:00"), np.datetime64("2026-01-01T00:15")]))
        assert idx[0] == 2, idx          # 00:00 15분봉 → 00:10 5분봉(00:15 에 닫힌다)
        assert idx[1] == 5, idx
        v = np.array([1.0, 2.0, 3.0, 4.0])
        d = np.array(["a", "a", "b", "b"], dtype=object)
        assert np.isnan(block_lo(v, d)), "블록이 5개 미만이면 부트를 내지 않는다"
        print("selftest OK — 15분↔5분 정렬(+10분) · 블록 부족 시 NaN")
        return 0

    panel = pd.read_parquet(RD.PANEL)
    D, F = RD.atoms(panel)
    d15 = R15.fetch_15m(a.symbol, a.start, a.end)
    s = R15.series(d15)
    r, ts = s["r"], s["ts"]
    sig, nxt, mag = -np.sign(r[:-1]), r[1:], np.abs(r[:-1])
    lo, hi = np.quantile(mag, [LO_Q, HI_Q])
    band = (mag >= lo) & (mag < hi)
    pidx = align(panel, ts[:-1])
    ok = band & np.isfinite(pidx)
    ev = np.flatnonzero(ok)
    prow = pidx[ev].astype(int)
    print(f"{a.symbol} 15분봉 {len(r):,} · 대역 {band.sum():,} · 패널 정렬 성공 **{len(ev):,}**")

    win = (np.sign(nxt[ev]) == sig[ev])
    net = np.where(win, 1, -1) * np.abs(nxt[ev]) * 1e4 - PEG_BP
    day = pd.to_datetime(ts[ev]).floor("D").to_numpy()
    half = len(ev) // 2
    base_lo = block_lo(net, day)
    print(f"기준선(조건 없음): n={len(ev):,} 적중 {win.mean():.4f} · 평균순익 {net.mean():+.2f}bp · "
          f"부트하한 {base_lo:+.2f}\n")

    # 원자: 방향 원자는 되돌림 «방향»과 측면이 맞을 때만 의미가 있다 → 부호 일치 여부로 건다
    cands: list[tuple[str, np.ndarray]] = []
    for name, (mask, side) in D.items():
        m = mask[prow] & (sig[ev] == side)          # 그 측면 신호가 켜졌고 되돌림 방향도 같다
        cands.append((f"{name}(측면일치)", m))
    for name, mask in F.items():
        cands.append((name, mask[prow]))
    pairs = [(f"{n1} + {n2}", m1 & m2) for (n1, m1), (n2, m2) in itertools.combinations(cands, 2)]
    allc = cands + pairs
    print(f"조건 후보 {len(allc):,}개 (단일 {len(cands)} · 쌍 {len(pairs):,})")

    def screen(w: np.ndarray, nt: np.ndarray) -> list[dict]:
        keep = []
        for nm, m in allc:
            n = int(m.sum())
            if n < MIN_N:
                continue
            h1 = w[m & (np.arange(len(ev)) < half)].mean() if (m & (np.arange(len(ev)) < half)).sum() else 0
            h2 = w[m & (np.arange(len(ev)) >= half)].mean() if (m & (np.arange(len(ev)) >= half)).sum() else 0
            if not (h1 > 0.5 and h2 > 0.5):
                continue
            lo_ = block_lo(nt[m], day[m], 400)
            if not (lo_ > 0):
                continue
            keep.append({"cond": nm, "n": n, "hit": float(w[m].mean()), "net": float(nt[m].mean()),
                         "boot_lo": lo_, "h1": float(h1), "h2": float(h2),
                         "per_day": n / max((pd.Timestamp(ts[ev][-1]) - pd.Timestamp(ts[ev][0])).days, 1)})
        return keep

    t0 = time.time()
    passed = screen(win, net)
    print(f"통과 {len(passed)}개 ({time.time()-t0:.0f}s)")
    rng = np.random.default_rng(20260912)
    fam = []
    for b in range(B_FAM):
        sh = int(rng.integers(50, len(ev) - 50))
        nt2 = np.roll(net, sh)
        w2 = np.roll(win, sh)
        fam.append(len(screen(w2, nt2)))
        print(f"  가족귀무 {b+1}/{B_FAM}: {fam[-1]}", flush=True)
    print(f"\n{'='*104}")
    print(f"조건 {len(allc):,}개 중 통과 **{len(passed)}** · **귀무 하 기대 {np.mean(fam):.1f}** (최대 {max(fam)})")
    print("=" * 104)
    if passed:
        df = pd.DataFrame(passed).sort_values("boot_lo", ascending=False)
        df.to_csv(R15.OUT / "panel_conditions.csv", index=False)
        print(f"{'조건':<52}{'n':>6}{'건/일':>7}{'적중':>8}{'전반':>7}{'후반':>7}{'순익':>8}{'부트하한':>9}")
        for x in df.head(25).itertuples():
            print(f"{x.cond[:50]:<52}{x.n:>6}{x.per_day:>7.2f}{x.hit:>8.4f}{x.h1:>7.3f}{x.h2:>7.3f}"
                  f"{x.net:>8.2f}{x.boot_lo:>9.2f}")
    else:
        print("  통과 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
