#!/usr/bin/env python3
"""liquidity_sweep Phase 1 — **메커니즘 기반 조건부 필터** 7축 (2026-09-14).

Phase 0(`research_eth_liquidity_sweep_block_independence_20260914.py`)에서 전건 sweep 의
1,741일 초과분이 음수로 나왔다. 그래도 «어떤 부분집합은 되는가»는 별개 질문이므로,
격자 최고점이 아니라 **메커니즘이 예측하는 방향의 단조성**으로 판정한다.

Osler(2003, JoF) 의 실제 발견 = 스톱주문이 직전 고/저점·라운드넘버 위에 군집해 캐스케이드를
만든다. 그 메커니즘이 참이면 다음이 **단조**여야 한다:
  D1 depth_atr   침투가 깊을수록(스톱을 더 많이 쓸어담을수록) 되돌림이 크다
  D2 touch_n     레벨을 여러 번 건드렸을수록(EQH/EQL 휴면유동성 축적) 크다
  D3 level_age   레벨이 오래 서 있었을수록 크다
  D4 wick        되돌림 꼬리가 길수록(빠른 거부) 크다
  D5 round_d     라운드넘버에 가까울수록(작을수록) 크다  <- Osler 본연
  D6 atr_pct     변동성이 클수록 배리어 b 가 커져 (2a-1)b 가 비용을 넘는다
  D7 btc_nonconf BTC 비확인(SMT 잔여분)일수록 크다

판정(사전등록): ①분위 단조(|Spearman rho| 유의) ②최고분위 초과 CI95 가 0 배제
③전·후반 부호 일치 ④최고분위 초과 > 5.52bp(peg 왕복 실측). 넷 다여야 후보.
7축 × 2측면 = 14셀이므로 «몇 개 통과»는 **랜덤 분위 귀무(B=200)** 와 대조한다.

출력: tmp/eth_liquidity_sweep_20260914/phase1_*.csv
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))
import research_eth_liquidity_sweep_block_independence_20260914 as P0  # noqa: E402

OUT = ROOT / "tmp/eth_liquidity_sweep_20260914"
H = 48                  # 09-09 의 그 셀
NQ = 5                  # 분위
LOOKBACK = 48           # SWEEP_LOOKBACK 과 동일
ROUND_GRID = 50.0       # ETH 라운드넘버 격자($50). Osler 의 FX 는 00/50 pip 였다
PEG_RT = 5.52           # peg 양다리 왕복 실측(bp)
RNG = np.random.default_rng(20260914)


def rolling_argextreme(arr: np.ndarray, w: int, want_min: bool) -> np.ndarray:
    """봉 i-1 까지 닫힌 w봉 창에서 극점이 **선 위치(절대 인덱스)**. 없으면 -1."""
    n = len(arr); pos = np.full(n, -1, dtype=np.int64)
    if n <= w: return pos
    win = np.lib.stride_tricks.sliding_window_view(arr, w)
    idx = np.arange(w, n); j = idx - w
    off = win[j].argmin(axis=1) if want_min else win[j].argmax(axis=1)
    pos[idx] = j + off
    return pos


def selftest() -> None:
    a = np.array([5., 1., 9., 3., 7.])
    # w=2: i=2 는 창 [5,1] -> min 위치 1 · max 위치 0
    p = rolling_argextreme(a, 2, True); assert p[2] == 1, p
    p = rolling_argextreme(a, 2, False); assert p[2] == 0, p
    # 라운드넘버 거리: 2475 -> 가장 가까운 50배수 2475 자신 -> 0 ; 2480 -> 5
    d = lambda x: abs(x - round(x / ROUND_GRID) * ROUND_GRID)
    assert d(2450.0) == 0.0 and d(2460.0) == 10.0
    print("selftest OK")


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest: selftest(); return 0
    selftest()
    OUT.mkdir(parents=True, exist_ok=True)
    import live_evidence_signal_dashboard_20260823 as EV  # noqa: E402

    kl = P0.load(P0.CSV); btc = P0.load(P0.BTC)
    sig = EV.compute_signals(kl, btc_df=btc, funding_df=None)
    op = sig["open"].to_numpy(float); cl = sig["close"].to_numpy(float)
    hi_a = sig["high"].to_numpy(float); lo_a = sig["low"].to_numpy(float)
    atr_abs = (sig["atr_pct"].to_numpy(float) * cl)
    day = sig["timestamp"].dt.floor("D").astype("int64").to_numpy()
    n = len(sig); lo, hi = P0.WARMUP, n - max(P0.HORIZONS) - 2

    swing_lo = pd.Series(lo_a).rolling(LOOKBACK, min_periods=LOOKBACK).min().shift(1).to_numpy()
    swing_hi = pd.Series(hi_a).rolling(LOOKBACK, min_periods=LOOKBACK).max().shift(1).to_numpy()
    pos_lo = rolling_argextreme(lo_a, LOOKBACK, True)
    pos_hi = rolling_argextreme(hi_a, LOOKBACK, False)
    # BTC 자기 극점 비확인(SMT 잔여) — 같은 봉에서 BTC 는 극점을 안 깼는가
    b = sig[["btc_low", "btc_high"]].to_numpy(float) if "btc_low" in sig.columns else None
    if b is None:
        bm = btc.set_index("timestamp").reindex(sig["timestamp"])
        b = bm[["low", "high"]].to_numpy(float)
    bsl = pd.Series(b[:, 0]).rolling(LOOKBACK, min_periods=LOOKBACK).min().shift(1).to_numpy()
    bsh = pd.Series(b[:, 1]).rolling(LOOKBACK, min_periods=LOOKBACK).max().shift(1).to_numpy()

    rows, passes = [], 0
    for side, long in (("top", False), ("bottom", True)):
        f = sig[f"{side}_liquidity_sweep"].fillna(False).to_numpy(bool)
        idx = np.flatnonzero(f); idx = idx[(idx >= lo) & (idx <= hi)]
        g = P0.gross_bp(op, cl, idx, H, long)
        null_mean = P0.same_side_null(op, cl, len(idx), H, long, lo, hi).mean()
        lvl = (swing_hi if side == "top" else swing_lo)[idx]
        a_i = atr_abs[idx]
        depth = ((hi_a[idx] - lvl) if side == "top" else (lvl - lo_a[idx])) / np.maximum(a_i, 1e-9)
        age = idx - (pos_hi if side == "top" else pos_lo)[idx]
        tol = 0.15 * a_i
        touch = np.array([                       # 레벨 ±0.15ATR 를 창 안에서 몇 봉이 건드렸나
            int(np.sum(np.abs((hi_a if side == "top" else lo_a)[max(i-LOOKBACK, 0):i] - L) <= t))
            for i, L, t in zip(idx, lvl, tol)])
        wick = sig[("upper_wick_ratio" if side == "top" else "lower_wick_ratio")].to_numpy(float)[idx]
        round_d = np.abs(lvl - np.round(lvl / ROUND_GRID) * ROUND_GRID) / np.maximum(a_i, 1e-9)
        atrp = sig["atr_pct"].to_numpy(float)[idx]
        nonconf = ((b[idx, 1] <= bsh[idx]) if side == "top" else (b[idx, 0] >= bsl[idx])).astype(float)
        AX = {"D1_depth_atr": depth, "D2_touch_n": touch.astype(float), "D3_level_age": age.astype(float),
              "D4_wick": wick, "D5_round_dist": round_d, "D6_atr_pct": atrp, "D7_btc_nonconf": nonconf}
        mid = len(idx) // 2
        for name, v in AX.items():
            ok = np.isfinite(v)
            gv, vv, dv = g[ok], v[ok], day[idx][ok]
            if len(gv) < 500: continue
            # 분위 (동점 많은 이산변수는 분위가 줄 수 있다 — 실제 분위 수를 보고한다)
            try: q = pd.qcut(pd.Series(vv).rank(method="first"), NQ, labels=False).to_numpy()
            except ValueError: continue
            qm = np.array([gv[q == k].mean() - null_mean for k in range(NQ)])
            rho = float(pd.Series(vv).corr(pd.Series(gv), method="spearman"))
            top = gv[q == NQ - 1] - null_mean
            clo, chi = P0.day_cluster_boot(top, dv[q == NQ - 1], B=1000)
            h1 = top[: len(top)//2].mean(); h2 = top[len(top)//2:].mean()   # 시간순 전·후반
            mono = bool(np.all(np.diff(qm) > 0) or np.all(np.diff(qm) < 0))
            crit = bool(mono and (clo > 0) and (np.sign(h1) == np.sign(h2)) and (top.mean() > PEG_RT))
            passes += int(crit)
            rows.append(dict(side=side, axis=name, n=len(gv), rho=round(rho, 4),
                             **{f"q{k+1}": round(float(qm[k]), 2) for k in range(NQ)},
                             top_excess=round(float(top.mean()), 2),
                             top_ci_lo=round(clo, 2), top_ci_hi=round(chi, 2),
                             h1=round(float(h1), 2), h2=round(float(h2), 2),
                             monotone=bool(mono), pass_all=crit))
    D = pd.DataFrame(rows); D.to_csv(OUT / "phase1_mechanism.csv", index=False)

    print(f"\n{'='*132}\nliquidity_sweep 메커니즘 필터 7축 — 분위별 **초과 bp**(같은측면 무작위 진입 대비), H={H}, 1,741일")
    print(f"{'측면':>5}{'축':<16}{'n':>7}{'rho':>8}{'q1':>8}{'q2':>8}{'q3':>8}{'q4':>8}{'q5':>8}"
          f"{'최고분위초과':>12}{'CI95':>20}{'전반':>8}{'후반':>8}{'통과':>6}")
    for r in D.itertuples():
        print(f"{'천장' if r.side=='top' else '바닥':>5}{r.axis:<16}{r.n:>7}{r.rho:>+8.3f}"
              f"{r.q1:>+8.2f}{r.q2:>+8.2f}{r.q3:>+8.2f}{r.q4:>+8.2f}{r.q5:>+8.2f}"
              f"{r.top_excess:>+12.2f}{f'[{r.top_ci_lo:+.2f},{r.top_ci_hi:+.2f}]':>20}"
              f"{r.h1:>+8.2f}{r.h2:>+8.2f}{'⭐' if r.pass_all else '·':>6}")
    print(f"\n판정: 14셀 중 4기준 전부 통과 = **{passes}개**  (기준 ④ 최고분위 초과 > peg 왕복 {PEG_RT}bp)")
    print("="*132)
    print(json.dumps({"done": True, "cells": len(D), "passes": passes}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
