#!/usr/bin/env python3
"""**파산 항을 목적함수에 넣을 수 있나** (2026-09-14).

2026-09-14 절단 연구의 «결론 7»: 배포된 목적함수 `g = L·μ − ½(L·σ)²` 에는 **파산 항이 없다**.
지금은 별도의 생존 상한이 막고 있지, 목적함수가 막는 게 아니다. 넣을 수 있는지 잰다.

## 사전점검이 질문을 바꾼다
손절 3% 에서 **단일거래** 파산은 `s·L >= 1`, 즉 L >= 33.3배에서만 가능하다. 하드캡이 25배라
**가용 구간 전체에서 항상 0인 죽은 항**이 된다. ⇒ 진짜 대상은 **경로 파산**(손절 반복 복리)이다.

## 그래서 세 가지를 잰다
  Q1 L 을 자유롭게 풀면 **배포 근사와 정확한 로그성장의 최적 L 이 갈리는가**.
     갈린다면 생존 상한이 «근사가 못 하는 일»을 하고 있다는 뜻이다.
  Q2 고차항(3·4차 큐뮬런트)을 더하면 그 격차가 닫히는가 -- 닫히면 **닫힌 형태로 배포 가능**하다.
  Q3 파산은 «항»이 아니라 «제약»이어야 하는가 (Busseti·Ryu·Boyd 2016 의 구조).
     경로 MDD·파산율을 재서 어느 쪽이 L 을 실제로 묶는지 본다.

⚠️경로 복리 중앙값은 **파라미터 선택에 쓰지 않는다** -- §5.33 에서 꼬리 경로가 지배해
   k 를 조금만 흔들어도 5배씩 튄다는 게 이미 확인됐다. 여기서는 **진단용**으로만 보고한다.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from scripts.live_eth_trade_plan_20260913 import expected_cost_bp  # noqa: E402
from scripts.research_stop_truncation_vs_gaussian_objective_20260914 import (  # noqa: E402
    HOLDS, STOP_PCT, STRIDE, TAPE, legs, regime_mask, simulate)

LEV_GRID = np.arange(1.0, 40.1, 1.0)
ACCS = (0.55, 0.60)
STUDY_H = (240, 1440)
N_PATH = 2000
RNG = np.random.default_rng(20260914)


def account_moments(move_ok, move_no, cost_ok, cost_no, a, L):
    """계좌 수익률 x = L·(move − cost) 의 적률과 **정확한** 로그성장."""
    x_ok = L * (move_ok - cost_ok / 1e4)
    x_no = L * (move_no - cost_no / 1e4)
    w = np.r_[np.full(len(x_ok), a), np.full(len(x_no), 1 - a)]
    x = np.r_[x_ok, x_no]
    mu = np.average(x, weights=w)
    m2 = np.average((x - mu) ** 2, weights=w)
    m3 = np.average((x - mu) ** 3, weights=w)
    m4 = np.average((x - mu) ** 4, weights=w)
    ruin = np.average((1.0 + x <= 0.0).astype(float), weights=w)
    safe = np.where(1.0 + x > 0, 1.0 + x, np.nan)
    exact = np.average(np.log(safe), weights=w) if ruin == 0 else -np.inf
    return mu, m2, m3, m4, ruin, exact, x, w


def paths(x, w, n_trades, n_path=N_PATH):
    """거래 수열을 리샘플해 1년 복리 결과와 MDD. **진단용**(파라미터 선택에 쓰지 않는다)."""
    p = w / w.sum()
    draw = RNG.choice(len(x), size=(n_path, n_trades), p=p)
    steps = np.log1p(np.clip(x[draw], -0.999999, None))
    dead = (1.0 + x[draw] <= 0).cumsum(axis=1) > 0      # 한 번 파산하면 끝
    steps[dead] = 0.0
    cum = steps.cumsum(axis=1)
    peak = np.maximum.accumulate(cum, axis=1)
    mdd = 1.0 - np.exp((cum - peak).min(axis=1))
    final = np.where(dead[:, -1], 0.0, np.exp(cum[:, -1]))
    return float(np.median(final)), float(np.median(mdd)), float(dead[:, -1].mean())


def main() -> int:
    if TAPE is None:
        print("테이프 없음 -- 건너뜀", file=sys.stderr)
        return 0
    d = pd.read_parquet(TAPE, columns=["px_last", "px_max", "px_min"])
    c, hi, lo = (d[k].to_numpy(float) for k in ("px_last", "px_max", "px_min"))
    print(f"표본 {len(c):,}분 = {len(c)/1440:.0f}일 · 손절 {STOP_PCT:.0%} · 경로 {N_PATH}개")

    rows = []
    for H in STUDY_H:
        idx = np.arange(0, len(c) - H - 2, STRIDE)
        r, tL, tS = simulate(c, hi, lo, idx, H)
        for with_stop in (True, False):
            tl, ts = (tL, tS) if with_stop else (np.zeros_like(tL), np.zeros_like(tS))
            mL, cL, hL, _ = legs(r, tl, +1.0, H, 0.0)
            mS, cS, hS, _ = legs(r, ts, -1.0, H, 0.0)
            long_right = r > 0
            m_ok = np.where(long_right, mL, mS); m_no = np.where(long_right, mS, mL)
            c_ok = np.where(long_right, cL, cS); c_no = np.where(long_right, cS, cL)
            h_ok = np.where(long_right, hL, hS); h_no = np.where(long_right, hS, hL)
            b_bp, sd_bp = np.abs(r).mean() * 1e4, r.std(ddof=1) * 1e4
            cost = expected_cost_bp(H, "LONG", 0.0)
            n_tr = int(min(735, 525_600 / H))
            for a in ACCS:
                et = a * h_ok.mean() + (1 - a) * h_no.mean()
                for L in LEV_GRID:
                    mu, m2, m3, m4, ruin, exact, x, w = account_moments(
                        m_ok, m_no, c_ok, c_no, a, L)
                    quad = L * ((2 * a - 1) * b_bp - cost) / 1e4 - 0.5 * (L * sd_bp / 1e4) ** 2
                    g2 = mu - m2 / 2
                    g3 = g2 + m3 / 3
                    g4 = g3 - m4 / 4
                    rows.append({"H": H, "stop": with_stop, "a": a, "L": L,
                                 "quad": quad, "g2": g2, "g3": g3, "g4": g4,
                                 "exact": exact, "ruin": ruin, "hold_real": et,
                                 "n_trades": n_tr})
        print(f"  H={H} 완료 (진입 {len(idx):,})")
    df = pd.DataFrame(rows)
    out = pathlib.Path(__file__).resolve().parents[1] / "tmp/ruin_term_in_objective_20260914.csv"
    df.to_csv(out, index=False)
    print(f"저장: {out.relative_to(out.parents[1])}")
    return df


EULER_GAMMA = 0.5772156649


def stage_mdd() -> None:
    """**파산 대신 MDD 를 닫힌 형태로 낼 수 있나.** 표류 있는 랜덤워크의 기대 최대낙폭
    (Magdon-Ismail·Atiya·Pratap·Abu-Mostafa 2004): 로그 낙폭 ≈ (v/2g)·(ln(2g²n/v) + γ).
    g·v·n 은 배포 코드가 **이미 가진 값**이라 새 표가 필요 없다.
    적합은 H∈{240,480} 에서 스케일 1개만, 검증은 **H∈{60,1440}** 에서 한다."""
    d = pd.read_parquet(TAPE, columns=["px_last", "px_max", "px_min"])
    c, hi, lo = (d[k].to_numpy(float) for k in ("px_last", "px_max", "px_min"))
    rows = []
    for H in (60, 240, 480, 1440):
        idx = np.arange(0, len(c) - H - 2, STRIDE)
        r, tL, tS = simulate(c, hi, lo, idx, H)
        mL, cL, _hL, _ = legs(r, tL, +1.0, H, 0.0)
        mS, cS, _hS, _ = legs(r, tS, -1.0, H, 0.0)
        lr = r > 0
        m_ok, m_no = np.where(lr, mL, mS), np.where(lr, mS, mL)
        c_ok, c_no = np.where(lr, cL, cS), np.where(lr, cS, cL)
        n = int(min(735, 525_600 / H))
        for a in (0.58, 0.60, 0.62, 0.66):
            for L in (2., 3., 4., 5., 6., 8., 10.):
                *_, x, w = account_moments(m_ok, m_no, c_ok, c_no, a, L)
                lg = np.log1p(np.clip(x, -0.999999, None))
                g = np.average(lg, weights=w)
                v = np.average((lg - g) ** 2, weights=w)
                if g <= 0 or 2 * g * g * n / v <= 1:
                    continue
                global RNG
                RNG = np.random.default_rng(7)
                _f, mdd, _p = paths(x, w, n, n_path=3000)
                rows.append({"H": H, "a": a, "L": L, "g": g, "v": v, "n": n, "mdd": mdd,
                             "pred_log": (v / (2 * g)) * (np.log(2 * g * g * n / v) + EULER_GAMMA)})
    df = pd.DataFrame(rows)
    df["mdd_log"] = -np.log(np.clip(1 - df.mdd, 1e-9, None))
    fit, test = df[df.H.isin([240, 480])], df[df.H.isin([60, 1440])]
    k = float((fit.pred_log * fit.mdd_log).sum() / (fit.pred_log ** 2).sum())
    print(f"\n=== MDD 닫힌 형태 (적합 H=240,480 에서 스케일 1개만: k={k:.3f}) ===")
    for name, sub in (("적합셀", fit), ("검증셀 H=60,1440", test)):
        pr = 1 - np.exp(-k * sub.pred_log)
        print(f"  {name:<18} n={len(sub):>3} 상관 {np.corrcoef(pr, sub.mdd)[0,1]:.3f}"
              f" · 평균절대오차 {np.abs(pr - sub.mdd).mean():.3f}")
    # 🔴k 가 1 근방이어야 «적합»이 아니라 «이론»이다. 멀어지면 근거가 사라진다.
    assert 0.9 < k < 1.2, k
    pr_t = 1 - np.exp(-k * test.pred_log)
    assert np.corrcoef(pr_t, test.mdd)[0, 1] > 0.9, "표본외 상관이 무너지면 못 쓴다"
    assert np.abs(pr_t - test.mdd).mean() < 0.10, "표본외 오차가 10pp 넘으면 못 쓴다"
    print("  ✅ 표본외 검증 통과 — 배포 코드의 g·v·n 만으로 MDD 를 낼 수 있다")


def verdict_ruin(df) -> None:
    """이 연구의 결론을 게이트로 고정한다."""
    stop = df[df.stop]
    # ① 손절이 있으면 가용 구간(L<=25)에서 **단일거래 파산이 0** -- 파산 항은 죽은 코드다.
    assert stop[stop.L <= 25].ruin.max() == 0.0, stop[stop.ruin > 0].L.min()
    # ② 파산이 시작되는 L 은 이론값 1/손절폭 근방이어야 한다.
    first = stop[stop.ruin > 0].L.min()
    assert abs(first - 1.0 / STOP_PCT) < 2.0, first
    # ③ 🔴손절이 **없으면** 근사가 위험하다: 파산이 훨씬 낮은 L 에서 시작한다.
    nostop = df[~df.stop]
    assert nostop[nostop.ruin > 0].L.min() < 10.0, "손절 없을 때 파산이 늦게 오면 전제가 바뀐다"
    # ④ 3차 큐뮬런트는 **발산한다** -- 고차항으로 못 고친다는 근거.
    g3max = df.loc[df.groupby(["H", "stop", "a"]).g3.idxmax(), "L"]
    assert g3max.max() >= 40.0, "3차항이 격자 끝을 안 고르면 «발산» 주장이 약해진다"
    print("\n✅ 게이트 통과 — 손절 하에서 파산 0(L<=25) · 파산 시작 L≈1/s · "
          "무손절이면 위험 · 3차항 발산")


if __name__ == "__main__":
    if "mdd" not in sys.argv:
        verdict_ruin(main())
    stage_mdd()
