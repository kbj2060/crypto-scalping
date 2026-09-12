#!/usr/bin/env python3
"""목적함수를 **정확도에서 순손익으로** 바꾼 방향 모델 (2026-09-12).

## 왜 이게 안 해본 축인가
오늘 방향을 여섯 번 쟀는데(규칙 11,934 · 극점확률 · 지평/배리어 · 문헌 3종 · 트레이더 6종 ·
재료 58피쳐 스택) **전부 정확도나 AUC 로 학습하고 정확도로 골랐다**. 그런데 같은 날 밝힌
구조는 **적중률과 건당 손익이 계열마다 반대로 움직인다**는 것이다:
  되돌림 계열 적중 >0.5 인데 건당 **음수**(자주 이기고 크게 진다)
  추세   계열 적중 <0.5 인데 건당 **양수**(적게 이기고 크게 번다)
⇒ **정확도를 최적화한 것 자체가 틀린 목적함수였다.** 이 저장소가 한 번 이긴 적이 있는데
(극점 탐지기 v2, 정밀도 .582→.689) 그때도 **피쳐가 아니라 목적함수를 고쳤다**.

## 무엇을 바꾸나
분류(부호 맞히기) → **부호 있는 순손익 bp 회귀**. 그러면 손익 비대칭이 라벨 안에 들어온다.
그리고 매 봉 베팅하지 않고 **|예측| 이 비용을 넘을 때만** 건다(기권 = 학습된 선택적 예측).
  라벨   fwd_bp = (종가[i+H+1] / 시가[i+1] − 1) × 1e4     ← 부호 있는 실현 bp
  결정   ŷ > +c 면 롱 · ŷ < −c 면 숏 · 아니면 **기권**
  평가   실제 부호수익 평균 − 비용. c 를 스윕한다.
⚠️피쳐는 **연속 하위값만**(사용자 지시 2026-09-12) — `ev_*`/`trg_*` 이진 발동 플래그 제외.

## 규약
전진 검증 매월 재적합·그 달만 평가(확장 창). 비용 **4bp(사용자)** / **5.52bp(실측 peg)** 병기.
🔴겹침 보정: 신호가 매 봉이라 H 봉짜리 보유가 겹친다. **블록 길이 = 보유기간**으로 부트하고
**독립 블록 수**를 함께 낸다(오늘 Donchian 이 여기서 13,000 → 245 로 무너졌다).
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
import research_eth_rule_direction_probability_20260912 as RD  # noqa: E402
import research_eth_stack_all_models_20260912 as S  # noqa: E402

FEES = {"사용자 4bp": 4.0, "실측 5.52bp": 5.52}
WARM, MIN_TRAIN = 900, 20_000
PANEL = RD.PANEL          # --panel 로 갈아끼운다(기간 확장판 비교용)
WALK_FROM = "2024-07"     # --walk-from


def signed_bp(H: int) -> np.ndarray:
    p = pd.read_parquet(PANEL)
    o = p["open"].to_numpy(float); c = p["close"].to_numpy(float)
    n = len(p)
    ent = np.roll(o, -1); ent[-1] = np.nan
    y = np.full(n, np.nan)
    y[: n - H - 1] = (c[H + 1 : n] / ent[: n - H - 1] - 1.0) * 1e4
    return y


def walk_pnl(X: pd.DataFrame, ts: np.ndarray, y: np.ndarray) -> np.ndarray:
    """전진 예측값. 목적함수는 **부호 있는 bp 회귀**(정확도 아님)."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    months = pd.PeriodIndex(pd.to_datetime(ts), freq="M")
    out = np.full(len(y), np.nan)
    Xv = X.to_numpy(np.float32)
    for m in [u for u in months.unique() if u >= pd.Period(WALK_FROM, "M")]:
        te = (months == m) & np.isfinite(y)
        tr = (months < m) & np.isfinite(y)
        tr[np.arange(len(tr)) < WARM] = False
        if tr.sum() < MIN_TRAIN or te.sum() < 200:
            continue
        mdl = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06, max_depth=6,
                                            random_state=0)
        mdl.fit(Xv[tr], y[tr])
        out[te] = mdl.predict(Xv[te])
    return out


def block_stats(v: np.ndarray, idx: np.ndarray, H: int, fee: float) -> dict:
    """블록 길이 = 보유기간. 독립 블록 수와 블록당 1건 t 를 같이 낸다."""
    blk = idx // H
    ub = np.unique(blk)
    by = {b: v[blk == b] for b in ub}
    rng = np.random.default_rng(20260912)
    bs = np.array([np.concatenate([by[b] for b in rng.choice(ub, len(ub))]).mean() - fee
                   for _ in range(1000)])
    # 🔴블록당 **첫 거래 하나**를 뽑던 초판은 임의 선택이라 대표성이 없다 — 순익이 −0.89bp 인
    # 셀에서 t=2.07 이 나왔다(2026-09-13 발견). **블록 평균**을 쓰면 그런 모순이 사라진다.
    bm = np.array([by[b].mean() for b in ub])
    t = ((bm.mean() - fee) / (bm.std(ddof=1) / np.sqrt(len(bm)))) if len(bm) > 2 else np.nan
    return {"blocks": int(len(ub)), "boot_lo": float(np.percentile(bs, 2.5)),
            "t_indep": float(t), "net": float(v.mean() - fee)}


def period_split(v: np.ndarray, ts_sel: np.ndarray, cut: str = "2024-01-01") -> tuple[float, float]:
    """구간을 갈라 본다. 기간을 늘렸을 때 **늘린 구간에서 부호가 뒤집히면** 그 효과는 시기 한정이다
    (이 저장소는 같은 패턴을 세 번 봤다)."""
    e = pd.DatetimeIndex(ts_sel) < pd.Timestamp(cut)
    a = float(v[e].mean()) if e.sum() > 200 else float("nan")
    b = float(v[~e].mean()) if (~e).sum() > 200 else float("nan")
    return a, b


def main() -> int:
    ap = argparse.ArgumentParser()
    global PANEL, WALK_FROM
    ap.add_argument("--panel", default=None, help="다른 기간으로 만든 패널 디렉토리")
    ap.add_argument("--walk-from", default=WALK_FROM)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.panel:
        PANEL = Path(a.panel) / "panel_5m.parquet"
    WALK_FROM = a.walk_from
    if a.selftest:
        y = signed_bp(12)
        p = pd.read_parquet(RD.PANEL)
        i = 5000
        exp = (p["close"].to_numpy(float)[i + 13] / p["open"].to_numpy(float)[i + 1] - 1) * 1e4
        assert abs(y[i] - exp) < 1e-9, (y[i], exp)
        st = block_stats(np.full(300, 10.0), np.arange(300), 12, 4.0)
        assert st["blocks"] == 25 and abs(st["net"] - 6.0) < 1e-9, st
        # 블록 평균 t 는 «순익이 음수면 t 도 음수»여야 한다(첫거래 방식은 이걸 어겼다)
        rngq = np.random.default_rng(0)
        vv = rngq.normal(1.0, 5.0, 600); st2 = block_stats(vv, np.arange(600), 12, 4.0)
        assert (st2["net"] < 0) == (st2["t_indep"] < 0), st2
        X, _ = S.build()
        ctx = [c for c in X.columns if not c.startswith(("ev_", "trg_"))]
        assert not any(c.startswith(("ev_", "trg_")) for c in ctx)
        assert "p_fast" in ctx and "delta_z" in ctx and "er" in ctx, "연속 하위값이 빠졌다"
        print(f"selftest OK — 부호bp 라벨 · 블록 집계 · 연속 하위값 {len(ctx)}열(이진 제외)")
        return 0

    S.RD.PANEL = PANEL                      # 스택 빌더도 같은 패널을 보게 한다
    X, ts = S.build()
    ctx = [c for c in X.columns if not c.startswith(("ev_", "trg_"))]
    X = X[ctx]
    print(f"피쳐 {len(ctx)}열 — **연속 하위값만**(ev_/trg_ 이진 제외, 2026-09-12 지시)\n")
    for H in (12, 48, 144):
        y = signed_bp(H)
        pred = walk_pnl(X, ts, y)
        ok = np.isfinite(pred) & np.isfinite(y)
        print(f"=== H={H}봉 ({H*5//60}시간) · 목적함수 = 부호 있는 bp 회귀 ===")
        print(f"예측-실현 상관 {np.corrcoef(pred[ok], y[ok])[0,1]:+.4f} · "
              f"예측 SD {pred[ok].std():.2f}bp · 실현 SD {y[ok].std():.1f}bp")
        for fname, fee in FEES.items():
            print(f"  비용 {fname}")
            print(f"{'임계 c':>8}{'거래율':>8}{'n':>8}{'독립블록':>9}{'적중':>8}{'건당bp':>9}"
                  f"{'순익':>8}{'부트2.5%':>10}{'블록t':>8}{'~2023':>9}{'2024~':>9}")
            for c in (0.0, 2.0, 5.0, 10.0, 20.0, 40.0):
                bet = np.where(pred > c, 1.0, np.where(pred < -c, -1.0, 0.0))
                m = ok & (bet != 0)
                if m.sum() < 200:
                    continue
                idx = np.flatnonzero(m)
                v = bet[m] * y[m]
                st = block_stats(v, idx, H, fee)
                e, l = period_split(v, ts[idx])
                print(f"{c:>8.1f}{m.sum()/ok.sum():>8.1%}{m.sum():>8,}{st['blocks']:>9,}"
                      f"{(v > 0).mean():>8.4f}{v.mean():>9.2f}{st['net']:>8.2f}"
                      f"{st['boot_lo']:>10.2f}{st['t_indep']:>8.2f}{e:>9.2f}{l:>9.2f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
