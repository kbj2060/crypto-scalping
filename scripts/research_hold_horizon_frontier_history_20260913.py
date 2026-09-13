"""**보유시간 프런티어 — 과거 데이터만으로** (2026-09-13, 사용자: *"원장은 정답이 아니야"*).

원장 68건의 보유시간별 엣지(μ_H) 대신, 869일 5분봉에서 **지평별 움직임 크기·분산·역행폭**을
재고 «정확도 a 인 사람이 H 분 들면 성장이 얼마인가»를 프런티어로 낸다. a 는 원장 값이 아니라
**파라미터**다 -- 표는 a 마다 최적 H 를 주고, 손익분기 정확도 a*(H) 를 같이 준다.

## 스케일 없는 계수
r_H ≈ atr_pct · k · z 라 가정하고 k 를 잰다(atr_pct = 워커와 같은 정의, 288봉 평균 |Δc|/c).
  k_b[H] = E[|r_H| / atr_pct]      (움직임 크기)
  k_s[H] = SD[ r_H / atr_pct ]     (분산)
  k_m[H] = q_{1−0.001}[ MAE_H / atr_pct ]  (역행폭; 라이브는 학습 모델이 대신한다)
계수가 변동성 5분위·연도별로 안정하면 «지금 atr_pct × k» 로 현재 상황의 표가 된다.

## 프런티어
  b_H = k_b·atr_pct   σ_H = k_s·atr_pct   L_H = min(상한, 1/(k_m·atr_pct))
  g(H, a) = L_H·((2a−1)·b_H − 비용) − ½(L_H·σ_H)²      a*(H) = ½ + 비용/(2·b_H)
⚠️(2a−1)·E|r| 은 «정확도가 움직임 크기와 독립»일 때만 맞다. 09-12 15분 되돌림 재현에서 이긴 봉이
진 봉보다 작아(36.7 vs 47.9bp) 이 식이 8배 과대였다. 그래서 a 는 «크기와 무관한 방향 정확도»로
읽어야 하고, 표는 절대값이 아니라 **H 사이의 순위**용이다.
"""
from __future__ import annotations

import argparse
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
KLINES = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
HOLDS = {60: 12, 120: 24, 240: 48, 480: 96, 1440: 288}
ATR_BARS = 288
COST_BP = 5.88
CAP_X = 8.0
TARGET_EXCEED = 0.001
TRAIN_END = "2025-06-30"
ACC_GRID = (0.52, 0.55, 0.58, 0.62, 0.66)


def load(path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["timestamp", "high", "low", "close"])
    df["ts"] = pd.to_datetime(df.timestamp)
    c = df.close.to_numpy(float)
    df["atr_pct"] = pd.Series(np.abs(np.diff(c, prepend=c[0]))).rolling(ATR_BARS, min_periods=200).mean().to_numpy() / c
    return df.dropna().reset_index(drop=True)


def horizon_frame(df: pd.DataFrame, h: int) -> pd.DataFrame:
    c, hi, lo, ap = (df.close.to_numpy(float), df.high.to_numpy(float),
                     df.low.to_numpy(float), df.atr_pct.to_numpy(float))
    n = len(c) - h
    r = c[h:] / c[:n] - 1
    fmin = pd.Series(lo).rolling(h).min().shift(-h).to_numpy()[:n]
    fmax = pd.Series(hi).rolling(h).max().shift(-h).to_numpy()[:n]
    mae = np.maximum((c[:n] - fmin) / c[:n], (fmax - c[:n]) / c[:n])   # 양측 중 큰 쪽(보수)
    return pd.DataFrame({"ts": df.ts.to_numpy()[:n], "ap": ap[:n], "z": r / ap[:n],
                         "az": np.abs(r) / ap[:n], "mz": mae / ap[:n]}).dropna()


def coefficients(f: pd.DataFrame) -> tuple[float, float, float]:
    return float(f.az.mean()), float(f.z.std()), float(np.quantile(f.mz, 1 - TARGET_EXCEED))


def growth(L: float, b_bp: float, s_bp: float, a: float) -> float:
    return L * ((2 * a - 1) * b_bp - COST_BP) / 1e4 - 0.5 * (L * s_bp / 1e4) ** 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--klines", default=str(KLINES))
    ap.add_argument("--atr-pct", type=float, default=None, help="지금 atr_pct. 없으면 전체 중앙값")
    a = ap.parse_args()
    df = load(a.klines)
    ap_now = a.atr_pct or float(df.atr_pct.median())
    print(f"5분봉 {len(df):,} ({df.ts.min().date()}~{df.ts.max().date()}) · atr_pct 지금 {1e4*ap_now:.2f}bp/봉"
          f" (전체 중앙 {1e4*df.atr_pct.median():.2f})\n")
    print(f"{'H':>6} {'k_b':>6} {'k_s':>6} {'k_m999':>7} {'분위안정(k_b min~max)':>22} {'TRAIN→TEST k_b':>15}")
    K = {}
    for H, h in HOLDS.items():
        f = horizon_frame(df, h)
        kb, ks, km = coefficients(f)
        q = pd.qcut(f.ap, 5, labels=False)
        by_q = [float(f.az[q == i].mean()) for i in range(5)]
        tr, te = f[f.ts <= TRAIN_END], f[f.ts > TRAIN_END]
        kb_tr, kb_te = float(tr.az.mean()), float(te.az.mean())
        K[H] = (kb, ks, km)
        print(f"{H:>5}분 {kb:>6.2f} {ks:>6.2f} {km:>7.1f} {min(by_q):>10.2f} ~ {max(by_q):<9.2f} "
              f"{kb_tr:>6.2f} → {kb_te:<6.2f}")
        assert 0.75 < kb_te / kb_tr < 1.33, f"H={H} 계수가 표본외에서 25% 넘게 움직였다 -- 상수로 못 쓴다"
    kbs = [K[H][0] for H in HOLDS]
    assert kbs == sorted(kbs), "움직임 크기는 지평에 단조여야 한다"
    assert 3.0 < kbs[-1] / kbs[0] < 7.0, f"√24≈4.9 스케일에서 크게 벗어났다: {kbs[-1]/kbs[0]:.2f}"

    print(f"\n지금 atr_pct 에서의 프런티어 (비용 {COST_BP}bp · 상한 {CAP_X}배)")
    print(f"{'H':>6} {'|r|bp':>6} {'σbp':>7} {'MAE999%':>8} {'L':>5} {'a*':>6} "
          + " ".join(f"g@{int(100*x)}%" for x in ACC_GRID))
    best = {x: (None, -1e9) for x in ACC_GRID}
    for H, (kb, ks, km) in K.items():
        b, s, m = kb * ap_now * 1e4, ks * ap_now * 1e4, km * ap_now * 100
        L = min(CAP_X, 100.0 / m)
        astar = 0.5 + COST_BP / (2 * b)
        gs = [growth(L, b, s, x) for x in ACC_GRID]
        for x, g in zip(ACC_GRID, gs):
            if g > best[x][1]:
                best[x] = (H, g)
        print(f"{H:>5}분 {b:>6.1f} {s:>7.1f} {m:>7.2f}% {L:>5.2f} {100*astar:>5.1f}% "
              + " ".join(f"{g:>+7.4f}" for g in gs))
    print("\n정확도별 최적 보유: " + " · ".join(f"{int(100*x)}%→{best[x][0]}분" for x in ACC_GRID))
    print("\n# live_eth_trade_plan_20260913.K_HORIZON 에 붙여넣는 값 (k_b, k_s)")
    print("K_HORIZON = {" + ", ".join(f"{H}: ({kb:.3f}, {ks:.3f})" for H, (kb, ks, _) in K.items()) + "}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
