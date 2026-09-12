#!/usr/bin/env python3
"""사이징 분모 — **배포 10피쳐 vs 재료 스택 41피쳐** (2026-09-12).

같은 날 다른 세션이 사이징 분모를 ATR 공식에서 **전방 변동성 예측 모델**로 바꿔 배포했다
(`live_eth_sizing_vol_model_20260912.py`, 커밋 690ebec: 예측상관 .630→.789 · 손익 SD −12.3% ·
50배 청산 도달률 7.72→6.11%). 이 스크립트는 **그 모델의 피쳐 10개**와
[재료 스택 41열](`research_eth_stack_all_models_20260912`)을 **같은 조건**에서 붙인다.

## 공정성 — 하나만 다르게 한다
타깃·분할·모델·지표를 **배포본에 맞춘다**. 피쳐 집합만 바꾼다.
  타깃  `rolling(48).std(log수익).shift(-48)` = **전방 48봉 실현변동성**(배포본 정의 그대로)
  분할  학습 ≤2025-08-31 · 평가 2025-09-01~ (배포본 TRAIN_END 그대로)
  모델  HistGradientBoostingRegressor(300, 0.06, depth 6) · **로그 타깃** · 8시드 기하평균
  지표  스피어만 rho(순위) · 로그공간 피어슨 · **손익 SD 대리**(1/예측 수량의 수익 산포)
⚠️내 스택의 타깃은 |다음 H봉 수익| 이었다 — 여기서는 **배포본 타깃으로 바꿔** 잰다.
그래야 «피쳐가 더 낫다» 와 «타깃이 달랐다» 가 안 섞인다.

## 비교 집합
  A 1/ATR (현행 이전 공식) — 모델 없음
  B 배포 10피쳐
  C 재료 스택 41열(문맥만: 증거신호·트리거 이진 제외 — 2026-09-12 지시)
  D B ∪ C
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
import live_eth_sizing_vol_model_20260912 as V  # noqa: E402
import research_eth_rule_direction_probability_20260912 as RD  # noqa: E402
import research_eth_stack_all_models_20260912 as S  # noqa: E402

HOLD, TRAIN_END = V.HOLD, V.TRAIN_END


def assemble() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """패널 격자에 배포 피쳐를 붙이고 배포본 타깃을 만든다."""
    kl = pd.read_csv(V.KL_CSV, usecols=["timestamp", "close", "quote_volume", "trades"],
                     parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    Xd = V.build_features(kl["timestamp"], kl["close"].to_numpy(float),
                          kl["quote_volume"].to_numpy(float), kl["trades"].to_numpy(float))
    lr = np.diff(np.log(np.maximum(kl["close"].to_numpy(float), 1e-12)), prepend=0.0)
    y = pd.Series(lr).rolling(HOLD, min_periods=HOLD).std().shift(-HOLD).to_numpy()
    Xd["timestamp"] = kl["timestamp"]
    Xd["_y"] = y

    p = pd.read_parquet(RD.PANEL)
    Xs, _ = S.build()
    ctx = [c for c in Xs.columns if not c.startswith(("ev_", "trg_"))]
    Xs = Xs[ctx].copy()
    Xs["timestamp"] = p["timestamp"]
    M = Xs.merge(Xd, on="timestamp", how="inner", suffixes=("", "_dep"))
    ts = M["timestamp"].to_numpy()
    yy = M["_y"].to_numpy(float)
    atr = M["atr_pct"].to_numpy(float)          # 패널의 원시 ATR = 현행 공식 분모
    F = M.drop(columns=["timestamp", "_y"])
    return F, yy, atr, ts


def fit_predict(F: pd.DataFrame, cols: list[str], y: np.ndarray,
                tr: np.ndarray, te: np.ndarray) -> np.ndarray:
    from sklearn.ensemble import HistGradientBoostingRegressor
    A = np.nan_to_num(F[cols].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
    preds = []
    for s in V.SEEDS:
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_depth=6,
                                          random_state=int(s))
        m.fit(A[tr], np.log(y[tr]))
        preds.append(m.predict(A[te]))
    return np.exp(np.mean(preds, axis=0))       # 배포본과 같은 로그공간(기하) 평균


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        F, y, atr, ts = assemble()
        assert all(c in F.columns for c in V.FEATURES), "배포 10피쳐가 안 붙었다"
        assert "atr_pct" in F.columns and F.shape[1] > 45, F.shape
        i = 3000
        assert np.isfinite(y[i]) and y[i] > 0
        print(f"selftest OK — 합친 프레임 {F.shape[0]:,}행 × {F.shape[1]}열 · 타깃 유한")
        return 0

    from scipy.stats import spearmanr
    F, y, atr, ts = assemble()
    ctx = [c for c in F.columns if c not in V.FEATURES and c != "atr_pct"] + ["atr_pct"]
    ok = np.isfinite(y) & (y > 0)
    tr = ok & (ts <= np.datetime64(TRAIN_END + "T23:59:59"))
    te = ok & (ts > np.datetime64(TRAIN_END + "T23:59:59"))
    print(f"학습 {tr.sum():,} (~{TRAIN_END}) · 평가 {te.sum():,} "
          f"({str(ts[te][0])[:10]} ~ {str(ts[te][-1])[:10]})\n")

    sets = {"A 1/ATR (모델 없음)": None,
            "B 배포 10피쳐": list(V.FEATURES),
            "C 재료 스택 41열": ctx,
            "D B ∪ C": sorted(set(V.FEATURES) | set(ctx))}
    print(f"{'피쳐집합':<22}{'열':>4}{'스피어만':>10}{'로그피어슨':>11}{'수익 SD':>10}{'SD 감소':>9}"
          f"{'|z|>3 비율':>11}")
    base_sd = None
    fwd = None
    for nm, cols in sets.items():
        if cols is None:
            pred = atr[te]
        else:
            pred = fit_predict(F, cols, y, tr, te)
        rho = float(spearmanr(pred, y[te]).statistic)
        pear = float(np.corrcoef(np.log(np.maximum(pred, 1e-12)), np.log(y[te]))[0, 1])
        # 손익 SD 대리: 수량 ∝ 1/예측 → 건당 수익 ∝ 실현변동성/예측
        z = y[te] / np.maximum(pred, 1e-12)
        z = z / np.median(z)
        sd = float(np.std(z))
        if base_sd is None:
            base_sd = sd
        print(f"{nm:<22}{(0 if cols is None else len(cols)):>4}{rho:>10.4f}{pear:>11.4f}"
              f"{sd:>10.4f}{(sd/base_sd-1)*100:>8.1f}%{float((z > 3).mean())*100:>10.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
