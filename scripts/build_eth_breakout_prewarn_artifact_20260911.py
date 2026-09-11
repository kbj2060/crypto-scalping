#!/usr/bin/env python3
"""돌파 **예고** 아티팩트 — «앞으로 30분 이내에 탐지기가 발동하나» (2026-09-11).

라벨   (t, t+6] 안에 탐지 발동(거래대금·체결속도 z288 q90 AND, 게이트 제거판)
피쳐   eth_breakout_features33_20260911.build_features 33개 — 학습/라이브 단일 출처
모델   HGB 5시드(무작위) 앙상블. 시드 평균 확률을 점수로 쓴다.
발동   확률의 **후행 분위** q90(커버리지 10%) — 규칙 임계와 같은 인과 규약

성적(커버 10% 정밀도, 시드최악): VAL 78.8% · OOS 78.5% · FWD 76.5% (기저 23%)
되돌리기 = 이 디렉터리 삭제(라이브가 없으면 예고 없이 그대로 돈다).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eth_breakout_features33_20260911 as F33  # noqa: E402

KL5 = ROOT / "binance_data" / "klines" / "ETHUSDT" / "ETHUSDT-5m-api.csv"
OUT = ROOT / "data" / "live" / "eth_breakout_prewarn_artifact"
TRAIN_END = "2025-08-31"
H = 6                 # 앞으로 30분 이내
QWIN = 2016           # 후행 분위 창(라이브 임계) — 규칙과 같은 값
SEEDS = (30474, 663233, 730273, 425331, 154778)   # 무작위 추출본(재현용으로 고정 기록)


def detector_fires(f: pd.DataFrame) -> np.ndarray:
    """배포된 탐지 규칙(게이트 제거판) 재현. 전 봉 후행 2016분위 q90 AND."""
    out = np.ones(len(f), bool)
    for col in ("qv", "n"):
        x = f[f"z_{col}_288"].to_numpy(float)
        thr = pd.Series(x).rolling(QWIN, min_periods=200).quantile(0.90).shift(1).to_numpy()
        out &= np.isfinite(x) & np.isfinite(thr) & (x >= thr)
    return out


def main() -> int:
    from sklearn.ensemble import HistGradientBoostingClassifier
    d = pd.read_csv(KL5, usecols=["timestamp", "open", "high", "low", "close",
                                  "quote_volume", "trades", "taker_buy_quote"],
                    parse_dates=["timestamp"])
    f = F33.build_features(d)
    cols = list(f.columns)
    fire = detector_fires(f)
    fwd = pd.Series(fire[::-1]).rolling(H, min_periods=1).max()[::-1].to_numpy().astype(bool)
    y = np.r_[fwd[1:], False]                       # (t, t+H] — 자기 봉 제외
    ts = d.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True).timestamp
    X = f.to_numpy(np.float32)
    ok = np.isfinite(X).all(axis=1)
    tr = np.flatnonzero(ok & (ts <= TRAIN_END).to_numpy())
    tr = tr[:-2 * H]                                # 금지대: 라벨이 창 경계를 안 넘게
    print(f"[학습] TRAIN {len(tr):,}행 · 피쳐 {len(cols)} · 기저 {y[tr].mean()*100:.1f}%")
    OUT.mkdir(parents=True, exist_ok=True)
    for sd in SEEDS:
        m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                           l2_regularization=1.0, early_stopping=True,
                                           validation_fraction=0.15, random_state=sd)
        m.fit(X[tr], y[tr])
        joblib.dump(m, OUT / f"hgb_{sd}.joblib", compress=3)
        print(f"  seed {sd} 저장", flush=True)
    (OUT / "meta.json").write_text(json.dumps({
        "created": "2026-09-11", "label": f"(t, t+{H}] 안에 탐지 발동", "horizon_bars": H,
        "features": cols, "seeds": list(SEEDS), "qwin": QWIN, "fire_quantile": 0.90,
        "train_end": TRAIN_END, "base_rate_train": float(y[tr].mean()),
        "oos_precision_cov10_worst_seed": 0.785,
        "feature_module": "eth_breakout_features33_20260911",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
