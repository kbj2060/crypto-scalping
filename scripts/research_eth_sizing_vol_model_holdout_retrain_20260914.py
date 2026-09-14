"""**변동성 모델 홀드아웃 재학습** — 2022~23 을 진짜 표본외로 만든다 (2026-09-14).

배포 아티팩트(`live_eth_sizing_vol_model_20260912`)는 `train_end=2025-08-31` 이고 학습셋이
**2021-12 부터 전부**라, 그걸 게이트로 쓰면 2022~23 과 TRAIN 이 **학습 안**이다. 그 창들이 바로
스트래들 판정의 핵심이었으므로 「게이트가 옛 창을 건졌다」가 아티팩트일 수 있다.

🔴ETH 5분봉이 2021-12-01 부터라 「2022 이전으로 train_end 를 당기기」는 불가능하다(1개월치).
대신 **학습 구간에서 2022~23 을 도려낸다**: 2024-01-01 ~ 2025-08-31 만 학습 ⇒ 2022~23 도,
VAL/OOS/TEST 도 전부 모델이 못 본 구간이 된다. (2022~23 은 시간상 **앞**이라 인과적 예측이
아니라 「그 시기를 안 보고도 그 시기의 변동성을 맞히나」를 재는 것 -- 게이트 아티팩트 판정에는 이걸로 충분하다.)

배포 아티팩트는 **건드리지 않는다**. 산출은 `data/research/` 로만 간다.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import live_eth_sizing_vol_model_20260912 as svm  # noqa: E402

OUT = ROOT / "data/research/eth_direction_barrier_label_20260914"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-start", default="2024-01-01")
    ap.add_argument("--train-end", default="2025-08-31")
    ap.add_argument("--symbol", default="ETHUSDT")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    import joblib
    from sklearn.ensemble import HistGradientBoostingRegressor

    kl = ROOT / f"binance_data/klines/{a.symbol}/{a.symbol}-5m-api.csv"
    out_name = a.out or f"vol_model_holdout2223_{a.symbol}.joblib"
    print(f"자산 {a.symbol} · {kl.name}")
    d = pd.read_csv(kl, usecols=["timestamp", "close", "high", "low", "quote_volume", "trades"],
                    parse_dates=["timestamp"]).dropna(subset=["timestamp"])
    d = d.sort_values("timestamp").reset_index(drop=True)
    X = svm.build_features(d["timestamp"], d["close"].to_numpy(float),
                           d["quote_volume"].to_numpy(float), d["trades"].to_numpy(float),
                           d["high"].to_numpy(float), d["low"].to_numpy(float))
    lr = np.diff(np.log(np.maximum(d["close"].to_numpy(float), 1e-12)), prepend=0.0)
    y = pd.Series(lr).rolling(svm.HOLD, min_periods=svm.HOLD).std().shift(-svm.HOLD).to_numpy()
    ok = np.isfinite(y) & (y > 0) & np.isfinite(X.to_numpy(float)).all(1)
    ts = d["timestamp"].to_numpy()
    is_tr = ok & (ts >= np.datetime64(a.train_start)) & (ts <= np.datetime64(a.train_end + "T23:59:59"))
    A = np.nan_to_num(X.to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0)
    print(f"학습 {a.train_start}~{a.train_end} · {is_tr.sum():,}행 (배포판 385,166행 대비 "
          f"{is_tr.sum()/385166:.0%}) · 시드 {len(svm.SEEDS)}", flush=True)
    assert is_tr.sum() > 50_000, "학습행이 너무 적다"

    models = []
    for s in svm.SEEDS:
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_depth=6,
                                          random_state=int(s))
        m.fit(A[is_tr], np.log(y[is_tr])); models.append(m)
    ref = float(np.median(svm.predict_vol(models, X[is_tr])))
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / out_name
    joblib.dump({"models": models, "features": svm.FEATURES, "ref_pred": ref, "seeds": list(svm.SEEDS),
                 "train_start": a.train_start, "train_end": a.train_end, "hold_bars": svm.HOLD,
                 "symbol": a.symbol,
                 "n_train": int(is_tr.sum())}, p)

    # 표본외 예측력 점검 -- 도려낸 2022~23 에서도 변동성을 맞히나(맞혀야 게이트로 쓸 자격이 있다)
    pred = svm.predict_vol(models, X)
    print(f"\n{'창':>12} {'n':>9} {'예측-실현 스피어만':>18}")
    for name, s0, s1 in (("2022~23", "2022-01-01", "2023-12-31"), ("학습", a.train_start, a.train_end),
                         ("VAL", "2025-09-01", "2025-12-31"), ("OOS", "2026-01-01", "2026-03-31"),
                         ("TEST", "2026-04-01", "2026-08-20")):
        m = ok & (ts >= np.datetime64(s0)) & (ts <= np.datetime64(s1 + "T23:59:59"))
        if m.sum() < 100:
            continue
        print(f"{name:>12} {m.sum():>9,} {spearmanr(pred[m], y[m]).statistic:>18.4f}")
    print(f"\n저장: {p} ({p.stat().st_size/1e6:.1f}MB) · 기준예측 {ref:.6g}")
    print("⚠️배포 아티팩트는 건드리지 않았다:", svm.ARTIFACT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
