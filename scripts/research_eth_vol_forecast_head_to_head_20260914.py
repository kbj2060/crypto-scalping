"""**변동성 예측 2종 정면 비교** — 5분봉 트레이더에게 어느 쪽이 유리한가 (2026-09-14).

사용자: *"위 변동성 예측 2가지 중 어떤 것이 5분봉을 보고 트레이딩하는 사람한테 유리한가?"*

  A. **사이징 모델** `live_eth_sizing_vol_model_20260912` — 앞으로 **48봉(4시간)** 실현변동성 **회귀**,
     5분 해상도, 입력은 klines 22열. 배포: 수량 = 기준수량 × 기준예측/현재예측.
  B. **대시보드 리본** `live_eth_vol_forecast_20260910` — 앞으로 **24시간** 변동성이 현재의 1.3배로
     **확장되나** 3등급 분류, **1시간** 해상도, 입력 HAR-RV + Deribit DVOL. 배포: 청산맵 아래 주황 리본.

두 모델은 지평·출력·해상도가 다르므로 **한쪽 잣대로만 재면 불공정**하다. 그래서 2×2 로 잰다:
  · 잣대① 앞으로 H봉 실현변동성 **수준** (스피어만)   · 잣대② 그게 지금의 1.3배 이상인가 (**AUC**)
  · H ∈ {12봉=1시간, 48봉=4시간, 288봉=24시간} — 5분봉 트레이더의 실제 보유(중앙 67분)를 포함한다

🔴창 주의: 리본 아티팩트는 TRAIN ≤2026-03-31 이라 **OOS(2026-01~03)는 그 모델의 학습 안**이다.
   DVOL 시간봉 CSV 가 2026-08-04 까지라 표본외 비교는 **2026-04-01~08-04** 한 구간뿐이다.
피쳐는 두 모듈의 **자기 빌더**를 그대로 부른다(하네스가 식을 다시 쓰면 하네스만 낡는다).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import live_eth_sizing_vol_model_20260912 as svm  # noqa: E402
import live_eth_vol_forecast_20260910 as vfc  # noqa: E402

OUT = ROOT / "data/research/eth_direction_barrier_label_20260914"
HS = (12, 48, 288)
WINS = {"OOS(리본 학습 안)": ("2026-01-01", "2026-03-31"),
        "표본외(2026-04~08-04)": ("2026-04-01", "2026-08-04")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=12)      # 1시간 간격 — 리본 해상도에 맞춘다
    a = ap.parse_args()
    kl = pd.read_csv(svm.KL_CSV, usecols=["timestamp", "close", "high", "low", "quote_volume", "trades"],
                     parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    c = kl.close.to_numpy(float); ts = kl.timestamp.to_numpy()

    # A. 사이징 모델 (5분 해상도)
    art = svm.load_model(); assert art is not None
    X = svm.build_features(kl.timestamp, c, kl.quote_volume.to_numpy(float),
                           kl.trades.to_numpy(float), kl.high.to_numpy(float), kl.low.to_numpy(float))
    predA = svm.predict_vol(art["models"], X)

    # B. 리본 (1시간 -> 5분으로 앞으로 채움: 그 시간 동안 화면이 보여주는 값 그대로)
    import joblib
    mB = joblib.load(ROOT / "data/live/eth_vol_forecast_artifact/model.joblib")
    dv = pd.read_csv(ROOT / "data/derivatives/deribit_dvol/ETH_dvol_hourly.csv",
                     parse_dates=["timestamp"]).rename(columns={"close": "dvol"})
    assert "dvol" in dv.columns and dv.timestamp.notna().all(), "DVOL CSV 형식이 다르다"
    hf = vfc.build_features(kl, dv)
    Xb = (hf[vfc.FEATS].to_numpy(float) - mB["mu"]) / mB["sd"]      # 라이브와 같은 표준화
    pB_h = pd.Series(mB["clf"].predict_proba(Xb)[:, 1], index=hf.index)
    rB_h = pd.Series(mB["reg"].predict(Xb), index=hf.index)          # 같은 아티팩트의 회귀 머리
    predB = pB_h.reindex(pd.DatetimeIndex(ts), method="ffill").to_numpy()
    # 리본 아티팩트에는 회귀 머리(`reg`)도 들어 있다 — 확률만으로 「수준」 잣대를 재면
    # 리본에 불리하므로 같이 싣는다.
    lvlB = rB_h.reindex(pd.DatetimeIndex(ts), method="ffill").to_numpy()

    lr = np.diff(np.log(np.maximum(c, 1e-12)), prepend=0.0)
    fwd = {h: pd.Series(lr).rolling(h, min_periods=h).std().shift(-h).to_numpy() for h in HS}
    cur = {h: pd.Series(lr).rolling(h, min_periods=h).std().to_numpy() for h in HS}

    rows = []
    print(f"{'창':>22} {'H':>5} {'잣대':>10} {'A 사이징':>9} {'B 리본':>9} {'B 회귀머리':>12} "
          f"{'기준 rv48':>9} {'n':>7}")
    for wname, (s0, s1) in WINS.items():
        m0 = (ts >= np.datetime64(s0)) & (ts <= np.datetime64(s1 + "T23:59:59"))
        for h in HS:
            ok = m0 & np.isfinite(predA) & np.isfinite(predB) & np.isfinite(fwd[h]) & \
                 np.isfinite(cur[h]) & np.isfinite(lvlB) & (fwd[h] > 0) & (cur[h] > 0)
            idx = np.where(ok)[0][::a.stride]
            if len(idx) < 200:
                continue
            y_lvl = fwd[h][idx]
            y_exp = (fwd[h][idx] >= 1.3 * cur[h][idx]).astype(int)
            base = pd.Series(lr).rolling(48, min_periods=48).std().to_numpy()[idx]
            sc = {"A 사이징": predA[idx], "B 리본": predB[idx], "B 회귀머리": lvlB[idx],
                  "기준 rv48": base}
            r1 = {k: float(spearmanr(v, y_lvl).statistic) for k, v in sc.items()}
            r2 = {k: (float(roc_auc_score(y_exp, v)) if 0 < y_exp.mean() < 1 else np.nan)
                  for k, v in sc.items()}
            for label, r in (("수준 ρ", r1), ("확장 AUC", r2)):
                print(f"{wname:>22} {h:>5} {label:>10} " +
                      " ".join(f"{r[k]:>9.4f}" if k != 'B 회귀머리' else f"{r[k]:>12.4f}"
                               for k in ("A 사이징", "B 리본", "B 회귀머리", "기준 rv48")) +
                      f" {len(idx):>7,}")
                rows.append({"window": wname, "h": h, "metric": label, **r, "n": len(idx)})
        print()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "vol_head_to_head.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    print(f"저장: {OUT/'vol_head_to_head.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
