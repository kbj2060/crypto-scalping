#!/usr/bin/env python3
"""진입 필터용 OOF 레짐 예측 생성 -- 누수 차단 (2026-09-03).

메타라벨과 같은 문제·같은 해법이다. clean S12_K3(ETH) / S24_K3(BTC)가 TRAIN <= 2025-08-31로
학습돼 필터 TRAIN을 전부 덮으므로, 필터 TRAIN 행의 레짐이 in-sample이다.

확장창 시계열 OOF:
  워밍업 2024-01 ~ 2024-04 (필터 TRAIN에서 제외)
  fold   2024-05 ~ 2025-08 4등분. fold k의 레짐은 fold k 시작 이전만 학습.
  최종   2025-09 이후는 TRAIN 전체(< 2025-09-01)로 학습.

⚠️라벨 임계값(T1/T2)도 각 단계의 자기 학습창에서만 백분위 보정한다 -- 미래 정보 유입 없음.
모델·피쳐·하이퍼파라미터·시드는 배포본과 동일. 바뀌는 건 학습 종료일뿐이다.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

OUT = ROOT / "tmp/eth_entry_oof_regime_20260903"
FOLDS = [pd.Timestamp(x) for x in
         ("2024-05-01", "2024-09-01", "2025-01-01", "2025-05-01", "2025-09-01")]
TRAIN_END = pd.Timestamp("2025-09-01")


def log(m): print(f"[oof_reg] {m}", flush=True)


def build(kind):
    import joblib
    if kind == "eth":
        from research_eth_regime_s12k3_label_train_20260902 import (
            GBM3_HP, GBM3_MODEL_PATH, SEED, load_frame, s12k3_label)
        from research_eth_regime_scalping_label_geometry_20260902 import TRAIN_START as TS
        src = joblib.load(GBM3_MODEL_PATH); df = load_frame(); lab = s12k3_label
    else:
        from research_btc_regime_s24k3_label_train_20260902 import (
            GBM3_HP, GBM3_MODEL_PATH, SEED, TRAIN_START as TS, load_btc_frame, s24k3_label)
        src = joblib.load(GBM3_MODEL_PATH)
        df = load_btc_frame(src["feature_cols"]); lab = s24k3_label
    cols, med = src["feature_cols"], src["feature_medians"]
    x = df[cols].apply(pd.to_numeric, errors="coerce")
    for c in cols:
        x[c] = x[c].replace([np.inf, -np.inf], np.nan).fillna(med.get(c, 0.0))
    return df["timestamp"], x, lab, df, GBM3_HP, SEED, TS


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for kind in ("eth", "btc"):
        ts, x, lab, df, hp, seed, TS = build(kind)
        out = pd.DataFrame({"timestamp": ts})
        out["regime_oof"] = -1
        out["oof_source"] = ""
        for k in range(len(FOLDS) - 1):
            lo, hi = FOLDS[k], FOLDS[k + 1]
            trm = ((ts >= TS) & (ts < lo)).to_numpy()
            fdm = ((ts >= lo) & (ts < hi)).to_numpy()
            if trm.sum() < 5000 or fdm.sum() == 0:
                log(f"  {kind} fold{k+1} 건너뜀"); continue
            y, t1, t2 = lab(df, trm)          # 임계값도 이 학습창에서만 보정
            m = HistGradientBoostingClassifier(random_state=seed, **hp).fit(x[trm], y[trm])
            out.loc[fdm, "regime_oof"] = m.predict(x[fdm])
            out.loc[fdm, "oof_source"] = f"fold{k+1}(train<{lo.date()})"
            log(f"  {kind} fold{k+1} {lo.date()}~{hi.date()} train {int(trm.sum()):,}봉 "
                f"→ {int(fdm.sum()):,}봉 | chop {float((out.loc[fdm,'regime_oof']==2).mean()):.3f} "
                f"| 라벨일치 {float((out.loc[fdm,'regime_oof'].to_numpy()==y[fdm]).mean()):.3f}")
        trm = ((ts >= TS) & (ts < TRAIN_END)).to_numpy()
        post = (ts >= TRAIN_END).to_numpy()
        y, t1, t2 = lab(df, trm)
        m = HistGradientBoostingClassifier(random_state=seed, **hp).fit(x[trm], y[trm])
        out.loc[post, "regime_oof"] = m.predict(x[post])
        out.loc[post, "oof_source"] = f"final(train<{TRAIN_END.date()})"
        log(f"  {kind} final train {int(trm.sum()):,}봉 → {int(post.sum()):,}봉 "
            f"| chop {float((out.loc[post,'regime_oof']==2).mean()):.3f} "
            f"| 라벨일치 {float((out.loc[post,'regime_oof'].to_numpy()==y[post]).mean()):.3f}")
        cov = float((out.regime_oof >= 0).mean())
        out.to_parquet(OUT / f"regime_oof_{kind}.parquet", index=False)
        log(f"{kind.upper()} 저장: {len(out):,}봉, 값 있는 비율 {cov:.1%} "
            f"(워밍업 2024-01~04는 −1)\n")
    json.dump({"folds": [str(x) for x in FOLDS], "train_end": str(TRAIN_END),
               "scheme": "expanding-window time-series OOF; thresholds recalibrated per fold"},
              open(OUT / "oof_regime_config.json", "w"), indent=2)
    log(f"산출: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
