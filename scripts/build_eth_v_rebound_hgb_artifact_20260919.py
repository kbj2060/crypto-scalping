#!/usr/bin/env python3
"""V자 급등락 **HGB 아티팩트** 빌더 — 라이브 TabPFN 을 대체한다 (2026-09-19, 사용자 지시).

근거 실측: `scripts/research_eth_v_rebound_hgb_vs_tabpfn_20260919.py` (같은 45,000행 맞대결)
    TabPFN 18k ctx   VAL .7025 · OOS .7102 · FWD .7029 · 라이브 1사이클 **6.62s**
    HGB 184k TRAIN   VAL .7027 · OOS .7125 · FWD .7036 · 라이브 1사이클 **12.7ms**
    HGB 같은 18k ctx VAL .6791 · OOS .6987 · FWD .6807      <- 컨텍스트를 맞추면 HGB 가 **진다**
⭐우위의 정체는 모델이 아니라 TRAIN 크기다. TabPFN 은 컨텍스트 상한(=6.6초의 원인) 때문에
  18,000행밖에 못 쓰고, HGB 는 1회 학습이라 184,207행을 다 쓴다.
🔴+0.0002~+0.0023 은 **앙상블 분산감소**다(시드폭 VAL .6976~.7018). 「AUC 가 이겨서」가 아니라
  「동률인데 575배 싸서」 바꾸는 것이다.

## 아티팩트 규약 — 극점 탐지기와 **같은 모양**
`data/live/eth_v_rebound_hgb_artifact/{model.joblib, meta.json}`.
`meta["features"]` 가 라이브의 FEATURES 와 다르면 라이브는 아티팩트를 **먹지 않는다**
(극점 탐지기 `load_art()` 와 같은 가드 -- 옛 형상을 조용히 삼키지 않는다).

## 임계값 — 확률값이 아니라 **발동률**로 옮긴다
`PROBA_THRESHOLD=0.60` 은 TabPFN 확률의 운영점이다. 모델이 바뀌면 같은 0.60 이 전혀 다른
발동률이 된다([[feedback_alert_threshold_must_be_declared_as_percentile_20260919]]).
배포본이 실측으로 남긴 **하루 13.25 발동봉**(라이브 스크립트 docstring, 0.60 기준)을 그대로
재현하는 HGB 확률을 찾아 `meta["proba_threshold"]` 에 적는다. 분위도 같이 적는다.
🔴이건 **분류 운영점을 보존**하는 것이지 경제성 게이트를 다시 통과했다는 뜻이 아니다 --
  0.60 이 통과했던 그 시험(data/research/eth_v_rebound_every_bar_tabpfn_costgate_20260901)은
  HGB 확률로 다시 돌려야 한다. 그 전까지 이 교체는 «같은 빈도·같은 순위»까지만 보증한다.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
for _p in (CODE, CODE / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import research_eth_v_rebound_hgb_vs_tabpfn_20260919 as X                    # noqa: E402
import research_eth_v_rebound_close_anchor_retrain_20260912 as R12           # noqa: E402
import live_eth_sweep_v_rebound_signal_20260829 as LIVE                      # noqa: E402

ART = X.ROOT / "data/live/eth_v_rebound_hgb_artifact"
DEPLOYED_FIRES_PER_DAY = 13.25   # 라이브 docstring 의 0.60 실측(발동봉/일). 이걸 재현한다.
BARS_PER_DAY = 288


def main() -> int:
    D, win, agree = X.assemble("matched")
    X_all = D[LIVE.FEATURES].to_numpy(float)
    y = D.y.to_numpy(int)
    tr = win["TRAIN"]

    from sklearn.ensemble import HistGradientBoostingClassifier
    import joblib
    models = []
    for sd in R12.SEEDS:
        m = HistGradientBoostingClassifier(          # 측정 스크립트와 **같은** 하이퍼파라미터
            max_iter=300, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.15, random_state=sd)
        m.fit(X_all[tr], y[tr])
        models.append(m)
    P = np.mean([m.predict_proba(X_all)[:, 1] for m in models], axis=0)

    # ── 임계값: 배포본의 «발동봉/일» 을 재현하는 확률 ──────────────────────────
    # 라이브는 봉마다 양측면을 채점하고 **높은 쪽**을 그 봉의 그림으로 쓴다(_every_bar_rows +
    # best_by_pos). 그러니 운영점도 «봉당 최대확률» 위에서 잡아야 한다.
    hold = win["VAL"] | win["OOS"] | win["FWD"]
    per_bar = (pd.DataFrame({"t": D.timestamp[hold], "p": P[hold]})
               .groupby("t")["p"].max().to_numpy())
    target = DEPLOYED_FIRES_PER_DAY / BARS_PER_DAY
    thr = float(np.quantile(per_bar, 1.0 - target))
    fires = float((per_bar >= thr).mean() * BARS_PER_DAY)
    print(f"[임계값] 목표 {DEPLOYED_FIRES_PER_DAY}발동봉/일(={target:.4%}) -> "
          f"proba {thr:.4f} (분위 {1 - target:.4f}) · 재현 {fires:.2f}발동봉/일", flush=True)

    scores = {}
    for nm in ("VAL", "OOS", "FWD"):
        m = win[nm]
        scores[nm] = {"n": int(m.sum()), "base_rate": float(y[m].mean()),
                      "model_auc": R12.auc(P[m], y[m]),
                      "single_feature_auc": R12.auc(D.rng_atr.to_numpy()[m], y[m])}
        print(f"  {nm:>4} n={m.sum():>7,} AUC {scores[nm]['model_auc']:.4f} "
              f"(단일피쳐 {scores[nm]['single_feature_auc']:.4f})", flush=True)

    ART.mkdir(parents=True, exist_ok=True)
    joblib.dump(models, ART / "model.joblib")
    (ART / "meta.json").write_text(json.dumps({
        "rule_id": "eth_v_rebound_hgb_allbars_20260919",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "replaces": "TabPFN in-context (frozen 18,000-row context, device=cuda)",
        "features": LIVE.FEATURES,
        "seeds": R12.SEEDS,
        "model": "HistGradientBoostingClassifier x5 (max_iter=300, lr=0.05, leaves=31, l2=1.0)",
        "train": {"n": int(tr.sum()), "label_rate": float(y[tr].mean()),
                  "first": str(D.timestamp[tr].min()), "last": str(D.timestamp[tr].max())},
        "population": "every bar x both sides, 3-state label (v_rebound / chop / ambiguous 제외) "
                      "-- 2026-09-01 모집단 복원, TRAIN 184k @14.6% (기록 182,969 @14.63%)",
        "label_parity_vs_frozen_context": agree,
        "proba_threshold": thr,
        "proba_threshold_percentile": 1.0 - target,
        "threshold_basis": f"배포 TabPFN 의 {DEPLOYED_FIRES_PER_DAY} 발동봉/일 재현 "
                           f"(실현 {fires:.2f}/일). 확률값이 아니라 발동률을 옮긴 것.",
        "scores": scores,
        "tabpfn_head_to_head_same_rows": {
            "source": "scripts/research_eth_v_rebound_hgb_vs_tabpfn_20260919.py",
            "VAL": {"tabpfn": 0.7025, "hgb": 0.7027},
            "OOS": {"tabpfn": 0.7102, "hgb": 0.7125},
            "FWD": {"tabpfn": 0.7029, "hgb": 0.7036},
            "live_cycle_sec": {"tabpfn": 6.62, "hgb": 0.0127},
            "caveat": "차이는 앙상블 분산감소 범위. 경제성 게이트는 아직 재통과하지 않았다."},
    }, ensure_ascii=False, indent=1))
    print(f"\n산출물 {ART}/model.joblib ({(ART / 'model.joblib').stat().st_size / 1e6:.1f}MB) · meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
