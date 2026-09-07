#!/usr/bin/env python3
"""MASHT 섀도우 아티팩트 동결 (2026-09-07).

사용자: *"조건 없이 그대로 새도우 러너를 만들어줘"*

섀도우 러너가 매 앵커마다 재현 가능하게 쓰려면 **세 가지를 동결**해야 한다:
  1. MultiRocket / Hydra 변환 -- 학습 구간에서 fit 한 그대로 (랜덤커널·다일레이션·바이어스)
  2. TabPFN in-context 학습 문맥 -- (X, y). TabPFN 은 파라미터를 적합하지 않으므로
     "모델 파일"이 아니라 **문맥 자체**가 아티팩트다.
  3. 진입 임계 -- **표본외(워크포워드) 예측**의 70분위(상위 30%).
     라이브에서 다시 계산하면 안 된다(그 순간 표본에 맞춰 움직여 사후 최적화가 된다).
     🔴1차 시도는 학습 문맥 자신에 대한 예측으로 임계를 잡았는데, TabPFN 은 in-context 라
     자기 문맥을 그대로 맞힌다(학습내 상위30% 적중 **1.0000**, 임계 0.8504). 그 임계는
     암기된 분포에서 나온 값이라 라이브 예측 분포와 어긋난다. 반드시 표본외에서 뽑는다.

## 동결 시점
학습 = y_bin 이 유효하고 timestamp < 2026-08-01 인 앵커 전부.
(전방 확인에 쓴 2026-06-30~07-31 구간도 **포함**한다 -- 그 창은 이미 열어봤으므로
 홀드아웃으로서 가치가 없고, 문맥에 넣는 편이 배포에 유리하다. 섀도우의 판정은
 앞으로 쌓일 새 데이터로만 한다.)

## 산출
`data/live/masht_wbin_shadow_artifact/` 에
  rocket.joblib   fit 된 MultiRocket + Hydra
  context.npz     X(float32), y(int8) -- TabPFN in-context 문맥
  meta.json       임계값·채널·K·비용·동결시각·학습범위
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
WT = ROOT / "tmp/eth_anchor_window_tensor_20260907"
WF_PRED = ROOT / "tmp/eth_anchor_masht_wf_20260907/preds_wbin_masht2784.npy"
FWD = ROOT / "tmp/eth_anchor_forward_window_20260907"
OUT = ROOT / "data/live/masht_wbin_shadow_artifact"
FREEZE_END = pd.Timestamp("2026-08-01")
SEED = 20260907
TOP_Q = 0.70          # 상위 30% 진입
N_EST = 4


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    from aeon.transformations.collection.convolution_based import MultiRocket, HydraTransformer
    from tabpfn import TabPFNClassifier
    import torch

    Xa = np.load(WT / "X.npy"); va = np.load(WT / "valid.npy")
    Ia = pd.read_parquet(WT / "index.parquet")
    Xb = np.load(FWD / "X.npy"); vb = np.load(FWD / "valid.npy")
    Ib = pd.read_parquet(FWD / "index.parquet")
    cols = [c for c in Ia.columns if c in Ib.columns]
    X = np.concatenate([Xa, Xb], axis=0)
    v = np.concatenate([va, vb])
    I = pd.concat([Ia[cols], Ib[cols]], ignore_index=True)
    ok = v & np.isfinite(I["y_bin"].to_numpy()) & (I["timestamp"] < FREEZE_END).to_numpy()
    print(f"[1/4] 후보 {len(I):,} → 동결 학습 {ok.sum():,} "
          f"({I.timestamp[ok].min()} ~ {I.timestamp[ok].max()})", flush=True)

    t0 = time.time()
    A = X[ok].astype(np.float32)
    mr = MultiRocket(n_kernels=252, random_state=SEED, n_jobs=8).fit(A)
    hy = HydraTransformer(n_kernels=8, n_groups=16, random_state=SEED, n_jobs=8).fit(A)
    F = np.concatenate([np.asarray(mr.transform(A)), np.asarray(hy.transform(A))],
                       axis=1).astype(np.float32)
    y = I["y_bin"].to_numpy()[ok].astype(np.int8)
    print(f"[2/4] 변환 fit+transform {F.shape} ({time.time()-t0:.0f}s) · 지속률 {y.mean():.4f}", flush=True)

    # 임계: **표본외** 워크포워드 예측의 70분위. 학습 문맥 자신으로 잡으면 안 된다.
    wf = np.load(WF_PRED)
    wf = wf[np.isfinite(wf)]
    thr = float(np.quantile(wf, TOP_Q))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # 참고용: 학습 문맥 자신의 예측 분포(암기 확인 -- 임계에는 쓰지 않는다)
    clf = TabPFNClassifier(device=dev, n_estimators=N_EST, random_state=SEED,
                           ignore_pretraining_limits=True, memory_saving_mode=True)
    clf.fit(F, y)
    p_in = clf.predict_proba(F)[:, 1]
    print(f"[3/4] 진입 임계(**표본외 워크포워드** {TOP_Q:.0%}분위, n={len(wf)}) = {thr:.6f}", flush=True)
    print(f"      워크포워드 예측 분위: 50% {np.quantile(wf,.5):.4f} · 70% {thr:.4f} "
          f"· 90% {np.quantile(wf,.9):.4f}", flush=True)
    print(f"      ⚠️참고: 학습문맥 자기예측 70분위 {np.quantile(p_in, TOP_Q):.4f} "
          f"(암기 -- 상위30% 적중 {y[p_in >= np.quantile(p_in, TOP_Q)].mean():.4f}, 임계에 쓰지 않음)",
          flush=True)
    print(f"      임계 초과 비율(표본외) {np.mean(wf >= thr):.3f}", flush=True)

    joblib.dump({"multirocket": mr, "hydra": hy}, OUT / "rocket.joblib")
    np.savez_compressed(OUT / "context.npz", X=F, y=y)
    (OUT / "meta.json").write_text(json.dumps({
        "rule_id": "masht_wbin_top30_20260907",
        "model": "TabPFN in-context (MultiRocket 2016 + Hydra 768 = 2784)",
        "arm": "wbin", "K": 48,
        "channels": json.loads((WT / "meta.json").read_text())["channels"],
        "cont_sign": "top=+1 / bottom=-1",
        "freeze_end": str(FREEZE_END), "n_context": int(ok.sum()),
        "train_range": [str(I.timestamp[ok].min()), str(I.timestamp[ok].max())],
        "base_rate": float(y.mean()), "entry_quantile": TOP_Q, "entry_threshold": thr,
        "threshold_source": "표본외 워크포워드 예측(tmp/eth_anchor_masht_wf_20260907)의 70분위 -- "
                            "학습문맥 자기예측은 TabPFN in-context 암기라 쓰지 않음",
        "label": {"barrier_pct": 1.0, "H_bars": 48, "resolution": "1m first touch",
                  "entry": "open[t+1]"},
        "cost_bp": {"taker": 10.0, "maker": 7.8}, "breakeven_acc": (100 + 7.8) / 200,
        "measured": {"walkforward_top30_acc": 0.5988, "ci": [0.5394, 0.6456],
                     "dayblock_null_p": 0.017, "pooled_auc": 0.5453,
                     "forward_check_2026_07": {"top30_acc": 0.64, "n_entries": 25,
                                               "random_entry_p": 0.282,
                                               "base_rate": 0.5714}},
        "n_estimators": N_EST, "seed": SEED, "built_utc": pd.Timestamp.utcnow().isoformat(),
    }, indent=1, ensure_ascii=False))
    sz = sum(f.stat().st_size for f in OUT.iterdir()) / 1e6
    print(f"[4/4] 저장: {OUT} ({sz:.1f}MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
