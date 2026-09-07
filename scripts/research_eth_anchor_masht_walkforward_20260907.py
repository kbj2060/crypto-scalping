#!/usr/bin/env python3
"""MASHT 후보를 월간 재학습 walk-forward 로 재측정 (2026-09-07, 서버 GPU).

사용자: *"각 피쳐별과 각 arm 별 최고 조합을 가지고 최고 Top1 을 새도우를 돌려보자"*

섀도우 후보를 같은 잣대로 비교하려면 **같은 프로토콜**이어야 한다.
TabICLv2 후보 2종(`three/perm20`, `wbin/perm20`)은 이미 월간 재학습 예측이 있다
(`tmp/eth_anchor_walkforward_20260907/preds_*.npy`). MASHT 후보(`wbin/masht2784`,
전 셀 중 **min3 최고 0.5325**)만 없어서 여기서 만든다.

비교 잣대는 AUC 가 아니라 **상위 30% 진입 정확도**다 -- 사용자 기준(55%)이 그 형태이고,
`hard/dirtop20` 이 AUC 0.544 인데 상위30% 리프트가 음수였던 전례가 있다.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
WT = ROOT / "tmp/eth_anchor_window_tensor_20260907"
OUT = ROOT / "tmp/eth_anchor_masht_wf_20260907"
EMBARGO = pd.Timedelta(hours=4)
SEED = 20260907
N_EST = 4


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    import torch
    from tabpfn import TabPFNClassifier
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    M = np.load(WT / "masht.npy").astype(np.float32)
    v = np.load(WT / "valid.npy")
    I = pd.read_parquet(WT / "index.parquet")
    I = I[v].reset_index(drop=True)
    y = np.nan_to_num(I["y_bin"].to_numpy()).astype(int)
    mask = np.isfinite(I["y_bin"].to_numpy())
    ts = I["timestamp"]
    print(f"[입력] {M.shape} · wbin 유효 {mask.sum()} · device {dev}", flush=True)

    months, cur = [], pd.Timestamp("2025-09-01")
    while cur <= ts.max():
        months.append(cur); cur = cur + pd.offsets.MonthBegin(1)
    preds = np.full(len(I), np.nan)
    for m0 in months:
        m1 = m0 + pd.offsets.MonthBegin(1)
        te = mask & (ts >= m0).to_numpy() & (ts < m1).to_numpy()
        tr = mask & (ts < (m0 - EMBARGO)).to_numpy()
        if te.sum() < 20 or tr.sum() < 300 or len(np.unique(y[tr])) < 2:
            continue
        t0 = time.time()
        clf = TabPFNClassifier(device=dev, n_estimators=N_EST, random_state=SEED,
                               ignore_pretraining_limits=True, memory_saving_mode=True)
        clf.fit(M[tr], y[tr])
        preds[te] = clf.predict_proba(M[te])[:, 1]
        print(f"   {m0.date()} 학습 {tr.sum():>5} → 예측 {te.sum():>4} ({time.time()-t0:.0f}s)", flush=True)

    np.save(OUT / "preds_wbin_masht2784.npy", preds)
    I.to_parquet(OUT / "index.parquet", index=False)
    sp = I["split"].to_numpy()
    s = mask & np.isfinite(preds) & np.isin(sp, ("VAL", "OOS", "HOLDOUT_SPENT"))
    a = roc_auc_score(y[s], preds[s])
    k = max(10, int(s.sum() * 0.30))
    top = np.argsort(-preds[s])[:k]
    print(f"\n풀링 n={s.sum()} · AUC {a:.4f} · 상위30% 진입 정확도 {y[s][top].mean():.4f} "
          f"(기저 {y[s].mean():.4f})", flush=True)
    (OUT / "summary.json").write_text(json.dumps(
        {"n": int(s.sum()), "pooled_auc": float(a),
         "top30_acc": float(y[s][top].mean()), "base": float(y[s].mean())}, indent=1))
    print(f"저장: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
