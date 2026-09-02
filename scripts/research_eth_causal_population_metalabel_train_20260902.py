#!/usr/bin/env python3
"""증거신호 8종 인과 모집단 메타라벨 학습 (2026-09-02, 재료화 1단계 -- 서버 GPU/TabPFN).

`research_eth_causal_population_metalabel_prep_20260902.py`가 만든 인과 발동집합(raw 트리거,
cluster_dedup 없음)으로 TabPFN을 재학습한다. 라벨(K/HORIZON)과 피처(Tier0 23)는 배포본과
동일하게 고정했으므로, AUC 차이는 순수하게 **모집단 효과**다.

시드는 원 파이프라인과 동일한 4개: [20260829, 141592, 271828, 577215]

⚠️**HOLDOUT AUC는 계산하지 않는다.** 확률은 전 구간(HOLDOUT 포함)에 대해 저장하는데, 그건
재료 텐서를 만들기 위한 추론이지 평가가 아니다. 홀드아웃 성능 판정은 하류 사용자 몫으로 남긴다.

실행: 서버 quant_ai 환경 (GPU + .env의 TABPFN_TOKEN 필요)
  python scripts/research_eth_causal_population_metalabel_train_20260902.py
"""
from __future__ import annotations

import json
import os
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
from sklearn.metrics import roc_auc_score  # noqa: E402

DIR = ROOT / "tmp/eth_causal_population_metalabel_20260902"
SEEDS = [20260829, 141592, 271828, 577215]
# 배포본(앵커 모집단) AUC -- 참고용. 모집단이 다르므로 직접 비교가 아니라 맥락이다.
DEPLOYED = {
    "taker_delta_z_climax": (0.622, 0.608), "short_term_return_z": (0.674, 0.649),
    "liquidity_sweep": (0.659, 0.637), "orthogonal_combo": (0.665, 0.680),
    "smt_divergence": (0.6613, 0.6253), "fib_extension_exhaustion": (0.6054, 0.6201),
    "demarker_extreme": (0.7527, 0.7157), "kalman_deviation_meanrev": (0.6569, 0.6311),
}


def log(m): print(f"[train] {m}", flush=True)


def main() -> int:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("TABPFN_TOKEN="):
                os.environ["TABPFN_TOKEN"] = line.split("=", 1)[1].strip().strip('"')
    from tabpfn import TabPFNClassifier

    cfg = json.loads((DIR / "config.json").read_text())
    feats = cfg["features"]
    out = []
    for name in cfg["cfg"]:
        f = DIR / f"{name}_causal_fires.csv"
        if not f.exists():
            log(f"⚠️ {name}: 파일 없음"); continue
        d = pd.read_csv(f, parse_dates=["timestamp"])
        X = d[feats].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        tr = (d.split == "TRAIN").to_numpy()
        X = X.fillna(X[tr].median())
        y = d["hit"].to_numpy()
        probas = []
        for sd in SEEDS:
            clf = TabPFNClassifier(device="cuda", random_state=sd)
            clf.fit(X[tr].to_numpy(), y[tr])
            probas.append(clf.predict_proba(X.to_numpy())[:, 1])
        P = np.vstack(probas)
        d["proba"] = P.mean(axis=0)
        d["proba_std"] = P.std(axis=0)
        rec = {"signal": name, "n_train": int(tr.sum()),
               "hit_rate_train": round(float(y[tr].mean()), 4)}
        for wn in ("VAL", "OOS"):
            m = (d.split == wn).to_numpy()
            aucs = [roc_auc_score(y[m], P[i][m]) for i in range(len(SEEDS))]
            rec[f"auc_{wn.lower()}"] = round(float(np.mean(aucs)), 4)
            rec[f"auc_{wn.lower()}_std"] = round(float(np.std(aucs)), 4)
            rec[f"n_{wn.lower()}"] = int(m.sum())
        dep = DEPLOYED.get(name)
        if dep:
            rec["deployed_val"], rec["deployed_oos"] = dep
            rec["d_val"] = round(rec["auc_val"] - dep[0], 4)
            rec["d_oos"] = round(rec["auc_oos"] - dep[1], 4)
        out.append(rec)
        d[["pos", "timestamp", "side", "is_bottom", "hit", "split", "proba", "proba_std"]] \
            .to_csv(DIR / f"{name}_causal_proba.csv", index=False)
        log(f"{name:26s} VAL {rec['auc_val']:.4f}±{rec['auc_val_std']:.4f} "
            f"OOS {rec['auc_oos']:.4f}±{rec['auc_oos_std']:.4f} "
            f"| 배포(앵커) {dep[0]:.3f}/{dep[1]:.3f} | Δ {rec.get('d_val',0):+.4f}/{rec.get('d_oos',0):+.4f}")

    s = pd.DataFrame(out)
    s.to_csv(DIR / "causal_auc_summary.csv", index=False)
    log("\n=== 요약 ===")
    print(s[["signal", "n_train", "hit_rate_train", "auc_val", "auc_oos",
             "deployed_val", "deployed_oos", "d_val", "d_oos"]].to_string(index=False))
    log(f"\n산출: {DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
