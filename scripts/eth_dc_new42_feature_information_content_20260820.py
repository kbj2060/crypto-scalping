#!/usr/bin/env python3
"""사용자 지시("그렇게 해줘" = 신호계산 진행)에 따른 첫 단계: 154피쳐 중 진짜 신규인 42개
(조합30 + financial-ML12)만 개별 정보량 체크. 기존 112개(VIF-clean)는 158개 전체가 이미
eth_dc_feature_set_information_content_20260820.py에서 개별 비유의(p=0.325~0.650) 확인됐으므로
재검증 불필요 -- 이번엔 진짜 한번도 개별테스트 안 된 42개만."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "tmp/eth_directional_change_triple_barrier_labels_dense_cashfill_20260819"
N_PERM = 200
RNG = np.random.default_rng(20260824)

sys.path.insert(0, str(ROOT / "scripts"))
import eth_dc_engineered_features_canonicaldata_20260820 as eng  # noqa: E402
omega = eng.omega

NEW_42 = sorted(c["name"] for c in eng.COMBO_FEATURES) + sorted(eng.FINML_NAMES)
assert len(NEW_42) == 42


def auc_dir_agnostic(y: np.ndarray, x: np.ndarray) -> float:
    valid = ~np.isnan(x)
    if valid.sum() < 30 or len(np.unique(y[valid])) < 2:
        return float("nan")
    auc = roc_auc_score(y[valid], x[valid])
    return max(auc, 1.0 - auc)


def main() -> None:
    train, eval_df = omega._load_omega_frames()[:2]
    frames = []
    for year, feat in ((2025, train), (2026, eval_df)):
        f = feat[["timestamp", *NEW_42]].copy()
        f["timestamp"] = pd.to_datetime(f["timestamp"])
        lbl = pd.read_csv(LABEL_DIR / f"zigzag_action_labels_{year}.csv", usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"])
        frames.append(f.merge(lbl, on="timestamp", how="inner"))
    data = pd.concat(frames, ignore_index=True)
    events = data[data["zigzag_action"] != 0].reset_index(drop=True)
    y = (events["zigzag_action"] == 1).to_numpy().astype(np.int64)
    print(f"이벤트bar {len(events):,}개 (LONG={int(y.sum())} SHORT={int((1 - y).sum())})", flush=True)

    real_auc: dict[str, float] = {}
    X: dict[str, np.ndarray] = {}
    for c in NEW_42:
        x = pd.to_numeric(events[c], errors="coerce").to_numpy(dtype=np.float64)
        X[c] = x
        real_auc[c] = auc_dir_agnostic(y, x)

    ranked = sorted(real_auc.items(), key=lambda kv: (kv[1] if not np.isnan(kv[1]) else -1), reverse=True)
    print("\n상위 15개(개별 AUC):", flush=True)
    for feat, a in ranked[:15]:
        tag = "[조합]" if feat.startswith("combo_") else "[금융ML]"
        print(f"    {tag} {feat:45s} auc={a:.4f}", flush=True)

    print(f"\npermutation null 계산 중 (N={N_PERM})...", flush=True)
    Xmat = np.column_stack([X[c] for c in NEW_42])
    null_max = []
    for i in range(N_PERM):
        y_perm = RNG.permutation(y)
        aucs = []
        for j in range(len(NEW_42)):
            xj = Xmat[:, j]
            valid = ~np.isnan(xj)
            if valid.sum() < 30:
                continue
            a = roc_auc_score(y_perm[valid], xj[valid])
            aucs.append(max(a, 1 - a))
        null_max.append(max(aucs) if aucs else float("nan"))

    null_arr = np.array(null_max)
    real_max = ranked[0][1]
    p95 = float(np.nanpercentile(null_arr, 95))
    empirical_p = float((null_arr >= real_max).mean())
    print(f"\n[신규42개] 실제 최고AUC={real_max:.4f}({ranked[0][0]}) vs null 95th={p95:.4f} "
          f"empirical_p={empirical_p:.3f}", flush=True)

    out = {"n_features": 42, "n_events": len(events), "top15": [(f, float(a)) for f, a in ranked[:15]],
           "real_max_auc": real_max, "null_p95": p95, "empirical_p": empirical_p}
    out_path = ROOT / "tmp/eth_dc_new42_feature_information_content_20260820.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[report] {out_path}")


if __name__ == "__main__":
    main()
