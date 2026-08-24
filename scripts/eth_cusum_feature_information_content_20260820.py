#!/usr/bin/env python3
"""후보②(CUSUM+TB) 라벨의 피쳐 정보량 저비용 선행체크 -- DC(①) 전체 파이프라인(TabM N=5+
N-HiTS+4가지 안정화기법)을 그대로 CUSUM에 반복하기 전에, DC θ=0.004/θ=0.015에 썼던 것과
정확히 같은 방법(AUC+permutation null)을 CUSUM 기본설정(cusum_k=1.0, 이미 빌드된
`tmp/eth_cusum_triple_barrier_labels_20260819/`)에 먼저 돌려본다 -- 학습 없이 몇 초 안에
"이 라벨에 애초에 배울 신호가 있는지"를 알 수 있어서, 있으면 전체투자, 없으면(DC와 같은
패턴이면) 투자 규모를 사용자와 다시 상의할 근거가 된다.

DC와 CUSUM은 이미 통계적으로 거의 포함관계(DC이벤트의 94.9%가 CUSUM과 5분이내 매칭,
방향일치율 94.1%)임이 확인돼 있어 결과가 비슷할 것으로 예상되지만, 추측 대신 직접 잰다."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path("/tmp/claude-1000/-home-kbj20-crypto-scalping/7445be14-7df6-4085-bc4a-6a5de4e4597d/scratchpad")
BASE_158 = json.loads((SCRATCH / "dc_base_158_cols.json").read_text())

sys.path.insert(0, str(ROOT / "scripts"))
import eth_directional_change_tabm_training_canonicaldata_20260819 as canon  # noqa: E402
omega = canon.omega

CUSUM_LABEL_DIR = ROOT / "tmp/eth_cusum_triple_barrier_labels_20260819"
RNG = np.random.default_rng(20260820)
N_PERM = 200
REF = {
    "dc_theta_0.004": {"n_events": 12193, "mean_auc": 0.5038, "max_auc": 0.5141, "empirical_p": 0.380},
    "dc_theta_0.015": {"n_events": 1817, "mean_auc": 0.5071, "max_auc": 0.5293, "empirical_p": 0.825},
}


def auc_dir_agnostic(y: np.ndarray, x: np.ndarray) -> float:
    valid = ~np.isnan(x)
    if valid.sum() < 30 or len(np.unique(y[valid])) < 2:
        return float("nan")
    auc = roc_auc_score(y[valid], x[valid])
    return max(auc, 1.0 - auc)


def main() -> None:
    train, eval_df = omega._load_omega_frames()[:2]
    feat_frames = []
    for feat in (train, eval_df):
        f = feat[["timestamp", *BASE_158]].copy()
        f["timestamp"] = pd.to_datetime(f["timestamp"])
        feat_frames.append(f)
    feat_all = pd.concat(feat_frames, ignore_index=True)

    lbl = pd.concat([
        pd.read_csv(CUSUM_LABEL_DIR / "zigzag_action_labels_2025.csv", usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"]),
        pd.read_csv(CUSUM_LABEL_DIR / "zigzag_action_labels_2026.csv", usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"]),
    ], ignore_index=True)

    data = feat_all.merge(lbl, on="timestamp", how="inner")
    events = data[data["zigzag_action"] != 0].reset_index(drop=True)
    y = (events["zigzag_action"] == 1).to_numpy().astype(np.int64)
    n_long, n_short = int(y.sum()), int((1 - y).sum())
    print(f"CUSUM(cusum_k=1.0, 2025+2026, 피쳐프레임 매칭) 이벤트: {len(events):,}개 (LONG={n_long} SHORT={n_short})\n", flush=True)

    for k, v in REF.items():
        print(f"참고 -- {k}: n={v['n_events']:,} mean_auc={v['mean_auc']:.4f} max_auc={v['max_auc']:.4f} empirical_p={v['empirical_p']:.3f}", flush=True)
    print(flush=True)

    X = {}
    real_auc = {}
    for c in BASE_158:
        x = pd.to_numeric(events[c], errors="coerce").to_numpy(dtype=np.float64)
        X[c] = x
        real_auc[c] = auc_dir_agnostic(y, x)
    vals = {k: v for k, v in real_auc.items() if not np.isnan(v)}
    arr = np.array(list(vals.values()))
    top5 = sorted(vals.items(), key=lambda kv: kv[1], reverse=True)[:5]
    mean_auc, max_auc = float(arr.mean()), float(arr.max())
    print(f"CUSUM: mean_auc={mean_auc:.4f} max_auc={max_auc:.4f}", flush=True)
    for feat, a in top5:
        print(f"  top: {feat:50s} auc={a:.4f}", flush=True)

    Xmat = np.column_stack([X[c] for c in BASE_158])
    null_max = []
    for _ in range(N_PERM):
        y_perm = RNG.permutation(y)
        aucs = []
        for j in range(Xmat.shape[1]):
            xj = Xmat[:, j]
            valid = ~np.isnan(xj)
            if valid.sum() < 30:
                continue
            a = roc_auc_score(y_perm[valid], xj[valid])
            aucs.append(max(a, 1 - a))
        null_max.append(max(aucs) if aucs else float("nan"))
    null_arr = np.array(null_max)
    p95 = float(np.nanpercentile(null_arr, 95))
    empirical_p = float((null_arr >= max_auc).mean())
    print(f"\npermutation null: 95th={p95:.4f} empirical_p={empirical_p:.3f} "
          f"({'유의' if empirical_p < 0.05 else '비유의(chance)'})", flush=True)

    out = {"n_events": int(len(events)), "n_long": n_long, "n_short": n_short,
           "mean_auc": mean_auc, "max_auc": max_auc, "top5": [(f, float(a)) for f, a in top5],
           "null_p95": p95, "empirical_p": empirical_p, "reference_dc": REF}
    out_path = ROOT / "tmp/eth_cusum_feature_information_content_20260820.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[report] {out_path}")


if __name__ == "__main__":
    main()
