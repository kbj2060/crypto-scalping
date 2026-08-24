#!/usr/bin/env python3
"""eth_dc_gbm_interaction_discovery_20260820.py가 2025(train)단독 LightGBM 트리구조에서 뽑은
top-30 후보쌍을, 완전히 분리된 2026(eval) 데이터에서만 rank-product AUC + permutation-null로
검증한다(discovery에 전혀 안 쓰인 데이터라 진짜 out-of-sample). 후보가 8,778개가 아니라 30개뿐이라
다중비교 부담이 훨씬 작다(K=30 귀무분포) -- 트리기반 발견이 실제로 신호를 못 보던 걸 찾아낸다면
여기서 유의하게 나와야 한다."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "tmp/eth_directional_change_triple_barrier_labels_dense_cashfill_20260819"
N_PERM = 200
RNG = np.random.default_rng(20260823)  # discovery(LightGBM seed 20260820)와 무관한 별개 시드

sys.path.insert(0, str(ROOT / "scripts"))
import eth_directional_change_tabm_training_canonicaldata_20260819 as canon  # noqa: E402
omega = canon.omega


def main() -> None:
    disc = json.loads((ROOT / "tmp/eth_dc_gbm_interaction_discovery_20260820.json").read_text())
    pairs = [(p["a"], p["b"]) for p in disc["top_pairs"]]
    cols = sorted({c for pr in pairs for c in pr})
    print(f"검증대상: 2025단독 트리구조 발견 top-{len(pairs)}쌍 ({len(cols)}개 고유피쳐)", flush=True)

    _, eval_df = omega._load_omega_frames()[:2]
    f = eval_df[["timestamp", *cols]].copy()
    f["timestamp"] = pd.to_datetime(f["timestamp"])
    lbl = pd.read_csv(LABEL_DIR / "zigzag_action_labels_2026.csv", usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"])
    events = f.merge(lbl, on="timestamp", how="inner")
    events = events[events["zigzag_action"] != 0].reset_index(drop=True)
    for c in cols:
        events[c] = pd.to_numeric(events[c], errors="coerce")
    complete = events.dropna(subset=cols).reset_index(drop=True)
    print(f"[2026단독] 이벤트bar {len(events):,}개 중 완전케이스 {len(complete):,}개", flush=True)

    y = (complete["zigzag_action"] == 1).to_numpy().astype(np.int64)
    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    assert 0 < n_pos < n
    print(f"n={n:,} LONG={n_pos:,} SHORT={n_neg:,}", flush=True)

    Z = {}
    for c in cols:
        x = complete[c].to_numpy(dtype=np.float64)
        order = np.argsort(x)
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1, dtype=np.float64)
        Z[c] = ranks / n - 0.5

    rank_const = n_pos * (n_pos + 1) / 2.0
    Y_perm = np.column_stack([RNG.permutation(y) for _ in range(N_PERM)]).astype(np.float64)
    assert np.allclose(Y_perm.sum(axis=0), n_pos)

    real_auc = []
    null_max_per_perm = np.zeros(N_PERM, dtype=np.float64)
    for a, b in pairs:
        score = Z[a] * Z[b]
        order = np.argsort(score)
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1, dtype=np.float64)

        rank_sum_pos = ranks[y == 1].sum()
        auc = (rank_sum_pos - rank_const) / (n_pos * n_neg)
        auc_da = max(auc, 1.0 - auc)
        real_auc.append((a, b, float(auc_da)))

        rank_sum_pos_perm = Y_perm.T @ ranks
        auc_perm = (rank_sum_pos_perm - rank_const) / (n_pos * n_neg)
        auc_perm_da = np.maximum(auc_perm, 1.0 - auc_perm)
        null_max_per_perm = np.maximum(null_max_per_perm, auc_perm_da)

    real_auc.sort(key=lambda t: -t[2])
    real_max = real_auc[0][2]
    p95 = float(np.percentile(null_max_per_perm, 95))
    empirical_p = float((null_max_per_perm >= real_max).mean())

    print(f"\n[2025발견-2026검증 최고AUC] {real_max:.4f} ({real_auc[0][0]} x {real_auc[0][1]})", flush=True)
    print(f"[귀무분포(K={len(pairs)}쌍, 훨씬 작은 다중비교부담)] 95th={p95:.4f} empirical_p={empirical_p:.3f}", flush=True)
    print("\n전체 30쌍 2026단독 AUC:", flush=True)
    for a, b, auc in real_auc:
        print(f"    {a:40s} x {b:40s} auc={auc:.4f}", flush=True)

    out = {"n_pairs_tested": len(pairs), "n_common_rows": n, "n_pos": n_pos, "n_neg": n_neg,
           "all_pairs_auc": real_auc, "real_max_auc": real_max, "null_p95": p95, "empirical_p": empirical_p}
    out_path = ROOT / "tmp/eth_dc_gbm_interaction_2026_validation_20260820.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[report] {out_path}")


if __name__ == "__main__":
    main()
