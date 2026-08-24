#!/usr/bin/env python3
"""정리된(133개) DC 피쳐셋 pairwise 상호작용("조합") 유의성 테스트 -- 사용자 요청 피쳐셋
연구 2단계(1단계=eth_dc_feature_redundancy_audit_20260820.py). 개별 피쳐 158개는 전부 이미
permutation-null 기각됐지만(eth_dc_feature_set_information_content_20260820.py), TabM이
스스로 학습하는 암묵적 상호작용과 별개로 "명시적으로 구성한 2-피쳐 조합"이 단독으로는
안 보이는 방향정보를 갖는지 직접 검증한다.

방법: 각 피쳐를 rank-percentile로 변환(z=rank/n-0.5, 스케일 무관/이상치에 강건) 후, 정리된
133개 중 모든 쌍(C(133,2)=8,778개)에 대해 interaction=z_i*z_j를 만들고 direction-agnostic
AUC(LONG/SHORT)를 계산한다. 8,778개를 동시에 테스트하는 다중비교 문제이므로, 라벨 200회
셔플로 "8,778개 조합 중 순전히 우연으로 나올 수 있는 최고 AUC" 귀무분포를 만들어 실제 최고
AUC의 유의성을 판정한다(개별 피쳐 정보량 스크립트와 동일 방법론, K(비교 대상 개수)만 다름).

성능: 상호작용 점수의 랭크는 라벨과 무관하므로 실데이터 1회만 계산 후 재사용하고, 퍼뮤테이션
200회는 행렬곱(Y_perm.T @ ranks)으로 한번에 처리한다. i를 축으로 블록 단위 처리해 피크 메모리를
O(n x 133)으로 제한(전체 8,778열을 동시에 메모리에 올리지 않음)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path("/tmp/claude-1000/-home-kbj20-crypto-scalping/7445be14-7df6-4085-bc4a-6a5de4e4597d/scratchpad")

PRUNED = json.loads((SCRATCH / "dc_pruned_features_20260820.json").read_text())
LABEL_DIR = ROOT / "tmp/eth_directional_change_triple_barrier_labels_dense_cashfill_20260819"
N_PERM = 200
RNG = np.random.default_rng(20260821)  # 정보량 스크립트(seed=20260820)와 다른 시드 -- 별개 검정

sys.path.insert(0, str(ROOT / "scripts"))
import eth_directional_change_tabm_training_canonicaldata_20260819 as canon  # noqa: E402
omega = canon.omega


def main() -> None:
    train, eval_df = omega._load_omega_frames()[:2]
    frames = []
    for year, feat in ((2025, train), (2026, eval_df)):
        f = feat[["timestamp", *PRUNED]].copy()
        f["timestamp"] = pd.to_datetime(f["timestamp"])
        lbl = pd.read_csv(LABEL_DIR / f"zigzag_action_labels_{year}.csv", usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"])
        frames.append(f.merge(lbl, on="timestamp", how="inner"))
    events = pd.concat(frames, ignore_index=True)
    events = events[events["zigzag_action"] != 0].reset_index(drop=True)
    for c in PRUNED:
        events[c] = pd.to_numeric(events[c], errors="coerce")

    complete = events.dropna(subset=PRUNED).reset_index(drop=True)
    print(f"이벤트bar {len(events):,}개 중 133피쳐 전부 non-NaN인 공통행 {len(complete):,}개 "
          f"({len(complete) / len(events):.1%}) -- 모든 쌍에 동일 모집단 사용(공정 비교)", flush=True)

    y = (complete["zigzag_action"] == 1).to_numpy().astype(np.int64)
    n = len(y)
    n_pos = int(y.sum())
    n_neg = n - n_pos
    assert 0 < n_pos < n, "LONG/SHORT 둘 다 있어야 함"
    print(f"n={n:,} LONG={n_pos:,} SHORT={n_neg:,}", flush=True)

    X = complete[PRUNED].to_numpy(dtype=np.float64)
    order = np.argsort(X, axis=0)
    ranks = np.empty_like(order, dtype=np.float64)
    rows = np.arange(1, n + 1, dtype=np.float64)
    np.put_along_axis(ranks, order, rows[:, None], axis=0)
    Z = ranks / n - 0.5

    k = len(PRUNED)
    n_pairs_total = k * (k - 1) // 2
    print(f"피쳐 {k}개 -> 쌍 {n_pairs_total:,}개, 퍼뮤테이션 {N_PERM}회", flush=True)

    Y_perm = np.column_stack([RNG.permutation(y) for _ in range(N_PERM)]).astype(np.float64)
    n_pos_perm = Y_perm.sum(axis=0)
    assert np.allclose(n_pos_perm, n_pos), "순열은 클래스 비율을 보존해야 함"

    real_pair_auc: list[tuple[str, str, float]] = []
    null_max_per_perm = np.zeros(N_PERM, dtype=np.float64)
    rank_const = n_pos * (n_pos + 1) / 2.0

    for i in range(k - 1):
        zi = Z[:, i]
        block = zi[:, None] * Z[:, i + 1:]
        b = block.shape[1]
        b_order = np.argsort(block, axis=0)
        b_ranks = np.empty_like(b_order, dtype=np.float64)
        np.put_along_axis(b_ranks, b_order, rows[:, None], axis=0)

        rank_sum_pos_real = b_ranks[y == 1].sum(axis=0)
        auc_real = (rank_sum_pos_real - rank_const) / (n_pos * n_neg)
        auc_real_da = np.maximum(auc_real, 1.0 - auc_real)
        for jj in range(b):
            real_pair_auc.append((PRUNED[i], PRUNED[i + 1 + jj], float(auc_real_da[jj])))

        rank_sum_pos_perm = Y_perm.T @ b_ranks
        auc_perm = (rank_sum_pos_perm - rank_const) / (n_pos * n_neg)
        auc_perm_da = np.maximum(auc_perm, 1.0 - auc_perm)
        null_max_per_perm = np.maximum(null_max_per_perm, auc_perm_da.max(axis=1))

        if (i + 1) % 20 == 0 or i == k - 2:
            print(f"  진행 {i + 1}/{k - 1} 피쳐 처리됨", flush=True)

    assert len(real_pair_auc) == n_pairs_total
    real_pair_auc.sort(key=lambda t: -t[2])
    real_max = real_pair_auc[0][2]
    p95 = float(np.percentile(null_max_per_perm, 95))
    p99 = float(np.percentile(null_max_per_perm, 99))
    empirical_p = float((null_max_per_perm >= real_max).mean())

    print(f"\n[실제 최고 상호작용 AUC] {real_max:.4f} ({real_pair_auc[0][0]} x {real_pair_auc[0][1]})", flush=True)
    print(f"[귀무분포({n_pairs_total:,}쌍 다중비교, N_PERM={N_PERM})] 95th={p95:.4f} 99th={p99:.4f} "
          f"empirical_p={empirical_p:.3f}", flush=True)
    print("\n상위 20개 조합:", flush=True)
    for a, b_, auc in real_pair_auc[:20]:
        print(f"    {a:40s} x {b_:40s} auc={auc:.4f}", flush=True)

    out = {
        "n_features_pruned": k, "n_pairs": n_pairs_total, "n_common_rows": n, "n_pos": n_pos, "n_neg": n_neg,
        "n_perm": N_PERM, "top20_pairs": real_pair_auc[:20], "real_max_auc": real_max,
        "null_p95": p95, "null_p99": p99, "empirical_p_of_max": empirical_p,
    }
    out_path = ROOT / "tmp/eth_dc_feature_interaction_significance_20260820.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[report] {out_path}")


if __name__ == "__main__":
    main()
