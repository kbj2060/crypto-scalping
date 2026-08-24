#!/usr/bin/env python3
"""eth_dc_feature_interaction_significance_20260820.py의 pooled(2025+2026) 결과(p=0.030,
top=eth_btc_beta_residual_z x oi_up_price_down, auc=0.5244)가 CUSUM pooled(p=0.000 -> 2026단독
재검증서 p=0.290로 붕괴, eth_cusum_2026only_feature_information_content_20260820.py 전례)와
같은 train기간-쏠림 아티팩트인지 검증. 방법 동일, 대상 데이터만 2026(eval_df) 단독으로 교체
-- pooled top pair를 그대로 보는 게 아니라 2026 데이터만으로 8,778쌍 전체를 처음부터
다시 랭킹해서(선택편향 없는 완전독립 재발견) 유의성을 판정한다."""
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
RNG = np.random.default_rng(20260822)  # pooled(20260821)과 다른 시드 -- 완전 독립 재검정

sys.path.insert(0, str(ROOT / "scripts"))
import eth_directional_change_tabm_training_canonicaldata_20260819 as canon  # noqa: E402
omega = canon.omega


def main() -> None:
    _, eval_df = omega._load_omega_frames()[:2]
    f = eval_df[["timestamp", *PRUNED]].copy()
    f["timestamp"] = pd.to_datetime(f["timestamp"])
    lbl = pd.read_csv(LABEL_DIR / "zigzag_action_labels_2026.csv", usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"])
    events = f.merge(lbl, on="timestamp", how="inner")
    events = events[events["zigzag_action"] != 0].reset_index(drop=True)
    for c in PRUNED:
        events[c] = pd.to_numeric(events[c], errors="coerce")

    complete = events.dropna(subset=PRUNED).reset_index(drop=True)
    print(f"[2026단독] 이벤트bar {len(events):,}개 중 133피쳐 전부 non-NaN인 공통행 {len(complete):,}개 "
          f"({len(complete) / max(len(events), 1):.1%})", flush=True)

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
    assert np.allclose(Y_perm.sum(axis=0), n_pos)

    real_pair_auc: list[tuple[str, str, float]] = []
    null_max_per_perm = np.zeros(N_PERM, dtype=np.float64)
    rank_const = n_pos * (n_pos + 1) / 2.0
    # pooled 테스트의 1위 쌍 -- 2026단독에서도 순위/AUC를 별도로 추적(재발견 여부 확인용)
    POOLED_TOP_PAIR = ("eth_btc_beta_residual_z", "oi_up_price_down")
    pooled_top_auc_2026 = None

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
            a, bb = PRUNED[i], PRUNED[i + 1 + jj]
            real_pair_auc.append((a, bb, float(auc_real_da[jj])))
            if {a, bb} == set(POOLED_TOP_PAIR):
                pooled_top_auc_2026 = float(auc_real_da[jj])

        rank_sum_pos_perm = Y_perm.T @ b_ranks
        auc_perm = (rank_sum_pos_perm - rank_const) / (n_pos * n_neg)
        auc_perm_da = np.maximum(auc_perm, 1.0 - auc_perm)
        null_max_per_perm = np.maximum(null_max_per_perm, auc_perm_da.max(axis=1))

    assert len(real_pair_auc) == n_pairs_total
    real_pair_auc.sort(key=lambda t: -t[2])
    real_max = real_pair_auc[0][2]
    p95 = float(np.percentile(null_max_per_perm, 95))
    p99 = float(np.percentile(null_max_per_perm, 99))
    empirical_p = float((null_max_per_perm >= real_max).mean())

    print(f"\n[2026단독 실제 최고 상호작용 AUC] {real_max:.4f} ({real_pair_auc[0][0]} x {real_pair_auc[0][1]})", flush=True)
    print(f"[2026단독 귀무분포] 95th={p95:.4f} 99th={p99:.4f} empirical_p={empirical_p:.3f}", flush=True)
    print(f"\n[pooled 1위 쌍 {POOLED_TOP_PAIR} 의 2026단독 AUC] = {pooled_top_auc_2026}", flush=True)
    print("\n2026단독 상위 20개 조합:", flush=True)
    for a, b_, auc in real_pair_auc[:20]:
        print(f"    {a:40s} x {b_:40s} auc={auc:.4f}", flush=True)

    out = {
        "n_features_pruned": k, "n_pairs": n_pairs_total, "n_common_rows": n, "n_pos": n_pos, "n_neg": n_neg,
        "n_perm": N_PERM, "top20_pairs": real_pair_auc[:20], "real_max_auc": real_max,
        "null_p95": p95, "null_p99": p99, "empirical_p_of_max": empirical_p,
        "pooled_top_pair_auc_on_2026only": pooled_top_auc_2026,
    }
    out_path = ROOT / "tmp/eth_dc_feature_interaction_significance_2026only_20260820.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[report] {out_path}")


if __name__ == "__main__":
    main()
