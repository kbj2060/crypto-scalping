#!/usr/bin/env python3
"""158개 DC 캐노니컬 피쳐 "정리"(리던던시 감사) -- 사용자 요청 피쳐셋 연구 1단계.
이미 알려진 정확중복쌍(smart_money_flow≡oi_change_rate, funding_z_score≡ou_funding_z,
feature_engineering_edge_research_20260817에서 확인) 외에 158개 전체에서
|pearson corr|>=0.95인 쌍을 그래프 연결요소(union-find)로 묶어 클러스터를 찾고,
클러스터당 대표 피쳐 1개(클러스터 내 individual direction-agnostic AUC 최고)만 남긴
pruned 리스트를 만든다. 158개 전부 이미 개별로는 비유의(eth_dc_feature_set_information_content_20260820.json,
empirical_p=0.325~0.650)이므로 대표선정은 "그나마 정보량 큰 쪽"이라는 결정론적 동률처리 규칙일 뿐,
통계적으로 의미 있는 선택이라 주장하지 않는다.

상관관계는 라벨과 무관한 X-only 구조 분석이라 2025(train)+2026(eval) 전체 bar를 풀링해서 계산한다
(레이블 리키지 우려 없음 -- 예측/학습이 아니라 피쳐 간 중복도 측정)."""
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
LABEL_DIR = ROOT / "tmp/eth_directional_change_triple_barrier_labels_dense_cashfill_20260819"
CORR_THRESHOLD = 0.95

sys.path.insert(0, str(ROOT / "scripts"))
import eth_directional_change_tabm_training_canonicaldata_20260819 as canon  # noqa: E402
omega = canon.omega


def auc_dir_agnostic(y: np.ndarray, x: np.ndarray) -> float:
    valid = ~np.isnan(x)
    if valid.sum() < 30 or len(np.unique(y[valid])) < 2:
        return float("nan")
    auc = roc_auc_score(y[valid], x[valid])
    return max(auc, 1.0 - auc)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def main() -> None:
    train, eval_df = omega._load_omega_frames()[:2]
    pooled = pd.concat(
        [train[BASE_158].apply(pd.to_numeric, errors="coerce"),
         eval_df[BASE_158].apply(pd.to_numeric, errors="coerce")],
        ignore_index=True,
    )
    print(f"상관관계 계산 대상: {len(pooled):,}행 x {len(BASE_158)}피쳐 (2025+2026 풀링)", flush=True)

    stds = pooled.std()
    dead_cols = sorted(stds[stds.fillna(0.0) < 1e-12].index.tolist())
    print(f"\n완전상수(std<1e-12, 순수 죽은 컬럼) {len(dead_cols)}개 -- 상관관계 분석에서 별도 제외: {dead_cols}", flush=True)
    live_cols = [c for c in BASE_158 if c not in dead_cols]

    corr = pooled[live_cols].corr(method="pearson")
    corr_abs = corr.abs().to_numpy().copy()
    np.fill_diagonal(corr_abs, 0.0)

    # 개별 AUC 재계산 (동률처리용) -- 정보량 스크립트와 동일 이벤트 모집단
    frames = []
    for year, feat in ((2025, train), (2026, eval_df)):
        f = feat[["timestamp", *BASE_158]].copy()
        f["timestamp"] = pd.to_datetime(f["timestamp"])
        lbl = pd.read_csv(LABEL_DIR / f"zigzag_action_labels_{year}.csv", usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"])
        frames.append(f.merge(lbl, on="timestamp", how="inner"))
    events = pd.concat(frames, ignore_index=True)
    events = events[events["zigzag_action"] != 0].reset_index(drop=True)
    y = (events["zigzag_action"] == 1).to_numpy().astype(np.int64)
    real_auc = {c: auc_dir_agnostic(y, pd.to_numeric(events[c], errors="coerce").to_numpy(dtype=np.float64)) for c in BASE_158}

    # union-find로 |corr|>=0.95 연결요소 클러스터링 (live_cols만 대상 -- dead_cols는 별도 제외됨)
    n = len(live_cols)
    uf = UnionFind(n)
    high_pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            if corr_abs[i, j] >= CORR_THRESHOLD:
                uf.union(i, j)
                high_pairs.append((live_cols[i], live_cols[j], float(corr_abs[i, j])))
    high_pairs.sort(key=lambda t: -t[2])

    clusters: dict[int, list[str]] = {}
    for i, c in enumerate(live_cols):
        clusters.setdefault(uf.find(i), []).append(c)

    pruned: list[str] = []
    cluster_report = []
    for members in clusters.values():
        if len(members) == 1:
            pruned.append(members[0])
            continue
        rep = max(members, key=lambda c: (real_auc[c] if not np.isnan(real_auc[c]) else -1.0))
        pruned.append(rep)
        cluster_report.append({"members": members, "representative": rep,
                                "member_aucs": {c: real_auc[c] for c in members}})

    pruned.sort()
    print(f"\n|corr|>={CORR_THRESHOLD} 쌍: {len(high_pairs)}개, 다중원소 클러스터: {len(cluster_report)}개", flush=True)
    print(f"158개 -> 죽은컬럼 {len(dead_cols)}개 제외 -> {len(live_cols)}개 중 중복클러스터 {len(cluster_report)}개 정리 -> 최종 {len(pruned)}개", flush=True)
    print("\n최고 상관관계 top15:", flush=True)
    for a, b, c in high_pairs[:15]:
        print(f"    {a:45s} <-> {b:45s} corr={c:.4f}", flush=True)
    print("\n다중원소 클러스터 상세:", flush=True)
    for cr in cluster_report:
        print(f"  대표={cr['representative']} (AUC={real_auc[cr['representative']]:.4f}) <- 원소: {cr['members']}", flush=True)

    # 정리 후 잔존 최고 상관관계 확인 (클러스터링이 실제로 리던던시를 제거했는지 sanity check)
    pruned_idx = [live_cols.index(c) for c in pruned]
    sub_corr = corr_abs[np.ix_(pruned_idx, pruned_idx)]
    n_nan = int(np.isnan(sub_corr).sum())
    remaining_max = float(np.nanmax(sub_corr)) if sub_corr.size else 0.0
    print(f"\n정리 후 잔존 최고 |corr| = {remaining_max:.4f} (임계값 {CORR_THRESHOLD} 미만이어야 정상, NaN항목 {n_nan}개)", flush=True)
    if n_nan:
        nan_mask = np.isnan(sub_corr)
        nan_features = sorted({pruned[i] for i in range(len(pruned)) if nan_mask[i].any()})
        print(f"  [경고] 예상외 NaN 상관관계 피쳐(재조사 필요): {nan_features}", flush=True)

    out = {
        "base_158_count": 158, "dead_cols": dead_cols, "live_count": len(live_cols),
        "pruned_count": len(pruned), "corr_threshold": CORR_THRESHOLD,
        "high_corr_pairs": [{"a": a, "b": b, "corr": c} for a, b, c in high_pairs],
        "clusters": cluster_report, "pruned_features": pruned, "remaining_max_corr_after_prune": remaining_max,
    }
    out_path = ROOT / "tmp/eth_dc_feature_redundancy_audit_20260820.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (SCRATCH / "dc_pruned_features_20260820.json").write_text(json.dumps(pruned, indent=2), encoding="utf-8")
    print(f"\n[report] {out_path}")
    print(f"[pruned list] {SCRATCH / 'dc_pruned_features_20260820.json'}")


if __name__ == "__main__":
    main()
