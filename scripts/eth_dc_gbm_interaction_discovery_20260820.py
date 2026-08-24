#!/usr/bin/env python3
"""문헌조사 결과 반영 2/2: 내 rank-product 상호작용테스트는 "매끄러운 단조 사분면"(bilinear)
형태에만 민감하고, 국소적/비단조/문턱형 상호작용(예: "A>p90 이고 B가 중간대일 때만")은 구조적으로
못 본다는 게 문헌조사에서 확인됨(H-statistic/SHAP-interaction/RIT류가 이걸 다룸). shap/statsmodels
패키지는 미설치(+공유 conda env에 새 의존성 추가 회피)라, RIT(Random Intersection Trees, Shah&
Meinshausen 2014)와 같은 개념으로 LightGBM 트리구조에서 "같은 경로에 같이 등장하는 조상-자손
split_feature 쌍"을 split_gain 가중치로 집계해 후보쌍을 랭킹한다.

⚠️ discovery는 2025(train)만 사용 -- 2026은 완전히 안 건드림, 이래야 다음 단계(2026단독 검증)가
진짜 out-of-sample이 된다. pooled로 하면 이전 rank-product 테스트가 겪은 것과 같은 문제(pooled서
유의해 보이다 2026단독서 붕괴)가 discovery 단계에서부터 재발할 위험이 있음."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path("/tmp/claude-1000/-home-kbj20-crypto-scalping/7445be14-7df6-4085-bc4a-6a5de4e4597d/scratchpad")

PRUNED = json.loads((SCRATCH / "dc_pruned_features_20260820.json").read_text())
# VIF 점검에서 발견된 정확 선형종속(bull+bear+chop=1.0 -- 모든 행에서 std=0으로 검증됨) 중
# 1개 제거(둘이 나머지 하나를 결정하므로) -- chop_prob 제거, bull/bear는 방향예측에 더 직접적.
PRUNED = [c for c in PRUNED if c != "regime3_current_sensitive_wide24_chop_prob"]
assert len(PRUNED) == 132

LABEL_DIR = ROOT / "tmp/eth_directional_change_triple_barrier_labels_dense_cashfill_20260819"
TOP_K = 30

sys.path.insert(0, str(ROOT / "scripts"))
import eth_directional_change_tabm_training_canonicaldata_20260819 as canon  # noqa: E402
omega = canon.omega


def _cooccurrence_scores(trees_df: pd.DataFrame) -> dict[tuple[str, str], float]:
    scores: dict[tuple[str, str], float] = defaultdict(float)
    split_rows = trees_df[trees_df["split_feature"].notna()]
    for tree_idx, tdf in split_rows.groupby("tree_index"):
        by_node = tdf.set_index("node_index")
        parent_map = tdf.set_index("node_index")["parent_index"].to_dict()
        for node_idx, row in by_node.iterrows():
            ancestors = []
            cur = parent_map.get(node_idx)
            seen = set()
            while isinstance(cur, str) and cur in by_node.index and cur not in seen:
                seen.add(cur)
                ancestors.append(by_node.loc[cur, "split_feature"])
                cur = parent_map.get(cur)
            this_feat = row["split_feature"]
            this_gain = float(row["split_gain"])
            for anc_feat in ancestors:
                if anc_feat == this_feat:
                    continue
                key = tuple(sorted((anc_feat, this_feat)))
                scores[key] += this_gain
    return scores


def main() -> None:
    train, _ = omega._load_omega_frames()[:2]
    f = train[["timestamp", *PRUNED]].copy()
    f["timestamp"] = pd.to_datetime(f["timestamp"])
    lbl = pd.read_csv(LABEL_DIR / "zigzag_action_labels_2025.csv", usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"])
    events = f.merge(lbl, on="timestamp", how="inner")
    events = events[events["zigzag_action"] != 0].reset_index(drop=True)
    for c in PRUNED:
        events[c] = pd.to_numeric(events[c], errors="coerce")

    y = (events["zigzag_action"] == 1).to_numpy().astype(np.int64)
    X = events[PRUNED].to_numpy(dtype=np.float64)
    print(f"[2025단독 discovery] n={len(y):,} LONG={int(y.sum()):,} SHORT={int((1 - y).sum()):,}", flush=True)

    model = lgb.LGBMClassifier(
        n_estimators=300, num_leaves=15, max_depth=4, learning_rate=0.05,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        random_state=20260820, verbose=-1,
    )
    model.fit(X, y, feature_name=PRUNED)
    trees_df = model.booster_.trees_to_dataframe()
    n_splits = int(trees_df["split_feature"].notna().sum())
    print(f"트리 {model.n_estimators_}개, 총 split노드 {n_splits}개", flush=True)

    scores = _cooccurrence_scores(trees_df)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    print(f"\n조상-자손 공동출현 쌍 {len(ranked)}개 발견, 상위 {TOP_K}개:", flush=True)
    top_pairs = []
    for (a, b), score in ranked[:TOP_K]:
        print(f"    {a:40s} x {b:40s} gain_weighted_score={score:.2f}", flush=True)
        top_pairs.append({"a": a, "b": b, "score": float(score)})

    out = {"n_events_2025": len(y), "n_trees": int(model.n_estimators_), "n_splits": n_splits,
           "top_k": TOP_K, "top_pairs": top_pairs, "pruned_132_used": PRUNED}
    out_path = ROOT / "tmp/eth_dc_gbm_interaction_discovery_20260820.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[report] {out_path}")


if __name__ == "__main__":
    main()
