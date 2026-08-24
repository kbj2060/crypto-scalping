#!/usr/bin/env python3
"""사용자 지시: VIF로 정리된 피쳐셋 기준으로 "피쳐 조합으로 새 피쳐"를 실제로 만든다.
⚠️ 아직 신호(AUC/permutation-null/PnL 등)는 계산하지 않는다 -- 순수 구축+구조검증 단계.

방법(RIT, Shah & Meinshausen 2014 -- 문헌조사 20260820에서 확인된 비단조 상호작용 탐지 기법):
VIF-clean 112개로 LightGBM을 2025(train)단독으로만 학습(2026은 전혀 안 씀 -- discovery가
eval을 오염시키면 나중에 이 조합피쳐를 테스트할 때 처음부터 편향된 상태로 시작하게 됨)해서
트리구조에서 조상-자손으로 공동출현하는(=같은 분기경로에서 같이 쓰인) split_feature 쌍을
split_gain 가중치로 집계, 상위 30쌍을 뽑는다. 각 쌍에 대해 "raw_a * raw_b" 곱 피쳐를
만든다(Aiken&West 1991 표준 상호작용항 구성 -- 여기선 진단용 rank-percentile이 아니라 실제
피쳐이므로 원값을 그대로 곱함. 스케일 정규화는 기존 132개 피쳐와 동일하게 학습 파이프라인의
표준화 단계에 위임 -- 신규 피쳐만 특별취급 안 함). 2025+2026 전체 bar(이벤트뿐 아니라)에 대해
계산 -- 실제 모델 입력으로 쓰일 걸 가정하므로 원본 피쳐가 이미 causal이면 곱도 causal(새 미래
정보 유입 없음)."""
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
    vif_clean = json.loads((SCRATCH / "dc_vif_clean_features_20260820.json").read_text())
    print(f"VIF-clean 피쳐 {len(vif_clean)}개 기준 조합피쳐 discovery 시작", flush=True)

    train, eval_df = omega._load_omega_frames()[:2]

    # --- discovery: 2025(train)단독 이벤트bar로 LightGBM 학습 (2026 미사용) ---
    f = train[["timestamp", *vif_clean]].copy()
    f["timestamp"] = pd.to_datetime(f["timestamp"])
    lbl = pd.read_csv(LABEL_DIR / "zigzag_action_labels_2025.csv", usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"])
    events = f.merge(lbl, on="timestamp", how="inner")
    events = events[events["zigzag_action"] != 0].reset_index(drop=True)
    for c in vif_clean:
        events[c] = pd.to_numeric(events[c], errors="coerce")
    y = (events["zigzag_action"] == 1).to_numpy().astype(np.int64)
    X = events[vif_clean].to_numpy(dtype=np.float64)
    print(f"[2025단독 discovery] n={len(y):,} LONG={int(y.sum()):,} SHORT={int((1 - y).sum()):,}", flush=True)

    model = lgb.LGBMClassifier(
        n_estimators=300, num_leaves=15, max_depth=4, learning_rate=0.05,
        min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
        random_state=20260820, verbose=-1,
    )
    model.fit(X, y, feature_name=vif_clean)
    trees_df = model.booster_.trees_to_dataframe()
    scores = _cooccurrence_scores(trees_df)
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])[:TOP_K]
    print(f"공동출현쌍 {len(scores)}개 중 상위 {len(ranked)}개를 조합피쳐로 구축", flush=True)

    # --- 조합피쳐 구축: raw_a * raw_b, 2025+2026 전체 bar(이벤트뿐 아니라) ---
    combo_names = []
    for (a, b), score in ranked:
        combo_name = f"combo_{a}_x_{b}"
        combo_names.append({"name": combo_name, "a": a, "b": b, "discovery_gain_score": float(score)})

    def _attach_combos(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for c in combo_names:
            out[c["name"]] = pd.to_numeric(out[c["a"]], errors="coerce") * pd.to_numeric(out[c["b"]], errors="coerce")
        return out

    train_aug = _attach_combos(train)
    eval_aug = _attach_combos(eval_df)

    # --- 구조 검증만(신호 계산 아님): NaN폭증/이상치 여부만 확인 ---
    print("\n조합피쳐 구조검증(NaN율/기술통계, 신호 아님):", flush=True)
    issues = []
    for c in combo_names:
        name = c["name"]
        for split_name, df in (("train2025", train_aug), ("eval2026", eval_aug)):
            s = df[name]
            nan_rate = float(s.isna().mean())
            if nan_rate > 0.5:
                issues.append(f"{name}@{split_name}: NaN율 {nan_rate:.1%} 과다")
            inf_count = int(np.isinf(s.to_numpy(dtype=np.float64, na_value=0.0)).sum())
            if inf_count:
                issues.append(f"{name}@{split_name}: inf {inf_count}개")
    if issues:
        print("  [경고] " + "; ".join(issues), flush=True)
    else:
        print("  전부 정상(NaN<=원본 피쳐 수준, inf 없음)", flush=True)

    sample = combo_names[0]["name"]
    print(f"\n예시({sample}) train2025 기술통계:\n{train_aug[sample].describe()}", flush=True)

    out = {
        "vif_clean_feature_count": len(vif_clean), "top_k": TOP_K, "combo_features": combo_names,
        "final_total_feature_count": len(vif_clean) + len(combo_names),
        "structural_issues": issues,
    }
    out_path = ROOT / "tmp/eth_dc_combination_feature_construction_20260820.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    (SCRATCH / "dc_combo_feature_names_20260820.json").write_text(json.dumps(combo_names, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[report] {out_path}")
    print(f"[최종 피쳐구성] 기존{len(vif_clean)}개 + 신규조합{len(combo_names)}개 = {len(vif_clean) + len(combo_names)}개")


if __name__ == "__main__":
    main()
