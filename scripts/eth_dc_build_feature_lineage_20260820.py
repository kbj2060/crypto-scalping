#!/usr/bin/env python3
"""사용자 지시: 154개 엔지니어링 피쳐셋을 "피쳐셋 리니지"로 보존. 손으로 다시 타이핑하지 않고
실제 저장된 리포트(redundancy audit/VIF elimination/combo construction/financial-ML construction)
에서 정확한 수치를 읽어 조립 -- 사람이 옮겨적으며 실수할 위험을 없앤다."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path("/tmp/claude-1000/-home-kbj20-crypto-scalping/7445be14-7df6-4085-bc4a-6a5de4e4597d/scratchpad")


def main() -> None:
    redund = json.loads((ROOT / "tmp/eth_dc_feature_redundancy_audit_20260820.json").read_text())
    vif = json.loads((ROOT / "tmp/eth_dc_feature_vif_iterative_elimination_20260820.json").read_text())
    combo = json.loads((ROOT / "tmp/eth_dc_combination_feature_construction_20260820.json").read_text())
    finml = json.loads((ROOT / "tmp/eth_dc_financial_ml_feature_construction_20260820.json").read_text())

    dead_cols = redund["dead_cols"]
    cluster_removed = [m for c in redund["clusters"] for m in c["members"] if m != c["representative"]]
    assert redund["base_158_count"] - redund["pruned_count"] == len(dead_cols) + len(cluster_removed)

    regime3_simplex_removed = "regime3_current_sensitive_wide24_chop_prob"

    lineage = {
        "feature_set_id": "eth_dc_engineered154_20260820",
        "final_feature_count": vif["final_feature_count"] + combo["top_k"] + len(finml["feature_names"]),
        "wrapper_script": "scripts/eth_dc_engineered_features_canonicaldata_20260820.py",
        "label": "ETH 5m Directional-Change (DC) triple-barrier direction label",
        "parent_data": {
            "train_csv": "data/splits/year_oos/training_features_2025.csv",
            "eval_csv": "data/splits/year_oos/training_features_2026_rebuilt.csv (REGIME3_CURRENT_2026-filtered, 51,746 rows)",
        },
        "stages": [
            {
                "stage": 0, "name": "canonical_base_158",
                "feature_count": 158,
                "description": "Auto-derived numeric feature columns from omega._numeric_feature_cols() on the canonical TRAIN/EVAL frames (includes regime3 HMM overlays).",
                "source_list": "dc_base_158_cols.json (scratchpad)",
            },
            {
                "stage": 1, "name": "dead_column_removal",
                "operation": "Exact zero-variance (std<1e-12, pooled 2025+2026) columns dropped -- placeholder REGIME3_CMAMBA/RISK overlays that were never populated with real data.",
                "removed_count": len(dead_cols), "removed": sorted(dead_cols),
                "input_count": 158, "output_count": 158 - len(dead_cols),
                "script": "scripts/eth_dc_feature_redundancy_audit_20260820.py",
                "report": "tmp/eth_dc_feature_redundancy_audit_20260820.json",
            },
            {
                "stage": 2, "name": "pairwise_correlation_cluster_dedup",
                "operation": "Pearson |corr|>=0.95 union-find clustering (mathematically single-linkage); one representative per cluster kept (highest individual direction-agnostic AUC vs DC label, tie-break only -- all individually non-significant).",
                "removed_count": len(cluster_removed), "removed": sorted(cluster_removed),
                "clusters": redund["clusters"],
                "input_count": 158 - len(dead_cols), "output_count": redund["pruned_count"],
                "script": "scripts/eth_dc_feature_redundancy_audit_20260820.py",
                "report": "tmp/eth_dc_feature_redundancy_audit_20260820.json",
            },
            {
                "stage": 3, "name": "probability_simplex_exact_collinearity_fix",
                "operation": "VIF diagnostic (correlation-matrix-inverse diagonal) found regime3_current_sensitive_wide24_{bull,bear,chop}_prob at VIF~3e13; verified bull+bear+chop=1.0 exactly on every row both years (probability simplex constraint, undetectable by pairwise correlation) -- dropped chop_prob (bull/bear jointly determine it, zero information loss).",
                "removed_count": 1, "removed": [regime3_simplex_removed],
                "input_count": redund["pruned_count"], "output_count": redund["pruned_count"] - 1,
                "script": "scripts/eth_dc_feature_vif_check_20260820.py",
                "report": "tmp/eth_dc_feature_vif_check_20260820.json",
            },
            {
                "stage": 4, "name": "iterative_vif_elimination",
                "operation": "Repeated: compute VIF for all surviving features (corr-matrix-inverse diagonal), drop the single highest-VIF feature, recompute -- until max VIF < 10 (O'Brien 2007 convention, not a derived threshold). Catches multivariate collinearity distributed across 3+ features that pairwise correlation structurally cannot see.",
                "removed_count": vif["n_removed"],
                "elimination_trace": vif["elimination_trace"],
                "input_count": vif["start_count"], "output_count": vif["final_feature_count"],
                "final_max_vif": vif["final_max_vif"],
                "script": "scripts/eth_dc_feature_vif_iterative_elimination_20260820.py",
                "report": "tmp/eth_dc_feature_vif_iterative_elimination_20260820.json",
            },
            {
                "stage": 5, "name": "rit_tree_discovered_combination_features",
                "operation": "LightGBM fit on 2025(train)-only DC events (2026 kept fully independent for later testing); ancestor-descendant split_feature co-occurrence within trees, split-gain-weighted, top-30 pairs; constructed as combo_a_x_b = raw_a * raw_b across all 2025+2026 bars.",
                "added_count": combo["top_k"], "added": [c["name"] for c in combo["combo_features"]],
                "construction_detail": combo["combo_features"],
                "input_count": vif["final_feature_count"], "output_count": vif["final_feature_count"] + combo["top_k"],
                "script": "scripts/eth_dc_combination_feature_construction_20260820.py",
                "report": "tmp/eth_dc_combination_feature_construction_20260820.json",
            },
            {
                "stage": 6, "name": "financial_ml_literature_gap_features",
                "operation": "Literature gap analysis (financial time-series ML + crypto-specific quant research, cross-checked against actual feature code to rule out false-positive overlaps) found 9 standard feature families absent from the 158-canonical set; implemented the computationally-tractable ones: fractional differentiation (d in {0.3,0.5,0.7}), return-sign Shannon entropy, Corwin-Schultz spread, Roll's implied spread (adapted to log_return for cross-period stationarity), Kyle's Lambda, VPIN approximation, realized semivariance ratio, realized kurtosis, Lo-MacKinlay variance ratio (q in {4,12}). SADF/multifractal-DFA/transfer-entropy/Hawkes deferred (implementation complexity/runtime cost).",
                "added_count": len(finml["feature_names"]), "added": sorted(finml["feature_names"]),
                "input_count": vif["final_feature_count"] + combo["top_k"],
                "output_count": vif["final_feature_count"] + combo["top_k"] + len(finml["feature_names"]),
                "script": "scripts/eth_dc_financial_ml_feature_construction_20260820.py",
                "report": "tmp/eth_dc_financial_ml_feature_construction_20260820.json",
                "literature_review": "docs/feature_redundancy_and_interaction_literature_review_20260820.md",
            },
        ],
        "signal_test_results": {
            "individual_new42_permutation_null": {"empirical_p": 0.460, "script": "scripts/eth_dc_new42_feature_information_content_20260820.py"},
            "tabm_n5seed_retrain": {
                "individual_cond_acc_range": [48.8, 51.0],
                "baseline_158feat_cond_acc_range": [48.2, 51.4],
                "oos_pnl_sign_consistent": False,
                "ensemble_cond_acc": 49.7, "ensemble_pnl": -19.96, "ensemble_mdd": -30.27,
                "script": "scripts/eth_dc_engineered154_training_runner_20260820.py",
                "analysis_script": "scripts/eth_dc_engineered154_analysis_20260820.py",
            },
            "verdict": "chance-level across every test -- see research_line_registry.json id eth_dc_feature_engineering_redundancy_combination_finml_20260820",
        },
        "docs": [
            "docs/experiments/eth_directional_change_tabm_nhits_training_20260819.md",
            "docs/feature_redundancy_and_interaction_literature_review_20260820.md",
        ],
    }

    out_path = ROOT / "docs/model_contracts/eth_dc_engineered_feature_set_lineage_20260820.json"
    out_path.write_text(json.dumps(lineage, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[lineage] {out_path}")
    print(f"최종 피쳐 개수: {lineage['final_feature_count']}")
    for s in lineage["stages"]:
        print(f"  stage{s['stage']} {s['name']}: {s.get('input_count', s.get('feature_count'))} -> {s.get('output_count', s.get('feature_count'))}")


if __name__ == "__main__":
    main()
