#!/usr/bin/env python3
"""h48qual/zig075 pinned102(canonical데이터, base_cols 원본과 동일 102개 고정, 진짜 sidecar,
VAL-best threshold 재튜닝) 번들의 Fresh-Forward 6창 평가 -- posfix 평가와 동일 파이프라인
재사용, BUNDLES만 교체(사용자 지시 "2,3,4번 진행" 전부 반영):
  (2) 피쳐셋 원본과 동일 102개로 고정 -- 158 vs 102 교란변수 제거
  (3) 이 번들 전용으로 새로 학습한 진짜 risk sidecar -- 원본 sidecar 빌려쓰던 근사치 제거
  (4) quality_threshold를 각 컴포넌트 자신의 VAL-best로 재선택(h48qual=0.40, zig075=0.80,
      원본 0.50/0.75 그대로 재사용 안 함) -- threshold 미조정 교란변수 제거
남은 교란변수는 (1) 단일시드뿐 -- 이건 이번 라운드에서도 미해결로 명시.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_eth_odyssey4_posfix_canonicaldata_freshforward_20260818 as ev  # noqa: E402

ev.OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_odyssey4_pinned102_freshforward_20260818"
ev.BUNDLES = {
    "h48qual": {
        "bundle": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_pinned102_20260818/true_3head_tabm_bundle.pt",
        "q_tag": "q040",
        "threshold": 0.40,
        "atr_window": 192, "tp_mult": 12.0, "sl_mult": 6.0,
        "min_tp": 0.075, "min_sl": 0.040, "max_tp": 0.22, "max_sl": 0.12,
        "sidecar_pkl": ROOT / "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_h48qual_pinned102_q040_20260818/risk_sidecar.pkl",
        "exit_threshold": 0.95,
    },
    "zig075": {
        "bundle": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_pinned102_20260818/true_3head_tabm_bundle.pt",
        "q_tag": "q080",
        "threshold": 0.80,
        "atr_window": 192, "tp_mult": 12.0, "sl_mult": 6.0,
        "min_tp": 0.075, "min_sl": 0.040, "max_tp": 0.22, "max_sl": 0.12,
        "sidecar_pkl": ROOT / "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_zig075_pinned102_q080_20260818/risk_sidecar.pkl",
        "exit_threshold": 0.95,
    },
}
# original bundles' 102-col set is what these are pinned to -- confirmed identical set/order to
# the original, so the same "zero cmamba/risk cols expected" exception used for posfix is not
# needed here either, but harmless to leave empty explicitly for clarity.
ev._EXPECTED_ZERO_COLS = set()

if __name__ == "__main__":
    result = ev.main()
    import json
    report_path = ev.OUT_DIR / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["risk_sizing_source"] = (
        "each_pinned102_bundle_own_newly_trained_real_sidecar -- trained specifically for these "
        "bundles via train_eval_omega4_2_risk_sidecar_eth_pinned102_{h48qual,zig075}_20260818.py, "
        "constraint_pass=True/fallback_used=False for both (confirmed from each sidecar's own "
        "report.json). NOT borrowed from the original bundles. Note: the validation_mdd_abs "
        "selection floor had to be relaxed from the original 8.0 to 50.0 (the default/original "
        "value made both grids infeasible: 'no eligible validation-only risk mapping') -- the "
        "achieved MDD is whatever the grid search actually selected under the wider floor, not "
        "constrained to be small; read the actual with_gate MDD numbers below at face value."
    )
    report["quality_threshold_note"] = (
        "VAL-best per component's own quality_threshold_ranking.csv (h48qual=0.40/q040, "
        "zig075=0.80/q080), NOT the original bundles' 0.50/0.75 -- re-tuned for these specific "
        "pinned102 bundles."
    )
    report["remaining_confound_note"] = (
        "Single seed only (--seed 260620) -- not addressed this round. Per "
        "tabm_hp_low_signal_pattern memory, a single-seed result cannot distinguish a genuine "
        "effect from seed noise."
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(result)
