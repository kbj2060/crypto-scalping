#!/usr/bin/env python3
"""일리아스1의 quality_threshold를 "VAL PnL 최대화"가 아니라 "baseline v1과 같은 자기분포
percentile"로 재선택해 6창 Fresh-Forward 재평가 -- eth_ilias1_h48qual_quality_gate_
selectivity_shift_20260818 발견([[..]]) 직후 사용자 요청.

VAL 기준 baseline v1의 quality_for_action(non-CASH 조건부) percentile:
  h48qual: threshold=0.50 -> 97.55백분위 (통과율 2.45%)
  zig075:  threshold=0.75 -> 91.58백분위 (통과율 8.42%)
일리아스1 자기 분포에서 같은 percentile에 해당하는 값(VAL에서 산출, 재사용 안 하고 매번 재계산
안 함 -- 원본 threshold 재선택 절차와 동일하게 VAL 1회 고정):
  h48qual: 0.40 -> 0.6730 (기존보다 훨씬 더 선별적으로 바뀜)
  zig075:  0.80 -> 0.7631 (기존보다 약간 덜 선별적으로 바뀜 -- zig075도 사실 percentile
           불일치가 있었음, 방향은 h48qual과 반대)

sidecar는 기존 pinned102 전용 sidecar(threshold=0.40/0.80 기준 entries로 학습됨)를 그대로
재사용 -- 이 새 threshold(0.67/0.76)로 생성되는 entry 분포는 그 sidecar가 학습한 분포와
정확히 같지 않다(더 selective한 h48qual entries는 average quality_score가 더 높음). 이건
research 진단 단계의 명시적 근사치이며, 이 결과가 유의미하면 전용 sidecar 재학습이 다음 단계다.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import eval_eth_odyssey4_posfix_canonicaldata_freshforward_20260818 as ev  # noqa: E402

ev.OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_ilias1_pctile_matched_threshold_freshforward_20260818"
ev.BUNDLES = {
    "h48qual": {
        "bundle": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_pinned102_20260818/true_3head_tabm_bundle.pt",
        "q_tag": "q067_pctile_matched",
        "threshold": 0.6730,
        "atr_window": 192, "tp_mult": 12.0, "sl_mult": 6.0,
        "min_tp": 0.075, "min_sl": 0.040, "max_tp": 0.22, "max_sl": 0.12,
        "sidecar_pkl": ROOT / "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_h48qual_pinned102_q040_20260818/risk_sidecar.pkl",
        "exit_threshold": 0.95,
    },
    "zig075": {
        "bundle": ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_pinned102_20260818/true_3head_tabm_bundle.pt",
        "q_tag": "q076_pctile_matched",
        "threshold": 0.7631,
        "atr_window": 192, "tp_mult": 12.0, "sl_mult": 6.0,
        "min_tp": 0.075, "min_sl": 0.040, "max_tp": 0.22, "max_sl": 0.12,
        "sidecar_pkl": ROOT / "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_zig075_pinned102_q080_20260818/risk_sidecar.pkl",
        "exit_threshold": 0.95,
    },
}
ev._EXPECTED_ZERO_COLS = set()

if __name__ == "__main__":
    result = ev.main()
    import json
    report_path = ev.OUT_DIR / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["threshold_selection_method"] = (
        "percentile_matched_to_baseline_v1 -- NOT VAL-PnL-maximized. h48qual threshold=0.6730 "
        "(baseline's 0.50 sits at VAL 97.55th percentile of baseline's own quality_for_action|"
        "non-CASH distribution; 0.6730 is Ilias 1's own VAL distribution value at that same "
        "percentile). zig075 threshold=0.7631 (baseline's 0.75 sits at VAL 91.58th percentile; "
        "0.7631 is the matching value in Ilias 1's own distribution). See "
        "eth_ilias1_h48qual_quality_gate_selectivity_shift_20260818 memory for the full derivation."
    )
    report["sidecar_confound_note"] = (
        "Risk sidecars are the SAME pinned102 sidecars trained for the VAL-PnL-optimal thresholds "
        "(0.40/0.80), NOT retrained for these new percentile-matched thresholds (0.6730/0.7631). "
        "The entry population at the new thresholds differs (higher average quality_score for "
        "h48qual, since it's now far more selective) -- a mild distribution-shift approximation, "
        "not a clean test. Flagged, not silently passed as exact."
    )
    report["remaining_confound_note"] = (
        "Single seed only (--seed 260620) -- not addressed this round."
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(result)
