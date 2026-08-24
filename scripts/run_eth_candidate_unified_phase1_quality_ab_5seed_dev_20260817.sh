#!/bin/bash
# Driver for eth_candidate_unified_single_component Phase 1 (see docs/experiments/
# eth_candidate_unified_single_component_redesign_20260817.md): quality-label A/B comparison,
# direction+quality only (exit head trained by the shared script as a side effect but NOT
# evaluated in Phase 1 -- see the design doc's Phase 1 scope). Reuses the exact pinned102 wrapper
# pattern already validated in docs/experiments/eth_omega461_zig075_direction_head_skill_formal_
# nseed_20260815.md (avoids the documented 102->172 feature-drift contamination trap). Both
# variants pinned to h48qual's live 102-column contract for a clean apples-to-apples comparison
# (arbitrary but consistent choice -- only the quality label differs between A and B).
#
# Variant A: quality_mode=quality_label_action + h48_conservative barrier label (h48qual's style).
# Variant B: quality_mode=same_as_direction (zig075's style, no separate quality label).
# Both share direction_label_dir=zigzag_action_labels_20260531 (already common to both live
# components -- confirmed via report.json this session, not assumed).
#
# 5 genuinely random seeds (np.random.SeedSequence(20260817101), NOT reused from any other
# experiment's seed batch), SAME 5 seed values used for both variants (paired design).
set -uo pipefail

cd "$(dirname "$0")/.."
SEEDS=(2559205075 1355646609 2549217127 1801478137 2105606360)
LOG_DIR="tmp/eth_candidate_unified_phase1_quality_ab_20260817"
PY="/home/kbj20/anaconda3/envs/quant_ai/bin/python3"
mkdir -p "$LOG_DIR"
FAILURES=()

DIRECTION_LABEL_DIR="tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531"
QUALITY_LABEL_DIR_A="tmp/causal_regen_20260516/omega_zigzag_fix_all_solutions_20260630/label_contracts/sltp_h48_conservative_padded_to_zigzag_timestamps"
THRESHOLDS="0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85"

run_variant() {
    local variant="$1"; shift
    local extra_args=("$@")
    for SEED in "${SEEDS[@]}"; do
        echo "########## VARIANT $variant SEED $SEED $(date -Iseconds) ##########"
        LOGFILE="$LOG_DIR/${variant}_seed${SEED}.log"
        echo "=== [$(date -Iseconds)] START ${variant}_seed${SEED} ===" | tee "$LOGFILE"
        if "$PY" scripts/train_eval_omega4_3head_parent72_pinned102_20260727.py \
            --pin-component h48qual \
            --epochs 2 \
            --direction-label-dir "$DIRECTION_LABEL_DIR" \
            --quality-thresholds "$THRESHOLDS" \
            --max-exit-samples 30000 \
            --max-train-rows 0 \
            --exit-label-mode entry_label_terminal_giveback \
            --out-suffix "eth_candidate_unified_phase1_${variant}_seed${SEED}" \
            --device cpu \
            --seed "$SEED" \
            "${extra_args[@]}" >>"$LOGFILE" 2>&1; then
            echo "=== [$(date -Iseconds)] OK ${variant}_seed${SEED} ===" | tee -a "$LOGFILE"
        else
            rc=$?
            echo "=== [$(date -Iseconds)] FAILED ${variant}_seed${SEED} (exit $rc) ===" | tee -a "$LOGFILE"
            FAILURES+=("${variant}_seed${SEED}")
        fi
    done
}

run_variant quality_A_barrier --quality-mode quality_label_action --quality-label-dir "$QUALITY_LABEL_DIR_A"
run_variant quality_B_samedir --quality-mode same_as_direction

echo "########## SUMMARY $(date -Iseconds) ##########"
if [ "${#FAILURES[@]}" -eq 0 ]; then
    echo "ALL 10 RUNS OK"
else
    echo "FAILURES (${#FAILURES[@]}): ${FAILURES[*]}"
fi
