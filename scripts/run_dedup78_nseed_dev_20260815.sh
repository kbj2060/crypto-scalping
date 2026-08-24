#!/bin/bash
# Driver for the Odyssey 78-feature (dedup of live 102) N=5-seed retrain, BOTH h48qual and
# zig075 (2026-08-15, dev machine). Reuses the exact deployed recipe for each component via the
# pinned78 wrapper (scripts/train_eval_omega4_3head_parent72_pinned78_20260815.py) -- only
# --seed varies across the SAME 5 genuinely random seeds already used for the 102-feature zig075
# formal N-seed test (paired comparison, seeds not freshly drawn).
set -uo pipefail

cd "$(dirname "$0")/.."
SEEDS=(946043153 932925759 74851798 975176982 542143953)
LOG_DIR="tmp/eth_dedup78_nseed_skill_retest_20260815"
PY="/home/kbj20/anaconda3/envs/quant_ai/bin/python3"
mkdir -p "$LOG_DIR"
FAILURES=()

DIR_LABEL_DIR="tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531"
H48_QUALITY_LABEL_DIR="tmp/causal_regen_20260516/omega_zigzag_fix_all_solutions_20260630/label_contracts/sltp_h48_conservative_padded_to_zigzag_timestamps"

for SEED in "${SEEDS[@]}"; do
    echo "########## zig075 SEED $SEED $(date -Iseconds) ##########"
    LOGFILE="$LOG_DIR/pinned78_zig075_seed${SEED}.log"
    echo "=== [$(date -Iseconds)] START zig075_pinned78_seed${SEED} ===" | tee "$LOGFILE"
    if "$PY" scripts/train_eval_omega4_3head_parent72_pinned78_20260815.py \
        --epochs 2 \
        --quality-mode same_as_direction \
        --direction-label-dir "$DIR_LABEL_DIR" \
        --quality-thresholds 0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95 \
        --max-exit-samples 30000 \
        --max-train-rows 0 \
        --exit-label-mode entry_label_terminal_giveback \
        --out-suffix "pinned78_zig075_dedup_seed${SEED}" \
        --device cpu \
        --seed "$SEED" >>"$LOGFILE" 2>&1; then
        echo "=== [$(date -Iseconds)] OK zig075_pinned78_seed${SEED} ===" | tee -a "$LOGFILE"
    else
        rc=$?
        echo "=== [$(date -Iseconds)] FAILED zig075_pinned78_seed${SEED} (exit $rc) ===" | tee -a "$LOGFILE"
        FAILURES+=("zig075_seed${SEED}")
    fi

    echo "########## h48qual SEED $SEED $(date -Iseconds) ##########"
    LOGFILE="$LOG_DIR/pinned78_h48qual_seed${SEED}.log"
    echo "=== [$(date -Iseconds)] START h48qual_pinned78_seed${SEED} ===" | tee "$LOGFILE"
    if "$PY" scripts/train_eval_omega4_3head_parent72_pinned78_20260815.py \
        --epochs 2 \
        --max-train-rows 0 \
        --max-exit-samples 30000 \
        --quality-thresholds 0.50 \
        --exit-label-mode entry_label_terminal_giveback \
        --direction-label-dir "$DIR_LABEL_DIR" \
        --quality-mode quality_label_action \
        --quality-label-dir "$H48_QUALITY_LABEL_DIR" \
        --out-suffix "pinned78_h48qual_dedup_seed${SEED}" \
        --device cpu \
        --seed "$SEED" >>"$LOGFILE" 2>&1; then
        echo "=== [$(date -Iseconds)] OK h48qual_pinned78_seed${SEED} ===" | tee -a "$LOGFILE"
    else
        rc=$?
        echo "=== [$(date -Iseconds)] FAILED h48qual_pinned78_seed${SEED} (exit $rc) ===" | tee -a "$LOGFILE"
        FAILURES+=("h48qual_seed${SEED}")
    fi
done

echo "########## SUMMARY $(date -Iseconds) ##########"
if [ "${#FAILURES[@]}" -eq 0 ]; then
    echo "ALL 10 RUNS OK"
else
    echo "FAILURES (${#FAILURES[@]}): ${FAILURES[*]}"
fi
