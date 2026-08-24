#!/bin/bash
# Driver for the Odyssey2 zig075 direction_head formal N=5-seed retrain (2026-08-15, dev machine).
# Reuses the exact deployed zig075 parent recipe via the pinned102 wrapper (pins base_cols to the
# live zig075 bundle's exact 102-column contract, avoiding feature drift from newer research CSVs)
# -- only --seed varies across the 5 genuinely random seeds.
set -uo pipefail

cd "$(dirname "$0")/.."
SEEDS=(946043153 932925759 74851798 975176982 542143953)
LOG_DIR="tmp/eth_zig075_direction_head_formal_nseed_20260815"
PY="/home/kbj20/anaconda3/envs/quant_ai/bin/python3"
mkdir -p "$LOG_DIR"
FAILURES=()

for SEED in "${SEEDS[@]}"; do
    echo "########## SEED $SEED $(date -Iseconds) ##########"
    LOGFILE="$LOG_DIR/pinned102_parent_seed${SEED}.log"
    echo "=== [$(date -Iseconds)] START zig075_pinned102_seed${SEED} ===" | tee "$LOGFILE"
    if "$PY" scripts/train_eval_omega4_3head_parent72_pinned102_20260727.py \
        --pin-component zig075 \
        --epochs 2 \
        --quality-mode same_as_direction \
        --direction-label-dir tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531 \
        --quality-thresholds 0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95 \
        --max-exit-samples 30000 \
        --max-train-rows 0 \
        --exit-label-mode entry_label_terminal_giveback \
        --out-suffix "pinned102_zig075_formal5seed_20260815_seed${SEED}" \
        --device cpu \
        --seed "$SEED" >>"$LOGFILE" 2>&1; then
        echo "=== [$(date -Iseconds)] OK zig075_pinned102_seed${SEED} ===" | tee -a "$LOGFILE"
    else
        rc=$?
        echo "=== [$(date -Iseconds)] FAILED zig075_pinned102_seed${SEED} (exit $rc) ===" | tee -a "$LOGFILE"
        FAILURES+=("seed${SEED}")
    fi
done

echo "########## SUMMARY $(date -Iseconds) ##########"
if [ "${#FAILURES[@]}" -eq 0 ]; then
    echo "ALL 5 SEEDS OK"
else
    echo "FAILURES (${#FAILURES[@]}): ${FAILURES[*]}"
fi
