#!/bin/bash
# Driver for eth_candidate_unified_single_component Phase 2 (see docs/experiments/
# eth_candidate_unified_single_component_redesign_20260817.md section 3-C / 5-1): retrain the
# exit head (ATR-barrier label, giveback_min recalibrated 0.65->0.25, feature-barrier bug already
# fixed by reusing research_eth_omega461_exit_head_liveatr_relabel_20260813.py's functions
# unmodified) on top of the frozen Phase-1-confirmed Variant B (same_as_direction quality) parent.
# N=5 genuinely random seeds for the exit-head retrain specifically (np.random.SeedSequence(
# 20260817201), NOT reused from any other experiment's seed batch, NOT the ad hoc smoke-test seed
# 3141592653 used to validate the pipeline before this batch).
set -uo pipefail

cd "$(dirname "$0")/.."
SEEDS=(548794457 3646016929 2988156591 858346535 2584959503)
LOG_DIR="tmp/eth_candidate_unified_phase2_exit_head_20260817"
PY="/home/kbj20/anaconda3/envs/quant_ai/bin/python3"
mkdir -p "$LOG_DIR"
FAILURES=()

for SEED in "${SEEDS[@]}"; do
    echo "########## SEED $SEED $(date -Iseconds) ##########"
    LOGFILE="$LOG_DIR/phase2_exit_head_seed${SEED}.log"
    echo "=== [$(date -Iseconds)] START phase2_exit_head_seed${SEED} ===" | tee "$LOGFILE"
    if "$PY" scripts/train_eth_candidate_unified_phase2_exit_head_giveback_recal_20260817.py \
        --seed "$SEED" \
        --out-suffix "seed${SEED}" >>"$LOGFILE" 2>&1; then
        echo "=== [$(date -Iseconds)] OK phase2_exit_head_seed${SEED} ===" | tee -a "$LOGFILE"
    else
        rc=$?
        echo "=== [$(date -Iseconds)] FAILED phase2_exit_head_seed${SEED} (exit $rc) ===" | tee -a "$LOGFILE"
        FAILURES+=("seed${SEED}")
    fi
done

echo "########## SUMMARY $(date -Iseconds) ##########"
if [ "${#FAILURES[@]}" -eq 0 ]; then
    echo "ALL 5 SEEDS OK"
else
    echo "FAILURES (${#FAILURES[@]}): ${FAILURES[*]}"
fi
