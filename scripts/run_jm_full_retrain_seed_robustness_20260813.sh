#!/bin/bash
# Driver for the ETH Omega4.6.1 JM-regime full-retrain N=5-seed robustness study (2026-08-13).
#
# Runs h48qual + zig075 parent retrains (pinned102, regime3=JM lambda=4) and their matching
# correctgate risk sidecars for 5 genuinely random seeds, STRICTLY SEQUENTIALLY (one job at a
# time), with a free-h memory check after every single job. This server also runs the live
# trading bot + BTC shadow loops + JM shadow bot continuously -- a prior job on this same box
# (exit-head retraining) caused a full server outage via memory exhaustion, so this script must
# never launch more than one training process at once and must abort rather than push through if
# available memory looks critically low.
#
# Usage: bash scripts/run_jm_full_retrain_seed_robustness_20260813.sh
set -uo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
SEEDS=(323033734 50011403 504028524 782182142 393423992)
LOG_DIR="tmp/jm_full_retrain_seed_robustness_20260813"
mkdir -p "$LOG_DIR"
MEM_FLOOR_GB=4
FAILURES=()

mem_check() {
    local label="$1"
    echo "[MEM][$label] $(date -Iseconds)"
    free -h
    local avail_gb
    avail_gb=$(free -g | awk '/^Mem:/{print $7}')
    echo "[MEM][$label] available=${avail_gb}GB floor=${MEM_FLOOR_GB}GB"
    if [ "$avail_gb" -lt "$MEM_FLOOR_GB" ]; then
        echo "[MEM][CRITICAL] available memory ${avail_gb}GB < ${MEM_FLOOR_GB}GB floor -- STOPPING sequence to protect the shared live-trading server"
        exit 90
    fi
    sleep 5
}

run_step() {
    # run_step <label> <logfile> <cmd...>
    local label="$1" logfile="$2"
    shift 2
    echo "=== [$(date -Iseconds)] START $label ===" | tee "$LOG_DIR/$logfile"
    if "$@" >>"$LOG_DIR/$logfile" 2>&1; then
        echo "=== [$(date -Iseconds)] OK $label ===" | tee -a "$LOG_DIR/$logfile"
        return 0
    else
        local rc=$?
        echo "=== [$(date -Iseconds)] FAILED $label (exit $rc) ===" | tee -a "$LOG_DIR/$logfile"
        FAILURES+=("$label")
        return "$rc"
    fi
}

echo "########## STEP 0: regenerate JM regime3 CSVs (CPU, cheap) ##########"
run_step "regime3_jmlam4_build" "step0_regime3_build.log" \
    python3 scripts/build_eth_regime3_jm_lam4_20260809.py
run_step "regime3_jmlam4_cleansource_2026" "step0_regime3_cleansource.log" \
    python3 scripts/regenerate_eth_jmlam4_regime3_2026_cleansource_20260809.py
mem_check "after_step0"

for SEED in "${SEEDS[@]}"; do
    echo "########## SEED $SEED ##########"

    H48_DIR="tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_pinned102_regime_jmlam4_20260809_h48qual_ext_seed${SEED}"
    run_step "h48qual_parent_seed${SEED}" "h48qual_parent_seed${SEED}.log" \
        python3 scripts/train_eval_omega4_3head_parent72_pinned102_h48qual_regime_jmlam4_20260809.py \
        --seed "$SEED" --out-suffix "pinned102_regime_jmlam4_20260809_h48qual_ext_seed${SEED}"
    mem_check "after_h48qual_parent_seed${SEED}"

    if [ -f "$H48_DIR/true_3head_tabm_bundle.pt" ]; then
        run_step "h48qual_sidecar_seed${SEED}" "h48qual_sidecar_seed${SEED}.log" \
            python3 scripts/train_eval_omega4_2_risk_sidecar_eth_regime_jmlam4_20260809.py \
            --baseline-bundle "${H48_DIR}/true_3head_tabm_bundle.pt" \
            --precomputed-prediction-dir "${H48_DIR}" \
            --precomputed-prediction-tag q070 --quality-threshold 0.70 \
            --min-validation-avg-notional 0.45 --max-validation-avg-notional 0.95 \
            --max-validation-mdd-abs 25 \
            --out-suffix "pinned102_jmlam4_q070_correctgate_seed${SEED}_20260813" \
            --device cpu
    else
        echo "[SKIP] h48qual_sidecar_seed${SEED}: parent bundle missing, parent training must have failed"
        FAILURES+=("h48qual_sidecar_seed${SEED}_SKIPPED_NO_PARENT")
    fi
    mem_check "after_h48qual_sidecar_seed${SEED}"

    ZIG_DIR="tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_pinned102_regime_jmlam4_20260809_zig075_seed${SEED}"
    run_step "zig075_parent_seed${SEED}" "zig075_parent_seed${SEED}.log" \
        python3 scripts/train_eval_omega4_3head_parent72_pinned102_zig075_regime_jmlam4_20260809.py \
        --seed "$SEED" --out-suffix "pinned102_regime_jmlam4_20260809_zig075_seed${SEED}"
    mem_check "after_zig075_parent_seed${SEED}"

    if [ -f "$ZIG_DIR/true_3head_tabm_bundle.pt" ]; then
        run_step "zig075_sidecar_seed${SEED}" "zig075_sidecar_seed${SEED}.log" \
            python3 scripts/train_eval_omega4_2_risk_sidecar_eth_zig075_regime_jmlam4_20260809.py \
            --baseline-bundle "${ZIG_DIR}/true_3head_tabm_bundle.pt" \
            --precomputed-prediction-dir "${ZIG_DIR}" \
            --precomputed-prediction-tag q080 --quality-threshold 0.80 \
            --min-validation-avg-notional 0.45 --max-validation-avg-notional 0.95 \
            --max-validation-mdd-abs 25 \
            --out-suffix "pinned102_jmlam4_q080_correctgate_seed${SEED}_20260813" \
            --device cpu
    else
        echo "[SKIP] zig075_sidecar_seed${SEED}: parent bundle missing, parent training must have failed"
        FAILURES+=("zig075_sidecar_seed${SEED}_SKIPPED_NO_PARENT")
    fi
    mem_check "after_zig075_sidecar_seed${SEED}"
done

echo "########## SUMMARY ##########"
if [ "${#FAILURES[@]}" -eq 0 ]; then
    echo "ALL 20 JOBS OK (5 seeds x {h48qual,zig075} x {parent,sidecar})"
else
    echo "FAILURES (${#FAILURES[@]}): ${FAILURES[*]}"
fi
echo "DONE $(date -Iseconds)"
