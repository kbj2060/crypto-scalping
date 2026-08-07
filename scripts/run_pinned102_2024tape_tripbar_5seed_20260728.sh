#!/usr/bin/env bash
# RESEARCH ONLY -- 5-seed retrain of h48qual/zig075 on the new triple-barrier direction label
# (tmp/causal_regen_20260516/eth_triple_barrier_maxdensity_20260728/label_contracts/
# triple_barrier_direction_maxdensity_20260728/), replacing zigzag_action as --direction-label-dir.
# Everything else matches round 19's control exactly: same 2024+2025 tape (pinned102_2024tape
# wrapper), same live 102-col feature contract, exit_loss_weight left at its default 1.15 (exit
# head present -- round 20/21 already closed the "remove exit head" question negatively, so this
# run isolates ONE change: the direction label).
#
# h48qual keeps its existing quality-label-dir (sltp_h48_conservative, a separate barrier-based
# quality target, unchanged) -- only its DIRECTION label is replaced.
# zig075 uses --quality-mode same_as_direction, so its quality label automatically follows the
# new triple-barrier direction label too.
set -u

cd /home/llewyn/crypto-scalping

S=scripts/train_eval_omega4_3head_parent72_pinned102_2024tape_20260727.py
L=tmp/research_20260728/pinned102_2024tape_tripbar
TRIPBAR_DIRLAB=tmp/causal_regen_20260516/eth_triple_barrier_maxdensity_20260728/label_contracts/triple_barrier_direction_maxdensity_20260728
QLAB=tmp/causal_regen_20260516/omega_zigzag_fix_all_solutions_20260630/label_contracts/sltp_h48_conservative_padded_to_zigzag_timestamps

mkdir -p "$L"

for SEED in 260620 260728 260729 260730 260731; do
  python "$S" --pin-component h48qual --out-suffix "pinned102_2024tape_tripbar_20260728_h48qual_seed_${SEED}" \
    --epochs 2 --max-train-rows 0 --max-exit-samples 30000 --quality-thresholds 0.50 \
    --exit-label-mode entry_label_terminal_giveback --seed "$SEED" \
    --direction-label-dir "$TRIPBAR_DIRLAB" --quality-mode quality_label_action --quality-label-dir "$QLAB" \
    > "$L/h48qual_seed_${SEED}.log" 2>&1 &

  python "$S" --pin-component zig075 --out-suffix "pinned102_2024tape_tripbar_20260728_zig075_seed_${SEED}" \
    --epochs 2 --max-train-rows 0 --max-exit-samples 30000 --quality-thresholds 0.75 \
    --exit-label-mode entry_label_terminal_giveback --seed "$SEED" \
    --direction-label-dir "$TRIPBAR_DIRLAB" --quality-mode same_as_direction \
    > "$L/zig075_seed_${SEED}.log" 2>&1 &
done

wait
echo ALL_DONE
