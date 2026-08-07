#!/usr/bin/env bash
# RESEARCH ONLY -- retrains h48qual/zig075 x 3 barrier widths (dense/medium/sparse) x 5 seeds,
# using the NEW matched direction+quality triple-barrier label pair for EACH component (zig075
# now gets its OWN quality label instead of same_as_direction -- fixes the 2026-07-28 catastrophic
# failure diagnosed as "two heads changed simultaneously"). Runs one barrier-width config at a
# time (10 jobs/wave: 2 components x 5 seeds) to keep CPU contention reasonable on this 12-core
# machine, matching this session's established pattern.
set -u

cd /home/llewyn/crypto-scalping

S=scripts/train_eval_omega4_3head_parent72_pinned102_2024tape_20260727.py
LABEL_ROOT=tmp/causal_regen_20260516/eth_triple_barrier_grid_20260728/label_contracts
L=tmp/research_20260728/pinned102_2024tape_tripbar_grid
mkdir -p "$L"

CONFIG="${1:?usage: run_pinned102_2024tape_tripbar_grid_20260728.sh dense-or-medium-or-sparse}"
DIRLAB="$LABEL_ROOT/direction_${CONFIG}"
QLAB="$LABEL_ROOT/quality_${CONFIG}"

for SEED in 260620 260728 260729 260730 260731; do
  python "$S" --pin-component h48qual --out-suffix "pinned102_2024tape_tripbargrid_20260728_${CONFIG}_h48qual_seed_${SEED}" \
    --epochs 2 --max-train-rows 0 --max-exit-samples 30000 --quality-thresholds 0.50 \
    --exit-label-mode entry_label_terminal_giveback --seed "$SEED" \
    --direction-label-dir "$DIRLAB" --quality-mode quality_label_action --quality-label-dir "$QLAB" \
    > "$L/${CONFIG}_h48qual_seed_${SEED}.log" 2>&1 &

  python "$S" --pin-component zig075 --out-suffix "pinned102_2024tape_tripbargrid_20260728_${CONFIG}_zig075_seed_${SEED}" \
    --epochs 2 --max-train-rows 0 --max-exit-samples 30000 --quality-thresholds 0.75 \
    --exit-label-mode entry_label_terminal_giveback --seed "$SEED" \
    --direction-label-dir "$DIRLAB" --quality-mode quality_label_action --quality-label-dir "$QLAB" \
    > "$L/${CONFIG}_zig075_seed_${SEED}.log" 2>&1 &
done

wait
echo "ALL_DONE_${CONFIG}"
