#!/usr/bin/env bash
# RESEARCH ONLY -- 3 more seeds (total n=5 with the existing 260620/260728) for the
# exit_loss_weight=0 ablation, to get a statistically informative read on "does removing the
# exit head from training change performance" instead of the inconclusive n=2 from round 20.
set -u

cd /home/llewyn/crypto-scalping

S=scripts/train_eval_omega4_3head_parent72_pinned102_2024tape_noexithead_20260727.py
L=tmp/research_20260727/pinned102_2024tape_noexithead_moreseeds
DIRLAB=tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531
QLAB=tmp/causal_regen_20260516/omega_zigzag_fix_all_solutions_20260630/label_contracts/sltp_h48_conservative_padded_to_zigzag_timestamps

mkdir -p "$L"

for SEED in 260729 260730 260731; do
  python "$S" --pin-component h48qual --out-suffix "pinned102_2024tape_20260727_h48qual_noexithead_seed_${SEED}" \
    --epochs 2 --max-train-rows 0 --max-exit-samples 30000 --quality-thresholds 0.50 \
    --exit-label-mode entry_label_terminal_giveback --seed "$SEED" \
    --direction-label-dir "$DIRLAB" --quality-mode quality_label_action --quality-label-dir "$QLAB" \
    > "$L/h48qual_seed_${SEED}.log" 2>&1 &

  python "$S" --pin-component zig075 --out-suffix "pinned102_2024tape_20260727_zig075_noexithead_seed_${SEED}" \
    --epochs 2 --max-train-rows 0 --max-exit-samples 30000 --quality-thresholds 0.75 \
    --exit-label-mode entry_label_terminal_giveback --seed "$SEED" \
    --direction-label-dir "$DIRLAB" --quality-mode same_as_direction \
    > "$L/zig075_seed_${SEED}.log" 2>&1 &
done

wait
echo ALL_DONE
