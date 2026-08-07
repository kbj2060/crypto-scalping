#!/usr/bin/env bash
# RESEARCH ONLY -- reproduction check for the noexithead run: same config (2024+2025 tape,
# exit_loss_weight=0, live 102-col contract) but --seed 260728 instead of the trainer's default
# 260620, to test whether zig075's OOS win over live (+65.12%/-8.42% vs +17.89%/-11.01%) is a
# real effect or a single lucky training draw (round 19 already showed large retraining variance
# even with exit_loss_weight unchanged).
set -u

cd /home/llewyn/crypto-scalping

S=scripts/train_eval_omega4_3head_parent72_pinned102_2024tape_noexithead_20260727.py
L=tmp/research_20260727/pinned102_2024tape_noexithead_seed2
DIRLAB=tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531
QLAB=tmp/causal_regen_20260516/omega_zigzag_fix_all_solutions_20260630/label_contracts/sltp_h48_conservative_padded_to_zigzag_timestamps
SEED=260728

mkdir -p "$L"

python "$S" --pin-component h48qual --out-suffix pinned102_2024tape_20260727_h48qual_noexithead_seed2 \
  --epochs 2 --max-train-rows 0 --max-exit-samples 30000 --quality-thresholds 0.50 \
  --exit-label-mode entry_label_terminal_giveback --seed "$SEED" \
  --direction-label-dir "$DIRLAB" --quality-mode quality_label_action --quality-label-dir "$QLAB" \
  > "$L/h48qual_noexithead_seed2.log" 2>&1 &

python "$S" --pin-component zig075 --out-suffix pinned102_2024tape_20260727_zig075_noexithead_seed2 \
  --epochs 2 --max-train-rows 0 --max-exit-samples 30000 --quality-thresholds 0.75 \
  --exit-label-mode entry_label_terminal_giveback --seed "$SEED" \
  --direction-label-dir "$DIRLAB" --quality-mode same_as_direction \
  > "$L/zig075_noexithead_seed2.log" 2>&1 &

wait
echo ALL_DONE
