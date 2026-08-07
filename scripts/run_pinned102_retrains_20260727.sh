#!/usr/bin/env bash
# RESEARCH ONLY -- launches the remaining pinned-102 parent retrains in parallel.
# Kept as a file (not an inline shell string) because the inline version lost its variable
# expansion through the wsl/PowerShell quoting layers and wrote its logs to /.
set -u

cd /home/llewyn/crypto-scalping

S=scripts/train_eval_omega4_3head_parent72_pinned102_20260727.py
L=tmp/research_20260727/pinned102
DIRLAB=tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531
QLAB=tmp/causal_regen_20260516/omega_zigzag_fix_all_solutions_20260630/label_contracts/sltp_h48_conservative_padded_to_zigzag_timestamps

mkdir -p "$L"

# h48qual and zig075 differ only in quality target: h48qual uses the separate sltp_h48 quality
# label contract at q=0.50, zig075 reuses its direction labels at q=0.75. Same as the live pair.
h48() {  # $1 = out-suffix tail, rest = extra exit-label args
  local tag="$1"; shift
  python "$S" --pin-component h48qual --out-suffix "pinned102_20260727_h48qual_${tag}" \
    --epochs 2 --max-train-rows 0 --max-exit-samples 30000 --quality-thresholds 0.50 \
    --exit-label-mode entry_label_terminal_giveback \
    --direction-label-dir "$DIRLAB" --quality-mode quality_label_action --quality-label-dir "$QLAB" \
    "$@" > "$L/h48qual_${tag}.log" 2>&1
}

zig() {
  local tag="$1"; shift
  python "$S" --pin-component zig075 --out-suffix "pinned102_20260727_zig075_${tag}" \
    --epochs 2 --max-train-rows 0 --max-exit-samples 30000 --quality-thresholds 0.75 \
    --exit-label-mode entry_label_terminal_giveback \
    --direction-label-dir "$DIRLAB" --quality-mode same_as_direction \
    "$@" > "$L/zig075_${tag}.log" 2>&1
}

zig control &
h48 gb045 --exit-giveback-min 0.45 &
zig gb045 --exit-giveback-min 0.45 &
h48 tw08 --exit-terminal-window 8 &
zig tw08 --exit-terminal-window 8 &
wait
echo ALL_DONE
