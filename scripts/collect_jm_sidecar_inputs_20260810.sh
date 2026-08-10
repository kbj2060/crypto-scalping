#!/usr/bin/env bash
# Collect the exact inputs that scripts/tune_btc_sidecar_regime_jmredesign_20260810.py needs,
# so the sweep can be run somewhere other than the machine that produced them.
#
#   bash scripts/collect_jm_sidecar_inputs_20260810.sh stage1   # probe seed 903174 only
#   bash scripts/collect_jm_sidecar_inputs_20260810.sh stage2   # all five seeds
#
# Writes tmp/jm_sidecar_inputs_<stage>.tar.gz and prints the per-item sizes, so the transfer
# route (git -f / release asset / lfs) can be chosen against real numbers rather than guesses.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAGE="${1:-stage1}"
case "$STAGE" in
  stage1) SEEDS=(903174) ;;
  stage2) SEEDS=(260620 481003 26611 903174 155827) ;;
  *) echo "usage: $0 [stage1|stage2]" >&2; exit 2 ;;
esac

TAG="jmredesign_20260810"
SUP="data/ensemble/supervised"
MANIFEST="$(mktemp)"
trap 'rm -f "$MANIFEST"' EXIT

add() {
  if [ -e "$1" ]; then
    printf '%s\n' "$1" >> "$MANIFEST"
  else
    echo "MISSING  $1" >&2
    MISSING=1
  fi
}

MISSING=0

# Shared feature source and the redesigned-JM regime overlay.
add "data/splits/year_oos/btc_features_2025.csv"
add "data/splits/year_oos/btc_features_2026.csv"
add "$SUP/btc_regime3_current_hmm_${TAG}_2025_maskedname.csv"
add "$SUP/btc_regime3_current_hmm_${TAG}_2026_maskedname.csv"

# Direction labels (omega4.LABEL_DIR).
add "tmp/causal_regen_20260516/btc_zigzag_action_labels_20260708/zigzag_action_labels_2025.csv"
add "tmp/causal_regen_20260516/btc_zigzag_action_labels_20260708/zigzag_action_labels_2026.csv"

# Per-seed parent bundle + the exact-threshold q050 prediction artifacts.
for s in "${SEEDS[@]}"; do
  d="$(ls -d tmp/causal_regen_20260516/*"${TAG}"_e3_r30000_s"${s}" 2>/dev/null | tail -1 || true)"
  if [ -z "$d" ]; then
    echo "MISSING  parent dir for seed $s (glob *${TAG}_e3_r30000_s${s})" >&2
    MISSING=1
    continue
  fi
  add "$d/true_3head_tabm_bundle.pt"
  for split in train validation oos; do
    add "$d/${split}_predictions_q050.csv"
  done
done

if [ "$MISSING" -ne 0 ]; then
  echo
  echo "one or more inputs are missing -- archive not written" >&2
  exit 1
fi

echo
echo "=== per-item size ==="
xargs -r -a "$MANIFEST" du -h | sort -h
echo
echo "=== total (uncompressed) ==="
xargs -r -a "$MANIFEST" du -ch | tail -1

OUT="tmp/jm_sidecar_inputs_${STAGE}.tar.gz"
tar -czf "$OUT" -T "$MANIFEST"
echo
echo "=== archive ==="
du -h "$OUT"
