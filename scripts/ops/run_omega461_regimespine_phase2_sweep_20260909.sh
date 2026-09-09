#!/usr/bin/env bash
# Phase 2 스윕 — 레짐 척추 balnobb vs wide24 대조, 2컴포넌트 × 5시드 × 2arm = 20회.
# 하위 프로젝트: omega461_regimegbm_rebuild_20260909
# 계약: docs/model_contracts/omega461_regimegbm_rebuild_contract.md
#
# arm A (후보) : train_eval_omega4_3head_parent72_regimespine_balnobb_20260909.py
#                base_cols 102개 중 레짐 6개를 regime3_balnobb_cut2509_* 로 치환
# arm B (대조) : train_eval_omega4_3head_parent72_pinned102_20260727.py
#                동일 트레이너·동일 시드·동일 인자, 레짐만 전신 wide24 유지
# 두 arm 의 차이는 레짐 척추 하나뿐이므로 짝지은 비교가 된다.
#
# 시드는 Seed-Diversity Gate(N>=5, 진짜 무작위, 고정간격 금지)를 충족한다.
set -uo pipefail
cd "$(dirname "$0")/../.."

SEEDS=(615372041 208844917 933105268 471926350 862017594)
LBL_ROOT=tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts
DIR_LBL="$LBL_ROOT/zigzag_action_labels_20260531"
QUAL_LBL=tmp/causal_regen_20260516/omega_zigzag_fix_all_solutions_20260630/label_contracts/sltp_h48_conservative_padded_to_zigzag_timestamps
LOGDIR=tmp/omega461_regimegbm_rebuild_20260909/phase2_logs
mkdir -p "$LOGDIR"

run_one () {  # $1=arm $2=script $3=component $4=seed
  local arm=$1 script=$2 comp=$3 seed=$4
  local suffix="regimespine_${arm}_${comp}_s${seed}_20260909"
  local log="$LOGDIR/${arm}_${comp}_${seed}.log"
  local -a extra
  if [ "$comp" = "zig075" ]; then
    extra=(--quality-mode same_as_direction --quality-thresholds 0.75)
  else
    extra=(--quality-mode quality_label_action --quality-label-dir "$QUAL_LBL" --quality-thresholds 0.50)
  fi
  echo "[$(date +%H:%M:%S)] START arm=$arm comp=$comp seed=$seed"
  timeout 3600 python -u "scripts/$script" \
    --pin-component "$comp" --epochs 2 \
    --direction-label-dir "$DIR_LBL" "${extra[@]}" \
    --max-exit-samples 30000 --max-train-rows 0 \
    --exit-label-mode entry_label_terminal_giveback \
    --out-suffix "$suffix" --device cpu --seed "$seed" > "$log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M:%S)] DONE  arm=$arm comp=$comp seed=$seed rc=$rc  -> $log"
  return 0
}

echo "=== Phase 2 sweep: 2 comps x 5 seeds x 2 arms = 20 runs ==="
echo "seeds: ${SEEDS[*]}"
for comp in zig075 h48qual; do
  for seed in "${SEEDS[@]}"; do
    run_one balnobb train_eval_omega4_3head_parent72_regimespine_balnobb_20260909.py "$comp" "$seed"
    run_one wide24  train_eval_omega4_3head_parent72_pinned102_20260727.py            "$comp" "$seed"
  done
done
echo "=== ALL DONE ==="
