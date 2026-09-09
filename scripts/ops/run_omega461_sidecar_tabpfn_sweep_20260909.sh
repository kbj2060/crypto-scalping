#!/usr/bin/env bash
# TabPFN 리스크 사이드카 스윕 — Phase 3-a 에서 HGB 사이드카가 성공한 부모 14개에 대해
# TabPFN 사이드카를 mean / q25 두 arm 으로 적합한다(=28회).
#
# 짝지은 비교: 같은 부모·같은 라벨·같은 매핑 격자·같은 선택 제약, **회귀기만** 다르다.
# 레짐 arm(balnobb/wide24)은 여기서 부모 변이의 원천일 뿐이라 짝이 5개가 아니라 14개로 늘어난다.
#
# arm 의미:
#   mean — 드롭인 교체(예측 평균). HGB 와 같은 의미의 점추정.
#   q25  — TabPFN 예측 **분포의 하위 25% 분위**. 사이드카는 리스크 사이징 모듈이므로
#          하방 인식을 모델 자체에서 얻는다. 의미가 달라지는 변경이라 별도 arm.
set -uo pipefail
cd "$(dirname "$0")/../.."

SEEDS=(615372041 208844917 933105268 471926350 862017594)
LOGDIR=tmp/omega461_regimegbm_rebuild_20260909/tabpfn_sidecar_logs
mkdir -p "$LOGDIR"
PSTEM=tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_regimespine
HSTEM=tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_regimespine

for OUTT in mean q25; do
  for comp in zig075 h48qual; do
    if [ "$comp" = "zig075" ]; then QTAG=q075; QTHR=0.75; else QTAG=q050; QTHR=0.50; fi
    for arm in balnobb wide24; do
      for seed in "${SEEDS[@]}"; do
        PDIR="${PSTEM}_${arm}_${comp}_s${seed}_20260909"
        # HGB 사이드카가 성공한 부모만 짝이 성립한다 (6개는 선택 격자에서 실패했다)
        [ -f "${HSTEM}_${arm}_${comp}_s${seed}_20260909/report.json" ] || { echo "[$(date +%H:%M:%S)] SKIP(HGB짝없음) $arm/$comp/$seed"; continue; }
        SUF="tabpfnA_${OUTT}_${arm}_${comp}_s${seed}_20260909"
        LOG="$LOGDIR/${OUTT}_${arm}_${comp}_${seed}.log"
        [ -f "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_${SUF}/report.json" ] && { echo "[$(date +%H:%M:%S)] SKIP(이미완료) $SUF"; continue; }
        echo "[$(date +%H:%M:%S)] START out=$OUTT arm=$arm comp=$comp seed=$seed"
        timeout 7200 python -u scripts/train_eval_omega461_risk_sidecar_tabpfn_20260909.py \
          --baseline-bundle "$PDIR/true_3head_tabm_bundle.pt" \
          --precomputed-prediction-dir "$PDIR" --precomputed-prediction-tag "$QTAG" \
          --quality-threshold "$QTHR" --exit-threshold 0.95 \
          --atr-window 192 --tp-mult 12.0 --sl-mult 6.0 \
          --min-tp 0.075 --min-sl 0.040 --max-tp 0.22 --max-sl 0.12 \
          --model-kind tabpfn --tabpfn-n-estimators 32 --tabpfn-output "$OUTT" \
          --risk-feature-mode parent_outputs \
          --side-split-model --dynamic-leverage --require-dynamic-leverage-mapping \
          --live-exposure-grid --min-validation-avg-notional 0.45 --max-validation-avg-notional 0.95 \
          --selection-objective log_risk --selection-scope validation_only \
          --max-validation-mdd-abs 25.0 --log-tail-budget 0.02 --log-tail-penalty 0.5 \
          --log-liquidation-buffer 0.12 --log-liquidation-penalty 0.25 \
          --risk-target-mode net --exit-sizing-input-mode actual \
          --exit-threshold-floor 0.55 --exit-threshold-cap 0.95 \
          --out-suffix "$SUF" > "$LOG" 2>&1
        echo "[$(date +%H:%M:%S)] DONE  out=$OUTT arm=$arm comp=$comp seed=$seed rc=$? -> $LOG"
      done
    done
  done
done
echo "=== TABPFN SIDECAR SWEEP ALL DONE ==="
