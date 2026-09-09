#!/usr/bin/env bash
# Phase 3-a — Phase 2 의 20개 부모 번들 각각에 리스크 사이드카를 적합한다.
# 하위 프로젝트: omega461_regimegbm_rebuild_20260909
#
# 왜 필요한가: Phase 2 의 PnL/MDD 는 사이드카가 없어 BASE_TEMPLATE 고정 사이징
# (notional 0.45 / leverage 2.0) 으로 계산됐다. 라이브는 사이드카가 봉마다
# margin/leverage 를 조절하므로 그 숫자는 "선별 실력"이지 라이브 성과가 아니다.
# 사이드카는 부모 예측을 입력으로 적합되므로 부모가 학습된 뒤에만 만들 수 있다.
#
# 설정은 배포본(omega4_2_trade_risk_sidecar_20260622_plus_t12_livepass_*_precomputed_20260630)
# 의 pkl 에서 그대로 복원했다:
#   model_kind=hgb · risk_feature_mode=parent_outputs · side_split_model · dynamic_leverage
#   selection_objective=log_risk · selection_scope=validation_only · risk_target_mode=net
#   exit_sizing_input_mode=actual · exit_threshold_floor .55 / cap .95
#   require_dynamic_leverage_mapping · live_exposure_grid
#   min/max_validation_avg_notional = 0.45 / 0.95
#   log_risk_params: tail_budget .02 / tail_penalty 0.5 / liq_buffer .12 / liq_penalty .25
#   max_validation_mdd_abs = 25.0   ← 기본값 8.0 으로 돌리면 "no eligible risk mapping" 으로
#                                     실패한다(2026-09-09 스모크에서 실제 발생). 배포본 실측값이
#                                     selection_rule 문자열 "validation_mdd >= -25.00" 이다.
#   ATR 배리어는 라이브 공식(192/12/6/.075/.040/.22/.12)
# quality/exit threshold 는 컴포넌트별 라이브 값(zig075 .75 / h48qual .50, exit .95).
set -uo pipefail
cd "$(dirname "$0")/../.."

SEEDS=(615372041 208844917 933105268 471926350 862017594)
LOGDIR=tmp/omega461_regimegbm_rebuild_20260909/phase3_logs
mkdir -p "$LOGDIR"
PSTEM=tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_regimespine

for comp in zig075 h48qual; do
  if [ "$comp" = "zig075" ]; then QTAG=q075; QTHR=0.75; else QTAG=q050; QTHR=0.50; fi
  for arm in balnobb wide24; do
    for seed in "${SEEDS[@]}"; do
      PDIR="${PSTEM}_${arm}_${comp}_s${seed}_20260909"
      SUF="regimespine_${arm}_${comp}_s${seed}_20260909"
      LOG="$LOGDIR/sidecar_${arm}_${comp}_${seed}.log"
      if [ ! -f "$PDIR/${QTAG}" ] && [ ! -f "$PDIR/oos_predictions_${QTAG}.csv" ]; then
        echo "[$(date +%H:%M:%S)] SKIP (부모 예측 없음) $PDIR"; continue
      fi
      echo "[$(date +%H:%M:%S)] START sidecar arm=$arm comp=$comp seed=$seed"
      timeout 5400 python -u scripts/train_eval_omega4_2_risk_sidecar_20260622.py \
        --baseline-bundle "$PDIR/true_3head_tabm_bundle.pt" \
        --precomputed-prediction-dir "$PDIR" \
        --precomputed-prediction-tag "$QTAG" \
        --quality-threshold "$QTHR" --exit-threshold 0.95 \
        --atr-window 192 --tp-mult 12.0 --sl-mult 6.0 \
        --min-tp 0.075 --min-sl 0.040 --max-tp 0.22 --max-sl 0.12 \
        --model-kind hgb --risk-feature-mode parent_outputs \
        --side-split-model --dynamic-leverage --require-dynamic-leverage-mapping \
        --live-exposure-grid \
        --min-validation-avg-notional 0.45 --max-validation-avg-notional 0.95 \
        --selection-objective log_risk --selection-scope validation_only \
        --max-validation-mdd-abs 25.0 \
        --log-tail-budget 0.02 --log-tail-penalty 0.5 \
        --log-liquidation-buffer 0.12 --log-liquidation-penalty 0.25 \
        --risk-target-mode net --exit-sizing-input-mode actual \
        --exit-threshold-floor 0.55 --exit-threshold-cap 0.95 \
        --out-suffix "$SUF" > "$LOG" 2>&1
      echo "[$(date +%H:%M:%S)] DONE  sidecar arm=$arm comp=$comp seed=$seed rc=$?  -> $LOG"
    done
  done
done
echo "=== PHASE3-A ALL DONE ==="
