# Entry/Exit Price Model Plan

## Goal
현재 시스템에서 부족한 것은 `진입 가격(entry price)` 예측이다.

이미 있는 것:
- 방향: `trend_xgb`, `DSAC`
- 기대 움직임/가격 범위: `quantile_forest -> m7_q10/q50/q90/qwidth`
- 품질/보유시간: `multi_target_lgbm -> m7_quality_pred`, `m7_hold_pred`, `m7_target_hold`

부족한 것:
- 현재가 기준으로 어느 가격에 진입하는 것이 유리한지
- 진입 후 어떤 가격을 TP/SL로 써야 하는지의 명시적 가격화

따라서 1차 구현 목표는 아래 두 가지다.

1. `Entry Price Model`
2. `Exit Price Mapping`


## Current System Mapping

### Direction Head
- source: `trend_xgb`
- live fields:
  - `m7_trend_xgb_dn`
  - `m7_trend_xgb_fl`
  - `m7_trend_xgb_up`

### Move/Range Head
- source: `quantile_forest`
- live fields:
  - `m7_q10`
  - `m7_q50`
  - `m7_q90`
  - `m7_qwidth`

### Trade Quality Head
- source: `multi_target_lgbm`
- live fields:
  - `m7_quality_pred`
  - `m7_hold_pred`
  - `m7_target_hold`

### Risk Gate
- source: GMM / ISO / VAE
- live fields:
  - `m7_gmm_*`
  - `m7_iso_*`
  - `m7_vae_*`


## Proposed Models

## 1. Entry Price Model

### Purpose
현재 캔들 시점 `t`에서 다음 `H`봉 안에 유리하게 진입할 수 있는 가격 offset을 예측한다.

### Why offset instead of raw price
raw price는 절대 가격 레벨 영향을 크게 받는다.
실전에서는 현재가 대비 몇 bp 아래/위가 더 중요한 신호다.

따라서 타겟은 다음처럼 둔다.

- long entry target:
  - `entry_long_offset = (future_min_low_H / close_t) - 1`
- short entry target:
  - `entry_short_offset = (future_max_high_H / close_t) - 1`

여기서:
- long offset은 보통 음수
- short offset은 보통 양수

추천 horizon:
- `H = 3` (15분)
- 보조 실험: `H = 2`, `H = 4`

### Training targets
- `entry_long_q25`
- `entry_long_q50`
- `entry_short_q50`
- `entry_short_q75`

1차 구현은 단순화를 위해 중앙값만 시작한다.

- `entry_long_q50`
- `entry_short_q50`

### Model type
- 1차: `LightGBMRegressor` quantile mode
- 이유:
  - 현재 코드베이스와 잘 맞음
  - tabular feature에서 빠르게 실험 가능
  - `q10/q50/q90` 기반 기존 quantile pipeline과 일관성 있음

### Features
우선 아래 feature만 사용한다.

- 방향 head
  - `m7_trend_xgb_dn/fl/up`
- 가격 범위 head
  - `m7_q10/q50/q90/qwidth`
- 품질 head
  - `m7_quality_pred`
  - `m7_hold_pred`
- 미시구조 / 변동성
  - `garch_vol_z`
  - `jump_z`
  - `evt_excess_z`
  - `oi_change_rate`
  - `cvp_volume_imbalance`
  - `net_taker_ratio`
  - `smart_money_flow`
  - `current_spread` 또는 대체 spread proxy
- 시간/세션
  - `session_us`
  - `hour_cos`

### Inference output
- `entry_long_offset`
- `entry_short_offset`

실거래에서는:
- long 진입 후보가 나오면:
  - `entry_price = close * (1 + entry_long_offset)`
- short 진입 후보가 나오면:
  - `entry_price = close * (1 + entry_short_offset)`

### Execution policy
1차는 `추천 가격`만 출력한다.
주문 타입까지 자동화하지 않는다.

즉:
- 봇이 `entry_price_reco`를 로그/패널에 보여줌
- 실제 주문 방식은 기존대로 유지하거나 별도 단계에서 연결


## 2. Exit Price Mapping

### Purpose
새 모델을 바로 추가하지 않고, 기존 `m7_q10/q50/q90`를 청산 가격에 직접 맵핑한다.

### Long position
- `tp_offset_long = max(m7_q90, min_tp_floor)`
- `sl_offset_long = min(m7_q10, -min_sl_floor)`

### Short position
- `tp_offset_short = min(m7_q10, -min_tp_floor)`
- `sl_offset_short = max(m7_q90, min_sl_floor)`

추천 floor:
- `min_tp_floor = 0.0008`
- `min_sl_floor = 0.0006`

### Dynamic scaling
`m7_qwidth`와 anomaly 상태로 TP/SL 폭을 조정한다.

- high uncertainty:
  - TP 축소
  - SL 보수화
- anomaly:
  - 신규 진입은 축소
  - 보유 중이면 조기 청산 조건 강화

### Hold-based exit
기존 `m7_target_hold` 유지

- `hold_count >= m7_target_hold` 이고
- `m7_composite_score` 또는 `direction strength`가 약화되면 시간 청산


## Integration Plan

## Phase 1: Explicit price outputs without new live order logic

### Files
- `ensemble/supervised/train_entry_price_model.py` 신규
- `ensemble/supervised/live_supervised_hub.py` 확장
- `ensemble/seven_model_ensemble.py` 확장
- `scripts/augment_rl_training_with_model7.py` 자동 반영
- `trading_bot.py` 요약 패널 및 exit 로직 일부 반영

### New live fields
- `m7_entry_long_offset`
- `m7_entry_short_offset`
- `m7_entry_long_price`
- `m7_entry_short_price`
- `m7_tp_offset`
- `m7_sl_offset`
- `m7_tp_price`
- `m7_sl_price`

### Trading bot display
요약 패널에 아래 추가:
- 추천진입가
- 추천익절가
- 추천손절가


## Phase 2: Use entry model in execution

### Policy
- 방향 확정 후 entry model이 추천한 가격이 현재가와 너무 멀지 않으면 지정가
- 너무 멀면 시장가 또는 스킵

### Guard rails
- spread가 큰 경우 market 진입 금지
- entry offset이 최근 실현 range보다 비현실적이면 clip


## Target Generation Details

### Entry model target builder
입력 행 `t`에 대해:

- `future_low = min(low[t+1 : t+H])`
- `future_high = max(high[t+1 : t+H])`
- `entry_long_offset = future_low / close[t] - 1`
- `entry_short_offset = future_high / close[t] - 1`

학습 시 extreme outlier clip:
- long offset: `[-0.02, 0.0]`
- short offset: `[0.0, 0.02]`


## Validation

### Entry model metrics
- MAE on offset
- directional fill usefulness:
  - long 진입 추천가가 실제 future low와 얼마나 근접한가
  - short 진입 추천가가 실제 future high와 얼마나 근접한가
- realized improvement vs market entry:
  - `market_entry_return`
  - `model_entry_return`

핵심 비교:
- 같은 방향 신호에서 현재가 진입 대비 추천 진입가가 평균 몇 bp 개선되는가

### Exit mapping metrics
- current stop/take 규칙 대비
  - 평균 손익
  - MDD
  - avg hold
  - early stop rate


## What not to do
- DSAC를 entry price predictor로 직접 바꾸지 않음
- direction/entry/exit를 한 모델에 다시 합치지 않음
- 1차에 CatBoost fill probability를 같이 넣지 않음


## Recommended Next Step
1. `train_entry_price_model.py` 추가
2. `live_supervised_hub.py`에 entry model loader 추가
3. `seven_model_ensemble.py`에서 `m7_entry_*` 출력 생성
4. `trading_bot.py` 패널에 entry/tp/sl 가격 표시
5. 이후에만 실행 로직에 지정가 진입 연결
