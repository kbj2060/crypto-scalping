# BTC CSALT DP-Advantage Search Protocol — 2026-07-15

Status: `predeclared_development_protocol_not_promotion_artifact`

## 목적과 성공 조건

T1 teacher smoke에서 absolute q10 DP value가 CASH로 수축한 실패를 고치기 위해,
teacher target을 `Q(action) - Q(CASH)`로 바꾼다. 후보 선택에는 T1–T4만 사용하고,
선택된 후보 하나를 고정한 뒤 T5와 T6을 정확히 한 번 평가한다.

최종 research pass 조건은 모두 만족해야 한다.

- T5 net PnL > 0
- T6 net PnL > 0
- T5+T6 closed trades >= 20
- T5+T6 aggregate 1.5x-cost net PnL > 0
- teacher prediction은 purged OOF이며 label-fold realized outcome을 label 변경에 사용하지 않음
- trade ledger, saved parent exit timestamp, future row를 entry input으로 사용하지 않음

T5/T6 결과를 본 뒤 threshold, feature, seed 또는 model hyperparameter를 바꾸지 않는다.

## 고정 후보군

### Target

1. `immediate_advantage`: lifecycle reward(action) - reward(CASH)
2. `dp_advantage`: finite SMDP Q(action) - Q(CASH)
3. `stress15_dp_advantage`: 1.5x cost finite SMDP advantage

CASH target은 항상 0이다. LONG/SHORT action만 회귀한다.

### Feature set

1. `derived11`: 기존 causal baseline의 자체 계산 11개 feature
2. `btc_native_stationary`: 아래 고정 column만 사용

```text
log_return, volatility_z, rsi, macd_hist, bb_width_z, hma_slope,
wick_ratio, garman_klass_vol, realized_vol_ratio, mtf_trend_1h,
mtf_trend_4h, chop_index, funding_z_score, long_squeeze_risk,
short_squeeze_risk, hurst_48, hurst_288, regime_trending,
ofi_acceleration, kalman_velocity, realized_skewness, funding_pressure,
cvd_slope_12, cvd_slope_48, bb_width_pct_rank_288,
atr_pct_rank_288, compression_score, vwap_dist_96,
distance_to_day_high_low_pct, crowding_pressure, execution_quality
```

모든 column은 entry 결정 시점 이전 completed bar 값만 사용한다. 각 fold의 label event에서
finite가 아니면 조용히 보정하지 않고 실패한다.

### Model and decision grid

- seeds: `310713, 310719, 310727, 310733, 310741`
- HGB quantile regressors: depth 3, iterations 100, leaf 40, L2 1.0, lr 0.05
- quantile: `0.10, 0.25, 0.50`
- ensemble score: `median - uncertainty_penalty * std`
- uncertainty penalty: `0.0, 0.5, 1.0`
- minimum predicted advantage: `0.0000, 0.00025, 0.00050, 0.00100`
- minimum seed vote: `0.60, 0.80`
- optional cost gate: none, or selected action의 1.5x-cost score > 0

각 조합의 label chart를 저장한다. 계산량을 줄이는 경우 model prediction은 재사용할 수 있지만,
후보 grid 자체를 결과에 따라 추가하거나 제거하지 않는다.

## 개발 선택 규칙

T1–T4에서 다음 lexicographic rule로 후보 하나를 선택한다.

1. 네 fold 모두 net PnL > 0인 후보 우선
2. 네 fold 합계 1.5x-cost net PnL > 0인 후보 우선
3. 합계 trades >= 40인 후보 우선
4. 그 안에서 최저 fold PnL 최대화
5. 동률이면 aggregate 1.5x-cost PnL 최대화
6. 다시 동률이면 feature 수, gate 수, threshold가 작은 후보

1–3을 만족하는 후보가 없으면 T5/T6을 열지 않고 `development_fail`로 종료한다.

## 산출물 계약

- 매 후보/매 fold label chart
- OOF label pack: event timestamp, entry-available timestamp, action id/name,
  predicted advantage, uncertainty, seed vote, sample weight
- dev candidate table과 frozen candidate manifest
- T5/T6 one-shot report 또는 명시적 `development_fail`
- 다음 provenance flag를 report에 기록

```text
fresh_forward_bar_by_bar=true
trade_ledgers_used_as_input=false
saved_parent_exit_timestamps_used=false
future_rows_used_for_entry=false
label_fold_realized_outcomes_used_to_change_labels=false
```

