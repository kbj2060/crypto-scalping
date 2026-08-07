# BTC CSALT DP Label Loop Final Report — 2026-07-15

Status: `development_fail_non_promotion_research`

## 결론

DP 기반 causal OOF 학습 라벨 생성·평가 루프는 구현되고 끝까지 실행됐지만, 사전 고정한
경제성 게이트를 통과한 라벨 후보는 없었다. 따라서 `research_pass=false`이며 T5/T6은 열지
않았다. BTC v1 live path, Omega artifact, risk sidecar는 변경하지 않았다.

총 1,320개 후보를 T1–T4에서 평가했고 후보·fold별 라벨 차트 5,280개를 저장했다.

| 실험 | 후보 | 차트 | 통과 | 핵심 결과 |
|---|---:|---:|---:|---|
| absolute DP advantage | 864 | 3,456 | 0 | 최대 합계 거래 1; q-regression이 CASH로 수축 |
| side-consensus advantage | 192 | 768 | 0 | 모든 후보 0거래 |
| balanced 7-class policy distillation | 144 | 576 | 0 | 가장 유망; 3/4 fold 양수 |
| two-stage meta-policy | 48 | 192 | 0 | 확률 calibration drift로 fold별 활동 붕괴 |
| causal rolling-rank gate | 24 | 96 | 0 | 거래는 증가했으나 비용 포함 성과 음수 |
| balanced 3-class side policy | 48 | 192 | 0 | 2/4 fold 양수, 비용 스트레스 음수 |

## 최선의 합법 후보

`btc_native_stationary__dp_policy__p60__m00__sv40__sg1`

- 구조: BTC-native 31 features, class-balanced 7-action DP policy distillation
- minimum side probability: 0.60
- 1.5x-cost teacher side gate: on
- T1: PnL +3.6817%, cost1.5 +2.3286%, 37 trades
- T2: PnL +6.7615%, cost1.5 +5.6715%, 18 trades
- T3: PnL -0.1400%, cost1.5 -0.1820%, 1 trade
- T4: PnL +3.8326%, cost1.5 +3.7020%, 3 trades
- aggregate: PnL +14.1358%, cost1.5 +11.5200%, 59 trades

T3가 음수이므로 “T1–T4 각각 양수” 개발 게이트를 통과하지 못했다. T3의 한 거래를 사후에
제거하거나 probability threshold를 미세 조정하면 leakage/selection bias가 되므로 하지 않았다.

## 산출물

- 첫 DP-advantage 결과: `tmp/causal_regen_20260516/btc_csalt_stage01_20260715/dp_advantage_dev`
- side-consensus 결과: `tmp/causal_regen_20260516/btc_csalt_stage01_20260715/side_consensus_dev`
- policy-distillation 결과: `tmp/causal_regen_20260516/btc_csalt_stage01_20260715/policy_distillation_dev`
- meta-policy 결과: `tmp/causal_regen_20260516/btc_csalt_stage01_20260715/meta_policy_dev`
- causal-rank 결과: `tmp/causal_regen_20260516/btc_csalt_stage01_20260715/causal_rank_dev`
- side-policy 결과: `tmp/causal_regen_20260516/btc_csalt_stage01_20260715/side_policy_dev`

각 디렉터리는 `report.json`, `development_candidate_table.csv`, 후보별 `label_charts/`를 포함한다.
후반 다섯 실험은 `candidate_fold_metrics.csv`도 포함한다. 7-class teacher 원시 확률은
`policy_distillation_dev/raw_oof_predictions/`에 보존했다.

## 무결성 판정

모든 최종 report는 다음을 기록한다.

```text
fresh_forward_bar_by_bar=true
trade_ledgers_used_as_input=false
saved_parent_exit_timestamps_used=false
future_rows_used_for_entry=false
label_fold_realized_outcomes_used_to_change_labels=false
holdout_T5_T6_opened=false
```

저장 ledger나 parent exit timestamp를 입력으로 사용하지 않았다. Label fold realized path는
사전등록 후보의 개발 평가에만 사용했고 해당 row의 label 생성에는 사용하지 않았다.

## 판단

현재 증거에서는 reward-shaped DP value regression보다 class-balanced DP policy distillation이
분명히 낫다. 그러나 이 성능은 아직 안정적인 학습 라벨이라고 부를 수 없다. 다음 연구는 같은
T1–T4 threshold 미세조정이 아니라 새 시간 구간을 확보한 뒤 7-class 최선 구조를 단일 사전등록
후보로 검증하는 것이 타당하다. 현재 결과를 BTC v1 live 후보로 승격해서는 안 된다.

