# Yush 잔여 후보 #1 — 평균 체결 규모 z-score 채택 전 사전점검 (2026-08-15)

## 배경

[[yush_orderflow_absorption_closed_20260815]]가 "부분적으로 취할 가치가 있는 것"으로 남겨둔 두 후보 중
첫 번째: `avg_print_z = zscore(quote_volume / trades)`, 5분봉 대용 "대량 체결" 신호(lift 1.36배 바닥/
1.19배 천장, 마스터 순위 하위권). 그 문서는 "채택 전 `spearmanr(avg_print_z, volume_z)` 및 가격 추세
오염 사전 점검이 필수"라고 명시하고 실행하지 않았다 — 이 문서가 그 사전점검을 실행한다.

## 방법

`scripts/analyze_eth_yush_orderflow_component_evidence_20260815.py`의 `load_frame`/`add_flow_features`
(avg_print_z·vol_z 계산 로직, 288bar 롤링 z-score, 미수정)를 그대로 import해서 재사용. 신규 계산
로직 없음 — 이미 계산된 두 컬럼과 종가/수익률 사이 상관관계만 측정.
[[feedback_raw_feature_price_trend_contamination]] 절차 그대로: `spearmanr(new_feature, price)` +
추세 오염 체크. 산출물: `scripts/diagnose_eth_yush_avg_print_z_contamination_20260815.py`.

데이터: `data/eth_5m_1year.csv` 전체(롤링 워밍업 이후 사용 가능 224,205행).

## 결과

| 점검 | spearman ρ | p-value | 판정 기준 |
|---|---:|---:|---|
| `avg_print_z` vs `vol_z`(중복성) | **+0.7396** | ≈0 | 0.6 초과 시 중복 판정 → **초과** |
| `avg_print_z` vs `close`(가격 레벨) | +0.0013 | 0.55 | 0.5~0.6 초과 시 오염 |
| `avg_print_z` vs 후행 1시간 수익률 | -0.0057 | 0.0071 | 〃 |
| `avg_print_z` vs 선행 1시간 수익률 | -0.0076 | 0.00035 | 〃 |
| `avg_print_z` vs \|선행 1시간 수익률\| | +0.0501 | ≈0 | 〃 |

추가로 `docs/experiments/eth_omega461_live_102feature_redundancy_audit_20260815.md`(같은 날 별도
실험)가 라이브 102-feature 집합에 이미 `quote_volume, taker_buy_base, taker_buy_quote, trades,
volume`로 구성된 상관 클러스터가 존재함을 확인해 뒀다 — `avg_print_z`의 두 원료(quote_volume,
trades)가 이미 개별 라이브 피처로 존재한다.

## 결론 — 폐기(CLOSED)

**가격 추세 오염은 없다**(모든 가격/추세 상관 |ρ|≤0.05, 문턱 0.5~0.6에 한참 못 미침) — 이 신호
자체는 drift-as-skill류 함정에 걸려 있지 않다. 그러나 **중복성 사전점검을 탈락**했다:
`avg_print_z`와 순수 거래량 z-score의 순위상관이 0.74로, [[yush_orderflow_absorption_closed_20260815]]
문서가 미리 예상했던 정확히 그 문제("거래량과 상관이 높을 것")가 실측으로 확인됐다. 원료(quote_volume/
trades)가 라이브 피처에 이미 존재한다는 점까지 더하면, 이 컬럼은 "라이브 102-feature에 없는 진짜
신규 정보"라는 애초의 채택 근거를 충족하지 못한다. 애초 lift(1.36배/1.19배)도 22개 신호 마스터
랭킹의 하위권이었던 점을 고려하면, 이 중복성 문제까지 겹친 candidate를 추가로 채택할 근거가 없다.

**채택하지 않음.** 재작업 불필요 — 이 축은 여기서 종결.

## 산출물

- `scripts/diagnose_eth_yush_avg_print_z_contamination_20260815.py`

## 준수 확인

신규 학습·신규 replay 없음. 이미 검증된 라벨/피처 계산 함수를 import로만 재사용, 수정 없음.
`trade_ledgers_used_as_input=false`(거래 결과와 무관한 순수 피처간 상관관계 진단).
