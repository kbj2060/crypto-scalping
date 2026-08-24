# ETH Omega4.6.1 라이브 102 base feature 중복도 감사 (2026-08-15)

## 성격

순수 서술적(descriptive) 통계 감사. **재학습/backtest/promotion 근거 아님.** live 파일 변경
없음. Fresh-Forward causal walk-forward 규칙은 적용 대상이 아님(정적 통계 속성 점검이라 causal
순서가 필요 없음) — 2025년 전체(105,064개 5분봉, 결측 0)를 한 번에 로드해 pairwise 상관을 계산.

## 배경

live h48qual/zig075 3-Head TabM 번들이 소비하는 102 base feature가 중복이 많은지 확인해달라는
요청. 과거 FINAL12(mRMR/knockoff 기반 축소, `docs/experiments/eth_h48qual_final12_feature_selection_20260811.md`)
는 후보 풀(h48qual 145개/zig075 138개) 대상이었고 실제로 wire-in되지 않았다(Odyssey1 미해결
이슈 5-7 — production panel bridge mismatch, FINAL12 vs FINAL13 카운트 불일치, dedup 스크립트
자체가 세션 scratchpad에만 존재해 재현 불가). 즉 현재 live 102개는 dedup을 거치지 않은 원본
세트다. 이 감사는 그 102개 자체를 대상으로 한다.

기존 문서 중 이 정확한 대상(현재 live 102개 base_cols)에 대한 상관행렬/VIF/mRMR 감사는 없었음
(grep 결과 FINAL12/knockoff 계열 문서는 모두 더 넓은 후보 풀 대상).

## 대상 feature 목록 확인

`tmp/causal_regen_20260516/.../true_3head_tabm_bundle.pt`의 `bundle["base_cols"]`를 직접
로드해 확인(코드/`features/engineering.py` 추정이 아니라 실제 배포 아티팩트 값):

- h48qual 번들(`FINAL_GOVERNOR_OMEGA4_6_1_H48QUAL_BUNDLE_PATH`)과 zig075 번들
  (`FINAL_GOVERNOR_OMEGA4_6_1_ZIG075_BUNDLE_PATH`)의 `base_cols`는 **길이 102, 순서까지 완전히
  동일**(`h48_cols == zig_cols` → True). 즉 두 컴포넌트가 정확히 같은 102개 raw/engineered
  feature를 공유한다.

## 데이터

- 원천: `tmp/causal_regen_20260516/alpha5_a5dir_2024_train_2025_score_20260529_cleanfunding_stable48/
  02_fixed_regime4_state24_sticky090_tp18_sl10_preprocess_2024_to_2025/trade_candidates_2025_regime4_state24_sticky090_tp18_sl10_fixed.csv`
  (2025 전체, 257컬럼, 96/102 base_cols 포함) + `data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/
  training_features_2025_regime3_current_sensitive_hmm_wide24.csv`(나머지 6개
  `regime3_current_sensitive_wide24_*` 컬럼)를 timestamp inner-join.
- 병합 후 (105,064, 263) — 102개 base_cols 전부 존재, 결측 0. 재계산 없이 기존 산출물만 사용.

## 방법

- Spearman 상관(순위 기반, outlier/비선형에 덜 민감 — Pearson 아님을 명시) 102x102 pairwise.
- 근사-상수/저분산 체크: std, unique value count.
- 클러스터링: |corr| > 0.9 임계값에서 connected-components(그래프 인접 = edge, 그 안에서
  BFS로 컴포넌트 추출).

## 결과

- **|corr| > 0.9 쌍: 36개** (사실상 동일 정보).
- **|corr| > 0.7 쌍: 108개** (강한 중복).
- std < 1e-8인 "진짜 상수" 컬럼은 **0개**.
- 저카디널리티 이진/범주 플래그 7개(`session_europe`, `session_us`, `is_hour_open`,
  `dual_momentum`, `regime_trending`, `jump_flag`, `evt_tail_flag`) — 이들은 값 분포가
  8~60% 사이로 합리적으로 균형 잡혀 있어(`jump_flag`/`evt_tail_flag`만 ~3.3~3.5% 양성률로
  희소 이벤트 지표) "정보 없음"으로 단정하지 않음. 별도 카테고리로만 기록.

### 0.9 임계 connected-components: 14개 클러스터, 38/102 feature 포함

| 클러스터 | 크기 | feature |
|---|---|---|
| OHLC + OI | 5 | `close, high, low, open, sum_open_interest_value` (r≈1.0) |
| 거래량군 | 5 | `quote_volume, taker_buy_base, taker_buy_quote, trades, volume` |
| 펀딩군 | 4 | `funding_abs, last_funding_rate, long_squeeze_risk, squeeze_power` |
| 변동성 추정량 3종 | 3 | `garman_klass_vol, parkinson_vol, rogers_satchell_vol` (서로 다른 OHLC 기반 변동성 공식이라 애초에 높은 상관이 기대되는 그룹) |
| mean-reversion 계열 | 3 | `fibonacci_level, mean_reversion_z, volume_profile_signal` |
| 나머지 8개 클러스터 | 2씩 | `quote_volume_btc/volume_btc`, `oi_change_rate/smart_money_flow`(r=1.0), `jump_z/log_return`, `mtf_trend_4h/rsi`, `hma_slope/kalman_velocity`, `breakout_strength/cvp_volume_imbalance`, `funding_z_score/ou_funding_z`(r=1.0), `evt_excess_z/evt_tail_flag` |

나머지 **64/102개는 어떤 0.9 클러스터에도 속하지 않음** — 상호 |corr| ≤ 0.9인, 상대적으로
독립적인 feature.

### 구체적 예시 (가장 legible한 것들)

- `smart_money_flow` ↔ `oi_change_rate`: r=1.0000 — 사실상 동일 컬럼.
- `funding_z_score` ↔ `ou_funding_z`: r=1.0000 — 동일.
- `open/high/low/close`: 서로 r≈0.9999~1.0000 (5분봉에서 당연), `sum_open_interest_value`까지
  이 클러스터에 합류(r>0.9) — OI value가 가격과 거의 동행한다는 뜻으로, 과거 메모에 기록된
  "raw feature가 price-trend를 그대로 실어나르는" 오염 패턴과 같은 계열(`spearmanr(feature,
  price)` 사전 점검이 필요했던 `whale_retail_ratio`/`CapMVRVCur`류 사례 참고).
- `garman_klass_vol`/`parkinson_vol`/`rogers_satchell_vol`: 세 개 모두 OHLC 기반 변동성 추정
  공식이 다를 뿐이라 r=0.986~0.997 — 설계상 예견된 중복.

## 해석 / 실용적 결론

38/102(37%)개 feature가 14개의 tight(>0.9) 클러스터로 뭉쳐 있어, 그 부분만 보면 실질 자유도가
feature 개수보다 훨씬 적다(예: 클러스터당 대표 1개씩만 남기면 38개 → 14개, 총 유효 차원은
대략 64(독립) + 14(클러스터 대표) ≈ 78/102). |corr|>0.7까지 넓히면 108쌍으로 중복 범위가 더
커진다.

**한 줄 요약**: 102개 중 대략 20~25%는 다른 feature와 거의 동일한 정보를 실어나르고 있어
통계적으로는 제거 가능해 보인다 — 다만 Odyssey1에서 이미 FINAL12를 포함한 30개 이상의
feature-set 변형을 시도했지만 어느 것도 direction_head skill을 만들어내지 못했으므로, 이
중복 자체가 no-skill의 원인이라는 근거는 아니다(별개 사실로만 기록).

## 재현

`/tmp/claude-1000/-home-kbj20-crypto-scalping/930e2f78-37fd-47e0-bb45-601fad343923/scratchpad/redundancy_analysis.py`
(세션 scratchpad, 레포 미커밋 — 재현 필요 시 위 데이터 경로 두 개를 timestamp inner-join 후
Spearman corr + 0.9 threshold connected-components만 실행하면 동일 결과 재현 가능, 로직이
단순해 재작성 비용 낮음).
