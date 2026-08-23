# ETH wide24 HMM 레짐분류기 — 학습이력 확장(2022-01~ vs 2024-01~) 비교 실험

- 날짜: 2026-08-23
- 대상: `regime3_current_sensitive_v2_hmm_wide24` (states=24, sticky=0.90, n_iter=22, seed=7529) —
  현재 유일하게 N=5 시드안정성이 CONFIRMED된 레짐분류기 축
  ([[eth_regime_classifier_wide24_vs_jm_sjm_investigation_20260821]])
- 목적: TRAIN 구간을 기존 2024-01-01~2026-06-30에서 2022-01-01~2026-06-30(+2년치 이력)으로
  늘리면 무엇이 달라지는가 — 순수 모델-변형 비교이며 트레이딩 전략/PnL 백테스트 아님
- Fresh-forward 관련: `fresh_forward_bar_by_bar=false`, `trade_ledgers_used_as_input=false`,
  `saved_parent_exit_timestamps_used=false`, `future_rows_used_for_entry=false` — 본 실험은
  레짐 HMM의 forward/causal `filter_proba`만 사용했고(재적합 없이 VAL을 causal하게 필터링),
  저장 원장/과거 청산 타임스탬프는 입력으로 쓰지 않았다.

## 데이터/파이프라인

- 신규 확장 입력: `tmp/eth_wide24_history_extension_20260823/eth_wide24_inputs_2021_2023.csv`
  (219,168행, 2021-12-01~2023-12-31), 이미 `gate_report.json`으로 wide24 21개 입력컬럼이
  2026 캐노니컬 대비 두 probe 구간에서 100% 일치(단 `bb_width_z`가 한 구간에서 99.97%,
  반올림 수준으로 무시가능) 검증 완료.
- 최종 직접 concat sanity check(본 세션 수행): STATE12_COLS/WIDE24_EXTRA_COLS를 구성하는
  22개 원시 입력 컬럼(`close/high/low/bb_width_z/chop_index/mtf_trend_1h/mtf_trend_4h/
  hma_slope/breakout_strength/dual_momentum/mean_reversion_z/net_taker_ratio/
  smart_money_flow/taker_acceleration/ofi_acceleration/log_return/garman_klass_vol/
  oi_change_rate/volatility_z/rsi/macd_hist/wick_ratio`)가 2021_2023 확장파일·2024·2025·
  2026_rebuilt 4개 파일 전부에 존재하고, 전부 `float64` dtype, 해당 22개 컬럼에 NaN 0건 —
  concat 안전성 확인. (`adx_14`는 2024/2025 캐노니컬에 아예 없는 컬럼이지만 wide24
  feature set에는 쓰이지 않는 라벨보조 컬럼이라 무관.)
- TRAIN 프레임 구성: 2024/2025/2026_rebuilt는 오늘(2026-08-23) 두 metrics-integrity 결함이
  수정된 최신본을 이 세션에서 직접 재로드해 새로 concat(기존 `tmp/eth_hmm_wide24_resweep_
  train2026h1_20260821/train_merged_*.csv`를 재사용하지 않고, 신뢰 지시에 따라 캐노니컬을
  그대로 다시 병합) — baseline/extended가 완전히 동일한 최신 데이터를 공유하도록 함.
  - baseline raw = concat(2024, 2025, 2026_rebuilt≤2026-06-30) → 262,609행
  - extended raw = concat(2021_2023 확장파일≤2023-12-31, 2024, 2025, 2026_rebuilt≤2026-06-30)
    → 481,777행(2021-12 워밍업 포함)
  - `_with_features()`(롤링 윈도우 최대 288bar=1일)를 전체 raw에 먼저 적용해 롤링 통계가
    2021-12 리드인을 보게 한 뒤, extended는 fit/eval 범위를 2022-01-01↑로만 잘라냄(2021-12는
    워밍업 전용, 날짜비교에는 안 들어감) — baseline 236,401행 vs extended 446,641행(TRAIN,
    2026-04-01 이전)
- 재사용한 원본 로직: `scripts/experiment_regime3_current_hmm_wide24_20260529.py`의
  `_with_features / _fit_obs / _labels / _state_class_matrix / _class_proba / _eval /
  GaussianStateModel / FEATURE_SETS["wide24"] / LABEL_CONFIGS["balancedish_adx16_slope03_bb006"]`을
  그대로 import해 조합했다(스케일링/피팅/평가 로직 재구현 없음). VAL split(`--val-start`
  동등 지점)은 2026-04-01~2026-06-30(2026Q2, 기존 컨벤션과 동일 26,208행)로 고정.
- 재현성 확인: 이번 세션에서 새로 병합한 baseline·seed=7529 결과가 8/21~23 재스윕에서 이미
  나온 `postfix_recheck_states24_sticky0.90_seed7529` 리포트의 VAL balanced_accuracy와
  **완전히 일치**(`0.7480087986918441`, abs_diff=0.0) — 파이프라인 재구성이 기존 라이브 모델
  결과를 정확히 재현함을 확인한 뒤 확장 비교로 넘어감.
- 스크립트: `tmp/eth_wide24_history_extension_20260823/run_history_extension_comparison.py`
  (baseline+extended × 5시드, 산출물은 전부 `tmp/eth_wide24_history_extension_20260823/`
  아래에만 기록, `data/`나 2026-07-01~09-30 OOS 구간은 건드리지 않음)

## 1) 시드안정성(N=5) — Seed-Diversity Ensemble Promotion Gate 대조

시드 `[7529, 534964, 116595, 666940, 505456]`(기존 wide24 N=5 검증과 동일 리스트, 진짜
무작위 추출, 고정간격 아님)로 두 variant를 각각 재적합, VAL(2026Q2) balanced_accuracy 표준편차:

| variant | VAL bal_acc 평균 | std | 기준선(≈0.0001) 대비 |
|---|---|---|---|
| baseline (2024-01~) | 0.748020 | **3.18e-05** | 통과(기존과 동급) |
| extended (2022-01~) | 0.742664 | **5.10e-05** | 통과 |

두 variant 모두 std가 0.0001 미만으로, extended variant도 CLAUDE.md Seed-Diversity
Ensemble Promotion Gate(N≥5, std가 신호와 시드분산을 구분 못 할 만큼 크지 않을 것)를
통과한다. 즉 "학습이력을 늘려도 시드에 불안정해지지는 않는다" — 이 축만 보면 승격 후보
자격은 유지.

## 2) VAL 구간 필터확률 괴리 (seed=7529, 동일 26,208행, forward/causal filter만, 재적합 없음)

| class | mean abs diff | max abs diff |
|---|---|---|
| bull | 0.0683 | 0.386 |
| bear | 0.0707 | 0.489 |
| chop | 0.0940 | 0.386 |
| 전체 | **0.0776** | **0.489** |

두 variant의 필터확률이 평균적으로 클래스당 약 7~9%p, 최악의 경우 한 시점에서 약 49%p까지
벌어진다 — 이는 "거의 같음" 수준이 아니라 실질적 재보정(recalibration)에 가까운 크기다.

## 3) balanced-accuracy / 라벨일치 비교 (`_current_labels3_thresholded`, VAL 2026Q2, seed=7529)

| 지표 | baseline (2024-01~) | extended (2022-01~) | 차이 |
|---|---|---|---|
| accuracy | 0.7466 | 0.7022 | **−4.44%p** |
| balanced_accuracy | 0.7480 | 0.7427 | **−0.53%p** |
| log_loss | 0.7003 | 0.7555 | **+0.055 (악화)** |
| recall(bull) | 0.681 | 0.869 | +0.188 |
| recall(bear) | 0.831 | 0.821 | −0.010 |
| recall(chop) | 0.732 | **0.538** | **−0.194** |
| flip_rate | 0.127 | 0.153 | +0.026(덜 지속적) |
| mean_state_duration(bar) | 7.86 | 6.53 | −1.33 |

extended는 bull 재현율만 크게 오르고(과다예측: pred bull 9,690 vs 실제 6,720) chop 재현율이
급락한다(2026Q2의 실제 chop 12,147행 중 상당수를 bull/bear로 오분류). log_loss도 확실히
악화. N=5 평균으로도 balanced_accuracy는 extended가 baseline보다 일관되게 −0.55%p 낮다
(0.7427 vs 0.7480, 표에는 seed7529 단일값 기재했으나 5시드 평균도 동일 방향).

## 정직한 caveat

2022-01~2023-12는 Terra/Luna 붕괴(2022-05)와 FTX 파산(2022-11) 등 이 리포에서 아직 한 번도
학습에 넣어본 적 없는 극단적 레짐을 포함한다. chop 재현율 급락과 state duration 단축은
이 이질적 구간이 HMM의 상태-이모션(mu/var) 추정과 상태→클래스 지도학습 캘리브레이션
(`_state_class_matrix`, ADX 정답라벨 기반)을 2026Q2와는 다른 방향으로 끌어당긴 결과로
보인다 — 즉 "이력을 늘려서 일반화가 좋아진다"는 통상적 직관과 반대로, 이 축에서는 이력
확장이 최근 레짐(VAL이 속한 2026Q2)에 대한 적합도를 오히려 희석시켰다. 이는 회귀
(regression)이지 우연한 노이즈가 아니다 — N=5 시드 전부에서 baseline이 일관되게 우위였고
격차도 시드std(≈0.00005)보다 훨씬 크다(baseline-extended 격차 ≈0.0055, std의 100배 이상).

## 결론 / 권고

**학습 윈도우를 2022-01~2026-06-30으로 바꾸는 것은 권장하지 않는다.** 시드안정성은
유지되지만(승격 게이트 자체는 통과), VAL(2026Q2) balanced_accuracy·log_loss·chop 재현율이
전부 baseline 대비 악화되고 필터확률도 최대 49%p까지 크게 갈린다. 현재 라이브 모델
(`regime3_current_sensitive_v2_hmm_wide24_2024.joblib`, TRAIN=2024-01~2026-06-30)을
그대로 유지하는 것이 낫다. 만약 이 축을 다시 열고 싶다면, 2022~2023 데이터를 통째로 넣는
대신 (a) Terra/FTX 붕괴 구간만 별도 제외하고 재시도하거나 (b) time-decay 가중치를 줘서
최근 구간(2024~2026H1)의 영향력을 우선하는 방식이 다음 시도 후보가 될 수 있다(아직
시도 안 함, 별도 실험 필요).

## 산출물

- `tmp/eth_wide24_history_extension_20260823/comparison_report.json` — 전체 수치(시드별
  balanced_accuracy/log_loss, 확률괴리, 혼동행렬 등)
- `tmp/eth_wide24_history_extension_20260823/models/{baseline,extended}_seed{seed}_val_model.joblib`
  — variant×seed별 val_model 페이로드(스케일러/state_class_val 포함, 재현용)
- `tmp/eth_wide24_history_extension_20260823/val_proba_seed7529.npz` — seed=7529 VAL 원시
  확률 배열(양쪽 variant, 확률괴리 재계산용)
- `tmp/eth_wide24_history_extension_20260823/run_history_extension_comparison.py` — 실행
  스크립트(재사용 가능)
