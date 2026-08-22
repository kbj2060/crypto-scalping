# 일리아스(Ilias) — 데이터 및 리소스 관리 (2026-08-17)

이 문서는 일리아스 서브 프로젝트
(`docs/model_contracts/ilias_eth_human_direction_risk_management_contract_20260817.md`)에서
실제로 만지거나 검토한 모든 데이터 소스/리소스를 모은 목록이다. 이번 세션은 부트스트랩(설계
단계)이라 새로 만든 리소스는 없고, 전부 오디세이4에서 **재사용**하는 기존 리소스다 — 아래 표는
전부 이 세션에서 실제 저장소를 확인(경로/파일 존재/행수 실측)한 것이다.

**새 리소스를 만질 때마다 그 턴에 행을 추가/갱신할 것** — 나중으로 미루지 않는다. 상태 값
컨벤션: `활성`, `인프라 확인됨-미착수`, `인프라 차단`, `검증 완료 — 긍정 결과`, `검증 완료 — 부정 결과`.

## 라벨/예측 데이터

| 리소스 | 위치 | 커버리지 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| h48qual TabM 체크포인트 | `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630/true_3head_tabm_bundle.pt` | TRAIN 2024-01~2025-09 학습 | h48qual 베이스 후보(Open Issue a), quality/exit head 재사용 | 활성(오디세이4 라이브 상속) | direction head 출력은 이 프로젝트에서 미사용(사람입력으로 대체) |
| zig075 TabM 체크포인트 | `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629/true_3head_tabm_bundle.pt` | 동일 | zig075 베이스 후보(Open Issue a) | 활성(오디세이4 라이브 상속) | zig075는 exit_head 발동 0/86건(구조적으로 exit-side 손잡이 없음) — 1차 연구 질문 대상으로는 h48qual이 더 유력 |
| h48qual HGB 리스크사이드카 | `tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_plus_t12_livepass_h48qual_q050_precomputed_20260630/risk_sidecar.pkl` | 위와 동일 학습 구간 | L7 사이징(margin_fraction/leverage) 그대로 상속 | 활성 | `HistGradientBoostingRegressor`, `selection_objective=log_risk` — CatBoost 아님(오디세이4 계약에서 정정 확인됨) |
| zig075 HGB 리스크사이드카 | `tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_plus_t12_livepass_zig075_q075_precomputed_20260630/risk_sidecar.pkl` | 위와 동일 | L7 사이징 상속 | 활성 | 동일 |
| h48qual 예측 CSV(train/val/oos) | `tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/h48qual/{train,validation,oos}_predictions_q050.csv` | train=2025-01~09(78,510행, 2025 Q1-Q3 context), validation=2025-10~12(26,490행), oos=2026-01~06(55,405행, oos_q1+oos_q2 통합 — 날짜 필터로 분리 필요) | direction/quality/exit head raw 확률 재사용 — post-entry 상태 벡터 설계의 1차 입력 후보 | 활성 | `train_predictions`는 전체 2024-2025 TRAIN(183,936행)이 아니라 2025 Q1-Q3만 커버(레인지장 재검정이 이미 확인) |
| zig075 예측 CSV(train/val/oos) | `tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/zig075/{train,validation,oos}_predictions_q075.csv` | 동일 커버리지 | 동일 | 활성 | 동일 주의 |
| 오디세이4 랜덤방향 어블레이션 산출물(N=30 최종) | `tmp/causal_regen_20260516/eth_odyssey4_random_direction_risk_management_ablation_20260817/{ablation_results.csv,report.json,exit_reason_distribution.csv,exit_reason_distribution_by_arm_type.csv}` | VAL/OOS-Q1/OOS-Q2 + 레인지 3구간, always_long/always_short/random(N=30 시드) | "사람 방향 입력" 시뮬레이션 프록시(방법 A) 및 1차 연구 질문 비교 기준선 데이터 — 재계산 없이 재사용 | 활성 | N=30 결과는 `docs/model_contracts/odyssey4_eth_entry_veto_baseline_contract_20260814.md` 실행 로그 #7에 통계 확정 요약됨 |

## 라이브 수집 데이터 (duckdb 등)

해당없음 — 이 서브프로젝트는 이번 세션 기준 순수 offline 연구/설계이며 라이브 duckdb를 사용하지
않는다.

## 외부 다운로더 / API

해당없음.

## 인프라

| 리소스 | 위치 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|
| Feature engine | `features/engineering.py`(`FeatureEngineer` 클래스) | 원시 피처 생성, 오디세이4와 동일 계약 상속 | 활성 | 102 base 피처 목록은 여기서 직접 나열되지 않고 라이브 번들의 `bundle["base_cols"]`가 실제 계약을 고정 |
| 다중구간 확인 게이트 | `scripts/eth_omega461_multiwindow_confirmation_gate_20260814.py` | VAL(2025-10-01~12-31)/OOS-Q1(2026-01-01~03-31)/OOS-Q2(2026-04-01~06-30) 표준 로더 + 단일터치 판정 인프라 | 활성 | 이 서브프로젝트의 Dataset Split 표 날짜가 이 모듈의 `WINDOW_DEFS`에서 그대로 확인됨 |
| 사람 방향입력 프록시 핵심 함수 | `scripts/research_eth_odyssey4_random_direction_risk_management_ablation_20260817.py`의 `prepare_component_direction_override`(129번째 줄, 시그니처: `(frame, pred_csv, cfg, device, *, oof, side_selector) -> dict`) | 원시 예측 CSV의 `final_action`/`quality_for_action`을 방향 선택자로 덮어쓰는 재사용 가능 헬퍼 — 1차 연구 질문의 사람방향 시뮬레이션(방법 A/B) 구현 시 그대로 재사용 예정 | 활성 | `dir_p_*`/`dir_action`은 의도적으로 유지(모델 자신의 방향 의견이 리스크 사이드카 context feature로 남는 human-in-the-loop 상태 재현) |
| 오디세이4 랜덤방향 파생 스크립트 3개 | `scripts/research_eth_odyssey4_random_direction_exit_reason_distribution_20260817.py`(exit 사유 분포), `scripts/research_eth_odyssey4_random_direction_ranging_market_retest_20260817.py`(레인지장 재검정), `scripts/research_eth_odyssey4_random_direction_large_n_reverification_20260817.py`(N=30 재검정) | 전부 위 어블레이션 스크립트를 무수정 재사용/import — 1차 연구 질문의 방법론 설계 참고 및 실제 구현 시 재사용 후보 | 활성 | 3개 전부 신규 코드는 최소(윈도우 로더/N 확장 정도)이며 핵심 로직은 원본 어블레이션 스크립트에 있음 |

## 1차 연구 질문 구현 산출물 (2026-08-17, 구현·테스트 세션)

| 리소스 | 위치 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|
| 반사실적 TP/SL 배리어 시뮬레이션 + 신규 replay 함수 | `scripts/research_ilias_eth_adaptive_exit_signal_common_20260817.py` | `simulate_private_barrier_trades`(라벨/평가 공용), `greedy_replay_new_exit_signal`(veto_mod.greedy_replay_entry_veto 문서화된 복사본, h48qual exit_head를 신규 분류기로 교체) | 활성 | 순환논리 회피 핵심 — exit_head 실제 발동 이력을 완전히 무시하고 가격 배리어만으로 라벨 구성 |
| h48qual TRAIN 라벨 데이터셋 | `tmp/causal_regen_20260516/ilias_eth_adaptive_exit_signal_baseline_20260817/train_labels_h48qual_2025q1q3.csv` | 65트레이드/60,694행, SL 양성률 63.9% — `scripts/research_ilias_eth_adaptive_exit_signal_labels_20260817.py` 산출 | 활성 | TRAIN 명목범위(2024-01~2025-09)의 진부분집합(2025-01~09만, 2024년 h48qual OOF 예측 CSV 부재) |
| 신규 exit 신호 분류기 번들 | `tmp/causal_regen_20260516/ilias_eth_adaptive_exit_signal_baseline_20260817/{new_exit_signal_bundle.pkl,new_exit_signal_bundle_secondary.pkl,train_report.json}` | 로지스틱회귀(1차)+HGB(2차, 참고용) 프리즌 모델, threshold=0.5 고정 | 활성 | GroupKFold(5) CV AUC 로지스틱 0.73/HGB 0.65, N=65트레이드로 표본 작음 |
| 6윈도우 성공/킬 평가 산출물 | `tmp/causal_regen_20260516/ilias_eth_adaptive_exit_signal_baseline_20260817/{arm_eval_criterion1_by_arm.csv,arm_eval_report.json}` | `scripts/research_ilias_eth_adaptive_exit_signal_arm_eval_20260817.py` 산출, 성공조건1/2 판정 전체 | 활성 | 상세 결과는 `docs/experiments/ilias_eth_adaptive_exit_signal_baseline_test_20260817.md` §3 |

## 정정 세션 산출물 (2026-08-17, 같은 날 side-blind 재검증)

`docs/experiments/ilias_eth_adaptive_exit_signal_baseline_test_20260817.md` §7 참고 — 위
"1차 연구 질문 구현 산출물" 표의 베이스라인 모델에서 `pos_side`/`pos_leverage`/`pos_notional`/
`pos_exposure` quasi-separation(|coef|=21~27)이 발견돼, 그 4개를 제외한 side-blind 버전으로
재학습·재검증했다. 원본 산출물(위 표)은 무수정 보존, 아래는 전부 신규/별도 파일.

| 리소스 | 위치 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|
| side-blind 공유 모듈 | `scripts/research_ilias_eth_adaptive_exit_signal_common_sideblind_20260817.py` | `FEATURE_COLUMNS`(10개, 방향/사이징 4개 제외), 일반화된 `score_new_exit_signal`/`greedy_replay_new_exit_signal_sideblind`(bundle["feature_columns"]를 이름으로 조회) | 활성 | `simulate_private_barrier_trades`는 원본 `..._common_20260817.py`에서 무수정 재수입(피처 부분집합과 무관하게 전체 raw 컬럼 출력) |
| side-blind 재학습 스크립트 | `scripts/research_ilias_eth_adaptive_exit_signal_train_sideblind_20260817.py` | 원본 `train_labels_h48qual_2025q1q3.csv`(재사용, 재생성 없음) 기반 재학습 + 표준화 계수 재확인/보고 | 활성 | 원본 `..._train_20260817.py`는 무수정 보존 |
| side-blind 모델 번들 | `tmp/causal_regen_20260516/ilias_eth_adaptive_exit_signal_baseline_20260817/{new_exit_signal_bundle_sideblind.pkl,new_exit_signal_bundle_sideblind_secondary.pkl,train_report_sideblind.json}` | 로지스틱회귀(1차, CV AUC 0.6488) + HGB(2차, CV AUC 0.4210) 프리즌 모델, threshold=0.5 고정 | 활성 | 계수 전부 정상 범위(최대 \|coef\|=0.44) — quasi-separation 재발 없음 확인됨 |
| side-blind 6윈도우 재검증 스크립트 | `scripts/research_ilias_eth_adaptive_exit_signal_arm_eval_sideblind_20260817.py` | 원본과 동일 성공/킬 기준·N=30 시드셋·가드레일로 side-blind 번들 재검증 | 활성 | 원본 `..._arm_eval_20260817.py`는 무수정 보존 |
| side-blind 6윈도우 재검증 산출물 | `tmp/causal_regen_20260516/ilias_eth_adaptive_exit_signal_baseline_20260817/{arm_eval_criterion1_by_arm_sideblind.csv,arm_eval_report_sideblind.json}` | 성공조건1(6/6 통과 유지)/성공조건2(3/6 통과, 구성 변경) 판정 전체 | 검증 완료 — 성공조건1 긍정 결과 재확인, 성공조건2 부분 통과 | 상세 결과는 실험문서 §7.5~§7.7 |

## 레짐게이팅 하이브리드 산출물 (2026-08-17, 같은 날 4차 세션)

`docs/experiments/ilias_eth_adaptive_exit_signal_baseline_test_20260817.md` §9 참고 — 위
"정정 세션 산출물"의 side-blind 신규신호를, Odyssey3의 기존 배포 레짐가드 탐지기로 게이팅해(ON=
h48qual 원본 exit_head 0.95, OFF=side-blind 신규신호 0.5) 재평가했다. 원본/side-blind 산출물은
전부 무수정 보존, 아래는 전부 신규/별도 파일.

| 리소스 | 위치 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|
| 레짐게이팅 공유 replay 모듈 | `scripts/research_ilias_eth_adaptive_exit_signal_common_regime_gated_20260817.py` | `greedy_replay_new_exit_signal_regime_gated`(§7 side-blind replay의 문서화된 복사본, 보유-bar 분기에서 탐지기 마스크를 먼저 확인하도록 변경) | 활성 | `simulate_private_barrier_trades`/`score_new_exit_signal`/`POS_VALUE_NAMES`/`FEATURE_COLUMNS`는 side-blind 공유 모듈에서 무수정 재수입 |
| 레짐게이팅 6윈도우 재평가 스크립트 | `scripts/research_ilias_eth_adaptive_exit_signal_arm_eval_regime_gated_20260817.py` | G0 identity check(게이팅 함수가 `new_exit_model` 미부착 시 배포 가드와 바이트 동일 재현함을 확인) + 창별 탐지기 배경활성률 + 성공조건1(재사용, 재실행 안 함)/조건2(재실행) + 3-way 비교(a=real_g0/b=side-blind단독/c=하이브리드) + 트리비얼 판정 | 활성 | 원본/side-blind arm-eval 스크립트는 무수정 보존, `arm_eval_report_sideblind.json`을 읽기 전용으로 재사용 |
| 레짐게이팅 6윈도우 재평가 산출물 | `tmp/causal_regen_20260516/ilias_eth_adaptive_exit_signal_regime_gated_20260817/arm_eval_report_regime_gated.json` | G0 identity check 결과(6/6 창 pnl/mdd/trades 완전일치), 창별 탐지기 활성률(VAL 7.55%/OOS-Q1 5.44%/OOS-Q2 8.19%/레인지① 15.98%/레인지② 11.53%/레인지③ 8.66%), criterion1/2, 3-way 비교표 전체 | 검증 완료 — 관성 아님이나 목표 실패창 3개 중 1개만 회복 | 상세 결과는 실험문서 §9.4~§9.6 |

## 미검증 후보 / 보류

- h48qual vs zig075 베이스 선택 — Open Issue (a), 1차 연구 질문은 h48qual로 진행(근본원인 진단이
  이미 확정한 우선순위). zig075 확장은 이번 세션 범위 밖으로 남음.
- exit_head 능동화를 위한 신규 라벨 — **구현 완료**(반사실적 TP/SL 배리어 재구성,
  `docs/experiments/ilias_eth_adaptive_exit_signal_baseline_test_20260817.md` §1). 결과 판정은
  같은 문서 §4 참고.
- post-entry HGB 리스크사이드카 재평가 가능 여부(bar마다 사이드카를 재호출해 리스크 추정치를
  post-entry 신호로 쓸 수 있는지) — 사이드카가 원래 entry 시점 1회 호출 설계인지 확인 필요, 이번
  세션에서 코드 레벨 확인 안 함.

## 라벨로직 후보축 이관 리소스 (2026-08-21, `eth_tabm_label_logic_retest_initiative`에서 이관)

`docs/model_contracts/ilias_eth_human_direction_risk_management_contract_20260817.md`의
"일리아스 라벨로직 후보축" 절 참고. Baseline v1/일리아스 1과 무관한 별도 154피쳐 공간.

### 154피쳐 데이터셋 (2024-01-01 ~ 2026-06-30, 신규 확정)

| 리소스 | 위치 | 커버리지 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| 154피쳐 최종 데이터셋(연도별) | `tmp/ilias_eth_154feature_dataset_20260821/ilias_eth_154feature_{2024,2025,2026}.csv` | 각 105,380/105,101/51,746행 | 154피쳐 재학습용 원자재 | 활성 — 완전성감사 통과(154개중143개 NaN0, 나머지11개도 최대0.076%) | 2025/2026은 기존 5-way 학습에 미사용(그건 canonical TRAIN_CSV/EVAL_CSV 원본을 wrapper가 실시간 계산), 이건 신규 2024확장 포함 materialized 버전 |
| 154피쳐 통합 데이터셋 | `tmp/ilias_eth_154feature_dataset_20260821/ilias_eth_154feature_2024_2026H1_combined.csv` | 262,227행, 2024-01-01~2026-06-30 | 위와 동일, 단일파일 | 활성 | 연도경계 gap 없음(combo/financial-ML 피쳐를 연속 시계열에서 한번에 계산) |
| 매니페스트(피쳐목록154개+NaN감사+소스파일) | `tmp/ilias_eth_154feature_dataset_20260821/manifest.json` | — | 재현/감사용 전체 메타데이터 | 활성 | |
| 154피쳐 데이터셋 빌드 스크립트 | `scripts/ilias_eth_154feature_dataset_build_20260821.py` | — | 재실행 가능(2024/2025/2026 base+regime overlay 병합→30 RIT조합+12 financial-ML 계산) | 활성 | |
| 154피쳐 runtime wrapper(원본, 5-way 학습이 실제 쓴 것) | `scripts/eth_dc_engineered_features_canonicaldata_20260820.py` | 2025 TRAIN+2026(REGIME3_CURRENT필터) EVAL만 | omega._load_omega_frames/_numeric_feature_cols 오버라이드로 실시간 154피쳐 부착 | 활성 | 위 materialized CSV와 다른 경로(runtime 계산 vs 저장파일) — 둘 다 동일 154개 피쳐명세 사용 |
| 2024 regime3_current HMM sidecar(신규 생성) | `data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/training_features_2024_regime3_current_sensitive_hmm_wide24.csv` | 105,380행, 2024-01-01~12-31 | regime3_current_sensitive_wide24_{bull,bear,chop}_prob/confidence/entropy/margin 6개 오버레이(154피쳐 중 3개가 사용) | 활성(정식 경로, 기존 SELECTED_MAIN 매니페스트가 명시했으나 누락돼있던 파일을 재생성) | 기존 fitted joblib(`regime3_current_sensitive_hmm_wide24_2024.joblib`, 같은 폴더) 재사용, 재적합 안 함. 안전성 검증은 [[eth_regime3_current_2024_training_data_compatibility_20260821]] 메모리 참고 |

### 라벨 소스 (zigzag/h48qual/cusum)

| 리소스 | 위치 | 커버리지 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| zigzag/h48qual 공유 direction 라벨 | `tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531/` | **2026-02-28에서 끊김**(2024/2025는 전체) | zigzag direction, h48qual direction(quality는 별도) | 활성이나 결함있음 | 근본원인 미조사, Open Issue (f) |
| h48qual quality 게이트 | `tmp/causal_regen_20260516/omega_zigzag_fix_all_solutions_20260630/label_contracts/sltp_h48_conservative_padded_to_zigzag_timestamps/` | 2024-01~2026-02-28(zigzag와 동일 제약 상속) | h48qual quality_label_action 소스 | 활성이나 결함있음 | 위와 동일 이슈 |
| cusum dense-cashfill 라벨 | `tmp/eth_cusum_triple_barrier_labels_dense_cashfill_20260820/` | 전체 2024-2026(끊김없음) | cusum direction 소스, 5-way 학습에 실사용 | 활성 | |

### 학습/분석 스크립트 + 산출물

| 리소스 | 위치 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|
| 5-way(4-way classification) 라벨스왑 러너 | `scripts/eth_tabm_label_logic_5way_seed_variant_20260820.py` | zigzag/h48qual/dc/cusum, --label/--seed로 스왑 | 활성 | exit-label-mode를 4개 다 independent_entry_hold_offsets로 통일(zig075/h48qual 라이브 설정과 다름) |
| 151피쳐 검증용 wrapper+러너(regime3 3개 제외판, 비교실험용) | `scripts/ilias_eth_engineered151_features_canonicaldata_20260821.py`, `scripts/ilias_eth_label_151feature_seed_variant_20260821.py` | 154→151 변경이 OOS부호를 뒤집는지 검증(zigzag 실제로 뒤집힘 확인) | 활성, 최종 채택 안 됨(154 유지 결정) | 이 두 스크립트는 "제거해도 되는지" 검증용으로만 쓰였고 최종 154피쳐 결정에는 미반영 |
| 5-way 학습 결과 report.json들 | `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_label5way_{label}_154feat_unified_single_model_seed{seed}_20260820/report.json` | zigzag/h48qual/dc/cusum × 3시드[133725056,176495706,796203462] | 활성 | |
| 구조분석 스크립트+결과 | `scripts/eth_zigzag_h48qual_cusum_structural_similarity_20260821.py`, `/tmp/claude-1000/.../scratchpad/structural_similarity_20260821.json` | 방향전환빈도/같은-bar매칭/순열귀무/ATR분포 | 활성, 산출물은 세션로컬 scratch(재실행으로 재현) | |
| 라벨비교 3패널 차트 | `scripts/chart_zigzag_h48qual_cusum_label_comparison_20260821.py`, `tmp/research_20260821/chart_zigzag_h48qual_cusum_label_comparison.png` | 2025-01-06~01-20 가격구간 시각비교 | 활성 | |
| OOS 트레이드원장 재구성 스크립트+원장 | `scripts/eth_zigzag_h48qual_cusum_oos_trade_ledger_20260821.py`, `/tmp/claude-1000/.../scratchpad/oos_trade_ledgers/*.csv` | zigzag/h48qual/cusum 개별 트레이드 목록(entry/exit ts·price·reason·pnl) | 활성(report.json과 cross-check 통과), 산출물은 세션로컬 scratch | omega._metrics() 로직을 그대로 계측 복제 — 재구현 아님 |
| OOS 등가곡선 차트 | `tmp/research_20260821/chart_zigzag_h48qual_cusum_oos_equity_curves.png` | 3라벨 등가곡선(cusum은 전체+Jan-Feb매칭 둘다) | 활성 | |
| 전체 실험기록 원본 | `docs/experiments/eth_tabm_label_logic_5way_comparison_20260820.md` | 5-way+구조분석+트레이드원장 전체 방법론/결과 | 활성 | 이관 전 원본 서브프로젝트 문서, 계속 참고용으로 유지 |

## 일리아스1 재편 리소스 (2026-08-21)

| 리소스 | 위치 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|
| (구)일리아스1 모델자산 전체 | 이관됨 → `docs/model_contracts/odyssey5_eth_position_feature_parity_fix_contract_20260821.md` | h48qual/zig075 버그수정 재학습판 + risk sidecar 2개 | 이관됨(구경로 심볼릭링크로 계속 동작) | 이 문서 상단 "라벨/예측 데이터" 절의 h48qual/zig075 TabM 체크포인트(20260630 빌드, 오디세이4 라이브 상속)와는 다른 리소스 — 그건 Baseline v1용, 이건 옛 "일리아스1"용이었음 |
| (신)일리아스1 zig075슬롯 TabM 번들 | `tmp/causal_regen_20260516/ilias1_eth_zig075slot_154feat_unified_single_model_seed133725056_20260821/true_3head_tabm_bundle.pt` | 154피쳐+단일모델, zigzag_action direction 라벨(zig075 정의와 동일) | 활성, N=3 예비 스크리닝만 거침 | `eth_tabm_label_logic_5way_seed_variant_20260820.py --label zigzag --seed 133725056` 산출물을 복사 |
| (신)일리아스1 h48qual슬롯 TabM 번들 | `tmp/causal_regen_20260516/ilias1_eth_h48qualslot_154feat_unified_single_model_seed133725056_20260821/true_3head_tabm_bundle.pt` | 154피쳐+단일모델, h48qual quality게이트 라벨 | 활성, N=3 예비 스크리닝만 거침 | `--label h48qual --seed 133725056` 산출물을 복사 |

## 데이터 무결성 대수술 산출물 (2026-08-23 추가)

| 리소스 | 위치 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|
| metrics 참조 진실값 3종 | `data/TOTAL_{ETH,BTC,SOL}USDT_metrics_2024_2026.csv` | OI/롱숏비 계열의 유일 신뢰 기준(+5분 종료라벨 보정본) | 활성 | 신규 다운로드시 +5분 보정 필수(아카이브 원본=시작라벨). 재생성: `scripts/download_eth_binance_metrics_archive_20260823.py` |
| 수정된 캐노니컬 전체 | 계약서 "데이터 무결성 현황" 절의 표 참고 | ETH 2026 BTC-오염 제거 + BTC/SOL vintage 수정(BTC 2024 미래참조 24% 제거) | 활성 | 08-23 이전 실험 재현은 `.bak_*_20260823` 백업으로 |
| 재생성된 wide24 오버레이 3종 | `tmp/ilias_labellogic_recheck_20260821/*_regime3_current_states24_sticky090.csv` | 레짐 소스 표준(수정된 캐노니컬 기반) | 활성 | 구 balancedish 사이드카(`data/ensemble/supervised/..._20260530/`)를 새 실험 레짐 소스로 쓰지 말 것 |
| 패치된 154피쳐셋 | `tmp/ilias_eth_154feature_dataset_20260821/` | 2026=51,841행(2/28 갭 삽입 반영), combined=262,322행 | 활성 | `manifest.json`의 `patched_20260823` 필드 참고 |
