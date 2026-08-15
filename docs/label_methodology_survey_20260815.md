# 저장소 라벨링 히스토리 종합 조사 (2026-08-15)

`core/event_label_engine.py`(새 범용 라벨 엔진) 설계 근거를 확인하려고 배경 에이전트 4개를
띄워 저장소의 라벨 생성 스크립트·문서 전체를 훑었다. 총 100개 이상의 스크립트와 60개 이상의
`docs/`를 읽고 40개 이상의 서로 다른 방법론으로 분류했다. 이 문서는 그 결과에서 **재사용
가능한 교훈만** 추린 것이다 — 각 방법론의 전체 세부사항(정확한 수식·파라미터·검증 수치)은
아래 표가 가리키는 원본 `docs/model_contracts/*`, `docs/experiments/*`, `docs/*.md`가 1차
자료이며, 이 문서는 그것들의 색인이다.

## 핵심 메타 발견: 방향성 alpha의 보편적 부재

BTC/ETH/SOL 5분봉(일부 1분·1시간봉)에서 시도된 **모든** 방향/진입 라벨 계열 — triple-barrier
(11개 변형), zigzag pivot(11개 변형), 순수 meta-labeling(7개 변형), trend-scanning(8개
변형), DP/Bellman oracle(4개 계열), CatBoost 라벨 스코어링(alpha5~7), panel rank, HMM
confluence — 이 엄밀하게(시드 평균, VAL+OOS, always_short/random 대비) 재검증됐을 때
**방향 예측 edge가 없다는 결론으로 수렴**했다.

반복되는 패턴: 라벨 자체는 종종 통계적으로 학습 가능하다(예: pivot-transition-hazard
분류기 AUC 0.90~0.95, wave3 action BAcc 0.565) — 하지만 이게 수익화로 이어지지 않는다
("signal real, monetization fails"가 최소 2개 독립 라인에서 재현). 진단된 원인은 거의
매번 배리어/샘플링 공식이 아니라:

1. **feature information content 부재** — 가장 흔한 결론. h48qual 하나에서만 10회 이상
   독립적으로 재확인(방향 헤드 교체, GBDT 백본, trend-scan MI/R² 게이트, quality-head
   9종 교체 후보 전부 포함).
2. **표본 유효 N 문제** — 배리어 구간이 겹쳐 라벨이 사실상 독립이 아님.
3. **발동률-정밀도 트레이드오프** — 고정밀 임계값은 발동이 너무 드물어 PnL을 못 움직임
   (pivot-transition-hazard: 정밀도 7~20배 상승해도 exit-capture ≈0).
4. **train/eval 비용함수 불일치**.
5. **시드 선택 아티팩트** — 단일 시드 "승리"가 N≥5 재검증에서 뒤집힘(SOL dual/h24wide,
   alpha5_29 router, BTC-110 이벤트-TB 등 3회 이상). [[tabm_hp_low_signal_pattern]]과 같은
   패턴.

**이 라벨 엔진은 원인 1을 해결하지 못한다.** 더 나은 라벨 구성·표본가중치·CV 위생을
제공할 뿐, 사용 가능한 feature set에 정보가 없다는 문제는 완전히 별개 축이다.

## `core/causal_event_labels.py`의 실제 위치

이 소규모 CUSUM+triple-barrier 유틸리티를 실제로 import하는 스크립트는 2개뿐이고
(`train_eval_btc110_cusum_tb_causal_20260804.py`, `..._expectancy_causal_20260804.py`),
**둘 다 2026-08-04 CAL/TEST 게이트에서 실패**했다. 조사된 나머지 40개 이상의 방법론은
전부 독자적으로 배리어/이벤트 로직을 재구현한다 — 이름의 유사성과 달리 실질적인 공유
확장점이 아니다. → `event_label_engine.py`를 `causal_event_labels.py`의 확장이 아니라
**독립 구현**으로 설계한 것은 사후적으로도 타당했다.

## 방법론 색인 (계열별 대표 항목만, 세부는 문서 링크 참고)

| 계열 | 방법론 | 검증 상태 | 대표 문서 |
|---|---|---|---|
| Triple-Barrier | Omega1.2 ATR-스케일 비대칭 dense grid (=h48qual 실배포 quality 라벨) | 메커니즘 유효, 방향 skill 없음(N≥5) | `odyssey_eth_h48qual_corrected_tabm_20260811_contract.md` |
| Triple-Barrier | ETH ground-truth 순차/포지션게이팅 대칭 배리어 | 검증 문서 미발견 | — |
| Triple-Barrier | Sigma1 배리어-매칭+연속바 평활 | **FAILED**, 0/9 게이트 | `sigma1_seq_barrier_20260704_contract.md` |
| Triple-Barrier | BTC/SOL "race" 라벨(수익분산 스케일 배리어+연성 conviction) | 오라클 유효, 엔트리 alpha **FAILED** | `btc_tripbarrier_zigzag_architecture_design_20260807.md` |
| Triple-Barrier | BTC-110 CUSUM+대칭 3-class(`causal_event_labels.py` 실사용) | **FAILED**, TEST PnL -23.61% | `btc_new_architecture_session_summary_20260804.md` |
| Triple-Barrier | BTC-110 CUSUM+독립 이진+등위보정 기댓값 | **FAILED**, CAL 전부 음수 | 상동 |
| Triple-Barrier | 3-way TP/SL/timeout 재구성 | **CLOSED**, 0/24 | `btc_3way_tpfirst_label_closed_20260804.md` |
| Triple-Barrier | 배리어/호라이즌 보정 스윕 150cell | **CLOSED**, 0/148 | `btc_barrier_horizon_calibration_sweep_closed_20260805.md` |
| Triple-Barrier | 멀티호라이즌 효용최대화(bar별 side×horizon×tp×sl 그리드) | 검증 문서 미발견 | — |
| Triple-Barrier | ETH 1분봉 스캘핑 전용(purged/embargo 가중 변형 포함) | research, BTC-feature 룩어헤드로 **전체 무효화** | `eth_scalp_1m_20260717_audit_findings.md` |
| Zigzag pivot | 정준 ATR-적응 zigzag(wave3/v2) — ETH 실배포 방향 라벨 | 학습 가능하나 BTC 엔트리 edge **CLOSED**(4회 독립 시도 금지) | `btc_tripbarrier_zigzag_architecture_design_20260807.md` |
| Zigzag pivot | 배리어-매칭(라이브 정책 버그 그대로 재현) | **FAILED**, OOS 부호반전 2회 확인 | `omega6_synthesis_v1_20260703_contract.md` |
| Zigzag pivot | corrected-vol 리빌드(12바 누적수익 분산 기반) | 검증 문서 미발견(근거는 견고) | `project-btc-deepfeat-acc-pnl-gap-diagnosis-20260806` |
| Zigzag pivot | pivot/transition-hazard("transition_soon") | 분류기 AUC .90-.95, exit-capture **≈0** | `btc_tripbarrier_zigzag_architecture_design_20260807.md`(G8) |
| Zigzag pivot | BTC H48 causal-hysteresis 평활 | **REJECTED** | `btc_v1_h48_causal_hysteresis_retrain_20260716.md` |
| Meta-labeling | h48qual/zig075 direction+ATR quality 하이브리드 | **CLOSED**, 방향 skill 없음(N≥5, 10+ 재확인) | `odyssey_eth_h48qual_corrected_tabm_20260811_contract.md` |
| Meta-labeling | HMM confluence v1/v2/v3 | **CLOSED**, OOS 전부 음수(v3도) | `eth_hmm_confluence_meta_labels_v2_20260724.md` |
| Meta-labeling | Sigma3 1h trend-scan+정통 López-de-Prado 메타라벨 | **FAILED**, 게이트-불량 시드 클러스터 사용 확인 | `sigma3_1h_trendscan_20260705_contract.md` |
| Meta-labeling | scalp_1m purged-CV 메타라벨(OOF, 가장 엄밀) | diagnostic-only, 소폭 악화 | 리포트 JSON만 존재 |
| Meta-labeling | evidence-signal agreement 메타라벨(oracle→실예측 2단계) | 통계적으로 실재하나 발동률 1~2%로 **실사용 불가**, CLOSED | `eth_direction_head_metalabel_evidence_signal_rank_correlation_20260815.md` |
| Trend-Scanning | 순정 De Prado(1h, Sigma3/6/7/9 + BTC v1/v2) | **CLOSED**, 8개 이상 독립 라인 전부 실패 — Sigma6-filtered가 Seed-Diversity Gate 정책의 실제 계기 사례 | `eth_tau1_sigma6_filtered_closed_20260807.md` |
| Trend-Scanning | h48qual 전용 MI/R² 게이트(8개 윈도우 그리드) | **결정적 음성**, R² 음수 양쪽 다 | `eth_h48qual_trend_scanning_label_mi_r2_gate_20260812.md` |
| Trend-Scanning | trend-scan+oracle-ceiling 하이브리드 | 오라클 체크 **FAILED**(승률 29.3%) | `oracle_validate_all_label_logics_20260806.py` 자체 출력 |
| DP/Oracle | omega1_2_1 age-augmented FLAT/LONG/SHORT DP | 검증 문서 미발견 | — |
| DP/Oracle | ETH action-grid(128액션) 비중첩 DP | `promotion_eligible: false` 자체 명시 | `eth_full_oracle_strategy_labels_v1_20260724.md` |
| DP/Oracle | CSALT(Cross-Fitted SMDP Advantage Teacher) | **CLOSED**, 1320후보 0/6, T1 스모크도 실패 | `btc_csalt_dp_label_loop_final_20260715.md` |
| DP/Oracle | BTC 레짐 3-state DP + DP-vs-triple-barrier 비교 | 검증 문서 미발견(스크립트 자체 경고 有) | — |
| 기타 | alpha5 라벨팩토리(8개 변형) | 아키텍처상 폐기, alpha5_27 OOS BAcc 0.434(랜덤 이하) | `alpha5_catboost_major_direction_deprecated_20260521.md` |
| 기타 | alpha6/7 라벨스코어링(19개 프리셋) | 원본 엔진 삭제됨(git 4c46d20), 미검증 | `alpha6_entry_quality_exit_5bucket_main_20260522_contract.md` |
| 기타 | panel 횡단면 rank(Rho1 Stage 2) | **CLOSED**, MSE 랜덤과 동급, 6config 전부 음수 | `btc_panel_crossasset_architecture_design_20260804.md` |
| 기타 | SOL dual-component(zig075+h24wide 앙상블) | 단일시드 승리 → 5시드 재검증 **뒤집힘** | `sol_dual_h24wide_seed_stability_20260729.md` |

## 발견된 실제 causality 버그 (설계에 직접 반영)

1. **단일-bar 변동성 노이즈** — BTC race 라벨과 zigzag corrected-vol 리빌드 양쪽에서
   독립적으로 진단, 같은 근본원인 문서 인용. → `return_dispersion_volatility()` 추가로 반영.
2. **메타라벨링 OOF 누수** — scalp_1m 메타라벨(가장 엄밀한 구현, purged 5-block CV)이
   기준. → `generate_labels()` docstring에 OOF 경고 + `purged_kfold_splits()` 사용법 명시.
3. **trend-scan 커널 시작-인덱스 버그** — 3개 독립 커널이 forward-window 결과를 구간
   시작 인덱스에 기록해, 그 결과물이 라벨이 아니라 "전체 feature parquet"에 실시간
   피처처럼 섞여 들어감(2026-08-04 발견, Sigma3/6/9 라인 오염). → `trend_scanning_labels()`
   docstring에 "절대 실시간 feature로 쓰지 말 것" 명시 경고 추가, 이 버그를 직접 인용.
4. **DP 전체구간 backward induction이 fold 경계를 넘어 leak** — `eth_scalp_1m` DP 오라클
   라벨이 2024~2026 전체 배열에 한 번에 backward induction을 돌려 TRAIN 타겟이 VAL/OOS
   가격 경로에 의존. → 해당없음(DP 미구현), 대신 모든 라벨을 이벤트-로컬로 설계하고
   `purged_kfold_splits()`로 CV 경계 안전장치 제공.
5. **zigzag_segment_id 연도 충돌**, **transition_buffer 스윙 시작부 오염** — 기존 메모리
   [[zigzag_segment_id_year_collision]] 재확인. 해당없음(segment_id·transition_buffer
   개념 자체가 없는 설계).
6. **라벨-실행 배리어 스케일 불일치** — exit_head/quality_head에서 반복된 가장 비싼
   버그 클래스(라벨의 배리어 가정이 실제 라이브 코드가 하는 일과 달라 OOS에서 부호 반전).
   → pt_mult/sl_mult/max_hold을 전부 config로 노출하고 기본값을 "이 저장소의 배포값"인
   것처럼 하드코딩하지 않음 — 소비자가 실제 쓸 배리어와 명시적으로 맞추게 강제.
7. **컬럼명 충돌** — 여러 무관한 방법론이 전부 `zigzag_action`이라는 컬럼명을 재사용해
   혼동 유발. → `event_idx/label/side/touch_type/trend_tstat` 등 저장소에서 겹치지 않는
   고유 명명 사용.

## 실모델 검증: h48qual TabM parent 메타라벨 테스트

`scripts/diagnose_eth_h48qual_dirhead_metalabel_via_event_label_engine_20260815.py` —
h48qual corrected TabM parent 아티팩트(`tmp/causal_regen_20260516/omega4_3head_parent72_
loose_entry_quality_20260620_h48qual_final12_h384_20260811_v2_e40_r30000_s179660/`)의
**validation+oos_predictions_q050.csv만**(train은 의도적으로 제외 — 위 버그#2와 같은
클래스의 in-sample 누수를 피하기 위함) 사용해 `direction_head`의 raw `dir_action`을
`generate_labels()`의 `side`로 넣어 실제 메타라벨을 만들어봄.

- 정합 표본 30,097개(LONG 15,796 / SHORT 14,301), 전체 적중률 **49.13%**
  (validation 구간 48.88%, oos 구간 49.76% — 동전던지기 수준)
- 이 엔진 자체의 메타라벨링 배관이 실제 외부 모델 출력으로 정상 동작함을 확인한
  스모크 테스트다. 여기 쓴 배리어(pt=sl=2.0×EWMA-vol, 48bar)는 h48qual 실배포 배리어
  (h48_conservative: tp=1.2/sl=0.8×ATR96)와 다르므로, 이 수치는 h48qual에 새 방향
  edge가 있다는 주장이 **아니다** — 위 표의 h48qual 행이 이미 N≥5 시드로 10회 이상
  재확인한 "방향 skill 없음" 결론과 정확히 일치하는 결과일 뿐이다.

## 관련 메모리

[[event_label_engine_design]], [[h48qual_label_mismatch_discovered]],
[[h48qual_standalone_replay_invalid]], [[zigzag_segment_id_year_collision]],
[[tabm_hp_low_signal_pattern]]
