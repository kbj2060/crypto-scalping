# Odyssey — 리소스 관리: 코드 · 데이터 · 모델 (2026-08-12)

이 문서는 Odyssey 서브 프로젝트(`docs/model_contracts/odyssey_eth_h48qual_corrected_tabm_20260811_contract.md`)에서 실제로 만지거나 검토한 모든 코드(스크립트·문서)·데이터 소스·모델 아티팩트를 한 곳에 모은 목록이다. 계약 문서의 "데이터 구간 정의" 절은 h48qual/zig075 예측·라벨 파일의 TRAIN/VAL/OOS 구간만 다루고, 나머지 리소스(스크립트 전체 목록, 모델 번들 경로, 라이브 duckdb, 외부 API, 다운로더 스크립트, GPU 서버 등)는 각 실험 문서에 흩어져 있었다 — 이 문서가 그걸 한 번에 찾을 수 있게 하는 표준 참조다.

**새 리소스를 만질 때마다 그 턴에 행을 추가/갱신할 것** — 나중으로 미루지 않는다.

**⚠ 동시성 참고 (2026-08-12 00:47 KST 스냅샷 기준)**: 이 문서는 이번 갱신 작업 시작 직전(00:36)에 다른 세션이 막 새로 만든 상태였고, 작업 도중에도 `..._fullwindow_recheck_20260812.*`, `eth_zig075_final15_vs_jmlam4_vs_live_comparison_20260812.md` 등 신규 파일이 실시간으로 더 늘어나는 걸 확인했다(`ps aux` 확인 결과 이 레포에서 동시 실행 중인 claude 세션이 다수 존재). 즉 Odyssey 관련 작업을 지금 이 순간에도 병행 진행 중인 세션이 최소 1개 더 있다는 뜻 — 아래 내용은 정확한 스냅샷이지만 계속 늘어나는 파일 집합의 한 시점일 뿐이다. 최신 상태가 필요하면 이 문서에 의존하지 말고 `git status --porcelain -uall | grep -iE "h48qual|zig075|zigzag|h384|odyssey"`를 직접 재실행할 것.

**⚠⚠ 범위 정정 — 매우 중요 (2026-08-12, 병행 세션이 사용자 지적으로 발견)**: 아래 "라이브 프로덕션 번들"·"격리 검증용 실험 번들" 절이 다루는 h48qual/zig075 번들은 **전부 2026-06-30 이전 학습된 구(舊) HMM 레짐 라우팅 번들**이며, 2026-08-11 하루 종일의 진단 체인(confidence-echo, swing-shape, calibration-instability 등) 전부가 이걸 대상으로 실행됐다. 그런데 사용자가 "오디세이 테스트 모델"이라고 부르는 건 이것과 **별개의 JM 리짐 재설계 라인**(jmlam4 → final15, 아래 새 절 참고)이라는 게 오늘 확인됨 — 구 번들은 대조군(비교 기준선)이지 "테스트 모델" 그 자체가 아니다. **이 리소스 문서를 "오디세이 테스트 모델" 리소스 관리 목적으로 읽는다면, 아래 "Odyssey 실제 테스트 모델 후보" 절을 최우선으로 볼 것** — 특히 `final15`는 오늘(2026-08-12 00:42) 처음 학습되어 이번 조사 전체에서 가장 큰 긍정적 결과를 낸 상태. 상세: [[odyssey_eth_h48qual_subproject]]의 2026-08-12 "MAJOR course-correction" 이후 업데이트들.

## 코드 (스크립트 · 문서)

Odyssey 착수(2026-08-11) 이후 생성된 파일만 집계 — 전부 커밋 안 됨(`git status` 기준 `??`, 2026-08-12 00:47 스냅샷: 스크립트 44개 + 문서 26개). 레포에는 이전부터 존재하던 h48qual/zig075 관련 **커밋된** 스크립트가 40개 이상 더 있음(`research_btc_h48qual_*`, `train_eval_omega4_3head_pinned102_extended_*_jmlam4_20260809.py` 등, 2026-07~08월 초의 다른/이전 프로젝트) — 이들은 Odyssey 범위 밖이므로 아래에 포함하지 않음.

### 문서 (26개)

- 계약: `docs/model_contracts/odyssey_eth_h48qual_corrected_tabm_20260811_contract.md`
- 리서치 근거: `docs/eth_omega4_6_1_accuracy_research_ideas_20260811.md` — 제안 방향 3-1(앙상블 불일치)/3-3(exit head)은 계약에 반영, 3-2(PLE)는 기존 구현, 3-4(레짐별 threshold)는 계약 미해결 이슈 4로 남음
- 이 리소스 레지스트리 (자기 자신)
- 실험 문서 22개, 전부 `docs/experiments/eth_*` — 각각 계약 문서 본문에 인라인 링크되어 있어 여기서 중복 나열하지 않음(계약 문서 참고). **최신 3개 중 갱신 상태 확인됨(2026-08-12)**: `eth_h48qual_direction_confidence_calibration_fullwindow_recheck_20260812.md`는 계약 문서 "데이터 구간 정의" 절 함정 1에 이미 반영됨(2024년 포함 전체구간 재확인 완료 — 아래 스크립트 항목 참고). `eth_zig075_final15_vs_jmlam4_vs_live_comparison_20260812.md`, `eth_zig075_jmlam4_candidate_confidence_echo_calibration_check_20260812.md` 2개는 아직 계약 문서에 미반영(병행 세션 작업 중으로 추정, 계속 확인 필요).
- ⚠ 범위 불확실: `docs/experiments/eth_signature_mmd_regime_sizing_overlay_20260811.json` — h48qual 13-피쳐 계약과 용어(`cvp_regime`, `ou_halflife`)를 공유하지만 계약 문서·리서치 근거 문서 어느 쪽도 인용하지 않고, 자체 상태가 `"OPENED -- blocked on IG0"`(대응 스크립트 없음). Odyssey 소속인지 별도 리서치 라인인지 미확인.

### 스크립트 (44개, 전부 `scripts/` 하위)

**진단 `diagnose_*` (13개)** — 순위상관·편향·calibration·앙상블 불일치 실증 진단:
`eth_h48qual_basis_rank_correlation`, `eth_h48qual_direction_confidence_calibration`, `eth_h48qual_direction_confidence_calibration_fullwindow_recheck`🆕, `eth_h48qual_ensemble_disagreement`, `eth_h48qual_longshort_winrate_h384`, `eth_h48qual_onchain_capmvrv_detrend`, `eth_h48qual_onchain_rank_correlation`, `eth_h48qual_quality_for_action_rank_correlation`, `eth_h48qual_quality_threshold_sweep`, `eth_h48qual_short_only_vs_always_short`, `eth_h48qual_train_rows_trend_mismatch`, `eth_zig075_quality_trend_bias`, `eth_zigzag_swing_asymmetry` (전부 `_20260811.py`, 🆕만 `_20260812.py`)

**검증/대조군 `verify_*` (7개)** — always-short 기준선, oracle 게이트, GBM 실전신호 검증:
`eth_h48qual_always_short_baseline_h384_v2`, `eth_h48qual_always_short_baseline_h48orig`, `eth_h48qual_always_short_baseline_live_bundle`, `eth_h48qual_quality_gbm_final12`, `eth_h48qual_quality_gbm_rel11`, `eth_h48qual_quality_oracle_gate`, `eth_zig075_final16_features` (전부 `_20260811.py`)

**라벨/피쳐 빌드 `build_*`, `pad_*` (5개)**:
`build_eth_h384_conservative_triple_barrier_label_20260811.py`, `build_eth_zigzag_exit_layer_labels_20260810.py`, `build_eth_zigzag_exit_layer_richfeatures_softlabel_20260810.py`, `pad_eth_h384_conservative_labels_to_zigzag_timestamps_20260811.py`, `pad_eth_h48_conservative_orig_labels_to_zigzag_timestamps_20260811.py`

**감사/재스크리닝 `audit_*`, `rescreen_*`, `knockoff_*` (5개)**:
`audit_eth_zig075_knockoff_mrmr_20260811.py`, `audit_eth_zig075_oracle_feature_analysis_jmredesign_20260811.py`, `knockoff_h48qual_only63_vs_zigzag_20260811.py`, `rescreen_eth_h48qual_quality_regression_dedup_20260811.py`, `rescreen_eth_h48qual_quality_regression_pool201_20260811.py`

**학습/튜닝 `train_eval_*`, `tune_*` (4개)**:
`train_eval_omega4_3head_parent72_eth_h48qual_final12_h384_20260811.py`, `train_eval_omega4_3head_parent72_eth_h48qual_final12_h48orig_20260811.py`, `train_eval_omega4_3head_parent72_eth_zig075_regime_jmredesign_final15_20260811.py`, `tune_eth_h48qual_final12_h384_20260811.py`

**스윕 `sweep_*` (3개)**:
`sweep_h48qual_barrier_floor_horizon_zigzag_match_20260811.py`, `sweep_h48qual_barrier_floor_zigzag_match_20260811.py`, `sweep_h48qual_horizon_wide_20260811.py`

**외부 다운로드 `download_*` (2개)**: `download_coinmetrics_onchain_eth_20260811.py`, `download_okx_eth_hourly_klines_20260811.py`

**차트 `chart_*` (3개)**: `chart_eth_h48qual_oracle_oos_1week_20260811.py`, `chart_eth_zig075_oracle_oos_1week_20260811.py`, `chart_eth_zigzag_exit_oracle_oos_1week_20260811.py`

**전체구간 재생성 (1개)**: `regenerate_eth_h48qual_fullwindow_train_predictions_20260812.py` — [[odyssey_eth_h48qual_subproject]]의 2026-08-12 "학습구간 갭"(`train_predictions_qXXX.csv`가 2025년 1~9월 서브셋뿐, 2024년 누락) 발견 후속. 라이브 번들로 순수 추론(재학습 아님) 재실행해 진짜 전체구간(183,936행, report.json과 diff=0) 예측을 재생성 — **완료, 결론 불변(오히려 강화)**: 2024 단독 구간이 2025 1~9월 단독 구간과 거의 동일한 클래스별 과신/과소신 패턴을 독립 재현. 이 재생성이 쓴 진짜 학습 시점(2026-06-30) 입력 피쳐 소스: `tmp/causal_regen_20260516/omega_clean_regime_only_24_25_inputs_20260629/`(레포 기본 피쳐 소스와 다름, 신규 확인된 경로). 결과: `docs/experiments/eth_h48qual_direction_confidence_calibration_fullwindow_recheck_20260812.md`, 계약 문서 "데이터 구간 정의" 절 함정 1에 반영됨.

**운영 (1개)**: `scripts/ops/run_h48qual_ensemble_disagreement_5seed_20260811.sh` — GPU 서버 5시드 재학습 런처, [[reference_dev_server_handoff]]로 실행

🆕 = 이번 스냅샷 작성 도중(2026-08-12) 새로 나타남, 아직 계약 문서에 미반영 — "동시성 참고" 콜아웃 참고.

**미커밋 스크립트 소재 문제 (계약 문서 미해결 이슈 7과 연결)**: FINAL12 분류기준 mRMR/knockoff dedup 스크립트 원본은 위 44개 어디에도 없다 — 이 레포가 아니라 다른(이전) 세션의 OS `/tmp` scratchpad(`mrmr_final_v2.py` 등 8개 후보 파일)에만 있어 지금 상태로는 독립 재현 불가능. 아래 "모델 아티팩트" 절의 백업 위험 참고.

## 라벨/예측 데이터

| 리소스 | 위치 | 커버리지 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| zigzag_action 원본 라벨 CSV | `tmp/causal_regen_20260516/zigzag_action_labels_20260531/zigzag_action_labels_{2024,2025,2026}.csv` | 2024-01-01 ~ (연도별 파일) | `direction_head`(h48qual/zig075 공통) 학습 라벨의 1차 소스. 스윙 단위 재구성(Calmar/edge 진단 등)에도 사용 | 활성 | ⚠ `zigzag_segment_id`가 연도별로 -1/0부터 재시작 — concat 후 `groupby(["year","zigzag_segment_id"])`로 반드시 연도를 키에 포함할 것. 2026-08-11 이 버그로 스윙 표본이 절반(918/925)으로 잘못 집계됐다가 수정(1718/1725)됨. 상세: [[feedback_zigzag_segment_id_year_collision]] |
| `train/validation/oos_predictions_qXXX.csv` (번들별) | 각 모델 산출 디렉토리 | 파일명과 달리 `train_predictions`는 2025-01~09(9개월, 전체 21개월의 43%)만 포함 — 2024년 0건 | h48qual/zig075 라이브 번들 예측값(`quality_for_action`, `dir_action` 등) 조인용 1차 소스 | 활성, 주의 | ⚠ "학습구간 전체"로 오인 금지. 전체 2024+2025 필요하면 원본 zigzag CSV를 직접 로드할 것. 계약 문서 "데이터 구간 정의" 절에 표준 구간 표 있음 |
| 라이브 번들 아티팩트 `true_3head_tabm_bundle.pt` + `report.json` | h48qual/zig075 각 번들 디렉토리 | 102 features, 실제 라이브 가중치 | always-short 대조, 게이트 편향 재현 등 "재설계판이 아닌 실제 라이브 재현" 검증 | 활성 | 게이트 통과율이 낮아(0.7~2.5%) 거래수 9~29건뿐 — 통계적 확인 수준으로만 취급 |

## 모델 아티팩트 (번들)

`true_3head_tabm_bundle.pt`(모델 가중치) 자체의 정확한 경로/크기 — 위 "라벨/예측 데이터" 표 3번째 행이 요약만 하던 것을 여기서 확정한다.

### 라이브 프로덕션 번들 (Odyssey가 만든 게 아님 — 대조·재현 대상으로만 읽음, 수정 금지)

| 구성요소 | 경로 | 크기 | 최종 수정 |
|---|---|---:|---|
| h48qual bundle | `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630/true_3head_tabm_bundle.pt` | 1.3M | 06-30 |
| h48qual sidecar | `tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_plus_t12_livepass_h48qual_q050_precomputed_20260630/risk_sidecar.pkl` | 120K | 06-30 |
| zig075 bundle | `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629/true_3head_tabm_bundle.pt` | 1.3M | 06-29 |
| zig075 sidecar | `tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_plus_t12_livepass_zig075_q075_precomputed_20260630/risk_sidecar.pkl` | 120K | 06-30 |

경로 근거: `trading_bot.py:3865-3868` → `runtime_config.py:348-363`의 `FINAL_GOVERNOR_OMEGA4_6_1_*` 환경변수(레포 전체 `.env`/yaml/json/sh에서 오버라이드 없음 확인 — 하드코딩 기본값이 실제 라이브에서 사용됨). `docs/model_contracts/CURRENT_LIVE_MANIFEST.json`(07-29 스냅샷, 2주 지나 참고용)이 동일 경로·크기·sha256로 교차 확인됨. 각 번들 디렉토리에는 `models/`(bull/bear/chop 3개 regime expert) + 위 `.pt` + `report.json` + `train/validation/oos_predictions_qXXX.csv`가 들어있음(h48qual 디렉토리 총 76M, zig075 총 34M).

### 🎯 Odyssey 실제 "테스트 모델" 후보 — JM 리짐 재설계 라인 (위 라이브 번들과는 다른 계열)

**이게 사용자가 말하는 "오디세이 테스트 모델"이다** — 위 라이브 프로덕션 번들(HMM 레짐 라우팅, 6월 학습)이 아니라, JM 리짐 재설계(`data/ensemble/reports/jm_redesign_20260810/`, 별개 프로젝트, git 커밋 `7eb2dd1`)를 라우터로 쓰는 zig075 계열. h48qual 라인에는 아직 이 계열의 ETH 번들이 없음(`*jmredesign*`으로 잡히는 건 전부 BTC).

| 후보 | 경로 | 크기 | 학습 시점 | 상태 |
|---|---|---:|---|---|
| jmlam4 (JM 레짐, 피쳐 제한 없음) | `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zig075_regime_jmlam4_20260809/` | 168M | 2026-08-09 | 완전 학습됨, q040~060 6개 threshold 예측 export 보유 |
| **final15 (JM 레짐 + FINAL15 mRMR/knockoff 큐레이션 피쳐)** | `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zig075_regime_jmredesign_20260810_final15/` | 168M | **2026-08-12 00:42, 시드 260620 1개, dev CPU (`--epochs 4 --max-train-rows 30000`)** | **오늘 처음 학습됨** — `scripts/train_eval_omega4_3head_parent72_eth_zig075_regime_jmredesign_final15_20260811.py`가 하루 넘게 미실행 상태였다가 방금 실행됨 |

**final15 헤드라인 결과**: direction_head의 confidence 비대칭(이 조사 전체의 핵심 문제)이 학습구간 기준 +0.048(구 라이브) → 거의 0(+0.0008)으로 붕괴하고, 게이트가 숏 쏠림을 증폭이 아니라 완화(-1.1~-4.1pp)하는 쪽으로 부호 자체가 뒤집힘. **단, 아직 단일 시드·PnL 미검증**(confidence 대칭성 개선이 곧 수익성 개선을 뜻하진 않음) — N≥5 다양한 시드 재현과 always-short 대조가 다음 단계. 상세: `docs/experiments/eth_zig075_final15_vs_jmlam4_vs_live_comparison_20260812.md`, [[odyssey_eth_h48qual_subproject]] 2026-08-12 최신 업데이트.

### 격리 검증용 실험 번들 (Odyssey가 직접 재학습, 전부 `tmp/causal_regen_20260516/` 하위)

| 세트 | 시드 (N) | 용도 | 총 크기 | `.pt` 보존 |
|---|---|---|---:|---|
| h48orig 5-seed | 260620/481003/26611/903174/155827 | 라이브 레시피(48bar) 그대로 always-short 대조 | 55M | ✗ 예측 CSV만 |
| h384 v2 15-seed | 위 5개 + 44452/51724/179660/240382/375044/378518/692713/711841/750878/821662 | FINAL12+384bar 격리검증 — OOS 통계 유의성 확보(이후 always-short 전패로 무효화됨) | 165M | ✗ 예측 CSV만(seed 260620만 validation 포함) |
| h384 v2b 5-seed (서버 GPU 재학습, `handoff.sh pull`로 회수) | h48orig와 동일 5개 | 앙상블 불일치(candidate C) 진단 — `.mean(dim=1)` 풀링 전 k=8 멤버 출력 필요 | 837M | ✓ 유일하게 `.pt` 보존 |

15-seed 세트는 레포의 Seed-Diversity Ensemble Promotion Gate(무작위 추출, 고정 간격 증가 금지)를 충족하는 진짜 다양한 시드임 — 참고용, 이 자체가 승격 주장은 아님.

### ⚠ 백업 위험 — 위 전부 git 추적 밖

`tmp/`와 `data/*`는 `.gitignore`에서 예외 없이 전부 제외됨(2026-08-12 직접 확인) — 위 라이브 번들·실험 번들·예측 CSV **전부가 git으로 백업되지 않는 로컬 디스크 전용 자산**이다.

**해소됨 (2026-08-12)**: 회귀 재스크리닝(REL11)이 의존하던 `fa_features.parquet`(127M)와 mRMR/knockoff dedup 스크립트 원본(`h48qual_knockoff_mrmr.py`, `mrmr_final_v2.py`, `zig075_knockoff_mrmr.py`)이 세션 scratchpad(`/tmp/claude-1000/.../f6f0940b-.../scratchpad/`)에만 있던 소멸 위험을 `tmp/eth_h48qual_fa_features_backup_20260812/`로 백업해 해소했다(`fa_features.parquet`, `fa_labels.npz`, `fa_meta.json` + 스크립트 3종). 여전히 git 추적 밖(레포 tmp/ 자체가 gitignore 대상)이지만 최소한 세션 scratchpad보다는 안전한 위치. 계약 문서 미해결 이슈 7의 "독립 재현 불가능" 표현은 이제 "가능하나 git 백업은 아님"으로 갱신 필요.

## 라이브 수집 duckdb (새 데이터소스 후보)

| 리소스 | 위치 | 커버리지 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| microstructure.duckdb | `data/live/microstructure.duckdb` | 2026-05-03 ~ | candidate 2: 마이크로구조 toxicity/queue/absorption/spoofing (`microstructure_scanner.py`, depth20@100ms+aggTrade) | 인프라 확인됨, **미착수** | VAL(2025-10~12)/OOS(2026-01~02)와 전혀 안 겹침 — "재학습 없이 조인만"이 불가능, 새 구간 causal inference로 재분류 필요 |
| tail_risk.duckdb | `data/live/tail_risk.duckdb` | 2026-05-03 ~ | candidate 1: 청산 이벤트 스트림 (`tail_risk_interceptor.py`, Binance `@forceOrder`) | 인프라 확인됨, **미착수** | 위와 동일 사유 |
| polymarket.duckdb | `data/live/polymarket.duckdb` | 2026-04-21 ~ 04-30 (9일치만) | candidate 3: Polymarket 예측시장 (`polymarket_engine.py`) | 인프라 확인됨, **미착수** | 커버리지가 가장 짧음 — 위와 동일 사유로 더 심함 |

## 외부 다운로더 / API

| 리소스 | 위치 | 커버리지 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| CoinMetrics ETH 온체인 CSV | `scripts/download_coinmetrics_onchain_eth_20260811.py` | 587일치, 2025-01-01 ~ | candidate 6: ETH on-chain 6개 무료 지표 | **검증 완료 — 부정 결과** | `CapMVRVCur`는 `corr(price)=0.95~0.97`로 심각한 가격추세 오염 확인(detrend 시 신호 붕괴, disqualifying threshold 0.561 대비 훨씬 심함). 나머지 5개 지표 무상관/부호불안정. 신규 raw-level 피쳐는 학습 전 반드시 오염도부터 체크 — [[feedback_raw_feature_price_trend_contamination]] |
| OKX 1시간봉 ETH perp 클라인 | `scripts/download_okx_eth_hourly_klines_20260811.py` | 2025-01 ~ 현재 | candidate 4 후반(price basis = binance_close vs okx_close) | **검증 완료 — 부정 결과** | 오염도는 낮음(`corr(price)`≤0.30)이나 h48orig vs h384 라벨 변형 간 순위상관 부호가 불일치해 기각 — 오염도 통과와 별개로 "라벨 변형 간 방향 일관성"도 독립적 신뢰성 기준임을 확인 |
| OKX `funding-rate-history` API | ccxt + 네이티브 REST | 보존기간 ~1개월 (`since` 사실상 무시) | candidate 4 전반(펀딩 스프레드) | **인프라 차단** | VAL/OOS 백필 불가능 — 검증 자체가 불가능해 이 절반은 테스트되지 못함 |
| Deribit 옵션 API | `get_book_summary_by_currency`(현재 스냅샷만), `get_instruments(expired=true)`(만기 계약 메타데이터만) | 과거 시점 조회 불가 | candidate 5: 옵션 스큐/GEX | **인프라 차단** | 특정 과거 시점의 옵션 체인을 조회하는 엔드포인트 자체가 없음 — VAL/OOS 백필 불가능, 재구성하려면 훨씬 큰 프로젝트 필요 |
| F4-C altdata collector | `scripts/run_f4c_altdata_collector.py` | 2026-08-10 ~ (실측 수집 중) | 거래소간 펀딩 스프레드 + Fear&Greed | 레포 전체 소비처 0건 확인, **미착수** | VAL/OOS 구간과 미중첩이라 duckdb 3종과 같은 사유로 아직 착수 안 함 |
| CoinGlass 청산/OI 히스토리 API | 미계약(요금제만 조사) — `docs.coinglass.com` | 플랜×인터벌별 상이: 일봉만 all-time, 시간봉대 최대 720일(Professional $699/월), 5분봉 최대 60일 | candidate 1·2(청산/OI) 백필 시도 | **조사 완료(결제 안 함) — 구조적 한계로 사실상 폐기** | 유료 등급을 올려도 TRAIN(2024-06~) 시작을 못 채움 — Professional 720일도 82일 부족, 5분봉은 OOS(2026-01~02)에도 못 닿음. 상세: `docs/experiments/eth_alt_data_source_feasibility_check_20260812.md` "추가 업데이트(2026-08-12)" 절 |
| LunarCrush 소셜/센티먼트 API | 미계약 — `github.com/lunarcrush/api`(기술스펙 확인), `lunarcrush.com/pricing`(요금제, SPA라 오늘 접근 실패) | **5분봉 상품 자체가 없음**(전체 시계열 엔드포인트가 `bucket=hour\|day`만 지원); 시간봉 "전체 히스토리" 주장은 ETH 토픽 실제 커버 시작일 미확인(계정 필요) | 감성 피쳐 후보(비검증, `galaxy_score`/`alt_rank`/`sentiment`/`social_dominance` 등) | **조사 일부 완료 — 5분봉은 구조적으로 폐기, 시간봉은 계정 없인 보류** | Dune과 동일하게 계정 가입이 전제조건. 요금제 정확한 $ 티어는 SPA라 오늘 미확인. 상세: 위 문서 "추가 업데이트(LunarCrush)" 절 |

## 인프라

| 리소스 | 위치 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|
| GPU 서버 핸드오프 | `scripts/ops/handoff.sh` | dev 머신은 GPU 없음(`torch.cuda.is_available()` False) — 5시드 재학습(앙상블 불일치 진단 등)을 서버에서 실행 | 활성, ~8분/5시드 완료 확인 | 사용법: [[reference_dev_server_handoff]] |

## 미검증 남은 후보

- candidate 7 (hazard/competing-risks relabeling): 새 데이터소스가 아니라 라벨 재구성 축 — 1~6 중 하나가 신호를 보이기 전까지 조건부 보류
- candidate 8 (전체 L2/L3 depth + 학술 VPIN): 가장 비싼 인프라, candidate 1(청산 스트림, 현재 미착수)에 조건부

전체 후보 랭킹과 검증 비용 순서는 `docs/experiments/eth_h48qual_quality_new_data_source_research_20260811.md` 참고.
