# ETH conformal 하방-LCB 거부게이트 후보 — 데이터 및 리소스 관리 (2026-08-16)

이 문서는 이 후보(`docs/model_contracts/eth_candidate_conformal_downside_veto_contract_20260816.md`)에서 실제로 만지거나 검토한 모든 데이터 소스/리소스를 한 곳에 모은 목록이다.

**새 리소스를 만질 때마다 그 턴에 행을 추가/갱신할 것** — 나중으로 미루지 않는다. 상태 값 컨벤션: `활성`, `인프라 확인됨-미착수`, `인프라 차단`, `검증 완료 — 긍정 결과`, `검증 완료 — 부정 결과`.

## 이식 원본 (BTC clean_base, 읽기 전용 참고)

| 리소스 | 위치 | 커버리지 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| BTC v1.5 계약/결과 | `docs/model_contracts/clean_base_causal_sleeve_conformal_veto_v1_5_contract.md`, `docs/experiments/clean_base_causal_sleeve_conformal_veto_v1_5.md` | - | 원 계약 프레이밍 확인 | 인프라 확인됨-미착수 | **정정됨**: 이건 sleeve-add 전용 veto다, Odyssey에 없는 개념 — 계약 본문 "이식 원본 재조사" 절 참고 |
| BTC v1.5 구현(conformal 계산 로직) | `scripts/train_eval_clean_base_causal_sleeve_conformal_veto_v1_5.py` | `_calibration`/`_residual_q`(93~110행), veto 적용(147~230행 부근) | 잔차분위수 LCB 계산식의 1차 출처 | 인프라 확인됨-미착수 | sleeve 관련 코드(`base._choose_sleeve` 등)는 이식 대상 아님 — conformal 계산부만 참고 |
| BTC v1.3 causal trade editor(회귀모델 원본) | `scripts/train_eval_clean_base_causal_trade_editor_v1_3.py` | `EDITOR_FEATURES`(50행), `_future_path_stats`(110행), `_train_editor_model`(142행), `_predict_editor`(185행) | v1.5가 그대로 import해서 쓰는 HGB 회귀 5개(full/h6/h12/h24/adverse) 학습·예측 로직 — ETH conformal veto가 실제로 이식할 핵심 자산 | 인프라 확인됨-미착수 | `_future_path_stats`는 entry_idx/side/exit 규칙만 있으면 계산되는 단독-트레이드 시뮬레이션 — ETH conformal veto의 causal 라벨 생성에 그대로 응용 가능 |
| BTC v1.4 conformal downside filter(자매 후보) | `docs/model_contracts/clean_base_conformal_downside_filter_v1_4_contract.md` | - | veto 대신 shrink+조기청산 하는 대안 프레이밍 | 인프라 확인됨-미착수 | 계약 미해결 이슈 2 — veto판이 기각되면 이걸 다음으로 검토 |

## Odyssey4 상속 자산 (그대로 재사용, 미변경)

| 리소스 | 위치 | 커버리지 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| Odyssey4 causal replay 하네스 | `scripts/research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814.py` | 6창 | G0 기준값, 컴포넌트 준비 함수 재사용 | 활성 | 이식 원본 그대로 — 수정 없이 import만 |
| 레짐 가드 컴포넌트 준비 | `scripts/research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814.py` (`prepare_regime_aware_components`, `build_detector`) | 6창 | h48qual/zig075 signal/quality/TP·SL/exit_head 컴포넌트 딕셔너리 생성 | 활성 | ETH conformal veto의 episode 라벨 생성이 이 컴포넌트 딕셔너리를 그대로 소비 |
| 사이징/exit_head 예측 | `scripts/train_eval_omega4_2_risk_sidecar_20260622.py`(`rs._predict_exit_prob_one`) | - | episode 단독-시뮬레이션 라벨 생성 시 exit_head 확률 재사용 | 활성 | - |
| Odyssey4 G0 기준선 | `docs/model_contracts/odyssey4_eth_entry_veto_baseline_contract_20260814.md` | 6창 | 비교 기준 | 활성 | 재계산 없이 그대로 인용 |

## ETH conformal veto 신규 실측 (2026-08-16, 코드로 직접 확인)

| 리소스 | 위치 | 커버리지 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| Episode 수 실측 | (일회성 인터랙티브 확인, 스크립트 미저장) | VAL/OOS-Q1/OOS-Q2, h48qual+zig075 | 학습 표본 크기 타당성 확인 — 계약 본문 표 참고(VAL h48qual 254 / zig075 789 episode) | 검증 완료 — 긍정 결과(표본 크기 충분) | **재현 필요**: 이 실측은 일회성 REPL 실행이라 저장된 스크립트가 없다 — 실제 구현 스크립트 작성 시 이 카운트를 재현하는 진단 단계를 반드시 포함해서 코드로 남길 것 |

## cheap_gate 산출물 (2026-08-16)

| 리소스 | 위치 | 커버리지 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| cheap_gate 스크립트 | `scripts/research_eth_candidate_conformal_veto_cheap_gate_20260816.py` | VAL만, h48qual/zig075 임계값 그리드 | quality_score 재게이팅 스윕 + G0 재현 | 검증 완료 — 결과 애매(아래 참고) | `_raise_quality_threshold`는 이 스크립트 전용 유틸 — h48qual@0.65/zig075@0.90이 각각 컴포넌트 완전비활성과 동일함을 별도 REPL로 교차검증(스크립트에 저장 안 됨, 재현 시 threshold=1.01로 재현 가능) |
| cheap_gate 리포트 | `tmp/causal_regen_20260516/eth_candidate_conformal_veto_cheap_gate_20260816/report.json` | G0 + 임계값 그리드 2종 | 원 수치 근거 | 검증 완료 | `docs/experiments/eth_candidate_conformal_veto_cheap_gate_20260816.md`에 표로 요약됨 |
| cheap_gate 결과 문서 | `docs/experiments/eth_candidate_conformal_veto_cheap_gate_20260816.md` | - | 전체 과정 + registry 충돌 진단 | 활성 | `research_line_registry.json`의 `global_exit_constant_tuning`(21회+ 실패)과의 겹침을 처음 발견·기록한 문서 |

## Episode 라벨 생성 산출물 (2026-08-16)

| 리소스 | 위치 | 커버리지 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| Episode 라벨 생성 스크립트 | `scripts/research_eth_candidate_conformal_veto_episode_labels_20260816.py` | 2025 Q1~Q3(학습) + VAL(캘리브레이션), h48qual+zig075 | quality-gate 통과 episode마다 causal 단독 시뮬레이션으로 `full`/`adverse` 라벨 생성 | 검증 완료 — **인접-episode 상관 심각**(아래 참고), 다음 단계 전 purge/embargo 필요 | OOS-Q1/OOS-Q2는 이 스크립트에서 로드조차 안 함 |
| Episode 라벨 데이터(parquet) | `tmp/causal_regen_20260516/eth_candidate_conformal_veto_episode_labels_20260816/episode_labels_<window>_<component>.parquet` | 8개 파일(4창×2컴포넌트) | HGB 학습 원재료 | 검증 완료 — 그대로 학습에 쓰면 안 됨(과신 위험) | purge/embargo/uniqueness weighting 적용 전까지 원본 그대로 사용 금지 |
| Episode 라벨 리포트 | `tmp/causal_regen_20260516/eth_candidate_conformal_veto_episode_labels_20260816/report.json` | 표본수/라벨통계/lag-1 자기상관/capped비율 | 원 수치 근거 | 검증 완료 | `docs/experiments/eth_candidate_conformal_veto_episode_labels_20260816.md`에 표로 요약됨 |
| Episode 라벨 결과 문서 | `docs/experiments/eth_candidate_conformal_veto_episode_labels_20260816.md` | - | 전체 과정 + 유효표본크기 진단 | 활성 | n_eff가 원표본의 1/6~1/8임을 처음 정량화한 문서 |

## Uniqueness 가중치 산출물 (2026-08-16)

| 리소스 | 위치 | 커버리지 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| Uniqueness 가중치 스크립트 | `scripts/research_eth_candidate_conformal_veto_uniqueness_weights_20260816.py` | 학습풀 3창+VAL, 두 컴포넌트 | concurrency 기반 가중치 계산 + lag-N ACF 진단, 라벨 parquet에 컬럼 추가 | 검증 완료 — **zig075 유효표본이 lag-1 근사보다 훨씬 작음**(학습풀 ~114건, VAL 창당 ~40건) | 라벨 parquet을 in-place로 덮어씀(uniqueness_weight 컬럼 추가) — 재실행 시 기존 라벨 재사용 가능, 재시뮬레이션 없음 |
| Uniqueness 리포트 | `tmp/causal_regen_20260516/eth_candidate_conformal_veto_uniqueness_weights_20260816/report.json` | 8개 창×컴포넌트 조합 | weighted_n/ACF 수치 근거 | 검증 완료 | `docs/experiments/eth_candidate_conformal_veto_uniqueness_weights_20260816.md`에 표로 요약됨 |
| Uniqueness 결과 문서 | `docs/experiments/eth_candidate_conformal_veto_uniqueness_weights_20260816.md` | - | 전체 과정 + 실현가능성 재평가 | 활성 | zig075 캘리브레이션 표본이 얇다는 한계를 처음 정량화 |

## HGB 학습 산출물 (2026-08-16)

| 리소스 | 위치 | 커버리지 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| HGB 학습 스크립트 | `scripts/research_eth_candidate_conformal_veto_hgb_train_20260816.py` | 학습풀 3창 pooled + VAL, 두 컴포넌트, 5시드 | full/adverse 회귀 학습 + VAL 가중상관/가중잔차분위수 | 검증 완료 — **zig075 방향 거꾸로(5/5 시드 음수), h48qual 미미(R²≈1%)** | `random_state`만으로는 HGB가 결정론적이라 첫 시도는 가짜 시드검증이었음 — uniqueness_weight 가중 부트스트랩 재표본으로 수정, 이 교훈이 스크립트 docstring에 기록됨 |
| HGB 학습 리포트 | `tmp/causal_regen_20260516/eth_candidate_conformal_veto_hgb_train_20260816/report.json` | 시드별 상관/분위수 전부 | 원 수치 근거 | 검증 완료 | `docs/experiments/eth_candidate_conformal_veto_hgb_train_20260816.md`에 표로 요약됨 |
| HGB 학습 결과 문서 | `docs/experiments/eth_candidate_conformal_veto_hgb_train_20260816.md` | - | 전체 과정 + 종합 판단(4단계 누적 증거) | 활성 | 이 세션에서 가장 결정적인(부정적) 결과 |

## 미검증 후보 / 보류 (축 종결 여부 결정 대기)

- **quality_score 재현 여부 진단**(cheap_gate가 새로 요구한 진단): 미착수 — 축이 계속될 경우에만 의미 있음.
- **VAL 임계값 그리드 선택, 포트폴리오 백테스트 통합, OOS 단일터치**: 미착수 — HGB 결과가 부정적이라 사용자가 계속 여부를 먼저 결정해야 함.
