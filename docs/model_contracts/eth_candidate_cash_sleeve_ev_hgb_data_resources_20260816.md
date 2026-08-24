# ETH 캐시 슬리브 EV-HGB 후보 — 데이터 및 리소스 관리 (2026-08-16)

이 문서는 이 후보(`docs/model_contracts/eth_candidate_cash_sleeve_ev_hgb_contract_20260816.md`)에서
실제로 만지거나 검토한 모든 데이터 소스/리소스를 한 곳에 모은 목록이다.

**새 리소스를 만질 때마다 그 턴에 행을 추가/갱신할 것** — 나중으로 미루지 않는다. 상태 값 컨벤션:
`활성`, `인프라 확인됨-미착수`, `인프라 차단`, `검증 완료 — 긍정 결과`, `검증 완료 — 부정 결과`.

## 이식 원본 / 재사용 대상 (읽기 전용 참고, 전부 미수정)

| 리소스 | 위치 | 커버리지 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| BTC 캐시 슬리브 메커니즘 스펙 | 오케스트레이팅 세션이 사전 조사해 과업 지시에 그대로 제공(원본 스크립트는 삭제됨, 재유도 아님) | - | 이식 스펙(라벨/시뮬레이션 로직, 고정 리스크 프로파일, ev_min) | 활성(참고용, import 불가) | `_simulate_label_detail`은 삭제된 스크립트라 스펙만으로 재구성, import 시도 안 함 |
| Odyssey4 잠금 베이스라인 replay 엔진 | `scripts/research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814.py`(`greedy_replay_entry_veto`, `_attach_veto_mask`) | val/oos_q1/oos_q2/2025q1~q3 | PRIMARY의 CASH 상태를 실제 계정 렛저에서 그대로 읽기 위한 엔진 | 활성 | `docs/model_contracts/odyssey4_eth_entry_veto_baseline_contract_20260814.md`의 G0 베이스라인 그 자체(h48qual regime-aware exit guard + zig075 SHORT 진입거부 포함) |
| h48qual regime-aware exit guard + 탐지기 | `scripts/research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814.py`(`prepare_regime_aware_components`, `build_detector`, `_detector_mask_for_frame`) | 위와 동일 | 위 replay 엔진이 요구하는 컴포넌트 준비/탐지기 | 활성 | 신규 자유변수 0개, 재계산된 threshold가 잠긴 값(0.8025793651)과 정확히 일치함을 스크립트 실행 로그로 확인 |
| 6-윈도우 정의/로더 | `scripts/eth_omega461_multiwindow_confirmation_gate_20260814.py`(`WINDOW_DEFS`, `load_all_windows`) | val=2025-10-01~12-31, oos_q1=2026-01-01~03-31 등 6개 | VAL/OOS 프레임+예측 CSV 정렬 로딩 | 활성 | VAL 시작이 CLAUDE.md 일반 기본값(09-01)과 다름 — h48qual/zig075 parent의 `SPLIT_TS=2025-10-01` 때문(실험 문서 참고) |
| ETH 5분봉 OHLCV + WIDE24 오버레이 피처 | `data/splits/year_oos/training_features_2025.csv`, `training_features_2026_rebuilt.csv`(+ 대응 WIDE24 CSV, `research_eth_omega461_exit_sweep_20260721.BASE_2025/2026`/`WIDE24_2025/2026` 경유) | 2025-01-01~2026-06-30 (이 후보는 2025-10~2026-03만 사용) | PRIMARY replay와 폴백 오라클 시뮬레이션 둘 다의 open/high/low/close/timestamp 소스 | 활성 | 신규 다운로드/가공 없음, 기존 파이프라인 그대로 재사용 |
| h48qual/zig075 예측 CSV(validation/oos_predictions_qXXX) | `tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/{h48qual,zig075}/` 하위 | val/oos_q1 | PRIMARY 컴포넌트의 dec/side/take_profit/stop_loss/exit_head 예측 | 활성 | 기존 아티팩트 재사용, 재학습 없음 |
| ETH fee/slip 상수 | `scripts/train_eval_omega1_2_tabm_diffusion_risk_20260603.py`(`FEE_RATE=0.0005`, `SLIP_RATE=0.0002`, `_load_fee_slip()`) | - | 정상 비용(폴백 시뮬레이션에서 3배로 스트레스) | 활성 | 이 계보의 다른 모든 ETH Omega4.6.1/Odyssey 스크립트와 동일 소스 |
| h48qual/zig075 raw ThreeHeadTabM 예측 → `tabm_*` 트레이스 피처 변환 | `scripts/train_eval_omega1_2_tabm_diffusion_risk_20260603.py`(`_source_state`) | val | Stage 1의 "primary-trace" 피처 그룹(43개, h48qual_/zig075_ prefix) — BTC 라이브 어댑터와 동일 `tabm_` 명명 | 활성 | Stage 1에서 처음 재사용, 미수정 |
| purged K-fold + embargo 유틸리티 | `core/event_label_engine.py`(`purged_kfold_splits`) | - | Stage 1의 purged CV(AFML Ch.7 purge+embargo) | 활성 | Stage 1에서 처음 재사용, 미수정. `t1_idx`는 각 이벤트의 보수적 상한(`event_idx+192`)으로 구성 |
| BTC 캐시 슬리브 라이브 어댑터(피처 이름/스케일링 참고) | `trading_bot_modules/omega1_2_3_cash_sleeve.py`(`Omega123CashSleeveAdapter`) | - | Stage 1 피처 3그룹(market/primary-trace/cash-state-history) 설계·명명·tanh 스케일링 참고(읽기 전용, import 안 함) | 활성 | 라이브 코드 미접촉·미수정, 참고만 함 |

## 신규 산출물

| 리소스 | 위치 | 커버리지 | 용도 | 상태 | 주의사항 |
|---|---|---|---|---|---|
| cheap_gate 스크립트 | `scripts/research_eth_candidate_cash_sleeve_ev_hgb_cheap_gate_20260816.py` | val(2025-10-01~12-31), oos_q1(2026-01-01~03-31) | PRIMARY CASH 상태 추출 + LONG/SHORT 폴백 오라클 시뮬레이션 + ev_min 게이팅 통계 | 검증 완료 — 헤드룸 있음(결정적 부정 아님) | G0 정합성 체크(no_gate pnl/mdd/trades) val/oos_q1 둘 다 정확히 일치, 재학습/GPU 없음 |
| cheap_gate 리포트 + bar별 CSV | `tmp/causal_regen_20260516/eth_candidate_cash_sleeve_ev_hgb_cheap_gate_20260816/report.json`, `cash_sleeve_oracle_bars_{val,oos_q1}.csv` | 위와 동일 | 윈도우별/합산 통계, bar별 long_net/short_net/reason 원본 | 활성 | `docs/experiments/eth_candidate_cash_sleeve_ev_hgb_20260816.md`에 요약 |
| IC-check 스크립트 | `scripts/research_eth_candidate_cash_sleeve_ev_hgb_ic_check_20260816.py` | val, oos_q1 | 학습 전 저비용 확인: 25개 causal 피처 vs 오라클 타겟 Spearman IC(가격오염/노이즈바닥 포함) | 검증 완료 — 애매한 중간지대(결정적 부정 아님) | 실험 문서 "후속 조사" 절에 전체 표/해석 |
| IC-check 출력 CSV/JSON | `tmp/causal_regen_20260516/eth_candidate_cash_sleeve_ev_hgb_ic_check_20260816/{ic_results.csv, val_oos_consistency.csv, ic_check_summary.json, feature_target_join_{val,oos_q1}.csv}` | 위와 동일 | 피처×타겟×윈도우 전수 IC/CI/셔플null/가격오염, 유망조합(12/50) 목록 | 활성 | Stage 1 피처 선택(market 25개)의 직접 근거 |
| Stage 1 학습 스크립트 | `scripts/research_eth_candidate_cash_sleeve_ev_hgb_train_stage1_20260816.py` | val(2025-10-01~12-31)만(OOS-Q1 미접촉) | 실제 `long_model`/`short_model` HGB 학습, purged 5-fold CV, fold당 30회 라벨 순열 대조군, 사전등록 결합 기준 검증 | **검증 완료 — 결정적 부정(FAIL)** | `docs/model_contracts/research_line_registry.json`의 `eth_candidate_cash_sleeve_ev_hgb_stage1_train_20260816` 항목으로 CLOSED 기록 |
| Stage 1 출력 | `tmp/causal_regen_20260516/eth_candidate_cash_sleeve_ev_hgb_train_stage1_20260816/{report.json, oof_predictions.csv, fold_purge_diagnostics.csv, permutation_null.csv}` | 위와 동일 | 전체 지표(real+null), fold별 purge 간격 검증, OOF 예측 원본, 순열 null 30회 분포 | 활성 | 실험 문서 "Stage 1" 절에 전체 표/해석 |

## 미검증 후보 / 보류

**없음 — 이 후보는 CLOSED다.** Stage 1(실제 EV-HGB 학습, purged/embargo CV, 라벨 순열 대조군)이
사전등록 결합 기준(순위 IC + 결정 관련 지표, 셋 다 순열 null 대비 z≥2.0)을 명백히 충족하지 못했다.
N≥5 시드 재현·fresh-forward walk-forward는 이 결과에 게이트되어 진행하지 않는다(계약 문서 "다음
단계" 참고).
