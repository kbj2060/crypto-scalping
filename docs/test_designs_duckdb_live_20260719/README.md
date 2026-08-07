# 실시간 DuckDB 데이터 활용 — 워크스트림별 테스트 설계도 (2026-07-19)

상위 설계: [`docs/duckdb_live_data_utilization_design_20260719.md`](../duckdb_live_data_utilization_design_20260719.md)

| 파일 | 워크스트림 | 성격 | 선행 조건 |
|---|---|---|---|
| [ws_a_cost_model_calibration.md](ws_a_cost_model_calibration.md) | 실행 비용 모델 보정 | Diagnostic (실증, 모델 없음) | 없음 — 즉시 실행 가능 |
| [ws_b_fill_probability_model.md](ws_b_fill_probability_model.md) | Maker 체결확률 모델 | 모델 학습 + shadow 검증 | WS-A 결과(비용 하한) |
| [ws_c_tail_risk_overlay.md](ws_c_tail_risk_overlay.md) | 청산/독성 리스크 오버레이 | 조건부 분포 검정 → 오버레이 | Step 1 통계 검정 통과 |
| [ws_d_parity_drift_monitor.md](ws_d_parity_drift_monitor.md) | 패리티/드리프트 모니터 | 운영 인프라 수락 테스트 | 없음 — 즉시 실행 가능 |
| [ws_e_data_flywheel.md](ws_e_data_flywheel.md) | 데이터 플라이휠 확장 | 수집 무결성/성능 수락 테스트 | 없음 — 즉시 실행 가능 |
| [ws_f_kronos_experiment.md](ws_f_kronos_experiment.md) | Kronos 파운데이션 모델 실험 | 연구 트랙 (단계별 kill 게이트) | 독립 |

## 공통 규칙 (모든 워크스트림에 적용)

1. **Fresh-Forward 규칙**: 성과 주장에 쓰이는 모든 평가는 bar-by-bar causal walk-forward.
   저장 원장/shadow_pnl replay는 diagnostic 전용, 승격 근거 금지.
2. **승격 게이트**: 라이브 반영은 Omega Artifact Integrity Promotion Gate 및
   기존 shadow 프로토콜(최소 4주)을 통과해야 한다.
3. **시간 분할 원칙**: 학습/검증/테스트는 시간순 분할 + 1일 purge gap.
   임계값·하이퍼파라미터 선택에 쓴 구간은 테스트 구간과 겹치면 안 된다
   (fold-overlap 사고 2회 전례 — threshold-선택 구간과 fold 중첩).
4. **통계 기준**: 경제적 효과 주장은 day-block bootstrap(일 단위 블록) 기반 t>3.
   1m 행 단위 t-stat은 시계열 상관으로 부풀려짐(트렌드 스캐닝 1m 사례) — 단독 사용 금지.
5. **다중 검정**: 지표×호라이즌 격자를 훑는 검정은 결과 전체를 보고하고 FDR(BH) 보정 적용.
   유의한 셀만 골라 보고하는 것 금지.
6. **BTC/SOL 데이터 금지**: 2026-10-14(3개월 누적) 전까지 BTC/SOL 테이블로 모델 학습 금지.
7. **결과 기록**: 각 테스트는 통과/실패와 무관하게
   `docs/test_designs_duckdb_live_20260719/results/` 아래 JSON+md로 결과를 남긴다.
   실패한 가설도 기록한다 (기각 이력이 이 프로젝트의 핵심 자산).
