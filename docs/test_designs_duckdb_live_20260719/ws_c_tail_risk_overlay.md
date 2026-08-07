# WS-C 테스트 설계도 — 청산 캐스케이드/독성 리스크 오버레이

성격: 조건부 분포 검정(diagnostic) → 통과 시에만 사이징 오버레이 설계 → shadow.
공수: Step 1 2~3일. 근거: VPIN(Easley et al.), Bitcoin order-flow toxicity(2026), 2025-10 캐스케이드 해부.

## 설계 원칙 (전례 반영)

- 1m 모델의 외부 regime veto는 **실패**했다 (모델이 이미 아는 정보의 중복 베토).
  이번 대상은 청산 스트림을 전혀 보지 않는 라이브 5m/1h 스택이므로 정보 중복이 없음 —
  단, 이 주장 자체를 T-C2에서 검증한다 (기존 피처와의 상관 체크).
- 진입 베토가 아니라 **사이징 감쇠**부터. Futures Risk Sizing Contract 준수:
  감쇠는 margin_fraction에만 적용, TP/SL price-move 기준 불변, leverage 재곱 금지.
- Omega4.6.1은 6개월 24트레이드 → 트레이드 PnL로 오버레이 검증 **불가능**.
  검증은 조건부 수익률 분포(1m, 96k행)로 한다.

## 가설

- H-C1: tail-risk 상태(청산 불균형/aftershock/독성 상위 분위) 직후 N분간 수익률 분포는
  무조건부 분포보다 유의하게 나쁘다 (평균 또는 하방 꼬리).
- H-C2: 이 상태 신호는 라이브 스택의 기존 피처와 낮은 상관(정보 중복 없음)이다.
- H-C3: 상태 조건부 margin_fraction 감쇠가 백테스트 MDD를 낮추고 PnL 훼손은 제한적이다.

## 입력 데이터

- `tail_risk.duckdb::tail_risk_1m` (96,013행): `long_usd_1m, short_usd_1m,
  shadow_aftershock_prob, shadow_decay_half_life, shadow_risk_bucket, liq_event_count_1m`
- `microstructure.duckdb::microstructure_1m`: `shadow_toxicity_score, shadow_queue_collapse,
  mark_price`
- 라이브 스택 피처 프레임 (H-C2 상관 체크용): `decision_feature_frame*`

## 테스트 절차

### T-C0. 데이터 품질 게이트
1. `valid_liq_stream=false`/`ws_stale=true` 구간 비율 보고, 해당 구간 제외 규칙 확정.
2. 청산 이벤트 희소성 확인: `liq_event_count_1m > 0` 비율. 상태 정의가 표본 부족이면
   (상태 발생 < 200회) 분위 완화(q95→q90).

### T-C1. 조건부 분포 검정 (핵심 diagnostic)
1. 상태 정의 격자 (사전 등록 — 사후 추가 금지):
   - S1: `shadow_aftershock_prob ≥ q95`
   - S2: `|long_usd − short_usd| / (합)` 청산 불균형 ≥ q95 (분모 하한 필터)
   - S3: `shadow_toxicity_score ≥ q95`
   - S4: S1∧S3 (복합)
2. 전방 수익률: N ∈ {5, 15, 60}분, mark_price 로그수익률. 방향: signed(청산 불균형
   방향 조건부) + absolute(변동성).
3. 검정: 상태 vs 비상태의 (a) 평균 차이 — day-block bootstrap t, (b) 하방 꼬리 —
   전방수익률 p5 분위 차이, (c) 실현변동성 차이.
4. **다중 검정 처리**: 4상태×3호라이즌×3지표 = 36셀 전체를 보고, BH-FDR 10% 보정.
5. **재현성 분할**: 전체 기간을 반분(05-03→06-10 / 06-11→07-19)해 유의 셀이
   양쪽에서 같은 부호로 재현되는지 확인. 한쪽만 유의 → "레짐 의존" 라벨, 채택 불가.
6. **판정**: FDR 통과 + 반분 재현 셀이 1개 이상 → H-C1 채택, T-C2로.
   0개 → **여기서 중단**, 결과 기록, 3개월 뒤 데이터 2배 시점에 1회 재시도.

### T-C2. 정보 중복 검사
1. 채택된 상태 신호 vs `decision_feature_frame`의 전체 피처: 스피어만 상관 상위 10개 보고.
2. |ρ| > 0.5인 기존 피처 존재 시: 해당 피처 조건부로 T-C1 재실행 (부분 정보 기여 확인).
   조건부 유의성 소멸 → 중복 정보로 판정, H-C2 기각, 중단.

### T-C3. 사이징 감쇠 오버레이 백테스트
1. 규칙: 상태 활성 시 margin_fraction × α, α ∈ {0.0, 0.25, 0.5} (α는 val 반분에서만 선택,
   test 반분에서 1개 값만 평가 — 격자 전체를 test에 대는 것 금지).
2. 대상: Omega4.6.1 ETH/SOL/BTC + Sigma6 백테스트 재채점.
   트레이드 수가 적으므로 판정 지표는 (a) MDD 변화, (b) 상태 활성 구간에 걸린
   트레이드의 조건부 PnL, (c) 총 PnL 훼손률.
3. **판정**: MDD 개선 ∧ 총 PnL 훼손 < MDD 개선폭의 절반 → H-C3 잠정 채택.
   PnL 훼손이 더 크면 α 재선택이 아니라 **기각 기록** (test 재사용 금지).

### T-C4. Shadow 검증 (4주)
1. `tail_risk_interceptor.py`에 재보정 임계값을 shadow 모드로 주입
   (실주문 영향 없음 — BINANCE_ACCOUNT_ENABLED=False 상태 확인 후).
2. 기록: 상태 발동 시각, 가상 감쇠량, 이후 N분 실현 수익률.
3. 판정: 발동 구간 조건부 분포가 오프라인 T-C1과 부호 일치. 4주 표본으로 t>3은
   비현실적이므로 shadow 목표는 **방향 재현 + 발동 빈도 예측치 일치(±50%)**로 한정.
   라이브 반영은 그 후 별도 게이트.

## Kill 기준 요약

| 단계 | 중단 조건 |
|---|---|
| T-C1 | FDR+반분 재현 통과 셀 0개 |
| T-C2 | 기존 피처 조건부 유의성 소멸 (정보 중복) |
| T-C3 | PnL 훼손 > MDD 개선 효과 |
| T-C4 | 발동 빈도/방향 오프라인과 불일치 |

## 산출물

- `results/ws_c_conditional_dist_YYYYMMDD.json` (36셀 전체 + FDR + 반분 재현)
- `results/ws_c_overlay_backtest_YYYYMMDD.md`
- 채택 시: tail_risk_interceptor 재보정 제안서 (임계값 + α + 발동 빈도 예측)
