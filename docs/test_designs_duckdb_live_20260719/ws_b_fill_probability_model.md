# WS-B 테스트 설계도 — Maker 체결확률 모델 (Fill-Probability)

성격: 모델 학습 + shadow 검증. 라이브 반영은 shadow 4주 + 게이트 통과 후.
공수: 1단계 3~5일. 선행: WS-A(비용 하한 수치).
근거 논문: Arroyo et al. 생존분석([2306.05479](https://arxiv.org/abs/2306.05479)), KANFormer([2512.05734](https://arxiv.org/html/2512.05734)).

## 가설

- H-B1: 스냅샷 시점 LOB 상태(imbalance, microprice edge, spread)와 1m 마이크로 지표로
  "best bid/ask maker 주문의 60초 내 체결 여부"를 캘리브레이션된 확률로 예측할 수 있다 (AUC > 0.55 + Brier 개선).
- H-B2: p_fill 조건부 placement 정책(체결확률 낮으면 taker 폴백/관망)이
  무조건 maker 정책 및 MicroExec v1.5 adaptive increment(+0.086bps) 대비 경제적으로 우월하다.

## 입력 데이터

- `orderbook_decision_snapshots` (ETH, 11,626행): `imbalance_{1,5,10,20}`, `microprice_edge_bps`,
  `spread_bps`, `bid_qty_1/ask_qty_1`, `recorded_at_kst`
- `microstructure_1m` (ETH): `obi, taker_buy_ratio, shadow_queue_collapse, shadow_absorption_score,
  recent_trade_notional_5m, mark_price` — 스냅샷 시점 기준 **직전 완결 1m bar만** 조인 (인과 조인)
- 1m mark_price 경로: 라벨 생성용

## 라벨 정의 (1단계 — 1m 테이프 근사)

스냅샷 t에서 가상 maker 매수 주문을 best_bid에 냈다고 가정:

- **체결 라벨** `filled_60s`: t 이후 60초 내 1m bar low가 best_bid **미만**으로 내려가면 1
  (트레이드-스루 보수 기준. `low == bid` 터치는 큐 위치를 모르므로 체결로 치지 않음).
- **역선택 라벨** `adverse_bps`: 체결 시 `(mid(t+60s) − best_bid(t)) / mid(t) × 10⁴`.
  음수면 체결 직후 손해(역선택).
- 매도측 대칭 라벨 동일 생성. long/short 라벨 분포 비대칭 여부 보고.
- **근사 한계 명시**: 1m OHLC 근사는 실제 큐 소진과 다르다. 이 한계 때문에 1단계 결과의
  용도는 "2단계 진행 여부 판단 + shadow 후보 선정"으로 제한. 라이브 반영 판단은 shadow 성과로만.

## 테스트 절차

### T-B0. 라벨 sanity 게이트
1. `filled_60s` 기저율이 5%~95% 범위 내 (극단이면 라벨 정의 재검토).
2. `adverse_bps` 분포: 체결 표본 평균이 음수(역선택 존재)인지 확인 — 문헌과 일치해야 정상.
3. naive-join 검사: 스냅샷 타임스탬프보다 늦게 닫힌 1m bar가 피처에 들어갔는지
   자동 체크 (조인 후 `feature_bar_close <= snapshot_ts` 전수 검증). 위반 0건 필수.

### T-B1. 베이스라인 모델 학습 (딥러닝 금지 — 표본 ~11k)
1. 시간순 분할: train 2026-05-13→06-20 / val 06-21→07-05 / test 07-06→현재. purge gap 1일.
2. 모델: (a) 로지스틱 회귀, (b) LightGBM (max 깊이 제한, 표본 대비 과적합 방지).
3. 지표: AUC, Brier score, reliability curve(10분위 캘리브레이션), 상수 예측 대비 Brier 개선율.
4. **판정**: test AUC < 0.55 또는 Brier 개선 없음 → **H-B1 기각, 여기서 중단.**
   결과 기록 후 WS-E 데이터 누적 대기 (2단계 조건 충족 시 재시도).

### T-B2. 경제성 백테스트 (오프라인)
1. 정책 정의: `p_fill ≥ θ`면 maker 시도, 아니면 taker 폴백. θ는 **val 구간에서만** 선택.
2. 트레이드당 기대 비용:
   `E[cost] = p_fill×(−spread/2 + adverse_bps) + (1−p_fill)×(taker cost from WS-A)`
3. 비교 대상 (test 구간, 동일 스냅샷 집합):
   - B0: 항상 taker (WS-A 실측 비용)
   - B1: 항상 maker (무조건)
   - B2: MicroExec v1.5 adaptive increment 룰 (기존 인증치 +0.086bps 재현 포함)
   - B3: 본 모델 정책
4. 통계: day-block bootstrap(일 단위 재표집 5,000회)으로 B3−B2 차이의 CI.
   **판정**: B3 > B2가 t > 3 → H-B2 잠정 채택, shadow 진행. 아니면 기각 기록.

### T-B3. Shadow 검증 (온라인, 최소 4주)
1. 기존 MicroExec shadow 프로토콜 재사용: 실주문 없음, 의사결정·가상체결을
   신규 테이블 `microstructure.duckdb::fill_prob_shadow_v1`에 기록.
2. 기록 항목: 스냅샷 피처, p_fill, 정책 결정, 가상 체결 결과, 실현 adverse_bps.
3. 주간 체크: 라이브 p_fill 캘리브레이션 곡선 vs 오프라인 곡선 (드리프트 시 조기 중단).
4. **최종 판정**: 4주 후 day-block bootstrap로 B3−B2 > 0, t > 3, 그리고
   오프라인 test 대비 효과 크기 감쇠 < 50%. 통과 시에만 라이브 제안서 작성
   (반영 자체는 별도 게이트).

### T-B4 (2단계, 조건부). 생존분석 업그레이드
- **개시 조건**: WS-E 연속 10초 스냅샷 + 체결 테이프 3개월(≥2026-10) 누적.
- 라벨을 체결 여부 이진 → 체결시간 분포(생존함수)로 교체. Arroyo 구조(conv-transformer
  encoder + 단조 decoder) 참고하되 표본 규모에 맞춰 축소.
- 1단계와 동일한 T-B1→T-B3 게이트 반복. 1단계 모델이 베이스라인 B2를 대체.

## Kill 기준 요약

| 단계 | 중단 조건 |
|---|---|
| T-B0 | 인과 조인 위반 발견 (수정 전 진행 금지) |
| T-B1 | test AUC < 0.55 또는 캘리브레이션 개선 없음 |
| T-B2 | B2(기존 adaptive increment) 대비 우위 없음 (t ≤ 3) |
| T-B3 | 4주 shadow에서 효과 소멸 또는 캘리브레이션 드리프트 |

## 산출물

- `results/ws_b_label_sanity_YYYYMMDD.json`, `results/ws_b_offline_eval_YYYYMMDD.md`
- shadow 테이블 `fill_prob_shadow_v1` + 주간 리포트
- 실패 시: 기각 사유 문서 (다음 재시도 조건 명시)
