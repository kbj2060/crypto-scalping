# WS-A 테스트 설계도 — 실행 비용 모델 보정

성격: **Diagnostic (모델 학습 없음, 순수 실증)**. 승격 게이트 무관.
공수: 1~2일. 선행 조건 없음.

## 가설

- H-A1: 백테스트의 고정 비용 상수(cost1/cost3)는 실측 스프레드+뎁스 기반 비용의
  시간대·레짐별 분포와 유의하게 다르다 (특정 구간에서 과소/과대 추정).
- H-A2: 실측 비용 함수로 재채점하면 Omega4.6.1/Sigma6 백테스트 PnL이
  의미 있게(±10% 이상) 달라지는 구간이 존재한다.

## 입력 데이터

| 소스 | 컬럼 | 용도 |
|---|---|---|
| `microstructure.duckdb::orderbook_decision_snapshots` | `recorded_at_kst, spread_bps, mid, bid_notional_{1,5,10,20}, ask_notional_{1,5,10,20}, imbalance_*` | 스프레드/뎁스 분포 |
| `microstructure.duckdb::microstructure_1m` | `ts, mark_price, recent_trade_notional_5m, taker_buy_ratio` | 변동성 레짐, 활동도 |
| `data/live/binance_execution_audit.jsonl` | 주문/체결 감사 로그 | 실제 체결가 vs mid 괴리 (표본 있으면) |
| 5m OHLCV (기존 피처 프레임) | realized vol | 레짐 버킷 정의 |

## 테스트 절차

### T-A0. 데이터 품질 게이트 (필수 선행)
1. `orderbook_decision_snapshots`: null률 컬럼별 < 1%, `recorded_at_kst` 단조성,
   `spread_bps <= 0` 행 비율 보고, `spread_bps > 50` 행은 수동 검사(이상치 vs 실제 이벤트).
2. **샘플링 편향 정량화 (핵심)**: 스냅샷은 의사결정 시점에만 기록되므로 조건부 표본이다.
   시간대별 스냅샷 수 분포 vs 균등 분포 비교(χ²), 스냅샷이 몰린 시간대의
   microstructure_1m 변동성 평균 vs 전체 평균 비교.
   → 편향이 크면(변동성 평균 괴리 >20%) 이후 모든 분위수에 "활동 구간 조건부" 라벨을 명시.
3. 통과 기준: 위 항목 전부 보고서에 기록. 편향 존재 자체는 실패가 아님(라벨링 의무만 발생).

### T-A1. 스프레드/슬리피지 분포 테이블 구축
1. 버킷: KST 시간대 4구간(00-06/06-12/12-18/18-24) × 변동성 3분위(5m realized vol 기준).
2. 버킷별 산출: `spread_bps` p50/p90/p99, 표본수.
3. 시장가 슬리피지 근사: 명목 $10k/$50k/$100k 주문이 depth 1→5→10→20 notional을
   소진한다고 가정한 선형 보간 walk (원시 레벨 부재로 근사임을 명시).
   버킷별 슬리피지 p50/p90 산출.
4. 최소 표본 규칙: 버킷당 n < 100이면 인접 버킷과 병합.

### T-A2. 기존 가정 대비 비교
1. 현재 백테스트 비용 상수(cost1/cost3의 bps 값)를 코드에서 추출해 명시.
2. 버킷별 실측 p50/p90과 비교 테이블 작성. `assumed < measured_p50`(과소) 또는
   `assumed > measured_p90`(과대) 버킷을 플래그.
3. 실제 체결 감사 로그 표본이 있으면: 체결가 vs 스냅샷 mid 괴리 분포를 근사 슬리피지와
   교차 검증 (표본 부족 시 "n<30, 참고용" 명시).

### T-A3. 백테스트 재채점 감도 분석
1. `cost(asset, hour_bucket, vol_regime)` 함수 구현 (조회 테이블).
2. Omega4.6.1(ETH/SOL/BTC), Sigma6 기존 백테스트를 고정 상수 대신 이 함수로 재채점.
   **주의**: 재채점은 저장된 트레이드의 비용 항만 교체하는 accounting 재계산이므로
   diagnostic이다. 신규 성과 주장 아님 (Fresh-Forward 규칙상 승격 근거 불가).
3. 산출: 전략별 PnL 변화율, MDD 변화, 비용 민감도 순위.

## 판정 기준

| 결과 | 판정 | 후속 조치 |
|---|---|---|
| 전 버킷에서 measured p50이 assumed ±20% 이내 | H-A1 기각 | 현행 상수 유지, 결과만 기록 |
| 일부 버킷 과소/과대 플래그 | H-A1 채택 | research 백테스트에 cost 함수 도입. 라이브 승격 수치는 기존 게이트 재실행 시에만 갱신 |
| 재채점 PnL 변화 ±10% 초과 전략 존재 | H-A2 채택 | 해당 전략의 취약 시간대/레짐 문서화, WS-C 사이징 감쇠 후보 구간으로 전달 |

## 산출물

- `results/ws_a_cost_calibration_YYYYMMDD.json` (버킷 테이블 + 플래그)
- `results/ws_a_cost_calibration_YYYYMMDD.md` (요약 + 감도 분석)
- `quant/cost_model_lookup.py` (조회 함수, research 전용 플래그 포함)

## 함정 목록

- 스냅샷의 의사결정 조건부 샘플링 편향 (T-A0-2에서 반드시 정량화).
- 뎁스 요약 기반 슬리피지는 근사 — 원시 레벨 저장(WS-E) 후 재검증 예정임을 명시.
- 77일 = 단일 레짐일 수 있음. 월별(05/06/07) 분할 재현성 체크를 모든 분위수에 병기.
