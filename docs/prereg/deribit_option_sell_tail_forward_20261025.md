# Deribit ETH 옵션 순 델타 매도 꼬리(z<−1) → 다음 4h 하락 — 전진 사전 등록 (2026-10-25, 전진 자료 보기 전)

## 왜
`deribit_option_flow_direction_20261025.md` 주 관문 불통과 뒤 **사후** 분해에서 z<−1 뒤 4h 평균이 검증 −39.9bp(79건)·시험 −13.4bp(56건)로 두 기간 같은 부호. 사후에 고른 꼬리이므로 새 자료로만 판정한다.

## 규칙(고정 — 주 시험 코드와 같은 정의)
- 블록·청산 뺀 ETH 역옵션 체결, BS 델타 × 테이커 부호 × amount, 4h 창 [t−4h, t) 합 → 직전 180칸 표준편차로 나눈 z.
- 사건 = z(t) < −1. 라벨 = 다음 4h 로그수익(close[t−1분] → close[t+4h−1분]).

## 전진 창·판정
- 창 2026-10-24 00:00 ~ 2027-03-23 23:59 UTC(151일, 사건 ~55건 예상). 판정일 **2027-03-24**, 그 전에는 코드가 거절(`--forward`), `forward_verdicts_20261025.py --status` 에 날짜만.
- 관문: 사건 평균 4h 수익이 같은 창 무작위 4h 칸(같은 개수, 2,000회) 평균보다 낮은 비율 > 95%(숏 상대 기준) 그리고 사건 ≥ 30.
- 보고만: 숏 비용 후(10bp) 건당 손익, 창 전체 평균.
- 자료: history.deribit.com(과거 체결 계속 제공) + data.binance.vision 일별 1분봉 — 서버 기록기 불필요.
- 실행: `python3 scripts/research_deribit_option_flow_direction_20261025.py --forward` (한 번만, 결과 tmp/deribit_optflow_20261025/forward_sell_tail_2027-03-24.json).
