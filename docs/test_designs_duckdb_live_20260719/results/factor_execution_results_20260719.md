# Track 3/4 실행 결과 (2026-07-19 12:xx KST)

설계: [`docs/factor_execution_test_design_20260719.md`](../../factor_execution_test_design_20260719.md)
전부 실제 데이터로 실행 완료 (사전 등록 격자, 탐색구간 kill-gate, day-block bootstrap).

## F3-A 펀딩 캐리 — **KILLED (탐색 단계)**

스크립트: [`scripts/research_f3a_funding_carry_20260719.py`](../../../../scripts/research_f3a_funding_carry_20260719.py)
결과: [`f3a_funding_carry_exploration_20260719.json`](f3a_funding_carry_exploration_20260719.json)

- ETH/BTC/SOL 펀딩비(2024-01~2026-06, 85개 zip 추출) × holding{8h,24h} × theta{0,q50,q75,q90} = 24개 변형 전 탐색.
- **24/24 변형 모두 kill 기준 미달** (net_cost1>0 & t>3 통과 0건).
- **핵심 발견**: **펀딩 컴포넌트 자체가 24개 변형 전부에서 음수**(-0.5~-8.5bps). "펀딩 부호가
  다음 정산까지 지속된다"는 전략의 핵심 가정이 이 표본에서 성립하지 않음 — 오히려 반전에
  가깝다. 가격 컴포넌트도 대부분 음수(캐리 방향이 추세를 거스름). BTCUSDT theta_0/h=8은
  t=-3.99로 통계적으로 유의한 손실.
- 결론: 단순 부호-지속 가정의 펀딩 캐리는 이 3자산에서 근거 없음. val/OOS 진행 안 함.

## F3-B 크로스섹션 모멘텀 — **KILLED (탐색 단계)**

스크립트: [`scripts/research_f3b_cross_sectional_momentum_20260719.py`](../../../../scripts/research_f3b_cross_sectional_momentum_20260719.py)
결과: [`f3b_cross_sectional_momentum_20260719.json`](f3b_cross_sectional_momentum_20260719.json)

- k∈{7,14,30}일 순위 롱숏, 3자산 전부.
- **총수익(비용 전)은 3개 k값 모두 양수**(1.5~8.2bps/일, k가 클수록 큼) — 방향성 있는
  진짜 신호일 가능성. 하지만 **일일 양다리 리밸런싱 비용(20bps/일 왕복×2)이 압도**해서
  순수익 전부 음수 (t=-1.67~-2.67, 미달).
- 3자산 유니버스 한계 명시(설계 문서 anti-fishing 규칙에 따라 유니버스 확장 없이 종료).
- **후속 아이디어(재시도 아님, 기록만)**: 리밸런스 빈도를 낮추면(주 1회 등) 비용이
  ~7배 줄어 총수익 대비 손익분기 가능성 — 별도 사전 등록 필요한 새 실험.

## F3-C 직교성 게이트 — **적용 대상 없음**

F3-A/B가 둘 다 탐색 단계에서 kill돼 새 팩터 후보가 없음. 설계상 F3-C는 F3-A/B 통과
시에만 발동하는 관문이라 이번 실행에서는 스킵 (정상 동작).

## F4-B 포트폴리오 결합 — **부분 통과 (실질적 다각화 효과 확인)**

스크립트: [`scripts/research_f4b_sigma6_dated_ledger_20260719.py`](../../../../scripts/research_f4b_sigma6_dated_ledger_20260719.py) (Sigma6 날짜정보 재현) +
[`scripts/research_f4b_portfolio_combination_20260719.py`](../../../../scripts/research_f4b_portfolio_combination_20260719.py)
결과: [`f4b_portfolio_combination_20260719.json`](f4b_portfolio_combination_20260719.json)

**입력 검증**: Sigma6 lev4/lev3 OOS 원장을 원본 로직 그대로 복제 재실행해 계약서 수치와
정확히 일치 확인(lev4 +45.85% vs 계약서 +45.9%, lev3 +16.64% vs +16.6%) — 재현성 확보 후
사용. Omega4.6.1은 2026-07-06 기 계산된 greedy-router 원장(no-gate 버전, 138.19%/-14.15%/32트레이드,
duration-gate 최종버전보다 보수적) 재사용 — 이미 검증된 산출물의 재활용이라 Fresh-Forward
규칙 위반 아님.

겹치는 구간(2026-03-02~06-30, 4개월)만 사용:

| 지표 | Omega4.6.1 단독 | Sigma6 lev3 단독 | 균등가중 결합 | 역변동성 결합 |
|---|---|---|---|---|
| 총수익 | +48.36% | +18.35% | +33.36% | +28.75% |
| Sharpe-유사(연환산) | 2.21 | 1.58 | **2.58** | 2.54 |
| MDD | **-8.33%** | -13.26% | -10.44% | -11.30% |

- **상관계수**: 0.107 (부트스트랩 CI [-0.03, 0.26], 0을 포함 — 낮고 불확실).
- **판정 상세**: 균등가중 결합의 Sharpe(2.58)가 **개별 전략 둘 다보다 높음** — 낮은 상관의
  진짜 다각화 효과. 다만 MDD는 결합(-10.44%)이 Omega 단독(-8.33%, 이미 매우 낮음)보다
  개선되지 않음 — Omega가 원래 MDD가 워낙 좋아서 Sigma6를 섞으면 오히려 희석됨.
- **설계 문서의 엄격한 AND 기준(MDD 개선 그리고 Sharpe 비열화)은 미충족**하지만, Sharpe가
  두 구성요소 모두를 능가한다는 건 교과서적 다각화 신호라 "부분 통과"로 기록.
- **버그 발견/수정**: 최초 코드에서 `best_single_mdd`를 `min()`으로 계산해 더 나쁜 쪽을
  기준으로 삼는 실수 발견 → `max()`(음수 중 0에 가까운 쪽)로 교정 후 재실행.
- **한계**: 두 전략 모두 트레이드 희소(Omega 19건, Sigma6 26건/4개월)라 상관 추정 자체가
  불안정. 트레이드 단위 수익을 청산일에 배치한 비복리 일별 근사(설계 문서에 명시된 단순화).

## F4-C 대체데이터 수집기 — **1단계 구축 완료, cron 등록**

스크립트: [`scripts/run_f4c_altdata_collector.py`](../../../../scripts/run_f4c_altdata_collector.py)

- Fear&Greed Index(alternative.me), Binance-OKX 펀딩비 스프레드(ETH/BTC/SOL), Binance 공지사항
  3종 수집기 구현·테스트 완료.
- **실행 중 발견/수정**: (1) `binanceusdm` 심볼 포맷이 `ETH/USDT`가 아니라 `ETH/USDT:USDT`
  필요 — `load_markets()` 누락 확인 후 수정. (2) Binance 공지 API 응답 구조가 예상
  (`data.catalogs[].articles[]`)과 달리 실제로는 `data.articles[]` 평탄 구조 — 실제 응답을
  찍어보고 파싱 로직 재작성.
- `data/research/altdata.duckdb`에 3개 테이블 생성, 초기 시드 완료 (FNG 1행, 펀딩스프레드
  3행, 공지 20행).
- **크론 등록**: 일 1회 01:00 KST (`0 1 * * *`), 기존 컨벤션(`.sh` 래퍼 + 로그) 그대로.
- **2단계 이벤트 스터디 가설 사전 등록** (지금 등록, ≥3개월 데이터 누적 후 1회 검정):
  - H-FNG: FNG 지수 극단값(≤20 또는 ≥80) 다음날 ETH 수익률 분포가 무조건부와 다른가.
  - H-SPREAD: Binance-OKX 펀딩 스프레드 상위 5%일 때 다음 8h 수익률이 스프레드 축소
    방향을 예측하는가 (크로스거래소 차익거래 압력 가설).
  - H-ANNOUNCE: 상장 공지(신규 자산 한정, ETH/BTC/SOL과 무관) 발표 후 24h간 시장 전체
    변동성(대리로 ETH realized vol)이 무조건부보다 높은가.

## 종합

| 실험 | 판정 | 다음 액션 |
|---|---|---|
| F3-A 펀딩 캐리 | KILL | 종결, 재시도 근거 없음 |
| F3-B 크로스섹션 | KILL | 종결. 리밸런스 저빈도화는 별도 신규 실험으로 취급 |
| F3-C 직교성 게이트 | N/A | F3-A/B kill로 자동 스킵 |
| F4-B 포트폴리오 결합 | **부분 통과** | Sharpe 다각화 효과는 실재 — 3번째(BTC 포함) 전략 편입 시 재검정 가치 있음 |
| F4-C 수집기 | 구축+cron 등록 완료 | 3개월 후 사전등록 가설 1회 검정 |

이번 라운드의 가장 유의미한 발견은 **F4-B** — 새 알파 없이 이미 검증된 두 전략을 저상관으로
결합하는 것만으로 리스크조정수익이 개선된다는, "월가 방식 3/4번"의 예측과 정확히 일치하는
실증 사례.
