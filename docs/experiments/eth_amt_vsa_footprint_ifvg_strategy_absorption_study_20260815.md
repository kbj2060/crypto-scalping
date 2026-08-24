# AMT/VSA/Footprint/iFVG 트레이더 프레임워크 조사 및 흡수 가능성 검증 (2026-08-15)

## 성격

[[eth_yush_orderflow_strategy_absorption_study_20260815]]의 후속. **데스크 조사 + 저장소 대조 +
미보유 구성요소의 실증 측정**. retrospective evidence study이므로 promotion/live 근거 아님.
Fresh-Forward 규칙 적용 대상 아님. 라이브 파일 변경 없음.

## 요청

"이런 트레이더들의 전략을 모두 조사해서 내 모델에 쓸만한 것들이 있는지 연구해줘." — Yush와
같은 계열(재량 오더플로우/Volume Profile/Auction Market Theory 기반 데이트레이더)의 다른
트레이더 4명/유파를 조사 대상으로 삼았다. Yush 본인 및 그와 겹치는 개념(단일봉 델타, 단일봉
흡수, 롤링 스윙 유동성 스윕, 세션/전일 레벨, Value Area 가장자리/중앙)은 이미 그 문서에서
결론이 났으므로 **재검증하지 않고 인용만** 했다.

---

## 1부 — 조사 대상 4개 프레임워크

### A. Jim Dalton — Auction Market Theory (Market Profile의 원조, Yush의 LAF가 인용하는 이론 그 자체)

"Market Profile의 대부"로 불리며 저서 *Mind Over Markets*, *Markets in Profile*로 확립. 핵심은
개별 레벨 터치가 아니라 **다중 봉 balance/rotation 레짐** 규칙 3개다.

- Rule 1: 가격이 balance 영역 **안으로 진입(accept)** 하면 반대편으로 되돌아가려는 경향.
- Rule 2: balance **내부**에서는 가장자리가 거부(reject)되고 변동이 chop함.
- Rule 3: balance **바깥으로 진입(accept)** 하면 불균형이 되어 새로운 value를 찾아가며, 목표는
  종종 이전 balance의 POC.

### B. Tom Williams — VSA (Volume Spread Analysis, Wyckoff 계승)

- **No Demand bar**: 상승봉인데 거래량이 최근 평균의 50% 이하, 몸통(spread)도 평균의 40%
  이하 — 매수 관심 부재, 약세 경고.
- **No Supply bar**: 하락봉인데 거래량·몸통 모두 낮음 — 매도 압력 부재, 강세 경고.
- 핵심 주장: **저거래량(효과의 부재)도 정보다.** 이 저장소가 지금까지 측정한 신호는 전부
  "고거래량 이벤트"였다는 점에서 극성이 반대인 주장이다.

### C. Footprint/Stacked Imbalance 트레이더 (오더플로우 플랫폼 커뮤니티 일반)

- **Stacked Imbalance**: 연속된 3개 가격 레벨에서 bid/ask 비율이 4:1 이상 — "확인용이지
  단독 진입 신호 아님."
- Absorption, exhaustion, unfinished auction, delta divergence, iceberg는 이미 Yush 문서에서
  다룬 개념과 본질적으로 같아 재검증하지 않았다.
- 5분봉에는 가격 레벨별 호가 데이터가 없어 "3레벨 4:1"을 그대로 재현할 수 없다. 유일하게 남는
  진짜 신규 질문은 **봉 단위 지속성(persistence)** — 단일봉 델타가 아니라 연속 3봉이 같은
  방향으로 쏠리는가.

### D. Dodgy DD(Ryan Wilson, @DodgysDD) — iFVG (Inversion Fair Value Gap)

- Yush와 유사한 프로필(검증된 펀디드 트레이더, 무료 커뮤니티 4.1만 명)이지만 계보가 완전히
  다르다 — ICT/Smart Money Concepts 파생.
- **FVG(Fair Value Gap)**: 3개 봉 중 1번째 고가 < 3번째 저가(강세 갭) 또는 그 반대(약세 갭) —
  "가격이 채워지지 않고 지나간 공백".
- **iFVG(Inversion)**: 이후 가격이 돌아와 그 공백을 **몸통 종가로** 뚫고 지나가면 반대 방향
  신호로 "반전"됨. "심지가 뚫는 것만으로는 부족하다 — 그건 거부이지 반전이 아니다. 반드시 몸통
  종가여야 한다."
- 실전 셋업: **유동성 그랩(스윕) → 반대 방향 FVG 형성 → 종가 반전 확인** → 진입, 손절은 FVG
  바로 아래/위, 목표 1:2 RR.

---

## 2부 — 저장소 대조

```
grep -rniE "fair_value_gap|\bfvg\b" → strategies/elite_builder.py의 fvg_dist뿐,
  실제로는 "3봉 갭"이 아니라 갭 자석-거리 개념이고 inversion 로직 없음. sig_orderblock은
  코드상 wick_ratio+log_return 조합일 뿐 실제 orderblock/FVG 판정이 아님(elite_builder.py:756
  주석에 "❌ 삭제됨: CVP_FVG" — 이미 폐기 표시된 실험 코드).
grep -rniE "balance_area|acceptance|rotation_factor|initial_balance" → 0건
grep -rniE "stacked_imbalance|delta_divergence|exhaustion_bar|unfinished_auction" → 0건
grep -rniE "no_demand|no_supply|effort.*result" → 0건
```

**4개 프레임워크 전부 저장소에 실질적으로 존재하지 않는다.** `strategies/elite_builder.py`의
"FVG" 관련 코드는 이름만 같을 뿐 다른 개념이고, 이미 자체적으로 삭제 표시가 남아 있다.

---

## 3부 — 미보유 구성요소의 실증 측정

### 방법

- 스크립트: `scripts/analyze_eth_amt_vsa_footprint_ifvg_component_evidence_20260815.py`
- 하네스는 Yush 문서와 완전히 동일하게 재사용(재구현 아님): `event_study`/`excess_move`/
  `load_zigzag_pivots`, `compute_indicators`, `race_outcomes`(연속성 신호 전용).
- 스윕(sweep_low/sweep_high) 정의는 저장소 기존 정의(48봉 causal 롤링 스윙)를 **그대로 재사용**
  — 새로 정의하지 않음.
- 창: VAL 2025-09-01~2025-12-31 + OOS 2026-01-01~2026-02-17. 48,853봉, 피벗 3,044개.
  (Yush 문서보다 표본이 큰 이유: 이번엔 `min_periods`를 완화해 rolling window 정착 이전 구간도
  일부 포함했다 — 방법 자체는 causal이라 인과성 훼손 없음.)
- 모든 레벨/레짐은 완료된 과거 데이터만 사용(shift/rolling().shift(1)).

### 결과 — 1시간(K12) 기준 lift

**바닥(bottom)**

| 신호 | n | 정밀도 | lift | 비고 |
|---|---|---|---|---|
| B1 excess_tail_deep (깊은 스윕, ≥1.0×ATR) | 87 | 36.8% | **2.94배** | 표본 작음, recall 2.6% |
| **B2 balance_edge_reject (저변동성 레짐 + 레인지 가장자리)** | 1,869 | 21.8% | **1.74배** | 표본 충분 |
| C1 VSA no_supply | 671 | 7.6% | **0.61배** | 근거 없음, 역방향 |
| D1 persistent_delta(3봉 연속 매수쏠림) | 2,024 | 5.5% | **0.44배** | 근거 없음, 역방향 |
| E1 FVG 터치 | 30,284 | 11.1% | 0.88배 | 근거 없음 |
| E2 FVG 반전(iFVG) | 11,225 | 6.0% | **0.48배** | 근거 없음, 역방향 |
| E3 스윕+iFVG 결합(Dodgy DD 셋업) | 1,943 | 8.1% | **0.65배** | 근거 없음 |

**천장(top)**

| 신호 | n | 정밀도 | lift |
|---|---|---|---|
| B1 excess_tail_deep | 80 | 32.5% | 2.77배 |
| **B2 balance_edge_reject** | 3,189 | 18.8% | **1.60배** |
| C1 VSA no_demand | 675 | 8.0% | 0.68배 |
| D1 persistent_delta(3봉 연속 매도쏠림) | 2,352 | 5.7% | 0.49배 |
| E1 FVG 터치 | 30,066 | 10.6% | 0.90배 |
| E2 FVG 반전 | 10,781 | 6.9% | 0.58배 |
| E3 스윕+iFVG 결합 | 1,820 | 6.5% | 0.56배 |

**B3 (AMT Rule 3 — balance 이탈 후 지속성, TP/SL race 방식)**: 상승 이탈 0.99배, 하락 이탈
0.98배 — **무작위(baseline)와 구분 불가**. 4시간/8시간에서도 동일.

### 발견 1 — Dalton Rule 2가 Yush의 Value Area 가장자리 실패를 뒤집는다 (이번 세션 최고 발견)

Yush 문서에서 "Value Area 가장자리에서 거래하라"는 게이트 없이 테스트했을 때 lift는
**0.96(바닥)/0.81(천장)** 로 근거가 없었다. Dalton의 진짜 규칙은 "**저변동성 balance 레짐일
때만**"이라는 조건이 붙어 있다. 이 조건(48봉 ATR% 백분위수 ≤30) 하나를 추가하자 lift가
**1.74/1.60**으로 반전됐다.

- VAL→OOS 안정성 확인 완료: 바닥 1.69→**1.89**(개선), 천장 1.66→1.42(완만한 감쇠지만 여전히
  양의 lift). 표본도 충분(VAL 1,360~2,500건, OOS 509~689건).
- 겹침 확인: 기존에 측정된 Bollinger %b 극단(마스터 순위 2.34배) 신호와의 봉 단위 겹침은
  **34%(바닥)/21%(천장)뿐** — 통계적 극단값과 "저변동성 레짐+레인지 경계"는 별개의 대상을
  잡고 있다. 재탕이 아니라 진짜 추가 정보다.
- 순위: 마스터 스코어카드 기준으로는 중위권(Bollinger 2.34배·%R+SlowK 2.28배보다 약하고,
  라운드넘버 1.79배보다 강함). 최상위권(3배대)은 아니지만 **유효하고 독립적인** 신호다.

### 발견 2 — VSA "No Demand/No Supply"는 크립토 5분봉에서 신화다

lift 0.61(바닥)/0.68(천장) — **무작위보다 나쁘다.** 이 저장소가 지금까지 측정한 모든 승자
신호는 고거래량 이벤트였는데, VSA의 "저거래량=정보"라는 반대 극성 주장을 직접 테스트해 봐도
성립하지 않는다. 펀딩비 극단 신화(기존 문서에서 이미 파괴됨)와 같은 계열의 결과다.

### 발견 3 — 오더플로우 "지속성"은 반전이 아니라 반전의 반대다

기존 마스터 순위의 "오더플로우 순매도 급증"(단일봉, 5위, 2.75배)은 강했다. 그런데 같은 방향이
**3봉 연속**되는 지속성(footprint stacked-imbalance의 봉 단위 대용치)은 lift 0.44/0.49로
역방향이다. 해석은 직관적이다 — 한쪽으로 쏠림이 단일봉에서 끝나면 소진(exhaustion) 신호지만,
3봉이나 이어지면 그냥 **추세 지속**이라 반전 신호로는 반대 방향이다. 이건 "확인이 많을수록
좋다"는 직관이 여기서도(Yush의 confluence-count 실패에 이어) 틀릴 수 있음을 보여준다.

### 발견 4 — iFVG는 어떤 형태로도 근거가 없다 (결정적)

- FVG 자체가 지지/저항 자석이라는 ICT 주장(E1 터치): 0.88/0.90, 근거 없음.
- FVG가 몸통 종가로 반전됐다는 확인(E2): 0.48/0.58, **오히려 역신호**.
- Dodgy DD의 실전 셋업 전체(스윕 → FVG 형성 → 반전 확인, E3): 0.65/0.56, **역신호**. 참고로
  스윕 단독은 마스터 순위 2위(3.01배)다. **FVG 반전 확인을 덧붙이면 스윕 단독보다 명백히
  나빠진다** — 이 저장소에서 반복 관측된 "확인을 더할수록 오히려 신호가 희석된다" 패턴의 또
  다른 사례.

### 발견 5 — Dalton Rule 3(balance 이탈 후 지속)은 무근거

TP/SL race 방식으로 측정한 방향 지속성이 baseline과 완전히 같다(0.98~0.99배). "레인지 이탈 =
추세 시작"이라는 직관은 ETH 5분봉에서 성립하지 않는다.

---

## 결론 — 흡수 가능성 판정

**부분 채택 가능(약함), 대부분은 흡수 불가.**

### 채택 후보

1. **B2 저변동성 레짐 게이트 + 레인지 가장자리 fade** — 이 두 문서(Yush + 이번)를 통틀어
   유일하게 새로 확보된 **양의 lift, VAL/OOS 안정, 기존 신호와 겹침 낮음**을 모두 만족하는
   신호. 마스터 스코어카드에 신규 항목으로 추가할 가치가 있다. 단, 이 문서는 방향성 lift만
   측정했고 비용(breakeven bp) 검증과 사이징 lag(1봉) 규율은 아직 적용 전이므로 — 실제 채택
   전에는 `docs/experiments/evidence_signal_quant_use_*` 계열의 비용 게이트를 통과해야 한다.

### 흡수하지 않는 것 (근거 무 또는 역신호)

- VSA no-demand/no-supply, 3봉 지속 델타, FVG 터치/반전, 스윕+iFVG 결합, balance 이탈
  지속성 — 전부 lift ≤1.0이거나 역신호. 코드화하지 않는다.
- B1(깊은 스윕)은 방향은 맞지만 표본이 너무 작아(n=80~87) 채택 근거로 쓰기엔 이르다. 향후
  더 긴 기간으로 표본을 늘려 재검증이 필요하면 그때 별도 실험으로 다룬다.

## 알려진 한계

- Footprint의 "3레벨 4:1 stacked imbalance"는 가격 레벨별 호가 데이터가 없어 봉 단위
  대용치(D1)로만 근사했다. 진짜 stacked imbalance의 성능은 이 문서로 판정되지 않는다.
- B1(excess_tail_deep)의 표본이 작아 신뢰구간이 넓다.
- retrospective event study이므로 거래 비용/슬리피지 미반영. lift는 수익성이 아니라 피벗 근접
  확률의 배수다.
- 조사 대상 트레이더 서술의 출처는 대부분 2차 정리 기사/영상(Dalton은 예외 — 저서 원전 개념).
  Dodgy DD 자료는 인디케이터 판매 페이지 성격이 섞여 있어 편향 가능성을 감안해야 한다.

## 산출물

- `scripts/analyze_eth_amt_vsa_footprint_ifvg_component_evidence_20260815.py`
- `tmp/eth_amt_vsa_footprint_ifvg_component_evidence_20260815/{reversal_component_evidence_table,balance_breakout_continuation_table}.csv`

## 출처

- [Intro to auction market theory and market profile — Topstep](https://www.topstep.com/blog/intro-to-auction-market-theory-and-market-profile)
- [Auction Market Theory — TradingRiot](https://blog.tradingriot.com/p/auction-market-theory)
- [Markets in Profile: Profiting from the Auction Process — Jim Dalton (Wiley)](https://www.wiley.com/en-us/Markets+in+Profile:+Profiting+from+the+Auction+Process-p-9781118044643)
- [VSA and cluster analysis. No Demand and No Supply — ATAS](https://atas.net/volume-analysis/basics-of-volume-analysis/vsa-and-cluster-analysis-no-demand-and-no-supply/)
- [Volume Spread Analysis — TradersUnion](https://tradersunion.com/technic-analysis/volume-spread-analysis/)
- [Footprint Chart Trading: Learn How to Use Order Flow and Delta — TradeThePool](https://tradethepool.com/fundamental/mastering-footprint-charts-trading/)
- [Footprint Chart Patterns Cheat Sheet for Order Flow — GoCharting](https://gocharting.com/blog/footprint-charts/footprint-chart-patterns-cheatsheet)
- [Catch the Reversal: Trading the Inverse Fair Value Gap (IFVG) Strategy — FTMO](https://ftmo.com/en/blog/catch-the-reversal-trading-the-inverse-fair-value-gap-ifvg-strategy/)
- [Spotlight: DodgySDD — The $200K/Mo Trading Indicator Empire — Whop Trends](https://whoptrends.com/blog/spotlight-dodgysdd-ifvg-2026)
- [Inversion Fair Value Gaps (IFVG) Explained — FluxCharts](https://www.fluxcharts.com/articles/inversion-fair-value-gaps-ifvg-explained)
