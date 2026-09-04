# 청산 스파이크 / V자 반등 — 진입·청산 방법론 외부 문헌 조사 (2026-08-30)

배경: `liquidity_sweep`(V자반등) 서브프로젝트가 라벨 v4→v7b로 분류 성능을 극적으로 개선했음에도
(VAL/OOS/홀드아웃 AUC 0.66~0.68 → 0.73~0.78) 트레일링스톱·고정TP/SL·무TP시간청산 3가지 exit
구조 전부 경제성 게이트를 통과하지 못했다(0/205, 이후 재검증들도 동일 패턴). "exit 구조를 바꿔도
안 되면, 애초에 진입 방향/타이밍 설계 자체가 문헌과 맞는지 확인해보자"는 취지로 외부 문헌을
재조사. 2개 병렬 에이전트가 각각 조사, 모든 인용은 WebFetch로 원문(초록/본문)을 직접 대조
검증했고, 검증 실패(SSRN 403 등)는 명시적으로 플래그했다.

**이미 답이 나온 질문이라 재조사하지 않은 것들**(중복 방지):
- "가격레벨 청산맵(포지션 밀도 클러스터링)이 유효한 모델인가" → 이미 기각
  ([[eth_liquidation_map_literature_review_cascade_mechanism_20260825]], arXiv:2607.27070/
  2608.03616 — 캐스케이드는 자기증폭 분기과정이 아니라 subcritical(λ≈0.1-0.2), 신호는
  포지션 밀도가 아니라 오더북/유동성 섹터에 있음).
- ML 라벨링 방법론 일반(triple-barrier, meta-labeling, deep optimal stopping) →
  [[reference_direction_quality_exit_label_methodology_20260819]].
- exit 이산화/TP·SL 결합최적화 일반(Broadie-Glasserman-Kou 1997, Hwang et al. 2023, Leung & Li
  2015, Grinold's Law) → V_REBOUND 실험문서 "매도방법 리서치" 섹션에 이미 정리됨.

---

## Part A — 청산 스파이크/캐스케이드: 진입 방향과 타이밍

### A1. Fade(역추세) vs Continuation(추세추종) — 이론과 실증

**되돌림(overshoot-then-revert) 쪽 근거가 이론적으로 탄탄하다.** Brunnermeier & Pedersen
(2005, *Predatory Trading*, J. Finance)의 모델이 핵심: 누군가 강제청산을 당하면 다른 트레이더들이
**먼저 같은 방향으로 팔아 가격을 과잉하락시킨 뒤 다시 사들여 되돌린다** — 즉 오버슈트와 되돌림이
같은 사건의 앞뒤일 뿐 서로 경쟁하는 결과가 아니다. Coval & Stafford (2007, *Asset Fire Sales
(and Purchases) in Equity Markets*, JFE)는 강제매도가 만든 가격압박이 이후 되돌려지며, 그 반대편에
서서 사들인 유동성공급자가 유의미한 수익을 낸다는 걸 실증했다.

**크립토 특화 근거**: De Nicola (2021, *On the Intraday Behavior of Bitcoin*, Ledger)는 BTC
1h/2h/4h 수익률에 유의미한 음(-)의 1차 자기상관을 발견했고, 원인으로 "투자자 과잉반응, 초과
변동성, **과도한 레버리지로 인한 청산 캐스케이드**"를 명시적으로 지목했다 — 청산이 클수록
반전도 크다는 뜻. 실제 사례로 2019년 OKEx 캐스케이드(단일 대량 시장가주문이 유동성 얕은
시간대에 BTC선물을 $7,000→$4,600까지 끌어내렸는데 현물은 $6,300 밑으로 안 내려감 — 선물이
현물 앵커로 되돌아올 수밖에 없었던 순수 청산 아티팩트)가 있다(Deribit Insights).

### A2. 진입 타이밍/확인 지표

업계 리서치(Glassnode, Coinglass, Amberdata 등, 전부 서술적 자료이며 백테스트는 없음)는
**OI 낙폭, 펀딩비 극값, 거래량/회전율 클라이맥스, 테이커 불균형 반전, 오더북 뎁스의
철수-후-복귀를 개별이 아니라 "여러 개가 동시에" 관찰하는 걸 안정화 신호로 삼는다**는 데
수렴한다. 다만 이 확인규칙 자체를 검증한 백테스트는 못 찾았다.

가장 근접한 정량적 근거는 Lim (2025/2026, SSRN 6636998, *Two-Regime Liquidity Recovery After a
Perpetual Futures Liquidation Cascade*, Hyperliquid 2025-10-10 데이터) — 캐스케이드 시간대
호가스프레드가 중앙값 9.4bp(95th pct 87.5bp)까지 벌어졌다가 **당일 내 대부분 정상화**, 23일
안에 사전 수준의 1.2배 이내로 완전 복구된다고 보고. **단, 이 논문은 SSRN 접근 차단(403)으로
원문 대조가 안 되고 검색스니펫으로만 확보됨 — 이 저장소 원칙상 미검증으로 취급**.

### A3. 이름 붙은 "캐스케이드 페이드" 전략 — 엄밀한 근거는 약함

"stop hunting"/"liquidation hunting"은 대개 **캐스케이드를 유발하는 쪽**(대형 트레이더가
스탑 밀집구간으로 가격을 밀어넣는 행위)을 서술하는 용어이지, 그 이후를 거래하는 방법론이
아니다. 실제로 정량화된 "캐스케이드 페이드" 백테스트는 개인 퀀트 블로그(curupira.dev) 하나뿐
— 5봉 속도+거래량 3배 필터로 진입, 5분 이내 시간청산, 3개 자산·800일 워크포워드에서 손익비
1.5~2.9·승률67% 보고했지만 **동료검토 없는 개인 블로그**라 증거 강도가 낮다. 학술/기관
리서치(Kaiko/Amberdata/Deribit) 중 실제 성과를 공개한 "청산 페이드" 전략은 발견하지 못했다.

### A4. 반대편 — 캐스케이드가 되돌리지 않고 계속 가는 경우

Brunnermeier & Pedersen (2009, *Market Liquidity and Funding Liquidity*, RFS)는 마진이
"destabilizing"할 수 있어 자기강화적 **유동성 스파이럴**(되돌림이 아니라 지속)이 가능함을
보여준다 — 이 저장소가 이미 확인한 arXiv:2608.03616 자체도 in-cascade 시그니처를 "B&P(2009)
의미의 유동성 스파이럴"로 명시적으로 프레이밍한다. 실측 사례: FTI Consulting에 따르면 2025-10
크래시는 당일 안정화됐음에도 **2.5개월 뒤에도 고점 대비 -30%** — 단기 되돌림과 중기 지속이
공존할 수 있다. 얕은 유동성(주말/야간), 단일 대량주문 트리거, 단일 거래소 쏠림이 문헌이
공통으로 지목하는 "되돌리지 않는" 증폭 조건.

---

## Part B — V자 반등: 진입 확인과 청산

### B1. 짧은 구간(30~60분) 반전의 학술적 지위 — 애매함을 정직하게 인정해야 함

진짜 반전 효과는 확인되지만 **구간이 이 프로젝트 라벨(30~60분)보다 훨씬 김**: Lehmann
(1990)은 **주간**(weekly) 반전, Bremer & Sweeney (1991)는 극단적 10일 하락 이후 **~2일**
반전을 찾았다. De Bondt & Thaler (1985)의 과잉반응은 **3~5년** 스케일로 완전히 다른 현상.
반대로 경고 신호도 있다: Atkins & Dyl (1990), Cox & Peterson (1994)는 대형 가격변동 이후
관측되는 "반전"의 상당수가 **진짜 반전이 아니라 매수-매도 호가 바운스(bid-ask bounce)/
비유동성 아티팩트**라고 결론짓는다(Cox & Peterson은 과잉반응 가설 자체를 명시적으로 기각).
**30~60분 해상도에서 직접 검증한 주식/고빈도 문헌은 발견하지 못했다** — 이 구간은 사실상
학술적 공백지대.

**크립토**: Zaremba et al. (2021)이 3,600개+ 코인에서 일간 횡단면 반전을 발견했지만, **가장
크고 유동성 높은 코인(ETH 등급)은 반전이 아니라 모멘텀**을 보인다는 단서가 있다 — ETH에
그대로 적용하면 안 됨.

### B2. Stop-hunt/유동성 스윕 문헌 — 실재하지만 크립토가 아니라 FX

Osler (2003, J. Finance)는 은행 자체 스탑로스/이익실현 주문 데이터로 라운드넘버 바로 너머의
스탑로스 밀집이 **돌파 후 가속**(스윕/지속 구간)을 설명한다는 걸 실증했다. Osler (2005, JIMF)는
스탑로스 주문이 자기강화적 "가격 캐스케이드"를 만들며 이게 **"며칠이 아니라 몇 시간" 지속**
된다고 명시 — 이 프로젝트의 30~60분 창과 자릿수는 맞지만, 이건 스윕/캐스케이드 구간 자체의
지속시간이지 "그 이후 반전"을 직접 다루진 않는다. Brunnermeier & Pedersen (2005), Carlin,
Lobo & Viswanathan (2007)은 포식적 트레이딩이 유발한 강제청산과 그 뒤 "가격 오버슈트"에서
수익 내는 걸 이론적으로 모델링. 전부 FX/이론이며 크립토 실측은 아니다.

### B3. ⭐평균회귀 최적 진입+청산 — 이미 가진 인용이 답을 갖고 있었음

**가장 실용적인 발견**: 이 프로젝트가 exit 이산화(TP:SL 결합최적화) 근거로 이미 쓰고 있는
**Leung & Li (2015)** 논문 자체가, **진입 구간도 exit와 동시에(이중 최적정지 문제로) 도출한다**
— 손절선 바로 위의 특정 구간에서 진입하는 게 최적이라는 걸 보여준다. 즉 이 신호의 진입설계를
더 원칙적으로 다듬고 싶다면 **새 논문을 찾을 필요 없이 이미 인용 중인 논문의 진입-구간 도출
부분을 다시 읽으면 된다**. 같은 계열로 Zhang & Zhang (2008, *Automatica*)·Song & Zhang (2013,
*Automatica*/arXiv:1302.6120)이 OU 평균회귀에서 "싸게 사서 비싸게 팔기"를 HJB 이중정지로
풀고, Bertram (2010, *Physica A*)이 최초통과시간 기반 진입/청산 문턱을 제시한다(단 Zhang&Zhang과
Bertram은 이번 조사에서 원문 접근이 막혀 서지사항만 확인, 초록 내용은 미검증 — ⚠️).

### B4. "반전 확인(confirmation)" 자체는 학술 용어가 아님

"reversal confirmation"이나 "dead cat bounce"라는 정확한 개념으로 된 학술문헌은 없음(트레이더
용어). 가장 가까운 건 Lo, Mamaysky & Wang (2000, J. Finance) — 더블바텀 등 기술적 패턴(구조상
V자와 유사)이 커널회귀로 객관화했을 때 실제 정보량을 갖는다는 실증. 이건 "몇 분/몇 봉을
기다려야 확인되는가"에 대한 규칙이 아니라, **패턴 검증 방법론 자체의 템플릿**으로 참고할 가치가
있다 — 이 프로젝트가 이미 하고 있는 "차트 육안검증+AUC/판정별 정밀도" 방식과 정신이 같다.

---

## 종합 — 이 프로젝트의 기존 실증 결과와 어떻게 연결되는가

1. **"왜 30분 고정 익절이 트레일링보다도 나빴는가"에 대한 문헌 기반 가설이 하나 생겼다.**
   Osler(2005)의 "캐스케이드는 며칠이 아니라 몇 시간 간다"와 Bremer & Sweeney(1991)의 "~2일
   반전 윈도우"를 함께 보면, 반전이 만드는 초과수익이 발생하는 시간축 자체가 이 신호의
   FAST_BARS=6(30분)보다 훨씬 길 가능성이 있다 — [[eth_v_rebound_fixed_tpsl_exit_structure_refuted_20260830]]에서
   실측한 "30분 익절은 최악, TP 없이 길게 들고가는 쪽이 최선"이라는 반직관적 결과와 방향이
   일치한다. 다만 그 결과 자체가 숏 51건 쏠림으로 얇았다는 것도 이미 확인된 사실 — 문헌이
   방향을 뒷받침한다고 그 결과의 소표본 취약성이 사라지는 건 아니다.
2. **"분류는 좋아졌는데 경제성은 그대로"에 대한 새로운(그리고 더 불편한) 가설**: Atkins & Dyl
   (1990)/Cox & Peterson (1994)가 경고하는 "겉보기 반전의 상당수는 진짜 알파가 아니라 매수-
   매도 호가 바운스/비유동성 아티팩트"라는 지적은, v7b가 OHLC 종가 기준으로 학습·분류하면서
   실제 체결(스프레드+비용)에서는 사라지는 순수 미시구조 잡음을 학습했을 가능성을 시사한다.
   기존 결론("손실원인=순수 분류오류")을 한 단계 더 구체화하면: **"분류기가 배우는 신호 중
   일부가 애초에 실현 가능한 가격움직임이 아니라 호가 바운스일 수 있다"**는 방향으로 좁혀볼
   수 있다 — 아직 검증 안 된 새 가설이며, 검증하려면 종가 대신 mid-price/실제 체결가 기준으로
   라벨을 다시 만들어 같은 AUC가 나오는지 보는 게 다음 단계가 될 것(미실행, 제안만).
3. **청산 스파이크 자체(liquidation_cascade 신호)에 대한 진입 설계 시사점**: fade가 이론적으로
   탄탄하지만(Brunnermeier & Pedersen 2005, Coval & Stafford 2007) 크립토 실전 백테스트 근거는
   약하고(개인 블로그 1건뿐), 되돌리지 않는 경우(유동성 스파이럴, B&P 2009)의 조건(얕은
   유동성·단일거래소 쏠림)이 명확히 문헌화돼 있다 — `eth_liquidation_cascade_sweep_vs_trend`
   신호가 이미 "스위칭 vs 지속"을 구분하려는 것 자체가 이 문헌이 말하는 정확한 구분과 일치한다.
4. **실용적 다음 단계 후보(제안만, 미실행)**: Leung & Li (2015)의 진입-구간 도출 부분을 다시
   읽고 V_REBOUND의 "몇 시간 시점에 진입하는가" 설계에 원칙적으로 반영할 수 있는지 검토, 또는
   mid-price 기준 재라벨링으로 호가바운스 가설을 직접 검증.

---

## 전체 인용 목록 (검증상태 표기)

### Part A — 청산 캐스케이드

- Brunnermeier, M. & Pedersen, L. (2005). "Predatory Trading." *J. Finance* 60(4). NBER WP
  10755. [원문대조완료]
- Brunnermeier, M. & Pedersen, L. (2009). "Market Liquidity and Funding Liquidity." *RFS*
  22(6). NBER WP 12939. [원문대조완료]
- Coval, J. & Stafford, E. (2007). "Asset Fire Sales (and Purchases) in Equity Markets."
  *JFE* 86(2). NBER WP 11357. [원문대조완료]
- Kirilenko, Kyle, Samadi, Tuzun (2017). "The Flash Crash." *J. Finance* 72(3). [초록만
  대조완료, "30분내 회복" 수치는 초록에서 미확인]
- De Nicola, G. (2021). "On the Intraday Behavior of Bitcoin." *Ledger* 6. [원문대조완료]
- Lim, B.C. (2025/2026). "Two-Regime Liquidity Recovery After a Perpetual Futures
  Liquidation Cascade." SSRN 6636998. [⚠️ SSRN 403 차단, 검색스니펫만 확보, 미검증]
- Lim, B.C. (2026). "Anatomy of a Crypto Cascade: Minute-Level Evidence from the October
  2025 Crash." SSRN 6579278. [⚠️ 동일사유 미검증]
- "Stop-loss Density and Liquidation Cascades." SSRN, doi:10.2139/ssrn.7293038. [⚠️ 이번에도
  접근 불가 재확인 — papers.ssrn.com/doi.org/bare ssrn.com 전부 403, 이 세션의 SSRN 접근
  자체가 막힌 것으로 보임(이 논문만의 문제 아님)]
- Deribit Insights. "Crypto Derivatives Exchanges: Liquidation Pioneers." [원문대조완료]
- Glassnode Research. "Pressure Points: Liquidation Heatmaps & Market Bias." [원문대조완료,
  서술적 자료로 백테스트 없음]
- FTI Consulting. "Crypto Crash Oct 2025: Leverage Meets Liquidity." [원문대조완료]
- curupira.dev. "Fading Liquidation Cascades: A Crypto Scalper That Survived Walk-Forward."
  [원문대조완료, 개인블로그·비동료검토]
- Amberdata. "How $3.21B Vanished in 60 Seconds." [원문대조완료]

### Part B — V자 반등/반전

- Jegadeesh, N. (1990). "Evidence of Predictable Behavior of Security Returns." *J. Finance*
  45(3). DOI:10.1111/j.1540-6261.1990.tb05110.x [OpenAlex 대조완료]
- Lehmann, B.N. (1990). "Fads, Martingales, and Market Efficiency." *QJE* 105(1).
  DOI:10.2307/2937816 [OpenAlex 대조완료]
- De Bondt, W. & Thaler, R. (1985). "Does the Stock Market Overreact?" *J. Finance* 40(3).
  DOI:10.1111/j.1540-6261.1985.tb05004.x [OpenAlex 대조완료]
- Da, Z., Liu, Q., Schaumburg, E. (2013). "A Closer Look at the Short-Term Return Reversal."
  *Management Science* 60(3). DOI:10.1287/mnsc.2013.1766 [원문대조완료]
- Nagel, S. (2012). "Evaporating Liquidity." *RFS* 25(7). DOI:10.1093/rfs/hhs066 [OpenAlex
  대조완료]
- Atkins, A. & Dyl, E. (1990). "Price Reversals, Bid-Ask Spreads, and Market Efficiency."
  *JFQA* 25(4). DOI:10.2307/2331015 [OpenAlex 대조완료]
- Cox, D. & Peterson, D. (1994). "Stock Returns Following Large One-Day Declines."
  *J. Finance* 49(1). DOI:10.1111/j.1540-6261.1994.tb04428.x [OpenAlex 대조완료]
- Bremer, M. & Sweeney, R. (1991). "The Reversal of Large Stock-Price Decreases."
  *J. Finance* 46(2). DOI:10.1111/j.1540-6261.1991.tb02684.x [OpenAlex 대조완료]
- Zaremba, A. et al. (2021). "Up or down? Short-term reversal, momentum, and liquidity
  effects in cryptocurrency markets." *IRFA* 78. DOI:10.1016/j.irfa.2021.101908 [OpenAlex
  대조완료]
- Osler, C. (2003). "Currency Orders and Exchange Rate Dynamics." *J. Finance* 58(5).
  DOI:10.1111/1540-6261.00588 [OpenAlex 대조완료]
- Osler, C. (2005). "Stop-Loss Orders and Price Cascades in Currency Markets." *JIMF* 24(2).
  DOI:10.1016/j.jimonfin.2004.12.002 [RePEc 대조완료]
- Carlin, Lobo, Viswanathan (2007). "Episodic Liquidity Crises." *J. Finance* 62(5).
  DOI:10.1111/j.1540-6261.2007.01274.x [OpenAlex 대조완료]
- Leung, T. & Li, X. (2015). "Optimal Mean Reversion Trading with Transaction Costs and
  Stop-Loss Exit." *IJTAF* 18(3). DOI:10.1142/S021902491550020X [OpenAlex 대조완료 — 진입구간
  도출 포함 재확인]
- Zhang, H. & Zhang, Q. (2008). "Trading a mean-reverting asset: Buy low and sell high."
  *Automatica* 44(6). DOI:10.1016/j.automatica.2007.11.003 [⚠️ 서지사항만 확인, 초록 미검증]
- Song, Q. & Zhang, Q. (2013). "An optimal pairs-trading rule." *Automatica* 49(10).
  arXiv:1302.6120 [원문대조완료]
- Bertram, W. (2010). "Analytic solutions for optimal statistical arbitrage trading."
  *Physica A* 389(11). DOI:10.1016/j.physa.2010.01.045 [⚠️ 서지사항만 확인, 초록 미검증]
- Lo, A., Mamaysky, H., Wang, J. (2000). "Foundations of Technical Analysis." *J. Finance*
  55(4). NBER WP 7613. DOI:10.1111/0022-1082.00265 [OpenAlex 대조완료]

## How to apply

"청산 스파이크/V자 반등 어떻게 진입·청산해야 하냐"류 재질문 시 이 문서부터 — 특히 (1) fade는
이론적으로 탄탄하나 크립토 백테스트 근거는 약함 (2) 30~60분 해상도 반전은 학술적 공백지대라
"검증됨"으로 과장하면 안 됨 (3) Leung & Li (2015) 재독해가 진입설계의 가장 저렴한 다음 수 (4)
호가바운스 아티팩트 가설이 V_REBOUND 경제성 미스터리에 대한 새 후보 설명이라는 점을 언급할 것.
SSRN 접근이 이 세션에서 전면 차단됐다는 사실도 재확인 시 참고(논문 특정 문제 아님).
