# 이런 지표를 퀀트 업계는 실제로 어떻게 쓰는가 — 문헌·실무 조사 (2026-08-15, 퀀트활용 서브프로젝트 #3)

상태: **완료(조사, 신규 실험 없음).** 결론: 우리가 측정한 크기(이벤트당 0.24bp, 횡단면 rank IC
0.027)는 업계 기준 **"알파 계층이 아니라 실행·마켓메이킹 계층의 숫자"**다. 이 계열 신호를 방향
베팅으로 쓰는 실무는 사실상 없고, 표준 용법은 (1) 호가 공정가 조정, (2) 실행 알고리즘의 자식주문
타이밍, (3) 회전율을 공격한 잔차 기반 팩터 3가지다.

## 왜 이 조사를 했는가

서브프로젝트 #1(횡단면 IC)·#2(브레드스)가 같은 형태로 닫혔다 — 신호는 진짜인데 더 단순한 공짜
벤치마크 대비 증분이 없거나, 있어도 비용의 1/40 크기다. 남은 질문은 "그럼 이런 신호를 실제로
돈으로 바꾸는 사람들은 무엇을 다르게 하는가"이며, 그건 우리 데이터가 아니라 외부 관행에 대한
질문이라 조사로 답한다.

## 발견 1 — 마이크로구조 신호의 표준 용법은 방향 베팅이 아니라 **호가 공정가 조정**

마켓메이킹에서 알파 신호는 포지션을 잡는 데 쓰이지 않고 **호가를 거는 중심 가격을 옮기는 데**
쓰인다. `hftbacktest`의 APT 튜토리얼이 그 구조를 그대로 보여준다:

```
fair_px  = (1 + beta * spot_return + alpha) * futures_past_px
bid_price = fair_px * (1 - relative_bid_depth)
ask_price = fair_px * (1 + relative_ask_depth)
```

즉 알파는 **어디를 중심으로 양쪽에 호가를 걸지**를 정하고, 수익은 여전히 스프레드에서 나온다.
Cartea & Wang의 정식화도 같다 — 마켓메이커는 알파 신호로 (a) 역선택 비용을 줄이고 (b) 재고
리스크를 관리하며, 방향 베팅은 부수적이다.

**이것이 결정적인 이유**: 이 구조에서는 **신호가 거래비용을 이길 필요가 없다.** 스프레드를 받고
(바이낸스 선물 최대 메이커 리베이트 0.005% = 0.5bp) 신호는 "역선택을 덜 당하게" 하는 역할만 한다.
우리 0.24bp짜리 신호가 무의미한 게 아니라, **부호가 반대인 비용 구조에 놓여야** 의미가 생긴다는
뜻이다. 우리 실험들은 전부 테이커로 스프레드를 **지불하면서** 이 신호로 방향을 맞히려 했다.

## 발견 2 — 마이크로구조 신호의 예측력은 초~분 단위로 감쇠하며, 크기는 원래 이 정도다

주문흐름 불균형(OFI)은 단기 가격 변화의 근사 선형 예측자로 마이크로구조에서 가장 재현성 높은
관계 중 하나이고, 트레이딩 회사들의 단기 예측 모델에 여전히 핵심 입력으로 들어간다. 다만 그
예측력은 **초에서 분, 길어야 시간 단위로 감쇠**한다.

참고로 한 OFI 연구가 보고한 IC는 평균 +0.0044, OOS +0.0022 수준이다. 우리 횡단면 rank IC
0.027과는 구성이 달라(관측 단위 수익률 상관 vs 횡단면 순위 상관) 직접 비교는 부적절하지만,
**우리 신호가 마이크로구조 기준으로 약한 편이 아니라는 정황**은 된다. 병목은 신호 강도가 아니라
구현 마찰이라는 우리 측정과 일치한다.

## 발견 3 — 단기반전이 비용 후에도 살아남는 조건은 "신호"가 아니라 "회전율·유니버스"

de Groot, Huij, Zhou(*Journal of Banking & Finance*, 2012)는 우리와 정확히 같은 문제를 다룬다.
핵심 결과:

- 단기반전 전략은 **거래비용 차감 후에도 주당 30~50bp**를 낸다.
- 비용이 수익성을 잡아먹는 현상은 전략의 결함이 아니라 **소형주를 과도하게 거래하는 데서** 온다.
- 처방 두 가지: **(a) 유니버스를 대형주로 제한**, **(b) 회전율을 낮추는 포트폴리오 구성 알고리즘**
  적용.

**우리 실험은 정확히 이 논문이 지목한 실패 구성이었다** — 60개 전부(1000RATS·WAXP·TLM·RIF 등
소형 알트 포함)를 h bar마다 전량 리밸런싱했다. 회전율을 공격하는 축은 미시도다.

**단, 크립토 특이사항 경고**: 크립토에서는 일간 반전이 **비유동 코인에 집중**되고 최상위 유동
코인들은 오히려 일간 모멘텀을 보인다는 연구가 있다. 주식의 "대형주로 좁혀라" 처방이 그대로
이식되지 않을 수 있고, **알파가 있는 곳이 곧 비용이 가장 큰 곳**이라는 고전적 함정일 수 있다 —
이건 우리 패널로 직접 검증 가능한 명제다(유동성 구간별 IC/비용 분해).

## 발견 4 — 잔차화(residualization)는 표준 처방이고, 우리는 이미 그 숫자를 갖고 있다

Blitz, Huij, Lansdorp, Verbeek(*Journal of Financial Markets*, 2013)의 단기 **잔차** 반전: 팩터
노출을 제거한 잔차 수익률로 반전을 구성하면 통상 반전 대비 **위험조정 수익이 2배**이고, **대형주로
제한해도 비용 후 유의**하다. 통상 반전이 시변 팩터 노출을 갖고 그게 수익에 마이너스·리스크에
플러스로 작용하는데, 잔차화가 그걸 제거한다는 게 메커니즘이다.

우리 #1이 계산한 **"반전 중립화 후 잔차 IC +0.0103(t=9.7, h=12)"**이 바로 이 계열의 양이다. 우리는
그것을 "증거 신호가 반전을 넘어 남기는 증분"으로만 해석했지만, 문헌 프레임에서는 **잔차 반전
전략의 입력**으로 볼 수 있다 — 같은 숫자의 다른 용도다.

## 발견 5 — 이론이 우리 상황을 정확히 예측한다 (Grinold 근본법칙)

`IR ≈ IC × √BR`. 우리 IC(0.027)에 60종목 × 연 수천 회 리밸런싱의 breadth를 곱하면 이론 IR은 매우
크고, 실제로 우리 총수익 기준 Sharpe가 1.74로 나왔다. 근본법칙은 **구현이 무마찰이라고 가정**하며,
마찰은 transfer coefficient로 새어나간다. 즉 우리가 관측한 "총수익은 좋은데 순수익이 −100%"는
이론적 이상현상이 아니라 **근본법칙이 예측하는 정확한 실패 지점**이다. 함의: 병목은 신호 탐색이
아니라 구현이며, **다음 노력은 새 지표가 아니라 마찰에 투입되어야 한다.**

## 우리 상황에 대한 함의 (세 갈래, 우선순위 순)

| 경로 | 하는 일 | 우리에게 필요한 변화 | 현실성 |
|---|---|---|---|
| **A. 실행·메이킹 계층으로 이동** | 신호를 진입 트리거가 아니라 지정가 배치/호가 스큐에 사용 — 스프레드를 **받으면서** 역선택을 줄임 | 테이커 방향베팅 → 메이커 운영으로 전환. 레이턴시·재고·거래소 티어·체결 인프라가 전부 새 문제 | **신호 문제가 아니라 사업 구조 결정.** 현 봇 구조와 불연속 |
| **B. 회전율·유니버스 공격** | 매 bar 전량 리밸런싱 폐기, 밴드/최적화 기반 부분 리밸런싱, 유동성 구간 제한, 홀딩 연장 | 기존 패널로 즉시 검증 가능(신규 데이터 불필요) | **가장 싸고 즉시 가능.** 단 크립토 유동성-반전 관계가 주식과 반대일 수 있어 먼저 분해 필요 |
| **C. 잔차 반전 + 팩터 결합** | 단독 알파 포기, 반전 중립 잔차를 다중신호 북의 한 축으로 | #1의 잔차 IC를 그대로 입력으로 사용 | 이미 계약 문서 후보 #2(패널 DL 피처)와 같은 방향 |

**정직한 요약**: 우리가 잰 0.24bp는 "이 지표가 쓸모없다"가 아니라 **"이 지표는 이 계층의 숫자"**라는
뜻이다. 같은 신호로 돈을 버는 곳은 비용 부호가 반대인 계층(메이킹)이거나, 회전율을 구조적으로
줄인 팩터 구현이다. 방향 베팅 계층으로 끌어올리려면 신호가 아니라 **비용 구조**를 바꿔야 한다.

## 다음 단계 제안 (미착수)

- **B-1 (권고, 저비용)**: 기존 패널에서 **유동성 구간별로 IC와 비용을 분해**한다 — 상위 10/20/60
  심볼 그룹별 IC, 그룹별 실효 비용 가정, 손익분기 bp. 크립토에서 "알파가 비유동 구간에만 있는가"를
  우리 데이터로 직접 확인하는 것이며, de Groot 처방이 이 시장에 이식 가능한지가 여기서 갈린다.
  킬 기준: 어느 유동성 그룹에서도 손익분기 bp가 현실 비용(테이커 왕복 ~10bp / 메이커 ~2~4bp)을
  넘지 못하면 B 경로 종결.
- **A는 제안하되 실험으로 열지 않는다** — 메이커 운영 전환은 리서치 결과가 아니라 운영 결정이며,
  이 서브프로젝트의 범위를 넘는다.

## 준수 확인

신규 실험·학습·백테스트 없음(외부 문헌·실무 자료 조사). 인용한 수치는 전부 출처 표기했고, 우리
측정치와 구성이 다른 경우(OFI 논문의 IC) 직접 비교 부적절함을 본문에 명시했다. 라이브 파일 미변경.

## 출처

- [Order Flow Imbalance Prediction (개관)](https://www.emergentmind.com/topics/order-flow-imbalance-prediction)
- [Order Flow Imbalance — A High Frequency Trading Signal (Dean Markwick)](https://dm13450.github.io/2022/02/02/Order-Flow-Imbalance.html)
- [Predictive Order Flow Imbalance: Cross-Asset Microstructure Alpha (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=7053198)
- [Market Making with Alpha — APT 튜토리얼 (hftbacktest)](https://hftbacktest.readthedocs.io/en/latest/tutorials/Market%20Making%20with%20Alpha%20-%20APT.html)
- [Cartea & Wang, Market Making with Alpha Signals (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3439440)
- [de Groot, Huij, Zhou — Another Look at Trading Costs and Short-Term Reversal Profits (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1605049) · [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0378426611002263)
- [Blitz, Huij, Lansdorp, Verbeek — Short-Term Residual Reversal (SSRN)](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1911449) · [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1386418112000468)
- [Up or down? Short-term reversal, momentum, and liquidity effects in cryptocurrency markets (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S1057521921002349)
- [Grinold — Fundamental Law of Active Management (해설)](https://blankcapitalresearch.com/learn/grinold-fundamental-law-active-management) · [The Fundamental Law: Time Series Dynamics (NYU)](https://math.nyu.edu/inmemoriam/avellaneda/FundamentalLawFT.pdf)
