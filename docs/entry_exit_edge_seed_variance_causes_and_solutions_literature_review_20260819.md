# 진입/청산 엣지 모델의 시드 분산 — 원인·증폭 메커니즘 외부 문헌 리뷰 (2026-08-19)

## 배경

2026-08-19 다른 세션이 오메가4.6.1 라이브(h48qual+zig075 dual) 승격 근거가 사실상 단일 시드
(260620)뿐이었음을 발견하고, N=3 예비 재현 시드검증(94046540/524707103 추가)을 실행했다
(`docs/experiments/eth_live_promotion_seed_robustness_3seed_20260819.md`). 결과: **6개 평가창
중 4개(2025q2/2025q3/VAL/oos_q2)에서 부호가 시드에 따라 뒤집혔다** — VAL 자체가
+54.88%→+91.44%→-16.48%로 반대 결론에 도달했다. N=3이라 이 저장소 정책(Seed-Diversity
Ensemble Promotion Gate, N≥5)에는 못 미치는 예비 신호이지만, 이 저장소에서 나온 시드분산
발견 중 가장 심각한 실사용 리스크다.

이 문서는 사용자 요청에 따라 (1) 이런 시드분산이 **왜** 생기고 무엇으로 줄일 수 있는지
(Part A), (2) 분류 성능의 작은 차이가 **왜 트레이딩 백테스트의 부호 자체**를 뒤집는 수준까지
증폭되는지(Part B)를 외부 문헌으로 조사한 결과다. 이 저장소가 이미 읽고 구현·테스트까지 마친
문헌(Prechelt UP_4, Arpit et al. 메모리제이션, GCE/ELR/mixup, AdaBelief/RAdam, LoMETab,
NCL, "Don't stop me now" arXiv:2602.22107, Google Tuning Playbook 등)은 재조사하지 않고
필요한 곳에서만 짧게 연결한다.

## 조사 방법 및 한계

- `paper-lookup` 스킬로 조사를 시작했으나, 세션 내내 arXiv의 `export.arxiv.org` API가
  `Rate exceeded` / HTTP 503으로 막혀 있었다(재시도·대기 후에도 동일) — 원인은 이 샌드박스
  환경의 공유 egress를 다른 세션들이 동시에 쓰고 있을 가능성이 높다고 판단했다. 우회책으로
  arXiv 논문은 전부 `WebFetch`로 `arxiv.org/abs/...` 웹 프론트를 직접 읽어 확인했다(API가
  아니라 사람이 보는 페이지라 스로틀이 달랐다). 이 우회 경로는 초록만 주는 경우가 많아,
  본문 수치가 필요한 항목(Jordan 2023, Colas et al. 2018)은 `ar5iv.labs.arxiv.org`의 HTML
  풀텍스트를 별도로 읽었다.
- 퀀트금융 4편(Bailey 그룹)은 arXiv에 없고 SSRN/저널에 있다. `davidhbailey.com/dhbpapers/`
  개인 사이트에서 원문 PDF를 직접 내려받아, `WebFetch`의 PDF 파싱이 실패한 3편은 `pypdf`로
  텍스트를 직접 추출해 원문을 읽었다(아래 각주에 파일명 명시). **이 4편은 초록이 아니라 본문
  전체(수식·수치 예시 포함)를 직접 읽고 인용한다.**
- 항목마다 "WebFetch 원문 확인" 여부를 명시한다. 확인 못 한 세부사항(예: 2603.01820의 정확한
  시드 개수)은 추측하지 않고 "확인 못 함"으로 남긴다.
- 이 문서는 순수 문헌 조사 + 저장소 기존 아티팩트 재해석이다. 재학습·코드 실행은 하지 않았고,
  §B2의 계산은 이 세션이 확인된 공식 위에서 직접 수행한 것이지 논문에 있는 숫자가 아니다(표에
  명시).

---

## Part A — 학습 자체의 시드분산: 원인과 완화

### A1. 진짜 딥앙상블 이론과 이 저장소의 BatchEnsemble 붕괴

| 문헌 | 확인 |
|---|---|
| Lakshminarayanan, Pritzel, Blundell, "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles," arXiv:1612.01474 (2016/2017) | WebFetch 원문(초록) 확인 |
| Fort, Hu, Lakshminarayanan, "Deep Ensembles: A Loss Landscape Perspective," arXiv:1912.02757 (2019) | WebFetch 원문(초록) 확인 |
| Wen, Tran, Ba, "BatchEnsemble: An Alternative Approach to Efficient Ensemble and Lifelong Learning," arXiv:2002.06715 (2020) | WebFetch 원문(초록) 확인 |
| Gorishniy, Kotelnikov, Babenko, "TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling," arXiv:2410.24210 (ICLR 2025) | WebFetch 원문(초록) 확인 |

Lakshminarayanan(2017)이 원조 딥앙상블이고, Fort et al.(2019)이 "왜 되는가"를 손실지형
관점에서 설명한 후속작이다. 핵심 주장(원문 인용): 무작위 초기화는 서로 "entirely different
modes"를 탐색하는 반면, 학습 궤적 자체는 예측공간에서 "cluster within single modes"에
머문다 — 즉 앙상블의 힘은 **init 다양성이 만드는 mode 간 이동**에서 나오지, 학습 자체가
다양성을 만드는 게 아니다. 논문은 이를 "subspace sampling"(같은 mode 내부에서 가중치를
흔드는 방법들) 과 명시적으로 대조하며, subspace sampling은 이 정도의 decorrelation을 못
준다고 결론짓는다.

**이 저장소와의 연결**: `eth_odyssey4_batchensemble_collapse_and_quality_head_duplication_20260816`
가 실측한 BatchEnsemble(k=8) 멤버 간 pairwise correlation 0.997-0.999 붕괴는 Fort et al.의
"subspace sampling은 다양성을 못 만든다"는 주장이 정확히 예측하는 실패모드다. BatchEnsemble의
`input_scale`/`expert_scale` rank-1 게이트는 공유 trunk 하나를 살짝 흔드는 구조 — Fort et
al.의 분류법으로는 "다른 mode 탐색"이 아니라 "같은 mode 내부 subspace sampling"에 해당한다.
원조 BatchEnsemble 논문(Wen et al. 2020) 자체도 초록에서 "효율적 근사"(3배속·3배 메모리
절감)만을 목표로 명시하고 멤버 간 다양성을 설계 목표로 주장하지 않는다 — 이 저장소의 붕괴
발견은 BatchEnsemble 저자들의 주장을 반박하는 게 아니라, 애초에 그 주장 범위 밖의 일(다양성
보장)을 기대했던 것이라는 재해석이 가능하다. 이 저장소가 실제로 쓰는 TabM(Gorishniy et al.
2024/2025) 원논문도 같은 톤이다: "the multiple predictions of TabM are weak individually,
but powerful collectively" — 개별 멤버가 약한 게 설계상 정상이라는 뜻이며, 이 역시 "다양성
없음"이 버그가 아니라 이 아키텍처 계열의 정직한 설계 한계라는 이 저장소의 기존 결론(6개
독립 개입 전부 CLOSED)과 정합적이다.

**새로운 시사점 — 저장소가 아직 안 짚은 구분**: 이 저장소가 "시드분산" 문제로 지금 겪는 건
BatchEnsemble 내부 k=8 멤버가 아니라, **전체 학습을 처음부터 다른 초기화·미니배치 순서로 다시
돈 outer 시드**(260620/94046540/524707103)다. Fort et al.의 이론에 따르면 이런 완전
독립재학습은 정확히 "서로 다른 mode를 탐색"하는 좋은 사례여야 한다 — 즉 VAL이 +54%/+91%/
-16%로 크게 갈린 것 자체가, 이 이론이 맞다면 "진짜 함수공간 다양성이 존재한다"는 신호로
해석할 수도 있다. 문제는 다양성의 부재가 아니라, 그 다양성을 딥앙상블 이론이 처방하는 방식
(§A2 참고, 평균/집계)이 아니라 "시드 하나를 골라 대표로 승격"하는 방식으로 쓰고 있다는
데 있을 가능성이 있다. §A5에서 이 가설을 검증할 구체적 다음 액션을 제안한다.

### A2. Mode connectivity — 왜 다른 시드는 다른 basin에 떨어지고, 사후 평균이 왜 위험한가

| 문헌 | 확인 |
|---|---|
| Garipov et al., "Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs," arXiv:1802.10026 (2018) | WebFetch 원문(초록) 확인 |
| Frankle, Dziugaite, Roy, Carbin, "Linear Mode Connectivity and the Lottery Ticket Hypothesis," arXiv:1912.05671 (2019/2020) | WebFetch 원문(초록) 확인 |
| Ainsworth, Hayase, Srinivasa, "Git Re-Basin: Merging Models modulo Permutation Symmetries," arXiv:2209.04836 (2022) | WebFetch 원문(초록) 확인 |
| Wortsman et al., "Model soups...," arXiv:2203.05482 (ICML 2022) | WebFetch 원문(초록) 확인 |

Garipov et al.은 독립적으로 학습된 두 최적점이 직선으로는 연결 안 되지만(그 사이 손실이
높다), 굽은 경로를 찾으면 "optima...are in fact connected by simple curves over which
training and test accuracy are nearly constant"임을 보였다 — 이게 FGE(Fast Geometric
Ensembling)의 근거다. Frankle et al.은 표준 비전 모델이 "SGD noise"(데이터 순서 등)에
대해 "become stable...early in training"함을 보여, 언제부터 서로 다른 시드가 (선형으로)
연결 가능한 basin에 정착하는지를 다룬다. Ainsworth et al.(Git Re-Basin)은 한 걸음 더 나가
은닉유닛의 순열대칭(permutation symmetry)을 정렬하면 "거의 하나의 basin"만 존재한다고
주장하며, 독립적으로 학습된 두 ResNet 사이에 "zero-barrier linear mode connectivity"를
직접 시연했다 — 단, 그러려면 **먼저 순열정렬**이 필요하다는 게 핵심 전제다.

Wortsman et al.(Model soups)은 여러 모델의 **가중치를 그냥 평균**내는 것만으로 정확도가
오르는 현상을 보였지만, 이는 **동일한 사전학습 체크포인트에서 시작한 파인튜닝들** 사이의
평균이다 — 즉 이미 같은 basin 근방에 있다는 전제가 깔려 있다.

**이 저장소와의 연결 — SWA보다 한 단계 더 간 버전이 있는가에 대한 답**: 사용자가 물은 "SWA
(미구현)보다 한 단계 더 간 버전"인 Model soups는, 이 저장소가 지금 그대로 가져다 쓰기에는
**전제 조건이 다르다**. 이 저장소의 N개 시드는 Wortsman의 세팅(같은 사전학습 시작점)이 아니라
Garipov/Frankle/Ainsworth이 다루는 세팅(처음부터 서로 다른 무작위 초기화로 학습)이다 — 이
경우 순열정렬 없는 naive 가중치 평균은 mode-connectivity 문헌이 정확히 경고하는 "두 최적점
사이 손실 언덕 위에 착지"가 될 위험이 크다. 즉 **"N개 시드 가중치를 그냥 평균해서 승격
후보로 쓰자"는 아이디어는 이 문헌들에 비추어 볼 때 위험하다** — 시도한다면 Git Re-Basin류
순열정렬을 먼저 적용하거나, 아예 가중치가 아니라 **예측(확률) 평균**(=원조 딥앙상블
그 자체, Lakshminarayanan)으로 가는 것이 안전하다. §A5·"지금 시도해볼 것"에서 이를
구체적 우선순위로 제안한다.

### A3. 재현성 위기 — RL 특유인가, 지도학습 분류기에도 적용되는가

| 문헌 | 확인 |
|---|---|
| Henderson, Islam, Bachman, Pineau, Precup, Meger, "Deep Reinforcement Learning that Matters," arXiv:1709.06560 (2017/AAAI 2018) | WebFetch 원문(초록) 확인 |
| Reimers & Gurevych, "Why Comparing Single Performance Scores Does Not Allow to Draw Conclusions About Machine Learning Approaches," arXiv:1803.09578 (2018) | WebFetch 원문(초록) 확인 |
| Bouthillier et al., "Accounting for Variance in Machine Learning Benchmarks," arXiv:2103.03098 (2021) | WebFetch 원문(초록) 확인 |
| Picard, "Torch.manual_seed(3407) is all you need...," arXiv:2109.08203 (2021) | WebFetch 원문(초록) 확인 |
| Jordan, "On the Variance of Neural Network Training with respect to Test Sets and Distributions," arXiv:2304.01910 (2023) | WebFetch 원문(**본문**, ar5iv HTML) 확인 |

사용자가 명시적으로 물은 질문("RL에서 시드가 결과를 지배한다는 게 지도학습 3-head
분류기에도 적용되는 메커니즘인지, RL 특유 문제인지")에 대한 답: **RL 특유가 아니다, 이미
순수 지도학습 분류기에서도 잘 문서화돼 있다.** Henderson et al.은 RL에 추가로 있는 증폭축
(정책이 곧 자신의 학습 데이터 분포를 만드는 비정상성, 보상 희소성 등)을 다루지만, Reimers &
Gurevych(2018)는 **NER 분류기**(RL과 무관, 이 저장소의 3-head 분류기와 훨씬 가까운 세팅)에서
단일 점수 비교 시 "in up to 26%의 사례에서 type I error(우연을 유의한 차이로 오판)"가
발생함을 보였다. Bouthillier et al.(2021)은 5개 딥러닝 태스크 전반(비전 포함, RL 아님)에서
데이터 샘플링·증강·초기화·HP 선택이 만드는 분산을 분해했고, Picard(2021)는 순수 비전
분류(CIFAR-10/ImageNet)에서 "평균보다 훨씬 잘/훨씬 못하는 아웃라이어 시드를 찾는 게
놀랍도록 쉽다"는 걸 최대 10^4개 시드로 실증했다.

가장 정량적으로 유용한 건 Jordan(2023, 본문까지 확인)이다. CIFAR-10 64-epoch 학습에서
**테스트셋 정확도의 시드 간 표준편차는 0.15%, 범위(최대-최소)는 1.4%**인데, 반해 **진짜
분포 기준 표준편차는 0.033%로 20배 작다**. 테스트셋을 반으로 쪼개 한쪽에서 상위 1/4 시드가
다른 쪽에서 얼마나 더 잘하는지 봤더니 겨우 0.02% — 즉 "이번에 좋아 보인 시드"는 거의 항상
유한샘플 노이즈이고 진짜 일반화 개선과 상관이 없다("improvements on the test-set due to
re-training will have near-zero correlation with improvements on unseen data"). 이는 이
저장소가 우려하는 정확히 그 지점 — "시드 260620이 우연히 좋았을 가능성" — 을 정량적으로
뒷받침하는 독립 선례다. 다만 Jordan의 세팅은 (가) 분류 정확도 자체를 보고 하류 트레이딩
로직이 없으며, (나) 셔플/증강 시드만 다르지 이 저장소처럼 데이터 분할 시점의 OOF 구조
자체가 프로즌돼 있진 않다는 차이는 있다.

**결론**: RL 문헌(Henderson)을 인용할 근거는 있지만, "우리는 RL이 아니니까 그 문제는
안 겪는다"고 이 저장소 문제를 예외 처리할 근거는 없다 — 지도학습 분류 문헌만으로 이미
충분히 예상 가능한 현상이다.

### A4. 몇 개 시드면 충분한가 — 통계적 검정력 문헌

| 문헌 | 확인 |
|---|---|
| Colas, Sigaud, Oudeyer, "How Many Random Seeds? Statistical Power Analysis in Deep RL Experiments," arXiv:1806.08295 (2018) | WebFetch 원문(**본문**, ar5iv HTML) 확인 |
| Card, Henderson, Khandelwal, Jia, Mahowald, Jurafsky, "With Little Power Comes Great Responsibility," arXiv:2010.06595 (EMNLP 2020) | WebSearch로 초록/핵심결과 확인(원문 미직접fetch) |

Colas et al.은 t-검정/부트스트랩 검정 각각에 대해 필요한 시드 수를 효과크기·분산·목표
검정력(β)의 함수로 유도한다. 자신들의 DDPG 예제에서 **N=5 시드로는 β=0.51(검정력 49%,
동전던지기와 다를 바 없음)**이었고, β=0.2(검정력 80%)를 달성하려면 **N=10**이 필요했다.
Henderson et al.(2017)이 리뷰한 딥RL 논문 대부분이 "5개 이하 시드"를 썼다는 걸 명시적으로
비판하며, 최종 권고는 "파일럿 스터디에 최소 N=20을 쓰고, 검정력분석이 처방하는 것보다 더
큰 N을 쓰라"다. Card et al.(NLP 검정력분석)도 같은 결론 계열이다 — 예컨대 기계번역
표준 2000문장 테스트셋은 1 BLEU 차이를 탐지하는 데 겨우 약 75% 검정력이라는 식으로, 표준
벤치마크 규모가 체계적으로 underpowered임을 보인다.

**이 저장소와의 연결**: CLAUDE.md의 "N≥5"는 이 저장소 스스로도 "관행이지 유도된 숫자가
아니다"라고 명시한 그대로다 — Colas et al.의 자기 예제 기준으로 보면 **N=5는 오히려
검정력이 절반도 안 되는(β=0.51) 수준일 수 있다.** 다만 이건 그들의 특정 효과크기·분산
조합에서 나온 숫자이지 보편 상수가 아니다("문헌은 있지만 안 맞을 수 있음" 절 참고) — 이
저장소 고유의 효과크기(오늘 N=3 데이터의 시드 간 표준편차)로 직접 재계산하는 게 다음
액션으로 더 정확하다(§우선순위 4).

### A5. 저SNR 표형 금융데이터 시드분산 — 공백 재확인, 부분적으로 채워짐 + 상충하는 결과

| 문헌 | 확인 |
|---|---|
| Saidd, "A Controlled Comparison of Deep Learning Architectures for Multi-Horizon Financial Forecasting: Evidence from 918 Experiments," arXiv:2603.16886 (2026) | WebFetch 원문(초록) 확인 |
| Saly-Kaufmann, Wood, Peter-Calliess, Zohren, "Deep Learning for Financial Time Series: A Large-Scale Benchmark of Risk-Adjusted Performance," arXiv:2603.01820 (2026) | WebFetch 시도, **PDF 본문에서 시드 관련 절을 못 찾음** — 초록의 "robustness to random seed selection" 언급만 확인, 수치 미확인 |

기존에 "저SNR 표형/금융 데이터의 시드분산을 전문으로 다룬 서베이는 없다"고 확인됐던 공백을
재검색한 결과, **완전 공백은 아니었다.** 두 편 다 2026년 논문이고 둘 다 시드를 평가축에
넣는다는 점에서 이전 확인 시점(이 세션 이전)보다 상황이 바뀌었을 수 있다.

`arXiv:2603.16886`은 초록에서 명시적으로 "architecture explains nearly all performance
variance, while **seed randomness is negligible**"라고 결론짓는다 — 이는 **이 저장소의
경험과 정면으로 배치**된다. `arXiv:2603.01820`(Zohren 그룹, 선물데이터 2010-2025)은
"robustness to random seed selection"을 평가축으로 명시하나, 이번 세션에서는 PDF에서
구체적 수치(시드 개수, 분산 크기, 순위 역전 여부)를 확인하지 못했다 — **솔직히 미확인으로
남긴다.**

**왜 상충하는지에 대한 이 문서의 해석(문헌이 직접 답하지 않음, 추정임을 명시)**:
1. 그 논문들은 아키텍처 간 **랭킹/평균 지표**(어떤 아키텍처가 이기는가)에서 시드분산을
   보는 반면, 이 저장소는 threshold-gated 진입/청산이 낀 경로의존적 **개별 캘린더 창** 별
   PnL을 본다 — 집계 시 상쇄되는 분산이 개별 창 단위에서는 증폭돼 보일 수 있다.
2. 그 논문들의 표본기간(918-exp 논문 범위 미확인, Zohren 그룹은 2010-2025 = 15년)이 이
   저장소의 VAL(3~4개월)/OOS(3개월)보다 훨씬 길 가능성이 높다 — Part B(MinBTL)의 논리대로
   표본이 길수록 같은 N에서 선택-노이즈의 진폭이 준다.
3. "seed randomness negligible"이 정확히 무엇을 고정하고 무엇을 바꾼 시드인지(데이터
   분할까지 포함한 outer 시드인지, 초기화·셔플만인지) 이번 세션에서 원문 본문까지는
   확인 못 했다.

**결론**: "저SNR 표형 금융데이터에서 시드분산을 다루는 문헌 자체가 없다"는 공백은 이제
정확한 서술이 아니다 — 있는데, 이 저장소와 **반대 결론**을 내는 논문이 최소 1편 있다.
하지만 "왜 작은 정확도 차이가 threshold-gated 백테스트의 부호반전으로 증폭되는가"라는
이 저장소 고유의 정밀한 질문에 답하는 문헌은 여전히 못 찾았다(Part B로 이어짐).

---

## Part B — 하류 백테스트 PnL 부호반전 증폭: 퀀트금융 문헌

이 절의 4편은 전부 David Bailey / Marcos López de Prado 그룹의 논문이다. 3편은 본문(PDF)을
`pypdf`로 직접 추출해 읽었다(초록만이 아니라 수식과 수치 예시까지): `overfitting.pdf`(SEBO
데모), `backtest-pseudo.pdf`(AMS Notices 이론), `sharpe-frontier.pdf`(PSR/MinTRL). 나머지
한 편(`deflated-sharpe.pdf`)은 `WebFetch`가 PDF에서 핵심 개념·공식을 성공적으로 요약해줬으나
(파싱 실패는 아니었다), `pypdf` 전체 텍스트 추출까지는 이번 세션에서 하지 않았다 — §B3에서
"초록·개념 수준"으로 명시한다.

### B1. PBO/CSCV — 이 저장소 상황(시드만 다른 N개 백테스트)에 직접 적용 가능한가

| 문헌 | 확인 |
|---|---|
| Bailey, Borwein, López de Prado, Zhu, "The Probability of Backtest Overfitting," SSRN 2326253 / *Journal of Computational Finance*, DOI: 10.21314/JCF.2016.322 (2014 SSRN / 2016 저널) | WebSearch+WMU 리포지토리 페이지로 초록·프레임워크 확인 (CSCV 원논문 본문 자체는 못 구함) |
| Bailey, Ger, López de Prado, Sim, Wu, "Statistical Overfitting and Backtest Performance," SSRN 2507040 (2014-10-07) — PBO의 자매 데모 논문(SEBO 툴) | **본문 전체 직접 읽음**(pypdf 텍스트 추출, 10쪽) |

PBO 원논문(CSCV 수식 자체)의 본문은 이번 세션에서 구하지 못했다(davidhbailey.com 개인
사이트 목록에 정확히 이 제목으로는 없었고, SSRN 배송 링크는 접근 실패) — 프레임워크
설명("model-free and nonparametric", CSCV=combinatorially symmetric cross-validation)은
초록·2차 출처 수준에서만 확인됐다. 대신 **같은 저자 그룹이 같은 해에 쓴 자매 논문(SEBO)의
본문 전체를 직접 읽었고, 이 논문이 사용자의 핵심 질문에 원저자 자신의 손으로 이미 답을
주고 있었다.**

SEBO 논문은 온라인 데모 툴로 400회 시뮬레이션을 돌리는데, 원문을 그대로 인용하면:

> "Half of the test runs use the same set of parameters except the seed for the
> pseudorandom number generator... The seeds for random number generator in SEBO
> vary from 201 to 400."

즉 **저자들 스스로가 "트라이얼"의 절반을 "전략은 완전히 동일, 오직 난수 시드만 다르게 한
N개 실행"으로 정의해서 오버피팅 실험을 했다.** 결과: In-Sample Sharpe는 0.9 근방에
집중되는데 Out-of-Sample Sharpe는 0(설계상 랜덤워크라 진짜 스킬 없음) 근방에 집중된다 —
"seed만 다른 N개 백테스트 중 최고를 고르는 것"과 "파라미터 조합 N개 중 최고를 고르는 것"이
**정확히 같은 수학**(N개 트라이얼의 수익률 시계열)으로 다뤄진다는 뜻이다.

**결론(사용자의 핵심 질문에 대한 답)**: PBO/CSCV 프레임워크는 "여러 전략 변형" 전용이
아니다 — **입력은 오직 "N개의 수익률 시계열 행렬"이며, 그 N개가 왜 서로 다른지(전략 로직이
다른지, 하이퍼파라미터가 다른지, 순수 시드만 다른지)는 방법론적으로 무관하다.** 따라서 이
저장소의 "같은 아키텍처, 시드만 다른 N개 백테스트"에 CSCV를 직접 적용하는 것은 방법론적으로
정당하다 — PBO 원논문의 CSCV 알고리즘(S개 부분집합으로 쪼개 조합적으로 IS/OOS를 구성,
각 조합에서 IS 최고 트라이얼의 OOS 순위를 logit으로 변환, logit≤0(=OOS에서 중앙값 이하로
떨어짐)의 경험적 빈도가 PBO)을 이 저장소의 N개 시드 × 6개 평가창 bar-by-bar equity curve에
그대로 먹일 수 있다.

**중요한 발견 — 이 조사 도중 이 저장소에 이미 구현·테스트된 버전이 있다는 걸 확인했다.**
외부 오픈소스(`github.com/esvhd/pypbo`, 코드 미감사)를 찾을 필요가 없었다 — `core/selection_stats.py`
(2026-07-26 생성, `test/test_selection_stats.py` 보유, `pipeline/architecture_workbench.py`의
`effect_size_gate`와 15개 이상의 `scripts/research_*_20260809.py`에서 실사용 중)가 이미
아래 5개 함수로 이 절과 §B2/§B3가 다루는 문헌 전체를 구현해 놓았다:

| 함수 | 구현 대상(원문 docstring 인용) |
|---|---|
| `expected_max_sharpe(n_trials, sr_std)` | E[max SR] over N trials under the null of no skill — §B2의 E[max_N] 공식과 **완전히 동일**(`EULER_MASCHERONI` 상수, `norm.ppf(1-1/n)` / `norm.ppf(1-1/(n*e))` 항까지 일치) |
| `probabilistic_sharpe_ratio(...)` | Bailey & Lopez de Prado (2012) — §B3의 PSR |
| `deflated_sharpe_ratio(...)` | Bailey & Lopez de Prado (2014) — §B3의 DSR, `noise_floor_sharpe`를 PSR의 benchmark로 대입 |
| `pbo_cscv(returns_matrix, n_splits)` | Bailey, Borwein, Lopez de Prado & Zhu (2017) — 이 절의 CSCV, `(periods × configurations)` 행렬을 그대로 받는다 |
| `falsification_audit(returns_matrix, ...)` | Nikolopoulos, "Spurious Predictability in Financial Machine Learning," arXiv:2604.15531 (2026, WebFetch로 원문 확인) — PBO보다 한 단계 더 나간 방법: 실제 best-of-N Sharpe를 (a) 동일 변동성의 순수 i.i.d. 가우시안 영(zero-predictability) null, (b) 각 트라이얼 자신의 수익률을 평균 0으로 강제한 뒤 블록부트스트랩한 microstructure-placebo null, 둘 다와 비교 |

즉 **§B2의 손계산은 이미 이 저장소 함수로 코드화돼 있고(수식이 정확히 일치해 상호검증이
됐다), §B1/§B3가 다루는 CSCV·PSR·DSR도 전부 이미 있다.** 더 중요한 사실: 이 도구는
**시드분산 축에는 아직 한 번도 적용된 적이 없다.** 지금까지의 용례는 전부 엔트리/피처
탐색(2026-08-09 오버나이트 세션의 17개 아이디어, VPVR 공식 탐색 등)이었고, 그 축에서는
"나이브 탐색 승자는 전부 기대 최대노이즈 바닥의 0.24배 이하, DSR은 사실상 0"이라는 결과가
이미 나와 있다(`docs/entry_exit_edge_root_cause_and_literature_review_20260809.md`) — 참고할
선례이지 시드축도 같은 결과가 나온다고 미리 단정할 근거는 아니다.

더 결정적인 사실은 `docs/pipeline_integrity_and_research_redesign_20260730.md`(C1 항목,
2026-07-30 감사, 이번 세션 이전에 이미 저장소가 스스로 남긴 기록)의 문장이다: **"다중검정
보정이 2026-07-26 이전엔 전무 — `core/selection_stats.py`가 그날 처음 생김. 그 이전 모든
승격 판단은 DSR/PBO 없이 이뤄졌다. 현재 라이브 스택도 마찬가지."** 즉 **오늘 시드분산이
발견된 바로 그 Omega4.6.1 라이브 승격(시드 260620) 자체가, 이 도구가 생기기 전에 이뤄져
DSR/PBO 검증을 한 번도 통과한 적이 없다** — 이는 이번 세션의 새 발견이 아니라 저장소가 3주
전부터 알고 있었지만 시드축에는 아직 적용하지 않은 미해결 항목이다.

같은 맥락에서, `pipeline/architecture_workbench.py`의 `validate_contract`는 이미 **코드
레벨로** CLAUDE.md의 Seed-Diversity Ensemble Promotion Gate(N≥5, 고정간격 아닌 진짜 랜덤)를
강제한다(`model.seed_ensemble_claim`이 true면 `model.seeds`가 5개 미만이거나 고정간격
클러스터면 거부) — 단, 이 계약 워크플로우 자체가 신규 라인(`architecture_workbench.py init`
경로)에만 적용되고, Omega4.6.1처럼 그 이전에 별도 스크립트로 직접 학습·승격된 라인은 애초에
이 게이트를 거친 적이 없다. 정책은 있고 코드로 강제도 되지만, 지금 문제가 된 그 승격 건
자체는 정책이 생기기 전에 그 정책 밖에서 이뤄졌다는 뜻이다.

### B2. MinBTL/E[max_N] — 이 저장소의 실제 창 길이에 대입한 계산

| 문헌 | 확인 |
|---|---|
| Bailey, Borwein, López de Prado, Zhu, "Pseudo-mathematics and financial charlatanism: The effects of backtest overfitting on out-of-sample performance," *Notices of the American Mathematical Society* 61(5):458-471 (2014) | **본문 전체 직접 읽음**(pypdf 텍스트 추출, 34쪽) |

이 논문(사실상 PBO의 이론적 자매편)이 **Minimum Backtest Length(MinBTL)**과 그 기반이 되는
**E[max_N]**(N개의 진짜 무기술(SR=0) 독립 트라이얼 중 In-Sample 최고 Sharpe의 기댓값) 공식을
유도한다. 원문에서 직접 확인한 공식(Proposition 2.1 / Eq. 2.4, 3.1):

```
E[max_N] ≈ (1-γ)·Φ⁻¹(1 - 1/N) + γ·Φ⁻¹(1 - 1/(N·e))          (연율화 전, y=1년 기준)
E[max_N, y년] ≈ E[max_N] / sqrt(y)                              (y년 배주기 재척도)
MinBTL(년) ≈ [ (1-γ)Φ⁻¹(1-1/N) + γΦ⁻¹(1-1/(Ne)) ]² / E[max_N]²  <  2·ln(N) / E[max_N]²
```
(γ≈0.5772 오일러-마스케로니 상수). 원문의 예시로 공식을 검증했다: "if only 5 years of data
are available, no more than 45 independent model configurations should be tried, or we
are almost guaranteed to produce strategies with an annualized Sharpe ratio IS of 1, but
an expected Sharpe ratio OOS of zero." 위 공식으로 직접 재현하면 N=45, y=5년일 때
E[max_N,5년]=1.000 — **정확히 일치**해 공식 이해가 맞았음을 확인했다. (§B1에서 다루듯 이
공식은 `core/selection_stats.py::expected_max_sharpe`가 이미 코드로 구현해 놓았고, 상수·
항 구조까지 동일하다 — 이 세션이 원문에서 독립적으로 재확인한 공식이 이 저장소 기존 구현과
정확히 일치한다는 것 자체가 서로에 대한 교차검증이다.)

**이 세션이 직접 계산함(논문에 없는 숫자, 이 저장소 창 길이 대입)**: 이 저장소의 실제
평가창 길이로 "진짜 기술=0인 N개 독립 트라이얼"의 기대 최대 연율화 Sharpe를 계산하면:

| N (독립 트라이얼 수) | VAL 4개월(y=0.333, CLAUDE.md 기본값) | VAL 3개월(y=0.25, 10-01 시작 시) / OOS 3개월 | OOS 분기 1.5개월(y=0.125, oos_q1/q2) |
|---|---|---|---|
| 2 | 0.90 | 1.04 | 1.47 |
| 3 | 1.48 | 1.71 | 2.41 |
| 5 | 2.07 | 2.39 | 3.37 |
| 10 | 2.73 | 3.15 | 4.45 |
| 45 | 3.87 | 4.47 | 6.32 |

역으로 "기대 최대 연율화 Sharpe=1.0이 되는 N"을 풀면 VAL(4개월)=2.13, OOS(3개월)=1.96,
OOS분기(1.5개월)=1.69 — **N이 2를 겨우 넘는 순간부터, 진짜 기술이 전혀 없는 트라이얼들만
가지고도 "그럴듯한" 연율화 Sharpe 1.0이 순전히 우연으로 기대된다.**

**한계(정직히 명시)**: (1) 이 계산은 트라이얼이 IID 정규분포·**완전 독립**이라는 가정 위에
있다 — 원논문도 "trials가 독립이 아니면 PCA 등으로 effective N을 따로 구해야 한다"고
명시한다. §A1/A2에서 다뤘듯 이 저장소의 outer 시드가 Fort et al. 이론대로 진짜 독립적
다양성을 갖는지, 아니면 상관돼 있는지는 미확인이다. (2) 이 표는 **Sharpe 비율** 단위이고,
2026-08-19 N=3 결과 문서가 보고한 것은 **누적 PnL%**다 — 둘은 관련 있지만 같은 양이
아니므로, 이 표를 "그러니 +54%→-16% 변동이 정확히 예측된다"는 식으로 1:1 대응시키면
과장이다. 정확한 비교는 §우선순위 3에서 다음 액션으로 제안한다. (3) 이 저장소의 실제 승격
이력은 "N개 시드 중 최고를 고른" 사례가 아니다(260620 하나만 시도됐다) — 하지만
`docs/experiments/eth_omega461_oos_selection_bias_scope_and_resolution_20260813.md`가 이미
확인한 대로, **quality_threshold는 `rows.sort(key=lambda r: (r["oos_pnl"], ...))`로 OOS
1순위 정렬**돼 있고, SCALE_MAP 탐색은 eligible_scales 53,986개(다만 이건 VAL-only/OOS-blind
였다는 완화 요인이 있음)였다 — **이건 이 저장소에서 이미 실제로 일어난, 훨씬 명백한
MinBTL/PBO 오염 사례다.** 새로 발견된 시드축 문제는 이미 있던 threshold축 문제와 같은
구조(많은 트라이얼 중 같은 지표로 최고를 고름)가 중첩된 것으로 봐야 한다.

### B3. DSR/PSR/MinTRL — 사후 보정 및 트랙레코드 충분성

| 문헌 | 확인 |
|---|---|
| Bailey & López de Prado, "The Deflated Sharpe Ratio: Correcting for Selection Bias, Backtest Overfitting and Non-Normality," *Journal of Portfolio Management* 40(5):94-107, SSRN 2460551 (2014) | WebFetch가 PDF에서 핵심 개념·공식 서술을 요약(성공) — `pypdf` 전체 텍스트 추출로 원문 수식을 줄 단위로 재확인하지는 않았다, 초록·개념 수준으로 표기 |
| Bailey & López de Prado, "The Sharpe Ratio Efficient Frontier," *Journal of Risk* 15(2), SSRN 1821643 (2012/2013) | **본문 전체 직접 읽음**(pypdf 텍스트 추출, 46쪽, 부록의 Python 구현까지 포함) |

DSR은 관측된 Sharpe ratio를, "N개 트라이얼을 시도했다면 순전히 우연으로 기대되는 최대
Sharpe"(§B2의 E[max_N]과 같은 계열, "expected max SR grows approximately with
√(ln N)")만큼 깎아내(deflate) 재평가하는 지표다 — 트라이얼 수가 많을수록, 트라이얼 간
분산이 클수록 페널티가 커진다.

PSR/MinTRL은 다른 각도다: "관측된 Sharpe가 특정 문턱 이상이라고 몇 %신뢰수준으로 말할 수
있으려면 트랙레코드가 최소 몇 기간이어야 하는가"를 스큐·첨도까지 반영해 계산한다. 원문에서
직접 확인한 공식(Eq. 12-13)과 부록의 Python 구현(`class PSR`, `set_TRL`)을 함께 확인했다.
실측 예시(HFR 헤지펀드 월간지수, 11.167년 표본): 관측 Sharpe와 왜도(-0.72)·첨도(5.78)를
넣으면 "annualized SR을 0.5 이상이라고 95% 신뢰로 말하려면 4.99년 필요" — Normal-월간
가정보다 54%, Normal-일간 가정보다 82.8% 더 긴 트랙레코드가 필요했다.

**이 저장소와의 연결**: 이 저장소의 VAL(3~4개월)/OOS(3개월) 창은 MinTRL 관점에서 매우
짧을 가능성이 높다 — 다만 정밀 계산에는 이 저장소 자체 트레이드 수익률의 왜도·첨도가
필요한데 이번 세션에서 실측하지 않았다(추정치를 임의로 넣어 숫자를 만들지 않는다, §우선순위
3의 다음 액션으로 남긴다). CLT가 요구하는 "30 관측치 이상"(Hogg & Tanis 기준, 논문이 명시)
자체는 4개월치 5분봉(약 34,560 bar)이면 개수로는 압도적으로 만족하지만, 이건 "표본이 30개
넘으면 정규근사가 성립한다"는 조건이지 "Sharpe를 신뢰할 수 있다"는 조건이 아니다 — 논문은
이 둘을 명시적으로 구분한다("MinTRL may demand less than 2.5 years... but the moments
inputted...must be computed on longer series for CLT to hold").

### B4. 증폭 메커니즘 자체("작은 정확도 차이 → PnL 부호반전")를 직접 다룬 문헌 — 못 찾음

WebSearch로 "classifier accuracy variance amplified into trading Sharpe/PnL sign flip",
"threshold sensitivity position sizing 증폭" 등을 재확인했다. 나온 결과는 일반적인 트레이딩
블로그·비핵심 저널(MDPI 계열) 수준이었고, "분류 정확도의 작은 시드분산이 threshold-gated
포지션사이징을 거쳐 PnL 부호 자체의 반전으로 증폭되는 메커니즘"을 정면으로 이론화한 논문은
찾지 못했다. **이 공백은 솔직히 그대로 남긴다** — 사용자가 예상한 대로, PBO/DSR/MinBTL이
"왜 증폭되는가"를 설명하진 않지만 "증폭된 결과가 통계적으로 유의미한가"를 판정하는 도구를
대신 제공한다.

이 문서 자체의 추정(**문헌 근거 아님, 이 세션의 추론임을 명시**): (1) threshold 게이팅은
연속 확률 출력을 이산 진입/비진입으로 바꾸는 계단함수라, 결정경계 근처에서는 확률의 작은
변화가 거래 여부 자체를 뒤집을 수 있다. (2) bar-by-bar 순차청산 구조에서는 한 바에서
뒤집힌 결정이 포지션 보유상태를 바꾸고, 이는 다음 바들의 진입가능여부·청산조건에 연쇄적으로
영향을 준다(경로의존적 복리효과) — 단순 "정확도 차이"가 아니라 "순차 의사결정의 복리효과"가
증폭원일 가능성이 있다. 둘 다 이 저장소의 실측 구조(quality threshold 게이트, `evaluate_exit`
류의 bar-by-bar 순차 청산)와 정합적이지만, 인용 가능한 외부 문헌은 아니다.

---

## 문헌은 있지만 이 저장소 상황엔 안 맞음 (명확히 구분)

- **Henderson et al. (RL 재현성)** — 인용 근거는 있으나, RL 특유 축(정책이 자기 학습데이터
  분포를 만드는 비정상성, 탐험 무작위성)은 이 저장소의 지도학습 3-head 분류기 세팅에 직접
  적용되지 않는다. §A3에서 다룬 대로 지도학습 계열 문헌(Reimers & Gurevych, Bouthillier,
  Picard)이 더 정확한 유사사례다.
- **Wortsman Model soups 원형 그대로 적용** — 파인튜닝(같은 사전학습 시작점) 세팅에서
  검증된 것이지, 이 저장소처럼 처음부터 다른 초기화로 학습한 N개 시드에 naive 가중치 평균을
  그대로 적용하면 Ainsworth의 순열정렬 경고를 건너뛰는 셈이라 위험하다.
- **918-exp(arXiv:2603.16886)의 "seed 무관" 결론을 그대로 가져와 "그러니 우리 문제도 아닐
  것"이라고 반박하는 것** — 부당하다. 평가지표(아키텍처 랭킹 vs 개별 창 PnL)와 표본 길이가
  다르다는 이 문서의 가설(§A5)이 맞다면 두 결과는 애초에 모순이 아니라 다른 걸 재는 것일
  수 있다 — 확인 없이 "그 논문이 맞으니 우리가 이례적"이라거나 "우리 문제니 그 논문이
  틀렸다"고 단정하면 안 된다.
- **Colas/Card의 N≥20 권고를 곧이곧대로 "N=5는 무효"로 치환하는 것** — 과하다. 그들의
  N 숫자는 자기 실험의 효과크기·분산에서 유도된 것이지 보편 상수가 아니다. 이 저장소
  고유의 효과크기로 다시 계산해야 정확한 숫자가 나온다(§우선순위 4).

---

## 지금 시도해볼 만한 것 — 우선순위

1. **[순수 통계 재해석, 새 코드 작성 불필요, 즉시 가능 — 최우선]** §B1에서 확인했듯 이
   저장소에 이미 `core/selection_stats.py`가 있고 시드축에는 한 번도 적용된 적이 없다. 이미
   있는 N=3 예비 데이터(`tmp/causal_regen_20260516/eth_live_promotion_seed_robustness_20260819_*/report.json`)
   의 6개 평가창 bar-by-bar equity curve를 `(periods × n_seeds)` 행렬로 재배열해
   `deflated_sharpe_ratio(...)`와 `pbo_cscv(...)`, 가능하면 `falsification_audit(...)`까지
   그대로 호출한다. 새 학습도 새 통계 코드 작성도 필요 없다 — 기존 산출물 재사용 + 함수 호출뿐.
   N=3은 `pbo_cscv`/`falsification_audit`의 최소 트라이얼 수 요건에 못 미칠 수 있으니(코드의
   `n_cfg < 2`/`n_periods < n_splits*3` 가드 확인 필요), N≥5 확장과 병행하면 더 안전하다.
2. **[코드실험, 중간비용]** N≥5로 정식 확장(공식 문서가 이미 권고한 것)한 뒤, "시드 하나를
   골라 승격"이 아니라 **N시드 예측(확률) 평균**을 별도 후보로 같이 평가한다 — Lakshminarayanan
   /Fort et al. 이론이 직접 처방하는 해법이다. **가중치 평균(모델수프)은 §A2의 Ainsworth
   경고 때문에 이 단계에서는 보류를 권고**하고, 먼저 추론 시점 예측 평균(안전한 버전)부터
   시도한다.
3. **[순수 통계, 저비용]** 이 저장소의 실제 trade-level 수익률 분포(스큐·첨도)를 실측해
   MinTRL/PSR(§B3)을 계산하고, VAL/OOS 창이 몇 %신뢰수준에서 얼마나 짧은지 정량화한다.
   같은 실측치로 §B2 표의 Sharpe 단위 계산을 이 저장소의 실제 PnL%·Sharpe로 정확히 환산해
   "N=3 결과가 이 프레임워크의 예측 범위 안인지"를 숫자로 확인한다.
4. **[검정력 계산, 저비용]** Colas/Card식 검정력분석을, 저 문헌들의 예제 숫자가 아니라 이
   저장소의 **실측 효과크기**(오늘 N=3 데이터의 창별 시드 간 표준편차)로 직접 수행해 "이
   저장소 상황에서 실제로 필요한 N"을 유도한다 — 관행적 N≥5를 검정력 근거로 업그레이드하거나
   하향 정당화한다.
5. **[코드실험, 낮은 우선순위/장기]** §A1에서 제기한 가설(outer 시드는 Fort et al. 이론대로
   진짜 다양성을 갖는가, 아니면 zigzag_action 라벨의 저스킬 노이즈가 다양성처럼 보이는가)을
   직접 검증한다 — N=5 시드 각각의 **예측 자체**(라벨 아님)에 대해 BatchEnsemble 진단과
   동일한 방법(pairwise correlation)을 outer 시드 간에 적용한다. 붕괴(0.99대)해 있다면
   "다양성 자체가 없음"이 문제, 상대적으로 낮다면(기대대로) "다양성은 있는데 진짜 스킬이
   낮아 노이즈로 보임"이 확증된다 — 두 경우 처방이 다르다(전자는 §A1/A2 못 씀, 후자는
   §A1/A2가 유효한 해법).

## 준수 사항

이 문서는 순수 문헌 조사 + 저장소 기존 아티팩트(오늘 N=3 결과, 2026-08-13 OOS 선택편향
감사) 재해석이다. 재학습·재백테스트를 포함한 어떤 코드도 이 세션에서 실행하지 않았다.
§B2의 표는 이 세션이 원문에서 확인한 공식 위에 이 저장소의 창 길이를 대입해 직접 계산한
것으로, 새로운 promotion·모델선택·test 근거로 쓰기 위한 것이 아니라(Fresh-Forward
Validation/OOS/Test Rule 대상 아님) 시드분산의 "예상 가능한 크기"를 가늠하기 위한 참고
계산이다.
