# 코인 트레이딩에서의 딥러닝 활용 — 논문·인터넷 전수 조사 (2026-08-17)

## 조사 개요

6개 하위 축(시계열 예측 / 강화학습 / NLP·LLM 센티먼트 / GNN·온체인그래프 / 오더북 마이크로구조 /
변동성·리스크+메타서베이+산업동향)으로 나눠 병렬 리서치 에이전트 6개를 띄워 arXiv, Semantic
Scholar, OpenAlex, Crossref, 일반 웹서치로 조사했다. **모든 인용은 arXiv ID 또는 DOI를 API로
직접 검증한 것만 포함**했고(환각 인용 배제), 각 논문마다 실제 walk-forward/OOS 백테스트가 있는지
아니면 in-sample/single-split 지표 비교에 그치는지를 검증(verdict)으로 명시했다 — 이 저장소의
[Fresh-Forward 규칙](../.claude/CLAUDE.md)이 요구하는 기준으로 걸러 읽기 위함이다.

**핵심 결론을 먼저 말하면**: 가장 방법론이 엄격한 소수의 연구(918-실험 벤치마크, DRW/G-Research
Kaggle 실전 대회)는 한결같이 "복잡한 아키텍처가 단순 모델/선형모델을 못 이긴다" 또는 "방향성
정확도는 동전던지기(~50%) 수준"이라는 결론에 도달했고, 이는 이 저장소가 이미 독립적으로 겪은
[[eth_odyssey_dl_rl_architecture_axis_closed_20260816|Odyssey DL/RL 아키텍처 축 종료]],
[[eth_odyssey4_rl_layer_axis_closed_20260815|RL 레이어 축 종료]],
[[eth_candidate_cash_sleeve_ev_hgb_closed_20260816|EV-HGB 후보 종료]] 등의 실패 패턴과
정확히 일치한다. 문헌 전체에서 압도적 다수(수십 편)는 in-sample 또는 single-split holdout
오차지표(RMSE/MAPE)만 보고하고, 실거래비용을 반영한 워크포워드 백테스트나 시드 다양성 검증은
극소수(5편 미만)만 갖췄다.

---

## 1. 시계열 가격·수익률 예측 DL (LSTM/GRU/CNN-LSTM/Transformer/TCN/N-BEATS/Mamba)

### 1.1 RNN 계열 (LSTM/GRU/BiLSTM)

- **McNally, Roche & Caton (2018)**, *Predicting the Price of Bitcoin Using Machine Learning*,
  IEEE PDP 2018, DOI:10.1109/pdp2018.2018.00060 (피인용 669). 베이지안 최적화 RNN vs LSTM vs
  ARIMA, 일봉 BTC-USD. LSTM이 최고였지만 **방향성 정확도 52%** — 동전던지기와 큰 차이 없음.
  **검증: 초기 대표 논문이지만 결과 자체가 이미 약한 신호를 보여줌.**
- **Mallqui & Fernandes (2018)**, *Predicting direction, max, min and closing prices of daily
  BTC*, Applied Soft Computing, DOI:10.1016/j.asoc.2018.11.038 (254회). **검증: holdout only,
  실전 백테스트 기준 이전 세대 연구.**
- **Chen, Li & Sun (2019)**, *Bitcoin price prediction using ML: sample dimension engineering*,
  J. Comp. Applied Math, DOI:10.1016/j.cam.2019.112395 (422회). 5분봉 BTC에서 **랜덤포레스트가
  LSTM/로지스틱을 능가**(샘플윈도우 크기를 제대로 튜닝했을 때). **검증: DL이 단순 모델에 진
  초기 사례 — "DL이 기본값"이라는 가정에 대한 경고.**
- **Seabe, Moutsinga & Pindza (2023)**, *Forecasting Crypto Prices Using LSTM, GRU, and
  Bi-LSTM*, Fractal and Fractional, DOI:10.3390/fractalfract7020203 (202회). BTC/ETH/LTC 일봉.
  **검증: holdout 오차비교, 트레이딩 백테스트 없음.**
- Huang et al. (2021, arXiv:2103.14804, LSTM+Weibo 감성), Fleischer et al. (2022,
  arXiv:2202.13874), Islam et al. (2025, arXiv:2508.01419, XRP+유동성 지표) — 전부 **in-sample
  회귀 예시 수준**, 워크포워드/백테스트 없음.

### 1.2 CNN-LSTM 하이브리드

- **Guo, Lei, Ye & Fang (2021)**, *MRC-LSTM: Multi-scale Residual CNN + LSTM for BTC*,
  arXiv:2105.00707. BTC/ETH/LTC. **검증: RMSE 개선만 보고, 실현손익 백테스트 없음.**
- **Gautam (2025)**, *crypto price prediction using lstm+xgboost*, arXiv:2506.22055. LSTM 특징
  →XGBoost. **검증: MAPE 비교, OOS 트레이딩 테스트 없음.**

### 1.3 Transformer 계열

- **Khaniki & Manthouri (2024)**, arXiv:2403.03606, Performer(선형어텐션)+BiLSTM+기술지표.
  **검증: in-sample 오차감소만.**
- **Herremans & Low (2022)**, *Forecasting BTC volatility spikes from whale transactions and
  CryptoQuant data using Synthesizer Transformer*, arXiv:2211.08281. 온체인+고래알림 트윗→
  익일 변동성 급등 분류. **검증: 극단 변동성 스파이크(방향이 아닌 변동성) 과제에서 진짜
  OOS 개선 보고 — 좁은 과제이지만 신뢰도 있음.**
- **Peik et al. (2025)**, *Adaptive Temporal Fusion Transformers for Cryptocurrency Price
  Prediction*, arXiv:2509.10542. ETH-USDT 10분봉, 2개월 held-out. **드물게 정확도뿐 아니라
  시뮬레이션 트레이딩 수익성까지 보고**하며 고정길이 TFT/LSTM을 능가. **검증: 단일
  자산·단일 구간, 거래비용 명시 없음 — 참고할 만하나 확정적이지 않음.**
- **Kehinde et al. (2025)**, *Helformer*, J. Big Data, DOI:10.1186/s40537-025-01135-4 (46회).
  **검증: holdout 오차지표만.**

### 1.4 최신 효율적 TS 아키텍처 (N-BEATS/N-HiTS/PatchTST/iTransformer/TSMixer/ModernTCN) — 가장 중요

- **Hassan & Ibrahim (2024)**, *Neural Foundations of Crypto Predictions*, SSRN 5051299. 시총
  상위 8개 코인에서 N-HiTS/N-BEATS가 가장 오차 낮고 iTransformer가 일부 코인에서 근소 우위 —
  **차이 자체가 작다고 명시**. **검증: 오차지표 비교뿐, 백테스트 없음.**
- **Saidd (2026)**, *A Controlled Comparison of Deep Learning Architectures for Multi-Horizon
  Financial Forecasting: Evidence from 918 Experiments*, arXiv:2603.16886. 9개 아키텍처
  (Autoformer/DLinear/iTransformer/LSTM/ModernTCN/N-HiTS/PatchTST/TimesNet/TimeXer)×
  크립토/외환/주식×4h/24h, **고정시드 HPO+멀티시드 재학습+통계검정**을 갖춘 이 조사 전체에서
  가장 방법론이 엄격한 논문. **결과: 방향성 정확도가 아키텍처와 무관하게 시간봉 기준 ~50%
  (동전던지기)에 수렴, 아키텍처 순위는 point-error 지표에서만 유의미(ModernTCN 최우수, 평균
  순위 1.33).** **검증: 최고 신뢰도 근거 — 이 저장소가 이미
  [[eth_odyssey_dl_rl_architecture_axis_closed_20260816]]에서 도달한 결론과 정확히 일치.**
  (이 논문은 이 프로젝트 memory에도 이미 등록되어 있으며, 이번 조사에서 두 개의 독립된
  리서치 에이전트가 각자 별도로 재발견해 교차검증됨.)

### 1.5 상태공간모델 (Mamba/SSM)

- **Sepehri, Mehradfar, Soltanolkotabi & Avestimehr (2025)**, *CryptoMamba*,
  arXiv:2501.01010. Mamba SSM vs ARIMA/GARCH/LSTM, 트레이딩 알고리즘과 결합해 "실전형" 결과
  주장. **검증: 백테스트 방법론(비용, 워크포워드 프로토콜) 초록에 불명확 — 원문 확인 전엔
  신뢰 보류.**
- **Sharma, Majumdar, Chouzenoux & Elvira (2023)**, arXiv:2311.14731, 확률적 SSM 일단위 예측.
  **검증: 점예측/불확실성 정량화 초점, 백테스트 아님.**

### 1.6 서베이

- *A survey of deep learning applications in cryptocurrency*, iScience 2023,
  DOI:10.1016/j.isci.2023.108509. DL 분류체계 정리, 인용맵으로 유용.
- Livieris et al. (2021), *Cryptocurrency price prediction using traditional statistical and ML
  techniques*, DOI:10.1002/isaf.1488 (176회). 통계기법 대비 ML 광범위 서베이.

### 1.7 실무 자료

- **Kaggle "G-Research Crypto Forecasting"** (2021-2022, $125K, 14개 코인, **3개월 실전
  라이브 예측 리더보드**). **상위 3팀 전부 LightGBM(그래디언트부스팅) 사용, DL 아님** —
  주최측도 "아키텍처보다 피처엔지니어링이 훨씬 중요했다"고 명시. **이 조사 전체에서 가장
  신뢰도 높은 실전 반증 사례 중 하나.**

### 1.8 이 축 종합

문헌의 압도적 다수는 in-sample/single-split 오차지표만 보고하며, 워크포워드+거래비용을 갖춘
논문은 손에 꼽는다(Peik et al. 정도). 방법론이 가장 엄격한 918-실험 논문은 아키텍처 차이가
point-error에는 실재하지만 **방향성 스킬로는 이어지지 않는다**(~50%)는 결론을 내렸고, 유일한
대규모 실전 검증(G-Research Kaggle)은 DL이 아닌 GBM이 이겼다. 이는 이 저장소의 TabM 대안
탐색(VSN/diffusion/Mamba/Transformer/TCN, N=5시드 전부 종료)과 정확히 같은 그림이다.

---

## 2. 강화학습(DRL) 기반 코인 트레이딩

### 2.1 가치기반 (DQN)

- **Lucarelli & Borrotti (2020)**, Neural Computing and Applications,
  DOI:10.1007/s00521-020-05359-8. 2단계 DQN(자산별 로컬+글로벌 에이전트), 4코인 포트폴리오.
  **검증: 개념증명 수준, 엄밀한 베이스라인 비교 없음.**
- **Sattarov & Choi (2024)**, Scientific Reports, DOI:10.1038/s41598-024-51408-w. M-DQN+트위터
  감성, 수익/리스크/거래빈도 균형 보상. 수익률 29.93%, Sharpe>2.7 주장. **검증: 단일 실행·단일
  자산, 시드분산/거래비용 공개 없음 — 낙관적일 가능성 높음.**

### 2.2 정책경사 (PPO/A2C)

- **Gort, Liu, Sun, Gao, Chen, Wang (2022)**, *Deep RL for Cryptocurrency Trading: Practical
  Approach to Address Backtest Overfitting*, arXiv:2209.05559. 다수의 PPO/A2C 에이전트를
  학습시킨 뒤 **백테스트 과적합 확률(PBO) 가설검정으로 걸러내는** 방법론. 저PBO 에이전트가
  동일가중/S&P DBM 벤치마크를 능가(2022년 5-6월, 두 번의 급락 포함 구간). **검증: 이 조사
  전체에서 과적합 통제를 명시적으로 갖춘 몇 안 되는 사례 — 방법론적으로 신뢰도 높음.**
- **Wang & Klabjan (2023)**, arXiv:2309.00626. 멀티검증구간 모델선택+정책앙상블. **검증:
  앙상블/모델선택 기여, 새 알고리즘은 아님.**

### 2.3 연속제어 (DDPG/SAC) + 원조 EIIE

- **Jiang, Xu & Liang (2017)**, arXiv:1706.10059. EIIE(독립평가자 앙상블)+포트폴리오벡터
  메모리, 30분봉, 수수료 0.25% 반영, 50일간 4배 이상 수익 주장. **검증: 원조 논문이지만
  백테스트뿐이며, 이후 문헌(Gort et al. 2022)이 이 세대 결과 상당수가 과적합이었을 가능성을
  시사.**
- **Paykan (2025)**, arXiv:2511.20678. SAC vs DDPG 직접비교, SAC가 노이즈 환경에서 더 안정적.
  **검증: 단일 연구, 시드다양성 검증 필요.**
- **Habibnia & Soltanzadeh (2024)**, arXiv:2408.05382. SAC+CNN-멀티헤드어텐션, **바이낸스
  무기한선물 12자산, 레버리지·대출까지 모델링**(실제 선물 메커니즘을 다룬 드문 사례).
  **검증: 이 조사에서 실전 선물 메커니즘을 진지하게 다룬 몇 안 되는 논문 — 이 저장소의
  [Futures Risk Sizing Contract](../.claude/CLAUDE.md)와 비교해볼 가치 있음.**
- **Huang & Tanaka (2023)**, *CryptoRLPM*, arXiv:2307.01599. 온체인 지표를 상태 피처로 사용.
  **검증: 차별점은 RL 알고리즘이 아니라 피처엔지니어링.**

### 2.4 리스크인지/샤프비 보상설계 (주식시장 연구, 보상설계 아이디어는 이식 가능)

- **Wang et al. (2021)**, *DeepTrader*, AAAI, DOI:10.1609/aaai.v35i1.16144. **음의 최대낙폭을
  보상으로 직접 사용**+자산관계 임베딩+매크로조건부 롱숏. **검증: 보상설계 아이디어는 이식
  가능, 크립토 미검증.**
- **Choudhary et al. (2025)**, DOI:10.1007/s44196-025-00875-8. 로그수익/차분샤프/MDD 3개
  보상으로 각각 학습한 에이전트를 CNN으로 융합. **검증: 단일 보상 RL의 취약성을 스스로 인정한
  설계 — 아이디어는 이식 가능, 크립토 미검증.**

### 2.5 멀티에이전트 RL

- **Kumlungmak & Vateekul (2023)**, IEEE Access, DOI:10.1109/access.2023.3289844. MAPPO+협력
  보상+연속손실 누진페널티. DQN/A2C/PPO/FinRL-Ensemble 대비 강세장 +46%, 약세장에서 유일하게
  수익(+2.36%) 주장. **검증: 이 조사에서 찾은 유일한 크립토 네이티브 멀티에이전트 RL 논문,
  다만 단일 train/test 분할·시드분산 미보고 — 헤드라인 수치는 신중하게.**

### 2.6 인과추론 결합 RL

- **Amirzadeh, Thiruvady, Nazari, Ee (2023)**, *CausalReinforceNet*, arXiv:2310.09462. 베이지안
  인과망을 상태표현에 결합, 5개 알트코인. **검증: 상태표현 아이디어, 새 RL 알고리즘 아님.**

### 2.7 프레임워크/벤치마크

- **FinRL** (Liu et al. 2021, arXiv:2111.09395), **FinRL-Meta** (2022, arXiv:2211.03107,
  NeurIPS Datasets&Benchmarks), **FinRL Contests** (2025, arXiv:2504.02281) — 크립토 환경 포함
  공개 벤치마크/라이브러리.
- **Holzer, Wang, Xiao, Liu (2025)**, arXiv:2501.10709. FinRL 그룹 자체가 **"정책 불안정성과
  샘플링 병목"을 여전히 풀리지 않은 문제로 명시** — 이 조사에서 찾은, DRL-금융 진영 내부의
  가장 솔직한 자기비판.

### 2.8 실무 자료 및 종합

GitHub `AI4Finance-Foundation/FinRL`(~12k★), `sadighian/crypto-rl`(DDQN+LOB 리플레이),
`TensorTrade`(2023년 이후 정체) 등이 표준 참고 구현이다. 알고리즘 선택(DQN vs PPO vs SAC)보다
**보상함수 설계, 과적합 통제(PBO), 실제 비용/레버리지 모델링**이 결과를 좌우한다는 것이 가장
방법론적인 논문들(Gort et al. 2022, Habibnia & Soltanzadeh 2024, Holzer et al. 2025)의
공통 결론이다. DRL이 단순 지도학습/규칙기반보다 크립토에서 우월하다는 합의는 없다 —
FinRL 그룹 스스로 "정책 불안정성"을 미해결 문제로 인정한다. 이는 이 저장소의
[[eth_odyssey4_rl_layer_axis_closed_20260815]](방향/베토/청산헤드/포트폴리오게이트/사이징
5개 삽입점 전부 HGB 사이드카에 패배)와 정확히 같은 방향의 결론이다.

---

## 3. NLP·LLM 기반 센티먼트·트레이딩 시그널

### 3.1 고전 센티먼트 분류기 (BERT/FinBERT)

- **PreBit** — Zou & Herremans (2022), arXiv:2206.00648. 트위터 FinBERT 임베딩→VAE 차원축소→
  기술지표+상관자산가격과 결합, BTC **극단 가격변동** 예측. **검증: 신뢰도 있는 초기
  대표논문이나, 일반 방향이 아닌 극단변동 과제로 범위가 좁음.**
- Hossain et al. (2024, arXiv:2411.12748), Dashtaki et al. (2024, arXiv:2409.18895) — FinBERT
  감성+기술지표 융합. **검증: 베이스라인 대비 개선폭 불명확, 약한 신호.**
- **DLT-Corpus** — Hernandez Cruz et al., KDD'26, arXiv:2602.22045. 29.8억 토큰 코퍼스+
  LedgerBERT+2.3만건 크립토뉴스 감성데이터셋. **핵심발견: 기술은 학술문헌→특허→소셜미디어
  순으로 먼저 나타나며, 소셜미디어 감성은 가격과 무관하게 항상 낙관적으로 유지된다** — 즉
  소셜감성 자체의 정보량이 낮다는 것을 저자 스스로 보임. **검증: 데이터셋으로는 유용하나,
  자체 발견이 오히려 소셜감성 신호를 약한 신호로 지목.**
- **RAML** — arXiv:2607.23370. 이 저장소가 이미
  [[eth_raml_sentiment_regime_gate_literature_review_closed_20260816|검토·종료]]함:
  헤드라인 AUC가 무작위와 통계적으로 구별불가(z<0.6), 시드다양성 0.

### 3.2 LLM 기반 추론/트레이딩 에이전트

- **Luo, Feng, Xu, Tasca, Liu (2025/2026)**, arXiv:2501.00826 v3. 3개 전문 에이전트(코인/
  뉴스/트레이딩)로 상위15 L1 토큰, 2025년 52주 백테스트. 최고 구성 누적수익 133.52%,
  Sharpe 1.502 주장(GPT-4o/GPT-5/Claude Sonnet 4.5 비교). **검증: 이 배치 전체에서 가장 화려한
  수치이나 단일연도·시드/모델 분산 없음·OOS 재확인 없음 — 이 저장소의 N≥5시드 기준으로 보면
  전형적 과적합 위험군.**
- **Singhi (2025)**, arXiv:2510.08068. 전문분화 LLM 에이전트, 단일 구간·단일 실행.
  **검증: 미확인 헤드라인 수치.**
- **CryptoTrade** (2024, arXiv:2407.09546) — 온체인+오프체인 데이터 결합 제로샷 반성형 에이전트,
  이후 논문들의 베이스라인으로 자주 인용.
- **QuantHarness** (2025/2026, arXiv:2509.09995) — HFT용 4개 전문 에이전트, FinMem류의
  "느린 서사적 추론" 패러다임이 HFT에 부적합하다고 명시적으로 비판. **검증: 방법론적 비판으로
  유용.**
- **"Exploring Sentiment Manipulation by LLM-Enabled Intelligent Trading Agents"**,
  arXiv:2502.16343. RL기반 LLM 트레이딩 에이전트가 감성/시장인식을 **조작하는 법을 학습할 수
  있는지** 조사 — 성과주장이 아닌 위험/비판 논문.
- **Wu (2026)**, *Auditing Asset-Specific Preferences in Financial LLMs (Bitcoin)*,
  arXiv:2606.02528 v3. 프론티어 LLM 9종 모두 **비트코인에 대해 프레임 의존적이고 내부적으로
  조작 가능한 편향**을 가짐(SAE 피처 증폭으로 시뮬레이션 포트폴리오 배분이 ±5%p 이동).
  **검증: 신뢰성 경고 — LLM의 "추론"은 입력 텍스트와 무관한 내재된 편향을 함께 실어나른다.**

### 3.3 온체인 포럼/백서 텍스트

- **"Are Whitepaper Claims Reflected in Market Structure?"**, arXiv:2601.20336. 백서 서사가
  시장구조를 예측하는지 오염(contamination) 통제 파이프라인으로 검증 — **초기의 개체수준
  신호가 전부 코퍼스 오염 아티팩트였음을 밝히고, 보정 후엔 검정력 부족 null 결과**로 정정.
  **검증: 이 저장소가 반복적으로 겪은 "허위신호=데이터오염" 패턴이 다른 하위분야에서도
  독립 재현됨.**

### 3.4 방법론적 비판

- **Glasserman & Lin (2023)**, arXiv:2309.17322. GPT 감성점수가 **LLM이 학습 데이터로 이미
  기억하고 있는 기업명/결과에 의해 오염**됨을 보임("distraction effect"). 개체명 익명화를
  해법으로 제안. **검증: 백테스트 기간이 GPT 계열 모델의 학습기간과 겹치는 크립토 LLM
  센티먼트 파이프라인 전부에 직접 적용되는, 구체적으로 점검 가능한 오염 위험.**

### 3.5 실무 자료 및 종합

산업계 데이터 제공자(LunarCrush, Santiment, Augmento — 자체주장 방향정확도 69%, 검증 안 됨)는
학술 검증이 전혀 없다. 문헌은 두 갈래다: (1) BERT/FinBERT 감성 융합 — 대부분 약한 신호이며
DLT-Corpus 스스로 소셜감성이 구조적으로 노이즈임을 보임. (2) 2024-2026년 급증한 "LLM 트레이딩
에이전트" — 화려한 수익률(100%+, Sharpe>1.5)을 보고하지만 거의 전부 단일실행·단일구간·
시드평균 없음. 이 조사에서 독립적으로 재확인된 RAML(z<0.6), 백서-NLP 오염 아티팩트, GPT
룩어헤드 편향 3건은 이 저장소가 이미 아는 실패패턴과 정확히 같다. **이 축 전체를 이 저장소의
평가 기준(N≥5시드, purged CV, 개체명 익명화)으로 재면, 사실상 전부 미확정(unconfirmed)이지
확정-음성(confirmed-negative)은 아니다** — 저비용 게이트 실험 후보로는 열려있으나, 어떤 논문의
헤드라인 수치도 그대로 신뢰할 근거는 없다.

---

## 4. GNN · 크로스에셋/온체인 그래프 기반 DL

### 4.1 온체인 트랜잭션 그래프 (사기/AML 탐지 — 성숙한 하위분야)

- Kanezashi et al. (2022, arXiv:2203.12363, 이더리움 피싱), Lo et al. (2022, arXiv:2203.10465,
  Inspection-L 자금세탁), Azad et al. (2023, arXiv:2306.07974, Chainlet Orbits — **GNN이
  풀체인 실시간 적용엔 계산비효율적이라고 명시적으로 지적**), Bellei et al. (2024,
  arXiv:2404.19109, Elliptic2 벤치마크), Jin et al. (2023, arXiv:2310.00856, 폰지스킴 탐지),
  Zheng et al. (2025, arXiv:2506.21382, ATGAT) 등. **검증: 전부 사기/AML 분류 과제이며 가격
  예측/트레이딩 시그널이 아니다 — Elliptic/Elliptic2가 공개 벤치마크로 확립된 성숙 분야.**

### 4.2 크로스에셋 상관관계 그래프 (가격예측)

- **CryptoGAT** — Peng, Khushi, Poon (2026), arXiv:2606.27670. 코인을 노드로, 학습된 자산간
  관계를 엣지로 삼는 GAT. 이 저장소가 이미
  [[eth_literature_review_cryptogat_and_918experiments_dl_architecture_20260816|딥리뷰]]함:
  N개 코인 중 상대순위를 맞추는 크로스섹션 랭킹 문제이지 단일자산 절대방향 문제가 아니며,
  베타중립 벤치마크 부재 + 강세장 편향 구간 + 시드 3회뿐이라는 구멍이 있음. **이 저장소가
  실제로 그 크로스에셋 상관관계 아이디어를 저비용 게이트로 재현했을 때
  [[eth_candidate_crossasset_correlation_cheap_gate_20260816|BTC-ETH 상관 IC가 VAL/OOS 간
  부호 불안정]]으로 종료됨 — CryptoGAT 논문 자체는 이 안정성 검증을 전혀 하지 않는다.**
- Zhou et al. (2025), *Financial Innovation*, DOI:10.1186/s40854-025-00768-x. 진화하는
  다중스케일 상관그래프+GNN **변동성** 예측(방향 아님).
- Uygun & Sefer (2025), DOI:10.1007/s00521-025-11586-8; Celik & Sefer (2025),
  DOI:10.1007/s10614-025-10940-1(온체인 트랜잭션 모티프→가격예측, 온체인그래프→트레이딩시그널
  범주와 가장 가까움); Yin et al. (2022), DOI:10.1080/13504851.2022.2141436(매크로스트레스
  지수 조건부). **검증: 전부 single-split 벤치마크뿐, 크로스에셋 IC 안정성 검증한 논문 전무.**

### 4.3 DeFi 프로토콜 그래프

- **DeFiGuard** — Wang et al. (2024), arXiv:2406.11157. DEX풀/토큰/거래 그래프로 가격조작
  공격 탐지(2018-2022년 손실 $50M+ 추정).
- **DeXposure / DeXposure-FM** — Wu et al. (2025, arXiv:2511.22314), Shu et al. (2026,
  arXiv:2602.03981). 4.37만개 항목·4,300개 프로토콜·602개 체인 규모의 토큰-프로토콜 TVL
  의존성 그래프, 그래프 파운데이션 모델로 시스템리스크/전염 예측. **검증: 이 조사에서 찾은
  가장 크고 새로운(2025-2026) DeFi 그래프 자원 — 알파 생성이 아닌 리스크오버레이 피처로서
  가치.**

### 4.4 종합

온체인 트랜잭션그래프 사기탐지는 Elliptic 벤치마크로 확립된 성숙 분야지만 가격예측과 무관하다.
크로스에셋 상관그래프 가격예측은 2025-2026년에 갑자기 늘어난 얇고 방법론이 약한 군집으로,
전부 single-split 벤치마크만 보고하며 IC 안정성 검증이 전무하다 — 이 저장소가 CryptoGAT의
핵심 아이디어를 직접 재현해서 얻은 "IC 부호 불안정" 결론과 대조할 논문이 문헌에 하나도 없다는
사실 자체가 의미있는 공백이다. DeFi 프로토콜그래프(DeXposure 계열)는 가장 활발히 성장 중인
축이지만 알파보다 시스템리스크/전염 예측 지향 — 리스크베토 피처 소스로는 검토할 가치가 있다.

---

## 5. 오더북·마켓 마이크로구조 DL

### 5.1 LOB 스냅샷 예측

- **DeepLOB** — Zhang, Zohren, Roberts (2018/2019), arXiv:1808.03668 → IEEE TSP,
  DOI:10.1109/tsp.2019.2907260 (278회). CNN+LSTM, 10레벨 LOB, 3클래스 중간가 방향. **주식
  전용, 이 분야 전체의 기준 아키텍처.**
- **TLOB** — Berti & Kasneci (2025), arXiv:2502.15757. 이중(공간+시간) 어텐션 트랜스포머,
  FI-2010/NASDAQ**+비트코인 데이터셋**에서 SOTA 능가. 시간에 따른 예측력 저하(F1 -6.68pt),
  추세를 평균스프레드 기준으로 재정의하면 이득이 사라짐을 명시. **검증: 크립토로 검증된
  드문 사례이며, 비용현실성 체크까지 갖춰 신뢰도 있음.**
- **T-KAN** — Makinde (2026), arXiv:2601.02310. LSTM 선형가중치를 학습가능 B-스플라인으로
  대체, FI-2010에서 DeepLOB이 1bp 비용에서 -82.76% 낙폭인데 T-KAN은 +132% 주장. **검증:
  단일논문·단일모델·주식벤치마크·크립토 미검증 — 신뢰 보류.**
- **Wang (2025)**, *Exploring Microstructural Dynamics in Cryptocurrency Limit Order Books:
  Better Inputs Matter More Than Stacking Another Hidden Layer*, arXiv:2506.05764. **크립토
  전용**. 깊이/파라미터 추가보다 입력 피처 품질이 더 중요하다고 직접 검증 — 이 저장소가 TabM
  대안 탐색에서 도달한 "피처가 아키텍처를 이긴다" 결론과 정확히 같은 결론을 LOB 도메인에서
  독립적으로 재현.
- **Jha, De Paepe, Holt, West, Ng (2020)**, arXiv:2010.01241 (코드/데이터 공개). **크립토
  전용**. 코인베이스 BTC 현물 LOB, TCN, 2초 호라이즌, **워크포워드 방향정확도 71%** 보고,
  코로케이션 배포 전제로 상용GPU 1일 이내 학습 가능.
- **Fang et al. (2020)**, arXiv:2003.00803 → European Journal of Finance 2021,
  DOI:10.1080/1351847x.2021.1908390 (89회). **크립토 전용**, 다중거래소 고빈도 중간가 틱데이터.
- **Shintate & Pichl (2019)**, *JRFM*, DOI:10.3390/jrfm12010017. BTC 고빈도 추세분류, LSTM보다
  낫지만 **매수후보유(buy-and-hold)를 이기지 못함** — 초기의 정직한 음성 결과.
- **Letteri (2025)**, arXiv:2507.14960. 비트코인 LOB 실시간 이상탐지(조작탐지), 13개 모델
  비교, "AITA-OBS" 테스트베드.

### 5.2 오더플로우 불균형(OFI)

- **Kolm, Turiel, Westray (2023)**, *Mathematical Finance*, DOI:10.1111/mafi.12413 (32회).
  OFI 파생 정상화 피처(원시 LOB 아님)→LSTM/ANN, 나스닥 115종목, OFI피처가 원시LOB입력보다
  우수. **주식 전용, 크립토 이식 사례 문헌에 없음.**
- **Cestari, Barchi, Busetto, Marazzina, Formentin (2023)**, arXiv:2312.16190. **크립토
  전용**. Hawkes 점과정 오더플로우 모델+연속출력오차모델, 신경망은 아니지만 크립토 OFI
  예측의 가장 가까운 유사사례.
- **Fabre & Challet (2025)**, arXiv:2504.15908. **크립토 전용**(중앙화 거래소 Level-3).
  다중스케일 Hawkes 파생 오더플로우 피처→신경망, 스푸핑 탐지.
- **Bieganowski & Ślepaczuk (2026)**, *Explainable Patterns in Cryptocurrency Microstructure*,
  arXiv:2602.00776. 바이낸스 선물 L2, CatBoost+SHAP — **DL이 아닌 GBM 강력 베이스라인**으로
  참고 가치.

### 5.3 마켓메이킹 DL

크립토 마켓메이킹은 검색된 문헌 대부분이 **지도학습이 아닌 RL**(arXiv:1911.08647,
arXiv:2004.06985, IEEE Access DOI:10.1109/access.2021.3074782 등, 2절에서 별도 취급)이며,
지도학습 DL의 유일한 구성요소인 체결확률 생존분석 모델(Arroyo, Cartea, Moreno-Pino, Zohren
2023, arXiv:2306.05479 등)은 **전부 주식/외환/채권 전용, 크립토 이식 사례 없음.**

### 5.4 실무 자료 및 종합

Kaiko Research, Amberdata가 유상 크립토 풀뎁스 LOB 데이터의 표준 공급자다. 공개 크립토 데이터가
대부분 OHLCV/체결틱뿐이라 풀뎁스 LOB 접근 자체가 재현의 구조적 장벽이며(나스닥 LOBSTER 같은
표준화된 저비용 대안 부재), 실제로 비용을 점검한 두 논문(TLOB, Shintate & Pichl)은 모두 **거래
비용 반영 시 엣지가 축소되거나 사라짐**을 보고했다. **이 축은 이 조사 전체에서 이 저장소가
아직 직접 탐색하지 않은 영역**이며, 유일하게 진짜 신뢰할 만한 크립토 검증 사례(TLOB)와
"피처가 깊이를 이긴다"는 독립 재확인(Wang 2025)이 존재해 저비용 게이트 실험 후보로 고려할
가치는 있다 — 다만 풀뎁스 LOB 데이터 확보 자체가 선결 문제다.

---

## 6. 변동성·레짐·리스크 DL + 메타서베이 + 산업동향

### 6.1 변동성 예측 / 레짐·이상탐지 / 리스크 모델링

- **Amirshahi & Lahmiri (2023)**, *Machine Learning with Applications*,
  DOI:10.1016/j.mlwa.2023.100465 (50회). LSTM/GRU+GARCH계열 하이브리드, 다중코인, GARCH·RNN
  단독보다 우수 주장. **검증: single-split, 시드평균 없음 — 시사적이나 확정적이지 않음.**
- **García-Medina & Aguayo-Moreno (2023)**, *Computational Economics*,
  DOI:10.1007/s10614-023-10373-8 (77회). LSTM-GARCH 하이브리드, VaR 백테스트 개선 주장.
  **검증: 동일한 한계(purged CV·멀티시드 없음).**
- *Forecasting Volatility with Machine Learning and Rough Volatility: Example from the
  Crypto-Winter*, arXiv:2311.04727. 2022 크립토윈터 스트레스구간에서 rough-volatility 피처가
  정보를 더한다고 주장. **검증: 단일논문, 미재현.**
- *Denoising Complex Covariance Matrices with Hybrid ResNet and Random Matrix Theory:
  Cryptocurrency Portfolio Applications*, arXiv:2510.19130. ResNet으로 공분산행렬 디노이징 —
  이 조사에서 찾은, 방향예측이 아닌 진짜 "리스크 모델링" DL에 가장 가까운 사례.
- **공백 확인**: 크립토 전용 딥 생존분석(포지션/낙폭 리스크) 모델은 문헌에 전무하다. 가장
  가까운 유사사례(LOB 체결확률 생존분석, arXiv:2306.05479)조차 크립토 미적용이다.

### 6.2 광범위 서베이/메타분석/방법론 비판

- **Saidd (2026)**, arXiv:2603.16886 — 위 1.4절과 동일 논문, 이 조사 전체의 앵커. **918개
  실험 전체 평균 방향정확도 50.08%, 통계적으로 동전던지기와 구별불가.**
- *A survey of deep learning applications in cryptocurrency*, iScience 2023,
  DOI:10.1016/j.isci.2023.108509 (47회). **스스로 "아키텍처 중심·평가 빈약(짧은 테스트구간,
  일관된 베이스라인 부재)"이라고 지적.**
- *Review of deep learning models for crypto price prediction: implementation and
  evaluation*, arXiv:2405.11431. 기존 발표된 DL 가격예측 주장들을 **공통 하네스로 재구현해
  재현성 자체를 검증**하는 드문 방법론 비판 논문.
- CryptoGAT(arXiv:2606.27670) — "크립토에 시계열모델이 통하는가?"를 정면으로 묻는 논문 자체가
  이미 회의적 프레이밍이며, 이 저장소의 재현 결과(4.2절)와 수렴.
- *Deep RL for Cryptocurrency Trading: Practical Approach to Address Backtest Overfitting*,
  arXiv:2209.05559 — "발표된 크립토 DL 백테스트 대부분이 과적합"이라는 주장을 최초로 엄밀하게
  실증한 선례(RL 특화, 2절 참조).

### 6.3 실무/산업 자료

- **Kaggle "DRW – Crypto Market Prediction"** (2025, DRW 후원, **실전 프로덕션 피처 데이터**,
  9,011명 참가, $25K). **1위 수상자 본인 발언: "피처 품질이 좋으면 선형모델이 강력하다."**
  ([writeup](https://www.kaggle.com/competitions/drw-crypto-market-prediction/writeups/drw-solution-1st))
  **이 조사 전체에서 가장 최신(2025)이고 직접적인 산업 증거** — 실제 퀀트펌이 실전 피처로
  주최한 대회에서 복잡한 ML/DL이 아닌 선형모델이 우승.
- **Kaggle "G-Research Crypto Forecasting"** (1.7절과 동일) — LightGBM 우승, DL 아님.
- **Freqtrade/FreqAI** (github.com/freqtrade/freqtrade, 3.99만★) — 공개 크립토 트레이딩봇,
  FreqAI 모듈이 임의 회귀/분류/신경망 지속재학습 지원. **검증: 인프라일 뿐 DL 우위를 주장하는
  공개 백테스트/라이브 수치 없음.**
- **바이낸스 리서치**: 트레이딩 알파 관련 DL 리서치 리포트는 발견되지 않음. 공개된 ML
  내용은 전부 사기탐지 지향("100개 이상 ML모델, $105억 사기 예방"). **주목할 부정적 발견 —
  최대 거래소의 공개 리서치가 트레이딩 알파용 DL을 발표하지 않는다는 사실 자체가 시사적.**
- Wintermute, Jump, DRW 등 퀀트 마켓메이커: AI 인프라 투자는 공개적($10억 규모 보도)이나
  기술 방법론은 비공개. DRW의 유일한 공개 기술적 단서가 위 Kaggle 대회 결과다.

### 6.4 종합

가장 엄격한 학술 근거(918-실험 논문)와 가장 신선한 산업 근거(DRW 2025 Kaggle 대회 1위 수상자의
"선형모델이 이긴다"는 육성 발언)가 **같은 방향**을 가리킨다. GARCH-하이브리드 변동성 예측
문헌은 표면적으로 더 고무적이지만 하나같이 single-split·비-시드평균이라 이 저장소의
Fresh-Forward 기준을 통과하지 못한다. 크립토 전용 딥 생존/낙폭리스크 모델은 사실상 빈
공백지대다. 바이낸스가 사기탐지 ML은 대대적으로 홍보하면서 트레이딩알파 DL은 침묵한다는
사실도, "AI가 크립토 트레이딩을 바꾼다"는 대중서사가 실증보다는 인프라/에이전트 마케팅에
가깝다는 정황이다.

---

## 7. 전체 종합 — 6개 축을 관통하는 메타 결론

**패턴 1 — 평가 방법론의 압도적 비대칭.** 6개 축 전체에서 수집된 수십 편 중 워크포워드+
거래비용+멀티시드를 모두 갖춘 논문은 **5편 미만**(918-실험 논문, Gort et al. 2022 PBO 논문,
TLOB, DRW/G-Research Kaggle 실전대회 정도)이었다. 나머지는 in-sample 또는 single-split
오차지표만 보고한다 — 이 저장소의 Fresh-Forward 규칙이 정확히 이 함정을 막기 위해 존재한다.

**패턴 2 — 방법론이 엄격할수록 결론이 회의적으로 수렴.** 시계열예측(1절)의 918-실험 논문,
변동성/메타서베이(6절)의 동일 논문 및 DRW Kaggle 실전대회, GNN(4절)에서 이 저장소가 직접
재현한 크로스에셋 IC 불안정성, RL(2절)의 FinRL 그룹 자체 "정책 불안정성" 인정, NLP(3절)의
RAML/백서-오염/GPT룩어헤드 3중 확인 — **전부 독립적으로 같은 결론**(복잡한 DL이 단순
모델/베이스라인을 유의미하게 이기지 못하거나, 이긴다는 주장 자체가 방법론적 결함을 안고
있다)에 도달했다.

**패턴 3 — 이 저장소의 종료된 축들과의 교차검증.** 아래는 이번 조사가 이 프로젝트의 기존
memory 기록과 독립적으로 수렴한 지점이다:

| 이 저장소의 기존 종료 축 | 이번 조사에서 발견한 외부 일치 근거 |
|---|---|
| [[eth_odyssey_dl_rl_architecture_axis_closed_20260816]] (TabM 대안 전부 종료) | 918-실험 논문(arXiv:2603.16886): 아키텍처 무관 방향정확도 ~50% |
| [[eth_odyssey4_rl_layer_axis_closed_20260815]] (RL 5개 삽입점 전부 HGB에 패배) | FinRL 그룹 자체 "정책 불안정성" 인정(arXiv:2501.10709); RL 문헌 대부분 시드분산 미보고 |
| [[eth_candidate_crossasset_correlation_cheap_gate_20260816]] (BTC-ETH IC 부호 불안정) | CryptoGAT류 크로스에셋 그래프 논문 전부 IC 안정성 검증 자체를 안 함 — 공백이 곧 반증은 아니지만 대조군 부재 |
| [[eth_raml_sentiment_regime_gate_literature_review_closed_20260816]] (RAML z<0.6) | 백서-NLP 오염 아티팩트(arXiv:2601.20336), GPT 룩어헤드편향(arXiv:2309.17322) — 같은 실패 유형 2건 추가 확인 |
| [[eth_candidate_cash_sleeve_ev_hgb_closed_20260816]] (HGB 사이드카도 결국 실패) | DRW 2025 Kaggle 1위: "피처 품질이 좋으면 선형모델이 강력하다" — 모델 복잡도가 아니라 피처가 핵심이라는 동일 결론 |

**패턴 4 — 이 저장소가 아직 시도하지 않은 축.** 오더북·마켓마이크로구조 DL(5절)은 이 조사
전체에서 이 프로젝트가 명시적으로 탐색한 적 없는 유일한 축이다. TLOB(arXiv:2502.15757)이
비트코인 데이터로 검증된 드문 사례이고, Wang(2025, arXiv:2506.05764)이 "피처가 깊이를 이긴다"는
결론을 LOB 도메인에서 독립 재현했다는 점은 흥미롭지만, **풀뎁스 크립토 LOB 데이터 확보 자체가
구조적 장벽**(공개데이터는 대부분 OHLCV/체결틱뿐)이라 저비용 게이트 실험을 시작하기 전에
데이터 소스 확보가 선결과제다.

**최종 평가.** 이번 전수조사는 "딥러닝이 크립토 트레이딩에서 진짜 재현가능한 엣지를
제공한다"는 명제에 대해 방법론 엄격도와 반비례하는 신뢰도 패턴을 보였다 — 대중적/산업
마케팅 서사(LLM 트레이딩 에이전트의 100%+ 수익률, AI인프라 투자 보도)는 화려하지만 근거가
얕고, 학술적으로 가장 엄밀한 소수 연구와 가장 최신의 실전 산업 증거(DRW 2025)는 한결같이
회의적이다. 이는 이 저장소가 지난 세션들에서 겪은 결과와 우연이 아니라 **업계 전체의 구조적
패턴**임을 시사한다.

---

## 조사 방법 메모

- 6개 병렬 리서치 에이전트가 각자 독립적으로 arXiv Atom API, OpenAlex works API, (rate-limit로
  제한적) Semantic Scholar, 일반 웹서치를 사용해 조사(2026-08-17 접근).
- 모든 인용은 arXiv ID 또는 DOI가 API로 실존 확인된 것만 포함 — 환각 인용 배제.
- 두 개의 독립 에이전트가 arXiv:2603.16886(918-실험 논문)을 각자 별도 경로로 재발견해
  교차검증됨.
- 이 문서는 리서치 결과의 종합이며, 상세 딥리뷰가 필요한 개별 논문(특히 오더북/LOB 축의
  TLOB, DeXposure)은 별도 `docs/experiments/` 문헌리뷰 문서로 후속 작업할 가치가 있음.
