# 외부 라벨링 로직 조사 + 라벨 불필요 패러다임 조사 (2026-08-19)

## 0. 배경과 방법론

이 문서는 두 개의 독립 질문에 답한다.

- **Part A**: 이 저장소가 아직 안 써본, 문헌상 근거가 강한 라벨링 방법론이 있는가?
- **Part B**: 지도학습 라벨링 자체를 회피하거나 최소화하는 패러다임이 있는가?

직접적인 계기는 오늘 같은 세션에서 나온 `zigzag_action` 라벨의 추세추종(momentum-chasing)
편향 발견(`eth_zigzag_action_label_entry_timing_momentum_bias_20260819`)과, 그 직후 진행된
1차 문헌조사(`reference_direction_quality_exit_label_methodology_20260819`, 메모리 노트)다.
이 문서는 그 1차 조사를 반복하지 않고 그 위에 (a) 더 깊은 검증(arXiv/OpenAlex/WebFetch로
원문 직접 확인), (b) Part B의 신규 축(마켓메이킹/스탯아브/온라인학습/규칙기반 트리거)을
추가한다.

### 0.1 반드시 먼저 확인한 저장소 자체 결론

- **`docs/label_methodology_survey_20260815.md`**: 40개 이상의 라벨 방법론(triple-barrier
  11종, zigzag 11종, meta-labeling 7종, trend-scanning 8종, DP/oracle 4계열)이 이미 시드평균+
  VAL+OOS+always_short 대비로 재검증됐고 전부 "방향 예측 edge 없음"으로 수렴했다. 진단된 근본
  원인은 거의 항상 **"사용 가능한 feature set에 애초에 방향 정보가 없다"**(h48qual 하나에서만
  10회 이상 독립 재확인)이지 배리어 공식이 아니다.
- **`docs/entry_exit_edge_root_cause_and_literature_review_20260809.md`**: 이 문서와 거의 같은
  형식의 선행 조사. **Part 5(A)**에서 이미 "CUSUM 정보기반 이벤트 샘플링 + 대칭 배리어 + 진짜
  신경망(MLP)"을 직접 구현해 테스트했다 — 이건 아래 A.1에서 다룰 Financial Innovation(2025)
  논문의 핵심 레시피와 사실상 동일한 조합이다. 결과: TRAIN AUC 0.55, DEV/VAL/OOS 전부
  0.49~0.51 (동전던지기), falsification_audit 35th percentile(무작위 셔플보다 나쁨). **이
  문서를 쓰기 전 반드시 알아야 할 사실 — "정보기반 이벤트 샘플링+대칭 배리어+DL"이라는, Part A가
  찾을 가능성이 있는 가장 유력한 후보 중 하나가 이미 사실상 재현·기각됐다.**
- **`docs/deep_learning_for_crypto_trading_literature_survey_20260817.md`**: 6개 축(시계열DL/
  RL/NLP·LLM/GNN/LOB/변동성) 전수조사. 방법론이 엄격한 문헌일수록 "복잡한 모델이 단순 모델을
  못 이긴다"로 수렴한다는 게 핵심 결론. **오더북 마이크로구조 DL(5절)이 유일하게 이 저장소가
  탐색하지 않은 축**으로 지목됐고, 데이터 블로커는 2026-08-17에 이미 해소됐다
  (`eth_candidate_lob_microstructure_contract_20260817.md`).
- **`docs/model_contracts/research_line_registry.json`**: 34개의 공식 CLOSED 연구 라인. RL,
  cross-sectional momentum, 기술지표, funding-carry-contrarian(F3-A) 등이 등재돼 있다 — 이
  문서에서 재추천하는 모든 항목은 이 레지스트리와 대조해 "왜 이번엔 다른가"를 명시한다.
- **`reference_direction_quality_exit_label_methodology_20260819`(메모리 노트)**: triple-barrier
  (direction 대안)/meta-labeling+SelectiveNet(arXiv:1901.09192)+conformal
  prediction(arXiv:2107.07511)(quality 대안)/deep optimal stopping(arXiv:1804.05394)(exit
  대안)을 이미 찾아뒀다. 이 문서는 그 결론을 반복하지 않고 더 최신 후속 연구·실제 구현체·
  크립토 적용 사례로 확장한다.

### 0.2 검증 원칙

인용된 모든 arXiv ID는 `arxiv.org/abs/{id}`를 WebFetch로 직접 열어 제목/저자/날짜/핵심주장을
확인했다(**검증됨**으로 표기). MDPI 논문 1건은 접근이 403으로 차단돼 WebSearch 스니펫으로만
교차확인했다(**부분검증**으로 표기, 아래 A.1.2). 이미 저장소에 인용돼 있던 항목(오늘자 메모리
노트 포함)은 원 출처를 표기하되 이 세션에서 재검증하지 않았다(**기존 인용**으로 표기).

---

## Part A — 외부 최고 라벨링 로직

### A.1 Triple-Barrier 최신/크립토 특화 변형

`core/event_label_engine.py`의 docstring이 이미 MDPI 2024 GA-triple-barrier
(DOI:10.3390/math12050780), Financial Innovation 2025 CUSUM+TB+DL
(DOI:10.1186/s40854-025-00866-w), Directional-Change/intrinsic-time(SSRN 5017215,
arXiv:2501.06032)를 인용해뒀다(**기존 인용**). 이번 조사에서 확인한 것:

| 항목 | 근거 | 이 저장소와의 연결 | 판정 |
|---|---|---|---|
| Financial Innovation(2025) CUSUM+TB+DL, 크립토 | DOI:10.1186/s40854-025-00866-w (**기존 인용**) | **이미 사실상 재현됨.** `entry_exit_edge_root_cause_and_literature_review_20260809.md` Part 5(A)가 정보기반 이벤트 샘플링(CUSUM)+대칭 배리어+실제 MLP를 독립 구현해 2026-08-10 테스트 — AUC 0.49~0.51, falsification_audit 실패. 논문의 핵심 레시피와 동일한 조합이 이미 기각됨. | **재시도 근거 없음** — 같은 정보, 다른 논문 포장 |
| GA-triple-barrier(MDPI 2024) | DOI:10.3390/math12050780 (**기존 인용**) | `event_label_engine.py`가 이미 "GA로 탐색하는 핵심 아이디어(매직넘버 배제)만 `calibrate_barriers()`의 grid search로 단순화해 반영"이라고 명시. h48qual 자체가 dense grid로 이미 이 아이디어를 수행 중. | **이미 흡수됨** |
| Directional-Change/intrinsic-time (Razmi & Barak, SSRN 5017215 / arXiv:2501.06032) | **기존 인용**, 이번에 코드 대조로 신규 확인 | `event_label_engine.py`에 `directional_change_events()`/`_directional_change_numba()`로 **이미 구현**돼 있고 `LabelEngineConfig.event_method='directional_change'`로 선택 가능하다(코드 537-597행). 그러나 이 엔진을 실제로 import하는 두 스크립트(`research_eth_candidate_cash_sleeve_ev_hgb_train_stage1_20260816.py`, `diagnose_eth_h48qual_dirhead_metalabel_via_event_label_engine_20260815.py`) 중 어느 것도 `directional_change`를 쓰지 않았다 — 후자는 명시적으로 `event_method="all_bars"`를 선택했다(83-84행, "부모가 매 bar 스코어링했으므로 CUSUM 재샘플링 없이"). 즉 **모듈 자체 스모크테스트(650-678행, 더미 데이터) 외에는 실전 라벨 생성에 단 한 번도 쓰인 적이 없다.** | **유일하게 "구현은 있는데 실전 검증 자체가 빠진" 항목** — CUSUM과는 이벤트 샘플링 메커니즘이 다르므로(누적합 임계값 vs 고정폭 가격반전), 이미 실패한 CUSUM 결과를 그대로 상속한다고 볼 근거는 없다. 다만 기대치는 낮게 잡아야 한다(근본원인이 이벤트 샘플링이 아니라 feature information content라는 진단과 동일 선상) |
| AEDL(Adaptive Event-Driven Labeling): multi-scale + Granger causality/transfer entropy + MAML | MDPI *Applied Sciences* 2025, https://www.mdpi.com/2076-3417/15/24/13204 (**부분검증** — 원문 403 차단, WebSearch 스니펫 교차확인) | `event_label_engine.py`가 이미 "프레임워크 자체가 별도 연구 과제 수준"이라며 의도적으로 제외했다. WebSearch로 확인한 신규 디테일: 16개 자산 2000-2025(크립토 특정 아님, TradFi 위주로 추정), baseline 대비 average Sharpe 0.48(baseline은 near-zero/negative). **Granger causality/transfer entropy 필터링은 기존 상관변수 중에서 고르는 것뿐, 새 정보를 만들지 않는다. MAML은 학습레시피 변경인데, 이 저장소는 "학습레시피는 방향 skill을 못 고친다"를 `ilias1_zig075_standalone_always_direction_benchmark`에서 이미 반복 확인했다.** | **재시도 근거 약함** — 저장소의 기존 제외 결정 유지 권고 |

**A.1 종합**: Triple-barrier 축은 사실상 소진됐다. 문헌에서 발견한 "새로운" 것들은 (a) 이미
반영돼 있거나(GA), (b) 이미 재현·기각됐거나(CUSUM+TB+DL), (c) 근본원인과 무관한 학습레시피
변경(AEDL)이다. 유일한 예외는 Directional-Change 이벤트 샘플링 — 구현은 있지만 실전
end-to-end 검증이 아예 빠져있다는 게 이번 조사의 실질적 신규 발견이다.

### A.2 이산 라벨을 피하는 분포적/회귀 접근

이 저장소는 이미 **한 번** 이 축을 시도했다: 2026-08-09 밤샘 세션 아이디어 #14, LightGBM
2-quantile(α=0.15/0.85) skew 방향 신호(`scripts/research_quantile_regression_skew_entry_eth_20260809.py`).
단일 오버나이트 실행, `falsification_audit` 게이트로 네거티브 판정. **단, 이건 "quantile
regression"이라는 큰 범주 중 아주 좁은 구현**(2-quantile GBM, skew 방향 신호로만 사용)이다 —
진짜 완전한 분포적 예측(모든 분위수 또는 분포 모수 직접 추정, proper scoring rule 최적화)은
아니다.

| 항목 | 근거 | 이 저장소와의 연결 | 판정 |
|---|---|---|---|
| "Forecasting Probability Distributions of Financial Returns with Deep Neural Networks", Michańków (2025) | arXiv:2508.18921 (**검증됨**) | Custom NLL loss로 Normal/Student-t/skewed-Student-t 분포 모수를 직접 추정, CRPS/LPS(proper scoring rule)로 평가. 이산 배리어/threshold 선택이 구조적으로 없다 — 배리어 선택이 반복 실패 원인이었던 이 저장소 관점에서 메커니즘 자체는 흥미롭다. **그러나 (a) S&P500/BOVESPA/DAX/WIG/Nikkei/KOSPI 등 TradFi 지수만 대상, 크립토 없음. (b) VaR 추정/캘리브레이션(PIT)만 검증, 실제 트레이딩 백테스트 전혀 없음.** | **재시도 근거 혼재** — "배리어 선택 축 제거"라는 방법론적 장점은 진짜이나, 같은 feature set 위에 다른 손실함수를 얹는 것뿐이라 정보량 문제는 그대로. 저비용 cheap-gate 후보이나 기대치는 낮게 |
| "Probabilistic Forecasting Cryptocurrencies Volatility: From Point to Quantile Forecasts", Dudek, Orzeszko, Fiszeder (2025) | arXiv:2508.15922 (**검증됨**) | QRS(Quantile Estimation through Residual Simulation), BTC 대상. **방향이 아니라 realized variance(변동성) 예측** — 예측 정확도만 검증, 트레이딩 백테스트 없음. | **direction_head 대안 아님.** 이 저장소가 유일하게 살아있는 축(사이징/리스크)에 변동성 피처로 검토할 가치는 있으나, 이 문서가 다루는 "라벨링 로직"과는 다른 층위 |
| 저장소 자체 아이디어 #14 (2026-08-09) | `research_quantile_regression_skew_entry_eth_20260809.py` (**저장소 내부**) | 2-quantile GBM skew 방향 신호, 단일 실행, 네거티브 | 위 두 논문이 제안하는 "완전한" 분포적 회귀보다 훨씬 좁은 구현이었다는 게 확인됨 — 완전판을 안 써봤다는 주장 자체는 사실 |

**A.2 종합**: "이산 라벨을 피한다"는 방법론적 매력은 진짜이지만, 이 저장소의 진단된 병목
(feature information content 부재)은 손실함수 선택과 무관한 축이다. 같은 정보로 학습하면
같은 정보량 상한에 부딪힌다 — 실제로 idea #14가 정확히 이 결과를 보였다. **완전한 분포적
회귀는 "재시도 근거 약함"이지만, quantile/distributional 손실은 "배리어 재보정" 계열보다는
싸게 검증 가능하므로 완전히 배제하기보다 최하위 우선순위 cheap-gate로 남겨둔다.**

### A.3 라벨 노이즈에 강건한 학습 (라벨 재설계 대체)

GCE(q=0.7)/ELR/mixup은 이미 시도되고 결론났다(**기존 인용/저장소 내부**,
`eth_odyssey4_gce_canonical_port_20260816.md` — N=5 시드, expert=bull, Δ(GCE−baseline) 5개
시드 중 1개만 양수(+0.0225), 나머지 4개 음수, 되돌림).

| 항목 | 근거 | 이 저장소와의 연결 | 판정 |
|---|---|---|---|
| Confident Learning, Northcutt, Jiang & Chuang (2021) | arXiv:1911.00068, *JAIR* 70 (**검증됨** — 논문 자체는 잘 알려진 표준 인용, 재확인함) | GCE/ELR/mixup(손실 재가중)과 **메커니즘이 다르다** — out-of-sample 예측확률+class-conditional noise process 추정으로 **오분류 의심 표본을 식별·제거/재라벨**하는 방식. 금융/크립토 시계열에 직접 적용한 논문은 이번 검색에서 찾지 못했다. **핵심 문제: Confident Learning의 전제는 "일부 라벨이 클래스조건부 무작위로 뒤집혔다"는 노이즈 모델이다. 그런데 이 저장소가 오늘 진단한 `zigzag_action`의 문제(`eth_zigzag_action_label_entry_timing_momentum_bias_20260819`)는 무작위 노이즈가 아니라 **구조적/체계적** 편향(사후확정 스윙 인식 → 24/24 시드×창 셀 전부 같은 방향의 추세추종)이다. Confident Learning은 "가끔 틀리는 라벨"을 잡아내지, "항상 같은 방향으로 치우친 라벨 정의"는 못 고친다.** | **가정 불일치 — 재시도 근거 약함.** 메커니즘은 진짜 새롭지만 이 저장소가 실제로 가진 문제 유형과 안 맞음 |

**A.3 종합**: 노이즈-강건 학습 축은 "GCE/ELR/mixum이 이미 실패했으니 Confident Learning도
당연히 실패할 것"이 아니라, **더 근본적으로 이 저장소의 라벨 문제가 애초에 "노이즈"가 아니라
"구조적 편향"이라서 이 방법론 계열 전체가 문제를 잘못 짚고 있다**는 게 이번 조사의 결론이다.

### A.4 크립토 특화 최신 사례 (백테스트/Sharpe 포함)

A.1의 AEDL과 Financial Innovation(2025) 논문이 이 항목의 가장 강한 후보였고 둘 다 위에서
다뤘다. 추가로 찾은 것은 대부분 방향 라벨링이 아니라 변동성 예측(A.2의 QRS 논문)이거나 이미
`dl_crypto_trading_literature_survey_20260817.md`가 다룬 것(DRW/G-Research Kaggle — 선형모델
우승)의 반복이었다. **신규로 다룰 가치가 있는 크립토 특화 라벨링 사례는 이번 조사에서 추가로
찾지 못했다** — 이 sub-bullet은 A.1/A.2를 넘어서는 독립적 결과가 없다는 걸 솔직히 명시한다.

---

## Part B — 라벨이 필요 없는 경우

### B.1 강화학습 (직접 보상 최적화)

**이미 CLOSED** — `eth_odyssey4_rl_layer_axis_closed_20260815`: 5개 삽입점(direction/quality,
zig075 veto, exit_head, 포트폴리오 게이트, 사이징) 전부 부정/배제. exit_head는 Gittins Index
Deep RL(arXiv:2405.01157)까지 시도 후 REJECTED_VAL_GATE. 사이징은 RL 사이드카+비-RL Kelly
변형 2종까지 전부 기존 HGB 사이드카에 패배.

이 저장소가 안 써본 RL 변형이 있는지 구체적으로 확인:

| 변형 | 근거 | 이번 저장소에서 안 써본 이유가 되는가? | 판정 |
|---|---|---|---|
| Decision Transformer (시퀀스 모델링 offline RL) | Chen et al. 2021 원조(OpenReview), 트레이딩 적용: "Pretrained LLM Adapted with LoRA as a Decision Transformer for Offline RL in Quantitative Trading", Yun (2024) — arXiv:2411.17900 (**검증됨**, 초록에 정량 백테스트 수치·자산군 명시 없음, 재현코드만 공개) | 저장소 자체 `portfolio_online_bandit_gate_native_20260709.md`가 "action space가 이진+거래표본 작아서 DT/CQL/IQL 대신 보수적 contextual bandit을 썼다"고 **명시적으로 설계 시점에 DT를 고려했다가 의도적으로 배제**했다(docstring 11-13행, arXiv:2301.01298/2110.06169/2305.14550 인용). 즉 "몰라서 안 씀"이 아니라 "검토 후 배제". | 리터럴하게는 미시도 |
| DT가 실제로 유리한 조건 | "When should we prefer Decision Transformers for Offline Reinforcement Learning?", Bhargava, Chitnis, Geramifard, Sodhani, Zhang (ICLR 2024) — arXiv:2305.14550 (**검증됨**, 저장소가 이미 인용한 논문 자체를 재검증) | 이 논문 자체가 "**데이터 부족, 희소 보상(sparse reward), 저품질(suboptimal) 데이터** 환경에서 DT가 유리"하다고 결론짓는다. 이 저장소의 실제 상황은 정반대다 — 매 bar마다 조밀한 보상(실현 PnL)이 있고, 2년+ 5분봉 데이터가 있으며, 병목은 데이터 부족이 아니라 **feature information content 부재**로 40회 이상 독립 확인됐다. | **DT가 유리한 조건 자체가 이 저장소 상황과 안 맞는다** — 논문 자신의 결론으로 판단할 수 있는 드문 케이스 |
| Exploratory(엔트로피 정규화) 확률적 제어 RL | "Reinforcement Learning for Speculative Trading under Exploratory Framework", Zhao, Tse, Zheng (2026) — arXiv:2604.02035 (**검증됨**) | Wang et al.(2020) 엔트로피 정규화 탐색적 제어 프레임워크, Cox 과정 기반 정지시간+HJB+Gibbs 분포 닫힌형 정책 — 기존에 닫힌 REDQ/TQC/CQL/DSAC-T(가치기반/정책경사 계열)와 **수학적 메커니즘이 명확히 다르다**. 페어트레이딩 사례는 언급되나 초록에 크립토/투기자산 백테스트 수치는 없음("37 pages, 14 figures"만 확인). | 메커니즘은 다르나 **실증 근거 자체가 초록 수준에서 확인 안 됨** |

**B.1 종합**: 문헌상 진짜 안 써본 RL 변형이 정확히 2개 존재한다(Decision Transformer,
엔트로피-정규화 탐색적 제어). 그러나 (a) DT는 저장소가 설계 단계에서 이미 검토 후 배제했고,
그 배제 근거가 된 바로 그 문헌(arXiv:2305.14550)이 "DT가 유리한 조건"으로 이 저장소와 정반대
상황을 지목한다. (b) 탐색적 제어 RL은 실증 근거가 사실상 없다(초록만으로는 백테스트 결과 확인
불가). **"더 볼 것 없다"고 완전히 닫아버리기엔 기술적으로 미시도인 게 맞지만, 재시도를
정당화할 근거는 거의 없다 — 우선순위 최하위.**

### B.2 마켓메이킹 / 스탯아브 — 진짜 라벨-불필요 패러다임 후보

저장소 전체(`docs/`, 메모리, `research_line_registry.json`)를 market-making/market-neutral/
statistical-arbitrage/cointegration/inventory-risk/Almgren-Chriss 키워드로 검색한 결과
**단 한 번도 시도된 적이 없다** — 유일한 인접 사례는 `btc_kappa1_invariant_composite_policy_design_20260807.md`의
"maker-first post-only 실행"인데, 이건 **방향성 신호의 체결 방식**(taker→maker)을 바꾼 것이지
방향 자체를 안 거는 전략이 아니다. 그 결과 자체가 이번 판단에 중요한 단서를 준다: **"메이커
실행은 비용을 절반으로 줄이지만 markout edge는 없다"** — 즉 체결 방식만 바꿔서는 edge가
안 생긴다는 걸 이 저장소가 이미 실증했다. 이건 오히려 "그럼 edge의 원천을 방향이 아니라
스프레드/재고 자체로 바꿔야 한다"는 이번 후보의 논리를 강화한다.

| 항목 | 근거 | 방향 라벨이 진짜 불필요한가? | 크립토 컨텍스트(24/7, 펀딩비, 무기한선물) |
|---|---|---|---|
| Avellaneda & Stoikov(2008) 원조 프레임워크 + RL 개선판, "A reinforcement learning approach to improve the performance of the Avellaneda-Stoikov market-making algorithm" | PLOS ONE, DOI:10.1371/journal.pone.0277042 (**검증됨**, 원문 fetch) | **그렇다.** "신경망이 직접 매수/매도 결정을 내리지 않으며" — RL이 조정하는 건 위험회피도(γ)와 스프레드 왜곡(skew) 파라미터뿐, 목표는 방향이 아니라 재고 위험 최소화+스프레드 수익. 실제 BTC-USD L2 tick 데이터(2020-12-07~2021-01-08, 33일)로 백테스트: Alpha-AS가 30일 중 24일(80%) Sharpe 우위, 25일(83%) Sortino 우위. **단, 단일 시드/단일 30일 구간/교차검증 없음 — 이 저장소의 N≥5 시드 기준에는 크게 못 미치는 얇은 증거.** | 순수 BTC 현물, 무기한선물/펀딩비 미반영 |
| "Funding-Aware Optimal Market Making for Perpetual DEXs", Nam Anh Le (2026) | arXiv:2605.06405 (**검증됨**, 원문 fetch) | **그렇다.** "재고는 시장 노출과 상태종속적 펀딩 현금흐름을 모두 생성한다"는 전제로, 펀딩비를 재고 보유 비용으로 취급하는 HJB 최적화 — 방향 베팅 불필요. **크립토 무기한선물 펀딩 메커니즘을 정면으로 다루는, 이번 조사에서 찾은 가장 이 저장소 컨텍스트에 정확히 맞는 논문.** ETH/BTC/SOL 시뮬레이션(100회 holdout)에서 funding-aware HJB가 Avellaneda-Stoikov 대비 평균 성과 개선+재고 RMS 감소. SOL은 risk-adjusted 지표 기준 Pareto 개선 아님. **절대 수익률/Sharpe 수치는 초록에 없음. 아직 peer-review 전 preprint(2026-05).** | ETH/BTC/SOL, 펀딩비 명시적 모델링, DEX 특화(CEX 아님) |
| Almgren & Chriss(2001) 최적집행 + 크립토 확장: "Slippage-at-Risk", "Dynamic Collateral Control for Permissionless Spot Perpetual Basis Trading" | Almgren & Chriss, *J. Risk* 3(2), 2001(고전, arXiv 없음); arXiv:2603.09164, arXiv:2605.05089 (**검증됨**, 제목·인용 관계 확인) | 부분적. 최적집행 자체는 "이미 방향이 결정된 뒤 어떻게 체결하나"의 문제라 방향 라벨을 대체하지 않는다. 단 `Dynamic Collateral Control...Basis Trading`(2605.05089)는 **현물+무기한선물 양다리(spot-perp basis)**를 다뤄 진짜 delta-neutral carry에 더 가깝다. | 크립토 무기한선물 담보/베이시스 특화 |

**저장소 내부 F3-A와의 정밀 대조(중요)**: `research_line_registry.json`에 이미 등재된
`global_funding_carry_contrarian`(CLOSED)은
`docs/test_designs_duckdb_live_20260719/results/factor_execution_results_20260719.md`
"F3-A 펀딩 캐리"를 가리킨다 — 실제로 테스트된 건 **"펀딩 부호가 다음 정산까지 지속된다"는
가정 하의 단일 레그(퍼프 한쪽만 보유) 방향성 베팅**이었다(24개 변형 전부 kill, 펀딩 컴포넌트
자체가 전부 음수). **이건 진짜 델타뉴트럴 cash-and-carry(현물 롱 + 퍼프 숏 동시 보유로 가격
방향 노출을 원천 제거하고 펀딩만 수취)가 아니다** — F3-A는 방향 노출이 있는 상태로 펀딩
부호만 베팅한 것이라, 사실상 방향 라벨링의 변형이었다. **따라서 F3-A의 CLOSED 판정이 진짜
market-neutral 마켓메이킹/베이시스 캐리 축을 선점하지 않는다** — 이 구분을 흐리면 안 된다.

**인프라 현실 점검(중요)**: `trading_bot_modules/position_router.py`/`binance_execution.py`를
확인한 결과, 이 봇은 이미 **단방향(one-sided) maker-first 진입**(post-only 시도 후 시장가
폴백, `binance_runtime_config.py`의 `maker_entry_fallback_market`) + TP는 resting LIMIT
주문 + SL은 STOP_MARKET 구조를 갖고 있다. 그러나 이건 "방향이 결정된 뒤 그 한쪽을 메이커로
체결"하는 것이지, **진짜 마켓메이킹(양방향 동시 호가, 실시간 재고 스큐 조정)과는 아키텍처가
근본적으로 다르다.** 이 저장소가 현재 spot 포지션을 보유하는지도 코드에서 확인되지 않았다 —
cash-and-carry에는 spot 레그가 필수다.

**B.2 종합 — 판정: 진짜 라벨-불필요, 근거 존재, 그러나 인프라/비즈니스모델 전제가 완전히
다르다.** 이건 "quantile regression은 여전히 forward return이라는 지도신호가 필요하다"는
구분에서 벗어나는 유일한 후보다 — 마켓메이킹의 핵심 손익원천(스프레드 캡처+펀딩 수취)은
방향 예측이 필요 없다는 게 문헌과 메커니즘 양쪽에서 확인된다. 단, 이걸 이 저장소에 붙이는
것은 "새 라벨을 시도"하는 수준의 실험이 아니라 **완전히 다른 상품/인프라를 새로 만드는
수준의 결정**이다. **다음 액션은 코드 실험이 아니라 스코핑**: (1) 실제 계정의 메이커
수수료/등급, (2) 거래소가 동시 양방향 resting 주문을 지원하는지, (3) spot 레그 보유 가능
여부, (4) 이 정도 자본/레버리지 규모에서 마켓메이킹 경제성이 성립하는지.

### B.3 자기지도 사전학습 — "방식이 문제였나, 구현이 문제였나"

JEPA(latent-prediction + InfoNCE temporal-contrastive, 소형 transformer)는 이미 BTC에서
시도되고 CLOSED됐다(`docs/btc_deepfeat_jepa_unified_panel_closed_20260804.md`, **기존
인용/저장소 내부**): 124개 원시피처 union 위에 사전학습, 0/9 threshold configs, **학습된
임베딩이 원본 raw 피처보다 다운스트림 LightGBM에서 순위가 낮았다**(중요도 랭크 56-121위 /
148개 중, 원본 DVOL/on-chain보다 못함).

이게 "자기지도 사전학습이라는 패러다임 전체의 실패"인지 "이 특정 JEPA 구현(latent-prediction+
InfoNCE)의 실패"인지 구분하기 위해 다른 pretext task 계열을 조사했다:

| Pretext task 계열 | 근거 | JEPA(latent-prediction+contrastive)와 메커니즘이 다른가? | 금융/크립토 백테스트 |
|---|---|---|---|
| Siamese 시간축 정렬(lineage embedding) | "TimeSiam", Dong, Wu, Wang et al. (2024) — arXiv:2402.02475 (**검증됨**) | 다름 — "무작위 마스킹이나 계열별 유사성 계산"(JEPA/contrastive 계열이 하는 것) 대신 과거-현재 부분계열 간 시간적 거리를 lineage embedding으로 명시적으로 학습 | 13개 표준 벤치마크, 금융 도메인 명시 없음 |
| 자기교정 적응 마스킹(pseudo-label 재생성) | "Not All Data are Good Labels: On the Self-supervised Labeling for Time Series Forecasting", Yang, Zhang, Liang, Lu, Chen, Li (NeurIPS 2025 Spotlight) — arXiv:2502.14704 (**검증됨**) | 다름 — 이 논문 자체가 "기존 self-supervised 방식이 시계열 예측에서 항상 효과적이지 않다"는 걸 정면 비판하며, 과적합 성분을 골라내 pseudo-label로 대체하는 SCAM 제안 | 11개 실제 데이터셋, 금융 도메인 명시 없음. **이 저장소의 JEPA 실패와 정확히 같은 방향의 메타비판** |
| 대조학습(CPC 기반), 트레이딩 적용 | "Trading through Earnings Seasons using Self-Supervised Contrastive Representation Learning", Ye & Schuller (2024) — arXiv:2409.17392 (**검증됨**) | InfoNCE 계열이라 JEPA의 contrastive 보조항과 유사하나, **다운스트림이 실제 트레이딩 적용**이라는 점에서 실용적으로 가장 가까운 선례 | **주식**(암호화폐 아님), 백테스트 수치는 초록에서 확인 안 됨 |
| Diffusion+autoregressive | "TimeDART" (arXiv:2410.05711), "LeNEPA: No-Augmentation Next-Latent Prediction"(arXiv:2607.00958, 2026) | LeNEPA는 이름 자체가 JEPA 계열의 최신 변형(augmentation 없는 next-latent prediction) — 메커니즘상 가장 가까움 | 리뷰 안 함(시간 제약, 제목 수준만 확인) |

**B.3 종합**: 엄밀히 말해 "자기지도 사전학습"이라는 패러다임은 완전히 소진되지 않았다 —
JEPA는 그 안의 한 pretext-task 계열(latent-prediction+contrastive)일 뿐이고, Siamese
정렬/적응 마스킹 등은 메커니즘이 다르다. **그러나** 이 저장소의 진단된 근본원인은 "표현학습
방식이 나빠서"가 아니라 "**같은 원시정보 위에서는 어떤 재구성도 새 정보를 만들지 못한다**"는
것이고, JEPA의 실제 실패 양상(임베딩이 원본보다 정보손실)이 정확히 이걸 실증한다 — 학습된
표현이 원재료보다 못했다는 건 "이 특정 목적함수가 나빴다"보다는 "압축 자체가 손해"라는
신호에 가깝다. **따라서 같은 OHLCV/파생지표 feature union 위에서 다른 pretext task를 또
시도하는 것은 재시도 근거가 약하다.** 예외는 **완전히 새로운 원시정보**(B.2/아래 결론의
LOB 원시 L2 데이터)에 적용하는 경우 — 그건 "정보를 재포장"이 아니라 "새 정보 위에 표현학습을
얹는" 것이라 다른 질문이 된다.

### B.4 온라인학습/밴딧 (실현 보상에서 직접 적응)

`portfolio_online_bandit_gate_native_20260709.md`가 이미 contextual bandit(CQL/IQL/DT를
검토 후 "action space 이진+표본 작음"을 이유로 보수적 버전으로 단순화)을 포트폴리오 진입
게이트에 시도했고 OOS -18.03%로 닫혔다(**기존 인용**, `eth_odyssey4_rl_layer_axis_closed_20260815`).

| 항목 | 근거 | 완전히 라벨-프리인가? | 이 저장소와의 연결 |
|---|---|---|---|
| 고전 온라인 포트폴리오 선택(Cover's Universal Portfolio, exponentiated gradient, follow-the-regularized-leader) | 고전 이론(Cover 1991 등), 최근 크립토 사례: "Improving Portfolio Optimization Results with Bandit Networks", Fonseca, Coelho e Silva, de Castro (2024/2025) — arXiv:2410.04217 (**검증됨**) | **그렇다** — 예측모델도 forward-return 타겟도 없이, 실현 수익률에서 직접 가중치를 온라인으로 재조정한다(가치함수/정책망 없음). 저장소가 시도한 contextual bandit(CQL/IQL 스타일)과 메커니즘이 다르다. | ADTS/CADTS(Thompson Sampling 기반) 결합 아키텍처, 크립토+S&P 백테스트, "OOS Sharpe가 최고 성능 classical 모델 대비 20% 높음" 주장(초록 확인, 절대수치는 본문 필요). **그러나 이 방법이 배분하는 "arm"(h48qual/zig075류 하위 신호) 자체가 이 저장소에서 OOS 안정적 edge가 없다는 게 반복 확인된 사실이다 — bandit이 재분배를 아무리 잘해도 배분 대상에 edge가 없으면 결과는 같다.** |

**B.4 종합**: 메커니즘 자체는 진짜 라벨-프리이고 미시도이지만, 이 저장소의 "레짐/신호
자체가 방향 정보를 못 담는다"는 반복 확인된 결론이 bandit의 상한도 같이 제한한다. **직접
재현 우선순위는 낮으나, 완전히 다른 메커니즘이라는 점에서 "더 볼 것 없음"이라고 말하기엔
정직하지 않다 — 사이징/자산배분 레이어(이 저장소에서 유일하게 살아있는 축)에 한정해
저비용으로 검토할 가치는 남아있다.**

### B.5 "신호는 실재하나 무료 벤치마크에 항상 흡수됨" → 규칙기반 실행 트리거

이 질문에 대한 가장 강력한 증거는 외부 문헌이 아니라 **이 저장소 자체에 이미 있다**:

- **`eth_odyssey3_zig075_short_entry_veto_uptrend_confirmed_20260815`**(CONFIRMED, 저장소
  내부): `dual_momentum` rolling 1주 percentile 기반 결정론적 규칙(ML 학습 없음)으로 zig075
  SHORT 진입을 지속상승장에서 거부. 게이트 3창은 무발동이었지만 2025-Q3 참고창에서
  PnL −15.86%→+20.17% 부호반전. 미러(LONG/하락장)도
  `eth_odyssey4_zig075_long_entry_veto_downtrend_confirmed_20260815`로 확인. **이게 정확히
  사용자가 묻는 "예측모델 라벨 대신 규칙기반 실행 트리거"의 실사례이고, 이 저장소 전체에서
  entry-side 개입 ~29회 연속 실패 끝에 나온 첫 성공이다.**
- **`eth_liquidation_feed_epoch_defect_20260817`**(저장소 내부, 진행중): 청산 데이터 오염이
  2026-07-18에 해소된 뒤 탐색 스캔에서 "청산 방향 컨트래리언"(`liq_net_z_12`, 롱 청산→상방)이
  일관된 부호로 나타났고, **결정 게이트(≥2026-09-15)에 이미 등록**돼 있다. 이것도 ML 라벨이
  아니라 임계값 기반 트리거로 설계 중이다.

문헌에서 이 패턴의 일반적 타당성을 뒷받침하는 근거:

| 항목 | 근거 | 연결 |
|---|---|---|
| 청산 캐스케이드의 산업 문헌 | WebSearch로 확인한 업계 자료(TradingView 인디케이터, 거래소 교육자료) — 학술 동료심사 아님, 신뢰도 낮음으로 명시 | "가격 가속도 임계값", "유동성 점수 임계값" 등 **룰기반/기계적 탐지가 표준**이라고 서술 — ML 예측이 아니라 임계값 트리거라는 이 저장소의 접근과 정합적이나, 학술 근거는 약함 |
| OFI를 직접 신호로(Kolm, Turiel & Westray) | SSRN 3900141, `entry_exit_edge_root_cause_and_literature_review_20260809.md`가 이미 인용(**기존 인용**) | 정확히는 "OFI 피처 위에 LSTM/ANN을 여전히 학습"시키는 방식이라, "모델 자체를 스킵"하는 순수 룰기반과는 다르다 — 참고할 건 "피처가 아키텍처를 이긴다"는 정도 |
| 고전 기술지표 규칙(RSI/MACD/MA) | 이미 문헌조사로 선제 기각(`eth_classical_technical_indicator_literature_check_20260817`, **기존 인용**) | **재추천 안 함** — 평균회귀형 룰기반 트리거는 이미 닫힌 축과 겹친다 |

**B.5 종합 — 이 문서 전체에서 가장 근거가 강한 항목.** zig075 veto는 (a) 실제로 이 저장소에서
CONFIRMED됐고, (b) 라벨/예측 없이 구조적 정보(추세지속 percentile)를 직접 트리거로 쓰며,
(c) 이 저장소의 자체 메타발견("레짐은 방향은 못 담아도 노출/구조 정보는 담는다")과 정합적이다.
**단, 이건 이 세션이 "새로 발견"한 게 아니라 이미 존재/진행 중인 축이다 — 이 문서의 기여는
그걸 Part B의 답으로서 명시적으로 정리하고, 외부 문헌으로 보강한 것이다.** 고전 기술지표
규칙과 혼동하지 않아야 한다 — 성공한 두 사례(zig075 veto, 청산 컨트래리언 후보) 모두
**"방향을 새로 만드는" 트리거가 아니라 "이미 나온 방향/노출을 거르는" 트리거**라는 공통점이
있다. 이는 A.3에서 도출한 결론(메타레이블링/거부옵션 형태가 안전하고, "새로 예측"하는 형태는
위험하다)과도 정확히 같은 방향이다.

---

## 종합 우선순위 랭킹 (강한 근거 순)

| 순위 | 항목 | Part | 근거 강도 | 다음 액션 |
|---:|---|:---:|---|---|
| 1 | 규칙기반 실행 트리거(exposure veto/gate) 확장 | B.5 | **최강** — 이 저장소 내부 CONFIRMED 사례 존재(zig075) + 2번째 후보 이미 파이프라인 등록(청산 컨트래리언, 09-15 게이트) | 새 리서치 아님 — 기존 청산 게이트를 계획대로 진행. 향후 유사 구조적 신호(OFI 등)를 만나면 "예측 모델"이 아니라 "거부/축소 트리거"로 먼저 검토 |
| 2 | 마켓메이킹/펀딩 델타뉴트럴 스코핑 | B.2 | 강함(문헌+메커니즘) 그러나 이 저장소 실증 없음, 인프라 전제 전무 | **코드 실험 금지, 스코핑 먼저**: 메이커 수수료 등급, 양방향 동시 호가 지원 여부, spot 레그 보유 가능성, 자본/레버리지 규모의 경제성부터 확인 |
| 3 | quality_head → 시계열전용 adaptive conformal prediction (GLCP/RCCP) | A(quality 축, 오늘 1차 조사 연장) | 강함(오늘 세션 arXiv:2607.23165/2608.10553 검증) + 겨냥하는 문제(`eth_ilias1_h48qual_quality_gate_selectivity_shift_20260818`의 threshold VAL-과적합)가 이미 구체적으로 진단돼 있음 | direction 축은 안 건드리고 quality_threshold 선택 메커니즘만 conformal 커버리지로 교체하는 저비용 cheap-gate |
| 4 | Directional-Change 이벤트 샘플링 실전 end-to-end 검증 | A.1 | 중간 — "구현됐으나 미검증"이 확인된 유일한 항목, 그러나 근본원인(정보량) 진단과 별개 축 | h48qual 또는 zig075 라벨 파이프라인에 `event_method='directional_change'`로 1회 cheap-gate, 기대치는 낮게 |
| 5 | 자기지도 사전학습 — LOB 원시데이터 위에서만 재검토 | B.3 (A.1의 LOB 축과 결합) | 중간 — 새 원시정보와 결합해야만 유효, 기존 피처 위 재시도는 근거 약함 | LOB raw L2 3-tier 축적기간(exploratory 2026-09-14) 도달 후, TLOB류 지도학습과 별개로 self-supervised pretext 1개(TimeSiam류) 저비용 비교 |
| 6 | 완전 분포적 회귀(CRPS/NLL 직접최적화) | A.2 | 약함 — 배리어축 제거는 방법론적 장점이나 크립토/백테스트 근거 전무, 정보량 문제 우회 못함 | 최하위 우선순위 cheap-gate, "재현 여부"보다 "방법론 갭 메우기" 목적으로만 |
| 7 | 고전 온라인 포트폴리오 선택(Cover/EG/FTRL) | B.4 | 약함 — 메커니즘은 순수 라벨프리이나 배분 대상 자체 무엣지 | 사이징/자산배분 레이어에 한정해서만 저비용 검토, direction 대체로는 부적합 |
| — | Confident Learning | A.3 | 약함 — 가정(무작위 노이즈) 불일치 | 재시도 비권장 |
| — | AEDL / GA-triple-barrier | A.1 | 약함 — 이미 배제/흡수 결정 유지 | 재시도 비권장 |
| — | Decision Transformer / 탐색적 엔트로피 RL | B.1 | 약함 — 자체 인용 문헌이 "이 저장소와 안 맞는 조건에서 유리"라고 명시, 또는 실증 근거 전무 | 재시도 비권장 |
| — | RL 일반(REDQ/TQC/CQL/DSAC-T 등) | B.1 | CLOSED 재확인 | 재추천 안 함 |

---

## 참고문헌

### 저장소 내부(1차 자료)
- `docs/label_methodology_survey_20260815.md` — 40+ 라벨 방법론 종합, 핵심 메타발견
- `docs/entry_exit_edge_root_cause_and_literature_review_20260809.md` — 선행 유사 조사, CUSUM+TB+DL 실증 재현(Part 5A) 포함
- `docs/deep_learning_for_crypto_trading_literature_survey_20260817.md` — 6축 DL 전수조사, LOB축 우선순위 지목
- `docs/model_contracts/research_line_registry.json` — 34개 공식 CLOSED 라인
- `docs/btc_deepfeat_jepa_unified_panel_closed_20260804.md` — JEPA CLOSED 원본
- `docs/experiments/eth_odyssey4_gce_canonical_port_20260816.md` — GCE(q=0.7) N=5시드 결과
- `docs/btc_kappa1_invariant_composite_policy_design_20260807.md` — 메이커 실행 vs markout edge 무관 실증
- `docs/test_designs_duckdb_live_20260719/results/factor_execution_results_20260719.md` — F3-A 펀딩캐리(단일레그) CLOSED
- `docs/model_contracts/eth_candidate_lob_microstructure_contract_20260817.md`,
  `..._data_resources_20260817.md` — LOB raw L2 데이터 연결 상태, 3-tier 축적기준
- `core/event_label_engine.py` — triple-barrier/trend-scanning/meta-labeling 통합 엔진, directional_change 구현 확인
- 메모리 노트(originSessionId 2a48cbbb, 2026-08-19): `reference_direction_quality_exit_label_methodology_20260819`,
  `eth_zigzag_action_label_entry_timing_momentum_bias_20260819`
- 메모리 노트: `eth_odyssey4_rl_layer_axis_closed_20260815`, `eth_candidate_lob_microstructure_data_scoping_20260817`,
  `eth_liquidation_feed_epoch_defect_20260817`, `eth_odyssey3_zig075_short_entry_veto_uptrend_confirmed_20260815`,
  `eth_odyssey4_zig075_long_entry_veto_downtrend_confirmed_20260815`, `evidence_signal_quant_use_subproject`,
  `ilias_eth_human_direction_risk_management_subproject`, `eth_classical_technical_indicator_literature_check_20260817`

### 외부 문헌(이번 세션 신규 검증, arXiv ID 순)
- 1804.05394 — Becker, Cheridito, Jentzen (2018), *Deep Optimal Stopping* (기존 인용, exit 대안)
- 1901.09192 — Geifman & El-Yaniv (2019), *SelectiveNet* (기존 인용, quality 대안)
- 1911.00068 — Northcutt, Jiang & Chuang (2021), *Confident Learning* (검증됨, A.3)
- 2107.07511 — Angelopoulos & Bates (2021), conformal prediction 서베이 (기존 인용)
- 2301.01298, 2110.06169 — CCQL, IQL (저장소 자체 인용, 검토 후 배제)
- 2305.14550 — Bhargava et al. (ICLR 2024), *When should we prefer Decision Transformers?* (검증됨, B.1)
- 2402.02475 — Dong et al. (2024), *TimeSiam* (검증됨, B.3)
- 2409.17392 — Ye & Schuller (2024), *Trading through Earnings Seasons* (검증됨, B.3)
- 2410.04217 — Fonseca et al. (2024/2025), *Bandit Networks* (검증됨, B.4)
- 2410.05711 — *TimeDART* (미검증, 제목만, B.3)
- 2411.17900 — Yun (2024), *LoRA Decision Transformer for Quant Trading* (검증됨, B.1)
- 2502.14704 — Yang et al. (NeurIPS 2025), *Not All Data are Good Labels* (검증됨, B.3)
- 2508.15922 — Dudek, Orzeszko & Fiszeder (2025), 크립토 변동성 quantile 예측 (검증됨, A.2)
- 2508.18921 — Michańków (2025), 금융 수익률 확률분포 예측 (검증됨, A.2)
- 2603.09164 — *Slippage-at-Risk*, 무기한선물 유동성 리스크 (검증됨, B.2)
- 2604.02035 — Zhao, Tse & Zheng (2026), 탐색적 프레임워크 RL (검증됨, B.1)
- 2605.05089 — *Dynamic Collateral Control for Perpetual Basis Trading* (검증됨, B.2)
- 2605.06405 — Le (2026), *Funding-Aware Optimal Market Making for Perpetual DEXs* (검증됨, B.2)
- 2607.00958 — *LeNEPA* (미검증, 제목만, B.3)
- 2607.23165, 2608.10553, 2608.01494, 2608.14106 — 오늘 세션 arXiv ID 재검증 완료(모두 실재 확인, A/B 전반 배경)
- DOI:10.1371/journal.pone.0277042 — RL-Avellaneda-Stoikov, BTC L2 백테스트 (검증됨, B.2)
- DOI:10.3390/math12050780 — GA-triple-barrier (기존 인용, A.1)
- DOI:10.1186/s40854-025-00866-w — CUSUM+TB+DL, 크립토 (기존 인용, A.1, 이미 재현·기각됨)
- MDPI *Applied Sciences* 15(24):13204 — AEDL (부분검증, A.1)
