# 왜 이 프로젝트는 "엔트리 엣지"를 찾지 못하고 밸런싱/리스크만 살아남는가
### 구조적 원인 분석 + 진입·청산 최신 문헌 리뷰 (2026-08-09)

이 문서는 두 파트로 구성된다.
1. **왜 그런가** — 메모리 180개 파일 + `docs/` writeup을 근거로 재구성한, 이 프로젝트 자체의 증거 기반 원인 분석.
2. **최신 문헌은 무엇을 말하는가** — 2025~2026 진입/청산/검증방법론 논문 리서치. 이미 닫힌 라인과 겹치지 않는 것 위주로 정리.

---

## Part 1. 왜 엔트리 엣지만 실패하는가

### 1.1 결론 먼저

이건 우연이나 실력 문제가 아니라 **이미 이 프로젝트가 2026-07-30에 문서로 명시적으로 내린 결정**이다. `docs/pipeline_integrity_and_research_redesign_20260730.md`:

> "지금까지 축적된 조사 결과를 한 줄로 정리하면... 병목은 '더 좋은 모델을 못 찾는 것'이 아니라 **측정을 신뢰할 수 없다는 것**이다." → **"방향 예측 개선은 접는다."**

그 뒤 P3 우선순위는 **(a) 리스크/사이징 레이어 — "새 알파를 요구하지 않기 때문에" 최우선, (b) 비상관 자산 추가, (c) 대체 데이터(2026-10 예정)** 순으로 명시돼 있다. 즉 지금 보이는 패턴(리서치 결과가 전부 밸런싱/리스크뿐)은 그 결정이 정확히 의도한 결과다. 문제는 "왜 실패하는가"가 아니라 "그 결정이 옳았는가, 그리고 대체 데이터/방법론 축에서 뭘 더 할 수 있는가"에 가깝다.

### 1.2 엔트리 엣지 실패 카탈로그 (증거)

에이전트가 메모리 전체를 훑어 재구성한 실패 목록 중 핵심만 발췌한다 (전체는 하단 부록):

| 축 | 결과 | 근거 |
|---|---|---|
| 108종 JM/czz 레짐 MoE | 0/108 VAL 통과 | `project-btc-regime-expert-lines-all-closed-20260808` |
| 레짐-조건부 엔트리 (모든 게이트 통과 후) | VAL +18.4% → OOS **-19.5%** (9번째 VAL→OOS 반전) | 동일 |
| 레짐 **차이(differential)** 자체의 부호 유지율 | train→VAL 84-96% → train→**OOS 36-52%** (랜덤 베이스라인 44-65%보다 낮음) | 동일, ETH도 재현 |
| SOL DL/RL 11개 아키텍처 서베이 | 오라클 천장 VAL +1,069%, **캡처 0%** | `project-sol-dl-rl-architecture-survey-20260807` |
| ETH 1h PatchTST 트랜스포머 | 5/5 시드 VAL 양수 → **5/5 시드 OOS 음수** (부호 만장일치) | `project-eth-1h-patchtst-new-architecture-failed-20260731` |
| BTC Kronos 파운데이션모델 임베딩 | frozen ΔAUC +0.0006(무의미), fine-tuned는 오히려 -0.0133 | `project-btc-kronos-layerA-stage0-closed-20260807` |
| BTC 6개 아키텍처(TabM/GRU/VSN/Chronos-2 등) | 전부 plain LightGBM에 패배, 600-config 스윕 **0개 VAL 수익** | `project-btc-5m-zigzag-architecture-session-arc-20260806` |
| BTC 지그재그 중앙값 파형 | 모든 θ에서 진폭≈1.9θ vs 확인비용 2θ → **63-74%가 구조적으로 수익화 불가능** | `project-btc-label-design-arc-and-next-session-queue-20260808` |
| BTC exit 피벗 분류기 | OOS AUC **0.946-0.951**(거의 완벽)인데도 exit 천장 캡처 ~0%, 가장 확신도 높은 구간이 가장 마이너스 EV | research_line_registry, `btc_zigzag_pivot_exit_primitives` |
| BTC event-gate (비방향성) | 8/8 롤링 스트레스 통과("역대 최강 신호")했지만 방향화 4가지 시도 전부 실패 | `project-btc-event-gate-stage1-stable-lift-20260804` 등 |

공통 패턴: **오라클 천장은 항상 크고 뚜렷한데(때론 +1,000%~+2,900%), 캡처율은 항상 0%에 가깝고, VAL 캡처와 OOS 캡처의 상관관계가 사실상 0(Spearman 0.061)이다.**

### 1.3 리스크/사이징이 살아남는 이유 (실패 사례 포함, 균형 잡힌 그림)

주의: **리스크/사이징도 대부분 실패한다.** ETH exit 로직만 21라운드 연속 실패했고, BTC TP/SL 재보정도 OOS를 더 악화시켰다. 심지어 "채택됐던" 레짐 사이징 오버레이도 나중에 자체 effect-size 감사에서 **다운그레이드**됐다 (bootstrap P=0.739였지만 실제 t=-0.99, p=0.33, 리스크 채널 방향 자체가 반대, VAL→OOS에서 67배 반전). 그러니 "리스크는 쉽다"가 아니라 **"좁은 부분집합만 산다"**가 맞는 설명이다. 살아남은 것들의 공통점:

1. **예측이 아니라 차단/재분배.** h48qual은 자기 트레이드로는 -0.59%를 벌지만, 제거하면 라우터가 51.42pp를 잃는다 — 이유는 "버는" 게 아니라 지그재그의 나쁜 진입 10건을 **막기** 때문 (`project-eth-h48qual-contribution-is-blocking-not-earning-20260808`).
2. **자유도가 극히 작다.** 채택/검토된 사이징 레버는 보통 1~4개 스칼라 파라미터(배율, 슬롯 수, 임계값 하나)를 그리드 몇 개 점에서 스윕한 것뿐이다. 반면 엔트리 탐색은 108종 MoE, 600-config 아키텍처 스윕, 60종목 스크린처럼 훨씬 큰 탐색 공간을 쓴다. `core/selection_stats.py`의 자체 계측: 이 프로젝트 지표 유니버스에서 나이브 탐색 "승자"는 전부 **기대 최대노이즈 바닥의 0.24배 이하**, DSR은 사실상 0 — 탐색 공간이 클수록 우연한 "발견"이 더 잘 만들어진다는 걸 프로젝트 스스로 계산으로 증명해놓은 상태다.
3. **레짐은 변동성/노출 정보는 담아도 방향 정보는 담지 않는다** — `project-eth-regime-differential-dead-20260808`에 명시: "regimes pay on exposure, not direction." 그래서 동일한 레짐 탐지기로 사이징을 걸면(정상성이 있는 변동성 상태 활용) 살아남을 여지가 있고, 엔트리를 걸면(정상성 없는 방향-페이오프 관계 필요) 죽는다.
4. **라벨의 숨은 유효표본수 손실.** 삼중장벽류 라벨은 겹치는 해상 윈도우 때문에 명목 43,798행이 실제 유효표본 **4,058개(10.8배 과대추정)**밖에 안 된다 (`project-btc-tripbarrier-baseline-is-seed-artifact-20260807`). 사이징/청산 판단은 이미 실현된 트레이드 원장(겹치지 않음) 위에서 평가되므로 이 함정이 없다.
5. **실패 양상의 비대칭.** 나쁜 엔트리 모델은 파국적으로 실패한다(Omega4.7-RL: OOS -88~-99%, 엔트로피 붕괴). 나쁜 사이징은 완만하게 저하된다. 그래서 애매한 리스크 결과는 "shadow 유지"로 오래 살아남고, 엔트리는 깔끔하게 CLOSED로 등록된다 — 이건 리스크가 실제로 더 잘 통과해서가 아니라 실패 모드 자체가 부드럽기 때문이다.

### 1.4 종합: 5가지 구조적 이유

1. 이 시장(BTC/ETH/SOL 5분봉)은 변동성/레짐 지속성은 갖지만, 레짐 조건부 방향-페이오프 관계는 정상적이지 않다 — 세 자산 모두 ΔAUC가 OOS에서 랜덤보다 낮게 무너진다.
2. 엔트리 탐색은 사이징 탐색보다 훨씬 큰 조합 공간(피처×아키텍처×라벨×탐지기)을 같은 ~2년 데이터에 반복적으로 던지므로, 프로젝트 자체 DSR/PBO 계측이 보여주듯 통계적으로 이미 과탐색된 상태다.
3. 순방향 라벨(삼중장벽/지그재그)은 겹침으로 인한 숨은 유효표본수 손실(~10배)을 갖지만 사이징/청산 판단은 그렇지 않다.
4. 방향 실패는 파국적이고 사이징 실패는 완만해서, 리스크 쪽 애매한 결과가 통계적으로 안 죽고 오래 살아남는다(관찰 편향).
5. 프로젝트의 "과거 성공"들 대부분은 실제로는 **버그 수정이거나 복잡도 제거**였다 (SOL v2는 하드코딩 나눗셈 상수 버그 수정, ETH notional multiplier 권고는 "레버 제거", duration-gate 권고는 "VAL로 튜닝한 필터를 쓰지 말라"). 이런 주장은 "새 정보를 찾았다"보다 훨씬 낮은 통계적 부담을 진다.

**BTC 자체 구조적 결론** (`project-btc-ceiling-and-eth-vs-others-structural-20260720`): "BTC가 가장 깊고 가장 많이 아비트라지/경쟁되는 시장이라는 게, 아직 못 찾은 버그라는 가설보다 더 그럴듯한 설명이다." — 이건 프로젝트가 이미 시장 효율성 가설로 수렴했다는 뜻이고, 아래 Part 2의 최신 문헌이 이걸 정량적으로 뒷받침한다.

---

## Part 2. 최신 문헌 리뷰 (2025-2026)

### 2.1 "발견"의 대부분이 가짜라는 검증방법론 문헌 — 프로젝트 경험과 정확히 일치

- **Spurious Predictability in Financial Machine Learning** (arXiv 2604.15531, 2026): "적응적 명세 탐색(adaptive specification search)"은 **완전한 마팅게일-차분 귀무가설(순수 랜덤워크) 하에서도** 통계적으로 유의한 백테스트를 만들어낸다. 저자들은 "falsification audit" — 제로예측력 합성 환경 + 미시구조 플라시보에 대해 같은 워크플로를 돌려서, 거기서도 유의하게 나오면 그 워크플로 자체를 무효로 판정 — 를 제안한다. **이 프로젝트가 사후적으로(닫힌 뒤에) 하던 순열검정(permutation test)을, 신호를 OOS에 태우기 전에 선제적으로 거치는 게이트로 승격시키라는 뜻.**
  - **구현 완료 (2026-08-09)**: `core/selection_stats.py`의 `falsification_audit()` — real best-of-N Sharpe를 (a) 제로예측력 i.i.d. 가우시안 널, (b) demean 후 circular block-bootstrap한 미시구조 플라시보 널, 두 개와 비교. 둘 다에서 `min_percentile`(기본 0.95) 이상이어야 게이트 통과. 테스트: `test/test_selection_stats.py`.
  - **배선 완료 (2026-08-09)**: `pipeline/architecture_workbench.py`의 `assert_effect_size_gate()`에 `falsification=` 파라미터 추가, contract schema v3에 `selection.effect_size_gate.falsification_audit_required`(필수 선언, true/false)와 `min_falsification_percentile`(기본 0.95) 필드 추가. `falsification_audit_required: true`인데 리포트가 없으면 즉시 실패, 있으면 두 percentile 모두 임계값을 넘어야 통과. 테스트: `test/test_architecture_workbench.py::EffectSizeGateFalsificationTest`. `docs/architecture_workbench.md` 갱신 완료.
- **Interpretable Hypothesis-Driven Trading — Rigorous Walk-Forward Validation Framework** (arXiv 2512.12924): 미국주식 100종목, 34개 독립 테스트 구간, 실거래비용 포함. 5개 미시구조 패턴군 테스트 → 통계적으로 유의하지 않은 결과(p=0.34)를 그대로 발표. 핵심 발견: **일봉 OHLCV 기반 미시구조 신호는 정보유입/거래활동이 높을 때만 작동하고 평상시엔 사라진다** — 이 프로젝트의 "레짐이 변동성/노출은 담지만 방향은 못 담는다"는 결론과 같은 방향.
- **Structural Limits of OHLCV-Based Intraday Signals in MNQ Futures: A Systematic Falsification Study** (arXiv 2605.04004): **5분봉**, 947일, **14개 신호군**을 엄격한 배포 기준(t≥2.0, ≥30 트레이드, 실비용 차감 후 순수익, 연도별 일관성)으로 테스트 → **전부 탈락**(총이익 0.07~1.50pt/트레이드, 왕복비용 2pt에 미달). 유일하게 통과한 2개는 순수 가격패턴이 아니라 **RTH 컨플루언스·런던세션**이라는 시간대/유동성 효과였다. 이 프로젝트의 5분봉·다중 라벨·다중 아키텍처 전멸과 거의 동형(isomorphic) 결과이며, 살아남은 두 신호가 **세션/유동성 효과**라는 점이 아래 2.6의 제안과 직결된다.
- **False discovery 일반론**: 표준 5% 유의수준 하에서 약 20번 반복 탐색하면 가짜 전략 하나가 "발견"된다는 건 이미 알려진 결과이고, 2025년 재검토는 "새로 발견된 팩터는 t>2.0이 아니라 t>3.0을 요구해야 한다"고 제안한다. Walk-Forward는 Combinatorial Purged CV(CPCV)보다 허위발견 억제력이 약하다는 비교 결과도 있다 — 다만 CPCV는 크립토처럼 히스토리가 짧은 자산에는 실용성이 떨어진다는 한계도 같이 보고된다.

### 2.2 시장 효율성 / 알파 붕괴 — "왜 지금 특히 더 어려운가"

- **AI-Driven Alpha Decay: Algorithmic Homogenization, Reflexive Signal Erosion, and the Paradox of Intelligent Markets** (arXiv 2605.23905, 2026-03): 신호 크라우딩·수행적 신호 침식(performative erosion)·"Red Queen" 경쟁이라는 세 채널을 게임이론 모델로 결합. 알파 반감기 h(φ)=ln2/[θ+δ(φ)] 도출. 현재 AI 채택 수준(φ≈0.7)에서 **신호 반감기가 AI 이전 5-7년에서 18개월로 단축**됐다고 추정. → 5분봉 방향예측처럼 흔한 문제 형태일수록 경쟁이 이미 심하게 침식시켰을 가능성이 높다.
- **Not All Factors Crowd Equally: Modeling, Measuring, and Trading on Alpha Decay** (arXiv 2512.11913): 팩터별로 크라우딩에 대한 민감도가 다르다는 걸 정량화 — 특이(idiosyncratic)하거나 구조가 복잡한 신호일수록 덜 붕괴한다는 시사점.
- **RobAlpha**: 크라우딩 리스크를 정량화하는 adversarial RL 프레임워크(Crowding Simulator + Market Antagonist) — 신호를 만들 때부터 "얼마나 쉽게 복제/경쟁될까"를 스코어링하자는 제안.

**시사점**: 프로젝트가 관측한 "레짐 조건부 엔트리는 VAL에서만 통하고 OOS에서 죽는다"는 패턴은, 크립토 파생상품이라는 매우 유동적이고 경쟁이 심한 시장에서 알파 반감기가 짧아졌다는 최신 이론과 정합적이다. 즉 "우리가 못 찾아서"가 아니라 "찾아도 반감기가 짧아서 프로젝트의 VAL/OOS 분리 구간(3~4개월) 동안 이미 죽어있을 확률이 높다"는 설명이 가능하다.

### 2.3 LOB/미시구조 엔트리 신호 — 있긴 한데 못 쓴다

- **Deep Order Flow Imbalance: Extracting Alpha at Multiple Horizons from the LOB** (Kolm, Turiel, Westray, SSRN 3900141): 오더플로우로 학습한 모델이 오더북 자체로 학습한 모델보다 우월. 다만 "유효 예측 구간은 평균 가격변화 약 2틱 정도"로 극히 짧다.
- **Deep Limit Order Book Forecasting: A Microstructural Guide** (Quant Finance 2025): "높은 통계적 예측력이 실행가능한 거래신호로 반드시 연결되지는 않는다"는 걸 정면으로 지적. 종목별 미시구조 특성이 딥러닝 방법의 효용을 좌우.
- **Predicting Adverse Selection in High-Frequency Cryptocurrency Markets Using Gradient Boosting** (Rajendran & Singaravelu, 2026): VPIN 기반 정보거래(informed trading) 예측에 그래디언트부스팅 사용 — 이 프로젝트가 이미 h48qual/GBDT 우위를 확인한 것과 같은 모델 패밀리.

**시사점**: LOB 알파는 존재가 보고되지만 (a) 호가창 원본 데이터가 필요하고(이 프로젝트는 OHLCV+파생 지표 기반으로 보임), (b) 유효 구간이 몇 틱 수준이라 5분봉 스케일에서는 거의 소멸해 있을 가능성이 높다. 새로 파고들 가치보다 "우리 데이터 해상도에서는 원리적으로 안 보일 신호"일 확률을 먼저 점검하는 게 낫다.

### 2.4 TSFM/딥러닝 vs GBDT — 프로젝트 결론의 외부 재확인

- **Pretrained Time-Series Foundation Models for Financial Return Forecasting** (arXiv 2606.27100): TSFM(TimesFM, Chronos, MOIRAI 등)은 **일별 초과수익률 제로샷 예측에서 CatBoost/LightGBM 앙상블에 열세**. 금융 사전학습을 해도 필요한 데이터 연수만 줄어들 뿐, 범용 사전학습만으로는 부족.
- **Rethinking Evaluation in the Era of TSFM: (Un)known Information Leakage** (arXiv 2510.13654): 15개 주요 TSFM의 사전학습·테스트셋 계보를 추적해 **훈련-테스트 오염(leakage) 문제**를 지적 — TSFM 벤치마크 자체가 부풀려져 있을 수 있다는 경고.
- **Algorithmic Complexity vs. Market Efficiency (Wavelet-Transformer)**: 2.1M 파라미터 웨이블릿-트랜스포머·ARIMA·XGBoost·LSTM·바닐라 트랜스포머 전부 **1일 호라이즌에서 단순 지속성(persistence) 베이스라인을 유의하게 이기지 못함**.

**시사점**: 프로젝트가 이미 6개 아키텍처, Kronos, JEPA, PatchTST, TSFM류를 다 시도해서 전멸시킨 건 지엽적 실수가 아니라 **2026년 현재 문헌 전반의 일치된 결론과 같다.** 새 아키텍처를 더 찾는 것보다 데이터(피처) 자체를 바꾸는 쪽이 문헌상으로도 더 근거 있는 다음 수다.

### 2.5 메타레이블링 비대칭성 — h48qual이 "왜" 작동하는지의 이론적 근거

Lopez de Prado의 메타레이블링 프레임워크 문헌은 이 프로젝트가 경험적으로 발견한 것(`project-eth-h48qual-contribution-is-blocking-not-earning-20260808`)에 정확한 이름을 붙여준다: **1차 모델(방향)과 2차 모델(베팅 여부/크기)을 분리하면, 2차 모델은 "이미 생성된 신호 중 뭘 거를지"라는 훨씬 제한된 문제만 풀면 되므로 과적합 위험이 구조적으로 작다.** 반대로 이 프로젝트의 3-way TP-first 메타라벨(P(TP)-P(SL) 직접 예측, `project-btc-3way-tpfirst-label-closed-20260804`)이 0/24로 실패한 이유도 설명된다 — 그건 "필터링"이 아니라 노이즈가 심한 라벨의 **경로 결과 확률을 직접 예측**하려 한 것이라, 2차 모델의 안전한 문제 형태가 아니라 1차 모델과 똑같이 어려운 문제였다.

**교훈**: 앞으로 "메타모델"을 만들 때는 반드시 "이미 나온 신호를 거른다"는 형태를 유지해야지, "결과 확률을 재추정한다"는 형태로 슬쩍 바뀌면 h48qual이 아니라 TP-first가 재현된다.

### 2.6 아직 안 건드린 축: 세션/유동성 윈도우 효과

앞서 2.1의 MNQ falsification 연구에서 유일한 생존 신호 2개가 세션 기반(RTH 컨플루언스, 런던 세션)이었다는 점, 그리고 이 프로젝트 자체가 "레짐은 변동성/노출 정보는 담는다"고 결론 낸 점을 겹쳐보면: **세션/유동성 윈도우는 가격패턴이 아니라 참여율(participation) 프록시라서, 프로젝트가 이미 죽은 걸 확인한 "방향-페이오프 관계"가 아니라 "변동성/유동성 상태"라는, 살아있는 것으로 확인된 정보 종류에 속할 가능성이 있다.** 크립토는 24/7이지만:
- 아시아/유럽/미국 세션 오버랩(특히 13:00-17:00 UTC)에 유동성이 집중되고, 주말 거래량은 평일 대비 20-40% 낮다는 게 여러 소스에서 일관되게 보고된다.
- 주말은 유동성 청산 캐스케이드 리스크가 커지고, CME 갭 같은 전통시장發 왜곡도 얽힌다.
이건 학술 논문이 아니라 업계 리포트 수준 근거라 신뢰도는 낮지만, **레짐 탐지기를 "세션+요일" 같은 결정론적 캘린더 변수로 보강해서 사이징(엔트리 아님) 레버로 테스트**해보는 건 이미 검증된 "레짐→사이징" 성공 패턴에 정확히 들어맞는 저비용 실험이다. (h48qual 필터처럼, 방향을 예측하지 말고 "이 시간대엔 슬롯을 줄인다"는 형태로만 걸 것.)

### 2.7 서명(Signature)-MMD 경로 클러스터링 레짐 탐지 — 프로젝트가 이미 "미검증"으로 표시한 후보

오늘 자 메모리(`project-regime-3class-literature-validation-20260809`)에 "미검증: signature-MMD path-clustering (arXiv:2306.15835)"라고 이미 적혀 있던 항목을 재확인했다: **Horvath & Issa, "Non-parametric online market regime detection and regime clustering for multidimensional and path-dependent data structures"** — rough-path signature + MMD 기반 비모수 온라인 체인지포인트 탐지. 코드가 공개돼 있다(`github.com/issaz/signature-regime-detection`). 2025-2026 후속 연구는 이 기법이 변동성 예측(HAR 모델), 위기 국면 탐지에 잘 맞는다고 보고한다. **이 프로젝트 맥락에서 중요한 건**: 이 방법은 방향을 예측하지 않고 "체제가 바뀌었는가/변동성 국면이 뭔가"만 비모수적으로 감지하므로, Part 1에서 확인한 "레짐이 담을 수 있는 정보(변동성/노출)"의 범위 안에 정확히 들어간다 — JM/HMM류보다 경로의존성을 더 잘 잡는다는 이론적 이점이 있으니, **사이징 오버레이용 레짐 탐지기 후보로 시도할 가치가 있다** (엔트리 신호로 쓰면 다시 실패할 것으로 예상됨).

### 2.8 청산(Exit) 최신 연구

- **Optimal Stop-Loss and Take-Profit Parameterization for Autonomous Trading Agent Swarm** (arXiv 2604.27150, 2026-04): 이 논문의 문제의식 자체가 인상적이다 — *"자율 크립토 트레이딩 시스템은 대개 엔트리를 찾는 데 설계 노력을 쏟고, 청산은 체계적으로 검증되지 않은 고정 규칙에 맡긴다."* 900개 이상의 과거 트레이드를 여러 청산 정책으로 재생(replay)한 결과: **더 타이트한 손절, 더 이른 익절, 더 촘촘한 트레일링이 위험조정수익을 유의하게 개선**했다. 다만 평가 시 "가장 최근 구간이 이례적인(전쟁 영향) 시장이라 결과를 왜곡시켰다"는 걸 스스로 지적 — 이 프로젝트의 "OOS 구간을 신중히 골라야 한다"는 규율과 같은 함정을 겪었다.
  - **주의**: 이 프로젝트는 이미 트레일링 스탑을 테스트해서 "분산/MDD 레버일 뿐 PnL 레버는 아니다, ETH/BTC 라이브 모델 둘 다에서 0/6"이라는 결론을 내렸다(`project-trailing-stop-risk-lever-keep-alive-20260807`). 저 논문의 "타이트한 손절+이른 익절"도 그대로 재시도하면 같은 결과가 나올 가능성이 크다 — **다만 저 논문처럼 900개 트레이드 규모의 재생 스윕을 직접 한 적은 없어 보이므로, 기존 발견을 재확인하는 저비용 크로스체크로는 가치가 있다.**
- **생존분석/해저드 모델 기반 청산**: DeepSurv/DeepHit류가 "트레이드가 지금 이 순간 청산될 위험(hazard)"을 이산시간으로 모델링하는 방식이 리미트오더 체결시간 예측(KANFormer, 2025)에 쓰이고 있다. 이 프로젝트의 21라운드 exit 실험(trailing/time-decay/reversal-classifier 등)은 전부 "지금 청산할지 말지"를 직접 분류/회귀했지만, **해저드-레이트로 모델링한 적은 없어 보인다** — 이론적으로는 다른 각도지만, exit 축 자체가 이미 21/21 실패한 걸 감안하면 기대치는 낮게 잡아야 한다.
- **Optimal stopping 일반론**: "청산 로직은 자율매매 시스템에서 가장 근거가 부실한 컴포넌트인 경우가 많다"는 지적이 여러 2025-2026 논문에서 반복된다 — 이 프로젝트는 이미 이 부분에 21+ 실험을 투입했으므로 상대적으로 앞서 있는 편.

### 2.9 대체 데이터 (프로젝트의 2026-10 계획 축) — 현재 증거는 약함

- LLM/뉴스 감성: TIE 감성신호가 2022-2025 백테스트 7일 구간의 68%에서 알파를 냈다는 보고, CryptoQuant의 48-72시간 선행 감성예측(정확도 61%) 베타 등 — 전부 **업계 리포트 수준, 동료심사 안 됨, 특정 벤더 자체 평가**라 신뢰도가 낮다. 학술 서베이(`The New Quant`, arXiv 2510.05533)는 LLM 트레이딩 신호 전반이 아직 초기 단계임을 인정한다.
- 온체인(CoinMetrics 등)은 이미 이 프로젝트가 0/9로 닫았고, 펀딩/청산 캐스케이드도 약한 상관관계 또는 데이터 부족으로 닫힌 상태 — 대체 데이터 축에서 "새로 시도해볼 것"은 사실상 뉴스/소셜 감성 정도로 좁혀져 있다.

**시사점**: 10월로 미룬 대체 데이터 축에 지금 시간을 더 쓰는 것보다, 현재 우선순위(리스크 레이어)에 집중하는 프로젝트의 계획이 문헌상으로도 합리적이다. 감성 데이터는 후보로 유지하되 기대치를 낮게 잡을 것.

---

## Part 3. 실행 제안 (기존 CLOSED 라인과 교차 확인 완료)

1. **[방법론, 최우선] Falsification audit을 사후 검정에서 사전 게이트로 승격.** 새 엔트리 신호 후보에 VAL 예산을 쓰기 전에, 동일 파이프라인을 (a) 순수 랜덤워크로 합성한 가짜 시장, (b) 라벨을 셔플한 미시구조 플라시보에 먼저 돌려서 유의한 결과가 나오면 그 파이프라인 자체를 폐기. 이미 하던 순열검정(사후)을 표준 사전 게이트로 문서화하면 됨. 근거: arXiv 2604.15531.
2. **[사이징 축, 중간 우선순위] Signature-MMD 경로 클러스터링을 레짐 탐지기 교체 후보로.** 방향이 아니라 변동성/체제전환 탐지 목적으로만, 기존 JM/HMM 사이징 오버레이와 나란히 놓고 effect-size gate(t≥2.0, permutation ≥0.90, risk-channel 검증)를 통과하는지만 확인. 코드 공개돼 있어 구현 비용 낮음.
3. **[사이징 축, 저비용] 세션/요일 캘린더 변수를 사이징 필터로 추가.** 방향 예측 아님 — "아시아/주말처럼 유동성이 얕은 구간엔 슬롯/배율을 줄인다"는 h48qual과 같은 차단형 룰로만 테스트. MNQ falsification 연구가 유일하게 살아남은 신호 종류라는 방향적 근거는 있지만 크립토 24/7 구조에서 그대로 전이될지는 검증 전.
4. **[Exit, 낮은 기대치] 900+ 트레이드 스윕 재생을 exit 스위트에 추가.** 기존 21라운드가 개별 규칙 단위 테스트였다면, arXiv 2604.27150처럼 전체 그리드를 원장에 대해 일괄 재생해서 "타이트한 손절+이른 익절"이 이 프로젝트 데이터에서도 재현되는지 저비용으로 크로스체크. 트레일링 스탑이 이미 0/6으로 닫혀 있으므로 큰 기대는 금물.
5. **[하지 말 것] 새 아키텍처/새 라벨 기하학으로 방향 예측을 다시 시도하는 것.** 2025-2026 문헌(TSFM<GBDT, 웨이블릿-트랜스포머도 persistence 못 이김, AI발 알파 반감기 18개월)이 프로젝트의 자체 전멸 기록과 정확히 일치한다. 프로젝트의 2026-07-30 "방향 예측 접기" 결정은 외부 근거로도 뒷받침된다.
6. **[대체 데이터, 계획대로 10월] LLM/뉴스 감성**은 후보로 유지하되, 업계 리포트 수준 근거뿐이므로 학술적으로 검증된 방법론(falsification audit 포함)을 반드시 통과시킬 것.

---

## Part 4. 오늘 밤 실증 후속 (2026-08-09 03:00-10:00, 17개 아이디어 전수 테스트)

사용자가 Part 3의 제안들을 실제로 코드로 돌려보라고 요청해서, `core/selection_stats.py`에 구현한 `falsification_audit` 게이트를 실전에 적용하며 밤새 17개 아이디어를 하나씩 검증했다. 전체 스크립트는 `scripts/research_*_20260809.py`, 개별 결과는 메모리 `project-eth-*-20260809.md` 파일들, 인덱스는 `project-eth-skip-filter-line-closed-20260809.md`.

### 4.1 결과 요약: 17/17 네거티브 (1개는 "찾았다"에서 철회)

| # | 아이디어 | 결과 |
|---|---|---|
| 1 | h48qual 확률에 conformal(APS) abstention | 네거티브 (t=0.27, p=0.79) |
| 2 | Path-signature (시간×수익률 Lévy area) | 네거티브, naive momentum보다 못함 |
| 3 | Path-signature (가격×거래량) | 게이트가 정확히 차단 |
| 4 | 144봉 naive momentum | **철회** — 라벨 구조 착시였음 (아래 4.2) |
| 5 | 세션/요일 캘린더 | 게이트 차단 |
| 6 | Order-flow imbalance (taker buy/sell) | 노이즈보다 나쁨 (0th percentile) |
| 7 | BTC→ETH lead-lag 모멘텀 | #6과 동일 패턴 |
| 8-9 | 이진 스킵필터 (모멘텀/OFI, 변동성/ATR) | 랜덤 스킵보다 못함 |
| 10 | 결합 피처 + 스킵비율 스윕 | 스킵할수록 단조 악화 |
| 11 | Hawkes 자기여기과정 (jump clustering) | #10과 동일 패턴 |
| 12 | SOL 전이 검증 (결합 필터) | ETH와 동일 네거티브 |
| 13 | 진짜 2-모델 우선순위 라우터 (실제 라이브 코드 확인 후 복제) | h48qual 실제 임계값에서 무승부, 느슨하면 네거티브 |
| 14 | Quantile regression skew | 새 패러다임인데도 네거티브 |
| 15 | Kaufman Efficiency Ratio 스킵필터 | 동일 패턴 |
| 16 | 44피처 kitchen-sink (전체 결합) | 5% 스킵에서도 baseline 미달 — 개별 무정보 피처들을 합쳐도 무정보 |
| 17 | Cross-sectional relative strength (ETH vs BTC+SOL 바스켓) | 동일 패턴, 절대 lead-lag(#7)와 다른 구성인데도 네거티브 |

### 4.2 밤중 발견한 핵심 방법론 버그 2건

**(a) Proxy 백테스트의 기준선은 반드시 always_long/always_short여야 하고, 절대 0이면 안 된다.** ETH·BTC 라벨의 tp_move:sl_move 비율이 정확히 2.083:1로 고정돼 있는데 실제 outcome 분포는 거의 1:1이라, 방향에 상관없이 일관되게만 베팅하면 이 프록시 구조에서 공짜 edge가 나온다. always_short는 21개월 중 21개월 전부 양수였고 — 이 기간 ETH 가격이 49% 폭락했는데도 — 이게 아이디어 #4의 "발견"을 철회하게 만든 결정적 증거다. 상세: `project-baseline-must-be-always-long-short-not-zero-20260809`.

**(b) 3-way 재분류는 "나쁜 봉 건너뛰기"(필터링, 통하는 패턴)와 "반대 방향 재선택"(예측, 안 통하는 패턴)을 뒤섞어서 조금만 틀려도 손해가 눈덩이처럼 커진다.** 아이디어 #6-7이 이걸로 실패한 뒤 이진 스킵필터(#8-15)로 전환했고, 실제로 필터 프레임이 더 공정하다는 게 확인됐다(그래도 결과는 여전히 네거티브였음).

### 4.3 아이디어 #13에서 나온 재해석 — 가장 가치 있는 발견

`trading_bot_modules/omega4_6_1_live.py`의 실제 코드를 직접 확인: `PRIORITY = ("h48qual", "zig075")`, h48qual quality_threshold=0.50, zig075=0.75. 즉 h48qual이 **더 낮은 문턱으로 먼저** 슬롯을 가져가는 우선순위 라우터다. 이건 h48qual의 "블로킹" 기여가 "봉 품질을 랭킹하는" 단일모델 메커니즘이 아니라 **다른(더 나쁜) 모델로부터 슬롯을 빼앗는 멀티모델 경쟁 메커니즘**이라는 뜻이고, 그래서 #8-12의 "제네릭 피처로 h48qual 재현" 시도들이 애초에 다른 메커니즘을 테스트하고 있었다는 걸 의미한다. 실제 라이브 코드와 대조한 2-모델 복제(#13)도 h48qual의 실제 임계값(0.50)에서는 거의 무승부였을 뿐 — h48qual의 진짜 edge는 몇 시간 만에 뚝딱 만든 제네릭 피처가 아니라 그 자체의 실제 피처/라벨 엔지니어링에서 나온다는 뜻이다.

### 4.4 종합 결론

17개 아이디어, 3가지 모델링 패러다임(3-way 분류/이진 확률/quantile regression), 7개 독립적 피처군(모멘텀, OFI, 변동성, Hawkes, Kaufman ER, cross-sectional relative strength, 그리고 이들을 전부 합친 44피처 kitchen-sink), 2개 자산(ETH, SOL), 실제 라이브 코드 대조 복제까지 — 전부 동일한 결론으로 수렴한다: **병목은 아직 안 써본 아키텍처나 피처가 아니라, 이 시장/시간프레임/데이터 해상도에서 뽑아낼 수 있는 정보량 자체다.** 개별 피처군을 다 합쳐도(#16) 여전히 무정보였다는 게 특히 결정적인데, 이는 "각 피처가 조금씩은 담고 있는데 개별적으로는 못 봤을 뿐"이라는 가설을 통계적으로 배제한다.

밤 전체를 관통하는 가장 깔끔한 숫자 하나: kitchen-sink 모델(#16)의 승/패 이진타겟 AUC가 **TRAIN(인샘플) 0.9564 vs DEV 0.5166 / VAL 0.5170**(n=17,855 / 35,136, 큰 샘플이라 노이즈 아님)이다. 모델이 훈련 데이터는 거의 완벽하게 외워버리면서(AUC 0.956) 완전히 같은 피처군으로 홀드아웃에서는 동전 던지기(0.50) 바로 위(0.517)밖에 못 하는 이 격차가, 오늘 밤 17개 아이디어 전부의 실패와 이 프로젝트 전체 역사의 "VAL 좋음→OOS 나쁨" 반복 패턴을 한 줄로 요약한다.

마지막으로 "혹시 과적합이 작은 진짜 신호를 가리고 있는 건 아닐까"를 확인했다: 훨씬 강하게 정규화한 두 모델(TRAIN AUC를 0.956→0.616, →0.565까지 낮춤)도 DEV/VAL AUC는 여전히 0.50~0.53 사이에서 요동칠 뿐, 개선되지 않았다. 즉 과적합을 억제해도 숨겨진 실력이 드러나지 않는다 — 애초에 없기 때문이다.

Part 1-3의 분석과 정확히 일치하며, 오늘 밤 작업은 그 결론을 독립적인 새 증거로 재확인한 것이다. 제안 5번("새 아키텍처로 방향 예측 재시도 금지")이 옳았다는 걸 실제로 17번 반복 검증한 셈이다.

### 4.5 남은 시간 계획

여기까지가 이 데이터 환경(로컬 ETH/BTC/SOL 5분봉 OHLCV + 파생 지표)에서 저비용으로 테스트 가능한 "일반 피처 기반 엔트리/필터 아이디어" 공간을 사실상 전수 조사한 결과다. 남은 시간은 (1) 정말 새로운 각도가 떠오르면 추가 테스트, (2) 최종 종합 리포트 정리에 쓴다. 사용자가 명시한 "10시까지, 좋아 보여도 멈추지 않기" 원칙은 계속 지킨다 — 억지로 사소한 변형을 더 만들어 숫자만 채우지 않는다는 뜻이다.

이 arc 전체는 `docs/model_contracts/research_line_registry.json`에 `eth_overnight_generic_feature_entry_filter_20260809`로 정식 등록해뒀다 — 이 프로젝트의 공식 "닫힌 연구 라인" 대장이라, 앞으로 이 정확한 17개 아이디어를 재시도하기 전에 반드시 참고하게 된다.

---

## Part 5. "딥러닝으로 이 턱을 넘긴 최신 사례가 있나?" 실증 후속 (2026-08-10)

사용자가 "최근 논문 중에 이 train/OOS 격차를 넘긴 딥러닝 사례가 있냐"고 물었고, 문헌 검색으로 세 가지 후보 메커니즘을 찾았다: (A) information-driven bars(CUSUM 이벤트 샘플링, 2025 크립토 triple-barrier DL 논문), (B) cross-sectional breadth(A주 5,000+ 종목 풀링, 2026 Financial Innovation 논문에서 LSTM/Transformer가 트리모델을 이긴 유일한 케이스), (C) Chronos 같은 오픈 웨이트 파운데이션 모델 zero-shot(2026 TSFM-in-Finance 논문에서 파인튜닝 시 "진짜" 통계적 개선을 보인 유일한 예외 케이스). 셋 다 실제로 ETH 데이터에 구현해서 테스트했다.

**(A) CUSUM 정보기반 이벤트 샘플링 + 대칭 배리어 + 진짜 신경망(MLP)**: 대칭 배리어로 라벨 편향을 원천 차단(P(long)=0.49, P(short)=0.51)하고, 균일 5분봉 대신 진짜 "이벤트"만 샘플링(22,754개 봉 중 21,754개 이벤트, 평균 10.3봉당 1개)했다. MLP는 GBDT보다 훨씬 덜 과적합했지만(TRAIN AUC 0.55 vs GBDT의 0.94) DEV/VAL/OOS는 똑같이 0.49~0.51에 머물렀고, VAL 트레이딩 전략 falsification 체크는 무작위 셔플 대조군보다도 나빴다(35th percentile). **결론: 겹침으로 인한 유효표본수 뻥튀기도, 모델 과적합도 이 프로젝트의 격차를 설명하는 핵심 원인이 아니다 — ETH 5분봉엔 애초에 안정적인 방향 정보가 없다.**

**(B) 60코인 cross-sectional 풀링(A주 논문의 "폭으로 넘는다" 메커니즘 재현)**: `data/panel/features` + `data/panel/tripbarrier`의 60개 USDT 무기한선물(1,628만 행)을 시점별로 cross-sectional z-score해서 풀링, 신경망을 학습시켜 ETH의 홀드아웃 성과를 ETH 전용 모델(대조군, 동일 아키텍처/피처)과 비교했다. 결과: **풀링 모델이 3개 구간 중 2개(DEV, OOS)에서 ETH 전용 모델보다 오히려 못했다** — 폭이 도움되기는커녕 해가 됨. A주 논문의 우위는 "5,000개 기업이 서로 다른 펀더멘털을 가진 진짜 이질적 정보원"이라는 전제에서 나오는데, 크립토 알트코인 60개는 대부분 BTC/ETH 베타에 종속돼 있어 행(row)만 늘고 정보는 안 늘어난다. 이건 이미 실패했던 BTC Rho1 60종목 패널-트랜스포머(랭킹 헤드가 랜덤 바닥, 6/6 백테스트 네거티브)를 완전히 다른 아키텍처로 독립 재확인한 셈이다.

**(C) Chronos(오픈 웨이트 TSFM) zero-shot**: 256봉 컨텍스트로 48봉 앞 확률적 예측(30 샘플)을 뽑아 median-direction/quantile-skew 두 신호를 만들어 always_long/short 기준선과 비교했다. **median-direction 신호는 VAL(t=-2.86, p=0.004)과 OOS(t=-3.14, p=0.002) 둘 다에서 기준선보다 통계적으로 유의하게 나빴다.** zero-shot이 가능성을 전혀 못 보였으므로(사전에 정한 규칙대로) 파인튜닝은 진행하지 않았다 — 문헌상 최선의 케이스(Chronos-large, 주식, 파인튜닝)조차 경제적 가치로 안 이어졌다는 걸 감안하면 합리적인 중단이다.

**종합**: 문헌에서 찾은 가장 신뢰할 만한 3가지 "DL이 이 턱을 넘는" 메커니즘을 실제로 이 프로젝트 데이터/평가 기준으로 테스트했고 전부 실패했다. 셋 다 서로 다른 방식으로 실패했다는 게 중요하다 — (A)는 샘플링/과적합 가설을 배제하고, (B)는 크립토 알트코인이 주식과 달리 횡단면 이질성이 없다는 걸 보여주고, (C)는 최신 파운데이션 모델도 이 특정 자산/호라이즌에서는 기준선보다 못하다는 걸 보여준다. 스크립트: `scripts/research_cusum_information_bars_dl_eth_20260810.py`, `scripts/research_cross_sectional_panel_dl_60coin_20260810.py`, `scripts/research_chronos_tsfm_eth_20260810.py`.

---

## 부록: 참고 문헌 (2025-2026 중심)

- Spurious Predictability in Financial Machine Learning — https://arxiv.org/html/2604.15531
- Interpretable Hypothesis-Driven Trading: Rigorous Walk-Forward Validation Framework — https://arxiv.org/html/2512.12924v1
- Structural Limits of OHLCV-Based Intraday Signals in MNQ Futures — https://arxiv.org/abs/2605.04004
- AI-Driven Alpha Decay: Algorithmic Homogenization, Reflexive Signal Erosion — https://arxiv.org/abs/2605.23905
- Not All Factors Crowd Equally: Modeling, Measuring, and Trading on Alpha Decay — https://arxiv.org/abs/2512.11913
- Deep Order Flow Imbalance: Extracting Alpha at Multiple Horizons from the LOB (Kolm, Turiel, Westray) — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3900141
- Deep limit order book forecasting: a microstructural guide — https://www.tandfonline.com/doi/full/10.1080/14697688.2025.2522911
- Pretrained Time-Series Foundation Models for Financial Return Forecasting — https://arxiv.org/pdf/2606.27100
- Rethinking Evaluation in the Era of Time Series Foundation Models (leakage) — https://arxiv.org/pdf/2510.13654
- Algorithmic Complexity vs. Market Efficiency: Wavelet–Transformer Architectures — https://doi.org/10.3390/a19020101
- Optimal Stop-Loss and Take-Profit Parameterization for Autonomous Trading Agent Swarm — https://arxiv.org/abs/2604.27150
- Predicting Adverse Selection in High-Frequency Cryptocurrency Markets Using Gradient Boosting — https://papers.ssrn.com/sol3/Delivery.cfm/6344338.pdf
- Non-parametric online market regime detection and regime clustering (signature-MMD) — https://arxiv.org/pdf/2306.15835 ; code: https://github.com/issaz/signature-regime-detection
- The Deflated Sharpe Ratio (Bailey & Lopez de Prado) — https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf
- Meta-Labeling (Lopez de Prado framework, secondary-model overfitting reduction) — https://en.wikipedia.org/wiki/Meta-Labeling
- The New Quant: A Survey of LLMs in Financial Prediction and Trading — https://arxiv.org/html/2510.05533v1

## 내부 근거 (메모리/프로젝트 문서)

- `docs/pipeline_integrity_and_research_redesign_20260730.md` — 방향예측 셸빙 결정 원문
- `docs/model_contracts/research_line_registry.json` — 닫힌 엔트리/청산 라인 전체 목록
- `project-btc-regime-arc-20260808-summary`, `project-btc-regime-expert-lines-all-closed-20260808`, `project-eth-regime-differential-dead-20260808` — 레짐 조건부 엔트리 전멸 + ΔAUC OOS 붕괴 수치
- `project-sol-dl-rl-architecture-survey-20260807`, `project-eth-1h-patchtst-new-architecture-failed-20260731`, `project-btc-deepfeat-jepa-unified-panel-closed-20260804` — DL/RL 아키텍처 전멸
- `project-btc-label-design-arc-and-next-session-queue-20260808`, `project-btc-oracle-label-selection-protocol-20260806` — 라벨 천장/오라클 갭
- `project-eth-h48qual-contribution-is-blocking-not-earning-20260808` — 메타레이블링 비대칭성의 실증
- `project-btc-regime-sizing-effect-size-downgrade-20260808`, `project-trailing-stop-risk-lever-keep-alive-20260807` — 리스크 축도 대부분 실패한다는 균형 근거
- `project-regime-3class-literature-validation-20260809` — signature-MMD가 이미 "미검증"으로 태깅된 근거
