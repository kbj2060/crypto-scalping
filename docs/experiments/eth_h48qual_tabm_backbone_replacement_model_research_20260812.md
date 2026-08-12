# ETH h48qual — TabM 백본 대체 모델 후보 문헌 리서치 (2026-08-12)

## 목적과 범위

사용자 지시(2026-08-12): "피쳐(FINAL12)·데이터·데이터 구간은 유지한 채, TabM을 대체할 다른 모델을
최신 논문 기준으로 리서치하라." **문헌 리서치만 — 학습/구현/코드 변경 없음.** 계약 문서 미해결
이슈 2(백본 A/B 보류)를 위반하지 않는다(연구 결과가 실제 학습 착수를 정당화하는지는 별도 결정).

- 방법: Model Architect 페르소나 단독 dispatch(Sonnet, 웹서치 32회) — 서브 프로젝트 표준 절차.
- 리드 세션 검증: (1) 핵심 인용(Drift-Resilient TabPFN) 저자·출처를 독립 웹서치로 재확인 —
  에이전트 보고서의 "den Breejen et al." 표기는 **오기**이며 실제 저자는 **Helli, Schnurr,
  Hollmann, Müller, Hutter (NeurIPS 2024, arXiv 2411.10634)**. 본 문서에는 정정본을 반영.
  (2) 권장 1순위(GBDT)의 전제 조건 — "이 레포에서 이미 시도됐는가"를 레포 실사로 직접 확인
  (아래 "레포 실사" 절). 그 외 벤치마크 수치·연도는 에이전트 웹서치 결과이며 출처 링크를 병기.

## 고정 제약 (사용자 지정, 변경 불가)

- 입력: FINAL12 — 12개 수치형 tabular 피쳐, bar 단위 단일 row (시퀀스 아님)
- 라벨: `zigzag_action` 3-class(CASH/LONG/SHORT), 보조 `h48_conservative` 3-class
- 데이터: TRAIN 183,936행(2024-01~2025-09) / VAL 26,496행(2025-10~12) / OOS 16,897행(2026-01~02)
- 모델(백본)만 대체 대상. 피쳐·라벨·구간 변경 제안은 범위 밖.

## 요약

TabM은 TabArena(2025 living benchmark)와 TALENT(약 300개 데이터셋) 양쪽에서 **플레인 DL 모델 중
최상위권**으로 확인된다 — 즉 "같은 계열 내 더 좋은 모델로 교체"의 headroom은 작다. TabArena 논문
스스로 "TabM·LightGBM·RealMLP 상위 3개 개별 모델 모두 post-hoc 앙상블 없는 CatBoost보다 못하다"고
쓸 정도로 최상위권 간 격차는 좁다. 실질적 여지는 두 곳뿐이다:

1. **행/피쳐 비율**: 183,936행 ÷ 12피쳐 ≈ 15,411은 문헌상(NeurIPS 2023 메타분석) GBDT가 구조적으로
   유리한 극단적 "저피쳐·대량행" 구간이다.
2. **인덕티브 바이어스 자체가 다른 계열**: 이 프로젝트의 확정된 실패 양상(N=5 시드에서 40칸 중
   38칸 always-short 패배)은 "용량 부족"이 아니라 시간적 분포 이동/신호 부재 문제인데, 이를
   정면으로 모델링하는 Drift-Resilient TabPFN 같은 계열이 존재한다.

단, 위 벤치마크들은 전부 **IID 랜덤 스플릿 기준**이다. TabReD(2024)가 "시간 기반 스플릿은 모델
순위를 크게 바꾼다"고 명시했으므로, 벤치마크 순위를 이 프로젝트의 walk-forward 평가에 그대로
적용할 수 없다.

## 후보 테이블

| 모델 | 출처 | TabM과 다른 prior | 184k×12 적용성 | 검증 비용 | 기대치 |
|---|---|---|---|---|---|
| **CatBoost / LightGBM** | Prokhorenkova+ NeurIPS 2018 / Ke+ NeurIPS 2017; Grinsztajn+ NeurIPS 2022 | 축-정렬 비미분 분할 트리 vs 연속 가중합 MLP 앙상블 | 매우 좋음 — 12피쳐·184k행은 오히려 GBDT 유리 구간 | **낮음** | 중간 — 이 정확한 계약(FINAL12→zigzag_action)으로는 미시도(레포 실사 절 참고), 단 유사 선례가 부정적 |
| **Drift-Resilient TabPFN** | **Helli, Schnurr, Hollmann, Müller, Hutter, NeurIPS 2024** (arXiv 2411.10634, 리드 세션이 저자 재확인) | 시간에 따라 변하는 SCM prior 위 베이지안 in-context 추론 — 정적 데이터 가정 자체를 버림 | 불확실 — TabPFN v1 기반이라 규모 제한 작을 가능성, 컨텍스트 서브샘플링 필요 (행수 상한 미확인) | 중간~높음 | 중간~높음 — 이 프로젝트의 실제 실패 양상(train/eval 분포 이동)과 문제 정의가 가장 정확히 일치하는 유일한 후보 |
| **TabICL** | Qu+, ICML 2025 (arXiv 2502.05564, INRIA) | 사전학습 트랜스포머 in-context learning (컬럼→로우 어텐션) | 좋음 — 500K행까지 실용 처리 명시, 오픈소스 | 중간 | 중간 — 10K행 초과 56개 데이터셋에서 TabPFNv2·CatBoost 능가 (IID 기준) |
| **ModernNCA** | Ye+, ICLR 2025 (arXiv 2407.03257) | 국소 이웃 유사도 기반 소프트 최근접이웃 분류 | 좋음 — 경량·고속, TALENT 300개 데이터셋 검증 | 낮음~중간 | 중간 |
| **TabPFN-2.5 / 3** | Hollmann+ Nature 2025(v2); Prior Labs 기술보고서(2.5: 2025-11, 3: 2026-05) | 합성 prior 사전학습 + gradient-free in-context 추론 | v2/2.5는 10K행 상한(부적합), **TabPFN-3은 1M행×200피쳐로 규격 내** | 낮음~중간 | 중간 — 단 **라이선스 리스크**: 상업/프로덕션 조건이 검색 결과 간 상충, 라이브 편입 전 법무 확인 필수 |
| **TabR** | Gorishniy+, ICLR 2024 (TabM과 동일 저자그룹) | retrieval-augmented (어텐션 이웃 검색) | 좋음 | 중간 | 중간 — 단 TabM 논문 자체가 "retrieval 복잡도는 결과적으로 이득 없음"이라고 후속 결론 |
| **xRFM** | Beaglehole+, 2025 (arXiv 2508.10053, ICLR 2026 accepted) | 커널 기반 Recursive Feature Machine + 트리 분기 | 중간 — TabArena-Lite 파레토 상단, 184k 실적 미확인 | 중간~높음 | 중간 — 신생 도구 리스크 |
| GRANDE | Marton+, ICLR 2024 | 미분가능 축-정렬 트리 | 중간 — 이진분류 19개 데이터셋만 검증 | 중간 | 낮음 — 순수 GBDT가 같은 prior를 더 싸게 제공 |
| Mitra | Amazon Science 2025 (AutoGluon 1.4) | 혼합 synthetic prior ICL | 낮음 — 강점 구간이 5천행 미만으로 정반대 | 낮음 | 낮음 |
| RealMLP | Holzmüller+, NeurIPS 2024 | **없음 — 동일 MLP 계열** | 좋음 | 낮음 | 낮음 — "다른 prior" 기준 탈락 |

## 후보별 상세

### 1. CatBoost / LightGBM (GBDT)

축-정렬 임계값 분할의 순차 부스팅. Grinsztajn et al.(NeurIPS 2022)이 중형 정형 데이터에서 트리
우위를 확립했고, 2024~2025 반박 문헌은 벤치마크 구성 결함을 지적했을 뿐 우위 자체를 뒤집지
못했다. "When Do Neural Nets Outperform Boosted Trees"(NeurIPS 2023, 176개 데이터셋)는 행수/피쳐수
비율이 클수록 GBDT 유리라는 메타피쳐 근거를 제시 — 이 프로젝트의 15,411은 전형적 GBDT 유리
구간이다. 적용: FINAL12 그대로, `zigzag_action` 3-class. 리스크: 최신 벤치마크 흐름은 "GBDT vs DL
논쟁 자체가 과장" 쪽이라, 신호가 정말 없으면 GBDT도 같은 방식으로 실패한다.

### 2. Drift-Resilient TabPFN

정적 prior 대신 "시간에 따라 파라미터가 이동하는 2차 SCM"을 명시적으로 모델링해 in-context
베이지안 추론(Helli et al., NeurIPS 2024). 18개 합성·실측 데이터셋에서 XGBoost·CatBoost·TabPFN·
Wild-Time 계열 전부를 능가(정확도 0.688→0.744, ROC-AUC 0.786→0.832). 이 프로젝트에 특별히
관련 있는 이유: 확정된 부정 결과의 반복 패턴("학습구간에서 학습된 표현이 VAL/OOS 하락장에서
붕괴", 캘리브레이션 진단·short 불안정 진단에서 재확인된 train/eval 레짐 불일치)이 정확히 이
논문의 문제 정의(temporal distribution shift)다. 리스크: TabPFN v1 아키텍처 기반이라 원 논문
실험 규모가 이 프로젝트보다 훨씬 작을 가능성이 높고(행수 상한 미확인), 184k행 적용에는 컨텍스트
서브샘플링/슬라이딩 윈도우 전략이 별도로 필요하며, 도구 성숙도가 낮다(automl/Drift-Resilient_TabPFN).

### 3. TabICL

컬럼별→로우별 어텐션으로 행을 고정차원 임베딩 후 트랜스포머 ICL(ICML 2025, 오픈소스
github.com/soda-inria/tabicl). TALENT 200개 분류 데이터셋에서 TabPFNv2 동급 성능을 최대 10배
빠르게, 1만행 초과 56개 데이터셋에서는 TabPFNv2·CatBoost 모두 능가. "합리적 자원에서 50만
샘플까지" 명시 — 184k행을 서브샘플링 없이 수용하는 몇 안 되는 파운데이션 모델. 리스크: 금융
시계열/저SNR 적용 사례 미확인, IID 벤치마크 기준.

### 4. ModernNCA

고전 NCA를 신경망 임베딩+PLR 인코딩+확률적 이웃 샘플링으로 현대화(ICLR 2025). TALENT 300개
데이터셋에서 트리·DL 모두를 능가하는 강한 베이스라인. 경량·고속. prior가 "국소 이웃 기하"로
TabM의 파라메트릭 앙상블과 명확히 다름. 리스크: IID 기준.

### 5. TabPFN 계열 (v2 / 2.5 / 3)

합성 prior로 사전학습된 트랜스포머가 gradient 업데이트 없이 학습셋 전체를 컨텍스트로 받아 1회
forward로 예측. v2(Nature 2025)/2.5는 1만행 권장 상한이라 부적합, TabPFN-3(2026-05, 100만행×
200피쳐)은 규격상 수용. **라이선스 주의**: TabPFN-3 웨이트의 상업 이용 조건이 검색 결과 간
상충("귀속 표시 시 허용" vs "프로덕션은 커머셜 계약 필요")하고 2026-05 SAP의 Prior Labs 인수까지
겹침 — 라이브 트레이딩 편입 전 정확한 조건 재확인 필수(이번 리서치로 확정 불가).

### 6~9. TabR / xRFM / GRANDE / Mitra / RealMLP

- **TabR**(ICLR 2024): retrieval-augmented, 수백만 행 검증. 단 동일 저자그룹의 후속작인 TabM
  논문이 "retrieval 복잡도는 이득 없음"이라 결론 — TabM 이후에도 우위가 유지되는지 불명.
- **xRFM**(2025, ICLR 2026 accepted): 커널 RFM+트리 하이브리드. TALENT 회귀 100개 태스크 1위,
  TabArena-Lite 파레토 상단. 신생 도구 리스크.
- **GRANDE**(ICLR 2024): 미분가능 트리. 검증 규모 작음(이진분류 19개) — GBDT가 같은 prior를 더
  싸게 제공하므로 후순위.
- **Mitra**(Amazon 2025): 설계 타겟이 5천행 미만 — 규모 미스매치로 제외.
- **RealMLP**(NeurIPS 2024): 성능은 좋으나 TabM과 같은 MLP 계열 — "다른 prior" 기준 탈락.

## 레포 실사 — GBDT 전제 조건 확인 (리드 세션, 2026-08-12)

에이전트 권장 1순위(GBDT)는 "이 레포에서 아직 안 해봤다면"이라는 조건부였다. 직접 확인 결과:

1. **FINAL12→`zigzag_action` 3-class GBDT 방향 분류기는 정식으로 돌린 적 없음.** 레포의 GBM
   사용처는 (a) `quality_head` 회귀 전환 검증(타겟이 `tb_*_quality` 연속값, R²≈0으로 기각,
   `eth_h48qual_quality_head_regression_conversion_attempt_20260811.md`), (b) knockoff/relevance
   스크리닝 도구 — 어느 쪽도 방향 분류가 아님.
2. **그러나 강한 부정적 선례가 있음**: 닫힌 리서치 라인
   `eth_overnight_generic_feature_entry_filter_20260809`(`research_line_registry.json`)가 일반
   기술적 피쳐(모멘텀/OFI/변동성/Hawkes/KER + 44피쳐 kitchen sink)로 방향 재추정을 **3개 모델링
   패러다임(3-way 분류·이진 승률·quantile 회귀) 전부에서 실패**시켰다 — kitchen sink 모델은
   in-sample AUC 0.9564 vs DEV/VAL 0.5166/0.5170의 전형적 암기. 그 라인의 retest_guidance는
   "로컬 OHLCV 파생 일반 피쳐의 재변형은 재개 근거가 아니다"라고 명시.
3. **차이점**: FINAL12는 그 kitchen sink와 피쳐 풀이 다르다(funding/whale/toptrader/OU/VAE 등
   h48qual 리서치 패널 파생 — 순수 OHLCV 파생이 아닌 축 포함). 따라서 GBDT 테스트가 순수한
   선례 재발견은 아니지만, **기대치는 선례에 맞춰 낮게 잡아야 한다.**

## 우선순위 권장안 (리드 세션 조정 반영)

1. **GBDT (CatBoost 또는 LightGBM)** — 비용이 거의 0(재학습 수 분, dev CPU로 충분)이고, 이 정확한
   계약으로는 미시도이며, prior가 확실히 다르다. 성공 기대보다는 **"TabM 탓인가, 데이터 탓인가"를
   가장 싸게 분리하는 진단**으로서의 가치 — GBDT도 always-short에 지면 백본 축 전체를 닫는 근거가
   되고, 이기면 그 자체로 발견이다.
2. **Drift-Resilient TabPFN** — 유일하게 이 프로젝트의 진단된 실패 메커니즘(train/eval 분포
   이동)을 정면으로 겨냥. 비용은 높지만 "다른 결과가 나올 이론적 근거"가 가장 강함.
3. **TabICL** — 184k행 무가공 수용 + 오픈소스 + 사전학습 prior. 비용 대비 균형.
4. **ModernNCA** — 저비용, 성숙한 도구(TALENT/LAMDA 레포), 명확히 다른 국소 기하 prior.
5. **xRFM** — 독특한 커널+트리 prior, 단 신생 도구 리스크로 후순위.

TabPFN-3은 규격상 적합하나 라이선스 확정 전 보류. TabR/GRANDE/Mitra/RealMLP는 제외(사유 상기).

## 검증 절차 요구사항 (어느 후보든 공통)

- 기준선: `max(always_long, always_short)` — 동일 구간·동일 비용 가정 (레포 표준).
- Seed-Diversity Gate: N≥5 진짜 다양한 시드. GBDT처럼 결정론에 가까운 모델은 bagging/subsample
  시드로 다양성 확보.
- 교차 라벨변형(h48orig vs h384) 부호 일관성 + VAL→OOS 순서 규율 (이 서브 프로젝트 표준).
- ICL 계열(TabPFN류)은 "학습이 없다"고 해도 Fresh-Forward 규칙이 동일 적용됨: 각 예측 시점의
  컨텍스트에 그 시점 이전 데이터만 포함(슬라이딩 컨텍스트 walk-forward). 전체 구간을 한 번에
  컨텍스트로 넣고 과거 bar를 예측하면 안 됨.

## 정직한 결론

**백본 교체가 문제를 풀 수 있는 조건**: 현재 실패가 "TabM의 파라메트릭 MLP 앙상블이 FINAL12→
`zigzag_action` 사이에 실재하는 (축-정렬적/국소적/시간-드리프트하는) 관계를 표현하지 못해서"라면,
prior가 근본적으로 다른 모델이 다른 결과를 낼 여지가 있다. 이는 계약 문서의 "모델 용량을 늘려도
신호가 안 생긴다"는 확정 결론과 모순되지 않는다 — 그 결론은 TabM 계열 내 용량 확장에 대한
것이고, prior 교체는 다른 실험이다.

**못 푸는 조건**: 이 라벨·구간·피쳐 조합에 애초에 FINAL12로 설명 가능한 신호가 없다면, 상한은
모델이 아니라 feature-label 상호정보량이 결정하며 어떤 prior도 always-short를 넘지 못한다.
40칸 중 38칸 패배라는 이 프로젝트 자체의 기록, TabArena의 최상위권 간 좁은 격차, 그리고
`eth_overnight_generic_feature_entry_filter_20260809` 선례를 함께 보면 **후자일 확률이 더 높다.**
따라서 이 리서치의 실용적 가치는 "백본 교체로 성능 도약"보다는, **1순위(GBDT)를 싸게 돌려
백본 축을 확정적으로 닫거나 열어서, 이미 사용자 결정으로 승격된 최상위 질문("direction_head의
방향 스킬이 어떤 피쳐/라벨/구간 조합에서든 존재하는가")에 집중할 근거를 강화하는 것**이다.
2~5순위는 1순위 결과와 최상위 질문의 진행 상황을 본 뒤 착수 여부를 판단할 것을 권장한다.

## 출처

리드 세션이 독립 재확인한 출처:
- [Drift-Resilient TabPFN (arXiv 2411.10634)](https://arxiv.org/abs/2411.10634) — 저자 Helli, Schnurr, Hollmann, Müller, Hutter 확인: [Semantic Scholar](https://www.semanticscholar.org/paper/7db6aa26864108d0e43cf1bf1ca39e836f069d34), [NeurIPS 2024 poster](https://neurips.cc/virtual/2024/poster/93581), [GitHub automl/Drift-Resilient_TabPFN](https://github.com/automl/Drift-Resilient_TabPFN)

에이전트 웹서치 출처:
- [TabArena: A Living Benchmark (arXiv 2506.16791)](https://arxiv.org/abs/2506.16791) / [리더보드](https://huggingface.co/spaces/TabArena/leaderboard)
- [TabM (ICLR 2025, arXiv 2410.24210)](https://arxiv.org/pdf/2410.24210)
- [TabPFN-3 기술보고서 (arXiv 2605.13986)](https://arxiv.org/pdf/2605.13986) / [Prior Labs 모델 문서](https://docs.priorlabs.ai/models) / [TabPFN-2.5 (arXiv 2511.08667)](https://arxiv.org/html/2511.08667v2)
- [TabICL (ICML 2025, arXiv 2502.05564)](https://arxiv.org/abs/2502.05564)
- [TabDPT (NeurIPS 2025)](https://openreview.net/forum?id=FDMlGhExFp)
- [ModernNCA (ICLR 2025, arXiv 2407.03257)](https://arxiv.org/pdf/2407.03257)
- [RealMLP (NeurIPS 2024)](https://openreview.net/forum?id=fwajDrDy89)
- [xRFM (arXiv 2508.10053)](https://arxiv.org/abs/2508.10053)
- [GRANDE (ICLR 2024, arXiv 2309.17130)](https://arxiv.org/abs/2309.17130)
- [TabR (ICLR 2024, arXiv 2307.14338)](https://arxiv.org/abs/2307.14338)
- [Grinsztajn et al. (NeurIPS 2022, arXiv 2207.08815)](https://arxiv.org/abs/2207.08815)
- [When Do Neural Nets Outperform Boosted Trees? (NeurIPS 2023, arXiv 2305.02997)](https://arxiv.org/abs/2305.02997)
- [TabReD: Tabular ML in-the-Wild (arXiv 2406.19380)](https://arxiv.org/pdf/2406.19380)
- [Mitra — Amazon Science](https://www.amazon.science/blog/mitra-mixed-synthetic-priors-for-enhancing-tabular-foundation-models)
- [Beyond IID: How General Are Tabular Foundation Models? (arXiv 2606.30410)](https://arxiv.org/pdf/2606.30410)
