# ETH h48qual — 오라클 라벨 설계 최신 문헌 리서치 (2026-08-12)

## 목적과 범위

사용자 지시: "기존에 트리플 배리어, 지그재그 모두 사용해봤는데 좋은 결과를 얻지 못했다 — 오라클
라벨에 대해 최신 논문 조사 및 심층 연구." **문헌 리서치만 — 학습/구현/코드 변경 없음.**

- 방법: Model Architect 페르소나 단독 dispatch(Sonnet, 웹서치 29회).
- 리드 세션 검증: 이 보고서의 가장 이례적인 3개 핵심 인용(2026년 arXiv 논문 2건 + MDPI 2025 논문
  1건)을 독립 웹서치로 재확인 — **전부 실재**: `arXiv:2602.03395`(Label Horizon Paradox, Song/
  Liu/Chen, 2026-06-08 게재), `arXiv:2604.15531`(Spurious Predictability, Nikolopoulos, 2026-04-16
  게재), `MDPI Applied Sciences 15(24):13204`(AEDL, 2025). 오라클 게이트 R²≈0 인용도 원본
  실험 문서(`eth_h48qual_quality_head_regression_conversion_attempt_20260811.md:118`)와 대조해
  정확함을 확인.

## 이 프로젝트가 이미 확정한 배경 (리서치 프롬프트에 반영)

- **zigzag_action**(ATR14 적응형 피벗, 8bar 이상 파동, ±2bar 전환버퍼): direction_head가 N=5~15
  시드 정식 검증에서 40칸 중 38칸 always-short 패배. SHORT 스윙이 LONG보다 경로가 통계적으로
  유의하게 "깔끔"(Calmar 비율 높음, p=2.87e-05)해 confidence가 숏에 3~5pp 쏠리는 비대칭이 학습에
  구워짐.
- **h48_conservative**(48/384bar 트리플 배리어, TP=max(0.6%,1.2·ATR96)/SL=max(0.4%,0.8·ATR96)):
  **오라클(미래 실제값)로 게이트하면 always-short를 15/15 시드 압도(메커니즘 유효)** — 그러나
  FINAL12/REL11(201개 풀 재스크리닝) 어느 쪽으로도 GBM 홀드아웃 R²가 0 근처(대부분 마이너스).
- 이 조합(라벨은 유효, 예측 불가)이 이번 리서치의 핵심 질문: **라벨 재설계로 이 간극이 메워지는가,
  아니면 피쳐 자체의 상호정보량 부족 문제라 라벨을 바꿔도 소용없는가?**

## 요약

문헌 전체가 수렴하는 결론: **라벨 설계 개선은 "예측 가능성"과 "거래 가능성"을 별개 축으로 다뤄야
하며, 대부분의 실패는 이 둘을 혼동한 데서 온다.** Triple-barrier/meta-labeling(De Prado 2018)
이후 연구는 (i) 라벨을 더 통계적으로 유의하게 만드는 방향(trend-scanning), (ii) 이산에서 연속/
분포로 바꿔 노이즈 바닥을 낮추는 방향(quantile/soft label), (iii) 예측 실패를 필터링으로 우회하는
방향(meta-labeling)으로 갈라졌다. **2026년 최신 논문 2건이 정확히 이 프로젝트가 부딪힌 문제를
정면으로 다룬다**: Label Horizon Paradox(최적 지도신호가 최종 목표와 다른 중간 호라이즌으로
이동한다)와 Spurious Predictability(검증 절차 자체가 가짜 예측가능성을 체계적으로 과대추정한다,
오라클 게이팅으로 안정 레짐 생존을 확인하라는 방법론 — 이 프로젝트가 이미 수행한 것과 동일).
이 프로젝트의 zigzag/triple-barrier 실패는 문헌상 드문 사례가 아니라 **전형적 패턴**(라벨은
메커니즘상 유효, MI는 0에 가까움)에 해당한다.

## 라벨링 방법 분류 체계

| 방법 | 대표 출처 | 핵심 아이디어 | 예측가능성 | 거래가능성 | 함정 |
|---|---|---|---|---|---|
| Fixed-horizon return | De Prado, *AFML*, 2018(책) | t+n 수익률 부호를 그대로 라벨화 | 최저 — 경로 무시 | 낮음 — 체결조건 미반영 | 벤치마크용으로만 권장(AEDL 등 최신 문헌 공통 지적) |
| **Triple-barrier**(현재 사용) | De Prado 2018; MDPI *Mathematics* 12(5):780(2024, 크립토); Springer *Financial Innovation*(2025, 크립토+DL) | TP/SL/시간 3중 배리어 중 먼저 닿는 것 | 중간 — 경로 반영하나 하드 임계값 | 중간~높음 — 실제 주문조건에 근접 | 배리어 간 변동성 미반영, 전환점 근처 라벨 플립, 클래스 불균형 |
| **Trend-scanning**(t-value) | De Prado, *ML for Asset Managers*(책); AEDL 논문 내 baseline | 구간 내 회귀기울기 t-value 최대인 호라이즌 자동선택 | AEDL 벤치마크 Sharpe 0.00(fixed −0.29, TB −0.03보다는 나음, 그 자체로 무의미) | 낮음~중간 | look-forward L 클수록 오버피팅, "통계적 유의"≠"거래 가능" |
| Meta-labeling | De Prado 2018; Hudson&Thames 케이스(S&P500 E-mini) | 1차 방향모델 신호를 2차 모델(베팅 여부)로 필터링 | 2차 타겟이 종종 더 쉬움 | 높음 — precision 직결 | 정확도 개선 착시(no-trade 비중 증가) — 해당 사례 정확도 17%→63% vs precision 0.17→0.20 |
| MFE/MAE 라벨 | 실무 문헌(Van Tharp 계열), 피어리뷰 ML 실증 미확인 | 보유구간 최대 유리/불리 이탈폭 | 미확인 | 개념상 높음 | 전환점 근처 극값 민감, 학술 검증 희소 |
| First-passage time | arXiv:2507.08101(2025, 리스크분류) | 확률과정이 임계값을 처음 넘는 시점 | 수리적으로 정교, 실거래 실증 희박 | 미확인 | triple-barrier와 사실상 동형 — 신규 정보 이득 의문 |
| Label smoothing/soft label | arXiv:2301.10458(2023) | 원핫 라벨을 α만큼 스무딩 | 과적합 완화, 신호 자체는 안 만듦 | 간접적 | 스무딩 강도가 또 다른 HP, 근본 신호부재는 못 고침 |
| Quantile/expectile 타겟 | arXiv:2404.09154(2024); arXiv:2406.00998(2024) | 방향 대신 수익률 분포/분위수 예측 | 이진 임계값 근처 노이즈 회피 가능 | 매매규칙 변환 별도 필요 | 5분봉 방향 트레이딩 직접 검증 사례 미확인 |
| Learning-to-rank | Poh/Lim/Zohren/Roberts, *JFDS* 3(2):70/arXiv:2012.07149(2020) | 절대수익 대신 자산간 상대순위 | 단일자산보다 안정적(주장) | Sharpe 약 3배 개선 보고(cross-sectional) | 본질적으로 cross-sectional 전제 — 단일자산엔 부적합 |
| Denoised/자기지도 라벨 | arXiv:2112.10139/ACM ICAIF 2022 | 오토인코더로 라벨 재구성해 노이즈 제거 | pretext task 정확도 개선 주장 | 실거래 백테스트 수치 미확인 | CV 기법의 금융 전이, 비정상성 검증 한계 |
| **AEDL**(regime-aware) | **MDPI *Applied Sci.* 15(24):13204(2025)** | 멀티스케일+Granger causality/transfer entropy+MAML로 라벨 파라미터 레짐별 적응 | 16자산 25년 벤치마크서 baseline 대비 우위 | **Sharpe 0.48 vs fixed −0.29/TB −0.03/trend-scan 0.00** | 구현 복잡도 최고, 단일 논문 재현성 미검증 |
| **Label Horizon Paradox** | **arXiv:2602.03395(2026)** | 최적 지도신호가 최종 목표와 다른 중간 호라이즌으로 이동 — bi-level 최적화로 자동 탐색 | 대규모 금융 데이터셋서 baseline 대비 일관 개선 | 미확인(direct 트레이딩 백테스트 아님) | 이 프로젝트의 48/384bar 스윕보다 급진적 — 호라이즌 조정만으론 부족하다는 주장 |
| **Spurious Predictability** | **arXiv:2604.15531(2026)** | falsification-audit로 검증절차 자체의 가짜예측력 과대추정을 탐지 | 방법론(라벨 자체 아님) | 방법론 | 이 프로젝트의 오라클게이트 실험과 동일 접근 — "이미 하고 있었다" |
| RL 가치함수(라벨 대체) | ScienceDirect(2022, 크립토); SITMO 블로그(보조) | 라벨을 PnL 최적화 행동으로 재정의 | 이 레포 RL 선례 다수(간략화) | 크립토 SL>RL 사례 있음(일반화 미확인) | 보상설계 문제로 이전할 뿐 |

## 상세 — 이 프로젝트에 특히 관련된 4건

**Label Horizon Paradox (Song, Liu, Chen, arXiv:2602.03395, 2026-06-08)**: "최적 지도 신호는
종종 최종 예측 목표와 다른 중간 호라이즌으로 이동한다"는 것이 핵심 — 신호-노이즈 트레이드오프의
동적 경쟁으로 이론화하고, bi-level 최적화로 단일 학습 런 안에서 최적 프록시 라벨을 자동 탐색하는
프레임워크를 제안, 대규모 금융 데이터셋에서 일관된 개선을 보고. 이 프로젝트의 48bar→384bar
호라이즌 스윕은 이미 이 방향의 초보적 시도였지만, 이 논문은 "호라이즌만 조정해선 부족하고 목표와
분리된 프록시 자체를 설계해야 한다"는 더 급진적인 제안이다.

**Spurious Predictability (Nikolopoulos, arXiv:2604.15531, 2026-04-16)**: 표준 검증 절차가 금융
데이터에서 진짜 예측가능성을 체계적으로 과대추정한다는 falsification-audit 방법론. 합성
참조계급(zero-predictability 환경, microstructure placebo) 대비 워크포워드 백테스트를 대조하고,
"통과한" 워크플로만 실제 신호로 인정. **이 프로젝트의 오라클 게이트 실험(실제 미래값으로 게이트해
always-short 대비 메커니즘 유효성만 먼저 확인한 뒤, 그 다음에야 예측가능성을 별도로 검증)은
이 논문이 사후에 권장하는 것과 방법론적으로 동일** — 이미 옳은 절차를 밟고 있었다는 교차검증.

**AEDL (MDPI *Applied Sciences* 15(24):13204, 2025)**: 멀티스케일(5개 시간축) + Granger
causality/transfer entropy 기반 인과 필터링 + MAML 메타러닝으로 라벨 파라미터를 레짐별 적응.
16개 자산 25년(2000-2025, val 2023-2025) 벤치마크에서 Sharpe 0.48로 fixed-horizon(-0.29)/
triple-barrier(-0.03)/trend-scanning(0.00)을 모두 앞섬 — **그러나 trend-scanning 단독은 이
벤치마크에서도 사실상 무의미(0.00)**, AEDL의 우위는 멀티스케일+인과필터+메타러닝의 결합에서
나온다. 구현 복잡도가 가장 높고 단일 논문 실증만 있어 재현성 검증이 안 됨.

**Meta-labeling 재조명**: Hudson & Thames 케이스(S&P500 E-mini, 블로그 보조출처)의 "정확도
17%→63%" 수치는 인상적이지만 precision은 0.17→0.20로 거의 개선이 없다 — 개선 대부분이 "거래
안 함" 비중 증가에서 왔을 가능성. 이 프로젝트 맥락에서는 오히려 유용한 재구성일 수 있다: zigzag의
always-short 대비 열위는 "틀린 방향에 베팅"의 문제이므로, 방향 라벨 자체를 재설계하는 대신
"이 방향 신호를 믿을지"를 별도 2차 타겟으로 학습하는 편이 더 쉬울 수 있다. 단, 이 프로젝트는
이미 유사한 "필터형" 시도(quality_head 자체가 사실상 메타라벨 게이트)를 상당 부분 소진했다는
점(계약 문서의 `quality_head` 대체 리서치 9개 후보 전부 소진)이 감안돼야 한다.

## 우선순위 권장안 (에이전트 원안, 리드 세션 그대로 채택)

기준: 검증 비용 낮음 & 이 프로젝트의 실패 메커니즘을 실제로 우회할 가능성 높음.

1. **MI/R² 사전 게이트 재실행 — 신규 라벨 정의 전 필수 관문.** 후보 라벨마다 GBM 홀드아웃 R²/
   AUC와 FINAL12 각 피쳐 Spearman을 h48_conservative와 동일 절차로 먼저 계산, 유의미하게 0보다
   큰 것만 TabM 풀 학습으로 승격. 이 프로젝트의 확정 교훈과 Spurious Predictability의
   falsification-audit 권고가 정확히 일치 — GPU 시간 전에 반드시 통과.
2. **Trend-scanning으로 zigzag 대체, t-value를 회귀/샘플가중치 병행.** L∈[8,96bar]에서 OLS
   기울기 t-value 최대 구간 선택. zigzag의 "8bar+±2bar 버퍼" 하드코딩을 제거하는 방향. 단
   AEDL 벤치마크에서도 trend-scanning 단독은 Sharpe 0.00이므로 게이트 필수.
3. **Meta-labeling: 기존 zigzag_action 위에 "베팅 여부"만 판별하는 2차 분류기.** 1차=기존 방향,
   2차 타겟="1차가 맞았나". 기존 zigzag 출력 재사용이라 검증비용 낮음 — 단 이 프로젝트가 이미
   quality_head 형태로 유사 구조를 9개 후보까지 소진했다는 점 고려.
4. **h48qual quality_head를 MFE 분위수(q10/50/90) 회귀로 전환.** 하드 임계값(TP/SL 히트)이
   노이즈 바닥 근처였을 가능성 — 연속 분포 타겟이 부분 신호를 흡수할 여지. 5분봉 방향성 직접
   실증 문헌은 없어 리스크 있음.
5. **(보류 권장) AEDL류 regime-aware/causal-filtered 라벨.** 구현 복잡도 최고, 단일 논문
   실증뿐 — 1~4위가 전부 게이트를 통과 못한 뒤에야 투자 가치.

## 정직한 결론

라벨 재설계가 문제를 풀 수 있는 조건은 **라벨 정의 변경이 그 라벨과 FINAL12 사이의 실제
상호정보량 자체를 바꾸는 경우**뿐이다(하드 임계값→연속 타겟, 데이터 기반 호라이즌 등). 이 경우는
이론적으로 정당하고 문헌(Label Horizon Paradox, AEDL)도 실제 이득을 보고한다.

그러나 이 프로젝트의 오라클 게이트 결과(h48_conservative가 오라클로는 always-short를 15/15
압도하지만 GBM R²≈0)는 **더 근본적인 한계를 시사한다**: 라벨이 담은 정보는 실전에서 돈이 되는
진짜 신호(메커니즘 유효)인데, 그런데도 예측이 안 된다는 것은 문제가 라벨의 "정의 방식"이 아니라
**FINAL12 12개 피쳐가 애초에 미래 48bar 경로의 방향/품질을 결정하는 정보를 담고 있지 않다**는
데 있을 가능성이 높다. 이 경우 라벨을 아무리 정교하게 재정의해도(soft label, quantile,
meta-labeling, trend-scanning 무엇이든) 상호정보량 예산 자체는 늘지 않는다 — Spurious
Predictability 논문이 경고하는 바와 정확히 같다: 검증 절차·라벨 정의를 바꿔서 나오는 "개선"은
새로운 형태의 노이즈 적합일 위험이 크다.

**실용적 함의**: 라벨 재설계 실험은 전부 저비용 MI/R² 사전 게이트를 통과한 것만 승격해야 하며,
권장안 2~4위가 모두 게이트에서 R²≈0로 반복되면 그것은 라벨 설계 문제가 아니라 **피쳐 확장(새
원시 피쳐 발굴) 없이는 이 자산·타임프레임에서 어떤 지도학습 라벨도 유의미한 신호를 갖지 못한다**는
결론으로 받아들여야 한다 — 이는 앞서 닫힌 새-데이터소스 리서치 라인(청산/마이크로구조/온체인/
Polymarket, 전부 인프라 문제 또는 부정 결과)과 지금 진행 중인 GBDT 백본 진단(같은 라벨/피쳐로
다른 모델을 시도) 두 축이 이미 겨냥하고 있는 바로 그 질문과 합류한다.

## 출처

리드 세션이 독립 재확인한 핵심 출처:
- [Label Horizon Paradox (arXiv:2602.03395)](https://arxiv.org/abs/2602.03395) — Song, Liu, Chen, 2026-06-08
- [Spurious Predictability in Financial ML (arXiv:2604.15531)](https://arxiv.org/abs/2604.15531) — Nikolopoulos, 2026-04-16
- [AEDL (MDPI Applied Sciences 15(24):13204)](https://www.mdpi.com/2076-3417/15/24/13204) — 2025

에이전트 웹서치 출처(그 외):
- De Prado, M.L. (2018). *Advances in Financial Machine Learning*. Wiley. (책, 보조출처)
- De Prado, M.L. *Machine Learning for Asset Managers*. (책, 보조출처)
- [Enhanced Genetic-Algorithm-Driven Triple Barrier Labeling (MDPI Mathematics 12(5):780, 2024)](https://www.mdpi.com/2227-7390/12/5/780)
- Springer *Financial Innovation* (2025), DOI 10.1186/s40854-025-00866-w — 페이월, 세부수치 미확인
- [Label Unbalance in HFT (arXiv:2503.09988, 2025)](https://arxiv.org/abs/2503.09988)
- [Learning to Rank cross-sectional momentum (arXiv:2012.07149, 2020)](https://arxiv.org/abs/2012.07149)
- [Denoised Labels via Self-Supervised Learning (arXiv:2112.10139, ACM ICAIF 2022)](https://arxiv.org/abs/2112.10139)
- [Label smoothing news-based stock classification (arXiv:2301.10458, 2023)](https://arxiv.org/abs/2301.10458)
- [Extreme quantile regression with deep learning (arXiv:2404.09154, 2024)](https://arxiv.org/abs/2404.09154)
- [Distributional Refinement Network (arXiv:2406.00998, 2024)](https://arxiv.org/abs/2406.00998)
- [First-passage risk classification (arXiv:2507.08101, 2025)](https://arxiv.org/abs/2507.08101)
- Hudson & Thames 블로그, "Does Meta Labeling Add to Signal Efficacy?" (보조출처)
- ScienceDirect (2022), "Outperforming RL trading systems: a supervised approach to crypto" (보조출처)

**미확인 항목**: MFE/MAE 라벨의 피어리뷰 ML 실증, first-passage 라벨의 트레이딩 직접 적용 실증,
quantile 회귀 라벨의 5분봉 방향성 직접 실증, Springer crypto 논문 세부 수치(페이월).
