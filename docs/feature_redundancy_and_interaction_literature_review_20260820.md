# ETH 피쳐 리던던시 제거 + 조합(상호작용) 방법론 문헌조사 (2026-08-20)

## 배경

`eth_tabm_label_logic_retest_initiative_20260819` 서브프로젝트에서 라벨/기법/데이터량 축을
전부 소진한 뒤, 사용자가 "새 피쳐셋은 못 구하니 정리하고 조합하자"고 지시. 1차로 ad-hoc 방법
(pairwise correlation 임계값 클러스터링 + rank-percentile 곱 상호작용 + permutation-null)으로
158→133개 정리, 8,778쌍 조합 전수조사를 했으나 둘 다 chance 결론이었다(`docs/experiments/
eth_directional_change_tabm_nhits_training_20260819.md` 참고). 사용자가 이 방법론 자체를
문헌으로 검증해달라고 재요청 — 두 개 독립 리서치 에이전트(각 5-8+회 웹서치, 논문/저서/1차
자료 우선)로 조사했다.

## 1. 리던던시 제거 — 문헌 대조

**발견된 방법**: mRMR(Peng/Long/Ding 2005, 지도학습·상호정보량 기반, target 정보를 선택 과정에
직접 씀) · VIF 기반 제거(Marquardt 1970 기원, Belsley/Kuh/Welsch 1980 정식화, O'Brien 2007이
"VIF>10" 류 관용 임계값이 유도값이 아니라고 비판) · 상관거리 계층적(Ward linkage) 클러스터링
(scikit-learn 공식 예제) · PCA/SAS PROC VARCLUS(변환이지 선택이 아님, 피쳐 이름 소실) · **López de
Prado의 ONC**(Optimal Number of Clusters — 거리 d=√((1−ρ)/2) 위 계층적 클러스터링, 클러스터
개수를 실루엣 t-stat으로 자동결정, López de Prado & Lewis 2019 *Quantitative Finance*) **+
CFI**(Clustered Feature Importance, *Machine Learning for Asset Managers* 2020 Ch.6 — 단
목적이 "트리앙상블 importance 편향 교정"이라 사전학습 리던던시 제거와는 다른 문제).

**내 방법과 비교**: |corr|≥0.95 연결요소(union-find)는 수학적으로 **single-linkage
클러스터링**과 동일(Everitt et al. *Cluster Analysis* 5판). Target-aware 대표선정(클러스터 내
최고 AUC)은 sklearn 공식예제(대표를 임의선택)나 순수 비지도 ONC보다 낫지만, 진짜 사각지대
2개는 남는다: (1) single-linkage 특유의 **chaining**(A-B, B-C가 각각 임계 넘으면 A-C가 안
넘어도 전이적으로 합쳐짐), (2) pairwise correlation은 **3개 이상 피쳐에 분산된 다중공선성**을
구조적으로 못 봄(D≈0.4A+0.4B+0.4C인데 어느 pairwise corr도 안 높은 경우) — 이건 VIF가 정확히
잡도록 설계된 문제.

**권고 및 실제 적용**: 문헌은 내 방법을 "정리/스크리닝 목적으로는 방어 가능, 다만 VIF 사각지대는
실제로 존재"로 평가. `scripts/eth_dc_feature_vif_check_20260820.py`로 correlation matrix
역행렬 대각원소(=VIF, statsmodels 없이 정확히 동일값) 실측 → **`regime3_current_sensitive_
wide24_{bull,bear,chop}_prob` 3개가 VIF~3×10¹³**(사실상 완전 특이) — 직접 확인 결과
**bull_prob+bear_prob+chop_prob=1.0이 전 구간(2025+2026) 예외없이 정확히 성립**(확률 단체
제약, std=0.0). Pairwise correlation으로는 절대 못 잡는 케이스였고 문헌이 지적한 정확히 그
사각지대였다 — `chop_prob` 1개 제거(133→132, 나머지 2개가 세번째를 완전결정하므로 정보손실
없음)로 수정. VIF≥10인 나머지 39개는 소프트 다중공선성(관용 임계값 자체가 비유도값이라는
O'Brien 비판 + TabM처럼 계수해석이 아닌 비선형모델에선 중요도가 상대적으로 작다는 문헌 뉘앙스
고려)이라 추가 제거는 보류 — 이미 10회 이상 "피쳐셋 바꿔도 chance"가 반복된 상황에서 이
132번째 vs 133번째 피쳐 하나 차이로 재학습 사이클을 또 도는 비용 대비 효익이 낮다고 판단.

## 2. 조합(상호작용) — 문헌 대조

**발견된 방법**: Deep Feature Synthesis/featuretools(Kanter & Veeramachaneni 2015, 다중테이블
전용이라 이 문제엔 부적합) · 유전프로그래밍 피쳐구축(Tran/Xue/Zhang 2019) · **GA2M/EBM**(Lou
et al. 2013, FAST 랭킹으로 상호작용쌍 자동선택) · **Friedman & Popescu H-statistic**(2008,
*Annals of Applied Statistics* — 적합된 모델에서 상호작용 강도를 직접 측정) · **RuleFit**(같은
논문, 트리노드 경로 전체를 규칙피쳐로) · **SHAP interaction values**(Lundberg et al. 2018/2020,
TreeExplainer) · **RIT/iRF**(Random Intersection Trees, Shah & Meinshausen 2014; Iterative
Random Forests, Basu et al. 2018 PNAS — 포레스트의 공유 의사결정경로에서 안정적인 고차
상호작용을 채굴). 고전통계: Aiken & West(1991)의 평균중심화 곱 상호작용항이 표준. 다중비교:
**Westfall & Young(1993) permutation maxT**(FWER 통제) vs Benjamini-Hochberg(1995, FDR
통제) — 목적이 다름.

**내 방법과 비교**: rank-percentile 곱(z_i×z_j)은 **Aiken & West 곱 상호작용항의 강건한
스케일불변 변형**(idiosyncratic이 아니라 표준적 구성)이고, 내 permutation max-statistic은
**정확히 Westfall-Young maxT**다(GWAS의 genome-wide-significance 관행과 동형). "8,778개 중
1위가 진짜인가"라는 내 질문에는 Bonferroni(과보수적)나 BH-FDR(다른 질문에 최적화)보다 maxT가
문헌상 더 적합한 도구라고 확인됨. Pooled p=0.030→2026단독 재검증 p=0.515 붕괴 패턴은 **교과서적
winner's-curse 시그니처**이고, held-out 재검증으로 잡아낸 것 자체가 "권장되는 표준 해법"이라는
평가. 단, 구조적 사각지대는 분명함: z_i×z_j는 **매끄러운 단조 사분면(bilinear) 형태에만
민감** — "A가 상위10%이고 B가 중간대일 때만" 같은 **국소적/비단조/문턱형 상호작용**은 곱셈
공식 자체가 못 본다(H-statistic/SHAP-interaction/RIT류가 이 형태를 다루도록 설계됨).

**권고 및 실제 적용**: 문헌은 "bilinear 계열에 한해 내 음성결과는 강건하다. 다만 tree-shape
상호작용은 다른 도구가 필요하며, 이미 이 프로젝트에서 수십 개 축이 chance로 수렴한 이력을
감안하면 사전확률이 낮으니 전면 재실행은 비권장, 단 저비용 점검(GBM 1개 적합 → 상위 20-50쌍만
→ 동일 held-out 재검증)은 해볼 만하다"고 결론. shap/statsmodels는 미설치(+공유 conda env 신규
의존성 추가 회피)라, **개념적으로 동일한 RIT 방식**(LightGBM 트리구조에서 같은 경로에 공동출현
하는 조상-자손 split_feature 쌍을 split_gain 가중 집계)으로 대체 구현
(`scripts/eth_dc_gbm_interaction_discovery_20260820.py`) — **discovery는 2025(train)만
사용**해 앞선 pooled-오염 문제가 재발하지 않도록 설계. Top-30쌍을 완전히 분리된 2026(eval)
단독으로 검증(`scripts/eth_dc_gbm_interaction_2026_validation_20260820.py`, K=30이라 다중비교
부담도 훨씬 작음) — **결과: 최고AUC=0.5232(cvp_regime×upper_wick_z), empirical_p=0.400
(chance)**. 비단조/트리형 상호작용을 전용으로 찾도록 설계된 방법으로도, 훨씬 관대한
다중비교 기준에서도 신호가 없었다.

## 결론

리던던시 제거(단일연결 클러스터링의 다중공선성 사각지대)와 조합(bilinear 상호작용의 비단조
사각지대) 둘 다 문헌이 지적한 구조적 한계가 실제로 존재했고, 각각 VIF·트리기반(RIT식)
방법으로 메웠다. VIF는 진짜 결함(확률 단체 완전선형종속)을 하나 찾아 수정했지만, 트리기반
상호작용 탐색은 (더 강력한 방법임에도) 여전히 chance를 재확인했다. 이걸로 피쳐 엔지니어링
축(정리+선형조합+비선형조합)이 문헌 검증까지 마친 상태로 완전히 소진됐다고 판단한다.

## 참고문헌

1. Peng, H., Long, F., & Ding, C. (2005). Feature selection based on mutual information criteria of max-dependency, max-relevance, and min-redundancy. *IEEE TPAMI*, 27(8), 1226–1238. DOI: 10.1109/TPAMI.2005.159
2. Guyon, I., & Elisseeff, A. (2003). An Introduction to Variable and Feature Selection. *JMLR*, 3, 1157–1182.
3. Hall, M. A. (1999). *Correlation-based Feature Selection for Machine Learning*. PhD thesis, University of Waikato.
4. Marquardt, D. W. (1970). Generalized Inverses, Ridge Regression, Biased Linear Estimation, and Nonlinear Estimation. *Technometrics*, 12(3), 591–612.
5. Belsley, D. A., Kuh, E., & Welsch, R. E. (1980). *Regression Diagnostics*. Wiley.
6. O'Brien, R. M. (2007). A Caution Regarding Rules of Thumb for Variance Inflation Factors. *Quality & Quantity*, 41(5), 673–690. DOI: 10.1007/s11135-006-9018-6
7. Jolliffe, I. T. (2002). *Principal Component Analysis* (2nd ed.). Springer.
8. scikit-learn developers. Permutation Importance with Multicollinear or Correlated Features (공식 예제).
9. López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Ch.8.
10. López de Prado, M., & Lewis, M. J. (2019). Detection of False Investment Strategies Using Unsupervised Learning Methods. *Quantitative Finance*, 19(9), 1555–1565.
11. López de Prado, M. (2020). *Machine Learning for Asset Managers*. Cambridge Elements in Quantitative Finance. Ch.6.
12. López de Prado, M. (2016). Building Diversified Portfolios that Outperform Out-of-Sample. *Journal of Portfolio Management*, 42(4), 59–69.
13. Everitt, B. S., Landau, S., Leese, M., & Stahl, D. (2011). *Cluster Analysis* (5th ed.). Wiley.
14. Kanter, J.M., & Veeramachaneni, K. (2015). Deep Feature Synthesis. *IEEE DSAA 2015*.
15. Tran, B., Xue, B., & Zhang, M. (2019). Genetic programming for multiple-feature construction on high-dimensional classification. *Pattern Recognition*, 93, 404–417.
16. Lou, Y., Caruana, R., Gehrke, J., & Hooker, G. (2013). Accurate Intelligible Models with Pairwise Interactions. *KDD '13*, 623–631.
17. Friedman, J.H., & Popescu, B.E. (2008). Predictive Learning via Rule Ensembles. *Annals of Applied Statistics*, 2(3), 916–954. (RuleFit + H-statistic)
18. Lundberg, S.M. et al. (2020). From Local Explanations to Global Understanding with Explainable AI for Trees. *Nature Machine Intelligence*, 2, 56–67.
19. Basu, S., Kumbier, K., Brown, J.B., & Yu, B. (2018). Iterative random forests to discover predictive and stable high-order interactions. *PNAS*, 115(8), 1943–1948.
20. Shah, R.D., & Meinshausen, N. (2014). Random Intersection Trees. *JMLR*, 15, 629–654.
21. Li, Y., Turkington, D., & Yazdani, A. (2020). Beyond the Black Box. *Journal of Financial Data Science*, 2(1).
22. Aiken, L.S., & West, S.G. (1991). *Multiple Regression: Testing and Interpreting Interactions*. Sage.
23. Westfall, P.H., & Young, S.S. (1993). *Resampling-Based Multiple Testing*. Wiley.
24. Benjamini, Y., & Hochberg, Y. (1995). Controlling the False Discovery Rate. *JRSS B*, 57(1), 289–300.

## 관련

- `docs/experiments/eth_directional_change_tabm_nhits_training_20260819.md` — 이번 조사 이전의
  ad-hoc 정리+조합 결과(1차) 및 서브프로젝트 전체 맥락
- `scripts/eth_dc_feature_vif_check_20260820.py` / `eth_dc_gbm_interaction_discovery_20260820.py`
  / `eth_dc_gbm_interaction_2026_validation_20260820.py` — 이번 조사에서 새로 만든 검증 스크립트
- 메모리: `eth_tabm_label_logic_retest_initiative_20260819`
