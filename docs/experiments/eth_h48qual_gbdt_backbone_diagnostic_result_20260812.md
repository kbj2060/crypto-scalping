# ETH h48qual — GBDT 백본 진단 풀런 결과 (2026-08-12)

**배경**: `docs/experiments/eth_h48qual_tabm_backbone_replacement_model_research_20260812.md`의
1순위 권장안("TabM 탓인가 데이터 탓인가"를 가장 싸게 분리하는 진단) 실행 결과. 스크립트:
`scripts/train_eval_eth_h48qual_final12_gbdt_backbone_diagnostic_20260812.py`(별도 세션 작성,
서버 GPU 머신에서 실행 — LightGBM 자체는 CPU 학습이지만 서버에서 돌림). 결과 아티팩트를 pull해서
스크립트 로직과 함께 직접 확인함(아래 "검증" 절).

**방법 요약**: h48orig 학습 파이프라인(`_prepare_frames`)을 그대로 재사용 — FINAL12 피쳐, 실제
라이브 zigzag_action 라벨, h48orig 5시드 재현판과 동일한 TRAIN(2025-01~09, 9개월,
78,568행)/VAL/OOS 윈도우(계약 문서 숫자와 직접 비교 가능하도록 의도적으로 맞춤). TRAIN 내부
월별 확장윈도우 CV(embargo 48bar) → Optuna 80 trials(목적함수: CV multi_logloss) → 상위 5개
CV 후보를 VAL 거래 시뮬레이션(cost_mult=3.0)으로 재평가해 **always_short 대비 마진이 가장 큰
것**을 채택(select-on-validation-only) → 그 HP로 N=8 진짜 무작위 시드(`random.SystemRandom`)
최종 학습, OOS는 이 단계에서 최초 1회만 읽음(blind). always_short/long은 (a) 모델과 동일 active
set에 강제, (b) 전체 bar 강제 두 버전 다 계산.

**주의(스크립트 자체가 명시)**: GBDT엔 TabM h48orig의 레짐별 expert_scale(0.75/0.90/0.90 notional
배율)에 대응하는 라우터가 없어 적용하지 않음 — GBDT 대 GBDT-always-short 비교는 완전히 대칭이지만,
GBDT의 절대 PnL 크기를 TabM h48orig의 절대 PnL 크기와 직접 비교할 땐 이 차이를 감안해야 함.

## 결과

| 구간 | cost | gbdt pnl | always_short(동일active) | always_long(동일active) | gbdt 승 | wilcoxon p |
|---|---|---:|---:|---:|---:|---:|
| VAL | cost1 | +1.75±4.60 | +11.30±1.64 | -12.89±1.29 | 0/8 | 1.0000 |
| VAL | cost2 | +0.53±2.98 | +10.72±2.13 | -12.93±1.21 | 0/8 | 1.0000 |
| VAL | cost3 | -7.91±3.55 | +9.76±1.76 | -13.49±1.21 | 0/8 | 1.0000 |
| OOS | cost1 | +8.03±4.53 | +21.27±3.07 | -20.67±0.39 | 0/8 | 1.0000 |
| OOS | cost2 | +3.97±7.41 | +20.97±2.96 | -20.44±0.46 | 0/8 | 1.0000 |
| OOS | cost3 | +2.07±5.75 | +20.74±2.25 | -20.80±0.48 | 0/8 | 1.0000 |

분류: VAL balanced_acc=0.469±0.001·macro_f1=0.446±0.002, OOS balanced_acc=0.470±0.002·
macro_f1=0.454±0.003 (3-class 무작위 chance=0.333 대비는 높지만 아래 해석 참고).

**HP 탐색 자체가 이미 마진을 못 찾음**: 채택된 HP(trial#55)는 상위 5개 CV 후보 중 VAL
always-short 대비 마진이 가장 큰 걸 골랐는데도 그 마진 자체가 **-13.88%p**(`winning_hp.json`).
즉 "margin 최대화"를 목적으로 한 선택 절차 자체가 양의 마진을 가진 후보를 하나도 못 찾았다.

**피쳐 중요도(gain, 8시드 평균)**: `vwap_dist_24`(105,291)와 `regime3_current_sensitive_wide24_chop_prob`(32,291),
`breakout_strength`(22,434)가 나머지(1,600~4,300, `mta_funding`은 191)를 압도 — 그런데도 PnL
엣지로 이어지지 않음.

## 검증 (트러스트 벗 베리파이)

스크립트 전체를 직접 읽고 확인: 워크포워드 CV+embargo, select-on-validation-only, OOS
1회·최후 blind read, N=8 진짜 무작위 시드(고정간격 아님), Wilcoxon 단측검정 — 전부 이
서브프로젝트의 기존 표준과 일치. `winning_hp.json`/`feature_importance_gain_mean.json`/
`final_classification_metrics.csv`를 직접 열어 로그 마지막 요약과 대조 — 일치 확인.
Confusion matrix 확인 결과 채택된 HP(`class_weight_mode=none`)가 CASH를 거의 예측하지 않음
(VAL 17,496행 중 예측 CASH 총합 ~90건대) — 즉 GBDT는 거의 매 bar 거래 중이라, TabM의
"ungated"(게이트 없이 거의 매 bar 거래) 실험과 비슷한 활성 프로필로 always-short와 대조되고
있음. 원본 CSV의 개별 시드 PnL도 요약 통계(평균±표준편차)와 정합.

## 해석 및 결론

**GBDT(LightGBM)도 8시드·6개 비용조합 전부 always-short에 완패, 통계적으로 결정적(p=1.0000
전부)** — TabM h48orig(N=40칸 중 2칸 승)와 사실상 같은 결론. 문헌상 이 행/피쳐 비율(9개월
78,568행÷12피쳐)이 GBDT에 유리한 구간이라는 점, 그리고 HP 탐색이 명시적으로 always-short
마진을 목적함수로 상위 후보를 재평가했는데도 양의 마진 후보가 하나도 없었다는 점까지 감안하면,
**"TabM이라는 특정 아키텍처의 용량/귀납편향 문제"라는 가설은 이걸로 상당히 약해진다** —
완전히 다른 계열(축-정렬 트리 앙상블 vs 연속 가중합 신경망)도 같은 데이터에서 같은 실패를
반복했기 때문이다. 오늘 완료된 오라클 라벨 설계 문헌 리서치(`docs/experiments/
eth_h48qual_oracle_label_design_literature_research_20260812.md`)의 결론("FINAL12 12개
피쳐 자체가 미래 경로 정보를 담고 있지 않을 가능성")과 정확히 같은 방향으로 수렴한다.

**이 결과가 뜻하는 것**: 백본 교체(TabM→GBDT나 다른 아키텍처)는 우선순위가 낮아진다 — 문제가
모델 용량이 아니라 피쳐/신호 자체일 가능성이 이걸로 한 번 더 강화됐기 때문. 오늘 스카우팅
문서(`eth_h48qual_direction_skill_new_directions_scouting_20260812.md`)가 1순위로 꼽은
direction-only 피쳐 재스크리닝, 그리고 오라클 라벨 리서치가 제안한 신규 라벨 후보(MI/R² 사전
게이트 필수)가 지금 가장 근거가 강한 다음 방향으로 남는다.
