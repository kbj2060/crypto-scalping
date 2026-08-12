# ETH h48qual — GBDT(LightGBM) 백본 진단: FINAL12 → zigzag_action (2026-08-12)

## 목적

TabM 백본 대체 후보 리서치([eth_h48qual_tabm_backbone_replacement_model_research_20260812.md](eth_h48qual_tabm_backbone_replacement_model_research_20260812.md))
1순위 권장안 실행. "TabM 탓인가, FINAL12/zigzag_action 데이터 조합 자체 탓인가"를 가장 싸게
분리하는 진단 — GBDT 승격 시도가 아니다.

## 방법

- 스크립트: `scripts/train_eval_eth_h48qual_final12_gbdt_backbone_diagnostic_20260812.py`
- 파이프라인은 h48orig 학습 스크립트의 `_prepare_frames()`를 그대로 재사용 — FINAL12 피쳐 구성
  (REGIME3_CURRENT 오버레이 + diff1/dt288 파생 4개), 실제 라이브 h48orig `zigzag_action` 라벨
  디렉토리, `BASE_TEMPLATE`(TP=2.6%/SL=1.4%/notional=0.45/leverage=2.0, max_hold/cooldown=0),
  검증된 bar-by-bar 거래 시뮬레이션(`omega._metrics`)을 완전히 그대로 씀. 재구현 없음.
- **TRAIN 윈도우**: 2025-01~09(9개월, 78,568행) — h48orig 5시드 재현판과 동일한
  isolated-verification 관례. 라이브 번들의 진짜 21개월 TRAIN(2024-01~2025-09, 183,936행)이
  아님(계약 문서 "함정 1"). 기존 TabM h48orig 5시드 숫자와 직접 비교 가능하게 하려고 의도적으로
  맞춤.
- **GBDT 자신의 거래 경제성**: notional/leverage/TP/SL은 고정값(레짐별 expert_scale 0.75/0.90/
  0.90 미적용 — GBDT에는 대응하는 라우터가 없음). GBDT와 그 자신의 always-short/always-long
  기준선 사이 비교는 완전 대칭이지만, 계약 문서의 기존 TabM 숫자(레짐 스케일 포함)와 절대 PnL
  크기를 직접 비교할 때는 이 차이를 감안해야 함.
- **데이터 분석**: 클래스 균형, 피쳐별 PSI drift(TRAIN vs VAL/OOS), Spearman 상관 재확인, 단변량
  mutual information(vs `zigzag_action`).
- **HP 튜닝**: TRAIN 내부 월별 확장윈도우 CV(5폴드, 2025-05~09 순차 검증, embargo 48bar) 위에서
  Optuna 80 trials, 목적함수=평균 multi_logloss. LightGBM 파라미터(num_leaves/max_depth/
  learning_rate/min_child_samples/reg_alpha/reg_lambda/feature_fraction/bagging_fraction/
  bagging_freq) + class_weight_mode(none/balanced) + cash_weight_mult(0.5~3.0) 탐색.
- **모델 선택**: CV 상위 5개 후보를 VAL 거래 시뮬레이션(cost3=3x)으로 재평가, always-short 대비
  마진이 가장 큰 후보를 최종 채택 — **select-on-validation-only**, OOS는 이 단계에서 건드리지
  않음.
- **최종 검증**: 채택된 HP로 N=8 진짜 무작위 시드(`random.SystemRandom().sample`, 고정간격
  증가 아님 — Seed-Diversity Ensemble Promotion Gate 준수) 학습, VAL/OOS × cost1/2/3에서
  always_short/always_long과 대조(동일 active-bar-set 강제전환 버전 + 전체 bar 강제 버전 둘 다).
  OOS는 이 단계에서 최초 1회만 읽음(blind).
- 실행: 서버(handoff.sh) GPU 불필요(CPU만), 전체 소요 약 7분(Optuna 80trial×5fold 379초 + 최종
  8시드 학습/평가 수 초).
- 사전 검증: 로컬 스모크(3 trial, 2시드) → 서버 스모크(동일 파라미터) → 두 환경 수치 완전 일치
  확인(결정론적 파이프라인) → 풀 실행.

## 데이터 분석 결과

- TRAIN 클래스 분포: CASH 9,243(11.8%) / LONG 36,283(46.2%) / SHORT 33,042(42.1%) — 기존 h48orig
  report.json과 일치(레포 다른 진단에서 이미 확인된 값 재확인).
- FINAL12 내 TRAIN 최대 |spearman| = 0.475(대각 제외) — mRMR/knockoff dedup 배제 기준(0.561)
  이내, 재선택 불필요 확인.
- 단변량 MI(TRAIN, `zigzag_action` 3-class) 상위: `cvp_regime`(0.414) > `funding_pressure_diff1`
  (0.226) > `ou_halflife`(0.204) > `funding_roc_48`(0.173) > `vwap_dist_24`(0.159).
- **PSI drift(TRAIN vs VAL/OOS) > 0.25**(중간~심한 이동): `funding_pressure_diff1`,
  `m7_vae_error_dt288`, `mta_funding`, `sig_whale_dt288` — 이 4개는 이 프로젝트가 이미 반복
  확인한 "train/eval 레짐 비정상성"과 같은 축의 정량적 재확인. 전체 수치:
  `tmp/eth_h48qual_gbdt_backbone_diagnostic_20260812/data_analysis.json`(서버, dev에 pull 완료).

## HP 탐색 결과

- CV 최저 multi_logloss = 0.7823 (80 trials). 참고: 균등분포 예측의 로그로스는 ln(3)=1.099,
  TRAIN 클래스 기저비율만 예측해도 이론상 ≈0.975 — 모델이 기저비율보다는 낮은 로그로스를
  달성해 **분포 수준에서는 어느 정도 구조를 포착함**(완전 무작위는 아님).
- 채택된 HP(trial#55, VAL margin=-13.88, 상위 5개 CV 후보 중 always-short 대비 손실이 가장
  작았던 후보): `num_leaves=7, max_depth=11, learning_rate=0.074, min_child_samples=15,
  reg_alpha=0.030, reg_lambda=5.15e-5, feature_fraction=0.634, bagging_fraction=0.764,
  bagging_freq=3`, `class_weight_mode=none`, `cash_weight_mult=1.063`, `n_estimators=70`.
  상위 5개 CV 후보 전부 VAL에서 always-short 대비 마이너스 마진(-13.88 ~ -22.17) — HP 탐색
  구간 전체에서 always-short를 이기는 설정 자체가 없었음.

## 최종 결과 — N=8 시드, VAL/OOS × cost1/2/3

| 구간 | cost | GBDT PnL | always_short(동일active) | always_long(동일active) | always_short(전체bar) | GBDT 승 | Wilcoxon p |
|---|---|---:|---:|---:|---:|---:|---:|
| VAL | cost1 | +1.75±4.60 | +11.30±1.64 | -12.89±1.29 | +10.94 | 0/8 | 1.0000 |
| VAL | cost2 | +0.53±2.98 | +10.72±2.13 | -12.93±1.21 | +10.91 | 0/8 | 1.0000 |
| VAL | cost3 | -7.91±3.55 | +9.76±1.76 | -13.49±1.21 | +10.24 | 0/8 | 1.0000 |
| OOS | cost1 | +8.03±4.53 | +21.27±3.07 | -20.67±0.39 | +21.41 | 0/8 | 1.0000 |
| OOS | cost2 | +3.97±7.41 | +20.97±2.96 | -20.44±0.46 | +22.08 | 0/8 | 1.0000 |
| OOS | cost3 | +2.07±5.75 | +20.74±2.25 | -20.80±0.48 | +20.91 | 0/8 | 1.0000 |

- **6개 구간(VAL/OOS × cost1/2/3) 전부, 8시드 전부 always-short 패배** — Wilcoxon one-sided
  p=1.0000 전 구간(가장 결정적인 형태의 부정 결과 — 우연이나 경계선 근처가 아니라 완전히
  한쪽으로 쏠림).
- GBDT 자체 PnL도 cost가 오를수록 VAL에서 마이너스로 전환(+1.75→-7.91), 승률 32.3~37.2%(VAL)/
  32.6~42.5%(OOS) — 이 거래 경제성(TP2.6%/SL1.4%)의 손익분기 승률 약 35% 근처를 맴돌거나 밑돎.
- 분류 지표: balanced_accuracy 0.469(VAL)/0.470(OOS), macro_f1 0.446(VAL)/0.454(OOS) — 3-class
  기저비율(46/42/12%)을 감안하면 "완전 무작위"보다는 낫지만 방향 판별력은 약함.
- 피쳐 중요도(gain, 8시드 평균): `vwap_dist_24`(105,291)가 압도적 1위, 이어서
  `regime3_current_sensitive_wide24_chop_prob`(32,291), `breakout_strength`(22,434) — 상위
  3개가 전체 gain의 대부분을 차지. `mta_funding`(191)은 사실상 미사용. GBDT가 신호를 못 찾은
  것이 "아무 피쳐도 안 씀"이 아니라 "특정 피쳐 조합에 강하게 의존했는데도 결과가 안 남"이라는
  점을 보여줌 — 데이터 자체의 정보량 부족을 시사.

## 검증

- 로컬 스모크와 서버 스모크가 결정론적으로 완전히 동일한 수치(데이터분석/CV폴드) 산출 — 파이프라인
  cross-machine 정합성 확인.
- 최종 CSV(`final_multiseed_results.csv`)의 거래수(39~69건/시드)·승률(32~43%)·always_short
  matched pnl의 시드간 근소한 변동(GBDT 자신의 active-bar-set이 시드마다 살짝 다름) 전부
  직접 대조해 이상 없음 확인.
- 산출물: `tmp/eth_h48qual_gbdt_backbone_diagnostic_20260812/`(서버 + dev pull 완료) —
  `data_analysis.json`, `optuna_trials.csv`, `top_candidates_val_reeval.csv`,
  `winning_hp.json`, `final_multiseed_results.csv`, `final_classification_metrics.csv`,
  `feature_importance_gain_mean.json`.

## 결론

**GBDT(LightGBM)는 진짜 HP 탐색(Optuna 80trial)과 시계열 CV, N=8 진짜 무작위 시드, VAL→OOS
blind 규율을 전부 갖춘 상태에서도 TabM과 동일하게 always-short에 완패했다.** 이는 TabM 백본
대체 후보 리서치가 예상한 두 갈래 중 "신호가 애초에 없다" 쪽을 뒷받침하는 세 번째 독립 증거다:

1. TabM h48orig 5시드: VAL/OOS 0/5 승
2. TabM h384 v2 15시드: 40칸 중 2칸만 승(VAL만)
3. **GBDT(이 진단) 8시드: VAL/OOS × cost1/2/3 전 구간 0/8 승, Wilcoxon p=1.0000**

인덕티브 바이어스가 근본적으로 다른(축-정렬 비미분 분할 트리 vs 연속 파라메트릭 MLP 앙상블)
두 모델 계열이 동일한 피쳐·라벨·구간에서 동일하게 실패했다는 것은, 문제가 "TabM의 표현력
한계"가 아니라 **FINAL12가 이 구간의 `zigzag_action`을 예측하는 데 필요한 정보를 담고 있지
않을 가능성**을 강하게 시사한다. 오라클 라벨 설계 문헌 리서치([eth_h48qual_oracle_label_design_literature_research_20260812.md](eth_h48qual_oracle_label_design_literature_research_20260812.md))의
"라벨을 재설계해도 상호정보량 예산 자체가 없으면 소용없다"는 결론과 정확히 합류한다.

**남은 백본 후보(Drift-Resilient TabPFN, TabICL, ModernNCA, xRFM)에 대한 함의**: 이들이 겨냥하는
"다른 prior가 다른 결과를 낼 가능성"의 사전 확률이 이번 GBDT 결과로 한 단계 더 낮아졌다.
완전히 다른 계열(트리 vs MLP)조차 차이를 못 만들었다면, 남은 후보들(대체로 여전히 신경망 기반
표현학습)이 다를 가능성은 더 낮게 봐야 한다. 다만 Drift-Resilient TabPFN처럼 "시간적 분포
이동을 정면으로 모델링"하는 계열은 여전히 질적으로 다른 시도이므로 완전히 배제하긴 이르다 —
단, 우선순위는 낮춰야 한다.

**서브 프로젝트 최상위 질문에 대한 함의**: 사용자가 이미 승격한 "direction_head가 어떤 피쳐/
라벨/구간 조합에서든 방향 스킬을 갖는가"라는 질문이 이제 "어떤 **모델**에서든"까지 포함하도록
더 넓게 확정됐다. 다음 단계는 모델 축이 아니라 피쳐·라벨·구간 축(오라클 라벨 리서치의 권장안 —
MI/R² 사전 게이트를 통과하는 새 라벨 후보 탐색, 또는 이 VAL/OOS 하락장 구간 밖 재검증)이 유일한
남은 생산적 방향으로 보인다.
