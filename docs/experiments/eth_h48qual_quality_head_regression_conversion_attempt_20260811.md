# ETH h48qual — quality_head 분류→회귀 전환 시도 (2026-08-11)

## 배경

계약 문서 미해결 이슈 9는 "`quality_head` 분류→회귀 재설계가 다른 세션에서 진행 중이라고
보고받았으나, 레포에 새 손실함수·라벨 스크립트·재스크리닝 산출물이 아직 없다"고 적고 있다. 이
문서가 그 "다른 세션"의 실제 작업이다 — 지금까지 전부 세션 scratchpad에만 있었고 레포에
커밋되지 않아서 안 보였던 것이지, 진행 자체는 실제로 있었다. 스크립트를 `scripts/`로 옮기고
이 문서를 남긴다.

계기: [FINAL12+h384 격리 튜닝 스윕](eth_h48qual_final12_h384_isolated_tuning_sweep_20260811.md)의
v2(15시드) 완료 후, `train_rows=30000`에서 OOS pnl이 5개 quality threshold 전부 통계적으로
유의미하게 양수로 나왔다(Bonferroni 보정 후에도 4/5 생존). 이게 진짜 신호인지 always-short
기준선과 대조해보니 **OOS 15/15 시드 전부 always_short이 모델을 이겼다**(p=0.00015). 원인을
파고들어 `dir_action`(direction_head 원본, 게이트 전)은 숏비중 53~55%로 균형잡혀 있는데
`final_action`(quality 게이트 통과 후)은 75~78%로 치솟는 걸 확인했다 — `quality_head`가
direction_head와 동일한 3-way 분류기이고 "품질 점수"(`quality_for_action`)는
`quality_proba[direction_head가 고른 클래스]`를 재활용한 것뿐이라, 추세에 민감한 자기 라벨로
학습된 quality_head의 편향이 최종 결정에 그대로 스며든다는 게 원인이었다. **이 문서는 그
문제의 수정안으로 "quality_head를 barrier builder가 이미 계산해두고 버리는 연속값
(`tb_long_quality`/`tb_short_quality`)에 대한 회귀로 바꾸면 되는가"를 검증한다.**

## 스크립트

| Script | Role |
|---|---|
| `scripts/verify_eth_h48qual_quality_oracle_gate_20260811.py` | 1단계 — 오라클(미래의 실제 `tb_long_quality`/`tb_short_quality` 값을 안다고 가정) 게이트가 always_short을 이기는지 확인. direction_head 원본 픽은 그대로 두고 게이트 조건만 `tb_quality>0`으로 교체 |
| `scripts/verify_eth_h48qual_quality_gbm_final12_20260811.py` | 2단계 — 오라클이 아니라 실제로 학습 가능한지: `HistGradientBoostingRegressor`로 FINAL12→`tb_long_quality`/`tb_short_quality` 회귀, TRAIN 적합 후 VAL/OOS 홀드아웃 R² 확인 |
| `scripts/rescreen_eth_h48qual_quality_regression_pool201_20260811.py` | 3단계 — FINAL12가 애초에 분류(MI vs discrete class) 기준으로 뽑힌 피쳐라 회귀엔 안 맞을 가능성 검증: h48qual 자체 패널(136 후보, PRICE_LIKE/CONST 제외) + zig075 소스 전용(56) = 201개 전체를 Spearman + `mutual_info_regression`으로 재스크리닝 |
| `scripts/rescreen_eth_h48qual_quality_regression_dedup_20260811.py` | 3단계 계속 — 위 201개 relevance 상위 30개에 mRMR식 중복제거(`|r|>0.5`, 낮은 relevance 탈락) 적용 → REL11 |
| `scripts/verify_eth_h48qual_quality_gbm_rel11_20260811.py` | 4단계 — REL11로 2단계와 동일한 GBM 홀드아웃 재검증 |

**재현성 참고**: 1·3·4단계는 h48qual 자체 리서치 패널(`fa_features.parquet`, 145컬럼)에
의존하는데, 이 파일은 세션 scratchpad에만 있고 레포에 없다 — [FINAL12 피쳐 선택 문서](eth_h48qual_final12_feature_selection_20260811.md)가 이미 지적한 것과 동일한 재현성 갭이다. 2단계는 프로덕션 패널(`omega._load_omega_frames()`)만 써서 이 갭이 없다.

## 결과

### 1단계 — 오라클 게이트: 메커니즘은 유효

direction_head 원본 픽 + `tb_quality>0` 오라클 게이트가 always_short을 **VAL/OOS 둘 다 15/15
시드 전부** 이김. OOS 평균 pnl 40.3%(오라클) vs 18.7%(always_short), paired t-test p<0.00001.
→ "게이트를 진짜 거래 품질로 걸면 좋아진다"는 것 자체는 확실하다.

### 2단계 — GBM(FINAL12): 실전에서는 안 잡힘

| | TRAIN R² | VAL R² | 부호-AUC(VAL) |
|---|---:|---:|---:|
| 정규화 약함(depth=5) | long 0.60 / short 0.60 | long -0.12 / short -0.14 | — |
| 강한 정규화(depth=2, early stopping) | long 0.08 / short 0.11 | long -0.02 / short -0.04 | long 0.53 / short 0.42 |

정규화를 강하게 걸수록 TRAIN 적합도는 낮아지는데(과적합 억제가 작동한다는 뜻) VAL은 그래도
0 근처(대부분 마이너스) — 즉 "정규화가 부족해서"가 아니라 이 피쳐셋엔 진짜 학습 가능한 신호가
거의 없다는 뜻으로 읽힌다.

### 3단계 — 201개 풀 재스크리닝: 방법론은 고쳤지만 결론은 같음

Spearman `|r|` 상위는 변동성 계열(`parkinson_vol` 등, |r|~0.09~0.13 — 약함)과
`regime3_current_sensitive_wide24_chop_prob`. `mutual_info_regression` 상위 30개에서
mRMR식 중복제거(TOP_N=30, `|r|>0.5` 충돌시 낮은 relevance 탈락) 후 **REL11**:

```text
funding_roc_288
funding_roc_48
funding_abs_dt288
cvp_regime
fibonacci_level
realized_skewness
cvd_288
garman_klass_vol
sum_toptrader_long_short_ratio_dt288
hurst_288
svps
```

corr(close) 전부 |r|<0.13(오염 없음, PRICE_LIKE/CONST 제외 확인됨 — 초기 스크리닝에서
`m7_entry_long_price`/`sum_open_interest_value` 등이 최상위로 새는 걸 발견해 deny-list를
추가한 뒤 재실행한 결과). FINAL12와 겹치는 건 4개뿐(`cvp_regime`, `realized_skewness`,
`sum_toptrader_long_short_ratio_dt288`, `funding_roc_48`) — 분류 기준과 회귀 기준이 실제로
다른 피쳐를 고른다는 걸 확인했다. 흥미롭게도 `funding_pressure_diff1`(FINAL12에서
`funding_roc_288`을 이겼던 피쳐)이 여기선 반대로 `funding_roc_288`에게 짐(r=0.995,
funding_roc_288의 MI가 더 높음) — 분류/회귀 기준이 다르면 dedup 승자도 뒤집힐 수 있다는 사례.

### 4단계 — GBM(REL11): 개선 거의 없음

| | VAL R² | VAL 부호-AUC |
|---|---:|---:|
| FINAL12(2단계) | long -0.02~-0.12 / short -0.04~-0.14 | — |
| REL11(4단계) | **long -0.01 / short -0.05** | long 0.53 / short 0.42 |

relevance 지표를 회귀에 맞게 고치고 후보 풀을 12→201로 넓혀도 VAL R²는 그대로 0 근처다.

## 해석

오라클(1단계)과 실전(2·4단계) 사이의 간극이 너무 크다. 이건 **피쳐 선택 방법론의 문제가
아니라 지금 확보한 피쳐 우주 자체에 384bar 앞선 연속 수익률을 예측할 신호가 없다는 뜻**으로
읽는 게 맞다 — 도구(분류 MI → 회귀 Spearman/MI-regression)를 바로잡고, 후보를 12개에서 201개로
넓히고, PRICE_LIKE 오염과 `corr(close)` 검증까지 다 통과시켰는데도 안 나왔기 때문이다. 오라클
결과 자체는 버리지 않는다 — "게이트 메커니즘"이 문제가 아니라 "지금 가진 피쳐로 그 메커니즘에
넣을 좋은 신호를 못 만든다"는 것이 정확한 진단이다.

이 진단은 [`quality_for_action` 스칼라 대안 연구](eth_h48qual_quality_scalar_alternatives_research_20260811.md)의 0단계(재학습 없는 진단: `quality_for_action` vs realized-outcome
순위상관)와 상보적이다 — 그 문서가 제안한 0단계 진단이 여기서 사실상 다른 방식(회귀 relevance
직접 스크리닝)으로 먼저 답을 준 셈이다: 순위상관이 있다 해도 그걸 재현할 만큼 강한 피쳐가 지금
없다.

## 계약 문서에 미친 영향

미해결 이슈 9의 "분류→회귀 전환 자체와 입력 피쳐 재스크리닝"이 이 문서로 완료됐다 — 결론은
"시도했고, 메커니즘은 유효하지만 지금 피쳐로는 실전에서 안 통한다"는 부정 결과다. "헤드별
라벨" 절의 `quality_head` 행은 **분류로 유지**해야 한다(회귀 전환 보류). "피쳐 계약 —
FINAL12" 절의 "재스크리닝 진행 중" 경고는 해제할 수 있다 — 재스크리닝은 끝났고 FINAL12를
바꿀 근거가 안 나왔다(REL11은 GBM에서 FINAL12 대비 유의미한 개선을 보이지 못했다).

## 결과 (계약 문서 반영용)

`quality_head` 분류→회귀 전환 시도 완료, **막다른 길**로 결론. 오라클 게이트는 always_short을
VAL/OOS 15/15 시드로 압도(메커니즘 유효, p<0.00001)하지만, FINAL12/REL11(201개 풀 재스크리닝
후 dedup) 어느 쪽으로도 GBM 홀드아웃 R²가 0 근처(대부분 마이너스)라 실전 신호가 없다. `quality_head`는
분류로 유지, FINAL12 피쳐셋도 변경 근거 없음. 다음 단계는 새 데이터소스(오더북/온체인 등) 없이는
이 방향 보류를 권장 — 대신 [h48orig 컨트롤](eth_h48qual_quality_trend_bias_h48orig_control_20260811.md)이 찾은 "게이트 자체가 문제"라는 진단으로 돌아가 게이트 방식(예: 앙상블 불일치, `quality_scalar_alternatives` 문서의 온도보정)을 봐야 한다.
