# ETH h48qual — Trend-scanning 라벨 MI/R² 사전 게이트 (2026-08-12)

## 목적

오라클 라벨 설계 문헌 리서치([eth_h48qual_oracle_label_design_literature_research_20260812.md](eth_h48qual_oracle_label_design_literature_research_20260812.md))
권장안 1(MI/R² 사전 게이트)+2(trend-scanning)를 함께 실행. TabM/GBDT 풀 학습 전에 저비용으로
"이 라벨이 애초에 학습 가능한가"만 확인 — zigzag_action/h48_conservative가 겪은 "풀 학습까지
가서야 실패를 확인" 패턴을 반복하지 않는 게 목적.

## 방법

- 스크립트: `scripts/verify_eth_h48qual_trend_scanning_label_mi_r2_gate_20260812.py`
- **Trend-scanning 정의**(De Prado): 각 bar t마다 `L ∈ {8,16,24,32,48,64,80,96}`(bar)에서
  `log(close)`를 `[t, t+L]` 구간에서 시간 인덱스에 OLS 회귀, 기울기 t-value 계산 → `|t-value|`
  최대인 `L*` 선택. 연속 타겟=그 t-value, 이산 라벨=`|t-value|≥임계값`이면 부호로 LONG/SHORT,
  미만이면 CASH(90%/95%/99% 근사 임계값 1.65/1.96/2.58 전부 리포트).
- **검증**: 벡터화 구현을 `scipy.stats.linregress`와 12개 지점(L=8/48/96 × 랜덤 4행)에서
  직접 대조 — 전부 소수점 6자리까지 일치. 음수 R² 결과가 계산 버그가 아님을 사전 확인.
- **데이터/피쳐**: h48orig 학습 파이프라인(`_prepare_frames`)의 FINAL12 + TRAIN(2025-01~09,
  78,568행)/VAL(2025-10~12)/OOS(2026-01~02) 그대로 재사용 — GBDT 백본 진단과 동일 관례.
  트렌드스캐닝 라벨은 zigzag_action과 동일하게 연도별 원본 CSV에 독립 적용(연도 경계 문제 회피).
- **MI 게이트**: `mutual_info_classif`(이산 라벨) + `mutual_info_regression`(연속 t-value),
  TRAIN 기준, FINAL12 전체.
- **GBM R² 게이트**: `quality_head` 회귀전환 시도(`verify_eth_h48qual_quality_gbm_final12_20260811.py`)와
  **정확히 동일한 두 설정** — 약한 정규화(`HistGradientBoostingRegressor(max_depth=5,
  max_iter=300)`) / 강한 정규화(`max_depth=2, learning_rate=0.03, l2_regularization=2.0,
  early_stopping=True`). TRAIN 적합 → TRAIN/VAL/OOS R², 부호-AUC(예측값 vs 실제 부호).
- **보조**: 튜닝 없는 단일 LightGBM fit(3-class, early stopping만) — 참고용, 게이트 판정에
  필수는 아님.
- 실행: dev 로컬(이 단계는 의도적으로 가벼움 — Optuna 없음, N=1 fit, 전체 소요 약 1분).

## 데이터 분석 결과 — 트렌드스캐닝 라벨 자체의 특성

- L* 분포(TRAIN): 8bar 2,137건 ~ 96bar(그리드 상한) 30,152건(38.4%) — **최댓값 L=96(그리드
  최상단)이 최빈값**. 이는 짧은 윈도우일수록 자유도(df=L-1)가 작아 t-value 분산이 커 우연히
  큰 |t|가 나오기 쉬운데도, 오히려 긴 윈도우가 더 자주 선택된다는 뜻 — 그리드를 96bar 너머로
  더 넓혔다면 최빈값이 더 늘어났을 가능성이 있어 **그리드 상한 자체가 결과에 영향을 준다**는
  한계로 남는다.
- **이산 라벨의 CASH 비중이 사실상 0%**(90/95/99% 임계값 전부 0.0~0.3%) — 8개 L 후보 중
  최대 `|t-value|`를 선택하는 절차 자체가 다중비교 문제를 안고 있어(8번 검정 중 최댓값을
  고르면 명목 유의수준보다 훨씬 자주 "유의"하게 나옴), 사실상 거의 모든 bar가 active로
  판정됨. De Prado 원 문헌의 표준 관례(다중비교 보정 없이 임계값만으로 필터링)를 그대로
  따른 결과이며, **이 구현의 알려진 한계로 명시적으로 남긴다** — 보정(Bonferroni 등)이나
  다른 L 선택 규칙이 이 문제를 완화할 수 있으나, 아래 R² 게이트 결과(연속 타겟 기준이라
  이 이산화 이슈와 무관)가 이미 결정적으로 부정적이라 이 이슈를 먼저 고치는 우선순위는 낮음.
- zigzag_action과의 방향 일치율(둘 다 active인 bar 기준): TRAIN 67.1% / VAL 67.3% / OOS
  67.8% — 완전히 다른 라벨은 아니지만 상당한 불일치가 있음(참고용, 게이트 판정과 무관).

## MI 게이트 결과 (TRAIN)

상위 5개(연속 t-value 기준): `funding_pressure_diff1`(0.416) > `m7_vae_error_dt288`(0.232) >
`funding_roc_48`(0.304, 소팅상 3위) > `sig_whale_dt288`(0.117) > `cvp_regime`(0.184). 이산
라벨 기준 상위는 `cvp_regime`(0.331) > `funding_pressure_diff1`(0.210) > `funding_roc_48`
(0.159). MI 자체는 zigzag_action(이산 기준 최고 `cvp_regime` 0.414, GBDT 진단 문서 참고)보다
약간 낮지만 0은 아님 — **MI만으로는 이 라벨이 즉시 폐기 대상인지 판단하기 부족**, R² 게이트가
결정적.

## GBM 홀드아웃 R² 게이트 결과 — 결정적 부정

| 설정 | n_iter | TRAIN R² | VAL R² | OOS R² | VAL 부호AUC | OOS 부호AUC |
|---|---:|---:|---:|---:|---:|---:|
| 약한 정규화(depth=5) | 300 | +0.6601 | **-0.1049** | **-0.2272** | 0.545 | 0.505 |
| 강한 정규화(depth=2+ES) | 1000(ES) | +0.1568 | **-0.0043** | **-0.0737** | 0.531 | 0.512 |

- **정규화를 강하게 걸수록 TRAIN 적합도가 낮아지는데도(0.66→0.16, 과적합 억제가 작동한다는
  증거) VAL/OOS R²는 여전히 0 이하** — h48_conservative 회귀전환 시도에서 나온 것과 동일한
  패턴("정규화 부족"이 아니라 "학습 가능한 신호 자체가 거의 없음"의 시그니처).
- 부호-AUC(예측값의 부호가 실제 t-value 부호를 맞추는지, ROC-AUC)는 VAL 0.531~0.545, OOS
  0.505~0.512 — **0.5(무작위) 근처에서 미미하게만 상회**, 방향 판별력이 사실상 없음.

## 보조 — 튜닝 없는 3-class 분류 홀드아웃

VAL balanced_accuracy=0.359, macro_f1=0.238 / OOS balanced_accuracy=0.331, macro_f1=0.227.
CASH가 사실상 없어(위 참고) 실질적으로 LONG/SHORT 2-class에 가까운 문제인데도 이 정도면
무작위(약 0.5) 대비 오히려 낮음 — 물론 이 단계는 튜닝이 전혀 없는 단일 fit이라 GBDT
백본 진단 수준의 결론력은 없지만, R² 게이트가 이미 결정적이라 추가 확인 이상의 의미는 없음.

## 결론 — 게이트 통과 실패, 이 라벨 폐기

**권장 판정 기준(VAL/OOS R² 유의미하게 >0, 부호-AUC가 0.5를 뚜렷이 상회)을 두 정규화 설정
모두, VAL/OOS 둘 다에서 통과하지 못했다.** h48_conservative 오라클 게이트(메커니즘 유효,
15/15 시드 always-short 압도)와 달리, 이번엔 오라클 단계 자체를 건너뛰고 바로 실전 학습
가능성(GBM R²)을 확인했는데도 결과가 부정적이다 — 즉 "메커니즘은 유효한데 예측이 안 되는"
h48_conservative의 패턴과도 다르고, 그냥 **FINAL12로는 이 라벨도 예측이 안 된다**는 더
단순한 결론이다.

이는 GBDT 백본 진단([eth_h48qual_gbdt_backbone_diagnostic_20260812.md](eth_h48qual_gbdt_backbone_diagnostic_20260812.md))의
결론(TabM·GBDT 두 계열 모두 zigzag_action에서 always-short에 완패)과 **네 번째 독립 증거**로
합류한다 — 이번엔 라벨 자체를 바꿔도(zigzag_action 대신 trend-scanning) 결과가 똑같다. 라벨
재설계 축(오라클 라벨 리서치 권장안 2위)이 이걸로 닫힌다.

**남은 권장안(3위 메타라벨링, 4위 MFE 분위수 회귀)에 대한 함의**: 둘 다 여전히 FINAL12를
입력으로 쓴다는 점에서 이번 결과가 사전 확률을 낮춘다 — 특히 3위(메타라벨링)는 이미 이
서브 프로젝트가 `quality_head`라는 구조적으로 유사한 형태로 9개 후보까지 소진한 전례가 있어
우선순위가 더 낮아짐. 4위(MFE 분위수)는 아직 미시도지만, 지금까지 축적된 증거(zigzag 실패,
h48_conservative 오라클-실전 간극, trend-scanning R² 음수, GBDT/TabM 공통 실패)를 종합하면
"FINAL12 자체의 정보량 부족"이 라벨 설계보다 더 근본적인 병목일 가능성이 계속 커지고 있다 —
다음 단계는 라벨 축보다 피쳐 확장(새 원시 데이터소스) 또는 이 VAL/OOS 하락장 구간 밖 재검증
쪽이 더 생산적일 것으로 보인다.

## 산출물

`tmp/eth_h48qual_trend_scanning_mi_r2_gate_20260812/` — `trend_label_analysis.json`,
`mi_gate.json`, `gbm_r2_gate.json`, `light_classification_holdout.json`.
