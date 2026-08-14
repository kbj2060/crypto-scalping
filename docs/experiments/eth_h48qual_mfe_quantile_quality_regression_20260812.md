# ETH h48qual — MFE 분위수 회귀 quality_head (2026-08-12)

## 배경

오라클 라벨 설계 문헌 리서치 권장안 4위: `quality_head`를 하드 임계값(TP/SL 히트) 분류 대신
MFE(Maximum Favorable Excursion) 연속/분위수(q10/50/90) 회귀로 전환. 이 세션에서 시도한
6종 모델(TabM/GBDT/오토인코더/TCN/CNN/one-vs-rest) + trend-scanning 라벨 재설계가 전부
저비용 MI/R² 사전 게이트 또는 실전 대조에서 부정으로 끝난 뒤, 유일하게 아직 안 써본 후보였다.

## 1단계 — MI/R² 사전 게이트 (통과)

스크립트: `scripts/verify_eth_h48qual_mfe_quantile_label_mi_r2_gate_20260812.py`. 타겟:
`build_omega1_2_triple_barrier_labels_20260619.py`가 h48_conservative 배리어 계산의
부산물로 이미 저장해둔 `tb_long_mfe_h48_conservative`/`tb_short_mfe_h48_conservative`(새로
계산 안 함)를 zigzag_action 방향에 맞춰 선택. h48orig 파이프라인의 FINAL12+TRAIN(2025-01~09,
78,568행)/VAL/OOS 그대로 재사용.

| 설정 | TRAIN R² | VAL R² | OOS R² | VAL spearman | OOS spearman |
|---|---:|---:|---:|---:|---:|
| 약한 정규화(depth=5) | +0.67(과적합) | -0.02 | +0.004 | +0.17 | +0.27 |
| 강한 정규화(depth=2+ES) | +0.16 | **+0.08** | **+0.14** | **+0.28**(p<0.001) | **+0.39**(p<0.001) |

분위수(q10/50/90) 회귀도 커버리지가 명목수준에 근접(q10 14.7%/17.1%, q50 51.4%/55.1%, q90
87.0%/86.0%)하고 spearman이 세 분위 전부 VAL/OOS 다 양(+0.25~+0.37). **이 세션에서 시도한
라벨×타겟 조합 중 유일하게 이 게이트를 결정적으로 통과했다.**

**Confound 체크**: MFE가 단순 변동성 프록시(고변동성 구간을 맞히는 것뿐이라면 MAE도 같이
커져야 함)가 아닌지 확인 — spearman(MFE, MAE크기) = TRAIN -0.32 / VAL -0.24 / OOS -0.10(전부
p<1e-30, MFE 높을수록 오히려 불리한 이탈은 작음), spearman(MFE, 실현손익) = 세 구간 전부
+0.43(p=0). 변동성 프록시가 아니라 진짜 거래품질(작은 드로다운+좋은 실현손익)과 결이 맞는
신호로 확인. `funding_pressure_diff1`(MI 최상위 피쳐) 가격추세 오염 체크도 통과(|spearman
with close| < 0.07, 기준 0.5~0.6 훨씬 아래).

## 2단계 — TabM 풀 학습 N=5 진짜 무작위 시드 (부정)

- 학습 스크립트: `scripts/train_eval_omega4_h48qual_mfe_quality_regression_20260812.py` —
  기존 `train_eval_omega4_quality_regression_20260621.py`의 `ThreeHeadQualityRegTabM`
  아키텍처+quantile-relative 게이팅(threshold=quantile(train 예측분포, 0.70))을 재사용,
  `_barrier_quality_targets`만 몽키패치해 게이트와 정확히 같은 사전계산 MFE를 재시뮬레이션
  없이 그대로 씀. `_prepare_frames` 시그니처 불일치(구버전 호출) 발견 및 수정(4개 필수
  kwarg 누락, h48qual 표준값으로 채움).
- 서버 GPU 디스패치, epochs=4/max-train-rows=30000(h48orig 재현판과 동일 관례), 시드
  13036874/747899465/799474674/570627141/842447243(진짜 무작위).
- 필수 검증(`scripts/verify_eth_h48qual_mfe_quality_reg_always_short_baseline_20260812.py`):
  같은 active bar 집합에서 방향만 강제숏/강제롱한 always_short/always_long과 cost1/2/3 전부
  대조.

| | VAL | OOS |
|---|---:|---:|
| model pnl | +5.6~+6.2(±17~18, 시드간 분산 큼) | +9.4~+10.2 |
| always_short pnl | +11.1~+11.5 | **+22.0~+22.8** |
| always_long pnl | -14.2~-15.2 | -18.3~-19.3 |
| 승(vs always_short) | 3/5(p=0.81~1.00) | **1/5**(p=0.06~0.19) |
| 승(vs always_long) | 4/5(p=0.125) | 5/5(p=0.0625) |

**always_long은 확실히 이긴다**(하락장 구조상 당연, 큰 의미 없음). **always_short 대조가
결정적**: VAL은 완전 무의미(거의 동전던지기), OOS는 1/5로 사실상 완패(유의성은 표본 5개라
약하지만 방향은 뚜렷). MI/R² 게이트에서 확인된 통계적 예측가능성이 always_short을 이기는
실전 엣지로 전환되지 않았다.

## 결론

**게이트 통과 → 실전 대조 실패**라는, h48_conservative 오라클 게이트가 겪은 것과 정확히
같은 패턴이 재현됐다. MFE는 진짜 학습 가능한 신호였지만(변동성 confound도 아니고, 실현손익과
강하게 상관됨), 그 신호가 이 특정 OOS 구간(2026-01~02, 매끈한 단조 하락)에서 always_short의
구조적 우위를 이길 만큼 강하지는 않았다.

**이 서브 프로젝트 전체 스코어카드**: TabM/GBDT/오토인코더/TCN/CNN/one-vs-rest(6개 모델) +
zigzag_action/trend-scanning/MFE회귀(3개 라벨×타겟) 조합을 다 합쳐 **7번째 독립 조합이
같은 곳(OOS에서 always_short 못 이김)에 수렴**. 라벨 재설계 권장안 1~4위(MI/R² 게이트,
trend-scanning, 메타라벨링, MFE 분위수)가 이제 전부 소진됐다(3위 메타라벨링은 quality_head
자체가 이미 유사구조로 9개 후보 소진해서 애초에 낮은 우선순위였음). 5위(AEDL류 regime-aware
라벨)만 미시도로 남았으나, 문헌 자체가 "1~4위가 전부 게이트 통과 못한 뒤에야 투자 가치"라고
명시한 최후순위이고 구현 복잡도도 최고다.

## 산출물

- `tmp/eth_h48qual_mfe_quantile_mi_r2_gate_20260812/` — MI/R² 게이트 산출물.
- `tmp/causal_regen_20260516/omega4_quality_regression_20260621_h48qual_mfe_quality_reg_20260812_s<seed>/` —
  5개 시드 학습 결과(report.json, VAL/OOS 예측).
- `tmp/eth_h48qual_mfe_quality_reg_5seed_always_short_baseline_20260812/pnl_comparison.csv` —
  always-short/long 대조.
