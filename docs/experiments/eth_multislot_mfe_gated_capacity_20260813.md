# ETH 멀티슬롯 MFE-게이트 용량 확장 — G0~G4 검증 (2026-08-13)

계약(사전등록, 결과 확인 전 작성): `docs/experiments/eth_multislot_mfe_gated_capacity_20260813.json`.
스크립트: `scripts/research_eth_multislot_mfe_gated_capacity_20260813.py`. 산출물:
`tmp/eth_multislot_mfe_gated_capacity_20260813/`.

## 배경

`eth_multislot_capacity_transfer_20260808.json`(균등예산 N=3, 진입정책 불변)가 이미 실패했다 —
VAL PnL +36.82%→+14.15%(-62%), 노출은 거의 안 줄었는데(time-weighted notional -7.8%) PnL만 크게
빠져 "한계신호 품질" 문제로 진단됐고, 그 계약 자체가 N 재스윕·예산배분 재시도를 금지했다. 이
계약은 N=3·균등예산(슬롯당 1/3 margin)은 그대로 두고, **2번째/3번째 슬롯(occupied_count≥1인
진입)에만** 추가 조건을 건다 — 그 bar의 방향별 예측 MFE(이 서브프로젝트에서 유일하게 자체 MI/R²
게이트를 통과한 신호, `eth_h48qual_mfe_quantile_quality_regression_20260812.md`)가 TRAIN 분포
기준 상위 70% 분위수(고정, 스윕 없음) 이상이어야 진입을 허용한다. 첫 슬롯(occupied_count==0)은
완전히 불변 — 08-08과 동일한 신호·우선순위·1/3 사이징.

## 방법 요약

- `multislot_replay()`(08-08, 재사용 불변)를 복사해 `multislot_replay_mfe_gated()`로 확장, 변경점은
  incremental 진입 조건 한 줄뿐.
- MFE 회귀모델: `research_eth_omega461_live_sltp_mfe_width_20260813.py`의 `base102_panel`/
  `train_mfe_models` 그대로 재사용(HistGradientBoostingRegressor, q=0.5, depth=2, TRAIN=
  2025-01-01~09-30, `tb_long/short_mfe_h48_conservative` 타겟). 컷오프는 방향별
  `quantile(TRAIN 예측분포, 0.70)`로 고정.
- **N=5 진짜 무작위 시드**: `[454090186, 918777617, 130430114, 828152837, 415921410]`
  (`random.SystemRandom().sample`, 등차수열 아님).
- G0(N=1 회귀, 08-08 원본 함수 그대로 재현) → G0b(VAL 컴포넌트 재구성 자체검증) → G0c(게이트 무효화시
  구조적 동치성) → G1(추가 admit 트레이드의 **복리 기여도** > 0, mean 아님) → G2(VAL falsification,
  N=1 대비 PnL·MDD 비악화) → **여기서 1차 정지, 오케스트레이터 보고** → 승인 후 G3(OOS 단일 확인,
  이번 한 번만) → G4(effect size, G3 통과시만).
- OOS 윈도우: 08-08 라인의 확장 윈도우 **2026-01-01~06-30**(2개 분기) — 오늘밤 다른 형제 스크립트
  (exit_sweep 계열)가 쓰는 짧은 2026-01-01~03-31과 다름, 혼동 방지를 위해 명시.
- `regime3_current_sensitive_wide24_*` 6컬럼의 ~95행 NaN 갭(2026-02-28 부근, 다른 동시 세션의 데이터
  재생성 과정에서 발생 추정)에 대해 `eth_omega461_live_sltp_wide_calibration_seed_robustness_20260813.md`
  가 쓴 것과 동일한 causal forward-fill 패치를 재사용(G0/G3 공통 적용).

## G0 / G0b / G0c — 전부 통과

| 게이트 | 내용 | 결과 |
|---|---|---|
| G0 | OOS N=1이 08-08 발행값(+77.11%/-15.48%/37건)을 0.05pp 이내로 재현 | **통과**(정확히 일치) |
| G0b | VAL용으로 독립 재구성한 컴포넌트가 `run_window`의 N=1 결과와 정확히 일치 | **통과**(1e-6 이내) |
| G0c | 게이트를 -∞로 무효화한 새 함수가 N=3 원본(08-08의 실패한 그 arm)과 정확히 일치 | **통과**(0.05pp 이내) |
| G0b_oos | OOS용으로 독립 재구성한 컴포넌트가 `run_window`의 OOS N=1 결과와 정확히 일치 | **통과**(1e-6 이내) |

## VAL — G1(복리 기여도)·G2(falsification), 5시드

N=1 baseline: PnL +36.82% / MDD -19.83% / 29건.

| 시드 | VAL PnL% | VAL MDD% | 거래수 | G1(복리기여%) | G1 pass | G2 pass |
|---:|---:|---:|---:|---:|:---:|:---:|
| 454090186 | 72.90 | -10.25 | 63 | +45.36 | O | O |
| 918777617 | 62.51 | -16.02 | 70 | +46.69 | O | O |
| 130430114 | 62.34 | -15.06 | 65 | +46.55 | O | O |
| 828152837 | 71.12 | -14.91 | 66 | +43.11 | O | O |
| 415921410 | 75.93 | -14.08 | 68 | +43.17 | O | O |
| **집계** | | | | | **5/5** | **5/5** |

G1·G2 전부 5/5(≥4/5 기준 초과 달성)로 **VAL 통과** — 08-08의 실패한 균등예산 arm과 뚜렷이
대비된다(08-08은 복리 기여도가 -0.11%로 마이너스, PnL -62%). 이 결과를 오케스트레이터가
`result_val_only_20260813.json`으로 직접 검토 후 G3 집행을 승인했다.

## OOS — G3(단일 확인, 이번 한 번만) · G4

N=1 baseline(OOS, 2026-01-01~06-30): PnL +77.11% / MDD -15.48% / 37건.

| 시드 | OOS PnL% | OOS MDD% | 거래수 | Q1 PnL% | Q2 PnL% | PnL≥N1+3pp | MDD≥-18.5% | Q1&Q2 양수 | G3 pass |
|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|
| 454090186 | 50.22 | -20.17 | 94 | +71.74 | **-12.53** | X | X | X | X |
| 918777617 | 45.54 | -20.17 | 99 | +50.31 | **-3.17** | X | X | X | X |
| 130430114 | 81.74 | -20.17 | 99 | +94.49 | **-6.56** | O | X | X | X |
| 828152837 | 40.63 | -21.57 | 97 | +60.93 | **-12.62** | X | X | X | X |
| 415921410 | 76.50 | -20.17 | 98 | +89.11 | **-6.67** | X | X | X | X |
| **집계(pass_count/5)** | | | | | | **1/5** | **0/5** | **0/5** | **0/5** |

**G3 = 완전 실패(0/5, 기준 ≥4/5).** 세 하위조건 중 어느 것도 4/5 문턱을 못 넘었고, 특히 MDD와
"두 분기 모두 양수" 조건은 **5개 시드 전부** 탈락했다 — 컷오프가 다른 5개의 서로 다른 MFE 모델인데도
**패턴이 완전히 동일**하다: Q1 2026은 전 시드 강하게 양수(+50~+94%)인데 Q2 2026은 전 시드 음수
(-3.2~-12.6%). MDD도 4/5 시드가 정확히 -20.17%로 동일(1개만 -21.57%)해서, 시드별 노이즈라기보다
**이 구체적 OOS 구간(특히 2026 Q2)에 구조적으로 물린 것**으로 보인다. G3가 실패했으므로 계약대로
**G4(effect size)는 실행하지 않았다.**

### 정직하게 덧붙이는 뉘앙스

08-08의 실패 원인("추가 admit된 트레이드 자체가 손해")과 이번 실패는 **메커니즘이 다르다.** 이번엔
OOS에서도 incremental(추가 admit)trades 자체의 복리 기여도는 5개 시드 전부 뚜렷한 **양수**였다
(+18.4%~+52.7%, 표 아래 참고). 즉 MFE 게이트가 고른 추가 신호 자체는 이번 OOS 구간에서도 여전히
말이 됐다 — 문제는 신호 품질이 아니라 **포트폴리오 레벨 리스크**(최대 3슬롯 동시보유가 만드는 동시
드로다운폭)와 **Q2 특정 레짐**이 겹친 것으로 보인다. 이건 08-08과 이번 계약이 사전에 명시한 adverse
evidence(오늘밤 다른 MFE 폭조정 실험도 VAL 5/5→OOS 0/5로 완전반전된 전례,
`eth_omega461_live_sltp_wide_calibration_seed_robustness_20260813.md`)와 같은 계열의 패턴 —
**VAL에서 시드 5개 전부 이겨도 OOS 레짐 차이 앞에서는 무의미할 수 있다**는 이 세션의 반복 교훈이
세 번째로 재현된 것이다. 다만 이번엔 "신호가 노이즈였다"보다는 "용량 확장이 만드는 동시노출 리스크가
특정 분기에 크게 물렸다"는, 조금 더 정밀한 실패 지점을 남겼다는 점만 기록해 둔다 — 이 관찰이
재시도나 재파라미터화의 근거는 아니다(아래 결론 참고).

incremental trades 상세(OOS, 참고용): 454090186 64건/+26.5%, 918777617 68건/+18.9%, 130430114
65건/+52.5%, 828152837 67건/+18.4%, 415921410 67건/+52.7%.

## 최종 결론

**CLOSE.** G3가 사전등록된 기준(≥4/5 시드, PnL·MDD·양분기 동시 충족)을 결정적으로 충족하지
못했다(0/5). 계약이 명시한 대로 이번이 **유일한 OOS 확인 기회**였고, 실패했으므로 **재튜닝·재확인
없이 이 축을 닫는다** — 컷오프 분위수(q=0.70)를 다른 값으로 바꾸거나, N을 조정하거나, 피쳐패널을
바꿔서 재시도하지 않는다. G4(effect size)는 계약대로 실행되지 않았다(G3 실패 시 스킵).

08-08(균등예산, 무게이트)과 이번(MFE 분위수 게이트)을 합쳐, ETH 멀티슬롯 용량 확장 축은 **두 가지
서로 다른 진입정책 모두 OOS 기준을 통과하지 못했다** — 하나는 한계신호 자체가 나빴고(08-08), 하나는
한계신호는 괜찮았지만 포트폴리오 리스크가 특정 분기에 물렸다(이번). 라이브 승격은 물론
섀도우 준비 단계조차 이 계약에서는 근거가 없다(계약의 outcome ceiling은 애초에 "섀도우가
최선"이었는데, G3 실패로 그 최선조차 도달하지 못했다).

## 산출물

- 계약: `docs/experiments/eth_multislot_mfe_gated_capacity_20260813.json`
- 스크립트: `scripts/research_eth_multislot_mfe_gated_capacity_20260813.py`
- VAL 전용 결과(오케스트레이터 검토본, `--stage val`): `tmp/eth_multislot_mfe_gated_capacity_20260813/result_val_only_20260813.json`
- 최종 결과(G0~G3 전체, `--stage all`): `tmp/eth_multislot_mfe_gated_capacity_20260813/result.json`
- 원장: `tmp/eth_multislot_mfe_gated_capacity_20260813/ledger_{val,oos}_n3_mfegated_seed<seed>.csv`,
  `{val,oos}_incremental_trades_seed<seed>.csv`
- 실행 로그: `tmp/eth_multislot_mfe_gated_capacity_20260813/run_log.txt`(VAL),
  `run_log_stage_all.txt`(G0~G4 전체)
- 인용 선행 계약/문서: `docs/experiments/eth_multislot_capacity_transfer_20260808.json`,
  `docs/experiments/eth_h48qual_mfe_quantile_quality_regression_20260812.md`,
  `docs/experiments/eth_omega461_live_sltp_mfe_width_20260813.md`,
  `docs/experiments/eth_omega461_live_sltp_wide_calibration_seed_robustness_20260813.md`
