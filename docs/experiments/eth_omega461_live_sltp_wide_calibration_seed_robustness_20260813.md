# ETH Omega4.6.1 라이브 SLTP — 넓은-쪽 캘리브레이션 시드 강건성 검증 (2026-08-13)

이 문서는 `docs/experiments/eth_omega461_live_sltp_mfe_width_20260813.md`(선행실험1)와
`docs/experiments/eth_omega461_live_sltp_asymmetric_tpsl_20260813.md`(선행실험2)에서 **독립적으로**
발견된 side-finding을 검증한다. 두 실험 모두 "SLTP 폭→거래수 감소" 문제 자체는 실패로 최종
결론지었지만(상수 플로어 재조정+MFE기반 대칭/비대칭 폭 학습 둘 다로도 해결 안 되는 구조적
트레이드오프), 그 그리드 탐색 과정에서 "MFE 예측 기반 폭을 baseline보다 넓히는 쪽" 설정 일부가
baseline 대비 PnL도 크게 높고 MDD도 뚜렷이 낮은 결과를 반복해서 보였다. **이 문서는 그 문제(거래수
증가)를 다시 풀려는 시도가 아니다 — 오직 "이 넓은-쪽 설정이 baseline을 이기는 게 시드에 걸쳐
재현되는가"만 순수하게 검증한다.**

## 방법

### 1단계 — 후보 셀 확정 (재확인)

두 선행실험의 `report.json`(`priority_combined_val`, 포트폴리오/우선순위결합 레벨)을 전수
스캔해서 baseline 대비 **no_gate·with_gate 양쪽 모두에서 PnL과 MDD가 동시에 개선**된 셀을
독립적으로 추출했다(오케스트레이터가 직접 원장을 열어 확인한 결과와 정확히 일치):

- **symmetric_scale9**(선행실험1, base102 피쳐셋, `tp_scale=9.0`, `sl_ratio=sl_mult/tp_mult=0.5`):
  no_gate +130.25%/-18.45%, with_gate +143.53%/-15.47% (baseline +36.82%/-24.34%, +54.88%/-31.11%).
- **asymmetric_tp9_sl1.5**(선행실험2, `tp_scale=9.0`, `sl_scale=1.5`): no_gate +123.68%/-15.51%,
  with_gate +123.53%/-22.22%.

이 두 셀 외에 두 그리드(대칭 스케일 7개×2피쳐셋, 비대칭 7×3=21개) 전체에서 이 조건을 만족하는
셀은 없었다.

### 2단계 — N=5 진짜 무작위 시드 재학습

CLAUDE.md Seed-Diversity Ensemble Promotion Gate와 같은 정신(고정 간격 증분 금지, 진짜 무작위
추출)으로 시드 5개를 OS 엔트로피 기반 `random.sample`로 생성: `[453827194, 121952941, 501601563,
139411872, 13643480]`(등차수열 아님, 직접 확인). 각 시드로 방향별 MFE 분위수회귀
(HistGradientBoostingRegressor, base102 피쳐 패널, TRAIN 구간 2025-01-01~09-30만) **재학습**
→ 두 설정(tp_scale=9.0의 대칭 폭 산식 / tp_scale=9.0+sl_scale=1.5의 비대칭 산식)을 **고정한 채**
VAL 우선순위결합 포트폴리오 백테스트를 5회씩 반복. 모델 레시피·TRAIN 윈도우·피쳐 패널·설정값 전부
불변, `random_state`만 시드별로 다름. baseline과 하네스는 선행실험과 완전 동일 재사용
(`research_eth_omega461_exit_sweep_20260721.py` + `replay_omega4_6_1_greedy_router_20260706.py`).
VAL만, OOS 미실행, 라이브 파일 미변경.
`fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false`.

## 결과

### symmetric_scale9 (5 시드, 포트폴리오 no_gate/with_gate)

| 시드 | no_gate pnl% | no_gate mdd% | with_gate pnl% | with_gate mdd% |
|---:|---:|---:|---:|---:|
| 453827194 | 126.44 | -14.87 | 120.34 | -14.87 |
| 121952941 | 122.66 | -15.01 | 122.33 | -15.01 |
| 501601563 | 77.48 | -17.51 | 71.08 | -23.58 |
| 139411872 | 73.44 | -20.36 | 71.45 | -23.58 |
| 13643480 | 77.25 | -20.00 | 74.41 | -22.09 |
| **평균±표준편차** | **95.45±23.83** | **-17.55±2.35** | **91.92±24.05** | **-19.83±4.03** |
| baseline | 36.82 | -24.34 | 54.88 | -31.11 |
| **승(vs baseline)** | **5/5** | **5/5** | **5/5** | **5/5** |

### asymmetric_tp9_sl1.5 (5 시드)

| 시드 | no_gate pnl% | no_gate mdd% | with_gate pnl% | with_gate mdd% |
|---:|---:|---:|---:|---:|
| 453827194 | 135.66 | -15.51 | 167.91 | -15.51 |
| 121952941 | 132.07 | -15.51 | 159.56 | -22.22 |
| 501601563 | 94.91 | -24.03 | 99.39 | -30.83 |
| 139411872 | 92.48 | **-24.69**(baseline보다 악화) | 98.66 | -30.83 |
| 13643480 | 51.85 | **-24.34**(baseline와 사실상 동률, 근소 악화) | 56.00 | -30.83 |
| **평균±표준편차** | **101.40±30.63** | **-20.82±4.34** | **116.30±41.88** | **-26.04±6.23** |
| baseline | 36.82 | -24.34 | 54.88 | -31.11 |
| **승(vs baseline)** | **5/5** | **3/5** | **5/5** | **5/5** |

## 판정

**symmetric_scale9: 재현됨(4/5 기준 통과 — 실제로는 4개 지표 전부 5/5).** PnL은 5개 시드 전부
baseline을 이겼고(평균 +95.45%, 최저 시드도 +73.44%로 baseline +36.82%보다 뚜렷이 높음), MDD도
5개 시드 전부 baseline보다 낮았다(평균 -17.55%, 최악 시드도 -20.36%로 baseline -24.34%보다
낮음). with_gate 기준도 동일하게 4/5 이상(5/5) 통과. **단일 시드 우연이 아니라 진짜 재현되는
효과다.**

**asymmetric_tp9_sl1.5: 재현 안 됨(PnL만 강건, MDD는 기준 미달).** PnL은 5/5로 강건하지만(평균
+101.40%, 최저 시드도 +51.85%로 baseline보다 높음), **no_gate MDD는 3/5로 4/5 기준에 못 미친다** —
2개 시드(139411872: -24.69%, 13643480: -24.34%)가 baseline(-24.34%)과 사실상 동률이거나
근소하게 더 나쁘다(부호가 뒤집히는 파국적 실패는 아니지만, "MDD도 뚜렷이 개선"이라는 원래 주장을
5개 시드 전부가 뒷받침하진 않음). with_gate MDD는 5/5로 통과하지만, 오케스트레이터의 판정 기준
("4개 지표 각각 4/5 이상")을 엄격히 적용하면 no_gate MDD 하나가 기준 미달이라 **전체 주장은
재현 안 된 것으로 처리한다.**

## 결론

**symmetric_scale9(선행실험1의 base102 피쳐셋, `tp_scale=9.0`, 대칭 sl_ratio=0.5)은 시드 강건성
검증을 통과한 진짜 후보다.** 5개 진짜 무작위 시드 전부에서 baseline 대비 PnL·MDD가 no_gate·
with_gate 양쪽 다 개선됐다 — 이 세션이 반복 겪은 "단일시드 승자는 노이즈"(TabM HP 패턴, 최종보스
v2/v3의 VAL개선/OOS반전) 함정에 해당하지 않는다. 다만 이 지점(scale=9.0)은 선행실험1이 이미
지적했듯 TP 중앙값이 원래 라이브 고정폭(7.5%)에 근접해 "보유기간 단축/거래수 증가"라는 원래
문제는 거의 개선하지 못한다(baseline 대비 거래수·보유기간 차이가 미미) — 이건 별개 문제로 이미
닫혔으므로 여기서 다시 논하지 않는다. **OOS 실행 여부와 승격 판단은 요청대로 진행하지 않았고,
오케스트레이터의 판단으로 남긴다.**

**asymmetric_tp9_sl1.5는 "재현 안 됨, 시드 분산 노이즈"로 폐기한다.** PnL 개선 자체는 강건해
보이지만, 애초에 이 셀을 후보로 만든 조건("PnL과 MDD 동시 개선")의 MDD 축이 5개 시드 중 2개에서
지켜지지 않아 원래 주장 그대로는 신뢰할 수 없다.

## OOS 확인(1회, symmetric_scale9 한정)

시드 강건성 검증을 통과한 `symmetric_scale9` 하나만, VAL 검증에 쓴 것과 **동일한 N=5 시드**로
OOS(`research_eth_omega461_exit_sweep_20260721.py` 표준 윈도우, 2026-01-01~03-31)에서 **이번
한 번만** 확인한다. `asymmetric_tp9_sl1.5`는 이미 시드 강건성에서 폐기됐으므로 건드리지 않았다.

### 방법

스크립트: `scripts/research_eth_omega461_live_sltp_wide_calibration_oos_confirm_20260813.py`.
MFE 회귀모델 TRAIN 윈도우(2025-01-01~09-30)는 불변, OOS는 스코어링/백테스트에만 사용하고 학습에
전혀 섞지 않았다. **OOS를 실제로 열기 전에, 같은 파이프라인을 먼저 VAL에 돌려 이미 발행된
시드강건성 결과(위 표)와 정확히 일치하는지 자체 검증(assert, 상대오차 1e-6)했다** — 이 자체검증이
통과해야만 스크립트가 OOS 구간으로 진행하도록 하드코딩해서, 파이프라인 버그로 1회성 OOS 확인
기회를 낭비할 위험을 줄였다. 자체검증은 5개 시드 전부 정확히 일치해 통과했다.

**데이터 이슈 발견 및 패치**: OOS 프레임 로딩 중 `regime3_current_sensitive_wide24_*` 6개 컬럼에
연속된 95행(2026-02-28 16:05~23:55, 전체 25,633행 중 0.37%) NaN 갭이 있어
`hard._route_id()`(전문가 라우팅에 쓰는, non-finite 값을 전혀 허용 안 하는 함수)가 크래시했다 —
`research_eth_omega461_exit_sweep_20260721.py`의 `replay_exit_variant`도 같은 호출을 하므로,
이건 이 스크립트만의 문제가 아니라 현재 시점 `WIDE24_2026` 원본 데이터 자체의 결측(이 저장소에
동시에 여러 세션이 작업 중이라 다른 세션의 데이터 재생성 과정에서 생긴 것으로 추정 —
`research_eth_omega461_tpsl_floor_portfolio_check_20260728.py`의 `_truncated_pred_csv` 주석에
기록된 것과 같은 패턴). 해당 6개 컬럼만, 오직 과거 값으로만(순방향 채움, causal, 미래데이터 사용
없음) 채워 넣었다 — baseline과 candidate 양쪽이 갈라지기 전에 프레임 단계에서 동일하게 적용해서
어느 쪽에도 유불리가 없다. 방향/품질/진입타이밍 로직은 전혀 건드리지 않은, 결측치 하나 메운 것뿐인
데이터 위생 패치다(패치 없이는 baseline조차 계산 불가능했다).

### 결과 — VAL 5시드 vs OOS 5시드

| split | 시드 | no_gate pnl% | no_gate mdd% | no_gate trades | no_gate avg_hold | with_gate pnl% | with_gate mdd% |
|---|---:|---:|---:|---:|---:|---:|---:|
| VAL | baseline | 36.82 | -24.34 | 29 | 676.5 | 54.88 | -31.11 |
| VAL | 453827194 | 126.44 | -14.87 | 25 | 738.1 | 120.34 | -14.87 |
| VAL | 121952941 | 122.66 | -15.01 | 26 | 706.3 | 122.33 | -15.01 |
| VAL | 501601563 | 77.48 | -17.51 | 27 | 674.8 | 71.08 | -23.58 |
| VAL | 139411872 | 73.44 | -20.36 | 27 | 677.1 | 71.45 | -23.58 |
| VAL | 13643480 | 77.25 | -20.00 | 26 | 702.3 | 74.41 | -22.09 |
| **OOS** | **baseline** | **51.19** | **-15.48** | 24 | 783.3 | **46.29** | **-15.48** |
| OOS | 453827194 | -14.47 | -25.13 | 12 | 1558.3 | -5.28 | -17.09 |
| OOS | 121952941 | -11.07 | -24.29 | 12 | 1559.4 | -2.67 | -17.14 |
| OOS | 501601563 | -5.30 | -24.78 | 13 | 1440.9 | 4.09 | -17.33 |
| OOS | 139411872 | -11.50 | -21.32 | 12 | 1559.4 | -0.81 | -14.08 |
| OOS | 13643480 | 3.77 | -23.81 | 13 | 1664.6 | 12.61 | -17.33 |
| OOS | **승(vs baseline)** | **0/5** | **0/5** | - | - | **0/5** | **1/5** |

(전체: `tmp/research_20260813/omega461_live_sltp_wide_calibration_oos_confirm/report.json`,
`val_vs_oos_comparison.csv`.)

### 판정 — 완전한 반전(재현 안 됨)

**VAL에서 5/5로 완벽하게 재현됐던 우위가 OOS에서 정확히 반대로 뒤집혔다.** OOS baseline 자체는
견조하다(+51.19%/-15.48%, 이 구간 라이브 성과가 준수했다는 뜻). 하지만 `symmetric_scale9`는
5개 시드 중 **no_gate pnl 0/5, no_gate mdd 0/5, with_gate pnl 0/5, with_gate mdd 1/5**로
baseline을 거의 전 지표에서 못 이겼다 — no_gate pnl은 5개 중 4개가 마이너스(-5.3~-14.5%)이고
최선의 시드조차 +3.77%로 baseline +51.19%에 한참 못 미친다. MDD도 5개 시드 전부
-21.3~-25.1%로 baseline -15.48%보다 뚜렷이 나쁘다. 평균 보유기간도 VAL에서는 baseline과
비슷했지만(674~738바 vs 676바) OOS에서는 1441~1665바로 baseline(783바)보다 오히려 훨씬
길어졌다 — TP가 OOS 구간의 실제 가격움직임과 안 맞아 거의 청산이 안 되고 있다는 뜻으로 보인다.

이건 이 세션이 반복 확인한 "VAL 개선 → OOS 반전" 패턴(최종보스 v2/v3, 2026-08-01 Sigma3-1h
5-seed 감사 등)의 또 하나의 사례다 — 시드 강건성 검증만으로는 잡아낼 수 없는, VAL 구간 자체에
과적합된 신호였다는 뜻이다(시드를 5개나 바꿔도 VAL 안에서는 전부 이겼으니 "노이즈"는 아니었지만,
그 신호 자체가 VAL 레짐에 특화된 것이었다).

## 결론 (최종)

**symmetric_scale9는 OOS에서 완패했다 — 승격 후보에서 제외한다.** VAL 5-시드 강건성만으로는
"진짜 후보"와 "VAL 레짐 과적합"을 구분할 수 없었다는 게 이번 OOS 확인의 핵심 교훈이다. 지시대로
이 config에 대한 OOS 확인은 이번 1회로 종료하며, 재튜닝 후 재확인은 하지 않는다.
`eth_omega461_live_sltp_mfe_width_20260813`/`eth_omega461_live_sltp_asymmetric_tpsl_20260813`가
이미 내린 "SLTP 폭→거래수 감소 문제는 두 메커니즘으로 해결 안 되는 구조적 트레이드오프" 결론과
합쳐, 이번 side-finding(넓은-쪽 baseline 초과)도 OOS 기준으로는 실체가 없는 것으로 최종
정리한다.

## 산출물

- 스크립트: `scripts/research_eth_omega461_live_sltp_wide_calibration_seed_robustness_20260813.py`
  (VAL 시드강건성), `scripts/research_eth_omega461_live_sltp_wide_calibration_oos_confirm_20260813.py`
  (OOS 1회 확인, 자체검증 게이트+데이터갭 패치 포함)
- 결과: `tmp/research_20260813/omega461_live_sltp_wide_calibration_seed_robustness/{report.json,
  seed_robustness_summary_VAL.csv, ledger_{symmetric_scale9,asymmetric_tp9_sl1.5}_seed<seed>_VAL.csv}`,
  `tmp/research_20260813/omega461_live_sltp_wide_calibration_oos_confirm/{report.json,
  val_vs_oos_comparison.csv, ledger_OOS_symmetric_scale9_seed<seed>.csv}`
- 시드: `[453827194, 121952941, 501601563, 139411872, 13643480]`(무작위 추출, 등차수열 아님, VAL·OOS
  전부 동일 시드)
- 선행실험(인용): `docs/experiments/eth_omega461_live_sltp_mfe_width_20260813.md`,
  `docs/experiments/eth_omega461_live_sltp_asymmetric_tpsl_20260813.md`
