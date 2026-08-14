# Odyssey2 — 다중구간 확인 게이트 구축 및 소급 검증 (2026-08-14)

## 배경

2026-08-13 밤, ETH Omega4.6.1 라이브 모델(h48qual/zig075) 위에 얹는 post-entry 후보 6개(레짐
threshold×2, GBDT/TCN exit_head, 대기압력, risk-controlled exit fallback) 중 여러 개가 "VAL에서
개선 → 단일 OOS(2026-01~03-31)에서 반전"됐다. 같은 날 독립적으로 작성된
`docs/experiments/eth_val_oos_regime_mismatch_investigation_20260813.md`가 다른 4개 후보(최종보스
v2/v3, SLTP 폭 재보정, 멀티슬롯 MFE게이팅)에서도 같은 패턴을 발견해 "VAL(단일 3개월 저표본 창)이
사이징→threshold→신규후보까지 3중으로 재사용되는 선택편향 때문에 저정보 신호가 된다"고 결론짓고,
실용적 권고 2번에서 다음과 같이 명시했다:

> 앞으로 새 후보는 최소 VAL+OOS-Q1+OOS-Q2, 가능하면 2025 Q1~Q3까지 포함한 **4개 이상의, 상승·하락
> 방향이 섞인 독립 구간**에서 부호 일치를 확인하기 전엔 "확인됨"이라고 쓰지 않는다.

사용자가 이 권고를 실제로 구현하는 재사용 가능한 검증 모듈을 만들 것을 다음 단계로 선택했다(3개
옵션 중 추천옵션). 이 문서는 그 결과 — 모듈 설계, 자체검증(G0), 이미 기각된 2개 후보(대기압력,
risk-controlled)에 대한 소급 스트레스테스트 — 를 기록한다.

**중요**: 이 작업은 대기압력/risk-controlled의 판정을 재심하는 것이 **아니다**. 두 후보 다 이미
OOS-Q1 단일 확인만으로 결정적으로 기각됐다(`docs/model_contracts/odyssey2_eth_live_injection_contract_20260813.md`
실행 로그 #7/#8). 이 문서의 4단계는 새 모듈이 실제 후보 로직에도 올바르게 작동하는지 확인하는
스트레스테스트이며, 결과가 뭐가 나오든 원래 판정은 바뀌지 않는다(아래 "판정이 바뀌는가?" 절에서
직접 확인).

## 산출물

- `scripts/eth_omega461_multiwindow_confirmation_gate_20260814.py` — 재사용 모듈(`load_all_windows`,
  `verify_windows`, `align_frame_and_predictions`, `run_portfolio_variant`, `summarize_multiwindow`)
  + G0 자체검증·4단계 소급적용을 실행하는 `main()`.
- `tmp/causal_regen_20260516/eth_omega461_multiwindow_confirmation_gate_20260814/report.json` — 이
  문서의 모든 수치의 근거(직접 읽어 대조 완료).

## 1. 데이터 가용성 재확인 (1단계)

`EXT_PRED_DIR = tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/`의 h48qual/zig075 각각
`train_predictions_q{050,075}.csv`(n=78509, 2025-01-01 04:55~09-30 23:55),
`validation_predictions_q{050,075}.csv`(n=26490, 2025-10-01~12-31 23:25),
`oos_predictions_q{050,075}.csv`(n=55405, 2026-01-01~07-12 09:00)를 직접 재확인했다(값 일치, 재학습
없음). `research_eth_omega461_exit_sweep_20260721.load_frame`이 임의 날짜범위를 받는 범용 로더이고,
`BASE_2025`/`WIDE24_2025`·`BASE_2026`/`WIDE24_2026`이 이미 존재함도 재확인했다.

## 2. 모듈 설계

### 2-1. 6개 창

| 창 | 기간 | 데이터 소스 | oof | tier |
|---|---|---|---|---|
| `2025q1` | 01-01~03-31 | `train_predictions_*.csv` | True | context |
| `2025q2` | 04-01~06-30 | `train_predictions_*.csv` | True | context |
| `2025q3` | 07-01~09-30 | `train_predictions_*.csv` | True | context |
| `val` | 10-01~12-31 | `validation_predictions_*.csv` | True | val |
| `oos_q1` | 2026-01-01~03-31 | `oos_predictions_*.csv` | False | oos_confirm |
| `oos_q2` | 2026-04-01~06-30 | `oos_predictions_*.csv` | False | oos_confirm |

**창 경계 발견사항(사전에 가정하지 않고 직접 확인)**: `val`/`oos_q1`은 `sweep.VAL_START/VAL_END`,
`OOS_START/OOS_END`를 시각 접미사 없이 그대로 쓴다 — 이미 발표된 모든 참조값(asymmetric_tabm_liveatr
46.59/77.31/93.27/67.25 등)이 정확히 그 방식(날짜 문자열이 자정으로 해석되어 마지막 날 대부분이
잘림)으로 계산됐기 때문이다. `2025q1/q2/q3`는 `"23:59:59"`를 붙인다 —
`audit_omega4_6_1_phase1_robustness_20260707.load_2025_quarter_components`의 `quarters` 튜플이
명시적으로 그렇게 정의했고, 실제로 확인해보니 이게 사소한 차이가 아니었다(2025-Q1만 해도
시각 접미사 없이는 25631행, 있으면 25918행 — 287행/거의 하루 차이, 없이 계산하면 공개된
28.54%/-20.62%/19건 참조값을 재현하지 못한다). `oos_q2`는 형제 창 `oos_q1`과 동일하게 시각 접미사
없이 처리한다 — 이 모듈을 만들며 `WIDE24_2026` 오버레이 파일이 정확히 2026-06-30 00:00:00에서
끝난다는 걸 발견했는데(기존에 알려진 "95-bar 중간 갭"과 같은 클래스의 커버리지 한계, 이번엔
파일 끝단), `"23:59:59"`를 썼다면 마지막 날 287bar에 route probability가 없어 드롭해야 했을
것이고, 시각 접미사 없는 경계는 정확히 같은 최종 프레임에 도달하면서 그 드롭 단계 자체를
생략한다.

### 2-2. 창 로딩 검증 (`load_all_windows`/`verify_windows`)

각 창의 `frame`과 각 컴포넌트의 원본 예측 CSV(전체, 자르지 않음) 교집합을 직접 계산해 확인했다.
결과(`report.json` `window_verification`):

| 창 | frame 행수 | route NaN 드롭 | h48qual 교집합 비율 | zig075 교집합 비율 |
|---|---:|---:|---:|---:|
| 2025q1 | 25,918 | 0 | 99.75% | 99.75% |
| 2025q2 | 26,204 | 0 | 100.00% | 100.00% |
| 2025q3 | 26,483 | 0 | 99.89% | 99.89% |
| val | 26,209 | 0 | 100.00% | 100.00% |
| oos_q1 | 25,538 | 95 | 100.00% | 100.00% |
| oos_q2 | 25,921 | 0 | 100.00% | 100.00% |

**발견 1(예상되고 무해함)**: `train_predictions_*.csv`가 2025q1에서 66행(2025-01-01~01-23에 흩어짐),
2025q3에서 30행(2025-07-22 13:50~16:15, 단일 연속 구간)이 기저 피처 프레임보다 적다. 이는 버그가
아니라 `research_eth_omega461_exit_sweep_20260721.prep_component`가 이미 문서화한 "predictions are
the authoritative row set"(원 학습 파이프라인이 워밍업/NaN 피처 등의 이유로 일부 행을 자름) 현상과
같은 종류다 — `align_frame_and_predictions`/`prep_component` 둘 다 교집합으로 자동 대응하도록
설계돼 있어 하위 계산에 영향 없다(아래 G0b가 이를 직접 재확인). 그래서 게이트 기준을 "100% 일치"가
아니라 "99% 이상 고커버리지"로 뒀다(99.75%/99.89%는 충분히 여유 있음).

**발견 2(이미 알려진 것의 재확인)**: OOS-Q1에서 95bar route probability 드롭 — 대기압력/risk-controlled
스크립트가 이미 발견한 2026-02-28 16:05~23:55 WIDE24_2026 커버리지 갭과 동일.

### 2-3. 사전등록 판정 기준 (결과 확인 전 확정, 변경 없음)

1. VAL 역할 불변 — 호출하는 쪽(각 후보 스크립트)이 기존처럼 주 튜닝/선택 창으로 사용.
2. **공식 OOS 확인은 oos_q1과 oos_q2를 한 번에 함께 여는 단일터치**(순차 아님) — 둘 다 with_gate
   PnL이 baseline 대비 비악화(또는 완화기준의 MDD 3%p 허용)여야 통과. 하나만 통과하면
   부호불일치로 기각.
3. `2025q1/q2/q3`는 참고용 맥락으로만 표에 표시 — pass/fail 기준에 넣지 않음(TRAIN 기간이라
   in-sample). OOS 개방 여부를 막지 않음.

`summarize_multiwindow(baseline_results, candidate_results, mdd_slack_pp=0.0|3.0)`가 이 기준을
코드로 구현한다 — 각 창의 no_gate/with_gate 계산 자체는 호출하는 쪽 책임(후보마다 개입 지점이
다르므로 만능 evaluate 함수를 만들지 않았다), 이 함수는 표 생성과 pass/fail 판정만 담당한다.

## 3. G0 자체검증 결과

### 3-1. G0a — asymmetric_tabm_liveatr, VAL·OOS-Q1

`run_portfolio_variant`(이 모듈의 새 함수, `greedy.prepare_component`/`portfolio._prepare_component_val`
+ `greedy.greedy_replay`+`portfolio._ledger_metrics`+`mfe_width._duration_gated` 무수정 재사용)로
재현:

| 창 | 지표 | 실제 | 참조값 | 일치 |
|---|---|---:|---:|---|
| val | no_gate | 46.59%/-21.70%/35건 | 46.59%/-21.70%/35건 | True |
| val | with_gate | 77.31%/-21.76%/26건 | 77.31%/-21.76%/26건 | True |
| oos_q1 | no_gate | 93.27%/-15.48%/24건 | 93.27%/-15.48%/24건 | True |
| oos_q1 | with_gate | 67.25%/-15.48%/19건 | 67.25%/-15.48%/19건 | True |

참조값 출처: `tmp/causal_regen_20260516/eth_omega461_risk_controlled_exit_fallback_20260814/report.json`
(`g0`/`g0b`/OOS 섹션). 4개 전부 정확히 일치(`report.json` `g0a_val_oosq1.pass=true`).

### 3-2. G0b — 2025 Q1/Q2/Q3, 두 단계 검증

**(i) baseline_both_original — 두 개의 독립 코드 경로로 검증**: 이 모듈의
`run_portfolio_variant` 경로와, `audit_omega4_6_1_phase1_robustness_20260707.load_2025_quarter_components`
+ `test_omega4_6_1_drop_h48qual_20260706._metrics(apply_gate=True)`를 직접 호출하는 독립 경로.

| 분기 | 이 모듈 | 독립 레거시 경로 | 참조값(2026-07-07) | 일치(둘 다) |
|---|---:|---:|---:|---|
| 2025-Q1 | 28.54%/-20.62%/19건 | 28.54%/-20.62%/19건 | 28.54%/-20.62%/19건 | True |
| 2025-Q2 | 39.99%/-10.82%/15건 | 39.99%/-10.82%/15건 | 39.99%/-10.82%/15건 | True |
| 2025-Q3 | -9.73%/-44.37%/19건 | -9.73%/-44.37%/19건 | -9.73%/-44.37%/19건 | True |

참조값 출처: `tmp/causal_regen_20260516/omega4_6_1_phase1_robustness_20260707/result.json`
`rolling_walk_forward_diagnostic`. 6개 전부(창×경로) 정확히 일치.

**(ii) asymmetric_tabm_liveatr — 신규 수치(참조값 없음, 이하 신규 참고용)**:

| 분기 | no_gate | with_gate |
|---|---:|---:|
| 2025-Q1 | 97.70%/-20.62%/28건 | 44.98%/-20.62%/20건 |
| 2025-Q2 | 106.45%/-13.23%/31건 | 31.49%/-15.85%/19건 |
| 2025-Q3 | -46.26%/-56.94%/38건 | -18.87%/-43.49%/30건 |

신뢰성은 (i)의 로딩 메커니즘 성공 재현으로 대신한다(같은 로딩 경로, h48qual 번들만 다름).
2025-Q3(강한 상승 분기, `eth_val_oos_regime_mismatch_investigation_20260813.md`가 이미 지목한
"zig075 SHORT 우위가 유일하게 반전되는 분기")에서 baseline_both_original·asymmetric_tabm_liveatr
둘 다 큰 폭의 손실이라는 점이 눈에 띈다 — 그 문서의 결론("숏 우위의 정체는 방향성 베타")과 방향이
일치한다.

### 3-3. G0c — `_metrics(apply_gate=True)` ↔ `_duration_gated` 수학적 동치성

VAL asymmetric_tabm_liveatr 렛저(26건) 하나로 직접 대조:

```
legacy (test_omega4_6_1_drop_h48qual_20260706._metrics): pnl=77.314939%, mdd=-21.756946%, trades=26
new    (research_eth_omega461_live_sltp_mfe_width_20260813._duration_gated): pnl=77.314939%, mdd=-21.756946%, trades=26
```

소수점 6자리까지 정확히 일치(`equivalent=true`). 두 함수는 하나가 hit 거래를 0%수익 스텝으로
유지하고 다른 하나는 아예 제거한다는 설계 설명 차이가 있을 뿐, `np.cumprod`에서 `(1+0)`을 곱하는 것과
행 자체를 빼는 것이 수학적으로 동일한 결과를 내므로 항상 동치다.

### 3-4. 부가 검증 — 세 독립 코드 경로의 베이스라인 렛저 바이트 단위 일치

G0/4단계 과정에서 baseline(asymmetric_tabm_liveatr) 렛저를 세 가지 독립 코드 경로로 계산했다:
(a) `run_portfolio_variant`(순수 `greedy.greedy_replay`), (b)
`qp_mod.greedy_replay_queue_pressure(..., queue_pressure_threshold=0.95)`(대기압력 후보 스크립트의
퇴화 모드), (c) `rc_mod.greedy_replay_risk_controlled(..., tau=TAU_NEVER_SWITCH)`(risk-controlled
후보 스크립트의 퇴화 모드). 6개 창 전부에서 (a)==(b)==(c)가 `diff`로 바이트 단위까지 정확히
일치함을 확인했다(oos_q2는 (a) 렛저를 별도로 안 만들었지만 (b)==(c)는 확인). 이는 이 모듈의 창
로딩·정렬이 기존에 검증된 두 후보 스크립트의 자체 로딩 방식과도 완전히 호환됨을 보여주는 강한
교차검증이다.

**G0 종합**: `gate_pass_g0=true`(전체 통과) — 4단계로 진행.

## 4. 4단계 — 대기압력·risk-controlled 소급 스트레스테스트

### 4-1. 대기압력(threshold=0.80)

`scripts/research_eth_omega461_queue_pressure_exit_threshold_20260814.py`의
`greedy_replay_queue_pressure`/`_zig075_pressure_mask`를 그대로 import해 6개 창에 적용:

| 창 | tier | baseline no_gate | baseline with_gate | 후보 no_gate | 후보 with_gate | with_gate 통과 |
|---|---|---:|---:|---:|---:|---|
| 2025q1 | context | 97.70%/-20.62%/28 | 44.98%/-20.62%/20 | 121.46%/-20.62%/30 | 59.42%/-20.62%/21 | PnL True / MDD False |
| 2025q2 | context | 106.45%/-13.23%/31 | 31.49%/-15.85%/19 | 80.68%/-21.78%/33 | 15.08%/-25.76%/21 | PnL False / MDD False |
| 2025q3 | context | -46.26%/-56.94%/38 | -18.87%/-43.49%/30 | -25.28%/-53.40%/37 | 0.48%/-45.87%/31 | PnL True / MDD False |
| **val** | val | 46.59%/-21.70%/35 | 77.31%/-21.76%/26 | 52.77%/-21.70%/38 | 95.46%/-20.69%/30 | **PASS** |
| **oos_q1** | oos_confirm | 93.27%/-15.48%/24 | 67.25%/-15.48%/19 | 59.08%/-15.48%/27 | 37.55%/-15.48%/22 | **FAIL**(-44.2% 상대하락) |
| **oos_q2** | oos_confirm | -9.55%/-20.76%/13 | -12.69%/-20.76%/10 | -14.70%/-22.45%/16 | -1.73%/-17.13%/13 | **PASS**(단독으로 보면) |

교차검증: VAL baseline no_gate, OOS-Q1 baseline no_gate, OOS-Q1 후보 no_gate(59.08%/-15.48%/27건)
전부 기존 `tmp/causal_regen_20260516/eth_omega461_queue_pressure_exit_threshold_20260814/report.json`과
정확히 일치(`cross_checks_vs_already_published` 전부 `true`). OOS-Q1 with_gate 37.55%도
`docs/experiments/eth_omega461_relaxed_gate_rescoring_20260814.md`의 기존 수치와 일치.

**단일터치 판정: oos_q1 FAIL, oos_q2 PASS → 부호불일치 → REJECTED_SIGN_MISMATCH**(`mdd_slack_pp=0.0`과
`3.0` 둘 다 동일 결과).

### 4-2. risk-controlled(eps_frac=0.90, τ̂=0.9995394945144653 고정, 재보정 없음)

`scripts/research_eth_omega461_risk_controlled_exit_fallback_20260814.py`의
`greedy_replay_risk_controlled`/`_gbdt_portfolio_fallback`을 그대로 import해 6개 창에 적용(GBDT
fallback 모델·τ̂ 전부 VAL에서 이미 확정된 값 그대로, oos_q2에도 재보정 없이 고정 적용):

| 창 | tier | baseline no_gate | baseline with_gate | 후보 no_gate | 후보 with_gate | with_gate 통과 |
|---|---|---:|---:|---:|---:|---|
| 2025q1 | context | 97.70%/-20.62%/28 | 44.98%/-20.62%/20 | 119.33%/-20.62%/31 | 59.10%/-20.62%/23 | PnL True / MDD True |
| 2025q2 | context | 106.45%/-13.23%/31 | 31.49%/-15.85%/19 | 93.46%/-20.87%/37 | 20.67%/-20.85%/24 | PnL False / MDD False |
| 2025q3 | context | -46.26%/-56.94%/38 | -18.87%/-43.49%/30 | -10.70%/-53.92%/46 | 3.63%/-41.73%/36 | PnL True / MDD True |
| **val** | val | 46.59%/-21.70%/35 | 77.31%/-21.76%/26 | 66.41%/-21.70%/34 | 85.50%/-23.59%/27 | PnL True / MDD False(strict) |
| **oos_q1** | oos_confirm | 93.27%/-15.48%/24 | 67.25%/-15.48%/19 | 21.18%/-28.70%/25 | 4.77%/-28.70%/20 | **FAIL**(-92.9% 상대하락) |
| **oos_q2** | oos_confirm | -9.55%/-20.76%/13 | -12.69%/-20.76%/10 | -8.83%/-20.76%/13 | -12.01%/-20.76%/10 | **PASS**(단독으로 보면, MDD는 부동소수점 오차 수준(~1e-14)까지 동일) |

교차검증: VAL baseline no_gate, OOS-Q1 baseline no_gate, OOS-Q1 후보 no_gate(21.18%/-28.70%/25건)
전부 기존 `tmp/causal_regen_20260516/eth_omega461_risk_controlled_exit_fallback_20260814/report.json`과
정확히 일치.

**단일터치 판정: oos_q1 FAIL, oos_q2 PASS → 부호불일치 → REJECTED_SIGN_MISMATCH**(`mdd_slack_pp=0.0`과
`3.0` 둘 다 동일 결과).

### 4-3. 핵심 발견 — 이 스트레스테스트가 방법론 자체를 실증적으로 뒷받침한다

**대기압력·risk-controlled 둘 다 OOS-Q1은 확실히 반전됐지만 OOS-Q2만 단독으로 봤다면 오히려
"통과"로 보였을 것**(대기압력: -1.73% > -12.69%; risk-controlled: -12.01% > -12.69%, MDD는
사실상 동일). 만약 이 서브프로젝트가 애초에 "OOS-Q2 하나만" 확인하는 방식이었다면 두 후보 다
잘못 채택됐을 것이다 — OOS-Q1과 OOS-Q2가 서로 부호가 갈리는 이 사례 자체가
`eth_val_oos_regime_mismatch_investigation_20260813.md`가 예측한 "3개월 단일 OOS 창도 그 자체로는
약한 증거"라는 진단을 새 데이터로 직접 재현한다. 단일터치(oos_q1 AND oos_q2)가 아니었다면 놓쳤을
반증이다.

## 5. 이 모듈로 과거 판정이 바뀌는가?

**아니다.** `report.json`의 `step4_retroactive_stress_test.this_module_changes_prior_verdict=false`.
두 후보 다 새 모듈의 엄격(`mdd_slack_pp=0`) 기준과 완화(`mdd_slack_pp=3`) 기준 둘 다에서
`REJECTED_SIGN_MISMATCH`로 나왔고, 이는 이미 확정됐던 기각과 동일한 결론이다. 예상된 결과였다 —
이 4단계의 목적은 재심이 아니라 모듈 자체의 스트레스테스트였다(§4-3의 발견은 오히려 기존 기각
판정을 더 강하게 뒷받침한다).

## 6. 준수 확인

- `fresh_forward_bar_by_bar=true`, `trade_ledgers_used_as_input=false`,
  `saved_parent_exit_timestamps_used=false`, `future_rows_used_for_entry=false` — `report.json`에
  명시, 모든 계산이 `greedy.greedy_replay`/`greedy_replay_queue_pressure`/`greedy_replay_risk_controlled`
  (전부 기존의 causal bar-by-bar 재생 루프, 무수정 재사용)를 통과.
- `research_eth_omega461_exit_sweep_20260721.py`, `replay_omega4_6_1_greedy_router_20260706.py`,
  `research_eth_omega461_exit_head_portfolio_asymmetric_20260813.py`,
  `research_eth_omega461_live_sltp_mfe_width_20260813.py`,
  `research_eth_omega461_queue_pressure_exit_threshold_20260814.py`,
  `research_eth_omega461_risk_controlled_exit_fallback_20260814.py` — 전부 import만, `git diff` 0줄.
- `trading_bot.py`/`trading_bot_modules/omega4_6_1_live.py`/`trading_bot_modules/runtime_config.py`/`.env`
  — `git diff` 0줄(세션 전후 직접 확인).
- 재학습 없음, GPU 불필요 — 전부 기존 `train_predictions`/`validation_predictions`/`oos_predictions`
  CSV 재사용.

## 7. 앞으로의 사용법

신규 후보(예: 문헌 3위 Conformal Kelly)는:
1. `load_all_windows()`로 6개 창을 한 번 로드.
2. VAL에서 자체 게이트(원 기준 또는 완화기준) 통과 여부 판단 — 이 단계는 기존과 동일, 이 모듈이
   대신하지 않음.
3. VAL 통과 시, `oos_q1`+`oos_q2`를 **같은 실행에서 함께** 열어 후보 고유 로직으로 no_gate/with_gate
   계산.
4. `summarize_multiwindow(baseline_results, candidate_results, mdd_slack_pp=...)`로 표·판정 생성.
5. `2025q1/q2/q3`도 계산해 표에 포함(참고용, 판정 제외).
