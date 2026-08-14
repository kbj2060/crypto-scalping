# ETH Omega4.6.1 Conformal Kelly 사이징 스케일 (2026-08-14, Odyssey2 #9)

상태: `tested_negative_closed` — **VAL 사전등록 게이트(원기준·완화기준 둘 다)를 그리드 3후보
전부 통과 실패**로 OOS는 열지 않았다(단일터치 원칙에 따라 미실행). 문헌스카우팅(#6) 3위 후보
(사이징 축)를 종결한다.

## 배경

`docs/experiments/eth_omega461_post_entry_literature_scouting_20260814.md`(Odyssey2 #6)가 3위로
랭킹한 **Conformal Kelly**(Ryan, arXiv:2608.01494, 2026-08-02)를 구현한다. 1위(대기압력, #7)·
2위(Risk-Controlled Post-Processing, #8)는 둘 다 **exit-timing 축**이었고 둘 다 VAL 통과 후
OOS-Q1 단일창에서 반전했다. 이번 3위는 **사이징 축**(포지션 크기 조절)으로 완전히 다른 레버다.

이 실험 직전에 이 세션이 `scripts/eth_omega461_multiwindow_confirmation_gate_20260814.py`(다중구간
확인 게이트)를 새로 구축했다 — `docs/experiments/eth_val_oos_regime_mismatch_investigation_20260813.md`가
"OOS-Q1 단일창은 약한 증거"라고 지적한 데 더해, 대기압력·risk-controlled 둘 다를 이 새 게이트로
소급 재확인한 결과 **OOS-Q2만 봤다면 통과처럼 보였을 결과**가 실제로 나왔다(직접 증거,
`docs/model_contracts/odyssey2_eth_live_injection_contract_20260813.md`의 "방법론 변경" 항목).
**이 실험부터 공식 OOS 확인은 OOS-Q1+OOS-Q2를 한 실행에서 함께 여는 단일터치**다 — 둘 다
with_gate PnL 비악화 필요, 하나만 통과하면 `REJECTED_SIGN_MISMATCH`.

다중구간 게이트 모듈은 "baseline-shape 개입"(레짐threshold, exit_head 교체) 위주로 설계돼
`evaluate` 로직 자체는 사이징(margin_fraction) 개입에 맞지 않는다 — 이 실험은 그 모듈의 창
로더(`load_all_windows`)와 판정 헬퍼(`summarize_multiwindow`)만 재사용하고, margin 스케일링을
적용하는 evaluate 함수는 새로 작성했다(`research_eth_omega461_regime_specific_quality_threshold_
20260813.evaluate_component`를 참고한 변형).

## 1단계 — 논문 메커니즘 (WebFetch로 PDF 원문 직접 확인)

arXiv:2608.01494 PDF(30쪽)를 직접 fetch해 텍스트를 추출하고 정독했다(저자: Robert Jacob Ryan,
ACSAthens; 2026-08-02 v1). 8개 ETF(SPY/QQQ/DIA/MDY/GLD/SLV/USO/DBC) 일봉 모멘텀 회귀 세팅이다.

### (a) 정확한 공식

```
sigma_hat_{i,t} = q_eff_{i,t} / z_{1-alpha/2},     z_{1-alpha/2} = 1.2816 (논문이 스스로 명시한
                                                     "disclosed inconsistency" — alpha=0.25면
                                                     이론값은 1.1503인데 이전 alpha=0.20의 z값이
                                                     그대로 남아있음. 그로스캡이 항상 바인딩돼
                                                     스케일이 사실상 inert이므로 영향 없다고 논문이
                                                     직접 확인)
f_{i,t} = kappa * mu_hat_{i,t} / sigma_hat_{i,t}^2,  kappa = 0.15, 자산별 ±0.75 윈저화

q_eff_{i,t} = (q_roll_{i,t})^(1-lambda) * (q_anchor_{i,t})^lambda,  lambda = 0.3
  q_roll   = 최근 W=500개 "도착 완료"(landed) 비순응점수(nonconformity score)의 (1-alpha) 경험분위수
             s_{i,t} = |R^(21)_{i,t} - mu_hat_{i,t}|  (21일 선행 수익합의 절대잔차)
  q_anchor = 지금까지 도착 완료된 전체 점수의 (1-alpha) 분위수, 21행마다 재계산 후 그 사이는 정지(stale)
```

### (b) "구간은 모델앙상블이 아니라 자산별 롤링 분위수가 최고"— 문헌스카우팅 요약 정확성 확인

**정확함.** 논문 §6.1(Locally adaptive conformal prediction)에서 변동성스케일 비순응점수/Adaptive
Conformal Inference/recency-weighted CP/asymmetric CQR/Mondrian·pooled calibration 등 "구간을 더
빠르게 적응시키는" 장치 전부가 손실(-0.7~-5.3pp/yr)을 낸다. 특히 §6.5(Epistemic model
disagreement)는 5개 호라이즌 컴포넌트의 분산(모델 앙상블 불일치와 정확히 같은 개념)을 Kelly
분모에 추가하는 실험을 직접 수행했고 — "평균 앙상블 불일치는 평균 conformal 반폭의 0.6~2.5%에
불과"(SPY 0.00053 vs 0.0365, USO 0.00297 vs 0.1187)해 **채택하지 않았다**("despite +0.1pp"). 이는
이 프로젝트가 Odyssey2 #2(앙상블 불일치→사이징 피처, `std≈0.003`으로 null result)에서 독립적으로
발견한 것과 정확히 같은 결론이다 — 우연이 아니라 두 프로젝트 모두에서 "모델 앙상블 분산은
예측불확실성의 극히 일부"라는 같은 현상.

### (c) "development window 성공이 2022+ 진짜 OOS에서 부분적으로만 재현" — 정확한 맥락 (문헌스카우팅 요약의 오류 수정)

문헌스카우팅 문서는 이를 "40개 설정 중 다수가 pre-registered holdout에서 저조"라고 요약했으나,
**원문을 직접 대조한 결과 이 요약은 부정확하다.** 논문의 "40"이라는 숫자는 전부 **DEV
윈도우(2016-2021, 에이전트 탐색 창) 안**에서 나온다: (i) miscoverage 기반 레버리지-컷 다이얼(이번
실험이 재사용하는 "구간폭→사이징 스케일" 메커니즘과는 **다른** 메커니즘)의 "40+ configurations"
전부 기각, (ii) 그 다이얼의 유의성을 검증한 40-way circular-shift 플라시보 테스트. **진짜
사전등록 true-OOS("lockbox", 2022-01-01~2024-09-20, 683일, DEV 작업 시작 전에 봉인)는 config
2개(A/B, predict_tail 변형 포함 4개 숫자조합)만 테스트했고 — 둘 다 저조했다**: DEV 연성장률
28.13%/25.52%(재등록 후 수정치)가 lockbox에서 8.47%/7.01%로 "development 값의 약 30%"로
급락, 11개 비교 대상 중 Sharpe·Calmar 둘 다 **꼴찌**. 반면 **conformal 커버리지 자체는 거의
그대로 전이**됐다(0.745 실현 vs 0.750 목표). 논문 §12 결론 원문: *"marginal calibration
transferred out of sample and the economic value of the sizing rule did not."*

이것은 문헌스카우팅이 암시한 "여러 설정 중 다수가 실패"보다 **더 정밀하고 더 강한 경고**다 —
사후선택(multiple-comparison) 여지가 전혀 없는 단 2개의 사전등록 설정조차, 그것도 "캘리브레이션은
정직하게 전이됐다"는 최선의 조건에서도, Kelly 사이징의 경제적 가치는 증발했다. 이 프로젝트가 오늘
밤 반복 관찰한 "VAL 승리 → OOS 반전"과 **같은 계열의, 그러나 원인이 다른** 위험 신호로 취급한다
(원인: DEV의 결과가 상당 부분 2020년 코로나 랠리 한 달에 쏠려 있었고 — "2020 growth +0.667 vs
6-year mean 0.285" — 그로스캡이 거의 항상 바인딩돼 그 특정 레짐에서만 유리한 구조였다는 것이
논문 자신의 사후 해석).

## 2단계 — 이 프로젝트 재해석 설계

원 논문은 "예측 수익률(mu_hat) vs 실현 수익률" 회귀 프레임인데, 이 프로젝트는 분류
(direction/quality/exit_head, 전부 동결) + 별도 사이징 GBM(`risk_sidecar.pkl`) 구조라 mu_hat이
따로 없다.

**채택한 설계**: 두 라이브 사이드카 pkl(h48qual/zig075)을 직접 로드해 확인한 결과
`risk_target_mode="net"`, `target_mae_penalty=0.0` — 즉 사이징 GBM의 `score` 출력은 **정확히**
`net_per_notional`(단위 notional당 실현 거래손익, 모든 리플레이 렛저에 이미 기록됨)을 예측하도록
학습돼 있다. 이는 논문의 mu_hat/R_H 잔차와 단위까지 일치하는 자연스러운 대응을 제공한다:

```
비순응점수(닫힌 거래 k) = | net_per_notional_k(실현) - score_k(진입 시점 예측) |
```

**기각한 대안**: mu_hat/sigma_hat^2 Kelly 공식을 처음부터 재도출(사이징 GBM의 이미 학습된 edge
추정을 새 Kelly 분자와 이중계산하게 되고, "기존 사이징 모델 출력을 base로 두고 스케일만 얹는다"는
과제 지시와 충돌해 기각).

**롤링 이력**: 논문의 W=500(일) 롤링 구간은 이 프로젝트에 그대로 못 쓴다 — VAL 한 분기 전체
거래수가 20~40건대다. 논문 §5.1 자체가 "느릴수록 좋다"(완전히 얼린 σ조차 최선의 롤링판 대비
1.5pp만 손해)는 것이 핵심 발견이므로, **롤링(q_roll) 레그를 생략하고 확장 앵커(q_anchor)만
사용**하는 것을 "W를 프로젝트가 허용하는 한 가장 크게 키운 논문 자신의 극한"으로 채택했다(임의
단순화가 아니라 논문 결론의 직접 연장).

**컴포넌트별 완전 분리(풀링 금지)**: h48qual/zig075를 "자산"으로 보고 롤링 이력을 절대 섞지 않는다
— 논문 §6.1의 가장 강한 실증 결론(자산 간 풀링은 전부 손실)을 그대로 따름.

**정확한 스케일 함수**:

```
q_eff(t)      = 닫힌 거래 잔차의 75th pct(alpha=0.25, 논문과 동일), causal: exit_timestamp < t인
                거래만, [VAL-이전 캘리브레이션셋] ∪ [이 실행 자체가 지금까지 실현한 거래]
q_ref         = q_eff를 VAL 시작(2025-10-01)에 딱 한 번 고정 — "느릴수록 좋다"를 극한까지 밀어
                스케일=1.0의 기준점 자체도 다시 앵커링하지 않음
kelly_scale(t)= clip( (q_ref / max(q_eff(t), 1e-6))^2, scale_floor, scale_cap )
                지수=2는 논문의 f∝mu/sigma^2, sigma∝q_eff(선형) → f∝1/q_eff^2에서 그대로 채택(그리드 아님)
margin_fraction_scaled[i] = margin_fraction_raw[i] * kelly_scale(timestamp_i)
```

`(scale_floor, scale_cap)`만 그리드 축(3후보, log공간 대칭): narrow(0.85,1.20) / medium(0.70,1.40)
/ wide(0.50,2.00).

## 3단계 — CLAUDE.md Futures Risk Sizing Contract 준수

`scripts/train_eval_omega4_2_risk_sidecar_20260622.py`의 `_risk_margins(...)`를 직접 읽고
확인했다: 이 함수는 `margin`(=margin_fraction) 배열을 floor/cap 클립까지 마친 최종형으로
그대로 반환한다. `research_eth_omega461_exit_sweep_20260721.replay_exit_variant`에서
`row_notional = row_margin * row_leverage`(정확히 `notional = margin_fraction * leverage`),
그리고 `notional_scaled_sltp=True`일 때만 `take_profit = base_tp * row_notional`(정확히
`take_profit = tp_price_move * notional`)이 된다 — **두 라이브 사이드카(h48qual/zig075) pkl을
직접 로드해 `notional_scaled_sltp=False`임을 확인**했으므로, TP/SL은 애초에 notional을 다시
읽지 않는다(레버리지 이중계산 경로 자체가 없음, 검증됨). 개입 지점은
`evaluate_component`류 함수의 `margin = base_sweep.rs._risk_margins(...)` 직후 —
`margin_fraction_scaled = margin_fraction_raw * kelly_scale`을 그 자리에 삽입했다. `leverage`는
전혀 건드리지 않는다.

## 4단계 — 구현

- 신규 스크립트: `scripts/research_eth_omega461_conformal_kelly_sizing_scale_20260814.py`.
- `research_eth_omega461_exit_sweep_20260721.prep_component`의 이름 바꾼 복사본
  (`_prep_component_with_score`)을 만들어 `score`(사이징 GBM 예측치) 필드 하나만 추가 반환 —
  direction/quality/exit_head 로직은 전혀 건드리지 않음.
- 캘리브레이션 창: **2025 Q1~Q3(VAL 이전)만**으로 초기 캘리브레이션셋을 구성(다중구간 게이트의
  "context" 창 재사용, `oof=True`/`train_predictions_*.csv`) — 이 세 창 자체는 스케일=1.0
  고정(자기 자신을 캘리브레이션해 스케일링하는 순환성 회피, 라이브 봇의 콜드스타트와 동일 원리).
  2025q1→q2→q3→VAL→OOS-Q1→OOS-Q2가 달력상 완전히 연속(캘린더 갭 없음)이므로, VAL부터는 하나의
  연속된 causal walk-forward로 취급 — 이전 창에서 실현된 거래이력이 다음 창으로 그대로 이월된다
  (저장된 과거 원장 파일을 읽는 게 아니라 **이 실행 자신이 방금 생성한** 원장을 이월하는 것 —
  `report.json`의 `kelly_calibration_uses_this_runs_own_fresh_trade_history=true`로 별도 표시).

### 인과성 관련 발견 및 정정 (설계 중 발견, 은폐하지 않음)

최초 구현에서 두 가지 문제를 직접 발견해 수정했다:

1. **창-내부 미갱신 버그**: 최초 구현은 창이 바뀔 때만 캘리브레이션 풀을 갱신해, VAL 구간
   전체에서 스케일이 상수 하나로 고정되는 버그가 있었다(로그로 직접 발견: `raw_ratio_range=
   [1.000,1.000]`). 원인은 "이번 창 자신의 거래가 닫히는 시점"을 풀 갱신에서 빠뜨린 것 —
   해당 창의 baseline(스케일=1.0) 렛저를 스케일 계산 **전에** 미리 풀에 합쳐 넣도록 고쳤다(거래
   시점 자체는 스케일과 무관하다는 아래 사실을 이용, 정정 후 재실행: `raw_ratio_range=[0.923,
   1.044]` 등 정상적으로 창 내부에서도 변동 확인).
2. **exit_head가 notional을 입력 피처로 쓴다는 사실 확인**: 처음엔 "notional_scaled_sltp=False면
   margin 스케일링이 진입/청산 타이밍에 전혀 영향 없다"고 가정했는데, `replay_exit_variant`가
   exit_head 모델에 넘기는 position-state 피처 목록(`notional`,`leverage`,`notional*leverage`
   포함, `notional_scaled_sltp` 플래그와 무관하게 항상 포함)을 직접 읽고 이 가정이 **틀렸음을
   발견**했다 — margin 스케일이 exit_head 확률 자체를 바꿔 청산 타이밍이 미세하게(bar 단위로)
   달라질 수 있다. 직접 측정한 결과(VAL, narrow 후보 예시): **zig075는 baseline과 완전히
   동일**(진입·청산 0건 차이, 모든 스코어드 창), **h48qual은 OOS-Q2 완전 동일, OOS-Q1은 63건 중
   2건만 4bar 이내로 청산 이동, VAL은 63건 중 진입 33건/청산 35건이 baseline과 다름**(같은
   거래수 63건이지만 재배치). 이는 미래정보 누설이 **아니다**(캘리브레이션 풀에 쓰인 baseline
   청산시각도 그 자체가 causal하게 실현된 값이고, `searchsorted`가 여전히 `exit_timestamp <
   bar_timestamp` 엄격부등호를 강제한다) — 다만 "캘리브레이션 이력이 스케일 적용 후 실제
   거래이력과 완전히 동일하다"는 가정이 2차 근사임을 뜻한다. 완전한 고정점(스케일 적용 실행
   자체의 결과로 풀을 재구성해 수렴할 때까지 반복)은 `replay_exit_variant`(수정 금지 대상)의
   bar-loop 내부에 스케일 계산을 끼워 넣어야 해서 이번 범위에서 구현하지 않았다 — 이미 결과가
   3개 그리드 후보 전부에서 만장일치로 VAL 게이트를 떨어뜨리는 상황이라, 이 2차 근사가 결론을
   바꿀 가능성은 낮다고 판단해 **구현 대신 정량적으로 공개**한다
   (`report.json["val"]["candidates"][*]["ledger_divergence_from_baseline"]`).

## G0 — 필수 재현 확인

과제 지정 4개 수치(오늘 밤 risk-controlled 실험 report.json에서 확립)를 **두 개의 독립 코드
경로**로 재확인했다: (a) 다중구간 게이트 모듈의 `run_portfolio_variant`(무수정 재사용), (b) 이
스크립트 자신의 walk-forward 머신에 스케일을 (1.0,1.0)로 강제해 항등연산으로 만든 경로.

| 창 | 지표 | 목표값 | (a) 게이트모듈 | (b) 자체머신(스케일≡1.0) |
|---|---|---:|---:|---:|
| VAL | no_gate | 46.59% / -21.70% / 35 | 46.59% / -21.70% / 35 (일치) | 46.59% / -21.70% / 35 (일치) |
| VAL | with_gate | 77.31% / -21.76% / 26 | 77.31% / -21.76% / 26 (일치) | 77.31% / -21.76% / 26 (일치) |
| OOS-Q1 | no_gate | 93.27% / -15.48% / 24 | 93.27% / -15.48% / 24 (일치) | 93.27% / -15.48% / 24 (일치) |
| OOS-Q1 | with_gate | 67.25% / -15.48% / 19 | 67.25% / -15.48% / 19 (일치) | 67.25% / -15.48% / 19 (일치) |

**G0 PASS** (`report.json["g0"]["pass"]=true`). 두 경로 모두 4개 수치 전부 정확히 일치 —
margin 스케일링 배관 자체(항등연산일 때)가 이미 검증된 baseline을 정확히 재현함을 확인했으므로
다음 단계로 진행.

## 캘리브레이션 진단

| 컴포넌트 | 사전-VAL 캘리브레이션 거래수(n) | q_ref(75th pct \|잔차\|) |
|---|---:|---:|
| h48qual | 194 | 0.04141 |
| zig075 | 75 | 0.06600 |

VAL~OOS-Q2 구간에서 실제로 관측된(unclipped) 스케일 비율 범위:

| 컴포넌트 | 창 | raw_ratio 범위 |
|---|---|---|
| h48qual | val | [0.923, 1.044] |
| h48qual | oos_q1 | [0.934, 0.971] |
| h48qual | oos_q2 | [0.962, 0.983] |
| zig075 | val | [0.941, 1.002] |
| zig075 | oos_q1 | [0.917, 0.966] |
| zig075 | oos_q2 | [0.914, 0.941] |

모든 창에서 raw_ratio가 가장 좁은 그리드 후보(narrow, 0.85~1.20) 안에 완전히 들어간다 — 즉
**3개 그리드 후보(narrow/medium/wide)가 실제로는 전혀 다른 결과를 만들지 않는다**(클립이 어느
후보에서도 바인딩하지 않음). 이는 그리드 선택의 우연이 아니라, 이 구체적인 히스토리에서
conformal 스케일 메커니즘 자체가 온건한 조정만 만든다는 사실을 보여준다 — 뒤집어 말하면, 아래
VAL 결과의 실패는 "그리드를 잘못 골라서"가 아니라 **메커니즘 자체가 방향을 잘못 짚었기
때문**이라는 뜻으로, 오히려 더 결정적인 부정 결과다.

## VAL 결과 (원기준 + 완화기준)

baseline(스케일=1.0, 무개입) = 확정 `asymmetric_tabm_liveatr`: no_gate 46.59%/-21.70%/35건,
with_gate 77.31%/-21.76%/26건.

| 후보 | bounds | no_gate | with_gate | 원기준(4지표 전부 비악화) | 완화기준(with_gate PnL 개선 + MDD 3%p 이내) |
|---|---|---:|---:|---|---|
| narrow | (0.85, 1.20) | 39.72%/-21.69%/35 | 52.09%/-21.91%/27 | **FAIL** | **FAIL** |
| medium | (0.70, 1.40) | 39.72%/-21.69%/35 | 52.09%/-21.91%/27 | **FAIL** | **FAIL** |
| wide | (0.50, 2.00) | 39.72%/-21.69%/35 | 52.09%/-21.91%/27 | **FAIL** | **FAIL** |

3개 후보 전부 숫자가 완전히 동일하다(위 진단대로 클립이 바인딩하지 않으므로 당연한 결과).
원기준은 no_gate PnL(46.59%→39.72%)·with_gate PnL(77.31%→52.09%)·with_gate MDD(-21.76%→
-21.91%) 3개 지표가 이미 악화라 실패. 완화기준도 with_gate PnL이 개선이 아니라 뚜렷한 악화라
실패(가드레일 항목은 exit_head 모델 자체를 바꾸는 실험에만 적용되므로 이 실험엔 해당 없음 —
h48qual/zig075 컴포넌트 단독 PnL도 참고용으로 확인: h48qual 9.23%→7.02%, zig075
40.31%→39.80%, 둘 다 소폭 악화로 포트폴리오 결과와 방향이 일치).

**`passing_original_gate=[]`, `passing_relaxed_gate=[]`, `val_winner=None`.**

## OOS 단일터치

**미실행.** VAL 사전등록 게이트(원기준·완화기준 둘 다)를 3개 후보 전부 통과하지 못해, 이
서브프로젝트의 표준 규율(VAL 실패 → OOS 금지)에 따라 OOS-Q1+OOS-Q2 단일터치를 열지 않았다 —
Odyssey2 #1(레짐threshold)·#4(GBDT)·#5(TCN)와 동일한 절차.

## 최종 판정

**`REJECTED_VAL_GATE`.** 사이징 축(문헌스카우팅 3위 후보)도 exit-timing 축(1위 대기압력, 2위
risk-controlled)과 마찬가지로 부정 결과로 종결한다 — 다만 **실패 지점이 다르다**: 1·2위는
VAL을 통과한 뒤 단일 OOS 창에서 반전했지만, 이번 3위는 **VAL 자체를 통과하지 못했다**(그리드
선택과 무관하게 만장일치 실패). VAL도 못 넘겼으므로 이번 실험은 애초에 "다중구간 확인 게이트"의
핵심 판정 로직(OOS-Q1+OOS-Q2 단일터치)을 시험할 기회조차 없었다.

원 논문의 약한 OOS 재현성 경고를 이 판정에 명시적으로 반영한다: 이 프로젝트의 재해석은 원
논문의 진짜 lockbox 테스트까지 가지 않고 VAL 단계에서 이미 멈췄지만, 원 논문 자신의 결론(*"진짜
2022+ 홀드아웃에서 캘리브레이션은 전이됐으나 Kelly 사이징의 경제적 가치는 전이되지 않았다,
development 값의 약 30%로 급락, 11개 비교 대상 중 꼴찌"*)은 이 재해석의 실패가 우연이 아닐
가능성을 시사한다 — conformal 구간폭 기반 사이징 스케일은 **캘리브레이션 자체는 정직해도
경제적 사이징 신호로서는 이 자산·이 시간대에 특별히 약할 수 있다**는 것이 이제 두 개의 독립된
데이터셋(원 논문의 8-ETF 일봉, 이 프로젝트의 ETH 5분봉 포지션 사이징)에서 같은 방향으로
관찰됐다. 이 서브프로젝트가 반복 관찰한 "VAL 승리 → OOS 반전" 패턴과는 **다른 실패 계열**(이번엔
VAL 자체가 실패)이라는 점도 명시한다 — 즉 이번 결과는 소표본 VAL 선택편향의 산물이 아니라, 이
구체적인 9개월 구간(2025-10~2026-06)에서 conformal 스케일이 실현 손익과 체계적으로 반대
방향으로 움직였다는, 메커니즘 자체에 대한 더 직접적인 부정 증거로 읽는다.

## 준수 확인

- `git diff` 대상 라이브 파일(`trading_bot.py`, `trading_bot_modules/omega4_6_1_live.py`,
  `trading_bot_modules/runtime_config.py`, `.env`) **0줄** — 직접 확인.
- 재학습 없음(두 risk_sidecar GBM 모두 frozen pkl 그대로 로드, margin_fraction 출력만 사후 스케일).
  GPU 불필요, CPU만 사용.
- `fresh_forward_bar_by_bar=true`, `trade_ledgers_used_as_input=false`(단, 캘리브레이션이 **이
  실행 자신이 생성한** 신선한 거래이력을 causal하게 소비한다는 점은 별도 플래그
  `kelly_calibration_uses_this_runs_own_fresh_trade_history=true`로 명시 — 과제가 승인한
  의도된 메커니즘이지 금지 패턴이 아님), `saved_parent_exit_timestamps_used=false`,
  `future_rows_used_for_entry=false` — report.json에 4개 플래그 전부 기록.
- 미수정 재사용(import만): `eth_omega461_multiwindow_confirmation_gate_20260814`
  (`load_all_windows`/`align_frame_and_predictions`/`run_portfolio_variant`/
  `summarize_multiwindow`/`COMP_CFGS_ASYMMETRIC_TABM_LIVEATR`/`_close`),
  `research_eth_omega461_exit_sweep_20260721`(`replay_exit_variant`/`COMPONENTS`/`COST_MULT`),
  `research_eth_omega461_exit_head_portfolio_asymmetric_20260813`(`_ledger_metrics`),
  `research_eth_omega461_live_sltp_mfe_width_20260813`(`_duration_gated`/`_as_router_component`),
  `replay_omega4_6_1_greedy_router_20260706`(`greedy_replay`/`DURATION_THRESHOLD`). direction_head/
  quality_head/exit_head 결정 로직·quality_threshold는 전혀 건드리지 않음(h48qual/zig075 둘 다
  라벨·threshold·exit_head 가중치 전부 동결, margin_fraction만 조건부 변경).
- `quality_threshold` 오염 캐비엇(Odyssey2 #8과 동일 상속): h48qual=0.50/zig075=0.75는
  2026-01-01~02-28 OOS-PnL 우선 선정으로 오염됨(`docs/experiments/
  eth_omega461_oos_selection_bias_scope_and_resolution_20260813.md`) — baseline과 후보가
  동일하게 공유하므로 상대비교는 유효하나, 이번 실험은 VAL에서 이미 기각돼 이 캐비엇이 최종
  판정에 실질적 영향을 주지 않았다.

## 산출물

- 스크립트: `scripts/research_eth_omega461_conformal_kelly_sizing_scale_20260814.py`.
- 결과: `tmp/causal_regen_20260516/eth_omega461_conformal_kelly_sizing_scale_20260814/report.json`
  (G0 양쪽 경로, 캘리브레이션 진단, VAL 3후보 원기준/완화기준, 렛저 괴리 진단, 정렬된 예측 CSV·
  렛저 CSV 다수).

## 출처

- [Conformal Kelly: Conformal Prediction Intervals as the Scale in Fractional Kelly Position Sizing (arXiv:2608.01494)](https://arxiv.org/abs/2608.01494) — Ryan, R.J., 2026-08-02. PDF 원문 직접 fetch+텍스트추출로 §3(Method)/§5(DEV results)/§10(Lockbox results)/§12(Conclusion) 확인.
- `docs/experiments/eth_omega461_post_entry_literature_scouting_20260814.md` (Odyssey2 #6, 랭킹 3위 후보 출처).
- `docs/experiments/eth_val_oos_regime_mismatch_investigation_20260813.md`, `docs/experiments/eth_omega461_multiwindow_confirmation_gate_20260814.md` (다중구간 게이트 도입 근거).
- `docs/experiments/eth_omega461_queue_pressure_exit_threshold_20260814.md`, `docs/experiments/eth_omega461_risk_controlled_post_processing_exit_fallback_20260814.md` (#7/#8, 같은 문헌스카우팅의 1·2위 후보, 실패 계열 비교 대상).
- `docs/experiments/eth_omega461_relaxed_gate_rescoring_20260814.md` (완화기준 정의 출처).
- `docs/model_contracts/odyssey2_eth_live_injection_contract_20260813.md` (서브프로젝트 계약, #9 실행 로그).
