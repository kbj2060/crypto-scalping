# ETH Omega4.6.1 — Selective Conformal Risk Control (SCRC) 2단계 exit fallback (2026-08-14, Odyssey2 #14)

번호 참고: 계약 문서 실행 로그에서 이 실험은 `#14`다 — `#13`은 동시 진행 중이던 다른 세션이 "Odyssey2
#11 라이브 섀도우 배포" 항목에 먼저 참조해둔 번호라 충돌을 피해 건너뛰었다(`docs/model_contracts/
odyssey2_eth_live_injection_contract_20260813.md` 참고).

상태: `tested_negative_val_gate_rejected` — **VAL 사전등록 게이트(원기준·완화기준) 둘 다 3개 eps
후보 전부 실패**, OOS는 이 프로젝트의 방법론(VAL 실패 시 OOS 미개방)에 따라 열지 않았다. #8
(risk-controlled post-processing, 2위 후보)과 **다른 이유로** 실패했다 — #8은 VAL을 통과하고
OOS 포트폴리오 레벨에서 반전됐지만, 이번 후보는 애초에 VAL 자체에서 기각됐다. 원인은 이 후보가
새로 추가한 **1단계 선별 자체가 2단계 calibration 표본을 극단적으로 좁혀(13,330개 중 52개,
0.39%) Algorithm 1의 eps-비율 설계를 모든 사전등록 eps에서 실행불가능(infeasible)하게 만들고,
그 결과 논문 자신의 "실행불가능하면 τ̂=0" fallback 규칙이 되레 아무 개입도 안 하는 것보다 훨씬
나쁜 정책(y-규칙 불일치율 5.66%→81.13%)을 선택**했기 때문이다 — 아래 "왜 #8과 다른 이유로
실패했는가" 절에서 직접 확인.

## 배경

문헌 스카우팅(`docs/experiments/eth_omega461_post_entry_literature_scouting_20260814.md`, Odyssey2
#6)이 4위로 랭킹한 후보를 구현한다. 1~3위(대기압력=#7, Risk-Controlled Post-Processing=#8,
Conformal Kelly=#9)는 전부 부정 결과로 종결됐다(#7·#8은 VAL 통과 후 OOS 반전, #9는 VAL 자체
실패). 오케스트레이터가 사전에 지적한 대로 **#8과 이번 후보(#14)는 개념적으로 가까운 계열**(둘
다 conformal/위험제어 기반 조건부 fallback 전환)이라 같은 실패 패턴을 물려받을 위험이 있었다 —
실제로는 **다른 실패 패턴**으로 확인됐다(아래 상세).

## 1단계: 논문 원문 확인 (WebFetch로 직접 확인, 스카우팅 문서 요약 재검증)

**출처**: Xu, Guo, Wei, "Selective Conformal Risk Control", arXiv:2512.12844 (v1 2025-12-14, v2
2026-04-27). arXiv abstract 페이지와 `arxiv.org/html/2512.12844v2` 전문 HTML을 둘 다 직접 fetch.

**초록 원문(발췌)**: "we propose Selective Conformal Risk Control (SCRC), a unified framework that
integrates conformal prediction with selective classification. The framework formulates uncertainty
control as a two-stage problem: the first stage selects confident samples for prediction, and the
second stage applies conformal risk control on the selected subset to construct calibrated
prediction sets." → **스카우팅 문서의 요약("1단계 선별 → 2단계 선별된 부분집합에만 conformal
위험 제어")이 정확함을 확인.**

**정확한 문제 설정** (K-클래스 분류 + prediction SET 문제, 이 프로젝트의 단일 hold/exit 결정과는
다른 문제 유형임에 유의): 기본 분류기 f(x)∈[0,1]^K, 별도의 신뢰도(선별) 점수 g(x)∈[0,1](**Y에
의존하지 않는 feature-only 함수** — 대칭적 선별을 위해 논문이 명시적으로 요구하는 조건), 임계값
쌍 (λ1,λ2). 선별 규칙: g(x)<1-λ1이면 **거부(⊛, 예측 안 함)**, 아니면 예측셋
C_λ2(x)={k:f(x)_k≥1-λ2}.

- **SCRC-T(transductive)**: λ1을 **calibration+test 표본에 대해 공동으로**(대칭함수로) 계산 —
  교환가능성(exchangeability)을 선별 이후에도 보존, **정확한 유한표본 보장**(Theorem 2: 조건부
  위험 E[ℓ|선별됨]≤α, 커버리지 P(선별됨)≥ξ 둘 다 exact). 단, test 시점 feature를 calibration
  시점에 봐야 하는 온라인/재계산 구조.
- **SCRC-I(inductive/calibration-only)**: λ1(Dvoretzky–Kiefer–Wolfowitz 하한보정)·λ2(Hoeffding
  보정, λ1×λ2 그리드) 둘 다 **calibration 데이터만으로 고정**, test 시점 접근 불필요 —
  **PAC 스타일**("확률 ≥1-δ로") 보장, SCRC-T보다 근소하게 보수적(prediction set이 약간 큼),
  calibration 표본이 커질수록 그 격차는 줄어든다고 논문 스스로 보고.
- **논문이 스스로 밝힌 한계**(결론부, 직접 확인): "거부된 표본에는 2단계 예측셋 보장을 제공하지
  않는다 — downstream fallback 메커니즘에 맡긴다"(범위 밖으로 명시적으로 남김). 2단계는
  m≥⌈1/α⌉-1개의 선별된 표본이 있어야 실행 가능(feasibility floor). 선별함수 g(x)/분류기 f(x)는
  고정("학습하지 않음")이 전제.

**#8(Joshi/Wang/Hassani/Dobriban, arXiv:2605.06479)과의 관계 — related work 절 직접 확인**: 이
논문의 related work(conformal prediction 블록, selective classification 블록, "통합" 블록 —
Fisch et al. 2022 "calibrated selective classification", Bao et al. 2024 "selective conditional
conformal prediction", Gazin et al. 2024)에 **Joshi/Wang/Hassani/Dobriban 인용이나 decision-policy
post-processing 계열 논문 언급이 전혀 없다** — 확인 결과 부재, 단순 미기재가 아니라 related work
전체를 훑어도 없음. 시간적으로도 SCRC v1/v2(2025-12-14/2026-04-27) 둘 다 Joshi et al.(2026-05-07)
보다 앞서 나와 SCRC가 Joshi를 인용할 수 없었고, 이 프로젝트의 #8 스크립트/문서가 확인한 Joshi
논문 쪽에서도 SCRC를 인용하지 않는다. **독립적인 병행 연구, 후속/파생 관계 아님.** 메커니즘상
차이도 명확하다: Joshi et al.은 **단일 스코어 Δ(x), 단일 임계값 τ로 두 액션(π0/π*) 중 하나를
고르는 단일 단계** 의사결정정책 문제이고, SCRC는 **거부(선별 실패) 자체가 별도 보장을 갖는 진짜
2단계** 분류+선별 문제다. #8은 이미 Joshi의 단일단계 메커니즘을 이 프로젝트에 전부 구현·검증
완료했다(VAL승리→OOS포트폴리오반전). 이번 스크립트의 과제는 **#8이 갖지 않았던 진짜 2단계(선별
먼저) 구조를 추가하는 것**이지 #8을 다시 도는 것이 아니다.

## 정직한 스코핑 선언

SCRC의 K-클래스 prediction-set 형식론이나 SCRC-T/SCRC-I의 DKW/Hoeffding calibration 알고리즘을
문자 그대로 이식하지 않는다(연속적인 단일 hold/exit 결정에는 "prediction set" 개념의 자연스러운
대응물이 없고, 이 프로젝트의 수십~수백 거래 규모 calibration 표본에 새로운 집중부등식 보정을
얹는 것은 검증되지 않은 복잡도만 추가한다). **이식하는 것은 논문의 핵심 구조적 주장** — 선별
먼저(feature-only 신뢰도 점수, Y에 의존하지 않음) → 그 부분집합에만 위험 제어 적용 — 이며, 2단계
위험제어의 실제 수식은 **#8이 이미 구현·검증한 Joshi et al. Algorithm 1을 무수정 재사용**한다.
이 재해석은 명시적으로 선언하며, 문자적 SCRC-T/SCRC-I 이식이라 주장하지 않는다 — #8이 Joshi
논문의 LLM 라우팅 예시를 이 프로젝트에 대응시킬 때 적용한 것과 동일한 정직성 기준이다.

## 2단계: 이 프로젝트 구조로 대응 — 2단계 구조의 명시적 구현

| 논문 개념 | 이 실험 대응 |
|---|---|
| g(x) [1단계 선별 점수] | TabM baseline exit_head 자신의 causal 확률 `prob_baseline(x)` — Y(y-규칙)에 의존하지 않는 feature-only 함수, 논문의 "L^(1)은 X에만 의존" 요구와 일치 |
| select_threshold ["1-λ1"] | **고정값 `sweep.BASELINE_EXIT_THRESHOLD`=0.95**(기존 EXIT_THRESHOLD와 동일, 새 하이퍼파라미터 아님). "선별됨" = 정확히 `{a0=1}`(TabM 자신이 이미 확신을 갖고 exit를 결정한 bar) — 스카우팅 문서의 "확신 있는 exit 신호만 선별"을 가장 직접적으로 구현. **VAL에서 적응적으로 캘리브레이션하지 않음** — `eth_val_oos_regime_mismatch_investigation_20260813.md`가 진단한 "risk-sizing→quality_threshold→신규후보, 전부 같은 ~26,000bar VAL 창에 3중 재사용" 선택편향 패턴 위에 4번째 VAL-fit 자유도를 얹지 않기 위한 의도적 설계(우연이 아님) |
| select_threshold(진단 전용, 게이트 미적용) | 0.50 — 선별을 명백히 넓히기 위한 확률 중립점. 게이트에 절대 쓰지 않음, #8과의 대조용 |
| 2단계 위험제어(선별된 부분집합에만) | #8과 완전히 동일한 메커니즘: π0=TabM baseline(0.95), π*=이미 학습된 GBDT exit_head(#4, `gbdt_exit_bundle.pkl`, 0.95), g(hold,x)=p_gbdt(x)/g(exit,x)=1-p_gbdt(x), Δ(x)=g(π0(x),x)-g(π*(x),x), y-규칙=pos_giveback≥0.65 OR pos_unrealized≤-0.010 — 전부 `research_eth_omega461_risk_controlled_exit_fallback_20260814`(rc_mod)에서 **무수정 import** |
| eps 그리드 | rc_mod.EPS_FRACTIONS=[0.90,0.70,0.50] 재사용(사전등록), 단 **baseline 불일치율을 선별된 부분집합에서만 측정**(논문의 Z_{λ̄1} 제한 — "선별된 부분집합에만 conformal 위험제어"의 정확한 구현) |

**#8과의 구조적 차이(설계 시점에 예측, 결과로 확인)**: select_threshold=0.95=baseline_threshold이므로
"선별됨"은 정확히 `a0=1`과 같다 — 이 부분집합에서는 Δ(x)가 오직 `astar=0`(GBDT가 반대)일 때만
0이 아니므로, **이 후보의 개입은 구조적으로 "TabM의 확신 있는 exit 신호를 취소"만 가능하고, TabM이
확신하지 못한 bar에 새 조기exit를 유발할 수 없다**(#8은 둘 다 가능했다). 아래 calibration 결과가
이 예측을 직접 확인한다(`selected_all_have_a0_eq_1=True`).

## 구현

`research_eth_omega461_exit_sweep_20260721.py`/`replay_omega4_6_1_greedy_router_20260706.py`/
`research_eth_omega461_risk_controlled_exit_fallback_20260814.py`/
`eth_omega461_multiwindow_confirmation_gate_20260814.py` 전부 무수정(import만). 신규 함수
`_selective_risk_controlled_action`(1단계 게이트, `selected=False`면 rc_mod의 stage-2 함수를 아예
호출하지 않음 — GBDT 순전파 자체를 skip해 계산도 절약) + 원본의 이름 바꾼 복사본
`replay_exit_variant_selective_risk_controlled`/`greedy_replay_selective_risk_controlled`(rc_mod의
동일 패턴 복사본에서 판정 블록만 추가 교체) + `eth_omega461_multiwindow_confirmation_gate_20260814`의
`load_all_windows`/`align_frame_and_predictions`/`summarize_multiwindow`(공개 API만 사용, 그
모듈이 예시로 보여준 자체 `_risk_controlled_variant` 패턴을 그대로 따름).

**G0c(신규, #8에 없던 자체검증)**: select_threshold=1.01(never-select 센티널)로 실행 — 1단계가
무력화된 상태에서도 baseline을 정확 재현해야 함, 2단계의 G0b(τ=NEVER_SWITCH)와 독립적으로 1단계
배관을 검증.

## G0 결과 — 오케스트레이터 지정 4개 수치 전부 정확 일치

| | no_gate | with_gate |
|---|---|---|
| VAL asymmetric_tabm_liveatr | 46.59%/-21.70%/35 (match) | 77.31%/-21.76%/26 (match) |
| OOS-Q1 asymmetric_tabm_liveatr | 93.27%/-15.48%/24 (match) | 67.25%/-15.48%/19 (match) |

`eth_omega461_multiwindow_confirmation_gate_20260814.run_portfolio_variant`을 통해 재현(단일
출처 재사용, 재입력 없음) — **PASS**. 컴포넌트 G0(baseline_original 5.45%/-11.62%/29,
tabm_liveatr 9.23%/-7.59%/63, `rc_mod.G0_REFERENCE` 대조)도 **PASS**. G0b(τ=NEVER_SWITCH, 2단계
무력화)·G0c(select_threshold=1.01, 1단계 무력화) 둘 다 컴포넌트·포트폴리오 정확 재현, G0b의
`rc_switch_bars=0`·G0c의 `rc_selected_bars=0` 확인 — **PASS**.

## Calibration 결과 — 1단계가 2단계를 실행불가능하게 만든 지점

VAL 컴포넌트 리플레이(13,330 보유 bar) 중 select_threshold=0.95로 **선별된 bar는 52개(0.39%)**
뿐이었다 — 전부 `a0=1`(설계대로, `selected_all_have_a0_eq_1=True`로 직접 확인). 이 52개 중
44개(84.62%)에서 GBDT가 TabM과 불일치한다. `τ=NEVER_SWITCH`(아무 개입 안 함) 상태의 y-규칙
불일치율(bumped) = **5.66%**.

| eps_frac | eps(=5.66%×frac) | τ̂ | feasible/grid | τ̂에서의 위험 | 논문 feasibility floor(⌈1/eps⌉-1) |
|---|---:|---:|---:|---:|---:|
| 0.90 | 5.09% | **0.0000** | **0/46** | **81.13%** | 19 (n=52로 충족) |
| 0.70 | 3.96% | 0.0000 | 0/46 | 81.13% | 25 (n=52로 충족) |
| 0.50 | 2.83% | 0.0000 | 0/46 | 81.13% | 35 (n=52로 충족) |

**핵심 발견**: eps는 항상 baseline 불일치율(5.66%)의 진분수이므로, "아무 개입도 안 함" 자체가
어떤 eps도 만족 못 한다(F_n을 non-empty로 만들려면 반드시 baseline보다 나은 중간 임계값이
존재해야 하는데, 이 52개 표본에서는 46개 그리드 포인트 전부 실패). Algorithm 1 자신의 명시적
fallback 규칙("F_n이 공집합이면 τ̂=0")이 발동됐고, **그 결과 채택된 정책(τ=0, 비음수 Δ에서는
전부 전환)이 아무것도 안 하는 것보다 오히려 14배 더 나쁜 불일치율(5.66%→81.13%)을 낸다** — 이
52개짜리 좁은 선별집합에서는 GBDT가 TabM과 불일치할 때 정말로 GBDT가 자주 틀렸다는 뜻이다
(84.62% 불일치 자체가 매우 높음).

**진단 전용 대조(select_threshold=0.50, 게이트 미적용)**: 선별 폭을 넓히면 6,738개(50.55%)가
선별되고 그중 `a0=1`은 0.77%뿐(=선별이 정말로 `{a0=1}`보다 훨씬 넓어짐, 설계대로), baseline
불일치율 54.84%, eps_frac=0.90 → **τ̂=0.9996(feasible 580/912, 정상 작동)** — #8이 전체
모집단(n=13,330)에서 찾은 τ̂=0.9995와 거의 같은 값이다. 즉 **calibration 수식 자체(rc_mod에서
무수정 재사용)는 정상이고, 붕괴는 순전히 1단계 선별 폭(select_threshold=0.95라는 선택)이
calibration 표본을 극단적으로 좁힌 데서 온다** — 원인이 정확히 특정된다. 이 진단점의 VAL PnL은
게이트 대상이 아니라 산출하지 않았다(범위 밖).

## VAL 결과 (select_threshold=0.95, 3개 eps 전부 τ̂=0.0으로 동일 정책에 수렴)

| | 컴포넌트 no_gate | 포트폴리오 no_gate | 포트폴리오 with_gate |
|---|---:|---:|---:|
| baseline | 9.23%/-7.59%/63 | 46.59%/-21.70%/35 | 77.31%/-21.76%/26 |
| 후보(eps 0.90/0.70/0.50, 전부 동일) | **8.53%/-10.33%/45**(악화) | **63.10%/-17.80%/31**(개선) | **84.63%/-25.16%/24**(PnL개선, MDD악화) |

포트폴리오 리플레이의 선별 bar=19, 전환 bar=18(→exit:0, →hold:16, 나머지 2는 τ̂=0에서 Δ=0인
합의 상황도 "전환" 플래그가 기술적으로 True가 되는 경계효과 — 실제 결정에는 영향 없음).

- **원기준(4개 지표 비악화)**: `component_pnl_nonworse=False`(8.53<9.23), `component_mdd_
  nonworse=False`(-10.33<-7.59) — **FAIL**(포트폴리오 2개는 통과하지만 컴포넌트 2개가 실패).
- **완화기준(with_gate PnL개선+MDD 3%p이내+가드레일)**: PnL개선 True(84.63>77.31), 가드레일
  True(8.53>0 and 8.53≥9.23×0.5=4.615), **MDD가 -25.16%-(-21.76%)=-3.40%p 악화로 3.0%p 슬랙을
  근소하게 초과 — FAIL**.

**3개 eps 후보 전부, 두 기준 모두 실패** → `val_passing_original=[]`, `val_passing_relaxed=[]`,
`val_winner=None`.

## 왜 #8과 다른 이유로 실패했는가

| | #8(Risk-Controlled Post-Processing) | #14(이번, SCRC) |
|---|---|---|
| 실패 지점 | **OOS**(VAL은 원기준·완화기준 둘 다 통과) | **VAL**(3개 eps 후보 전부, 두 기준 모두 실패) |
| 메커니즘 | Δ 임계값 정책 자체는 VAL에서 건강하게 작동(τ̂=0.9995, 전환 7건뿐, 컴포넌트도 개선) — OOS에서 단 4건의 전환이 **공유슬롯 재순환**을 촉발해 포트폴리오 PnL이 72pp 넘게 흔들림(원장 직접 대조로 확인) | 1단계 선별(select_threshold=0.95)이 calibration 표본을 52개(0.39%)로 좁혀 **eps-비율 설계 자체가 모든 사전등록 eps에서 실행불가능**해지고, 논문 자신의 fallback(τ̂=0)이 "개입 안 함"보다 14배 나쁜 정책을 선택 |
| 공통점 | 둘 다 conformal/위험제어 기반 조건부 fallback 전환, 둘 다 GBDT exit_head를 fallback으로 재사용 | 좌동 |
| 차이의 본질 | **포트폴리오-레벨 상호작용 아티팩트**(이 프로젝트의 단일계좌·공유슬롯 구조 특유의 문제, 논문 메커니즘 자체의 결함 아님) | **calibration 표본 크기와 위험예산 설계의 구조적 상호작용**(선별을 좁힐수록 안전해질 것이라는 SCRC의 직관이, 이 프로젝트 규모의 VAL 표본에서는 정반대로 calibration을 무력화시킴 — SCRC 메커니즘 자체가 이 프로젝트 규모에서 드러낸 한계) |

오케스트레이터가 사전에 경고한 "같은 실패 패턴을 물려받을 위험"은 **현실화되지 않았다** — 표면
현상(둘 다 부정 결과)은 같지만, 실패가 발생한 파이프라인 단계(OOS 대 VAL)와 근본 원인(포트폴리오
슬롯 재순환 대 calibration 표본 붕괴)이 완전히 다르다. 이는 #14가 #8을 단순 재실행한 게 아니라
**진짜 다른 구조(2단계 선별)를 구현했고, 그 다른 구조가 다른 방식으로 실패했다**는 직접적 증거이기도
하다.

## OOS

**열지 않음** — 이 프로젝트 방법론("VAL 사전등록 게이트 통과 후에만 OOS 단일터치 개방", #9
Conformal Kelly가 이미 확립한 전례와 동일)에 따라 VAL에서 3개 후보 전부 두 기준 모두 실패했으므로
OOS-Q1/OOS-Q2/2025 Q1~Q3 어느 창도 실행하지 않았다. `report.json`의 `oos_opened=false`로 확인.

## 결론

**채택 불가.** SCRC의 핵심 구조(선별 먼저 → 선별된 부분집합에만 위험제어)를 명시적으로 구현했고
(#8과 구별되는 진짜 2단계 파이프라인, G0c로 독립 검증), 그 결과 #8과는 **다른 지점(VAL)**, **다른
메커니즘**(calibration 표본 붕괴)으로 부정됐다 — 표면적 유사성("둘 다 conformal 계열, 둘 다
GBDT fallback")에도 불구하고 실패 원인은 완전히 갈린다. 진단 전용 대조(select_threshold=0.50)가
calibration 수식 자체는 건강함을 보여주므로, 이 부정 결과는 구현 버그가 아니라 **"선별을 좁힐수록
안전해진다"는 SCRC의 직관이 소표본 VAL calibration에서 무너지는 실제 현상**으로 해석된다.
Odyssey2 문헌 스카우팅(#6) 4위 후보는 이것으로 **부정 결과 종결**한다.

## 미해결 / 다음 단계

- select_threshold를 0.95와 0.50 사이 중간값(예: 0.80~0.90)으로 재시도하면 calibration이
  건강해지는 지점이 있을 수 있으나, 이는 이미 종결한 이 후보의 재튜닝이며 이 프로젝트 규율(결과를
  본 뒤 재튜닝 금지)에 어긋난다 — 만약 향후 재검토한다면 **새 사전등록**으로 별도 실험이어야 한다.
- Algorithm 1의 "F_n 공집합 → τ̂=0" fallback이 좁고 불일치율 높은 calibration 집합에서 "개입
  안 함"보다 나쁜 정책을 낼 수 있다는 것은 이번 실험이 직접 관찰한, 향후 유사 선별-후-위험제어
  설계에 적용 가능한 일반적 주의사항이다(논문 자체는 이 조합을 다루지 않음).
- 채택 가능한 변경 0건, 라이브 파일 미변경.

## 준수 확인

`fresh_forward_bar_by_bar=true`(두 리플레이 복사본 모두 단일 순방향 causal 루프, y-규칙도 그
bar까지 이미 실현된 position-state만 사용, 1단계 선별도 같은 causal `prob_baseline`에서 계산).
`trade_ledgers_used_as_input=false`. `saved_parent_exit_timestamps_used=false`.
`future_rows_used_for_entry=false`. direction_head/quality_head/quality_threshold/encoder 양쪽
컴포넌트 전부 동결(h48qual exit_head **모델**도 TabM 라이브ATR·GBDT 둘 다 원래 가중치 그대로,
선택 로직만 조건부). zig075는 완전 동결(fallback 부착·선별 대상 아님).

`git diff` 확인(세션 전/후 모두 0줄): `scripts/replay_omega4_6_1_greedy_router_20260706.py`,
`scripts/research_eth_omega461_exit_sweep_20260721.py`,
`scripts/research_eth_omega461_risk_controlled_exit_fallback_20260814.py`,
`scripts/eth_omega461_multiwindow_confirmation_gate_20260814.py`, 라이브 파일(`trading_bot.py`,
`trading_bot_modules/omega4_6_1_live.py`, `trading_bot_modules/runtime_config.py`, `.env`).

Seed-Diversity Ensemble Promotion Gate: 해당 없음(결정론적 post-processing 정책, 시드 앙상블
승격 주장 없음). Omega Artifact Integrity Promotion Gate: 해당 없음(신규 parent 예측 아티팩트
없음, 기존 TabM·GBDT 아티팩트 그대로 재사용).

## 산출물

- 새 스크립트: `scripts/research_eth_omega461_selective_conformal_risk_control_20260814.py` —
  `_selective_risk_controlled_action`(1단계 선별 게이트, rc_mod의 stage-2 함수를 무수정 호출) +
  `replay_exit_variant_selective_risk_controlled`/`greedy_replay_selective_risk_controlled`(원본의
  이름바꾼 복사본) + G0(오케스트레이터 지정 4수치)/G0b(2단계 무력화)/G0c(1단계 무력화, 신규) +
  calibration(선별된 부분집합 전용) + 진단 대조(select_threshold=0.50) + VAL 3후보 스윕(이중
  게이트) 전부 단일 스크립트. VAL 기각으로 OOS 멀티윈도우 단계는 코드는 존재하나 미실행.
- report.json: `tmp/causal_regen_20260516/eth_omega461_selective_conformal_risk_control_20260814/
  report.json`(G0/G0b/G0c/calibration/진단대조/VAL 3후보 전부 포함, `oos_opened=false`).
- 거래 원장(diagnostic, 참고용): 같은 디렉토리의
  `portfolio_ledger_val_g0b_tau_never_switch.csv`,
  `{component,portfolio}_ledger_val_eps{0.90,0.70,0.50}.csv`,
  `portfolio_ledger_{val,oos_q1}_asymmetric_tabm_liveatr.csv`(G0 재현용).
- 인용 문서: `docs/experiments/eth_omega461_post_entry_literature_scouting_20260814.md`(이론적
  근거, Odyssey2 #6), [Selective Conformal Risk Control (arXiv:2512.12844)](https://arxiv.org/abs/2512.12844)
  — Xu, Guo, Wei, v1 2025-12-14/v2 2026-04-27, [Risk-Controlled Post-Processing of Decision Policies
  (arXiv:2605.06479)](https://arxiv.org/abs/2605.06479) — Joshi, Wang, Hassani, Dobriban(#8, 2단계
  중 stage-2 수식의 출처), `docs/experiments/eth_omega461_risk_controlled_post_processing_exit_
  fallback_20260814.md`(#8 본문, 대조 대상), `docs/experiments/eth_omega461_multiwindow_
  confirmation_gate_20260814.md`(재사용 인프라), `docs/experiments/eth_val_oos_regime_mismatch_
  investigation_20260813.md`(select_threshold를 VAL-적응형으로 만들지 않은 근거),
  `docs/model_contracts/odyssey2_eth_live_injection_contract_20260813.md`(서브 프로젝트 계약).
