# ETH Omega4.6.1 — Risk-Controlled Post-Processing exit fallback (2026-08-14, Odyssey2 #8)

상태: `tested_negative_closed` — **VAL 사전등록 게이트를 원 기준·새 기준 둘 다로 통과한 유일한
후보(eps_frac=0.90, tau_hat=0.9995)가 나왔으나, 1회 한정 OOS 확인에서 포트폴리오 레벨이 뚜렷이
반전**(no_gate PnL +93.27%→+21.18%, MDD -15.48%→-28.70%; with_gate PnL +67.25%→+4.77%, MDD
-15.48%→-28.70%). 컴포넌트 레벨은 OOS에서도 개선(+0.53%→+9.05%, MDD -9.02%→-3.24%)이 유지됐지만,
반전의 원인은 컴포넌트 경제성이 아니라 공유 슬롯 재순환(아래 "해석" 절 참고)으로 확인된다. 규율대로
재튜닝 없이 결과만 보고하고 **채택 불가로 종결**한다.

## 배경

Odyssey2 문헌 스카우팅(`docs/experiments/eth_omega461_post_entry_literature_scouting_20260814.md`,
#6)이 2위로 랭킹한 후보를 구현한다. GBDT/TCN exit_head(#4/#5)가 공통으로 드러낸 패턴 — "exit를
공격적으로 할수록 컴포넌트 단독 PnL은 악화하는데 공유 슬롯을 자주 비워 포트폴리오 지표는
개선된다" — 를 "GBDT/TCN을 전면 교체가 아니라 위험이 높다고 판단되는 좁은 조건에서만 개입하는
fallback으로 재활용"하는 방향으로 재시도한다.

### 논문 원문 확인 — Risk-Controlled Post-Processing of Decision Policies (Joshi, Wang, Hassani,
Dobriban, arXiv:2605.06479, 2026-05-07 제출)

문헌 스카우팅 문서의 1~2문장 요약이 아니라 **WebFetch로 arXiv abstract·본문(HTML)을 직접 읽어**
확인한 정확한 메커니즘:

1. **문제 정식화**: "바꾸길 꺼리는 기존 baseline 정책" π₀(x)와 fallback/oracle 정책 π*(x)가
   있을 때, baseline과의 일치도를 최대화하면서 위험 제약(marginal chance constraint)을 만족하는
   새 정책을 찾는다 —
   `max_π P(π(X)=π₀(X))  s.t.  P(ℓ(π(X),Y)≥c)≤ε`
   (ℓ은 손실함수, c는 "위반" 임계값, ε는 위험 예산, 제약은 population 전체 평균 기준).
2. **Theorem 3.1(population-level 최적해의 threshold 구조)**: oracle score
   `Δ(x) := g(π₀(x),x) - g(π*(x),x)`(g(a,x) = 행동 a를 취했을 때의 조건부 위반위험)를 정의하면,
   최적 정책은 `Δ(x)<τ`이면 π₀(x), `Δ(x)≥τ`이면 π*(x)로 전환하는 **threshold 정책**이다. τ는
   위험예산 ε를 정확히 소진하는 지점에서 정해진다 — agreement를 가장 적게 희생하면서(=가장 작은
   부분집합에서만 전환하면서) 제약을 만족시키려면, Δ(x)가 가장 큰(=전환했을 때 위험감소가 가장
   큰) 지점부터 전환해야 하기 때문이다.
3. **Algorithm 1(유한표본 calibration)**: calibration 데이터 `{(Xᵢ,Yᵢ)}ᵢ₌₁ⁿ`와 미리 적합된
   (별도 학습 데이터로, calibration/test와 독립적으로 고정된) score `Δ̂`가 주어지면, 임계값 격자
   `T_n = {0,⊤} ∪ {Δ̂ᵢ}`를 만들고 각 후보 t에 대해 **"bumped"(split-conformal 스타일 +1/(n+1)
   보정) 경험적 위반위험** `R̂ₙ⁺(t) = (Σᵢ 1{ℓ(π̂(Xᵢ,t),Yᵢ)≥c} + 1)/(n+1)`을 계산, 실행가능집합
   `F_n = {t∈T_n : R̂ₙ⁺(t)≤ε}`에서 **최대값** `τ̂ = max(F_n)`(공집합이면 0)을 택한다 — "최소
   전환으로 예산을 만족시키는 가장 보수적인 임계값"을 고르는 것.
4. **Theorem 4.2(유한표본 초과위험 보장)**: 정규성 조건(i.i.d.) 하에서
   `E[위반위험(배포된 정책)] ≤ ε + C₃·log(n+1)/(n+1)` — O(log n/n) 초과위험. **"정확한 안전
   fallback"(모든 x,y에 대해 ℓ(π_safe(x),y)<c) 특수 사례에서는 exchangeability 하에 초과항이
   0인 정확한 위험제어**가 성립하지만, 이 프로젝트의 GBDT fallback은 그 자체로 규칙과 어긋날 수
   있어 "정확한 안전"이 아니다 — 따라서 일반 사례(O(log n/n) 근사) 보장만 적용된다는 점을 이
   문서는 명시한다(억지로 더 강한 보장을 주장하지 않음).
5. **실험**: COVID-19 흉부영상 4클래스 진단→행동 매핑(Inception-V3), **LLM 라우팅**(Qwen3-4B
   fast=baseline, Qwen3-32B thinking=fallback, `Δ̂(x)=(ĝ₀(x)-ĝ_s(x))₊`를 hidden state에서 학습한
   두 로지스틱회귀로 구성), 합성 4클래스 분류(부록). LLM 라우팅 사례가 이 프로젝트와 구조적으로
   가장 가깝다 — baseline/fallback 각각의 "자기 자신의 오류 확률"을 별도 모델로 추정해 Δ를
   구성한다.

### 이 프로젝트로의 대응

| 논문 개념 | 이 실험 대응 |
|---|---|
| baseline 정책 π₀ | h48qual exit_head, TabM 라이브ATR 재라벨(현재 확정 Odyssey2 베이스라인), `EXIT_THRESHOLD=0.95` 고정 |
| fallback 정책 π* | 이미 학습된 GBDT exit_head(Odyssey2 #4, `gbdt_exit_bundle.pkl`) 자신의 임계값 0.95 결정(TabM과 동일 관례) — TCN은 시간 제약상 이번엔 미시도(GBDT가 더 가벼워 우선권장이라는 지시대로) |
| g(a,x) (조건부 위반위험) | GBDT 자신의 확률 p_GBDT(x)를 TabM과 독립적인 위험추정 모델로 재사용: g(hold,x)=p_GBDT(x), g(exit,x)=1-p_GBDT(x). LLM 라우팅처럼 "각 정책 전용" 모델 두 개를 새로 학습하는 대신, 이미 존재하는 독립 모델(GBDT)을 그대로 arbiter로 씀 — 재학습 0 |
| Δ(x) | `g(π₀(x),x) - g(π*(x),x)` — π₀·π*가 합의하면 정확히 0(전환 불필요), 불일치할 때만 GBDT 확신도에 비례해 커짐. "fallback과 baseline이 강하게 불일치할 때"(오케스트레이터가 제시한 예시)를 논문의 정식 Δ로 구현한 것 |
| Y(ground truth, calibration 전용) | `pos_giveback≥0.65 OR pos_unrealized≤-0.010` — exit_head 라벨의 98.1%를 차지하는 것으로 이미 확인된 규칙(GBDT 문서), 매 bar 이미 계산되는 position-state 값에서 그대로 읽음(causal, 추가 피처 0개) |
| ℓ(a,y), c | 0-1 불일치 손실(`ℓ(a,y)=1{a≠y}`), c=1 — "위반" = 선택한 행동이 이 규칙과 어긋남 |
| ε(위험예산) | VAL로만 계산한 baseline 자체의 bumped 불일치율의 {0.90, 0.70, 0.50}배 — 결과를 보기 전에 고정(사전등록) |
| calibration 데이터 | VAL(2025-10-01~12-31)만, h48qual 컴포넌트 단독(전액가상자본) 리플레이의 보유 bar 전부 |

## 방법

### 스코어·정책 — `_risk_controlled_action`(신규 함수, 두 리플레이 복사본이 공유)

매 bar(h48qual이 포지션 보유 중) TabM 확률 `prob`과 GBDT 확률 `prob_fb`를 **둘 다** 계산해
`a0=1{prob≥0.95}`, `a*=1{prob_fb≥0.95}`, `Δ=g(a0,x)-g(a*,x)`(위 정의)를 구하고, `Δ≥τ`면 `a*`,
아니면 `a0`을 최종 행동으로 쓴다. τ는 아래 calibration이 정한 값을 그대로 대입 — exit_head
**모델**(TabM 가중치) 자체는 손대지 않는다.

### 구현 — `greedy_replay`/`replay_exit_variant` 무수정, 이름 바꾼 복사본만

`scripts/replay_omega4_6_1_greedy_router_20260706.py`(`greedy_replay`)와
`scripts/research_eth_omega461_exit_sweep_20260721.py`(`replay_exit_variant`)는 **전혀 수정하지
않았다** — 새 스크립트에 이름 바꾼 복사본(`greedy_replay_risk_controlled`,
`replay_exit_variant_risk_controlled`)을 만들어 exit_head 판정 블록 한 곳만 위 로직으로
바꿨다(GBDT(#4)의 duck-typing, TCN(#5)의 윈도우 슬라이싱, 대기압력(#7)의 조건부 threshold와
같은 패턴). GBDT 쪽은 기존 `GBDTExitHeadWrapper`/`_gbdt_loaded_models`/`_inject_gbdt_exit_runtime`
(`research_eth_omega461_gbdt_exit_head_val_20260813.py`, #4가 이미 검증)를 **무수정 재사용** —
다만 그 함수들은 원래 TabM `exit_runtime`을 GBDT로 통째로 바꿔치기하는 용도였는데, 이 실험은
TabM과 GBDT 확률이 매 bar 둘 다 필요하므로 GBDT 런타임을 **별도 키**(`fallback_exit_runtime`)로
덧붙이는 방식으로만 다르게 조합했다(래퍼 클래스 자체는 한 줄도 새로 안 씀).

### G0/G0b 자체검증

- **G0(과제 지정 범위)**: 새 하네스로 컴포넌트 baseline_original(+5.45%/-11.62%/29건),
  컴포넌트 TabM 라이브ATR(+9.23%/-7.59%/63건), 포트폴리오 baseline_both_original
  (+36.82%/-24.34%/29건), 포트폴리오 asymmetric_tabm_liveatr(+46.59%/-21.70%/35건) 4개 전부
  발표값과 정확 일치(`h48cons._evaluate_val`/`portfolio.run_variant` 무수정 재호출). **PASS.**
- **G0b(자체 정합성)**: 두 리플레이 복사본을 `tau=10.0`(sentinel, `|Δ|≤1`이므로 항상 baseline만
  선택)으로 실행 — 컴포넌트(+9.23%/-7.59%/63건)·포트폴리오 no_gate(+46.59%/-21.70%/35건) 둘 다
  정확 일치, `rc_switch_bars=0`(실제로 한 번도 전환 안 함)까지 확인. **PASS.**
- **포트폴리오 with_gate 베이스라인 — 이번 실험에서 새로 확립**: `_duration_gated`를 이 G0b
  리플레이 원장에 적용해 **+77.31%/-21.76%/26건**(skipped=9)을 얻었다. **주의**: 기존
  `docs/experiments/eth_omega461_relaxed_gate_rescoring_20260814.md`가 "baseline with_gate
  PnL+54.88%/MDD-31.11%"로 적어둔 값은 이 asymmetric_tabm_liveatr(35건) 원장이 아니라
  `baseline_both_original`(29건, 둘 다 원본) 원장을 `_duration_gated`한 값임을 **직접 재현으로
  확인**했다(`portfolio_ledger_baseline_both_original.csv`에 같은 함수를 적용하면 정확히
  54.88/-31.11이 나옴). 과거 GBDT/TCN/레짐threshold/대기압력의 판정 자체는 이 구분과 무관하게
  전부 그대로 유지된다(가드레일·부호반전 등 결정적 사유가 별도로 있었음, 아래 참고) — 다만 이
  실험은 이 구분을 반영해 **asymmetric_tabm_liveatr 자신의 with_gate(77.31/-21.76)를 새 기준의
  기준선으로 사용**했다(같은 baseline의 no_gate/with_gate 두 렌즈를 일관되게 비교하기 위해).

### Calibration(Algorithm 1) — VAL 전용

컴포넌트 단독(h48qual, 전액가상자본) VAL 리플레이의 보유 bar 전부(13,330개, `tau=10.0` 축퇴
실행에서 부수적으로 로그)에서 `(Δᵢ, yᵢ, a0ᵢ, a*ᵢ)`를 수집했다. 포트폴리오(공유슬롯) 리플레이의
보유 bar(3,678개)보다 훨씬 크다 — h48qual이 zig075 우선순위에 막히지 않고 자기 신호가 뜰 때마다
매번 거래하기 때문이며, calibration 표본 크기를 키우기 위해 의도적으로 컴포넌트 레벨에서
calibration했다(paper Theorem 4.2의 O(log n/n) 항이 n=13,330이면 `log(13331)/13331≈0.00071`로
매우 작아짐 — 이 프로젝트 대부분의 실험이 30~70건대 **거래** 표본으로 돌아가는 것과 달리, 이번
calibration은 13,330개의 **bar** 표본으로 돌아간다는 게 통계적 이점이다).

- baseline 자신의 bumped 불일치율(τ=10.0, 즉 전환 없음): **31.21%**.
- a0≠a* (TabM·GBDT 행동 불일치) 빈도: **1,080/13,330 = 8.10%** — 대기압력(#7)의 "6.6~7.9%" 개입
  빈도와 같은 자릿수로, "거의 매번 발동"(GBDT/TCN 자체)과 질적으로 다르다.
- ε 그리드(사전등록, baseline 불일치율의 배수): {0.90×→0.2809, 0.70×→0.2185, 0.50×→0.1561}.

| eps_frac | ε | τ̂ | feasible/grid | risk(τ̂) |
|---|---:|---:|---:|---:|
| 0.90 | 0.2809 | **0.9995** | 647/1022 | 0.2808 |
| 0.70 | 0.2185 | **0.0000**(공집합→0 규칙) | 0/1022 | 0.2418 |
| 0.50 | 0.1561 | **0.0000**(공집합→0 규칙, 0.70과 동일) | 0/1022 | 0.2418 |

`eps_frac=0.90`(관대한 목표)은 τ̂=0.9995 — Δ가 거의 최대값(1)에 가까울 때만 전환하는 극도로
보수적인 정책을 찾아냈다. `eps_frac=0.70/0.50`(공격적 목표)은 이 grid 안에서 **달성 불가능**해
Algorithm 1의 공집합 규칙대로 τ̂=0(=Δ≥0이면 무조건 전환, 사실상 "불일치할 때마다 GBDT를
따른다")으로 떨어졌다 — 이는 GBDT 단독 전면교체(#4)와 사실상 같은 정책에 근접한다(아래 결과가
이를 확인).

## 결과 — VAL 후보 스윕(포트폴리오 2025-10-01~12-31)

| eps_frac | τ̂ | 컴포넌트 no_gate | 포트폴리오 no_gate | 포트폴리오 with_gate | 전환 bar(포트폴리오) | 원 기준 | 새 기준 |
|---|---:|---:|---:|---:|---:|---|---|
| baseline | — | +9.23%/-7.59%/63건 | +46.59%/-21.70%/35건 | +77.31%/-21.76%/26건 | — | — | — |
| **0.90** | **0.9995** | **+9.61%/-6.61%/75건**(개선) | **+66.41%/-21.70%/34건**(개선) | **+85.50%/-23.59%/27건**(개선, MDD +1.89pp 악화) | **7건**(→exit 7, →hold 0) | **PASS** | **PASS** |
| 0.70 | 0.0000 | +3.73%/-7.94%/72건(악화, -59.6%) | +101.27%/-19.81%/38건 | +120.20%/-19.59%/30건 | 1,666건(→exit 14, →hold 17) | FAIL(컴포넌트 2개) | FAIL(가드레일) |
| 0.50 | 0.0000 | (0.70과 완전 동일 — 같은 τ̂) | (동일) | (동일) | (동일) | FAIL | FAIL |

**0.90만 원 기준(4개 지표 전부 비악화)과 새 기준(with_gate PnL개선+MDD 3%p이내+가드레일)을 둘 다
통과**하는 유일한 후보다 — 이 실험의 특이점은 **컴포넌트 레벨도 그냥 비악화가 아니라 개선**됐다는
점(+9.23%→+9.61%, MDD도 -7.59%→-6.61%)이다. GBDT/TCN/대기압력 전부 "컴포넌트를 희생하고
포트폴리오만 좋아지거나(GBDT/TCN)" "포트폴리오만 위태롭게 좋아졌다가 OOS에서 반전(대기압력)"했던
것과 달리, 0.90 후보는 **포트폴리오 전 슬롯 보유 bar 3,678개 중 단 7개(0.19%)만 건드리고 그
7개가 전부 "더 일찍 나가기" 방향**이었다 — 논문이 의도한 "좁은 조건에서만 개입하는 fallback"이
정확히 이 규모로 구현된 것으로 해석된다.

0.70/0.50(τ̂=0)은 예상대로 GBDT 전면교체(#4, 포트폴리오+101.27%/컴포넌트+2.72%)와 거의 같은
숫자(포트폴리오+101.27%/컴포넌트+3.73%)로 수렴했다 — τ̂=0은 "불일치하면 무조건 GBDT를 따른다"는
정책이라 사실상 GBDT 전면교체의 근사이기 때문이다. 원 기준(컴포넌트 PnL·MDD 둘 다 악화)과 새
기준(가드레일: `+9.23%→+3.73%`는 -59.6% 상대악화로 50% 기준 초과) 둘 다에서 일관되게 기각된다.

**VAL 승자: eps_frac=0.90(τ̂=0.9995)** — passing_original=['0.90'], passing_relaxed=['0.90'].

## 결과 — OOS 단일 확인(2026-01-01~03-31, τ̂=0.9995 고정, 재보정 없음)

⚠️ **유보**: h48qual/zig075 `quality_threshold`가 2026-01-01~02-28 프레임에 OOS-pnl-1순위로
선택된 값이라(`eth_omega461_oos_selection_bias_scope_and_resolution_20260813.md`), baseline·후보가
동일하게 이 오염을 공유한다 — 상대비교는 유효하나 절대수치는 "깨끗한 미접촉 검증"이 아니다.

| | 컴포넌트(단독) | 포트폴리오 no_gate | 포트폴리오 with_gate |
|---|---:|---:|---:|
| baseline | +0.53%/-9.02%/28건 | +93.27%/-15.48%/24건(발표값과 cross-check 일치) | +67.25%/-15.48%/19건(skipped 5) |
| 후보(τ̂=0.9995) | **+9.05%/-3.24%/33건**(개선) | **+21.18%/-28.70%/25건**(반전) | **+4.77%/-28.70%/20건**(반전) |
| Δ | +8.52pp / +5.78pp개선 | **-72.09pp** / **-13.22pp악화** | **-62.48pp** / **-13.22pp악화** |

`oos_gate_original_survives=False`, `oos_gate_relaxed_survives=False` — **둘 다 반전**. 포트폴리오
전환 bar는 **단 4건**(→exit 3, →hold 1, `rc_hold_bars=585`)뿐이었다(대기압력 OOS의 "소수 거래만
건드림" 패턴과 같은 자릿수).

## 해석 — 왜 컴포넌트는 개선인데 포트폴리오만 반전됐는가(원장 직접 대조로 확인, 추정 아님)

OOS 포트폴리오 원장을 직접 대조했다: h48qual 자신이 관여한 4개 거래(전환 3건+무전환 1건 포함)는
baseline 대비 **개별적으로도 대부분 개선**됐다(`-0.0473→-0.0201`, `-0.0005→+0.0004`,
`-0.0002→-0.0009`(근소 악화), `-0.0170→+0.0012`) — 컴포넌트 레벨 개선이 우연이 아니라 실제
거래별로 확인된다. 그런데 baseline·후보의 총 거래수가 24건→25건(+1건)으로 달라졌고, **zig075
쪽 최저 5개 거래를 비교하면 후보에만 나타나는 새 손실거래**(`2026-01-21 stop_loss -7.86%`,
`2026-03-20 stop_loss -7.20%`)**가 있다** — h48qual의 청산 타이밍이 미세하게(전환 4건만) 바뀌면서
공유 슬롯이 zig075에게 풀리는 시점 자체가 달라지고, 그 결과 zig075가 잡는 거래의 **구성 자체가
바뀌어** OOS 3개월 구간에서 우연히 더 나쁜 stop_loss 거래를 추가로 물게 됐다 — GBDT/TCN/대기압력
문서가 반복 관찰한 "슬롯 재순환" 상호작용과 같은 메커니즘이지만, 이번엔 **컴포넌트를 개선시키는
전환이 포트폴리오 쪽에서 불리한 재순환을 촉발**했다는 점에서 그 문서들과 방향이 다르다(GBDT/TCN은
"컴포넌트 희생↔포트폴리오 이득", 이번은 "컴포넌트 이득↔포트폴리오 손실"). 단 4번의 전환이
포트폴리오 PnL을 72pp 넘게 흔든 것은, 단일계좌·우선순위 공유슬롯 구조에서 나비효과처럼 초기의
작은 청산시점 변화가 이후 수개월의 진입기회 시퀀스 전체를 재배열하기 때문으로 해석된다.

## 결론

**채택 불가.** VAL 사전등록에서 원 기준·새 기준을 둘 다 통과한 최초의 post-entry 후처리
후보(GBDT/TCN/레짐threshold/대기압력 중 유일)였고, 컴포넌트 레벨 경제성도 함께 개선됐다는 점에서
이전 실험들과 질적으로 달랐다. 그러나 OOS 단일 확인에서 포트폴리오 레벨이 뚜렷이 반전됐다 —
원인은 컴포넌트 경제성 훼손이 아니라(오히려 OOS 컴포넌트도 개선됨을 직접 확인) **단 4건의 전환이
촉발한 공유슬롯 재순환의 우연한 불리한 방향**으로 원장 대조를 통해 확인된다. 논문의 형식적
초과위험 보장(Theorem 4.2)은 **정의한 대리 위반사건(y-규칙 불일치)에 대한 것**이지 **실현
포트폴리오 PnL에 대한 것이 아니므로**, 이 실험의 실패가 논문 메커니즘 자체의 결함을 뜻하지는
않는다 — 다만 이 프로젝트의 단일계좌·공유슬롯 구조에서는 "매우 좁고 컴포넌트에 유리한 개입도"
포트폴리오 레벨 소표본 반전에 취약할 수 있다는 사실을 보여준다. Odyssey2 문헌 스카우팅(#6) 2위
후보는 이것으로 **부정 결과 종결**한다.

## 미해결 / 다음 단계

- Algorithm 1의 형식적 O(log n/n) 보장은 y-규칙(불일치) 위반율에 대한 것이지, 실현 거래 PnL에
  대한 것이 아니다 — "위반위험을 통제하면 실제 경제성도 통제된다"는 연결고리는 이 실험이
  가정했을 뿐 논문이 증명하지 않는다(논문 자신도 이런 격차를 명시하지 않음, 이 프로젝트가
  대응시키며 발견한 한계). 향후 유사 후처리 설계 시 y를 "실현 거래결과"에 더 가깝게 정의하는
  대안(예: 이번 실험 범위에서 시도하지 않은, 포지션별 사후 롤아웃 PnL 기반 y)을 고려할 수
  있으나 계산비용이 훨씬 높다.
- 공유슬롯 재순환이 "작은 컴포넌트 개선도 포트폴리오 레벨에서 증폭된 불리한 재배열을 낳을 수
  있다"는 이번 발견은, 향후 이 프로젝트가 post-entry 개입을 설계할 때 "컴포넌트 개선 여부"만으로
  안전하다고 판단해선 안 된다는 방법론적 시사점을 남긴다(추정, 별도 검증 없이 다음 실험에
  그대로 적용하지 말 것).
- TCN을 fallback으로 쓰는 변형은 시간 제약상 미시도(오케스트레이터가 GBDT를 우선권장) — TCN은
  VAL 단독으로도(#5) GBDT보다 컴포넌트 악화가 더 컸으므로, 이 메커니즘에서 TCN이 GBDT보다 나을
  근거는 약하다(추정, 미검증).
- 채택 가능한 변경 0건, 라이브 파일 미변경.

## 준수 확인

`fresh_forward_bar_by_bar=true`(두 리플레이 복사본 모두 단일 순방향 causal 루프, y-규칙도 그
bar까지 이미 실현된 position-state만 사용). `trade_ledgers_used_as_input=false`(원장은 출력
전용). `saved_parent_exit_timestamps_used=false`. `future_rows_used_for_entry=false`.
direction_head/quality_head/quality_threshold/encoder 양쪽 컴포넌트 전부 동결(h48qual exit_head
**모델**도 TabM 라이브ATR·GBDT 둘 다 원래 가중치 그대로, 선택 로직만 조건부). zig075는 모델·
threshold·exit 로직 전부 무변경.

`git diff` 확인(세션 전/후 모두 0줄): `scripts/replay_omega4_6_1_greedy_router_20260706.py`,
`scripts/research_eth_omega461_exit_sweep_20260721.py`,
`scripts/research_eth_omega461_gbdt_exit_head_val_20260813.py`,
`scripts/train_eval_omega461_gbdt_exit_head_liveatr_20260813.py`, 라이브 파일(`trading_bot.py`,
`trading_bot_modules/omega4_6_1_live.py`, `trading_bot_modules/runtime_config.py`, `.env`).
`replay_exit_variant_risk_controlled`/`greedy_replay_risk_controlled`는 각각
`replay_exit_variant`/`greedy_replay`의 이름 바꾼 복사본임을 직접 대조로 확인 — 실질 변경은
새 키워드 인자(`fallback_loaded_models`/`fallback_exit_runtime`/`tau` 등)와 exit_head 판정
블록 한 곳(risk-controlled 로직 삽입), 나머지 로직은 100% 동일.

Seed-Diversity Ensemble Promotion Gate: 해당 없음(재학습 모델이 아닌 결정론적 post-processing
threshold 정책이고, 시드 앙상블 승격 주장도 없음 — GBDT/TabM 둘 다 이미 확정된 단일 아티팩트를
그대로 재사용). Omega Artifact Integrity Promotion Gate: 해당 없음(신규 parent 예측 아티팩트를
만들거나 승격 주장하지 않음, 기존 TabM 라이브ATR·GBDT parent 아티팩트를 그대로 재사용).

## 산출물

- 새 스크립트: `scripts/research_eth_omega461_risk_controlled_exit_fallback_20260814.py` — 논문
  Algorithm 1 구현(`_calibrate_threshold`/`_bumped_risk`) + `_risk_controlled_action`(Theorem 3.1
  threshold 정책) + `replay_exit_variant_risk_controlled`/`greedy_replay_risk_controlled`(무수정
  원본의 이름바꾼 복사본) + G0/G0b 자체검증 + calibration + VAL 후보 스윕(이중 판정) + OOS 단일
  확인 전부 단일 스크립트.
- report.json: `tmp/causal_regen_20260516/eth_omega461_risk_controlled_exit_fallback_20260814/
  report.json`(G0/G0b + calibration + VAL 3후보(이중 게이트) + OOS 전부 포함).
- 거래 원장(diagnostic, 참고용): 같은 디렉토리의
  `portfolio_ledger_val_g0b_tau_never_switch.csv`,
  `{component,portfolio}_ledger_val_eps{0.90,0.70,0.50}.csv`,
  `component_ledger_oos_candidate.csv`,
  `portfolio_ledger_oos_{baseline_tabm_liveatr,candidate}.csv`(위 "해석" 절의 원장 대조가 이
  마지막 두 파일에서 나옴).
- 인용 문서: `docs/experiments/eth_omega461_post_entry_literature_scouting_20260814.md`(이론적
  근거, Odyssey2 #6), [Risk-Controlled Post-Processing of Decision Policies
  (arXiv:2605.06479)](https://arxiv.org/abs/2605.06479) — Joshi, Wang, Hassani, Dobriban,
  2026-05-07, `docs/experiments/eth_omega461_gbdt_exit_head_20260813.md`(#4, GBDT 번들·y-규칙
  출처), `docs/experiments/eth_omega461_tcn_exit_head_20260813.md`(#5),
  `docs/experiments/eth_omega461_queue_pressure_exit_threshold_20260814.md`(#7, 같은 계열의
  좁은 개입·OOS 반전 선례), `docs/experiments/eth_omega461_relaxed_gate_rescoring_20260814.md`
  (새 기준 정의 출처, with_gate 베이스라인 혼동 발견의 계기), `docs/experiments/
  eth_omega461_live_exit_head_liveatr_relabel_20260813.md`(TabM 베이스라인 근거),
  `docs/model_contracts/odyssey2_eth_live_injection_contract_20260813.md`(서브 프로젝트 계약).
