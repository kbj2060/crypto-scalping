# ETH 포트폴리오 Drawdown 예산 거버너 후보 — 데이터 계약 (2026-08-15)

이 문서는 **공식 Odyssey 계보(Odyssey1~4)에 속하지 않는다** — 버전 번호는 확정된 성과가 있을 때만 올린다는 원칙에 따라, 이 후보는 CLOSED 상태이므로 "Odyssey5"로 명명하지 않는다(2026-08-16, 사용자 결정). 조사·결과 자체는 향후 참고를 위해 그대로 보존한다.

## 상태

| 컴포넌트 | 상태 |
|---|---|
| **ETH 드로다운 거버너 후보** | **`CLOSED (2026-08-16, 사용자 결정)`**. cheap_gate(진입 전 스로틀)와 L9.1/L9.2 순환청산 ablation 둘 다 "BTC 원본 그대로는 부적합" 결론에 도달했고, 그 결론이 서로 정면충돌(cheap_gate는 진입 전 스로틀을 후순위로 미뤘는데, 그 미룸 자체가 순환청산의 재진입 처닝을 유발)했다. 해법(재진입 쿨다운)은 BTC 원본에 없는 신규 발명이 필요해 "신규 자유변수 0개" 원칙에서 벗어나므로, 사용자가 이 축을 접고 다른 후보로 넘어가기로 결정했다. 아래 "종결" 절 참고. |

## 범위

- 모델 id: `eth_candidate_drawdown_budget_governor_20260815`
- 목적: Odyssey4 베이스라인(h48qual 레짐인지형 exit 가드 + zig075 지속상승장/하락장 진입거부) 위에 **포트폴리오 레벨 drawdown 예산 거버너**를 얹는다. 트레이드 단위 사이징(L7 HGB risk sidecar)만 있고 계좌 equity-peak 기반 drawdown 예산이 전혀 없다는, `docs/experiments/eth_odyssey_internal_architecture_zoo_cross_pollination_survey_20260815.md`(Tier 1-1)가 지적한 구조적 공백을 메운다.
- 아키텍처 유형: **학습 모델이 아니다.** 관측된 cash/mark-to-market 경로/과거 equity peak만 입력으로 쓰는 결정론적 causal state-machine이며, config는 소수의 스칼라 파라미터(캡/임계값)로만 구성된 그리드다. 따라서 이 저장소의 N≥5 시드-다양성 게이트는 적용 대상이 아니다(결정론적 룰 — Odyssey4 진입거부 계약과 동일 논리).
- Owner agent: Model Architect(단독, Sonnet) — `feedback_architect_team_single_agent_sonnet` 컨벤션에 따름.
- 이식 원본: BTC `clean_base_deep_gated_drawdown_budget_v5`(레드팀 `APPROVED_AS_SHADOW_FRONTIER`) 중 **거버너 오버레이 부분만**. 원본은 별도의 Deep GRU 상태 인코더(`train_eval_clean_base_deep_state_hybrid_v2.py`)가 만드는 HIGH/MID/DEFENSIVE 확신도 버킷 위에 거버너를 얹지만, **그 신호생성 계층은 이식 대상이 아니다** — Odyssey는 이미 자신의 3-Head TabM(h48qual/zig075) + HGB risk sidecar가 그 역할을 한다. 이식하는 것은 오직 "account/daily drawdown 캡 + loss-streak 캡 + 손절/추세추종 청산 3종"이라는 **범용 리스크 오버레이 로직**뿐이다.
  - 원본 계약: `docs/model_contracts/clean_base_deep_gated_drawdown_budget_v5_contract.md`
  - 원본 레드팀: `docs/experiments/clean_base_deep_gated_drawdown_budget_v5_redteam.md` (verdict `APPROVED_AS_SHADOW_FRONTIER`)
  - 원본 구현: `scripts/train_eval_clean_base_deep_gated_drawdown_budget_v5.py`(그리드 선택/리포트), `scripts/train_eval_clean_base_deep_drawdown_min_v4.py`(`DrawdownMinConfig`, `backtest_drawdown_min` — 실제 게이팅 로직 원본)
- Odyssey 삽입 대상 스크립트: `scripts/research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814.py`의 `greedy_replay_entry_veto`(153행) — Odyssey4가 이미 쓰고 있는 causal bar-by-bar replay 하네스. 새 스크립트는 이 함수를 복사해 아래 "삽입 지점"에 훅을 추가하는 방식으로 작성한다(Odyssey4가 `guard.greedy_replay_regime_aware_exit_guard`를 복사해 진입거부 한 줄만 추가한 것과 동일한 패턴).
- 리소스 레지스트리: `docs/model_contracts/eth_candidate_drawdown_budget_governor_data_resources_20260815.md`
- 관련 문서: `docs/model_contracts/odyssey4_eth_entry_veto_baseline_contract_20260814.md`(G0 기준선), `docs/model_contracts/odyssey4_eth_full_stack_architecture_20260814.md`(L0~L10 전체 파이프라인, 2026-08-15 정정 완료), `docs/experiments/eth_odyssey_internal_architecture_zoo_cross_pollination_survey_20260815.md`(이 후보를 1순위로 선정한 조사)

## G0 기준선 (Odyssey4, 변경 없이 그대로 재사용)

Odyssey4 계약의 G0 표를 그대로 상속한다 — 이 거버너는 **이 표의 판정 3창(VAL/OOS-Q1/OOS-Q2)을 다시 계산하지 않는다.** 대신 동일한 6창(Q1/Q2/Q3 참고 + VAL/OOS-Q1/OOS-Q2 판정) 위에서 거버너 on/off 쌍으로 비교한다.

| 창 | 티어 | Odyssey4 `with_gate` (거버너 없음) |
|---|---|---|
| 2025-Q1(참고) | context | 44.98%/−20.62%/20 |
| 2025-Q2(참고) | context | 5.62%/−23.59%/19 |
| 2025-Q3(참고) | context | +20.17%/−19.72%/17 |
| VAL | val | 77.31%/−21.76%/26 |
| OOS-Q1 | oos_confirm | 67.25%/−15.48%/19 |
| OOS-Q2 | oos_confirm | −12.69%/−20.76%/10 |

출처: `docs/model_contracts/odyssey4_eth_entry_veto_baseline_contract_20260814.md` G0 표(PnL/MDD/trades, `with_gate` 열).

**이 계약의 승격 기준은 "PnL 극대화"가 아니라 "MDD 개선"이다.** OOS-Q2(−12.69%/−20.76%)처럼 MDD가 큰 창에서 거버너가 실제로 손실을 줄이는지가 핵심 질문이고, VAL(77.31%/−21.76%)처럼 PnL이 크지만 MDD도 큰 창에서 PnL을 얼마나 희생해야 MDD가 줄어드는지의 트레이드오프를 정직하게 보고해야 한다. BTC 원본도 같은 철학이었다: "validation은 10%-range MDD band 안에서 PnL을 최대화"(원본 계약 Architecture 절).

## 이식 원본 메커니즘 (BTC, 정확한 파라미터 인용)

`DrawdownMinConfig`(`train_eval_clean_base_deep_drawdown_min_v4.py:40`)와 `backtest_drawdown_min`(동 파일 200행)에서 확인:

**진입 전 노셔널 스로틀(매 bar, flat 상태에서 진입 직전 적용, `min()`으로만 축소):**

| 캡 | 트리거 조건 | 효과 |
|---|---|---|
| account DD soft | `account_dd = 1 - cash/closed_peak >= account_dd_soft`(예: 0.06~0.12) | 노셔널을 `account_dd_notional`로 캡 |
| account DD hard | `account_dd >= account_dd_hard`(soft보다 큰 값) | 노셔널을 더 낮은 `account_dd_hard_notional`로 재캡 |
| daily DD | `daily_dd = 1 - cash/daily_peak >= daily_dd_soft`(캘린더-day마다 peak 리셋) | 노셔널을 `daily_dd_notional`로 캡 |
| loss-streak | 연속 손실 마감 거래 수 `loss_streak >= loss_streak_soft`(예: 3~5) | 노셔널을 `loss_streak_notional`로 캡 |

**보유 중 매 bar 순환 청산 체크(TP/SL 판정과 같은 루프, 원본은 cost-stress ≥2x면 스킵):**

1. `equity_dd = 1 - eq/peak >= equity_mdd_stop` → `equity_mdd_budget_stop`(계좌 전체 mark-to-market MDD 예산 소진 시 강제청산)
2. `unreal <= -abs(hard_loss)` → `hard_loss_stop`(개별 트레이드의 unrealized 계좌-PnL 하한)
3. `best_unreal >= trail_activation AND unreal <= best_unreal - trail_gap` → `profit_trailing_lock`(MFE 대비 giveback 하한)

**불변식(원본 계약 Runtime Invariants, 이식 시에도 그대로 유지):**

- entry side는 절대 바꾸지 않는다.
- effective exit index는 원래(core) exit보다 **이르거나 같을 수만** 있다 — 절대 연장하지 않는다.
- 캡은 항상 `min()`으로만 적용된다 — 노셔널을 늘리는 방향으로는 절대 작동하지 않는다.
- 거버너 상태(peak/streak)는 관측된 cash·mark-to-market 경로·과거 peak만 사용한다 — 미래 데이터 없음.

## Odyssey 삽입 지점 (코드 레벨, `greedy_replay_entry_veto` 기준)

기존 하네스가 **이미 포트폴리오 전체 단일 `cash`/`peak`/`mdd`를 매 bar 갱신**하고 있다는 게 이식을 단순하게 만드는 핵심 사실이다(단일 공유 슬롯 구조라 h48qual/zig075를 구분할 필요가 애초에 없다 — account_dd/daily_dd/loss_streak는 자연스럽게 두 컴포넌트 풀링(pooled)이 된다, 별도 설계 결정이 아니라 기존 구조의 직접적 귀결이다):

| 신규 계층 | 훅 위치 | 신규 상태 |
|---|---|---|
| L4.9 진입 전 노셔널 스로틀 | 284~290행(`row_margin/row_leverage/row_notional` 계산 직후, entry 확정 전) | `closed_peak`, `daily_peak`(day_key 리셋), `loss_streak` — 셋 다 신규 추가 필요 |
| L9.1 계좌 DD 예산 순환청산 | 199~204행(`eq = cash*(1+unreal)`, `peak = max(peak, eq)` — **이미 존재하는 값을 그대로 재사용**) | 없음 — 기존 `peak`/`eq`에 조건 하나만 추가 |
| L9.2 hard-loss/trailing-lock 순환청산 | 206~216행(기존 `take_profit`/`stop_loss`/`trailing_stop` 체크 체인) | `best_unreal`(=기존 `mfe`와 동일값 재사용 가능 — `mfe`가 이미 `max(mfe, move)`로 추적됨) |
| 마감 시 상태 갱신 | 250~266행(`if reason:` 블록, `cash = cash*(1.0 + raw_exit*notional)` 직후) | `loss_streak = 0 if trade_return > 0 else loss_streak + 1`, `closed_peak`/`daily_peak` 갱신 |

**equity_mdd_budget_stop은 사실상 무료로 붙는다** — 기존 코드가 매 bar `peak = max(peak, eq)`를 이미 계산하고 있으므로, `eq/peak - 1.0 <= -equity_mdd_stop` 조건 하나를 `if not reason:` 체인에 추가하는 것으로 끝난다. `trailing_stop`(원본은 TP 대비 giveback)과 `profit_trailing_lock`(원본은 순수 unrealized-PnL 대비 giveback)이 이름은 비슷해도 기준이 다르다는 점에 주의 — 후자를 이식할 때 전자와 혼동하지 말 것(미해결 이슈 3 참고).

## 상태/피처 계약 (Feature Contract 대체)

- **외부 피처나 학습 모델 입력이 전혀 없다.** 상태는 전부 자기 자신의 렛저에서 파생된다: `cash`(관측), `peak`/`closed_peak`/`daily_peak`(과거 cash의 running max), `loss_streak`(과거 마감 거래의 부호), `mfe`/`unreal`(현재 보유 포지션의 mark-to-market, 이미 하네스에 존재).
- 이 때문에 leakage 위험은 구조적으로 낮다(미래 피처를 끌어올 경로 자체가 없음) — 하지만 **accounting identity 감사는 여전히 필수**다(peak/cash 갱신 순서가 하루라도 틀리면 참조 시점이 미래로 새는 부트스트래핑 버그가 가능 — BTC 원본도 `_audit`으로 이를 강제함).
- `daily_dd`의 day 경계는 BTC 원본이 `pd.Timestamp(...).date().isoformat()`(캘린더 날짜, 원본 소스의 로컬 timezone 그대로)로 정의한다. **Odyssey 라이브 피처 타임스탬프의 timezone 컨벤션과 일치하는지 구현 전에 확인 필요**(미해결 이슈 참고).

## Layer Contract

| Layer | Input state | Output | Artifact |
|---|---|---|---|
| L4.9 진입 전 스로틀 | `account_dd`, `daily_dd`, `loss_streak`(전부 자체 렛저 파생) | `row_notional`에 대한 추가 `min()` 캡(margin_fraction/leverage 자체 공식은 불변, CLAUDE.md Futures Risk Sizing Contract와 정합 — notional만 하향, `notional = margin_fraction * leverage` 관계는 유지한 채 상한만 낮춤) | 신규 스크립트(경로는 데이터 리소스 레지스트리에 등록) |
| L9.1/L9.2 순환청산 | `eq`, `peak`, `unreal`, `mfe`(전부 하네스에 이미 존재하거나 자명하게 파생) | `exit_idx` 조기화 + `stop_reason` ∈ {`equity_mdd_budget_stop`, `hard_loss_stop`, `profit_trailing_lock`} | 동일 |

## Cost/Risk Assumptions

- Fee/slip: Odyssey4와 동일하게 `omega._load_fee_slip()` 재사용, 1x/2x/3x 스트레스 3종 모두 리포트.
- Notional/leverage 캡: 기존 `LEVERAGE_CAP=5.0`, `NOTIONAL_CAP=1.8`(`replay_omega4_6_1_greedy_router_20260706.py`) 불변 — 거버너는 이 상한 **아래로** 추가 캡을 걸 뿐, 상한 자체를 재정의하지 않는다.
- CLAUDE.md Futures Risk Sizing Contract 준수: 거버너가 손대는 것은 `notional`(= `margin_fraction * leverage`)의 상한뿐이다. `margin_fraction`과 `leverage`를 각각 재계산하거나 TP/SL price-move에 leverage를 다시 곱하는 이중계산을 하지 않는다 — 노셔널 캡을 걸 때 `leverage_v = row_notional / max(row_margin, 1e-12)`로 역산해 margin_fraction 자체는 sizing sidecar 산출값을 그대로 보존한다(기존 289~290행 패턴과 동일).
- 원본과 동일하게 **cost-stress ≥2x에서는 순환청산 3종을 스킵**할지, 아니면 Odyssey 자체 cost1/2/3 관례에 맞춰 항상 적용할지는 미해결 이슈로 남긴다(원본은 저빈도 모드로 전환하는 게 목적이었지만 Odyssey는 이미 cost stress를 승격 게이트가 아니라 리포트 항목으로만 쓴다 — 원본 그대로 가져올 이유가 약함).

## Output Contract

렛저에 추가할 컬럼(원본 `train_eval_clean_base_deep_drawdown_min_v4.py` 361행 이하 패턴 준용):

```text
account_dd_prior
daily_dd_prior
loss_streak_prior
stop_reason        # 기존 reason에 추가: equity_mdd_budget_stop / hard_loss_stop / profit_trailing_lock
notional_capped_by  # "" | account_dd_soft | account_dd_hard | daily_dd | loss_streak (복수 가능, "|" join)
```

리포트 필수 지표: 기존 Odyssey4 리포트 항목(PnL/MDD/trades/wr) + `early_stop_fraction`(거버너가 개입한 거래 비율) + `governor_off` 대비 diff 표(원본 v5의 `v2_reference` 패턴처럼, "거버너 없음" 버전을 항상 같은 실행에서 나란히 리포트).

## Red Team Gates

- [ ] `greedy_replay_entry_veto`를 거버너 없이(config 전부 off) 그대로 재실행해 Odyssey4 G0 표(위 표)를 6창 전부 정확히 재현하는지 확인(Odyssey4 자신의 `G0b_copy_fidelity` 패턴 — 회귀 없음을 먼저 증명).
- [ ] Accounting identity 감사: cash/peak/closed_peak/daily_peak 갱신 순서가 causal한지, fee 반영이 원본과 일치하는지.
- [ ] Causality 감사: 거버너 상태가 전부 `i` 시점까지 관측된 값만 쓰는지(미래 bar 참조 없음) — CLAUDE.md Fresh-Forward 규정 그대로 재확인.
- [ ] **VAL/OOS 단일터치 프로토콜 재사용**: Odyssey4와 동일하게 `scripts/eth_omega461_multiwindow_confirmation_gate_20260814.py`로 VAL에서 config를 고정한 뒤 OOS-Q1+OOS-Q2를 한 번에 연다(순차/반복 확인 금지).
- [ ] N≥5 시드 게이트: **해당 없음**(결정론적 룰, 학습 파라미터 없음) — 단, config 그리드 서치 자체가 VAL 과적합이 되지 않도록 그리드 크기와 선택 기준을 계약에 사전 고정한다.
- [ ] G0 대비 비교: PnL 유지+MDD 개선, 또는 PnL 트레이드오프가 있다면 그 폭을 정직하게 표로 보고(ETH 드로다운 거버너는 "MDD 개선"이 채택 근거지 "PnL 극대화"가 아님을 위 G0 절에서 이미 명시).

## 필수 저비용 게이트 (cheap_gate) — **완료 2026-08-16, 결과: 둘 다 기각, 설계 우선순위 재조정**

이 저장소의 RL 사이징 서브프로젝트가 이미 확립한 방법론(`eth_odyssey4_rl_layer_axis_closed_20260815` — "RL보다 먼저, 더 싼 걸로 이길 수 있는지부터 확인")을 그대로 적용해 VAL 윈도우에서 실행했다(OOS 미개봉). 전체 과정·수치: `docs/experiments/eth_candidate_drawdown_budget_governor_cheap_gate_20260816.md`.

1. **고정 `NOTIONAL_CAP` 인하**(1.8→1.5/1.2/0.9/0.6) — **기각**: 모든 레벨에서 MDD 1pp 개선에 PnL 7~16pp를 지불하는 명백히 나쁜 트레이드(1.8→0.6: MDD −21.76%→−12.22%, PnL 77.31%→11.08%). 균일 축소는 승자/패자를 구분 못 하기 때문 — 오히려 상태의존형 거버너를 만들 이유를 강화하는 결과.
2. **단순 "일일 손실 N% 시 그날 신규진입 정지"** — **기각**: 8%/12% 임계값은 사실상 무영향(VAL에서 거의/전혀 미발동), 5% 임계값은 발동은 하지만 MDD를 **악화**시켰다(−21.76%→−24.15%, 슬롯 재배분 경로효과로 추정). 이 하네스는 포지션 보유 중에만 `peak`/`mdd`를 갱신하므로, 진입 전 정지 규칙은 **이미 열려 있는 포지션의 위험을 전혀 줄이지 못한다**는 게 핵심 원인.
3. **결론 — 구현 순서 재조정**: 두 저비용 대안 모두 "이거면 충분하다"를 주지 못했으므로 전체 거버너 구현으로 진행한다. 단, 결과가 시사하는 바에 따라 **L9.1/L9.2(보유 중 순환청산: equity_mdd_budget_stop/hard_loss_stop/profit_trailing_lock)를 L4.9(진입 전 노셔널 스로틀)보다 먼저 구현한다** — Odyssey는 거래 빈도가 낮고(VAL 4개월 26~35건) 단일 슬롯이라, MDD를 만드는 주 원인이 "몰린 신규 진입"이 아니라 "이미 보유 중인 포지션의 불리한 가격변동"이기 때문이다. 원래 계약의 "Layer Contract" 절이 두 계층을 동등하게 제시했던 것에서 변경.

## 미해결 이슈

1. **`hard_loss_stop`이 기존 SL과 실질적으로 겹치는지 미확인.** Odyssey의 `stop_loss`는 가격변동률 임계값(`move <= -abs(stop_loss)`, ATR floor 포화로 사실상 0.040 고정)이고, `NOTIONAL_CAP=1.8` 상한을 감안하면 단일 트레이드의 이론상 최대 계좌손실은 `0.040 * 1.8 = 7.2%`로 이미 SL만으로 bound된다. 그러나 sizing sidecar가 트레이드마다 notional을 다르게 산출하므로, 실제 관측 notional 분포에서 `hard_loss_stop`이 기존 SL보다 먼저 발동하는 사례가 있는지는 실측 전까지 모른다 — 구현 시 "발동했지만 SL이었다면 몇 bar 늦게 나갔을지" 카운트를 반드시 리포트할 것.
2. **daily_dd의 day 경계 timezone 컨벤션 미확인.** BTC 원본은 원본 소스 타임스탬프의 로컬 날짜를 그대로 쓴다. Odyssey 라이브 피처 프레임의 타임스탬프 timezone(UTC 여부)과 일치하는지 구현 전 확인 필요.
3. **`profit_trailing_lock`과 기존 `trailing_stop`(하네스 211~215행)의 관계 미정.** 기존 `trailing_stop`은 TP 대비 armed 상태에서 MFE 대비 SL폭만큼 giveback되면 발동(TP 활성화 여부에 의존). `profit_trailing_lock`은 TP와 무관하게 순수 unrealized-PnL의 MFE 대비 giveback으로 발동. 둘을 동시에 켜면 어느 게 먼저 발동하는지, 중복 로직인지 실측 전 판단 불가 — 구현 시 두 조건을 분리 리포트해서 겹침 여부를 먼저 확인.
4. **forward에서 실제 큰 drawdown 이벤트를 겪은 적이 없다.** Odyssey1~4와 동일한 정직한 한계 패턴 — VAL/OOS 구간 자체의 MDD(−20%대)는 존재하지만, 거버너가 겨냥하는 "계좌 자체의 연속 손실/급락" 이벤트가 판정 3창 안에서 실제로 몇 번이나 발동하는지는 구현 후 실측해야 안다. 발동 0건인 창이 대부분이면 Odyssey3/4처럼 "무해성 증명"에 그칠 수 있음을 미리 인지.
5. **그리드 크기.** BTC 원본은 3개 프로파일 x 손절 2~3종 x trailing 2종 x equity-stop 2종 = 수십 개 config를 VAL에서 스윕한다. Odyssey는 판정 윈도우가 더 짧고(VAL 4개월) 거래 수도 적어(26건) 같은 크기의 그리드를 돌리면 VAL 과적합 위험이 커진다 — 그리드를 원본보다 작게(예: 프로파일 1개, 손절/trailing/equity-stop 각 2종 이하) 시작할 것을 권고.

## L9.1/L9.2 순환청산 구현 결과 (2026-08-16)

전체 과정: `docs/experiments/eth_candidate_drawdown_governor_intrabar_stops_20260816.md`. G0 회귀는
6개 창 전부 통과(구현 자체는 정확). 그러나 3개 메커니즘을 한 번에 하나씩 켠 ablation(VAL만) 결과:

- `equity_mdd_budget_stop`(0.12/0.16/0.20): **파국적 붕괴** — 거래 26건→1000건 이상, PnL/MDD
  둘 다 −87~−92%. 렛저 확인 결과 보유기간 중앙값 0 bar — 강제청산 직후 같은 bar/다음 bar에
  같은 신호로 즉시 재진입하는 처닝이 원인. 튜닝으로 해결되는 임계값 문제가 아니라 재진입을
  막는 장치의 부재라는 설계 결함.
- `hard_loss_stop`(0.03/0.05/0.07): 처닝은 없지만 최선의 설정(0.03)도 MDD 1pp 개선에 PnL
  7.5pp를 지불 — cheap_gate가 이미 기각한 NOTIONAL_CAP 인하와 같은 급의 나쁜 트레이드오프.
  0.05는 MDD를 오히려 악화시킴(슬롯 재배분 경로효과).
- `profit_trailing_lock`(2개 조합): 둘 다 PnL·MDD 동시 악화 — 순손실, 채택 불가.

**근본 원인**: BTC 원본은 이 3개 stop을 항상 진입 전 노셔널 스로틀과 함께 배치해 재진입 처닝을
암묵적으로 억제했는데, cheap_gate 단계가 그 스로틀을 Odyssey에선 무력/역효과라며 뒤로 미룬 결정과
정면으로 충돌한다.

## 종결 (2026-08-16)

1. ~~cheap_gate~~ — 완료(2026-08-16), 둘 다 기각.
2. ~~L9.1/L9.2 순환청산 구현·ablation~~ — 완료(2026-08-16), 3종 전부 기각, 근본원인 진단됨.
3. **사용자 결정: (B) 축 종결.** 재진입 쿨다운(안 A)은 BTC 원본에 없는 신규 메커니즘을 발명해야
   해서 시도하지 않기로 함. OOS는 끝까지 개봉되지 않았다.

**재개 조건**: 이 문서가 다룬 두 가지(진입 전 스로틀, 보유 중 순환청산)를 BTC 원본 그대로
포팅하는 형태로는 재시도하지 않는다. 재개하려면 다음 중 하나가 필요하다 — (a) 재진입 처닝을
막는 원리적으로 다른 설계(쿨다운이 아닌 다른 방식으로), 또는 (b) Odyssey의 단일-슬롯·저빈도
거래 구조에 맞게 처음부터 다시 설계된 거버너. 단순 임계값 재스윕은 재개 근거가 아니다(3개
메커니즘 모두 그리드 전체에서 안정적으로 나쁜 패턴을 보였음).

전체 과정: `docs/experiments/eth_candidate_drawdown_budget_governor_cheap_gate_20260816.md`,
`docs/experiments/eth_candidate_drawdown_governor_intrabar_stops_20260816.md`.
