# ETH h48qual/zig075 동시-2슬롯 포지션 후보 — 데이터 계약 (2026-08-16)

이 문서는 **공식 Odyssey 계보(Odyssey1~4)에 속하지 않는다** — 버전 번호는 확정된 성과가 있을 때만
올린다는 원칙에 따라(2026-08-16, 사용자 결정), 결과가 확정되기 전까지 "Odyssey5"로 명명하지 않는다.

## 상태

| 컴포넌트 | 상태 |
|---|---|
| **동시-2슬롯 후보** | **`CLOSED (2026-08-16) — cheap_gate에서 상한선 자체가 음수로 기각`**. h48qual에 막힌 zig075 episode 185건의 counterfactual 수익 합이 −149.91%(notional 제약도 무시한 가장 관대한 상한선인데도 음수) — 전체 엔진 구현 없이 여기서 접는다. 상세: `docs/experiments/eth_candidate_dual_slot_cheap_gate_20260816.md`. "막힌 15.8% 기회비용" 조사 3단계(우선순위 스왑/max-hold/동시-2슬롯) 전부 부정 결과로 종결. |

## 범위

- 목적: 현재 h48qual/zig075는 **포트폴리오 전체에 열린 포지션이 항상 최대 1개**인 단일 공유
  슬롯 구조다. 이 제약 때문에 zig075의 quality-gate 통과 신호 episode 중 **15.8%(765/4844,
  6개 창 합산)가 h48qual이 이미 슬롯을 쥐고 있어 완전히 막힌다** — 2025-Q3(38.7%)·VAL(23.4%)에서
  특히 심하다(2026-08-16 실측, 아래 "여기까지 오게 된 경위" 참고). h48qual은 direction_head
  자체가 N≥5 시드로 무스킬 확정된 컴포넌트라, 이 컴포넌트가 zig075의 기회를 막고 있다면 순수
  포트폴리오 구성 변경(사이즈 분할)만으로 회수할 여지가 있는지 확인한다.
- 아키텍처 유형: **학습 모델이 아니다.** 리스크 예산을 두 컴포넌트로 나누는 결정론적 규칙이며,
  h48qual/zig075의 방향/품질/exit 헤드는 전혀 건드리지 않는다.
- 다른 후보들과의 결정적 차이: 이 세션에서 시도한 다른 것들(conformal veto, mid-trade exit
  trigger)은 전부 "같은 102개 피처에서 예측 능력을 더 짜내려는" 시도였고 전부 실패했다. 이건
  **새 예측을 주장하지 않는다** — 이미 검증된 두 신호(h48qual/zig075) 사이의 자원배분만
  바꾼다. 다만 "제로 신규 자유변수" 원칙(entry veto가 유일하게 성공한 이유)에서는 벗어난다 —
  notional 분할 규칙 자체가 새 설계 요소다.
- Owner agent: Model Architect(단독, Sonnet).
- 관련 문서: `docs/model_contracts/odyssey4_eth_entry_veto_baseline_contract_20260814.md`(G0
  기준선), `docs/experiments/eth_candidate_priority_swap_cheap_gate_20260816.md`(기각된 대안
  1), `docs/experiments/eth_candidate_h48qual_max_hold_cheap_gate_20260816.md`(기각된 대안 2),
  `docs/experiments/eth_candidate_conformal_veto_episode_labels_20260816.md`(이 계약의 cheap_gate가
  재사용하는 zig075 episode 라벨 산출물).

## 여기까지 오게 된 경위 (요약)

1. 사전 빈도 확인(2026-08-16, ad hoc REPL, 스크립트 미저장): h48qual/zig075가 **반대 방향으로
   동시에 신호를 내는 경우는 6개 창 전부 0건**(919개 동시신호 bar 전부 같은 방향). 이건 이
   후보의 설계를 크게 단순화한다 — 헤지/넷팅 로직이 애초에 필요 없다.
2. 우선순위 스왑(`PRIORITY=("zig075","h48qual")`) 테스트: VAL 결과가 소수점까지 무변화. 원인:
   `PRIORITY`는 둘 다 flat 상태에서 동시에 신호를 낼 때만(VAL의 0.43% bar) 승자를 가른다 —
   "h48qual이 이미 포지션을 들고 있어서 막힌" 765건의 케이스는 flat 루프 자체가 실행이 안 되니
   순서와 무관하다. **잘못된 지렛대였음이 확인됨.**
3. h48qual max-hold cap(150/250/400 bar) 테스트: 그리드가 비단조({150:나쁨, 250:PnL
   77→117%·MDD 개선, 400:나쁨})였고, 250에서도 zig075 거래수는 오히려 21→20으로 줄어 원래
   가설(zig075 기회 회복)이 검증되지 않았다. 게다가 "time exit" 재튜닝은
   `research_line_registry.json`의 `global_exit_constant_tuning`(21회+ 실패) 축과 정확히
   겹쳐 기각.

두 대안 다 15.8%의 기회비용을 회수하지 못했다 — **슬롯 자체를 늘리는 것 외에 남은 지렛대가
없다.**

## 메커니즘 설계

```text
기존: 열린 포지션 최대 1개, PRIORITY=(h48qual, zig075) 순서로 먼저 조건 만족한 쪽이 슬롯 차지.

신규: 열린 포지션 최대 2개(h48qual 1개 + zig075 1개, 컴포넌트당 최대 1개는 불변).
- 반대 방향 동시 보유 케이스는 실측상 0건이므로 별도 넷팅 로직 불필요(발생 시 정책 미정, 미해결 이슈 참고).
- 사이징: 각 컴포넌트는 자기 몫의 HGB risk sidecar가 산출한 margin_fraction/leverage를
  그대로 쓴다(변경 없음) — 단, 집계 notional이 기존 NOTIONAL_CAP(1.8)을 넘지 않도록 제약한다:
    first_notional  = min(first_margin * first_leverage_scaled, NOTIONAL_CAP)          # 기존과 동일
    remaining_budget = NOTIONAL_CAP - first_notional                                    # 신규
    second_notional  = min(second_margin * second_leverage_scaled, remaining_budget)    # 신규 — 남은 예산 이하로만
    (remaining_budget <= 0이면 기존과 동일하게 두 번째 진입 스킵 — 즉 max_concurrent=1로 두면
    이 신규 로직은 항상 remaining_budget=0이 되어 기존 동작과 정확히 같아진다)
- LEVERAGE_CAP은 각 포지션에 개별 적용(불변, 집계 레버리지 캡은 두지 않음 — notional 집계 캡이
  간접적으로 레버리지 총합도 제한).
- Exit: 컴포넌트별로 완전히 독립 — 기존 TP/SL/exit_head/h48qual 레짐가드 로직 무변경, 두 포지션을
  동시에 추적만 한다.
```

## Layer Contract

| Layer | Input | Output | Artifact |
|---|---|---|---|
| L5 신규(동시-슬롯 배분) | 현재 열린 포지션 수·컴포넌트, 신규 신호의 sidecar margin/leverage | `remaining_budget` 기반 notional 제약 | 신규 replay 엔진(리소스 레지스트리에 등록) |

## Cost/Risk Assumptions

- CLAUDE.md Futures Risk Sizing Contract 준수: `notional = margin_fraction * leverage` 관계는
  각 포지션 내부에서 불변 — 집계 캡은 notional 합에만 적용, margin_fraction/leverage 개별
  재계산 없음.
- Fee/slip: 기존과 동일, 포지션별로 독립 부과(Odyssey4와 동일 회계).
- 전체 gross/net notional 상한(NOTIONAL_CAP=1.8)은 불변 — 이 후보는 리스크 총량을 늘리지
  않고 "누가 그 총량을 쓸 자격이 있는가"만 두 컴포넌트로 나눈다.

## Red Team Gates

- [ ] **G0 회귀**: `max_concurrent=1`로 설정하면 Odyssey4 G0(6개 창)과 정확히 일치해야 한다.
- [ ] 신규 상태(두 번째 포지션의 entry/exit)가 causal한지(미래 데이터 없음) 확인.
- [ ] VAL 캘리브레이션(있다면 그리드 크기 최소화) → OOS-Q1+OOS-Q2 단일터치.
- [ ] N≥5 시드: **해당 없음**(결정론적 규칙, 학습 없음).
- [ ] G0 대비 비교: PnL/MDD 트레이드오프를 정직하게 보고.

## 필수 저비용 게이트 (cheap_gate, 전체 엔진 구현 전 먼저 통과)

전체 동시-2슬롯 replay 엔진(포지션 2개를 동시 추적하는 새 상태기계)을 만들기 전에, **이미
계산해 둔 자산을 재사용**해서 상한선(upper bound) 추정부터 한다: `docs/experiments/
eth_candidate_conformal_veto_episode_labels_20260816.md`의 zig075 episode 라벨(각 episode를
"단독으로 진입했다면" 벌어졌을 causal 시뮬레이션 결과, `full`=fee/slip 반영 net 가격변동률)을
h48qual의 실제 보유구간과 다시 대조해서, **h48qual에 막힌 episode들만 모아 그 counterfactual
수익 합을 본다.** 이건 실제 집계 notional 제약·컴포넌트 간 상호작용을 무시한 순수 상한선이라
과대추정이지만, 이게 작거나 음수면 전체 엔진을 만들 이유가 없다는 것만은 확실히 걸러낸다.

## 미해결 이슈

1. **반대 방향 동시 신호 정책 미정** — 실측 0건이라 지금은 발생 안 하지만, 새 데이터에서
   발생할 가능성은 배제 못 함. 발생 시: (a) 먼저 진입한 쪽 유지 + 나중 신호 스킵(기존과
   유사) vs (b) 순노출(net exposure)만 계산해 넷팅 중 하나를 정해야 함 — cheap_gate 통과 후
   결정.
2. **두 번째 진입자의 남은 예산이 아주 작을 때(예: NOTIONAL_CAP의 5%) 진입을 허용할지** —
   너무 작은 노출은 수수료 대비 무의미할 수 있음. 최소 notional 하한 추가 여부는 cheap_gate
   결과를 보고 결정.
3. **exit_head/TP-SL이 두 포지션에서 같은 bar에 동시에 발동할 때 처리 순서** — 독립 추적이라
   원칙적으로 문제없어야 하지만 구현 시 명시적으로 테스트 필요.

## 다음 단계

1. cheap_gate(재사용 라벨 기반 상한선 추정) 먼저 실행.
2. 상한선이 유의미하게 양수면 실제 동시-2슬롯 replay 엔진 구현(G0 회귀 확인 필수) → VAL →
   OOS 단일터치.
3. 결과는 `docs/experiments/eth_candidate_dual_slot_concurrent_positions_<date>.md`에 기록.
