# ETH 동시-2슬롯 포지션 후보 — cheap_gate 결과 (2026-08-16)

상태: **기각. 상한선 추정 자체가 뚜렷하게 음수 — 전체 replay 엔진을 만들 필요도 없이 여기서
접는다.** 공식 계보 아님(성과 없음). OOS 미개봉.

## 배경

`docs/model_contracts/eth_candidate_dual_slot_concurrent_positions_contract_20260816.md`의
cheap_gate: 전체 동시-2슬롯 엔진을 새로 만들기 전에, 이미 계산해 둔 zig075 episode 라벨
(conformal veto 후보 산출물, causal 단독-시뮬레이션 `full` 순수익 — fee/slip 반영)을 h48qual의
실제 보유구간과 대조해서, **h48qual에 막힌 episode들만 모아 그 counterfactual 수익 합**을 상한선으로
추정했다. 스크립트: `scripts/research_eth_candidate_dual_slot_cheap_gate_20260816.py`.

## 결과

VAL 기준, zig075 episode 789건 중:
- h48qual에 막힘: **185건**
- zig075 자기 자신에 막힘(중복 보유 불가): 469건
- 자유(슬롯 비어 있었음): 135건

**h48qual에 막힌 185건의 counterfactual 수익 합: −149.91%**(price-move 단위 단순 합산, 계좌
PnL 아님) — 평균 −0.81%/건, 양수 비율 29.7%뿐. 참고로 같은 기간 h48qual 자신의 실제 실현
수익 합은 −1.83%, zig075의 실제 실현 수익 합은 +48.27%(계좌 PnL 단위, 직접 비교 불가 — 참고용).

## 판단

이건 notional 제약도, 복리 효과도 다 무시한 **가장 관대한 상한선**인데도 뚜렷하게 음수다.
집계 notional 제약까지 반영한 실제 엔진에서는 이보다 더 나빠질 일만 있지 좋아질 일은 없다 —
전체 엔진을 만들 이유가 없다.

**왜 막힌 episode들이 유독 나쁜가**: episode 밀도가 높고(중앙값 길이 1 bar) 자기상관이 심하다는
건 이미 conformal veto 후보에서 확인했다(`docs/experiments/
eth_candidate_conformal_veto_uniqueness_weights_20260816.md`). h48qual이 길게 보유하는 동안
"막힌" zig075 신호들은, 신선한 새 기회가 아니라 **같은 지속적 신호가 h48qual의 긴 보유기간
내내 계속 재점화되는 것**에 가깝다 — 그리고 그 재점화가 계속된다는 것 자체가 방향이 안
풀리고 있다는(=손실 중이라는) 신호일 가능성이 높다. 즉 h48qual의 긴 보유가 **의도치 않게
zig075의 가장 나쁜 재점화 신호들을 걸러내는 부수효과**를 내고 있었을 수 있다.

## 결론 — "막힌 15.8% 기회비용" 조사 전체를 여기서 닫는다

우선순위 스왑(무효) → max-hold cap(노이즈, 닫힌 축과 충돌) → 동시-2슬롯(상한선 자체가 음수) —
세 가지 지렛대를 전부 시도했고 전부 부정적이다. "h48qual이 zig075의 기회를 막아서 손해"라는
원래 직관 자체가 틀렸을 가능성이 높다 — 오히려 그 반대(막힌 신호들이 원래 나쁜 신호였다)에
더 가까운 증거가 나왔다.

## 아티팩트

- 스크립트: `scripts/research_eth_candidate_dual_slot_cheap_gate_20260816.py`
- 리포트: `tmp/causal_regen_20260516/eth_candidate_dual_slot_cheap_gate_20260816/report.json`
