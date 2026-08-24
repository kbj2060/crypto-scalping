# ETH 드로다운 거버너 후보 — L9.1/L9.2 순환청산 구현 및 ablation 결과 (2026-08-16)

상태: **구현 완료, ablation 3종 전부 채택 기준 미달.** 2종(equity_mdd_budget_stop,
profit_trailing_lock)은 재진입 처닝(churn)으로 파국적 붕괴, 1종(hard_loss_stop)은 최선의 설정에서도
cheap_gate가 이미 기각한 것과 같은 나쁜 PnL/MDD 트레이드오프. **근본 원인이 진단됐고, 다음 결정은
사용자 승인이 필요해 이 문서에서 결론짓지 않는다.** VAL만 사용, OOS 미개봉.

## 배경

`docs/model_contracts/eth_candidate_drawdown_budget_governor_contract_20260815.md`의 cheap_gate
결론에 따라(`docs/experiments/eth_candidate_drawdown_governor_cheap_gate_20260816.md`), 진입 전
스로틀보다 먼저 **보유 중 순환청산 3종**을 구현했다:

- `equity_mdd_budget_stop`: 포트폴리오 전체 mark-to-market peak 대비 현재 equity의 드로다운이
  임계값 이상이면 강제청산.
- `hard_loss_stop`: 현재 트레이드의 unrealized 계좌-PnL이 임계값 이하로 떨어지면 강제청산.
- `profit_trailing_lock`: 트레이드 자신의 최고 unrealized-PnL(mfe) 대비 giveback이 임계값 이상이면
  강제청산(기존 TP-상대적 `trailing_stop`과 독립적인 별도 메커니즘).

스크립트: `scripts/research_eth_candidate_drawdown_governor_intrabar_stops_20260816.py`. Odyssey4의
`greedy_replay_entry_veto`를 복사해 보유-중 분기에 3개 체크를 TP/SL보다 먼저(포트폴리오 리스크
예산이 개별 트레이드 판단보다 우선) 삽입했다. 계약의 미해결 이슈 5(그리드 과적합 위험)에 따라
**한 번에 한 메커니즘만 켜는 ablation**으로 설계했다(3+3+2=8회 재생, 3×3×2 전체곱 아님).

## G0 회귀 확인

3개 메커니즘 전부 비활성(`None`) 상태로 6개 창 전부 재생 → Odyssey4 G0 기준값과 정확히 일치
(`pass=True`, 전 창 no_gate/with_gate 모두 매치). 아래 ablation 결과를 신뢰할 수 있다.

## 결과 A: `equity_mdd_budget_stop` 단독 (VAL, with_gate)

| 임계값 | PnL | MDD | trades | 발동 횟수 |
|---:|---:|---:|---:|---:|
| 0.12 | −89.46% | −89.46% | 1077 | 1218 |
| 0.16 | −87.76% | −87.76% | 995 | 1131 |
| 0.20 | −88.32% | −88.32% | 942 | 1062 |

**파국적 붕괴.** 기준선(26건)의 40배가 넘는 거래가 발생했다. 직접 렛저를 열어 확인한 결과, 이
거래들의 **보유 기간 중앙값은 0 bar** — 즉 진입한 바로 그 bar에 즉시 청산된다:

```
entry_signal_i  entry_i  exit_i  side  component  reason                   trade_return
487             488      510     -1    h48qual    equity_mdd_budget_stop   -0.018012   (여기까지는 정상적인 드로다운 진입)
528             529      529     -1    zig075     equity_mdd_budget_stop   -0.001049   (hold=0)
531             532      532     -1    zig075     equity_mdd_budget_stop   -0.000390   (hold=0)
533             534      534     -1    zig075     equity_mdd_budget_stop    0.001560   (hold=0)
...             ...      ...     ...   ...        ...                      ...
```

**원인**: `peak`는 백테스트 전체에서 한 번도 리셋되지 않는 전역 최고치이고, `cash`(실현 자산)는
과거 손실을 영구히 반영한다. 487~510번 트레이드에서 실제 드로다운이 한 번 발생해 `cash`가
`peak` 대비 12~20% 낮은 수준으로 영구히 눌리자, **그 이후의 모든 신규 진입은 진입 직후(가격이
전혀 움직이기 전, `unreal≈0`)부터 이미 `equity_dd = 1 - cash/peak ≥ threshold`를 만족**한다.
그 결과 신호가 살아있는 매 bar마다 "진입 → 같은 bar 강제청산 → 다음 bar 재진입 → 강제청산"이
반복되며 수수료만 계속 깎아먹는다. 이건 튜닝으로 고칠 수 있는 임계값 문제가 아니라 **설계
자체의 구조적 결함**이다 — BTC 원본은 이 메커니즘을 진입 전 노셔널 스로틀(account/daily DD
캡)과 항상 함께 배치했는데(원본 그리드 어디에도 스로틀 없이 이 stop만 단독 배치한 설정이
없음), 이번 ablation은 의도적으로 단독 분리했기 때문에 이 결함이 드러났다.

## 결과 B: `hard_loss_stop` 단독 (VAL, with_gate)

| 임계값 | PnL | MDD | trades | 발동 횟수 |
|---:|---:|---:|---:|---:|
| 0.03 | 34.32% | −16.00% | 42 | 33 |
| 0.05 | 30.03% | −25.55% | 29 | 14 |
| 0.07 | 36.80% | −21.76% | 27 | 8 |

기준선(77.31%/−21.76%/26)과 비교. 세 설정 전부 PnL을 40~47pp 희생한다. 0.07은 MDD가 기준과
**완전히 동일**(발동 8회가 실제 최대낙폭 지점에 영향을 못 줌). 0.05는 MDD가 오히려 **악화**
(−21.76%→−25.55%, cheap_gate의 daily-loss-halt에서 봤던 것과 같은 슬롯 재배분 경로효과로
추정). 유일하게 MDD가 개선된 0.03(−21.76%→−16.00%, 5.76pp 개선)도 PnL을 42.99pp 지불한다 —
비율로 환산하면 MDD 1pp당 PnL 7.5pp, cheap_gate가 이미 기각한 NOTIONAL_CAP 인하(비율 7~16pp)와
같은 급의 나쁜 트레이드오프다. **판정: 세 설정 다 기각.**

## 결과 C: `profit_trailing_lock` 단독 (VAL, with_gate)

| activation/gap | PnL | MDD | trades | 발동 횟수 |
|---:|---:|---:|---:|---:|
| 0.03/0.02 | −17.28% | −31.05% | 51 | 31 |
| 0.05/0.03 | −5.01% | −22.62% | 43 | 18 |

**둘 다 기준선보다 PnL·MDD 동시에 악화** — 개선이 아니라 순수 손실이다. 거래 수가 기준(26건)의
1.7~2배로 늘어난 것도 결과 A와 같은 재진입 처닝 패턴을 시사한다(신호가 살아있는데 lock으로
조기청산되면 다음 bar에 재진입 가능 — 다만 A만큼 극단적이지 않은 건 `best_unreal`이 매 진입마다
0에서 다시 쌓여야 하므로 hold=0에서는 발동 불가능하기 때문으로 보인다).

## 근본 원인 (3종 공통 진단)

**신호가 여러 bar에 걸쳐 지속되는데, 거버너가 트레이드를 강제청산해도 그 신호를 소비하거나
쿨다운을 거는 메커니즘이 없다.** 그래서 flat 루프가 바로 다음 bar에 같은 신호로 재진입한다.
BTC 원본(`train_eval_clean_base_deep_drawdown_min_v4.py`)에는 이 위험이 없거나 약한데, 그건
원본이 이 세 stop을 **항상 진입 전 노셔널 스로틀과 함께** 배치했고(그리드 전체에 스로틀-only
설정이 없음), ETH 드로다운 거버너의 cheap_gate 단계는 오히려 진입 전 스로틀이 Odyssey에서 무력/역효과라는
걸 이미 보였다(`docs/experiments/eth_candidate_drawdown_governor_cheap_gate_20260816.md`) — 즉
**BTC 원본이 이 결함을 피할 수 있었던 바로 그 장치를, ETH 드로다운 거버너는 cheap_gate 결과에 따라 의도적으로
뒤로 미뤘다.** 두 조사 결과가 서로 충돌한다: cheap_gate는 "진입 전 스로틀은 이미 열린 포지션을
못 지키니 순환청산부터 하라"고 했는데, 순환청산은 "재진입을 막는 장치 없인 처닝으로 자멸한다."

## 종합 판단 — 갈림길 (사용자 결정 필요)

이식된 그대로는 3종 다 채택 기준 미달이다. 이건 임계값을 더 스윕한다고 해결되지 않는 구조적
문제이므로, 다음 두 방향 중 하나를 사용자가 선택해야 한다:

**(A) 재진입 쿨다운을 신규 메커니즘으로 추가해 재시도.** 거버너가 트레이드를 강제청산한 뒤
N bar 동안(또는 그 신호가 사라질 때까지) 같은 컴포넌트/방향 재진입을 금지. **주의**: 이건 BTC
원본에 없던 것을 새로 발명하는 것이다 — 계약의 "신규 자유변수 0개" 철학(Odyssey3/4가 지켰던
원칙)에서 벗어난다. 그래도 시도한다면 쿨다운 길이 자체가 새 하이퍼파라미터가 되므로 그리드를
더 작게 유지해야 한다.

**(B) 순환청산 축을 여기서 접는다.** cheap_gate(진입 전 스로틀)와 이 ablation(보유 중
순환청산) 둘 다 "이식 원본 그대로는 안 된다"는 결론에 도달했다 — ETH 드로다운 거버너 전체를, 최소한 BTC
drawdown-budget-governor를 그대로 포팅하는 형태로는, 재고할 시점일 수 있다.

이 문서는 (A)/(B) 중 하나를 결정하지 않는다 — 계약 문서와 메모리에 이 갈림길을 그대로
기록해뒀다.

## 아티팩트

- 스크립트: `scripts/research_eth_candidate_drawdown_governor_intrabar_stops_20260816.py`
- 리포트: `tmp/causal_regen_20260516/eth_candidate_drawdown_governor_intrabar_stops_20260816/report.json`
