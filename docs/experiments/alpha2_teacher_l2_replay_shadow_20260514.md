# Alpha2 Teacher + L2 Replay Shadow Experiment

## Result

`Teacher-Constrained Deep Parent Overlay + L2 conservative replay` is promoted to the `alpha2` shadow candidate.

It is not a live trading promotion. The Red Team verdict is `shadow_collect_l2` because the large PnL lift depends on conservative maker replay assumptions that still need enough forward L2 snapshots.

## Baselines

| Model | Cost1 PnL | MDD | Cost2 PnL | Cost3 PnL |
|---|---:|---:|---:|---:|
| Alpha1 taker baseline | +361.19% | -31.74% | +88.74% | +0.58% |
| Alpha1 L2 replay only | +642.43% | -30.54% | +434.61% | +402.96% |
| Alpha2 shadow | +699.14% | -29.72% | +463.54% | +420.80% |

## Interpretation

The teacher overlay removes a small number of weaker parent trades while preserving parent CASH bars, so the V27 deep scout still has room to operate. The L2 replay router is the main cost durability source; the teacher overlay adds incremental PnL and MDD improvement on top.

## Audit

- `selection_uses_2026`: false
- Train window: 2025-01-01..2025-09-30
- Selection window: 2025-10-01..2025-12-31
- OOS: fixed 2026 after selection
- V27 deep scout: preserved
- V21.2 jackpot runner: preserved
- V31 exit: preserved
- Verdict: `shadow_collect_l2`

## Next Improvement Path

1. Keep collecting live L2 `orderbook_decision_snapshots`.
2. Rebuild the replay layer with real fill statistics once there are enough decision snapshots and actual action rows.
3. Test `alpha2.1` as a teacher runtime risk sweep before changing deep architecture.
4. Do not increase notional until maker fill assumptions are validated with live shadow data.

## Alpha2.1 Runtime Sweep

Script: `scripts/eval_alpha2_teacher_l2_runtime_sweep_20260514.py`

This kept the deep teacher checkpoint, HGB parent, V27 scout, V21.2 runner, V31 exit, and L2 replay mechanism fixed. It only swept teacher confidence and parent notional scaling on 2025Q4 selection.

Selected OOS candidate:

| Model | Cost1 PnL | MDD | Cost2 PnL | Cost3 PnL |
|---|---:|---:|---:|---:|
| Alpha2 reference | +699.14% | -29.72% | +463.54% | +420.80% |
| Alpha2.1 `c0.56 parent_scale1.10` | +718.70% | -26.66% | +443.82% | +360.15% |

Decision: do not replace Alpha2 shadow. Alpha2.1 is useful as an aggressive branch because cost1 PnL and MDD improved, but the combined score worsened due to lower cost2/cost3 resilience.

## Defensive Mode Memory

`hgb_meta_task_attention_focal` from `alpha2_1_teacher_arch_ablation_extra_20260514` is recorded as a defensive-mode candidate only.

| Candidate | Cost1 PnL | MDD | Cost2 PnL | Cost3 PnL | Use |
|---|---:|---:|---:|---:|---|
| Alpha2.1 reference | +718.70% | -26.66% | +443.82% | +360.15% | main/aggressive reference |
| HGB meta task-attention focal | +223.63% | -15.34% | +148.04% | +127.16% | defensive/risk-off candidate |

Do not promote this candidate as the main model without a new selection result that beats the Alpha2/Alpha2.1 reference on the project score. It is useful as a future drawdown recovery or risk-off sleeve candidate.

## Default Execution Contract Memory

`next_open_limit_offset2_entry_fallback_fee20` is now the preferred execution contract for future Alpha2.1-style tests and live routing:

- post-only limit first
- passive offset `2 bps`
- entry miss -> market fallback
- exit miss -> market fallback
- maker fee multiplier in backtest `0.20`
- maker slippage `0`

Latest OOS comparison:

| Execution | Cost1 PnL | MDD | Cost2 PnL | Cost3 PnL |
|---|---:|---:|---:|---:|
| next-open taker | +354.53% | -32.45% | +23.13% | +10.80% |
| old L2 replay fee20 | +718.70% | -26.66% | +443.82% | +360.15% |
| post-only limit + market fallback | +747.76% | -27.37% | +510.83% | +436.68% |

Use this as the default execution comparison branch, but keep `taker-only` and `old synthetic L2` as mandatory controls until real L2 queue/fill validation is available.
