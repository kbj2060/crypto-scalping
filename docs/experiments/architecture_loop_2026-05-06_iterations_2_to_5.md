# Architecture Improvement Loop 2-5

Date: 2026-05-06 KST

Final status: `shadow_not_promote`

## Baseline

Promotion baseline stayed fixed to `current_top_muzero_az_stage2_azexit_2026`:

| Metric | Baseline |
|---|---:|
| PnL | `752.648580%` |
| MDD | `-18.755787%` |
| Trades/day | `6.017045` |
| Avg leverage | `1.596029` |

Baseline reproduction passed exactly in loops 2-5 after switching Stage2 to:

```text
data/ensemble/supervised/zero_style/remaining_layers_walkforward/wf_stage2_sleeve_mz.pt
sha256=442a9dae1b46e94c4bd93d7123c437483d62e17590804337e010008988ae64b4
```

Stage3 and Stage4 remained excluded.

## Loop Results

| Loop | Model id | Report | PnL | MDD | Trades/day | Decision |
|---:|---|---|---:|---:|---:|---|
| 2 | `muzero_az_rank1_microadd_v2_2026` | `data/ensemble/reports/muzero_az_rank1_microadd_v2_2026.json` | `6.893035` | `-2.490148` | `2.198864` | reject |
| 3 | `muzero_az_rank1_flat_microadd_v3_2026` | `data/ensemble/reports/muzero_az_rank1_flat_microadd_v3_2026.json` | `730.302554` | `-18.755787` | `6.051136` | reject |
| 4 | `muzero_az_rank1_flat_microadd_v4_2026` | `data/ensemble/reports/muzero_az_rank1_flat_microadd_v4_2026.json` | `749.498517` | `-18.755787` | `6.119318` | reject |
| 5 | `muzero_az_rank1_flat_microadd_v5_2026` | `data/ensemble/reports/muzero_az_rank1_flat_microadd_v5_2026.json` | `753.653924` | `-18.755787` | `6.034091` | shadow only |

## Loop 5 Gate State

Loop 5 is the only credible near-miss:

| Gate | Result |
|---|---|
| Baseline reproduction | pass |
| PnL > baseline | pass, `+1.005344` |
| MDD better than baseline | fail, effectively tied but strict compare is slightly worse |
| Trades/day > baseline | pass, `+0.017045` |
| Avg leverage target | pass |
| Selected from validation-eligible config | pass |
| Exact flat-state audit | pass |
| Invariant audit | pass |
| Cost 1x/2x/3x survival | pass |
| Contract cost2 target | pass |
| Contract cost3 target | fail, `75.738535 < 75.84` |

Red Team and Model/Data Architect both concluded:

```text
Do not promote.
Allow paper/shadow tracking of loop 5.
Keep current rank-1 baseline as production candidate.
```

## Next Direction

Do not broaden micro-add further. The observed edge is small and cost-sensitive.

If another loop is authorized later, use a narrow `v5_cost3_mdd_tiebreak` refinement:

```text
cost3_pnl >= baseline_cost3_pnl + safety_margin
mdd <= baseline_mdd - epsilon
pnl > baseline_pnl
trades/day > baseline_trades/day
```

Treat changes below reporting precision as ties, not wins.
