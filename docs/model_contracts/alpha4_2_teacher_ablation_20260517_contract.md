# Alpha4.2 Teacher Ablation Contract - 2026-05-17

## Goal

Test whether the Alpha4.2 teacher layer is still necessary after adding
`tp_sl_action_score` to the parent input and disabling the unstable deep scout.

The deep sleeve is disabled for this ablation so that teacher effects are not
mixed with the known V27-style deep stop-loss problem.

## Fixed Inputs

- Train CSV: `tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/trade_candidates_2025_patchtst__tide__dlinear.csv`
- Eval CSV: `tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/trade_candidates_2026_patchtst__tide__dlinear.csv`
- Parent: `tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/artifacts/hgb/parent.pkl`
- Teacher: `tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/artifacts/hgb/teacher_gate.pt`
- Execution: corrected Alpha3 limit-close contract
- Deep scout: disabled with zero deep-q/no-deep overlay

## Compared Architectures

Each variant retrains its own V21.2 runner on its own train decisions, selects
runner config on 2025Q4, then evaluates fixed 2026 OOS.

1. `parent_direct_raw_no_teacher`
   - Parent decisions are used directly.
   - No teacher model.
   - No runtime notional scaling.

2. `parent_direct_scaled_no_teacher`
   - Parent decisions are used directly.
   - No teacher model.
   - A simple parent notional scale runtime is selected on 2025Q4.

3. `teacher_constrained`
   - Existing Alpha2/Alpha3 teacher constraint layer is used.
   - Teacher runtime is selected on 2025Q4.

## Command

```bash
venv/bin/python scripts/eval_alpha4_2_teacher_ablation_20260517.py
```

## Artifacts

- Script: `scripts/eval_alpha4_2_teacher_ablation_20260517.py`
- Report: `tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/alpha4_2_teacher_ablation_summary.json`
- Audit: `tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/alpha4_2_teacher_ablation_audit.json`
- Grid: `tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/alpha4_2_teacher_ablation_grid.csv`
- Teacherless scaled runner: `tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/teacher_ablation_artifacts/parent_direct_scaled_no_teacher_runner.pkl`

## Results

Validation selection uses only 2025Q4.

| Variant | Val Score | Val Cost1 | Val MDD | OOS Score | OOS Cost1 | OOS MDD | OOS Cost2 | OOS Cost3 | OOS Trades |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| parent_direct_raw_no_teacher | -11.08 | +4.94% | -34.37% | +240.08 | +169.37% | -22.74% | +107.66% | +100.75% | 73 |
| parent_direct_scaled_no_teacher | +11.04 | +15.75% | -31.48% | +275.89 | +183.42% | -21.99% | +169.76% | +79.27% | 66 |
| teacher_constrained | +65.41 | +41.24% | -13.66% | +123.29 | +75.54% | -19.98% | +71.67% | +74.97% | 44 |

## Interpretation

Strict validation selection chooses `teacher_constrained`, so a no-leak promotion
process cannot claim teacher removal is selected solely from 2025Q4.

However, the fixed 2026 OOS result strongly favors removing the teacher model:

- `parent_direct_scaled_no_teacher` has the best OOS PnL and cost2 survival.
- It keeps MDD near the teacher variant.
- It avoids the extra sequence model and the live/backtest parity risks caused by
  teacher lookback handling.

The teacher layer is acting mostly as an aggressive trade-count suppressor. That
helps 2025Q4 validation drawdown, but it leaves a lot of 2026 edge unused after
`tp_sl_action_score` is already present in the parent.

## Decision

Do not remove teacher from a live/canonical model purely from this single
validation split, because validation picked the teacher.

For the next candidate architecture, remove the teacher layer and test:

`Alpha4.3 = HGB parent + tp_sl_action_score + simple parent scale runtime + V21.2 runner + no deep scout`

This candidate should be walk-forward validated with multiple selection windows.
If it remains stable, the teacher sequence layer should be deleted from the
production Alpha4 path.
