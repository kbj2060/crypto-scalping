# Portfolio RL Gate 2-Action - 2026-07-08

## Contract

- Router action space: `0=SKIP`, `1=TAKE_TOP`.
- Rule router first selects one top candidate per timestamp from ETH/SOL/BTC.
- RL does not create entries, change exits, or change sizing.
- Training uses validation only; OOS is reported once after training.
- Implementation is offline Fitted Q Iteration with a ridge linear Q-function.

## Results

| policy | split | PnL | MDD | trades | WR |
|---|---|---:|---:|---:|---:|
| rule_take_all | validation | 35.02% | -11.29% | 25 | 44.00% |
| rule_take_all | oos_extended | 30.92% | -22.43% | 36 | 38.89% |
| rule_take_all | oos_frozen_q1_2026 | 33.70% | -10.41% | 18 | 44.44% |
| rl_gate | validation | 72.67% | -10.56% | 23 | 47.83% |
| rl_gate | oos_extended | 70.26% | -20.51% | 29 | 48.28% |
| rl_gate | oos_frozen_q1_2026 | 58.56% | -14.62% | 18 | 50.00% |

## Notes

- Validation events: `60`.
- OOS events: `87`.
- RL validation selected trades: `23`.
- RL OOS selected trades: `29`.

This is a research prototype, not a promotion-grade live router. The dataset is small, so the policy must be red-teamed before any live use.

## Red-Team Audit

Audit artifact:

- `tmp/causal_regen_20260516/portfolio_rl_gate_2action_20260708/redteam_audit.json`
- `docs/audits/portfolio_rl_gate_2action_redteam_20260708.md`

Audit result:

- Promotion pass: `false`
- Blockers: `4`

Blocking reasons:

- `fresh_forward_bar_by_bar=false`
- `trade_ledgers_used_as_input=true`
- `saved_parent_exit_timestamps_used=true`
- `promotion_grade=false`

Ledger integrity checks passed:

- selected-position overlap count is `0` for validation and OOS
- notional identity errors are numerical noise only
- selected RL rows satisfy `q_take > q_skip`

## Native Fresh-Forward Requirement

To become promotion-grade, this router must be moved below the per-asset
candidate-decision layer and above the per-asset exit simulator:

1. At each 5m bar, generate ETH/SOL/BTC candidate decisions from causal
   features only.
2. Apply the rule top-candidate selector at that bar.
3. Evaluate the frozen 2-action RL gate from causal portfolio state only.
4. If `TAKE_TOP`, open the selected asset with its own existing margin,
   leverage, TP, SL, and exit-head contract.
5. While a position is open, advance one 5m bar at a time and close only when
   the selected asset's live-equivalent TP/SL/exit-head/time-exit condition
   fires.
6. Do not read any saved candidate trade ledger, saved exit timestamp, or
   precomputed parent trade outcome during routing.

Until that native integration exists, the results above should be treated as
an event-level research signal only, not as live-promotion evidence.
