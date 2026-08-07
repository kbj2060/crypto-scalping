# Alpha3 Offline RL Exit Layer Contract - 2026-05-14

## Scope

This experiment adds a compact discrete offline-RL exit layer on top of Alpha3.

Alpha3 remains frozen:

- HGB parent
- Alpha2.1 teacher/L2 decision layer
- V21.2 jackpot runner/add-on layer
- frozen V27/V31 deep scout and rule exits
- existing entry sizing, leverage, cooldown, and entry execution contract

The RL layer is reduce-only. It is not allowed to create new entries or alter parent/deep ownership.

## Algorithm

Selected architecture: compact fitted-Q network with CQL-style conservatism.

- State: Alpha3 live position state, owner, side, hold, unrealized PnL, MFE/MAE, notional, Alpha3 decision summary, frozen V27 q values, effective TP/SL context, and current-bar features.
- Actions: `hold`, `baseline_exit2_pen05`, `exit0_pen0`, `exit1_pen0`, `exit2_pen0`, `exit3_pen0`, `exit4_pen0`.
- Training reward: counterfactual one-step reduce-only net reward for each exit placement arm, with a conservative penalty on non-baseline exit arms.
- Runtime selected policy: `placement_only_q_exit4_fallback`.

The selected runtime policy only calls Q at existing Alpha3 exit events. It does not promote validation-favorable early-exit policies, because those failed OOS and repeat the prior global-exit overlay failure mode.

## Split

- Train: 2025-07-01 through 2025-09-30.
- Selection: 2025-10-01 through 2025-12-31.
- OOS report: current `v31.DEFAULT_EVAL` file at execution time.
- `selection_uses_2026=false`.

Current execution warning: at run time, `tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv` contained `2026-01-01 00:00:00` through `2026-02-28 16:00:00`, not the full historical report horizon. Results below are valid for the current replay input, while the canonical Alpha3/front-run report remains the promotion benchmark when full 2026 data is restored.

## Current Replay Result

Artifacts:

- Model: `data/ensemble/supervised/alpha3_offline_rl_exit_layer_20260514/offline_rl_exit_q.pt`
- Summary: `data/ensemble/reports/alpha3_offline_rl_exit_layer_20260514_summary.json`
- Grid: `data/ensemble/reports/alpha3_offline_rl_exit_layer_20260514_grid.csv`
- Audit: `data/ensemble/reports/alpha3_offline_rl_exit_layer_20260514_audit.json`
- Dataset: `data/ensemble/reports/alpha3_offline_rl_exit_layer_20260514_dataset.json`

Current replay metrics:

| Candidate | Cost1 PnL | Cost1 MDD | Cost2 PnL | Cost3 PnL | Score |
|---|---:|---:|---:|---:|---:|
| Alpha3 baseline exit2 pen05 | +358.84% | -27.59% | +283.14% | +215.42% | 541.22 |
| Fixed front-run exit4 pen0 | +369.62% | -27.14% | +285.00% | +218.20% | 553.83 |
| Offline-RL placement-only | +356.79% | -27.81% | +316.13% | +242.24% | 561.99 |
| Validation-best early RL | -1.83% | -35.33% | -33.28% | -54.78% | -45.60 |

## Decision

Do not promote RL timing/early-exit policies.

The offline-RL placement-only layer improves cost-stressed PnL on the current replay, but loses cost1 PnL and MDD versus fixed `exit4_pen0`. It is therefore a shadow candidate only, not the Alpha3 production replacement.

For the current Alpha3 branch, the best actionable exit improvement remains the simpler model-wide fixed front-run layer (`exit4_pen0`) until full-horizon data and real L2 queue/partial-fill validation prove that RL placement adds stable value.

## Audit Notes

- Backtest still uses 5m high/low immediate-limit touch proxy.
- Real post-only reject, queue position, partial fill, and L2 replay behavior are not modeled.
- Dataset is small: 258 train episodes and 10,044 position states.
- The learned target distribution is dominated by `hold`; early-exit selection overfits validation and collapses OOS.
- Promotion requires comparison against both canonical Alpha3 `+747.76% / -27.37%` and fixed front-run `+792.42% / -26.91%` on the same restored full 2026 input.
