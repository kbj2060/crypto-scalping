# State Option MoE 2026 V2 Contract

Status: `experimental_challenger`

## Purpose

Lift alpha from the audited clean base and SOMoE v1 without reusing the rejected MuZero/AZ rank-1 path as the promotion baseline.

## Data

- Train: `/home/llewyn/crypto-scalping/tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv`
- Validation split: rows before `2025-11-01` train; `2025-11-01` through `2025-12-31` validation.
- OOS: `/home/llewyn/crypto-scalping/tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv`
- Feature source: current SOMoE feature subset from `scripts/train_eval_state_option_moe_2026.py::FEATURE_COLS`.

## Layer IO

| Layer | Inputs | Outputs |
|---|---|---|
| State tokenizer | causal feature matrix | `state_token`, `state_distance` |
| Option catalog | side, notional, hold candidates | expanded LONG/SHORT option ids |
| Distributional critics | feature matrix + state + option params | q05/q50/q95, cost3, MAE, large-loss probability |
| Upside/risk selector | critic outputs + validation-selected config | side, notional, hold, utility |
| Execution risk profile | selected option stream | hard stop, trailing lock, daily loss/DD locks, global DD scaling |
| Accounting | fills with fee/slippage stress | PnL, MDD, trades/day, ledger |

## Promotion Reference

- Clean base PnL: `177.329809%`
- Clean base MDD: `-17.759665%`
- SOMoE v1 PnL: `362.656101%`

Leak-prone event label columns are dropped by default: `evt_candidate_label, evt_candidate_side, evt_side_margin`.

V2 is allowed to be aggressive, but it must report cost 2x/3x and invariant audit separately.
