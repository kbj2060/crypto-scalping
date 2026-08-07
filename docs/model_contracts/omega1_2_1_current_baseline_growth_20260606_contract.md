# Omega1.2.1 Current Baseline Growth Contract - 2026-06-06

## Status

- Alias: `omega1.2.1`
- Model family: `omega1_2_1_current_baseline_growth_20260606`
- Parent baseline: `omega1_2_true_3head_tabm_20260603_final_tp_sl_on_e28_exit30k_q080`
- Parent contract: `docs/model_contracts/omega1_2_true_3head_tabm_final_tp_sl_current_20260606_contract.md`
- Status: `research_growth_branch_not_live_promoted`

Omega1.2.1 does not replace the parent entry alpha. It records growth candidates on top of the current Omega1.2 final TP/SL baseline.

## Static Candidates

Balanced candidate:

- Name: `omega1_2_1_balanced_compensated_exposure_scale135_cap055`
- Transform: compensated TP/SL + exposure scale
- Scale / cap: `1.35 / 0.55`
- Validation: PnL `+61.14%`, MDD `-7.32%`, WR `63.64%`, trades `33`
- OOS: PnL `+45.31%`, MDD `-5.54%`, WR `72.22%`, trades `18`

Aggressive candidate:

- Name: `omega1_2_1_aggressive_compensated_exposure_scale200_cap090`
- Transform: compensated TP/SL + exposure scale
- Scale / cap: `2.00 / 0.90`
- Validation: PnL `+100.54%`, MDD `-10.68%`, WR `63.64%`, trades `33`
- OOS: PnL `+72.76%`, MDD `-8.11%`, WR `72.22%`, trades `18`

Raw notional-only scaling is rejected for this branch because it changes TP/SL hit geometry and caused validation trade explosion/collapse.

## Learned Exposure Selector Test

Script:

- `scripts/train_eval_omega1_2_1_exposure_selector_20260606.py`

Selector input:

- Parent 3-head routed Direction/Quality probabilities and confidence features.
- Router expert one-hot.
- Causal OHLC-derived return, ATR, range, EMA-gap, and time-of-day features.
- Parent side and fixed TP/SL/notional context.

Forbidden inputs:

- `clean_regime4_*`
- `regime4_pred_*`
- `tp_sl_action_score`
- `teacher_*`

Selector target:

- 2025 validation active-signal independent replay win/net labels.
- Validation ranking uses expanding OOF selector scores.
- OOS uses a selector refit on all validation active signals.

Risk transform:

- Apply exposure scale only when selector confidence is high.
- Notional is scaled and capped.
- TP/SL equity thresholds are multiplied by the same realized notional ratio to preserve the parent price-hit geometry.

Result:

- Best strict selector: `omega1_2_1_learned_extra_win_top40_scale200_cap090`
- Validation: PnL `+54.18%`, MDD `-5.47%`, WR `63.64%`, trades `33`
- OOS: PnL `+35.97%`, MDD `-4.14%`, WR `72.22%`, trades `18`
- Delta vs parent OOS: `+3.83pp` PnL with effectively unchanged MDD.
- Selector reliability warning: OOF win AUC is weak (`extra_win=0.3714`, `hgb_win=0.3570`, `hgb_net=0.3657`).

Decision:

- The learned selector is better than the parent baseline but worse than the static balanced candidate.
- Do not promote learned selector over `omega1_2_1_balanced_compensated_exposure_scale135_cap055`.
- Treat learned selector as a diagnostic showing that the current features do not yet separate high-confidence exposure upgrades well.

## Promotion Rule

Omega1.2.1 candidates are research-only until they pass:

- parent baseline replay reproduction,
- validation and OOS comparison against the parent,
- no forbidden feature audit,
- runtime-native parity if considered for `trading_bot.py`.
