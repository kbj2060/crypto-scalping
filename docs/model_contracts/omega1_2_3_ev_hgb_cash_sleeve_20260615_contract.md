# Omega1.2.3 EV-HGB Cash Sleeve 20260615

## Status

`walkforward_pass_live_wired`

This model supersedes the Omega1.2.2 classifier cash sleeve as the next Omega
cash-sleeve candidate. It keeps the
`omega1_2_1_tp_runner_clean_repair_20260613` primary and replaces the
three-class cash-sleeve classifier with long/short expected-value regressors.

## Why This Replaces Omega1.2.2

Omega1.2.2 used a direct CASH/LONG/SHORT classifier. Its OOS fallback-only
performance was close to breakeven:

- fallback-only OOS PnL: `+0.30%`
- fallback trades: `22`
- WR: `54.55%`
- PF: `1.04`
- stop rate: `45.45%`

Omega1.2.3 uses an EV gate and trades fewer, more selective CASH rows:

- fallback-only OOS PnL: `+3.33%`
- fallback trades: `16`
- WR: `56.25%`
- PF: `1.33`
- stop rate: `43.75%`

The prior `ev_min=0.004` diagnostic produced a stronger OOS point result
(`+7.91%` fallback-only), but failed monthly walk-forward stability. The robust
candidate is therefore `ev_min=0.002`.

## Runtime Contract

- `model_id`: `omega1_2_3_ev_hgb_cash_sleeve_20260615`
- base primary: `omega1_2_1_tp_runner_clean_repair_20260613`
- sleeve family: HGB expected-value regressors
- long model: `HistGradientBoostingRegressor`
- short model: `HistGradientBoostingRegressor`
- sleeve risk: `base_tp026_sl014_n0405_h192`
- label `min_edge`: `0.002`
- execution EV gate: `ev_min=0.002`
- sleeve eligibility: primary must be CASH and no open position
- takeover rule: close fallback by `fallback_primary_takeover` when primary
  becomes active

## Feature Contract

Omega1.2.3 starts from the Omega1.2.2 cash-sleeve feature set and adds
CASH-specific context features:

- `cash_ret_sum_12`
- `cash_ret_sum_48`
- `cash_ret_vol_12`
- `cash_ret_vol_48`
- `cash_range_ratio_12_48`
- `tabm_dir_entropy`
- `tabm_long_short_gap`
- `tabm_abs_side_gap`
- `tabm_quality_side_gap`
- `tabm_quality_abs_gap`
- `time_since_primary_exit`
- `last_primary_active_len`
- `last_primary_side`

Forbidden active features remain fail-fast:

- `tp_sl_action_score`
- `teacher_*`
- `regime4_pred_*`
- `clean_regime4_*`
- `clean_regime_2024_unsup_v4_*`

## Selection And Verification

Candidate discovery:

- script: `scripts/train_eval_omega1_2_3_cash_sleeve_upgrade_20260615.py`
- report:
  `tmp/causal_regen_20260516/omega1_2_3_cash_sleeve_upgrade_20260615/report.json`
- tested candidates: `846`

Robust candidate confirmation:

- summary:
  `tmp/causal_regen_20260516/omega1_2_3_cash_sleeve_upgrade_20260615/robust_ev002_selected_summary.json`
- stress:
  `tmp/causal_regen_20260516/omega1_2_3_cash_sleeve_upgrade_20260615/ev_hgb_base_confirm_stress.json`

Monthly walk-forward:

- script:
  `scripts/walkforward_omega1_2_3_ev_hgb_cash_sleeve_20260615.py`
- report:
  `tmp/causal_regen_20260516/omega1_2_3_ev_hgb_cash_sleeve_walkforward_20260615/report.json`
- selected `ev_min=0.002` improved `3/4` folds
- total fold combo delta: `+6.10p`
- total fold fallback-only PnL points: `+4.06p`

## Metrics

Omega1.2.1 clean-repair baseline replay:

- validation: `+160.22%` PnL, MDD `-27.64%`, WR `59.46%`, trades `37`
- OOS: `+85.70%` PnL, MDD `-15.64%`, WR `66.67%`, trades `18`

Omega1.2.3 robust EV-HGB sleeve:

- validation: `+160.50%` PnL, MDD `-28.45%`, WR `59.57%`, trades `47`
- validation fallback-only: `+0.11%`, trades `10`, WR `60.00%`, PF `1.08`
- OOS: `+91.89%` PnL, MDD `-15.64%`, WR `61.76%`, trades `34`
- OOS fallback-only: `+3.33%`, trades `16`, WR `56.25%`, PF `1.33`

## Live Wiring

Wired into `trading_bot.py` through
`trading_bot_modules/omega1_2_3_cash_sleeve.py`.

- live bundle:
  `data/ensemble/supervised/omega1_2_3_ev_hgb_cash_sleeve_20260615/ev_hgb_cash_sleeve.joblib`
- activation env:
  `OMEGA123_CASH_SLEEVE_ENABLE=true` by default
- active branch: only after Omega1.2.1 primary returns CASH while no position is
  open
- fallback exits: `omega1_2_3_fallback_take_profit`,
  `omega1_2_3_fallback_stop_loss`, `omega1_2_3_fallback_max_hold`,
  `omega1_2_3_fallback_primary_takeover`
- verification: `py_compile`, bundle load, and synthetic inference passed on
  2026-06-15
