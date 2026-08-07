# Alpha7 Decontam Deep Alpha Control Sweep 2026-05-28

## Scope

- Active baseline: `alpha7_submodel_01965_decontam_v2_tp_20260528`
- Test path only; live path and active model artifacts were not modified.
- Frozen components: parent/v21_2, decision source, feature/data contracts, limit execution, Cost3 fee/slip.
- Variable component: `deep_alpha` fallback/scout controls only.

## Artifacts

- Script: `/home/llewyn/crypto-scalping/scripts/sweep_decontam_deep_alpha_controls_20260528.py`
- Grid: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/decontam_deep_alpha_controls_20260528/grid.csv`
- Summary: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/decontam_deep_alpha_controls_20260528/summary.json`
- Best OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/decontam_deep_alpha_controls_20260528/deep_stop_cd18_bear_long_veto_oos_cost3_ledger.csv`

## Result

| Variant | Val PnL | Val MDD | Val WR | OOS PnL | OOS MDD | OOS WR | OOS Trades | OOS Deep | OOS SL Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 109.74 | -16.59 | 0.510 | 162.28 | -17.99 | 0.439 | 107 | 88 | 0.131 |
| deep_stop_cd18 | 113.53 | -17.78 | 0.513 | 198.78 | -18.22 | 0.440 | 109 | 88 | 0.110 |
| deep_stop_cd18_bear_long_veto | 94.03 | -34.05 | 0.483 | 198.76 | -18.22 | 0.441 | 102 | 81 | 0.127 |
| deep_stop_cd18_dual_regime_veto | 44.34 | -27.40 | 0.431 | 133.68 | -20.30 | 0.430 | 86 | 67 | 0.140 |
| deep_stop_cd18_side_specialist_mild | 94.41 | -34.28 | 0.489 | 128.89 | -25.89 | 0.423 | 111 | 81 | 0.153 |
| deep_stop_cd18_side_specialist_no_short_veto | 97.92 | -35.11 | 0.482 | 128.22 | -25.89 | 0.431 | 109 | 79 | 0.156 |
| deep_stop_cd18_long_defensive | 88.29 | -34.31 | 0.476 | 127.84 | -25.89 | 0.434 | 106 | 76 | 0.170 |
| deep_stop_cd18_side_specialist | 40.16 | -30.69 | 0.423 | 96.09 | -25.89 | 0.418 | 91 | 63 | 0.154 |
| deep_stop_cd24 | 136.94 | -16.29 | 0.536 | 194.34 | -19.08 | 0.439 | 107 | 86 | 0.112 |
| deep_notional_050 | 82.45 | -18.88 | 0.538 | 120.62 | -15.68 | 0.476 | 103 | 81 | 0.078 |
| deep_short_only | 60.96 | -30.66 | 0.435 | 155.46 | -25.89 | 0.479 | 94 | 66 | 0.191 |
| deep_disabled | 46.21 | -23.90 | 0.326 | 100.78 | -25.89 | 0.353 | 34 | 0 | 0.294 |

## Interpretation

- `deep_stop_cd18` is best by OOS PnL/score: OOS PnL improves from `+162.28%` to `+198.78%`.
- `deep_stop_cd18_bear_long_veto` is a live-shadow risk patch prompted by a BEAR-regime paper LONG stop-out. OOS stays essentially flat (`+198.76%`) with fewer trades/deep entries, but validation MDD weakens sharply. Keep it shadow-only until live observations justify the veto.
- Side-specialist gates were tested as split LONG/SHORT thresholds and symmetric regime vetoes. They did not improve the model: the specialist variants reduced deep entries but damaged parent/deep sequencing and raised OOS MDD/SL ratio. Do not promote the side-specialist threshold versions.
- A learned LONG/SHORT specialist meta-veto was also tested. It trained separate `logreg_balanced` veto heads on 2025 `deep_alpha` entries (`LONG=64`, `SHORT=109`) using V31 q-values, state24 regime, regime4_pred, flow, and `tp_sl_action_score`. It overfit: validation improved, but 2026 OOS Cost3 fell to `91.07%` with MDD `-25.89%` and SL ratio `15.45%`. Do not promote the learned meta-veto version.
- A neural LONG/SHORT specialist was tested with a shared MLP trunk and side-specific heads. Training used all 2025 V31 edge/margin candidate bars (`12,822` rows) rather than only executed trades. It was better than the logistic veto but still below baseline: 2026 OOS Cost3 `150.73%`, MDD `-23.07%`, SL ratio `12.38%`. Combining it with `bear_long_veto` improved to `166.22%`, but still trailed `deep_stop_cd18`/`bear_long_veto`. Do not promote the neural specialist.
- The first neural specialist dataset was audited and rejected as a training source because it duplicated every candidate index into both LONG and SHORT counterfactual rows, producing an artificial `50/50` side balance. A corrected chosen-side dataset was generated from the actual V31 selected side only: 2025 all rows `6,411` (`LONG=2,472`, `SHORT=3,939`), strict rows `3,268` (`LONG=1,289`, `SHORT=1,979`). The corrected shared-trunk, side-head NN trained on CUDA but did not improve OOS: `deep_stop_cd18_chosen_nn_side_specialist` reached 2026 OOS Cost3 `198.59%`, MDD `-17.95%`, WR `43.12%`, trades `109`, SL ratio `11.01%`; `chosen_nn_plus_bear_long_veto` reached `196.58%`, MDD `-17.95%`, WR `44.12%`, trades `102`, SL ratio `12.75%`. Baseline `deep_stop_cd18` remains better by PnL/score. Do not promote the chosen-side NN specialist.
- `deep_stop_cd24` is the more validation-stable candidate: Val PnL improves from `+109.74%` to `+136.94%` and Val MDD slightly improves.
- `deep_stop_cd06` and `deep_stop_cd12` are identical to baseline because the frozen overlay already has a deep cooldown near that range; extra cooldown below or equal to the existing guard has no behavioral effect.
- `deep_short_only` improves raw deep contribution but damages parent/v21_2 sequencing and MDD. It should not be promoted.
- `deep_notional_050` reduces MDD and SL ratio, but loses too much PnL. It is a defensive profile, not the main profile.
- `deep_disabled` confirms that deep_alpha is still needed as a sequencing/fallback component; removing it degrades both PnL and MDD.

## Candidate Recommendation

- Main candidate for next retest: `deep_stop_cd18`.
- Shadow-only diagnostic candidate: `deep_stop_cd18_bear_long_veto`.
- Conservative candidate: `deep_stop_cd24`.
- Do not promote: `deep_short_only`, `deep_threshold_*`, `deep_disabled`, notional-only reductions, `deep_stop_cd18_side_specialist*`, `deep_stop_cd18_dual_regime_veto`, `deep_stop_cd18_long_defensive`, `deep_stop_cd18_meta_logreg_balanced_train_selected`, `deep_stop_cd18_nn_side_specialist`, `deep_stop_cd18_nn_plus_bear_long_veto`, `deep_stop_cd18_chosen_nn_side_specialist`, `deep_stop_cd18_chosen_nn_plus_bear_long_veto`.

## Chosen-Side Specialist Retrain

Artifacts:
- Dataset builder: `/home/llewyn/crypto-scalping/scripts/build_deep_side_specialist_chosen_dataset_20260528.py`
- Train/eval script: `/home/llewyn/crypto-scalping/scripts/train_eval_deep_side_specialist_chosen_nn_veto_20260528.py`
- Output: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/deep_side_specialist_chosen_nn_veto_20260528/summary.json`

Dataset contract:
- One row per actual V31 selected side only.
- No LONG/SHORT counterfactual duplication.
- Side-specific aligned features are explicit, for example `side_state24_trend_alignment`, `side_pred_trend_alignment`, `side_net_taker_ratio`, and side-adjusted regime directional bias.
- Strict labels use `path_return >= +0.8%` as good and `path_return <= -0.4%` as bad; middle rows are excluded from strict training.

Result:

| Variant | OOS PnL | OOS MDD | OOS WR | Trades | Deep Entries | L/S | SL Ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| deep_stop_cd18 | 198.78 | -18.22 | 0.440 | 109 | 88 | 34 / 75 | 0.110 |
| deep_stop_cd18_chosen_nn_side_specialist | 198.59 | -17.95 | 0.431 | 109 | 87 | 35 / 74 | 0.110 |
| deep_stop_cd18_chosen_nn_plus_bear_long_veto | 196.58 | -17.95 | 0.441 | 102 | 80 | 23 / 79 | 0.127 |

Conclusion:
- The corrected data fixes the side-balance artifact and is suitable for future model research.
- The current chosen-side NN specialist is not a promotion candidate because it does not beat `deep_stop_cd18` on OOS PnL/score.

## Parent/Fallback Side-Specialist Retrain

Artifacts:
- Train/eval script: `/home/llewyn/crypto-scalping/scripts/train_eval_alpha7_01965_parent_fallback_side_specialists_20260528.py`
- Output: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_01965_parent_fallback_side_specialists_20260528/summary.json`
- Grid: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_01965_parent_fallback_side_specialists_20260528/grid.csv`

Design:
- `primary_parent` and `fallback_alpha43` were retrained in an experiment-only path.
- Each layer uses separate LONG and SHORT binary action heads.
- Each layer also uses side-specific risk bucket heads for `notional`, `leverage`, `take_profit`, `stop_loss`, `max_hold`, and `cooldown`.
- The active/live artifacts were not modified.

Training label distribution:
- Full label candidates: `CASH=11,145`, `LONG=905`, `SHORT=997`.
- The side-specialist heads therefore had enough non-counterfactual LONG/SHORT labels, but the trade labels are still sparse relative to CASH.

Parent/fallback-only Cost3 result:

| Variant | Selection | Val PnL | Val MDD | Val WR | OOS PnL | OOS MDD | OOS WR | OOS Trades |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_primary_fallback | baseline | 39.43 | -22.75 | 0.151 | 123.63 | -23.88 | 0.210 | 81 |
| both_side_specialized | best by validation, p=0.55/m=0.00 | 120.67 | -17.50 | 0.135 | 79.62 | -19.99 | 0.165 | 85 |
| primary_side_fallback_base | best by OOS, p=0.65/m=0.00 | 25.83 | -24.09 | 0.145 | 123.35 | -22.40 | 0.175 | 80 |

Conclusion:
- Side-specializing primary/fallback action and risk heads did not produce a robust promotion candidate.
- The validation-selected model overfit: validation PnL improved sharply, but OOS PnL fell from `123.63%` to `79.62%`.
- The best OOS side-specialist was essentially flat versus baseline (`123.35%` vs `123.63%`) and had weak validation. Do not promote primary/fallback side-specialist retrains.

## Full LONG/SHORT Specialist Stack With Router

Artifacts:
- Train/eval script: `/home/llewyn/crypto-scalping/scripts/train_eval_alpha7_01965_full_long_short_router_20260528.py`
- Output: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_01965_full_long_short_router_20260528/summary.json`
- Grid: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_01965_full_long_short_router_20260528/grid.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_01965_full_long_short_router_20260528/router_oos_cost3_ledger.csv`

Design:
- This is the corrected interpretation of the LONG/SHORT specialist request.
- The entire `deep_stop_cd18` stack was duplicated into two specialist paths:
  - LONG stack: LONG-only `primary_parent`, LONG-only `fallback_parent`, and LONG-only deep-alpha permission.
  - SHORT stack: SHORT-only `primary_parent`, SHORT-only `fallback_parent`, and SHORT-only deep-alpha permission.
- A learned middle router selects `LONG stack` or `SHORT stack`.
- Router inputs include both specialists' action/confidence/quality/risk outputs, `deep_q_long`, `deep_q_short`, `deep_q_margin`, state24 regime probabilities, `regime4_pred_*`, flow, and volatility features.
- Active/live artifacts were not modified.

Training distribution:
- LONG specialist parent/fallback labels: `CASH=12,142`, `LONG=905`.
- SHORT specialist parent/fallback labels: `CASH=12,050`, `SHORT=997`.
- Router was first tested as `CASH/LONG/SHORT`, but it collapsed to mostly CASH. It was then corrected to a binary `LONG stack` vs `SHORT stack` router trained only on non-CASH labels (`LONG=905`, `SHORT=997`).

Cost3 result:

| Variant | Split | PnL | MDD | WR | Trades | Deep Entries | L/S | SL Ratio |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| deep_stop_cd18 baseline | val | 113.53 | -17.78 | 0.513 | 195 | 173 | 71 / 124 | 0.118 |
| deep_stop_cd18 baseline | oos | 198.78 | -18.22 | 0.440 | 109 | 88 | 34 / 75 | 0.110 |
| full long/short router | val | 27.56 | -23.28 | 0.459 | 148 | 112 | 79 / 69 | 0.182 |
| full long/short router | oos | 92.50 | -22.93 | 0.416 | 89 | 58 | 58 / 31 | 0.146 |

Conclusion:
- The requested architecture was implemented and tested, but it is not a promotion candidate.
- The binary router over-shifted OOS toward LONG (`58 / 31`) and damaged both PnL and MDD.
- Compared with baseline, OOS PnL fell from `198.78%` to `92.50%`, MDD worsened from `-18.22%` to `-22.93%`, and SL ratio worsened from `11.01%` to `14.61%`.
- The next viable direction is not duplicating the full stack by side, but improving router labels/objective so it optimizes realized stack-level PnL rather than inherited single-bar action labels.

### Stack-PnL Router Relabel

The middle router was retrained again with stack-level counterfactual PnL labels:
- For each train timestamp, the LONG stack and SHORT stack were each evaluated as a one-trade counterfactual with Cost3 fees/slippage.
- Router labels became `CASH`, `LONG stack`, or `SHORT stack` based on the higher net stack-level PnL.
- A return-regression router was also trained to predict `LONG stack return` and `SHORT stack return` directly, then threshold/margin swept on validation.

Counterfactual training label distribution:
- Inherited parent labels: `CASH=11,145`, `LONG=905`, `SHORT=997`
- Stack-PnL labels: `CASH=8,472`, `LONG=4,037`, `SHORT=538`

Result:

| Router | Selection | Val PnL | Val MDD | Val WR | OOS PnL | OOS MDD | OOS WR | OOS Trades | L/S |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deep_stop_cd18 baseline | baseline | 113.53 | -17.78 | 0.513 | 198.78 | -18.22 | 0.440 | 109 | 34 / 75 |
| stack-PnL classifier | fixed | -30.74 | -35.13 | 0.377 | -14.23 | -26.75 | 0.483 | 29 | 27 / 2 |
| stack-return regressor | best by validation | 62.89 | -20.19 | 0.440 | 79.57 | -25.32 | 0.439 | 82 | 16 / 66 |
| stack-return regressor | best by OOS, not selectable | 43.66 | -26.64 | 0.442 | 129.71 | -25.32 | 0.466 | 88 | 341 / 13773 route rows |

Conclusion:
- Stack-level relabeling fixed the earlier CASH collapse but still did not beat baseline.
- The return-regression router is better than the stack-PnL classifier, but validation-selected OOS remains far below `deep_stop_cd18`.
- The OOS-best threshold is not promotable because it is selected on OOS and validation is weak.
- Do not promote the full LONG/SHORT specialist stack or either router version.

## Router Overlay Refinements

Artifacts:
- Train/eval script: `/home/llewyn/crypto-scalping/scripts/train_eval_alpha7_01965_router_overlay_refinements_20260528.py`
- Output: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_01965_router_overlay_refinements_20260528/summary.json`
- Grid: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_01965_router_overlay_refinements_20260528/grid.csv`
- Short-override OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_01965_router_overlay_refinements_20260528/short_override_oos_cost3_ledger.csv`

Purpose:
- The full LONG/SHORT duplicated stack failed because the learned router over-shifted OOS toward LONG and damaged baseline sequencing.
- The follow-up test kept the strong `deep_stop_cd18` baseline intact and only added narrow router overlays:
  - `baseline_side_veto`: veto baseline entries when specialist side confidence is too weak.
  - `portfolio_window_router`: choose baseline/LONG-stack/SHORT-stack by recent realized route portfolio performance.
  - `trade_opportunity_router`: route only when the specialist has a strong per-trade expected edge.
  - `short_override_overlay`: keep baseline by default, but allow a high-confidence SHORT specialist override.

Cost3 result:

| Variant | Split | PnL | MDD | WR | Trades | Deep Entries | L/S | SL Ratio | Score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deep_stop_cd18 baseline | val | 113.53 | -17.78 | 0.513 | 195 | 173 | 71 / 124 | 0.118 | 92.64 |
| deep_stop_cd18 baseline | oos | 198.78 | -18.22 | 0.440 | 109 | 88 | 34 / 75 | 0.110 | 176.69 |
| baseline_side_veto | val | 65.59 | -24.11 | 0.500 | 32 | 22 | 9 / 23 | 0.125 | 36.40 |
| baseline_side_veto | oos | 3.86 | -20.95 | 0.190 | 21 | 10 | 10 / 11 | 0.095 | -31.06 |
| portfolio_window_router | val | 64.20 | -16.04 | 0.509 | 53 | 27 | 21 / 32 | 0.264 | 50.90 |
| portfolio_window_router | oos | 158.55 | -23.61 | 0.465 | 43 | 11 | 24 / 19 | 0.209 | 128.65 |
| trade_opportunity_router | val | -29.28 | -32.61 | 0.416 | 77 | 65 | 65 / 12 | 0.208 | -80.19 |
| trade_opportunity_router | oos | 64.93 | -12.97 | 0.535 | 43 | 32 | 33 / 10 | 0.140 | 59.09 |
| short_override_overlay | val | 86.99 | -18.80 | 0.505 | 200 | 176 | 72 / 128 | 0.125 | 63.60 |
| short_override_overlay | oos | 204.91 | -18.56 | 0.422 | 109 | 87 | 34 / 75 | 0.119 | 181.39 |

Conclusion:
- `baseline_side_veto` is unusable; it removes too many good baseline trades and collapses OOS PnL.
- `portfolio_window_router` improves validation MDD but degrades OOS PnL, MDD, and SL ratio. Do not promote.
- `trade_opportunity_router` improves OOS MDD and WR, but validation is strongly negative and OOS PnL is far below baseline. Do not promote.
- `short_override_overlay` is the only refinement that beats baseline OOS PnL/score (`204.91%` vs `198.78%`), but the edge is small and fragile: validation PnL is worse, OOS MDD is slightly worse, and SL ratio rises. It is a retest candidate only, not a live promotion candidate.

Next action:
- Run precision retest and monthly walk-forward only for `short_override_overlay`.
- Keep live/active path on the existing `deep_stop_cd18` candidate until the override proves stable across month splits and shadow/capped validation.

## Short-Override Precision Retest

Artifacts:
- Precision retest script: `/home/llewyn/crypto-scalping/scripts/precision_retest_alpha7_01965_short_override_20260528.py`
- Output: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_01965_short_override_precision_retest_20260528/summary.json`
- Threshold/period grid: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_01965_short_override_precision_retest_20260528/threshold_period_grid.csv`
- Monthly Cost3 grid: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_01965_short_override_precision_retest_20260528/monthly_cost3.csv`
- Selected OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_01965_short_override_precision_retest_20260528/selected_short_override_oos_cost3_ledger.csv`

Method:
- Only `short_override_overlay` was retested.
- Threshold selection was locked to validation full-period score.
- OOS was not used to choose the threshold.
- Monthly validation/OOS splits and capped-notional checks were added.
- Active/live artifacts were not modified.

Threshold sweep:

| Threshold | Val PnL | Val MDD | Val WR | Val Score | OOS PnL | OOS MDD | OOS WR | OOS Score | OOS Override Rows |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 69.80 | -18.80 | 0.498 | 46.07 | 226.97 | -13.71 | 0.430 | 213.53 | 33 |
| 0.55 | 72.82 | -18.80 | 0.500 | 49.23 | 196.22 | -18.56 | 0.418 | 172.52 | 27 |
| 0.60 | 82.75 | -18.80 | 0.500 | 59.16 | 196.22 | -18.56 | 0.418 | 172.52 | 25 |
| 0.625 | 82.75 | -18.80 | 0.500 | 59.16 | 204.91 | -18.56 | 0.422 | 181.39 | 21 |
| 0.65 | 86.99 | -18.80 | 0.505 | 63.60 | 204.91 | -18.56 | 0.422 | 181.39 | 19 |
| 0.675 | 86.99 | -18.80 | 0.505 | 63.60 | 204.99 | -18.56 | 0.422 | 181.48 | 16 |
| 0.70 | 86.99 | -18.80 | 0.505 | 63.60 | 204.99 | -18.56 | 0.422 | 181.48 | 14 |
| 0.75 | 86.99 | -18.80 | 0.505 | 63.60 | 204.99 | -18.56 | 0.422 | 181.48 | 7 |
| 0.80 | 113.53 | -17.78 | 0.513 | 92.64 | 205.30 | -18.48 | 0.431 | 182.31 | 5 |

Validation-selected threshold:
- `0.80`.
- At this threshold validation behavior is effectively identical to baseline.
- OOS improves from `198.78%` to `205.30%`, but the edge comes from only `5` OOS override rows.

Monthly Cost3 comparison for selected threshold:

| Variant | Period | PnL | MDD | WR | Trades | L/S | SL Ratio | Score |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | val 2025-10 | 27.36 | -17.78 | 0.426 | 61 | 21 / 40 | 0.180 | 7.02 |
| selected override | val 2025-10 | 27.36 | -17.78 | 0.426 | 61 | 21 / 40 | 0.180 | 7.02 |
| baseline | val 2025-11 | 50.65 | -13.22 | 0.548 | 73 | 29 / 44 | 0.096 | 43.94 |
| selected override | val 2025-11 | 50.65 | -13.22 | 0.548 | 73 | 29 / 44 | 0.096 | 43.94 |
| baseline | val 2025-12 | 6.89 | -15.11 | 0.532 | 62 | 22 / 40 | 0.097 | -3.91 |
| selected override | val 2025-12 | 6.89 | -15.11 | 0.532 | 62 | 22 / 40 | 0.097 | -3.91 |
| baseline | oos 2026-01 | 16.98 | -18.22 | 0.381 | 42 | 9 / 33 | 0.190 | -5.47 |
| selected override | oos 2026-01 | 19.54 | -18.48 | 0.357 | 42 | 9 / 33 | 0.214 | -4.40 |
| baseline | oos 2026-02 | 73.74 | -27.23 | 0.440 | 75 | 33 / 42 | 0.107 | 34.63 |
| selected override | oos 2026-02 | 73.74 | -27.23 | 0.440 | 75 | 33 / 42 | 0.107 | 34.63 |

Capped shadow check:

| Variant | Split | Cap | PnL | MDD | WR | Trades | SL Ratio | Score |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | val | 0.25 | 7.73 | -7.24 | 0.530 | 151 | 0.026 | 9.92 |
| selected override | val | 0.25 | 7.73 | -7.24 | 0.530 | 151 | 0.026 | 9.92 |
| baseline | oos | 0.25 | 11.38 | -3.98 | 0.547 | 86 | 0.023 | 22.71 |
| selected override | oos | 0.25 | 12.69 | -3.97 | 0.541 | 85 | 0.024 | 23.85 |
| baseline | val | 0.50 | 34.83 | -9.62 | 0.548 | 157 | 0.083 | 32.79 |
| selected override | val | 0.50 | 34.83 | -9.62 | 0.548 | 157 | 0.083 | 32.79 |
| baseline | oos | 0.50 | 14.54 | -10.05 | 0.484 | 93 | 0.054 | 11.02 |
| selected override | oos | 0.50 | 17.51 | -10.05 | 0.484 | 91 | 0.055 | 14.03 |

Conclusion:
- The OOS improvement at threshold `0.80` is too small and too sparse to promote.
- Threshold `0.50` has the best OOS result, but validation degrades sharply, so it is not selectable.
- The capped checks show the override does not introduce obvious notional-cap instability, but the incremental edge remains too small.
- Final decision: do not promote `short_override_overlay` to active/live. It can remain a shadow diagnostic only.

## Regime Direction Veto On Deep Stop CD18

Artifacts:
- Test script: `/home/llewyn/crypto-scalping/scripts/test_alpha7_deep_stop_cd18_regime_direction_veto_20260528.py`
- Output: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_deep_stop_cd18_regime_direction_veto_20260528/summary.json`
- Full grid: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_deep_stop_cd18_regime_direction_veto_20260528/grid.csv`
- Monthly grid: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_deep_stop_cd18_regime_direction_veto_20260528/monthly_cost3.csv`

Rule:
- Dominant regime source: `clean_regime4_state24_sticky090_v2_*`.
- BEAR regime: block LONG.
- BULL regime: block SHORT.
- Tested three scopes:
  - deep-only: only deep-alpha fallback entries are blocked.
  - parent-only: only parent/fallback decision rows are blocked.
  - global: both parent/fallback and deep-alpha entries are blocked.

Cost3 full-period result:

| Variant | Split | PnL | MDD | WR | Trades | Deep Entries | L/S | SL Ratio | Score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deep_stop_cd18 baseline | val | 113.53 | -17.78 | 0.513 | 195 | 173 | 71 / 124 | 0.118 | 92.64 |
| deep_stop_cd18 baseline | oos | 198.78 | -18.22 | 0.440 | 109 | 88 | 34 / 75 | 0.110 | 176.69 |
| deep-only regime veto | val | 44.34 | -27.40 | 0.431 | 144 | 109 | 44 / 100 | 0.125 | 2.44 |
| deep-only regime veto | oos | 133.68 | -20.30 | 0.430 | 86 | 67 | 27 / 59 | 0.140 | 107.72 |
| parent-only regime veto | val | 143.09 | -19.33 | 0.508 | 189 | 171 | 71 / 118 | 0.127 | 119.08 |
| parent-only regime veto | oos | 174.08 | -15.89 | 0.467 | 105 | 89 | 33 / 72 | 0.105 | 157.83 |
| global regime veto | val | 20.17 | -33.01 | 0.426 | 141 | 113 | 38 / 103 | 0.149 | -33.06 |
| global regime veto | oos | 119.42 | -16.35 | 0.476 | 82 | 68 | 26 / 56 | 0.134 | 103.29 |

Parent-veto row counts:
- Validation: `102` parent rows blocked (`44` BEAR/LONG, `58` BULL/SHORT).
- OOS: `370` parent rows blocked (`330` BEAR/LONG, `40` BULL/SHORT).

Monthly notes:
- Parent-only veto improves 2026-01 (`16.98% -> 29.67%`, MDD `-18.22% -> -13.35%`) but damages 2026-02 (`73.74% -> 19.09%`).
- Deep-only veto damages both OOS months versus baseline.
- Global veto improves some risk metrics but removes too much profitable sequencing and is far below baseline PnL.

Conclusion:
- Do not apply a hard symmetric regime-direction veto to deep-alpha. It blocks too many profitable counter-regime deep entries.
- Do not promote the global veto.
- Parent-only veto is a risk-control candidate, not a main candidate: it lowers OOS MDD and SL ratio but loses too much OOS PnL. If revisited, it should be softened with regime confidence/edge thresholds instead of a hard dominant-regime block.

## Soft Regime Direction Veto

Artifacts:
- Test script: `/home/llewyn/crypto-scalping/scripts/test_alpha7_deep_stop_cd18_soft_regime_veto_20260528.py`
- Output: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_deep_stop_cd18_soft_regime_veto_20260528/summary.json`
- Full grid: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_deep_stop_cd18_soft_regime_veto_20260528/grid.csv`
- Monthly grid: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_deep_stop_cd18_soft_regime_veto_20260528/monthly_cost3.csv`

Method:
- Hard dominant-regime veto was replaced with softer conditions.
- Deep soft veto:
  - If the position side is counter-regime and counter-regime probability is above `deep_conf`, block only when `deep_q_edge` and `deep_q_margin` are not strong enough.
  - This preserves strong counter-regime deep-alpha entries.
- Parent soft veto:
  - If the parent side is counter-regime and counter-regime probability is above `parent_conf`, block only weak parent rows.
  - Weak parent row means low model confidence or low quality score, depending on the variant.
- The effective state24 regime probabilities are capped around `0.715`, so earlier `0.80/0.90` gates are no-op conditions.

Cost3 full-period result:

| Variant | Val PnL | Val MDD | Val WR | Val Score | OOS PnL | OOS MDD | OOS WR | OOS Score | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 113.53 | -17.78 | 0.513 | 92.64 | 198.78 | -18.22 | 0.440 | 176.69 | current baseline |
| parent_soft_c65_conf70_q040_any | 112.86 | -17.78 | 0.513 | 91.97 | 250.39 | -18.22 | 0.463 | 229.24 | best OOS, validation slightly below baseline |
| parent_soft_c55_conf70_q040_any | 112.86 | -17.78 | 0.513 | 91.97 | 250.39 | -18.22 | 0.463 | 229.24 | same realized result as c65, blocks more rows |
| parent_soft_c65_conf70_q040_both | 113.53 | -17.78 | 0.513 | 92.64 | 198.78 | -18.22 | 0.440 | 176.69 | effectively no trade-level change |
| deep_soft_c70_e125_m125 | 72.24 | -24.44 | 0.505 | 37.82 | 113.38 | -18.52 | 0.427 | 90.13 | damages deep-alpha sequencing |
| deep_soft_c65_e110_m110 | 64.51 | -23.56 | 0.479 | 30.86 | 108.83 | -18.52 | 0.439 | 86.15 | damages deep-alpha sequencing |
| global_soft_c65 | 64.00 | -23.56 | 0.479 | 30.34 | 142.27 | -17.52 | 0.449 | 121.96 | deep damage offsets parent benefit |

Interpretation:
- Deep-level soft veto is still harmful. Even when strong counter-regime signals are allowed through, blocking weaker deep entries damages the sequence and reduces PnL.
- Global soft veto also fails because it inherits the deep veto damage.
- Parent-only soft veto is the only useful direction. `parent_soft_c65_conf70_q040_any` blocks weak counter-regime parent rows and keeps deep-alpha unchanged.
- The OOS full-period improvement is large (`198.78% -> 250.39%`) with WR improving (`0.440 -> 0.463`) and MDD unchanged, but validation score is slightly below baseline (`92.64 -> 91.97`). This is promising but not enough for active promotion under strict validation selection.

Monthly note:
- The parent-soft candidate is not uniformly better by reset-month PnL. It keeps 2026-01 unchanged and lowers reset-month 2026-02 PnL.
- The full OOS gain comes from sequence/compounding changes across the continuous OOS ledger, so it requires ledger-level precision retest before any promotion.

Decision:
- Do not promote deep or global soft veto.
- Keep `parent_soft_c65_conf70_q040_any` as the only new shadow/precision-retest candidate.
- Next validation should record the full OOS ledger and compare removed parent rows against the baseline ledger to confirm the gain is not a small sample artifact.

## Parent Soft Regime Veto Precision Retest

Artifacts:
- Precision script: `/home/llewyn/crypto-scalping/scripts/precision_retest_alpha7_parent_soft_regime_veto_20260528.py`
- Output: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_parent_soft_regime_veto_precision_20260528/summary.json`
- Grid: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_parent_soft_regime_veto_precision_20260528/grid.csv`
- Monthly grid: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_parent_soft_regime_veto_precision_20260528/monthly_cost3.csv`
- Blocked parent rows: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_parent_soft_regime_veto_precision_20260528/blocked_parent_rows.csv`
- Ledger diff summary: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_parent_soft_regime_veto_precision_20260528/ledger_diff_summary.csv`
- OOS baseline ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_parent_soft_regime_veto_precision_20260528/oos_baseline_cost3_ledger.csv`
- OOS candidate ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_parent_soft_regime_veto_precision_20260528/oos_parent_soft_cost3_ledger.csv`

Candidate rule:
- Name: `parent_soft_c65_conf70_q040_any`.
- Only parent/fallback decision rows are affected; deep-alpha is untouched.
- A parent row is blocked when:
  - side is counter-regime,
  - counter-regime probability is `>= 0.65`,
  - and either parent confidence `< 0.70` or quality score `< 0.040`.

Full-period Cost3:

| Variant | Split | PnL | MDD | WR | Trades | Deep | L/S | SL Ratio | Score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | val | 113.53 | -17.78 | 0.513 | 195 | 173 | 71 / 124 | 0.118 | 92.64 |
| parent soft | val | 112.86 | -17.78 | 0.513 | 195 | 173 | 71 / 124 | 0.118 | 91.97 |
| baseline | oos | 198.78 | -18.22 | 0.440 | 109 | 88 | 34 / 75 | 0.110 | 176.69 |
| parent soft | oos | 250.39 | -18.22 | 0.463 | 108 | 88 | 34 / 74 | 0.111 | 229.24 |

Blocked-row audit:
- Validation: `16 / 638` active parent rows blocked (`2.51%`), mostly BEAR/LONG (`14`) with `2` BULL/SHORT.
- OOS: `107 / 1020` active parent rows blocked (`10.49%`), mostly BEAR/LONG (`105`) with `2` BULL/SHORT.
- OOS blocked BEAR/LONG rows average counter-regime probability `0.7067`, confidence `0.6595`, quality `0.1110`, notional `2.15`.

Ledger-level attribution:

| Split | Common Trades | Baseline-Only | Candidate-Only | Baseline-Only Return | Candidate-Only Return | Blocked Parent Trades Actually Removed | Blocked Parent Return | Gross Return Delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| val | 194 | 1 | 1 | -0.0043 | -0.0074 | 1 | -0.0043 | -0.0031 |
| oos | 99 | 10 | 9 | 0.0936 | 0.2559 | 1 | 0.0881 | 0.1622 |

Important interpretation:
- The OOS improvement is not explained by directly removing a bad parent trade.
- The one executed blocked parent trade in baseline was actually profitable (`+0.0881` raw return).
- The candidate improves because blocking that parent entry changes the later sequence and creates different deep-alpha opportunities. Candidate-only trades add `+0.2559` raw return versus `+0.0936` raw return in baseline-only trades.
- This is a sequence/compounding effect, not a simple “bad trade removed” effect.

Monthly reset check:
- Validation monthly behavior is nearly identical and slightly worse in October.
- OOS January is unchanged.
- OOS February reset-month PnL is worse (`73.74% -> 32.75%`) even though continuous full OOS PnL is better.

Decision:
- Do not promote `parent_soft_c65_conf70_q040_any` to active/live.
- It remains a research/shadow candidate only.
- The OOS full-period lift is interesting, but because it comes from sequence path dependence after one profitable blocked parent trade, it is too artifact-prone for deployment without wider walk-forward validation.

## Parent Soft Regime Veto Walk-Forward

Artifacts:
- Walk-forward script: `/home/llewyn/crypto-scalping/scripts/walk_forward_alpha7_parent_soft_regime_veto_20260528.py`
- Output: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_parent_soft_regime_veto_walk_forward_20260528/summary.json`
- Grid: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_parent_soft_regime_veto_walk_forward_20260528/grid.csv`
- Blocked rows: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_parent_soft_regime_veto_walk_forward_20260528/blocked_parent_rows.csv`

Method:
- Runtime-parity frame construction was used:
  - `v31.DEFAULT_TRAIN/EVAL`,
  - state24 sidecar merge,
  - alpha7 feature augmentation,
  - same decision source and `deep_stop_cd18` backtest engine.
- A first raw-CSV attempt was discarded because it did not reproduce the known OOS baseline. The final script reproduces the known OOS baseline `198.78%`.
- This is still not untouched validation: 2025 is model development/training history, and 2026 OOS is the same January-February frame already used above.

Blocked-row summary:
- 2025 runtime frame: `248 / 9747` active parent rows blocked (`2.54%`), `182` BEAR/LONG and `66` BULL/SHORT.
- 2026 OOS frame: `107 / 1020` active parent rows blocked (`10.49%`), `105` BEAR/LONG and `2` BULL/SHORT.

Continuous full-period result:

| Variant | Period | PnL | MDD | WR | Trades | Deep | L/S | Score |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 2025 full | 60,324,547.17 | -23.01 | 0.745 | 564 | 437 | 222 / 342 | 60,324,514.03 |
| parent soft | 2025 full | 70,656,171.74 | -21.01 | 0.754 | 569 | 437 | 221 / 348 | 70,656,142.52 |
| baseline | 2026 OOS full | 198.78 | -18.22 | 0.440 | 109 | 88 | 34 / 75 | 176.69 |
| parent soft | 2026 OOS full | 250.39 | -18.22 | 0.463 | 108 | 88 | 34 / 74 | 229.24 |

Monthly/quarterly stability:

| Period | Baseline PnL | Parent Soft PnL | PnL Delta | MDD Delta | Score Delta |
|---|---:|---:|---:|---:|---:|
| 2025-01 | 241.08 | 241.08 | 0.00 | 0.00 | 0.00 |
| 2025-02 | 368.05 | 368.05 | 0.00 | 0.00 | 0.00 |
| 2025-03 | 369.26 | 368.87 | -0.39 | 0.00 | -0.39 |
| 2025-04 | 261.36 | 261.36 | 0.00 | 0.00 | 0.00 |
| 2025-05 | 519.64 | 601.40 | +81.76 | 0.00 | +85.24 |
| 2025-06 | 298.31 | 298.91 | +0.60 | +0.03 | +0.75 |
| 2025-07 | 318.78 | 318.78 | 0.00 | 0.00 | 0.00 |
| 2025-08 | 391.26 | 395.13 | +3.87 | 0.00 | +3.27 |
| 2025-09 | 111.00 | 111.00 | 0.00 | 0.00 | 0.00 |
| 2025-10 | 13.50 | 16.45 | +2.95 | +2.00 | +7.22 |
| 2025-11 | 50.65 | 50.65 | 0.00 | 0.00 | 0.00 |
| 2025-12 | 6.89 | 7.06 | +0.16 | 0.00 | +0.54 |
| 2025Q1 | 7,573.69 | 7,567.26 | -6.43 | 0.00 | -6.43 |
| 2025Q2 | 8,889.37 | 10,090.78 | +1,201.41 | 0.00 | +1,202.62 |
| 2025Q3 | 4,385.81 | 4,421.18 | +35.37 | 0.00 | +35.01 |
| 2025Q4 | 90.30 | 95.24 | +4.94 | +2.00 | +9.07 |
| 2026-01 | 16.98 | 16.98 | 0.00 | 0.00 | 0.00 |
| 2026-02 | 73.74 | 32.75 | -40.99 | +3.21 | -34.83 |
| 2026Q1/full | 198.78 | 250.39 | +51.62 | 0.00 | +52.55 |

Interpretation:
- The rule is not uniformly harmful on runtime-parity 2025 history; it is usually neutral and occasionally beneficial.
- The strongest 2025 improvement comes from Q2/May, again suggesting sequence/compounding effects rather than a simple per-trade filter.
- The same instability remains in 2026: continuous OOS full improves strongly, but monthly reset February worsens materially.
- The parent-soft rule is therefore path-dependent. It can improve a continuous ledger by changing later opportunities, but it is not a robust per-period edge.

Decision:
- Keep `parent_soft_c65_conf70_q040_any` out of live/active.
- It can be kept as a shadow experiment or as a feature for a future learned router, but not as a deterministic hard rule.
- A deployable version would need a path-aware validation target, for example a learned meta-router trained on realized sequence impact, not a static row-level veto.

## Parent Soft Path-Aware Meta Router

Artifacts:
- Script: `/home/llewyn/crypto-scalping/scripts/train_eval_alpha7_parent_soft_meta_router_20260528.py`
- Output: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_parent_soft_meta_router_20260528/summary.json`
- Grid: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_parent_soft_meta_router_20260528/grid.csv`
- Daily training frame: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_parent_soft_meta_router_20260528/daily_training_frame.csv`
- Router predictions: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_parent_soft_meta_router_20260528/router_predictions.csv`

Method:
- Runtime-parity frame construction and `deep_stop_cd18` backtest engine were kept unchanged.
- The router chooses, once per day, whether that day uses baseline decisions or `parent_soft_c65_conf70_q040_any` decisions.
- Router inputs are previous-day aggregates only, including realized market movement, active decision mix, blocked-active ratio, confidence/quality summaries, counter-regime pressure, state24 regime probabilities, and DeepAlpha Q summaries.
- Router labels are day-level realized sequence score deltas: `parent_soft_score > baseline_score`.
- Fit window: `2025-01-01` through `2025-09-30`.
- Validation window: `2025Q4`.
- OOS window: 2026 runtime eval frame.
- 2026 labels are diagnostic only and are not used for fitting or threshold selection.

Daily label base rate:
- 2025 fit: `3.30%` of days favor parent-soft.
- 2025Q4 validation: `3.26%` of days favor parent-soft.
- 2026 OOS diagnostic: `3.45%` of days favor parent-soft.
- Most days are exactly neutral because parent-soft does not change an executed path.

Cost3 results:

| Variant | Split | PnL | MDD | WR | Trades | Deep | L/S | SL Ratio | Score | Soft Days |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 2025Q4 | 90.30 | -23.01 | 0.500 | 198 | 174 | 69 / 129 | 0.126 | 58.34 | - |
| parent soft static | 2025Q4 | 95.24 | -21.01 | 0.503 | 197 | 175 | 70 / 127 | 0.127 | 67.41 | all |
| meta HGB t0.35 | 2025Q4 | 95.85 | -20.76 | 0.503 | 197 | 175 | 70 / 127 | 0.127 | 68.52 | 3 / 92 |
| prev blocked ratio t0.02 | 2025Q4 | 95.85 | -20.76 | 0.503 | 197 | 175 | 70 / 127 | 0.127 | 68.52 | 9 / 92 |
| baseline | 2026 OOS | 198.78 | -18.22 | 0.440 | 109 | 88 | 34 / 75 | 0.110 | 176.69 | - |
| parent soft static | 2026 OOS | 250.39 | -18.22 | 0.463 | 108 | 88 | 34 / 74 | 0.111 | 229.24 | all |
| meta HGB t0.35 | 2026 OOS | 198.78 | -18.22 | 0.440 | 109 | 88 | 34 / 75 | 0.110 | 176.69 | 3 / 58 |
| prev blocked ratio t0.02 | 2026 OOS | 250.39 | -18.22 | 0.463 | 108 | 88 | 34 / 74 | 0.111 | 229.24 | 8 / 58 |

Interpretation:
- The learned HGB router improved validation but failed to select the OOS days where parent-soft changes the executed path, so its OOS behavior equals baseline.
- Logistic variants also produced baseline-equivalent OOS behavior.
- The simple previous-day blocked-active-ratio gate matched the best validation score and reproduced the static parent-soft OOS improvement while using parent-soft on only a small number of days.
- This suggests the useful signal is not a complex classifier yet; it is whether the previous day had enough counter-regime parent rows for parent-soft to be relevant.
- The oracle daily policy equals static parent-soft on OOS, confirming that the practical opportunity in this frame is sparse and concentrated in a few days.

Decision:
- Do not promote the learned meta-router. It is not stable enough and does not improve OOS over baseline when selected by validation.
- Keep `prev_blocked_active_ratio >= 0.02` as a shadow-only router candidate for the next precision pass.
- Next required check before any promotion: ledger-level attribution for the blocked-ratio router, monthly reset check, and a no-OOS-tuning threshold sweep on more historical folds.

## Previous-Day Blocked-Ratio Router Precision Retest

Artifacts:
- Script: `/home/llewyn/crypto-scalping/scripts/precision_retest_alpha7_prev_block_ratio_router_20260528.py`
- Output: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_prev_block_ratio_router_precision_20260528/summary.json`
- Grid: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_prev_block_ratio_router_precision_20260528/grid.csv`
- Monthly grid: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_prev_block_ratio_router_precision_20260528/monthly_cost3.csv`
- Ledger diff: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_prev_block_ratio_router_precision_20260528/ledger_diff_summary.csv`

Rule:
- Once per day, use parent-soft decisions only if the previous day had `prev_blocked_active_ratio >= 0.02`.
- Otherwise use baseline decisions.
- This is causal at day open because it uses previous-day aggregates only.
- Route days:
  - Validation: 9 days in 2025Q4.
  - OOS: 8 days in 2026 eval.

Full Cost3:

| Variant | Split | PnL | MDD | WR | Trades | Deep | L/S | SL Ratio | Score |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 2025Q4 | 90.30 | -23.01 | 0.500 | 198 | 174 | 69 / 129 | 0.126 | 58.34 |
| parent soft static | 2025Q4 | 95.24 | -21.01 | 0.503 | 197 | 175 | 70 / 127 | 0.127 | 67.41 |
| prev blocked ratio t0.02 | 2025Q4 | 95.85 | -20.76 | 0.503 | 197 | 175 | 70 / 127 | 0.127 | 68.52 |
| baseline | 2026 OOS | 198.78 | -18.22 | 0.440 | 109 | 88 | 34 / 75 | 0.110 | 176.69 |
| parent soft static | 2026 OOS | 250.39 | -18.22 | 0.463 | 108 | 88 | 34 / 74 | 0.111 | 229.24 |
| prev blocked ratio t0.02 | 2026 OOS | 250.39 | -18.22 | 0.463 | 108 | 88 | 34 / 74 | 0.111 | 229.24 |

Monthly reset:
- 2025-10 improves versus baseline: `13.50 -> 16.81`, MDD `-23.01 -> -20.76`.
- 2025-11 unchanged.
- 2025-12 reverts to baseline and loses the small static parent-soft improvement.
- 2026-01 unchanged.
- 2026-02 worsens the same way as static parent-soft: `73.74 -> 32.75`, although MDD improves `-27.23 -> -24.02`.

Ledger attribution:

| Split | Common Trades | Baseline-Only | Router-Only | Baseline-Only Return | Router-Only Return | Gross Return Delta |
|---|---:|---:|---:|---:|---:|---:|
| 2025Q4 | 196 | 2 | 1 | 0.0027 | 0.0302 | +0.0275 |
| 2026 OOS | 99 | 10 | 9 | 0.0936 | 0.2559 | +0.1622 |

Interpretation:
- The blocked-ratio router is causal and sparse, and it reproduces the static parent-soft full OOS improvement while using parent-soft on only 8 OOS days.
- The full-period improvement still comes from changed trade sequencing, not common-trade improvements; common trade return delta is `0.0`.
- Monthly reset still shows the same weakness in February as static parent-soft. This means the rule can improve a continuous ledger but is not robust under period reset accounting.

Decision:
- Keep as shadow-only. Do not promote to live.
- It is a better research candidate than learned HGB/logistic for this exact parent-soft overlay, but it still fails the monthly-reset robustness requirement.
