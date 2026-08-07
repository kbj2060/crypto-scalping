# Omega1.2.3 Full Model Evaluation - 2026-06-16

## Executive Verdict

Current model:
`omega1_2_3_ev_hgb_cash_sleeve_20260615`

Status:
`walkforward_pass_live_wired`

Overall judgment:
the model is acceptable as a live-wired shadow/dry-run candidate, but it is not
yet a fully production-cleared strategy. The offline logic, feature contract,
and monthly walk-forward checks pass under the current Red Team rule set. The
main remaining work is runtime-native parity, richer live telemetry, and
stability improvement for the cash fallback sleeve.

The model improves the Omega1.2.1 clean-repair baseline on 2026 OOS by
`+6.19p` total PnL with unchanged OOS MDD, but the validation lift is small
(`+0.29p`) and one monthly walk-forward fold is negative. Treat this as a
conservative incremental overlay, not as a broad architecture breakthrough.

## Model Structure

- Base primary:
  `omega1_2_1_tp_runner_clean_repair_20260613`
- Live candidate:
  `omega1_2_3_ev_hgb_cash_sleeve_20260615`
- Sleeve family:
  parent-CASH-only long/short expected-value regressors
- Models:
  `HistGradientBoostingRegressor` for `long_ev` and `short_ev`
- Feature count:
  `61`
- Training rows:
  `19,984` parent-CASH label rows
- Gate:
  enter only if `max(long_ev, short_ev) > 0.002`
- Selected risk:
  `base_tp026_sl014_n0405_h192`
- Risk details:
  TP `0.026`, SL `0.014`, notional/exposure `0.405`, leverage `2.0`,
  max hold `192` bars

Live wiring:

- Adapter:
  `trading_bot_modules/omega1_2_3_cash_sleeve.py`
- Entrypoint:
  `trading_bot.py`
- Bundle:
  `data/ensemble/supervised/omega1_2_3_ev_hgb_cash_sleeve_20260615/ev_hgb_cash_sleeve.joblib`
- Activation branch:
  Omega1.2.1 primary must return CASH and no position can be open
- Fallback exits:
  `omega1_2_3_fallback_take_profit`,
  `omega1_2_3_fallback_stop_loss`,
  `omega1_2_3_fallback_max_hold`,
  `omega1_2_3_fallback_primary_takeover`

## Offline Performance

### Baseline Replay

Omega1.2.1 clean-repair baseline:

| Split | PnL | MDD | WR | Trades | Primary Entries |
|---|---:|---:|---:|---:|---:|
| Validation | `+160.22%` | `-27.64%` | `59.46%` | `37` | `37` |
| OOS | `+85.70%` | `-15.64%` | `66.67%` | `18` | `18` |

### Omega1.2.3 Selected Candidate

| Split | PnL | Delta vs Base | MDD | WR | Trades | Fallback Entries |
|---|---:|---:|---:|---:|---:|---:|
| Validation | `+160.50%` | `+0.29p` | `-28.45%` | `59.57%` | `47` | `10` |
| OOS | `+91.89%` | `+6.19p` | `-15.64%` | `61.76%` | `34` | `16` |

Interpretation:

- OOS lift is meaningful relative to the small trade count: `+6.19p` on only
  `16` fallback entries.
- Validation lift is weak. The sleeve mostly adds trade count without large
  validation alpha.
- OOS WR drops from `66.67%` baseline to `61.76%` combined because the added
  sleeve trades have lower hit rate than the primary, but the EV profile still
  improves total PnL.
- Validation MDD worsens by about `0.81p`; OOS MDD is unchanged.

## Cash Sleeve Standalone Quality

| Split | Fallback PnL | MDD | WR | PF | Trades | Stop Rate |
|---|---:|---:|---:|---:|---:|---:|
| Validation | `+0.11%` | `-1.32%` | `60.00%` | `1.08` | `10` | `0.00%` |
| OOS | `+3.33%` | `-4.49%` | `56.25%` | `1.33` | `16` | `43.75%` |

Exit mix:

- Validation fallback exits:
  `10/10` were `fallback_primary_takeover`.
- OOS fallback exits:
  `5` take profit, `7` stop loss, `1` primary takeover, `2` max hold,
  `1` forced end.

Interpretation:

- The sleeve is not just a tiny primary-takeover scalper in OOS; it sometimes
  runs independently to TP/SL/max-hold.
- The OOS stop rate is high enough that a stop-risk veto or calibrated EV
  uncertainty filter is a natural next improvement.
- The validation fallback result is close to flat, so the OOS gain should not
  be overinterpreted as universally stable alpha.

## Walk-Forward Evaluation

Monthly expanding-window check:

- Fold count: `4`
- Positive delta folds: `3/4`
- Total combo delta: `+6.10p`
- Total fallback-only PnL points: `+4.06p`
- Total fallback trades: `28`
- Mean fallback WR: `45.83%`
- Mean fallback stop rate: `17.71%`

Fold summary:

| Fold | Combo Delta | Fallback PnL | Fallback WR | Stop Rate | Comment |
|---|---:|---:|---:|---:|---|
| 2025-10 -> 2025-11 | `+0.18p` | `+0.11p` | `66.67%` | `0.00%` | small positive |
| 2025-10/11 -> 2025-12 | `-0.57p` | `-0.67p` | `33.33%` | `0.00%` | negative fold |
| 2025-Q4 -> 2026-01 | `+0.31p` | `+0.23p` | `33.33%` | `33.33%` | small positive |
| 2025-Q4/2026-01 -> 2026-02 | `+6.19p` | `+4.39p` | `50.00%` | `37.50%` | dominant positive fold |

Interpretation:

- The model passes the current walk-forward gate, but most aggregate gain comes
  from the final fold.
- `ev_min=0.004` was rejected because it was stronger in point OOS but less
  stable in monthly walk-forward. Keeping robust `ev_min=0.002` is the right
  conservative choice.
- The walk-forward report metadata still has a stale method sentence mentioning
  `selected ev_min=0.004`, while diagnostics and selected aggregate use
  `0.002`. This is a documentation/artifact hygiene issue, not a model logic
  blocker.

## Red Team / Contract Evaluation

Pass areas:

- Parent-CASH-only eligibility is explicit.
- Feature count and bundle contract load correctly.
- Forbidden feature families remain excluded:
  `tp_sl_action_score`, `teacher_*`, `regime4_pred_*`,
  `clean_regime4_*`, `clean_regime_2024_unsup_v4_*`.
- `py_compile` passes for:
  `trading_bot.py`, `trading_bot_modules/omega1_2_3_cash_sleeve.py`,
  `trading_bot_modules/omega1_2_1_live.py`.
- Bundle load passes in the live Python environment.
- JSON manifests parse correctly.
- Live process is running after restart.

Current live status observed on 2026-06-16 KST:

- PID: `192810`
- Command:
  `/home/llewyn/miniconda3/envs/quant_ai/bin/python trading_bot.py`
- Position:
  `NONE`
- Dashboard state:
  updating
- Binance execution:
  log shows `binance_execution=OFF dry_run=True testnet=True`
- Initial post-restart data status:
  microstructure briefly stale during warmup, then recovered to LIVE

Non-blocking but important issues:

- The cash sleeve currently has in-memory primary history. After a process
  restart, `primary_active_roll_*`, `primary_cash_streak`,
  `last_primary_active_len`, and `last_primary_side` cold-start. The model can
  still run, but this is not identical to the continuous historical feature
  construction.
- Live telemetry does not expose every cash sleeve EV decision when the sleeve
  also returns CASH. That makes live calibration and false-negative analysis
  harder.
- The current verification is compile + bundle load + synthetic inference. It
  is not yet a full runtime-native replay parity proof.
- The live log line reports the Omega1.2.1 bundle path as the underlying TabM
  artifact path, while the active model ID contract is the clean-repair TP
  runner model. This is expected from the adapter stack, but the startup log
  should include the Omega1.2.3 sleeve bundle explicitly to avoid ambiguity.

## Risk Assessment

| Risk | Severity | Assessment |
|---|---|---|
| Full runtime-native parity not yet run | High | Required before calling this production-cleared |
| Restart cold-start for sleeve history | Medium | Can shift early post-restart decisions |
| OOS lift concentrated in final WF fold | Medium | Good but not broad enough |
| High OOS fallback stop rate | Medium | Needs stop-risk gating or EV calibration |
| Weak validation fallback-only PnL | Medium | Suggests limited robust standalone alpha |
| Live telemetry for sleeve EV is sparse | Medium | Slows diagnosis |
| Current execution is dry-run/testnet | Operational | Real money path not enabled in observed logs |

## Improvement Roadmap

### Priority 1 - Runtime Parity And Observability

1. Build a runtime-native replay harness for Omega1.2.3.
   It should call `FinalGovernorRuntime` over historical bars and compare
   decisions against the vectorized sleeve evaluation ledger. Required outputs:
   action parity, side parity, EV value drift, entry/exit reason parity, PnL
   drift, and row-level mismatch CSV.

2. Persist or reconstruct sleeve primary history.
   Store enough fields in runtime state to preserve:
   `primary_active_history`, `primary_cash_streak`,
   `last_primary_active_len`, and `last_primary_side`. On restart, either load
   persisted state or deterministically reconstruct recent primary decisions
   from the latest frame.

3. Add cash sleeve telemetry.
   Write a JSONL row whenever Omega1.2.1 returns CASH:
   `long_ev`, `short_ev`, selected action, EV margin, gate threshold, feature
   hash, primary router expert, and resulting live action. This should record
   both sleeve trades and sleeve-CASH decisions.

4. Add startup logging for Omega1.2.3.
   The governor startup line should show:
   `omega1_2_3_cash_sleeve=<bundle path or OFF>`,
   `ev_min`, `feature_count`, and `risk`.

### Priority 2 - Model Quality Improvements

1. Stop-risk veto.
   Train a lightweight stop-risk model or reuse the tested stop-veto family,
   but select it only by validation and monthly walk-forward. Target:
   reduce OOS fallback stop rate below `35%` without destroying trade count.

2. EV calibration.
   Calibrate `long_ev` and `short_ev` with chronological residual bands or
   conformal lower bounds. Trade on lower-confidence EV, not raw mean EV.
   Target:
   improve fold consistency and reduce the negative 2025-12 fold.

3. Regime-aware EV threshold.
   Sweep `ev_min` by simple causal context:
   bull/bear/chop, volatility bucket, cash streak bucket, and primary router
   expert. Keep the active path fail-fast and avoid legacy aliases.

4. Time-decay / max-hold refinement.
   OOS has `2` max-hold and `1` forced-end fallback exits. Add a time-decay
   exit gate for sleeve positions that fail to gain EV confirmation after a
   fixed bar count.

5. Side-specific risk.
   Current sleeve uses symmetric risk. Test long/short-specific TP/SL and
   max-hold under the same validation-only and walk-forward selection policy.

### Priority 3 - Promotion Gates

Before enabling real execution, require:

- runtime-native replay parity report with no material contract mismatch
- at least one live shadow window with cash sleeve telemetry
- no feature contract violations in live snapshots
- no unexpected sleeve decisions during post-restart cold-start window
- explicit Red Team sign-off for real execution costs, exchange account sync,
  and position reconciliation

## Final Recommendation

Keep Omega1.2.3 live-wired in shadow/dry-run mode and collect telemetry. Do not
upgrade it to real-money production until runtime-native parity and restart
state persistence are done.

The most promising next version is not a larger model. It should be:

`omega1_2_4_ev_calibrated_cash_sleeve`

Expected changes:

- same Omega1.2.1 primary
- same CASH-only ownership boundary
- same 61-feature base contract plus explicit telemetry hash
- calibrated EV lower-bound gate
- optional stop-risk veto selected by walk-forward
- persisted sleeve history
- full runtime-native parity artifact

This keeps the working structure intact while addressing the actual observed
weaknesses: stop-heavy OOS fallback trades, fold concentration, sparse live
observability, and restart-state mismatch.

## Implementation Update - 2026-06-16

Completed after the initial report:

- Added direct cash-sleeve parity harness:
  `scripts/check_omega1_2_3_cash_sleeve_parity_20260616.py`
- Added live cash-sleeve history snapshot/restore:
  `omega1_2_3_cash_sleeve_state` in the final governor runtime state JSON
- Added cash-sleeve decision telemetry:
  `data/live/omega1_2_3_cash_sleeve_decisions.jsonl`
- Added startup logging for Omega1.2.3 bundle path, telemetry path, `ev_min`,
  feature count, and selected risk

Verification completed:

- `py_compile` passed for:
  `trading_bot.py`,
  `trading_bot_modules/omega1_2_3_cash_sleeve.py`,
  `scripts/check_omega1_2_3_cash_sleeve_parity_20260616.py`
- adapter snapshot/restore round-trip passed
- direct parity sampled validation: `366/366` cash rows compared,
  `0` mismatches, max feature diff `1.67e-16`, EV diff `0`
- direct parity sampled OOS: `364/364` cash rows compared,
  `0` mismatches, max feature diff `1.67e-16`, EV diff `0`

Remaining limitation:

- full-row direct parity was started for validation and OOS, but the current
  direct replay implementation recomputes rolling features over the growing
  frame prefix per row and became too slow. It was stopped without mismatches
  observed. The next parity improvement should vectorize the rolling feature
  replay while keeping the same strict feature and EV comparison.

## Source Artifacts

- Manifest:
  `data/ensemble/supervised/omega1_2_3_ev_hgb_cash_sleeve_20260615/candidate_manifest.json`
- Bundle:
  `data/ensemble/supervised/omega1_2_3_ev_hgb_cash_sleeve_20260615/ev_hgb_cash_sleeve.joblib`
- Robust summary:
  `tmp/causal_regen_20260516/omega1_2_3_cash_sleeve_upgrade_20260615/robust_ev002_selected_summary.json`
- Walk-forward report:
  `tmp/causal_regen_20260516/omega1_2_3_ev_hgb_cash_sleeve_walkforward_20260615/report.json`
- Live adapter:
  `trading_bot_modules/omega1_2_3_cash_sleeve.py`
- Live runtime:
  `trading_bot.py`
- Cash sleeve parity harness:
  `scripts/check_omega1_2_3_cash_sleeve_parity_20260616.py`

## Roadmap Execution Update - Omega1.2.4 Probe

New candidate:

`omega1_2_4_ev_calibrated_cash_sleeve_20260616`

Change:

- same Omega1.2.1 primary
- same parent-CASH-only ownership boundary
- same `61` feature contract
- same HGB long/short EV models
- EV lower-bound calibration added:
  subtract validation median absolute residual from raw EV
- calibration offsets:
  long `0.0010768821918855976`,
  short `0.0010608525999764662`
- gate remains `max(calibrated_long_ev, calibrated_short_ev) > 0.002`

Probe results:

| Candidate | Validation PnL | Validation Fallback PnL | OOS PnL | OOS Delta vs Base | OOS Fallback PnL | OOS Fallback Trades | OOS Stop Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Omega1.2.3 raw EV002 | `+176.81%` | `+6.38%` | `+91.89%` | `+6.19p` | `+3.33%` | `16` | `43.75%` |
| Omega1.2.4 cal q50 EV002 | `+180.80%` | `+7.91%` | `+93.49%` | `+7.79p` | `+4.20%` | `14` | `35.71%` |

Walk-forward comparison:

| Candidate | Positive Folds | Total Combo Delta | Total Fallback PnL | Fallback Trades | Mean Fallback WR | Mean Stop Rate |
|---|---:|---:|---:|---:|---:|---:|
| Omega1.2.3 raw EV002 | `3/4` | `+6.10p` | `+4.06p` | `28` | `45.83%` | `17.71%` |
| Omega1.2.4 cal q50 EV002 | `4/4` | `+4.91p` | `+3.49p` | `18` | `63.64%` | `19.70%` |

Interpretation:

- Omega1.2.4 trades less than Omega1.2.3 in walk-forward.
- Aggregate WF PnL is lower than Omega1.2.3, but fold consistency improves
  from `3/4` to `4/4`.
- OOS probe improves both total PnL and stop rate versus Omega1.2.3.
- This is a stability-oriented upgrade, not a max-PnL upgrade.

Verification completed:

- `py_compile` passed for the updated runtime and evaluation scripts.
- New bundle load passed in the live Python environment.
- Full validation parity:
  `20,102/20,102` cash rows compared, `0` mismatches.
- Full OOS parity:
  `13,345/13,345` cash rows compared, `0` mismatches.

Artifacts:

- Bundle:
  `data/ensemble/supervised/omega1_2_4_ev_calibrated_cash_sleeve_20260616/ev_calibrated_cash_sleeve.joblib`
- Manifest:
  `data/ensemble/supervised/omega1_2_4_ev_calibrated_cash_sleeve_20260616/candidate_manifest.json`
- Calibration probe:
  `tmp/causal_regen_20260516/omega1_2_4_ev_calibrated_cash_sleeve_probe_20260616/report.json`
- Walk-forward:
  `tmp/causal_regen_20260516/omega1_2_4_calibrated_cash_sleeve_walkforward_20260616/report.json`

Updated recommendation:

Use Omega1.2.4 as the live-wired shadow/dry-run candidate. Keep real execution
disabled until at least one live shadow window confirms telemetry, restart
state recovery, and no unexpected calibrated sleeve behavior.

## Parent Numeric Utility vs RL Q-Value Experiment - 2026-06-16

Experiment id:

`omega1_2_5_parent_numeric_vs_rlq_20260616`

Purpose:

Test whether the Omega parent model should be retrained with either existing
numeric utility labels or RL policy/value-network Q-value labels, then compare
parent-only and parent-plus-cash-sleeve structures against the current Omega
baseline.

Assumptions and label construction:

- Numeric utility labels are deterministic path-simulation labels, not full RL.
- RL Q-value labels come from the DSAC critic checkpoint:
  `data/ensemble/ckpt/best_dsac_agents_clean_retrain_v1.pth`.
- RL labels use fixed candidate actions:
  short `-0.45`, cash `0.0`, long `+0.45`.
- Parent features use the Omega parent state feature set and exclude decision
  columns plus forbidden feature families.
- Tested structures:
  parent-only replacement and parent plus retrained cash sleeve.

Label diagnostics:

| Label Set | Rows | Long Mean | Short Mean | Long Positive | Short Positive | Extra |
|---|---:|---:|---:|---:|---:|---|
| Numeric utility | `26,295` | `-0.005900` | `-0.004143` | `7,858` | `8,444` | long stop `27.45%`, short stop `20.66%` |
| RL Q advantage | `26,490` | `-0.002912` | `-0.008205` | `12,087` | `8,801` | cash baseline Q mean `-0.809276` |

Baseline inside this experiment:

| Split | PnL | MDD | Trades | Win Rate |
|---|---:|---:|---:|---:|
| Validation | `+160.22%` | `-27.64%` | `37` | `59.46%` |
| OOS | `+85.70%` | `-15.64%` | `18` | `66.67%` |

Validation-selected candidate:

| Candidate | Family | Val PnL | Val Delta | OOS PnL | OOS Delta | OOS Fallback PnL | OOS Fallback Trades | OOS Stop Rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `utility_thr0.002_parent_plus_sleeve_ev0.004` | parent + retrained sleeve | `+54.47%` | `-105.75p` | `+8.00%` | `-77.70p` | `-10.03%` | `19` | `26.32%` |

Best OOS diagnostic candidate:

| Candidate | Family | Val PnL | Val Delta | OOS PnL | OOS Delta | OOS MDD | OOS Win Rate | OOS Trades |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `utility_thr0.001_parent_only` | parent-only | `+42.74%` | `-117.48p` | `+26.99%` | `-58.71p` | `-5.90%` | `63.64%` | `22` |

Best RL Q-value diagnostic candidate:

| Candidate | Family | Val PnL | Val Delta | OOS PnL | OOS Delta | OOS MDD | OOS Win Rate | OOS Trades |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `rlq_thr0.002777_parent_only` | parent-only | `-30.11%` | `-190.32p` | `+8.61%` | `-77.09p` | `-12.70%` | `46.67%` | `30` |

Redteam result:

- Status: `redteam_pass_probe`.
- No feature-contract, data-contract, or artifact-contract blockers were found.
- This is not a promotable model because performance is materially worse than
  both the local baseline and the current Omega1.2.4 live-wired candidate.

Interpretation:

- Replacing the parent with numeric utility labels weakened the parent decision
  surface.
- The RL Q labels were internally usable, but they are not aligned enough with
  the Omega parent action/risk space to improve execution.
- Parent-plus-retrained-sleeve did not rescue the parent degradation. In RLQ
  variants, the parent often left no useful CASH ownership region for the cash
  sleeve, so the structure collapsed back toward parent-only behavior.

Recommendation:

- Do not promote Omega1.2.5.
- Do not replace the Omega1.2.1 parent with these numeric utility or RLQ labels.
- Keep Omega1.2.4 as the live-wired dry-run candidate.
- If RL labels are retried, generate them from a parent-aligned critic trained
  on the exact Omega parent action set, position state, risk limits, and
  execution reward surface, rather than from a generic flat-state DSAC critic.

Artifacts:

- Script:
  `scripts/train_eval_omega1_2_5_parent_numeric_vs_rlq_20260616.py`
- Report:
  `tmp/causal_regen_20260516/omega1_2_5_parent_numeric_vs_rlq_20260616/report.json`
- Ranking:
  `tmp/causal_regen_20260516/omega1_2_5_parent_numeric_vs_rlq_20260616/parent_numeric_vs_rlq_ranking.csv`

## Parent Numeric Utility vs RL Q-Value Full OOF Evaluation - 2026-06-16

Correction:

The first `omega1_2_5_parent_numeric_vs_rlq_20260616` run was a fast probe.
It trained HGB regressors on all validation labels and used same-sample
validation predictions for candidate screening. It did not retrain the RL
policy/value network and did not produce a promotable parent bundle.

Full evaluation id:

`omega1_2_5_parent_numeric_vs_rlq_full_20260616`

Full-eval method:

- no row subsampling
- validation rows: `26,490`
- OOS rows: `16,820`
- parent feature count: `34`
- utility and RLQ labels generated across the full validation segment
- expanding OOF validation:
  `35->50%`, `50->65%`, `65->80%`, `80->100%`
- final refit on all validation labels
- OOS remains diagnostic only and is not used for selection
- RLQ labels still use the existing DSAC critic checkpoint; this is not an RL
  policy/value network retrain

Baseline inside the full evaluation:

| Split | PnL | MDD | Trades | Win Rate |
|---|---:|---:|---:|---:|
| Validation | `+160.22%` | `-27.64%` | `37` | `59.46%` |
| OOS | `+85.70%` | `-15.64%` | `18` | `66.67%` |

Full OOF ranking:

| Candidate | Val PnL | Val Delta | OOS PnL | OOS Delta | Val Trades | OOS Trades |
|---|---:|---:|---:|---:|---:|---:|
| `utility_thr0.002_full_oof` | `+6.81%` | `-153.40p` | `+21.87%` | `-63.83p` | `27` | `25` |
| `utility_thr0.000_full_oof` | `+2.62%` | `-157.60p` | `+17.16%` | `-68.55p` | `27` | `30` |
| `rlq_thr0.003686_full_oof` | `-1.40%` | `-161.61p` | `-3.67%` | `-89.38p` | `28` | `36` |
| `utility_thr0.001_full_oof` | `-5.80%` | `-166.02p` | `+24.91%` | `-60.79p` | `33` | `29` |
| `rlq_thr0.001659_full_oof` | `-17.42%` | `-177.64p` | `-1.53%` | `-87.23p` | `37` | `34` |
| `rlq_thr0.000000_full_oof` | `-20.63%` | `-180.85p` | `+1.82%` | `-83.89p` | `40` | `34` |

Full-eval redteam result:

- Status: `redteam_pass_full_eval`.
- No data/feature/artifact contract blockers were found.
- Performance is not promotable.
- The full OOF result is worse than the fast probe and confirms that the
  parent relabel idea should not replace Omega1.2.1.

Updated recommendation:

- Keep Omega1.2.4 as the live-wired dry-run candidate.
- Do not promote Omega1.2.5.
- Do not call the first Omega1.2.5 run a full training run; it was a probe.
- Do not use the current DSAC critic Q-value labels for parent replacement
  without retraining a parent-aligned critic.

Full-eval artifacts:

- Script:
  `scripts/train_eval_omega1_2_5_parent_numeric_vs_rlq_full_20260616.py`
- Report:
  `tmp/causal_regen_20260516/omega1_2_5_parent_numeric_vs_rlq_full_20260616/report.json`
- Ranking:
  `tmp/causal_regen_20260516/omega1_2_5_parent_numeric_vs_rlq_full_20260616/parent_numeric_vs_rlq_full_oof_ranking.csv`

## RL Q-Value Cash Sleeve Only Full Evaluation - 2026-06-16

Experiment id:

`omega1_2_5_rlq_cash_sleeve_full_20260616`

Scope:

- Omega parent is unchanged.
- Only primary-CASH rows are used for fallback cash sleeve training.
- Labels are DSAC critic Q-value advantages:
  `long_adv = Q(long) - Q(cash)`,
  `short_adv = Q(short) - Q(cash)`.
- This uses the existing DSAC critic checkpoint to generate labels. It does not
  retrain the RL policy/value network itself.
- Validation uses expanding OOF predictions.
- OOS uses final refit on full validation cash-label rows.
- OOS remains diagnostic only.

Data and feature contract:

- validation rows: `26,490`
- OOS rows: `16,820`
- sleeve feature count: `61`
- cash train rows:
  micro `20,052`,
  base `19,984`,
  mid `19,984`
- redteam status: `redteam_pass_full_eval`

Baseline inside this experiment:

| Split | PnL | MDD | Trades | Win Rate |
|---|---:|---:|---:|---:|
| Validation | `+160.22%` | `-27.64%` | `37` | `59.46%` |
| OOS | `+85.70%` | `-15.64%` | `18` | `66.67%` |

Validation-selected RLQ cash sleeve:

| Candidate | Val PnL | Val Delta | Val Fallback PnL | Val Fallback Trades | OOS PnL | OOS Delta | OOS Fallback PnL | OOS Fallback Trades | OOS Stop Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `base_tp026_sl014_n0405_h192_rlq_extra_qmin0.00000000` | `+157.90%` | `-2.32p` | `-0.89%` | `23` | `+60.37%` | `-25.34p` | `-13.64%` | `38` | `36.84%` |

Best OOS diagnostic RLQ cash sleeve:

| Candidate | Val PnL | Val Delta | Val Fallback PnL | Val Fallback Trades | OOS PnL | OOS Delta | OOS Fallback PnL | OOS Fallback Trades | OOS Stop Rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `mid_tp030_sl018_n055_h192_rlq_agree_qmin0.02642761` | `+158.69%` | `-1.53p` | `-0.59%` | `4` | `+88.98%` | `+3.27p` | `+1.76%` | `22` | `40.91%` |

Interpretation:

- The validation-selected RLQ sleeve fails OOS badly.
- The best OOS diagnostic candidate does improve OOS versus the local baseline,
  but its validation fallback PnL is negative and it is still weaker than
  Omega1.2.4's OOS result.
- Stop-loss rate remains too high in OOS for the best diagnostic candidate.
- RLQ labels appear useful as a ranking signal for a narrow diagnostic case, but
  not reliable enough as the sole cash sleeve training target.

Recommendation:

- Do not promote the RLQ-only cash sleeve.
- Keep Omega1.2.4 as the live-wired dry-run candidate.
- A better next retry is a hybrid sleeve label:
  deterministic EV lower-bound plus RLQ agreement or RLQ veto, instead of
  replacing the sleeve target with RLQ alone.

Artifacts:

- Script:
  `scripts/train_eval_omega1_2_5_rlq_cash_sleeve_full_20260616.py`
- Report:
  `tmp/causal_regen_20260516/omega1_2_5_rlq_cash_sleeve_full_20260616/report.json`
- Ranking:
  `tmp/causal_regen_20260516/omega1_2_5_rlq_cash_sleeve_full_20260616/rlq_cash_sleeve_full_ranking.csv`
