# Omega1.2.8 Full-Retrain Numeric Cash Sleeve Contract - 2026-06-16

## Status

- Alias: `omega1.2.8_full_retrain_numeric_cash_sleeve`
- Model ID: `omega1_2_8_full_retrain_numeric_cash_sleeve_20260616`
- Status: `current_omega_research_baseline_not_live_wired`
- Parent artifact: `tmp/causal_regen_20260516/omega1_2_true_3head_tabm_20260603_full_retrain_cash_alpha43_20260608`
- Entrypoint: `scripts/train_eval_omega1_2_8_full_retrain_numeric_cash_sleeve_20260616.py`
- Report: `tmp/causal_regen_20260516/omega1_2_8_full_retrain_numeric_cash_sleeve_20260616/report.json`
- Manifest: `data/ensemble/supervised/omega1_2_8_full_retrain_numeric_cash_sleeve_20260616/candidate_manifest.json`

## Architecture

- Preserve the full-retrained 3-head TabM parent artifact as the primary action owner.
- When the parent is active long/short, the parent owns the entry and position lifecycle.
- When the parent is CASH, a deterministic HGB EV lower-bound sleeve can propose long/short fallback entries.
- Numeric HGB utility regressors provide agreement/veto over the EV sleeve. This is a numeric-label hybrid, not RLQ and not full RL.
- Selected config is validation-only:
  `full_retrain_ev_cal0.50_ev0.001_numcfg1_u0.002_m0.000`.
- Selected thresholds:
  `cal_q=0.50`, `ev_min=0.001`, `utility_cfg_id=1`, `utility_min=0.002`, `margin_min=0.000`.
- Utility config 1:
  `stop_penalty=0.003`, `mae_penalty=0.20`, `time_penalty=0.0`.
- Risk template:
  `base_tp026_sl014_n0405_h192`, TP `0.026`, SL `0.014`, notional `0.405`, leverage `2.0`, max hold `192`.
- Feature count: `42`.

## Baseline Metrics

Full-retrained parent primary-only baseline:

- Validation: PnL `+100.542729%`, MDD `-10.677653%`, WR `63.636364%`, trades `33`.
- OOS: PnL `+72.760041%`, MDD `-8.108171%`, WR `72.222222%`, trades `18`.

Selected HGB numeric hybrid:

- Validation: PnL `+116.524792%`, MDD `-10.789000%`, WR `69.047619%`, trades `42`, fallback entries `9`.
- OOS: PnL `+82.251474%`, MDD `-8.108171%`, WR `62.500000%`, trades `32`, fallback entries `14`.
- Delta vs full-retrained parent: validation `+15.982062p`, OOS `+9.491433p`.

Diagnostics only:

- Best OOS diagnostic hybrid was `full_retrain_ev_cal0.50_ev0.004_numcfg1_u0.000_m0.000`, OOS PnL `+87.381332%`.
- Best EV-only control was `full_retrain_ev_cal0.65_ev0.003`, OOS PnL `+85.375222%`.
- These diagnostic rows are not selection defaults because the selection policy is validation-only.

## Promotion Boundary

- This model is the current Omega research baseline for new candidate comparisons.
- Live runtime remains `omega1_2_3_ev_hgb_cash_sleeve_20260615` until a separate runtime-native implementation, parity check, walk-forward/stress pass, and explicit live promotion.
- Future Omega research candidates should compare against the selected validation row above, not the best OOS diagnostic row.
- Do not reintroduce RLQ as a standalone target/action owner for this cash sleeve.
- Do not replace the HGB numeric sleeve with TabM unless a new validation-only test beats this baseline and keeps the same fail-fast feature/artifact contract.
- Do not use `tp_sl_action_score`, legacy regime prefixes, or compatibility fallbacks in active candidate paths.
