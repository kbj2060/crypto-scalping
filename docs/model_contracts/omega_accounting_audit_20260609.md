# Omega Accounting Audit - 2026-06-09

## Full Inventory Scope

This audit is the canonical Omega accounting-state inventory as of 2026-06-09.

Artifacts generated:

- `tmp/causal_regen_20260516/omega_full_accounting_audit_20260609/omega_report_accounting_inventory.csv`
- `tmp/causal_regen_20260516/omega_full_accounting_audit_20260609/omega_script_accounting_inventory.csv`
- `tmp/causal_regen_20260516/omega_full_accounting_audit_20260609/summary.json`

Inventory coverage:

- JSON reports inspected: `284`
- Omega/RL-related scripts inspected: `94`
- Omega model-contract files inspected: `14`
- Report/ranking files discovered in Omega-named result trees: `356`

First-pass report classification:

- `explicit_true_leverage_exposure`: `38`
- `documented_metadata_effective_notional`: `72`
- `metadata_effective_notional_unstated`: `33`
- `deprecated_invalid_true_leverage`: `9`
- `mixed_or_suspect_needs_manual_review`: `4`
- `unknown_needs_manual_review`: `128`

The `unknown_needs_manual_review` bucket is not promotion-safe. Most entries are non-replay training, feature, chart, or ablation artifacts without an explicit replay accounting contract. If any of those artifacts becomes a candidate, it must be reclassified from its generating script and rerun if needed before comparison.

## Verdict

Not all Omega models are accounting-invalid. The invalid group is the research path that stored `leverage` as if it were active exposure control while replay accounting only applied `notional_exposure`.

Active and future Omega research must use one explicit accounting contract:

- `margin_fraction`: account margin committed to the trade
- `leverage`: exchange leverage multiplier
- `effective_exposure = margin_fraction * leverage`
- entry/exit fee base: entry and exit effective notional
- PnL/MDD base: effective exposure

If a model intentionally uses `notional_exposure` as already-effective account exposure, it must state that explicitly and must not claim leverage-amplified PnL.

## Status Taxonomy

- `explicit_true_leverage_exposure`: eligible for true-leverage ranking. PnL, fee, and MDD are based on effective exposure, or the report explicitly sets `use_leverage_exposure=true` / `accounting.mode=leverage_exposure`.
- `documented_metadata_effective_notional`: accounting is valid only as effective-notional replay. `leverage` is metadata or execution metadata. These reports must not be compared against true-leverage candidates as leverage-amplified results.
- `metadata_effective_notional_unstated`: replay appears to use `notional_exposure` as effective exposure, but the report does not state the contract. Keep as research-only until the contract is updated or the result is rerun.
- `mixed_or_suspect_needs_manual_review`: leverage appears in reward/features/risk scoring, but replay PnL/fee/MDD appears notional-only. Keep as research-only and do not stack/promote.
- `deprecated_invalid_true_leverage`: known mismatch where a candidate claimed or implied leverage semantics but replay did not account on effective exposure.

## Deprecated Due To Accounting Semantics

These artifacts are historical-reference only and must not be used for live promotion or active candidate stacking:

- `omega2_1_label_atr1_h24_hgb_12seed_ensemble_thr055`
- `omega2_1_hgb_calibration_exposure_20260609`
- `omega1_2_1_cash_fallback_extra_base_edge006_thr055_20260606`
- `omega1_2_1_cash_fallback_mlp_base_edge006_thr085_20260606`
- `omega1_2_1_cash_fallback_tb08_mlp_zigsame_c075_e065_20260607`
- Any candidate inheriting `scripts/train_eval_omega1_2_1_cash_fallback_sleeve_20260606.py` metrics without a leverage-exposure rerun.
- `omega2_1_dsac_overlay_20260609`
- `omega2_architect_priority_experiments_20260609`

Reason: fallback risk included both `notional` and `leverage`, but the legacy replay treated `notional_exposure` as the only PnL/fee/MDD exposure base. Under true leverage-exposure replay the headline Omega2.1 HGB baseline changed from:

- legacy metadata-leverage OOS: `+102.611483%`, MDD `-8.108171%`, WR `60.975610%`, trades `41`
- corrected leverage-exposure OOS: `+33.877901%`, MDD `-23.976364%`, WR `43.410853%`, trades `129`

## Research-Only / Revalidation Required

These families are not automatically invalid, but they are not eligible for promotion until their accounting contract is made explicit:

- `train_eval_omega1_2_mamba_sac_lifecycle_controller_20260604.py`: replay uses `notional_exposure` as effective exposure; leverage treatment is not explicit in all reports.
- `train_eval_omega1_2_mamba_sac_3head_feature_coordinator_20260604.py`: leverage appears in reward/features while replay appears notional-only.
- `train_eval_omega1_2_tabm_diffusion_risk_20260603.py`: diffusion reward/scorer uses leverage/exposure, but `_simulate_trade` accounts PnL/fee on `notional_exposure`.
- `train_eval_omega1_2_risk_scale_selector_20260605.py`: inherits the diffusion risk replay semantics and must be rerun if used as a candidate.
- `train_eval_omega1_2_exposure_governor_20260606.py`: uses post-lifecycle adapter helpers and must state whether `notional_exposure` is effective exposure or true leveraged exposure before ranking.

## Highest PnL Among Accounting-Normal Results

Accounting-normal means the report explicitly uses `use_leverage_exposure=true` or `accounting.mode=leverage_exposure`.

### Highest OOS PnL, Rejected For Promotion

- Candidate: `omega1_2_post_lifecycle_bucket_adapter_20260605_hgb_fixed_ultra_wide_lev5_eff_cap120_nogate_traink3_replayk2_s260710`
- OOS: `+81.350538%`
- OOS MDD: `-8.540793%`
- OOS WR: `80.000000%`
- OOS trades: `30`
- Validation: `-31.607681%`
- Validation MDD: `-34.669962%`
- Status: `research_only_oos_overfit_risk`

This is the highest explicit true-leverage OOS result in the full inventory, but it is not promotion-safe because validation collapses.

The corrected Omega2.1 true-leverage calibration sweep also produced a high OOS-only result:

- Candidate: `omega2_1_hgb_calibration_exposure_levexp_20260609::static_t0.45_m0.00_a0.84_s2.5_cap0.9`
- OOS: `+97.821995%`
- OOS MDD: `-29.584606%`
- OOS WR: `49.032258%`
- OOS trades: `155`
- Validation: `-10.750938%`
- Validation MDD: `-45.702435%`
- Status: `research_only_oos_overfit_risk`

This sweep result is kept as a corrected accounting diagnostic, not as a model promotion candidate.

### Best Stable High-PnL Candidate

- Candidate: `omega1_2_post_lifecycle_bucket_adapter_20260605_hgb_base_lev5_eff_cap150_comp_tpup_voltarget_trainall_c96_replayk2_s260726`
- OOS: `+79.762537%`
- OOS MDD: `-10.787332%`
- OOS WR: `68.571429%`
- OOS trades: `35`
- Validation: `+11.692292%`
- Validation MDD: `-15.899477%`
- Status: `best_current_accounting_normal_stable_research_candidate`

This is the best current candidate when requiring positive validation PnL and validation MDD above `-20%`.

Stable true-leverage promotion-filter top results:

| Rank | Candidate | OOS PnL | OOS MDD | WR | Trades | Val PnL | Val MDD |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `s260726` | `+79.762537%` | `-10.787332%` | `68.57%` | `35` | `+11.692292%` | `-15.899477%` |
| 2 | `s260727` | `+77.907955%` | `-11.360088%` | `68.57%` | `35` | `+12.467642%` | `-16.921473%` |
| 3 | `s260724` | `+70.702466%` | `-11.360088%` | `68.57%` | `35` | `+1.749506%` | `-16.425596%` |
| 4 | `s260716` | `+68.623600%` | `-11.360088%` | `68.57%` | `35` | `+2.081900%` | `-16.814987%` |
| 5 | `s260723` | `+63.459587%` | `-9.429411%` | `68.57%` | `35` | `+0.309119%` | `-16.955534%` |

## Evidence Artifacts

- Full inventory summary:
  `tmp/causal_regen_20260516/omega_full_accounting_audit_20260609/summary.json`
- Report-level inventory:
  `tmp/causal_regen_20260516/omega_full_accounting_audit_20260609/omega_report_accounting_inventory.csv`
- Script-level inventory:
  `tmp/causal_regen_20260516/omega_full_accounting_audit_20260609/omega_script_accounting_inventory.csv`
- Corrected Omega2.1 leverage-exposure report:
  `tmp/causal_regen_20260516/omega2_1_hgb_calibration_exposure_levexp_20260609/report.json`
- Corrected accounting audit:
  `tmp/causal_regen_20260516/omega2_1_hgb_scale25_levexp_accounting_audit_20260609/report.json`
- Stable high-PnL true leverage-exposure candidate:
  `tmp/causal_regen_20260516/omega1_2_post_lifecycle_bucket_adapter_20260605_hgb_base_lev5_eff_cap150_comp_tpup_voltarget_trainall_c96_replayk2_s260726/report.json`
