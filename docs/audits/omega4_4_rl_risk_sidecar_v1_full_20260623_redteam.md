# Omega4.4 RL Risk Sidecar v1 Full-Test Audit - 2026-06-23

## Verdict

`omega4_4_rl_risk_sidecar_v1_full_20260623`: **`REDTEAM_PASS_CLEAN_RESEARCH_FULL_TEST_NOT_BASELINE_UPGRADE`**

The run completed all four recommended policy classes on the full Omega4.4 train/validation/OOS ledgers. It is clean as a research test, but it should not replace the Omega4.4 HGB risk sidecar.

## Checks

- `report_exists`: `True`
- `artifact_exists`: `True`
- `ranking_exists`: `True`
- `all_four_policies_tested`: `True`
- `selection_scope_validation_only`: `True`
- `oos_readout_only`: `True`
- `notional_contract_declared`: `True`
- `sltp_not_notional_scaled`: `True`
- `selected_validation_improves_hgb`: `True`
- `selected_oos_improves_hgb`: `False`
- `selected_validation_mdd_within_7`: `True`

## Result

- Selected policy: `iql_awac`
- Validation: `+23.62%`, MDD `-5.59%`
- OOS readout: `+15.40%`, MDD `-5.54%`
- Omega4.4 HGB baseline OOS: `+22.21%`

`bandit_qnet` produced the best OOS readout (`+22.48%`) and best OOS full replay (`+25.24%`), but it failed the validation MDD guard. The selected `iql_awac` improves validation but gives weaker OOS than the HGB baseline.
