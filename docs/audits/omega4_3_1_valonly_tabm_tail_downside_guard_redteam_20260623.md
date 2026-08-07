# Omega4.3.1 Validation-Only TabM Tail-Downside Guard Red-Team Audit - 2026-06-23

## Verdict

`omega4_3_1_valonly_tabm_tail_downside_guard_20260623`: **`REDTEAM_PASS_CLEAN_RESEARCH_TEST_NOT_UPGRADE`**

Scope: clean research-test audit only. This run replaces the HGB score and q10 risk models with side-split TabM regressors. It is not an upgrade candidate because performance is materially below the Omega4.3 baseline.

## Selection Hygiene

- Selected variant: `risk_4866`

- Validation-only recomputed top: `risk_4866`

- OOS excluded from filter/sort/tie-break; OOS is selected-row readout only.

## Metrics

| Split | PnL | MDD | WR | Trades | Avg Notional | Avg Margin | Avg Lev | Log-Risk Utility |

| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

| Baseline validation | `+30.33%` | `-7.91%` | `67.06%` | `85` | `0.5536` | `0.2346` | `2.2574` | `0.205548` |

| TabM validation | `+19.07%` | `-6.85%` | `67.06%` | `85` | `0.4451` | `0.2361` | `1.8199` | `0.152264` |

| Baseline OOS readout | `+32.44%` | `-5.72%` | `66.15%` | `65` | `0.5613` | `0.2374` | `2.2699` | `0.262865` |

| TabM OOS readout | `+17.17%` | `-4.61%` | `66.15%` | `65` | `0.4017` | `0.2261` | `1.7194` | `0.150114` |

## Full Replay Diagnostic

| Split | PnL | MDD | WR | Trades | Log-Risk Utility |

| --- | ---: | ---: | ---: | ---: | ---: |

| Baseline validation full replay | `+31.34%` | `-10.52%` | `68.35%` | `79` | `0.191867` |

| TabM validation full replay | `+16.19%` | `-7.09%` | `65.88%` | `85` | `0.123740` |

| Baseline OOS full replay | `+33.73%` | `-5.73%` | `66.10%` | `59` | `0.271981` |

| TabM OOS full replay | `+17.72%` | `-4.62%` | `67.69%` | `65` | `0.154722` |

## Checks

- `report_exists`: `True`

- `runtime_contract_exists`: `True`

- `risk_sidecar_exists`: `True`

- `risk_sidecar_loads`: `True`

- `model_kind_is_tabm`: `True`

- `sidecar_has_no_hgb_model_object`: `True`

- `validation_only_top_matches_selected`: `True`

- `selection_policy_declares_validation_only`: `True`

- `notional_contract_ok`: `True`

- `sltp_not_notional_scaled`: `True`

- `full_replay_selection_not_applied`: `True`

- `feature_count_matches`: `True`

- `forbidden_feature_hits_zero`: `True`

## Upgrade Checks

- `validation_pnl_improved_vs_baseline`: `False`

- `validation_utility_improved_vs_baseline`: `False`

- `oos_pnl_improved_vs_baseline_readout`: `False`

- `oos_utility_improved_vs_baseline_readout`: `False`

## Artifacts

- Report: `tmp/causal_regen_20260516/omega4_3_1_valonly_tabm_tail_downside_guard_20260623/report.json`

- Runtime contract: `tmp/causal_regen_20260516/omega4_3_1_valonly_tabm_tail_downside_guard_20260623/runtime_contract.json`

- Risk sidecar: `tmp/causal_regen_20260516/omega4_3_1_valonly_tabm_tail_downside_guard_20260623/risk_sidecar.pkl`

- JSON audit: `tmp/causal_regen_20260516/omega4_3_1_valonly_tabm_tail_downside_guard_redteam_20260623/report.json`
