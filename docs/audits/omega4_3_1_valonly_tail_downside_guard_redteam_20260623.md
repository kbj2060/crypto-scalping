# Omega4.3.1 Validation-Only Tail-Downside Guard Red-Team Audit - 2026-06-23

## Verdict

`omega4_3_1_valonly_tail_downside_guard_20260623`: **`REDTEAM_PASS_CLEAN_RESEARCH_CANDIDATE_NOT_LIVE_WIRED`**

Scope: clean research-candidate audit only. This is not a live exchange promotion PASS because runtime-native parity, exchange leverage/margin sync, and shadow or paper smoke were not run.

## Selection Hygiene

- Selected variant: `risk_0488`

- Validation-only recomputed top: `risk_0488`

- OOS excluded from filter/sort/tie-break; OOS is selected-row readout only.

## Metrics

| Split | PnL | MDD | WR | Trades | Avg Notional | Avg Margin | Avg Lev | Log-Risk Utility |

| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |

| Baseline validation | `+30.33%` | `-7.91%` | `67.06%` | `85` | `0.5536` | `0.2346` | `2.2574` | `0.205548` |

| Candidate validation | `+30.09%` | `-6.94%` | `67.06%` | `85` | `0.5327` | `0.2230` | `2.2517` | `0.206614` |

| Baseline OOS readout | `+32.44%` | `-5.72%` | `66.15%` | `65` | `0.5613` | `0.2374` | `2.2699` | `0.262865` |

| Candidate OOS readout | `+32.28%` | `-5.54%` | `66.15%` | `65` | `0.5348` | `0.2231` | `2.2639` | `0.263117` |

## Full Replay Diagnostic

| Split | PnL | MDD | WR | Trades | Log-Risk Utility |

| --- | ---: | ---: | ---: | ---: | ---: |

| Baseline validation full replay | `+31.34%` | `-10.52%` | `68.35%` | `79` | `0.191867` |

| Candidate validation full replay | `+30.72%` | `-9.65%` | `68.35%` | `79` | `0.190013` |

| Baseline OOS full replay | `+33.73%` | `-5.73%` | `66.10%` | `59` | `0.271981` |

| Candidate OOS full replay | `+33.56%` | `-5.56%` | `65.00%` | `60` | `0.272202` |

## Checks

- `report_exists`: `True`

- `runtime_contract_exists`: `True`

- `risk_sidecar_exists`: `True`

- `validation_only_top_matches_selected`: `True`

- `selection_policy_declares_validation_only`: `True`

- `notional_contract_ok`: `True`

- `sltp_not_notional_scaled`: `True`

- `full_replay_selection_not_applied`: `True`

- `validation_utility_improved_vs_baseline`: `True`

- `validation_mdd_improved_vs_baseline`: `True`

- `oos_readout_utility_not_worse`: `True`

- `oos_readout_mdd_improved`: `True`

- `risk_sidecar_loads`: `True`

- `feature_count_matches`: `True`

- `forbidden_feature_hits_zero`: `True`

## Artifacts

- Report: `tmp/causal_regen_20260516/omega4_3_1_valonly_tail_downside_guard_20260623/report.json`

- Runtime contract: `tmp/causal_regen_20260516/omega4_3_1_valonly_tail_downside_guard_20260623/runtime_contract.json`

- Risk sidecar: `tmp/causal_regen_20260516/omega4_3_1_valonly_tail_downside_guard_20260623/risk_sidecar.pkl`

- JSON audit: `tmp/causal_regen_20260516/omega4_3_1_valonly_tail_downside_guard_redteam_20260623/report.json`
