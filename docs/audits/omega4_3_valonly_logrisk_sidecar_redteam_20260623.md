# Omega4.3 Validation-Only Log-Risk Sidecar Red-Team Audit - 2026-06-23

## Verdict

`omega4_3_valonly_logrisk_tail050_margin_leverage_sidecar_20260623`: **`REDTEAM_PASS_CLEAN_RESEARCH_BASELINE_NOT_LIVE_WIRED`**

Scope: clean research-baseline audit only. This is not a live exchange promotion
PASS because runtime-native parity, exchange leverage/margin sync, and shadow or
paper smoke were not run.

## What Changed From Failed 4.3

- Previous blocker: OOS was used in mapping selection via `oos_mdd` eligibility
  and OOS log-risk tie-break.
- New selection rule: `validation-only log_risk max with validation_mdd >= -8.00 and trades >= 0.95 * baseline trades; OOS excluded from filter/sort/tie-break`
- Selected variant: `risk_3473`
- Validation-only top check: `True`

## Passed Checks

- Forbidden feature hits: `0`
- Sidecar feature count: `29`
- Train ledger is self-contained in the promoted directory: `True`
- Sizing contract: `notional = margin_fraction * leverage`
- SLTP remains raw price-move based and `notional_scaled_sltp = false`
- Full replay selection applied: `False`
- Runtime dynamic-risk exit enabled: `False`

## Metrics

| Split | PnL | MDD | WR | Trades | Avg Notional | Avg Margin | Avg Lev | Log-Risk Utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | `+30.33%` | `-7.91%` | `67.06%` | `85` | `0.5536` | `0.2346` | `2.2574` | `0.205548` |
| OOS readout | `+32.44%` | `-5.72%` | `66.15%` | `65` | `0.5613` | `0.2374` | `2.2699` | `0.262865` |

OOS is a selected-row readout, not a filter/sort/tie-break input.

## Warnings

- Risk sidecar uses only `242` train trades. This is acceptable for research PASS
  but still fragile for live deployment.
- Full dynamic-risk exit replay is still diagnostic only: validation `+31.34%`, MDD `-10.52%`.
- Live parity and exchange execution checks remain separate.

## Artifacts

- Manifest: `data/ensemble/supervised/omega4_3_valonly_logrisk_tail050_margin_leverage_sidecar_20260623/candidate_manifest.json`
- Runtime contract: `tmp/causal_regen_20260516/omega4_3_valonly_logrisk_tail050_margin_leverage_sidecar_20260623/runtime_contract.json`
- JSON audit: `tmp/causal_regen_20260516/omega4_3_valonly_logrisk_sidecar_redteam_audit_20260623/report.json`
- Risk sidecar: `tmp/causal_regen_20260516/omega4_3_valonly_logrisk_tail050_margin_leverage_sidecar_20260623/risk_sidecar.pkl`
