# Omega4.3 Log-Risk Sidecar Red-Team Audit - 2026-06-23

## Verdict

`omega4_3_logrisk_tail050_margin_leverage_sidecar_20260623` is **not a full red-team PASS** for clean OOS or live promotion.

Verdict: `REDTEAM_FAIL_FOR_CLEAN_OOS_AND_LIVE_PROMOTION__RESEARCH_CONTRACT_PARTIAL_PASS`

The promoted sizing-only research contract passes the direct feature/accounting
checks, but risk mapping selection used OOS metrics. Therefore the reported OOS
`+31.42%` must not be cited as untouched clean OOS evidence.

## Blocking Findings

1. OOS was used in risk mapping selection.
   - Selection rule: `validation log_risk max with validation_mdd >= -8.00, oos_mdd >= -5.70, and trades >= 0.95 * baseline trades`
   - Code path applies `oos_mdd >= -5.70` eligibility and uses OOS log-risk as a tie-break.
   - Validation-only top row: `risk_3473` with validation log-risk `0.205548`, validation PnL `+30.33%`, OOS MDD `-5.7160%`.
   - Promoted row: `risk_1673` with validation log-risk `0.204769`, validation PnL `+29.39%`, OOS MDD `-5.3669%`.
   - Impact: OOS `+31.42%` is diagnostic, not clean untouched OOS.

2. Full dynamic-risk exit replay is not promotion-safe.
   - Full replay validation: `+30.25%`, MDD `-10.25%`, trades `79`.
   - The promoted contract correctly disables dynamic-risk exit timing.
   - Any live path that lets sidecar margin/leverage alter exit-head state must be retrained and re-audited.

## Passed Checks

- Registry, manifest, and runtime contract point to `omega4_3_logrisk_tail050_margin_leverage_sidecar_20260623`.
- Risk sidecar feature count: `29`.
- Forbidden feature hits: `0`.
- Parent outputs for train/validation are generated with `oof=True`; OOS uses `oof=False` only after selection.
- Sizing ledgers satisfy `risk_notional = risk_margin_fraction * risk_leverage` within floating-point tolerance.
- Replay ledgers satisfy `notional = margin_fraction * leverage` within floating-point tolerance.
- SLTP contract remains raw price-move based; `notional_scaled_sltp = false`.
- Runtime declares fail-fast behavior for missing sidecar or contract mismatch.

## Feature Audit

Sidecar input columns:

```text
parent_router_confidence
parent_router_margin
parent_dir_p_cash
parent_dir_p_long
parent_dir_p_short
parent_dir_confidence
parent_dir_side_edge
parent_dir_trade_prob
parent_dir_action
parent_quality_p_cash
parent_quality_p_long
parent_quality_p_short
parent_quality_for_action
parent_quality_threshold
parent_final_action
parent_router_expert_bear
parent_router_expert_bull
parent_router_expert_chop
decision_action
decision_side
decision_quality_score
decision_confidence
decision_notional_exposure
decision_leverage
decision_position_fraction
decision_take_profit
decision_stop_loss
decision_rr
atr_pct_runtime
```

No direct hits for forbidden prefixes/tokens:
`clean_regime4_`, `clean_regime_2024_unsup_v4_`, `regime4_pred_`,
`regime3_pred_`, `teacher_`, `teacher_oof_`, `a5dir_`, `target`, `future`,
`label`, `pnl`, `zigzag`, `wave3`, `tp_sl_action_score`.

## Split Audit

| Split | Rows | Long | Short | Entry Start | Entry End |
| --- | ---: | ---: | ---: | --- | --- |
| Train | `242` | `75` | `167` | `2025-01-01 05:00:00` | `2025-09-29 22:35:00` |
| Validation | `85` | `22` | `63` | `2025-10-01 00:05:00` | `2025-12-19 07:35:00` |
| OOS | `65` | `20` | `45` | `2026-01-01 04:55:00` | `2026-02-28 10:25:00` |

## Metrics

Sizing-only promoted contract:

| Split | PnL | MDD | WR | Trades | Avg Notional | Avg Margin | Avg Lev |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | `+29.39%` | `-7.66%` | `67.06%` | `85` | `0.5297` | `0.2237` | `2.2574` |
| OOS | `+31.42%` | `-5.37%` | `66.15%` | `65` | `0.5382` | `0.2272` | `2.2699` |

Because OOS was part of selection constraints, these OOS metrics are diagnostic.

## Warnings

- The sidecar target has only `242` trade-level train rows. Side-split HGB has overfit risk.
- The promoted 4.3 directory does not contain `train_baseline_trade_ledger.csv`; it references the source run ledger instead.
- Dynamic leverage output is research-safe only under the explicit accounting contract. Live exchange leverage/margin/liquidation parity remains unaudited.

## Required Remediation For Full PASS

1. Re-run risk mapping selection with validation-only criteria. Do not filter, sort, or tie-break by OOS.
2. Evaluate the selected mapping once on 2026 OOS, or better, on a fresh later holdout/walk-forward window.
3. Keep dynamic-risk full replay disabled unless exit head/state features are retrained under the new margin/leverage contract.
4. Copy or regenerate the train ledger into the promoted audit bundle for self-contained reproducibility.
5. Run runtime-native parity and shadow/paper smoke before any live wiring.

## Artifacts

- JSON report: `tmp/causal_regen_20260516/omega4_3_logrisk_sidecar_redteam_audit_20260623/report.json`
- Manifest: `data/ensemble/supervised/omega4_3_logrisk_tail050_margin_leverage_sidecar_20260623/candidate_manifest.json`
- Runtime contract: `tmp/causal_regen_20260516/omega4_3_logrisk_tail050_margin_leverage_sidecar_20260623/runtime_contract.json`
- Risk sidecar: `tmp/causal_regen_20260516/omega4_3_logrisk_tail050_margin_leverage_sidecar_20260623/risk_sidecar.pkl`
