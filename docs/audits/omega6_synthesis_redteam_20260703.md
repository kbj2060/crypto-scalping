# Omega6 Synthesis Red-Team Audit - 2026-07-03

- Verdict: `CONDITIONAL_PASS_WITH_WARNINGS`
- Blockers: 0
- Warnings: 1

## Checks

| Check | Severity | Pass |
| --- | --- | --- |
| `fail_fast_on_broken_artifact_path` | blocker | True |
| `l2_train_val_non_overlap` | blocker | True |
| `l4_sidecar_train_window_before_val` | blocker | True |
| `l3_gate_train_window_before_val` | blocker | True |
| `forbidden_feature_prefixes_absent` | blocker | True |
| `l6_governor_reduce_only` | blocker | True |
| `futures_sizing_contract_notional_eq_margin_times_leverage` | blocker | True |
| `cost_stress_trade_count_non_collapse` | blocker | True |
| `sizing_sensitivity_informational` | warning | False |

## Warnings (non-blocking, must be read before promotion)
- **sizing_sensitivity_informational**: {"current_val_pnl": -13.482727414450302, "note": "2026-07-03 session found validation PnL flips from +27.69% (uncapped L4 mapping, cap=0.6) to -13.48% (MDD-capped L4 mapping, cap=0.3) using the SAME L2/L3 signal. This is a sizing-sensitivity finding, not a pass/fail gate by itself, but it is a material caveat: the underlying directional signal's edge is not clearly robust to reasonable resizing choices. See docs/model_contracts/omega6_synthesis_v1_20260703_contract.md."}

- JSON: `/home/llewyn/crypto-scalping/docs/audits/omega6_synthesis_redteam_20260703.json`
