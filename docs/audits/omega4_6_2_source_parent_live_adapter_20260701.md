# Omega4.6.2 Source Parent Live Adapter Audit - 2026-07-01

- Verdict: `SOURCE_PARENT_LIVE_ADAPTER_PASS`
- Adapter implementation pass: `True`

## Checks

- `ledger_replay_live_decision_blocked`: `True` (must) {'adapter': '/home/llewyn/crypto-scalping/trading_bot_modules/omega4_6_2_runtime_adapter.py'}
- `source_parent_live_adapter_exists`: `True` (must) {'adapter': '/home/llewyn/crypto-scalping/trading_bot_modules/omega4_6_2_source_parent_live.py'}
- `source_parent_adapter_does_not_read_ledgers`: `True` (must) {'forbidden': ['selected_validation_ledger', 'selected_oos_ledger', 'pd.read_csv']}
- `source_parent_uses_predictive_artifacts`: `True` (must) {'components': ['h48qual', 'zig075']}
- `source_parent_reconstructs_policy_contract`: `True` (must) {'policy_layers': ['cap220', 'fine_exposure', 'loss_governor', 'short_rsi_gate']}
- `trading_bot_has_guarded_source_parent_switch`: `True` (must) {'default': 'enabled; Omega5 fails fast if source parent is disabled or missing'}
- `omega5_entry_uses_selected_parent_provider`: `True` (must) {'entrypoint': 'FinalGovernorRuntime._decide_omega5_entry'}

## Interpretation

- The selected Omega4.6.2 ledger replay adapter is now explicitly historical-only.
- The new source-parent adapter is a forward policy path: it loads the two TabM bundles and applies h48qual/zig075 routing, cap220 exposure, and source-parent exposure/governor logic without reading validation/OOS ledgers.
- The trading bot switch is default-on and Omega5 fails fast if the source parent is disabled or missing.

## Artifacts

- JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_source_parent_live_adapter_audit_20260701/omega4_6_2_source_parent_live_adapter_audit_20260701.json`
