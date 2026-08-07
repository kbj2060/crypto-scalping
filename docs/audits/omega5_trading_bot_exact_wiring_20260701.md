# Omega5 Trading Bot Exact Wiring Audit - 2026-07-01

- Verdict: `OMEGA5_EXACT_STATIC_WIRING_PASS`
- Exact static wiring pass: `True`

## Checks

- `omega5_source_parent_default_on`: `True` (must) {'expected': 'source parent is enabled by default when Omega5 is enabled'}
- `omega5_requires_source_parent_no_substitute`: `True` (must) {'forbidden_substitute': 'Omega1.2.1/Omega3 parent'}
- `omega5_parent_provider_only_source_parent`: `True` (must) {'method': 'FinalGovernorRuntime._omega5_parent_decision'}
- `omega5_adapter_validates_parent_identity`: `True` (must) {'adapter': '/home/llewyn/crypto-scalping/trading_bot_modules/omega5_live.py'}
- `source_parent_uses_runtime_native_forward_artifact`: `True` (must) {'adapter': '/home/llewyn/crypto-scalping/trading_bot_modules/omega4_6_2_source_parent_live.py', 'meaning': 'uses causal TabM bundle + risk sidecar runtime inference, not validation/OOS event replay'}
- `source_parent_forbids_historical_replay_provider`: `True` (must) {'adapter': '/home/llewyn/crypto-scalping/trading_bot_modules/omega4_6_2_source_parent_live.py', 'forbidden': 'historical validation/OOS ledger or interval replay as live decision source'}
- `ledger_replay_live_path_blocked`: `True` (must) {'adapter': '/home/llewyn/crypto-scalping/trading_bot_modules/omega4_6_2_runtime_adapter.py'}
- `omega5_open_journal_recovery_requires_contract_fields`: `True` (must) {'method': 'FinalGovernorRuntime._recover_omega5_state_from_open_journal'}
- `omega5_runtime_state_load_fail_fast_no_trace_fallback`: `True` (must) {'method': 'FinalGovernorRuntime._load_runtime_state + _recover_omega5_state_from_open_journal', 'reason': 'Omega5 active state must fail fast when sizing provenance is absent'}
- `omega5_active_position_no_reconcile_fallback`: `True` (must) {'method': 'FinalGovernorRuntime._manage_omega5_position'}
- `omega5_source_exit_event_is_runtime_owner`: `True` (must) {'method': 'FinalGovernorRuntime._manage_omega5_position'}
- `omega5_decision_priority_before_omega121`: `True` (must) {'method': 'FinalGovernorRuntime.decide'}
- `omega5_entry_persists_risk_contract`: `True` (must) {'method': 'FinalGovernorRuntime._decide_omega5_entry'}
- `omega5_entry_persists_sizing_trace_contract`: `True` (must) {'method': 'FinalGovernorRuntime._decide_omega5_entry + GovernorPositionRouter._journal_audit_fields', 'reason': 'Omega5 OPEN/CLOSE journal rows must retain sizing provenance'}
- `omega5_entry_persists_live_native_source_parent_provenance`: `True` (must) {'method': 'GovernorPositionRouter._journal_audit_fields', 'reason': 'live-native Omega5 rows must retain source model and sidecar provenance'}
- `omega5_risk_math_uses_price_move_times_notional`: `True` (must) {'adapter': '/home/llewyn/crypto-scalping/trading_bot_modules/omega5_live.py'}
- `trading_bot_does_not_import_ledger_replay_adapter`: `True` (must) {'file': '/home/llewyn/crypto-scalping/trading_bot.py'}
- `runtime_native_walkforward_full_pass`: `True` (must) {'report': '/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega5_source_parent_runtime_native_walkforward_20260701/report.json'}

## Scope

- This audit verifies trading-bot wiring and fail-fast contracts.
- It does not replace a runtime-native walk-forward backtest.
- Historical ledger replay remains audit-only and is forbidden as a live decision provider.

## Artifacts

- JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega5_trading_bot_exact_wiring_audit_20260701/omega5_trading_bot_exact_wiring_audit_20260701.json`
