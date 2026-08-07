# Omega5 Live Backtest Parity Audit - 2026-07-01

- Verdict: `LIVE_BACKTEST_PARITY_PASS`
- Can reproduce source backtest in live: `True`
- Runtime route pass: `True`

## Contract Snapshot

- Source model: `omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701`
- Source backtest parent: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`
- Live parent model: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701`
- Source validation PnL: `675.3209%`
- Source OOS PnL: `212.6850%`
- Source avg notional val/oos: `2.5969` / `2.6981`
- Recent live avg notional: `0.8505`

## Checks

- `omega5_runtime_native_uses_signal_immediate_route`: `True` (must) {'route': 'runtime_native_signal_immediate_maker_limit'}
- `omega5_live_parent_matches_source_backtest_parent`: `True` (blocker) {'live_parent': 'omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701', 'source_parent': 'omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701'}
- `omega5_adapter_declares_overlay_on_external_parent`: `True` (blocker) {'adapter': '/home/llewyn/crypto-scalping/trading_bot_modules/omega5_live.py', 'meaning': 'Omega5 sizes an externally supplied parent decision'}
- `source_parent_predictive_artifact_available`: `True` (blocker) {'adapter': '/home/llewyn/crypto-scalping/trading_bot_modules/omega4_6_2_source_parent_live.py', 'runtime_contract': '/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_cap220_short_boost125_time_stop120h_20260630/runtime_contract.json', 'source_parent_live_adapter_audit': '/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_source_parent_live_adapter_audit_20260701/omega4_6_2_source_parent_live_adapter_audit_20260701.json', 'source_parent_live_adapter_verdict': 'SOURCE_PARENT_LIVE_ADAPTER_PASS', 'exact_threshold_parent_predictions_required': True, 'historical_trade_ledger_fallback_allowed': False, 'meaning': 'live parity requires runtime-native source-parent forward inference, not validation/OOS event-window replay'}
- `omega462_adapter_is_historical_replay_only`: `True` (blocker) {'adapter': '/home/llewyn/crypto-scalping/trading_bot_modules/omega4_6_2_runtime_adapter.py', 'allowed_use': 'historical validation/OOS replay only'}
- `omega5_source_parent_switch_default_enabled`: `True` (evidence) {'reason': 'Omega5 source parent is mandatory; Omega1.2.1/Omega3 substitution is forbidden'}
- `omega5_open_journal_persists_sizing_trace`: `True` (blocker) {'reason': 'OPEN/CLOSE journal rows must preserve Omega5 sizing provenance'}
- `active_omega5_open_journal_trace_contract`: `True` (blocker) {'active_open_count': 0, 'bad_trade_ids': []}
- `runtime_native_walkforward_replay_completed`: `True` (blocker) {'expected_report': '/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega5_source_parent_runtime_native_walkforward_20260701/report.json', 'observed_verdict': 'OMEGA5_RUNTIME_NATIVE_PROOF_PASS', 'observed_pass': True, 'reason': 'static wiring is not enough to claim source backtest PnL parity'}
- `recent_live_notional_matches_source_scale`: `False` (evidence) {'recent_live_avg_notional': 0.8505, 'source_oos_avg_notional': 2.6980956006526533, 'recent_open_count': 5}
- `current_open_position_created_after_next_open_repair`: `False` (evidence) {'trade_id': '', 'entry_price_source': ''}

## Required Contract

- Omega5 live parity is source-immediate, not next-open.
- The live path must use the Omega4.6.2 loss-cluster parent for base notional and the Omega5 reference source-policy event artifact for entry/exit timing.
- The final Omega5 ledger must not be used as the live decision provider.
- The historical ledger replay adapter is acceptable for audits only; it must not be used as a live future-timestamp decision provider.

## Artifacts

- JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega5_live_backtest_parity_20260701/omega5_live_backtest_parity_audit_20260701.json`
