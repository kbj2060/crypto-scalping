# Omega5 Current Live Snapshot Contract - 2026-07-02

- Verdict: `OMEGA5_LIVE_CURRENT_SNAPSHOT_BLOCKED_AS_EXPECTED`
- Contract proof pass: `True`
- Current snapshot timestamp: `2026-07-01 16:45:00`
- Current snapshot rows: `500`

## Checks

- `current_live_snapshot_source_parent_failfast`: `True` (must) {'snapshot': '/home/llewyn/crypto-scalping/data/live/decision_feature_frame_snapshot.pkl.gz', 'timestamp': '2026-07-01 16:45:00', 'adapter_returned_decision': False, 'error': 'Omega4.6.2 reference policy has no promoted artifact coverage for timestamp 2026-07-01 16:45:00; live decisions outside the promoted validation/OOS policy window must fail fast'}
- `predictive_source_parent_artifact_not_declared`: `True` (must) {'runtime_contract': '/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_cap220_short_boost125_time_stop120h_20260630/runtime_contract.json', 'runtime_native_replay_available': False, 'historical_trade_ledger_fallback_allowed': False, 'exact_threshold_parent_predictions_required': True}

## Meaning

- PASS here means the current live timestamp is blocked as expected because the promoted source-parent artifacts only cover validation/OOS windows.
- This prevents Omega5 from silently using historical event-window replay as a future live decision provider.
- A future live promotion must provide a predictive source-parent artifact and update this proof accordingly.

## Artifacts

- JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega5_live_current_snapshot_contract_20260702/report.json`
