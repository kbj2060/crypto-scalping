# Omega5 Source Parent Predictive Distill - 2026-07-02

- Verdict: `SOURCE_PARENT_PREDICTIVE_ARTIFACT_NEEDS_REVIEW`
- Proof bundle: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_source_parent_predictive_distill_20260702/proof_validation_only_bundle.joblib`
- Live bundle: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_source_parent_predictive_distill_20260702/live_val_oos_bundle.joblib`
- Feature count: `195`
- Live feature coverage: `1.0000`

## Proof Metrics

- Validation event F1: `0.5735`
- OOS event F1: `0.0000`
- OOS predicted/true events: `22` / `97`

## Contract

- Proof model trains on validation only and evaluates OOS without OOS training.
- Live model trains on validation+OOS for deployment; it is not used as OOS proof.
- The artifact predicts current-bar source-parent action/side/notional from causal features, not historical policy rows.

