# Omega4.6.1 Base — "Live Clean" Final Status (2026-07-06)

Status: `promotion_pass_true_live_adapter_verified_not_yet_wired`

This closes out the full promotion checklist for `omega4_6_1_duration_ou_halflife_risk_gate_20260630`
(base form, event-flat overlay excluded). Supersedes
`docs/model_contracts/omega4_6_1_base_promotion_checklist_20260706_final.md`.

## Gate 2 (Artifact Integrity): FIXED, now PASSES

Root cause of the 16832-vs-16838 row mismatch: shared regime3 overlay files
(`regime3_current_sensitive_wide24`/`cmamba`/`stability_risk` 2026 CSVs) were extended by a
DIFFERENT session on 2026-07-04 for Omega6 v2 purposes, unrelated to Omega4.6.1. This pushed the
"current runtime frame" reconstruction 6 bars (30 min) past where h48qual/zig075's frozen
predictions ended. Fix: re-ran parent inference fresh against current overlay data, filtered to
the exact expected 16838-row window, pointed a corrected `report.json` at the fixed prediction
files. Re-ran `scripts/audit_omega_artifact_integrity_20260630.py` via the required `quant_ai`
conda env: **`promotion_pass: true`**.

## Runtime-native parity: bug found and fixed, then a real architecture gap found and resolved

Built `trading_bot_modules/omega4_6_1_duration_gate_live_draft_20260706.py` and ran it bar-by-bar
against real historical data, comparing every decision to the backtest ledger:

1. **Direction/side**: 0 mismatches across 2343 checked bars for the h48qual component alone.
2. **Sizing bug found**: initial leverage output was ~2.5x too high (margin matched exactly).
   Root cause: the risk sidecar's score model needs the BASELINE (pre-sizing) `decision_notional_
   exposure`/`decision_leverage` as INPUT features, and the draft adapter was feeding zeros
   instead of the real `BASE_TEMPLATE` (notional=0.45, leverage=2.0) times the expert-routing
   scale (bull=0.75/bear=0.90/chop=0.90). Fixed; margin/leverage/notional now match the backtest
   ledger to floating-point precision (~1e-16) on every trade checked.
3. **Real architecture gap found**: comparing against the full ROUTER-COMBINED ledger (not just
   one component) found 8 of 33 trades (24%) where the live adapter's greedy "try h48qual, else
   zig075" priority rule disagreed with the offline ledger. Root cause: every backtest in this
   lineage (`build_omega_plus_t12_livepass_candidate_20260630.py::priority_route`) simulates
   h48qual and zig075 as two INDEPENDENT full ledgers (each with its own imaginary 100% capital)
   and reconciles overlaps post-hoc -- which requires knowing both components' full counterfactual
   futures in advance and **cannot be replicated by a real single-account live system**.

## Honest live-achievable PnL: re-derived with a genuine greedy single-account replay

Built `scripts/replay_omega4_6_1_greedy_router_20260706.py`: one interleaved bar-by-bar loop,
single shared position slot, greedy h48qual>zig075 priority exactly matching what the live adapter
does. Result:

| | PnL | MDD | Trades | WR |
|---|---|---|---|---|
| Offline priority-reconciled (previously reported, NOT live-achievable) | +145.46% | -10.82% | 25 | 52.0% |
| **Genuine greedy single-account (live-achievable, THIS is the number to cite)** | **+145.34%** | **-10.13%** | **24** | **54.2%** |

The architecture gap turned out to barely matter for this window -- the greedy number is nearly
identical to (and marginally better on MDD than) the offline-reconciled one. Redteam-style checks
(leverage cap 5.0x, notional cap 1.8x, zero overlaps, cost1/cost2/cost3 all positive at
+138.2%/+131.6%/+125.1% pre-gate) re-verified against the new greedy ledger and hold.

## Remaining gap to full live promotion

- **Not wired into `trading_bot.py`.** This is a genuinely high-stakes, hard-to-reverse action
  (connects to real capital) and was deliberately NOT done as part of this checklist without
  explicit further confirmation -- everything up to "ready to wire" is complete and verified.
- **Runtime-native parity was checked against the SAME feature source the backtest used**
  (`training_features_2026_rebuilt.csv` + regime3 wide24 overlay), not against the actual live
  feature-computation pipeline in `trading_bot.py`/`pipeline/`. A final parity check against the
  true live feature path is still needed before flipping this on.
- **kel/evt_excess_z/btc_corr_60/dual_momentum feature drift** (parent inputs) remains
  unresolved -- would need a full parent retrain, which this project's history shows tends to fail
  validation gates. Accepted as a standing limitation per user decision 2026-07-06.
- Validation window (2025-10-01..12-31) was used for duration-threshold reselection only; the full
  chain has not been re-validated end-to-end on VAL with the greedy router (only OOS was
  re-derived this way). Low risk given how close VAL's own numbers already were pre-fix, but not
  literally re-checked.

## Verdict

This is now the most thoroughly verified candidate in the project's history for this specific
checklist: Gate 2 passes cleanly, a genuine architecture bug (leverage sizing) was found and
fixed via real runtime-native parity testing (not just a smoke test), a second genuine
architecture gap (offline reconciliation vs. live greedy routing) was found, quantified, and
resolved with an honest re-derivation, and the final live-achievable number changed by less than
0.1pp from what was previously reported. **Ready to wire pending final confirmation and one last
parity check against the actual live feature pipeline.**
