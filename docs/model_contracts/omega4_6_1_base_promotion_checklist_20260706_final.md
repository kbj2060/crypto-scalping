# Omega4.6.1 Base (No Event Overlay) — Promotion Checklist Final Status (2026-07-06)

Status: `research_strong_but_not_promotable_gate2_fail`

Scope: closes out the promotion-checklist gates enumerated in
`docs/model_contracts/omega4_6_1_event_flat_live_promotion_audit_20260706.md`, for the BASE
Omega4.6.1 candidate only (extended Jan-Jun 2026 OOS, event-flat overlay excluded -- that idea
failed cleanly, see `omega4_6_1_event_flat_fresh_forward_correction_20260706.md`).

## Gate-by-gate final status

| Gate | Status | Detail |
|---|---|---|
| 1. Fresh-Forward Rule | `partial` | Parent inference + risk-sidecar sizing + duration gate reuse the genuinely causal bar-by-bar `_replay_with_risk` engine (same one validated for the event-flat correction). The extended-window construction (feature frame, VAL-only duration-threshold reselection) is causal. Not re-verified as a single from-scratch end-to-end walk the way the event-flat correction was -- treat as strong but not identically rigorous. |
| 2. Omega Artifact Integrity Audit | **FAIL** | Ran `scripts/audit_omega_artifact_integrity_20260630.py` for real (via the `quant_ai` conda env) against a constructed `report.json`. Result: `promotion_pass=false`. Both components fail on `oos_prediction_timestamps_match_runtime_frame`: the frozen `oos_predictions_q050.csv`/`q075.csv` have 16832 rows, but freshly rebuilding the runtime frame from the CURRENT codebase (even pointed at the ORIGINAL, un-extended label/eval sources) produces 16838 rows -- a 6-row (0.036%) mismatch that **pre-dates today's work**. This is a genuine, pre-existing artifact-integrity gap in the frozen h48qual/zig075 components, not something introduced by the extension or the duration-gate reselection. |
| 3. Feature-vintage drift | `documented, partially resolved` | `ou_halflife` re-selected via genuine VAL-only grid search on the current formula (landed at 0.005417, ~identical to the frozen 0.005415348) -- this specific concern is closed. `kel`/`evt_excess_z`/`btc_corr_60`/`dual_momentum` (parent inputs) remain drifted; fixing requires a full parent retrain, declined by user (this project's retrain attempts on this family have a documented history of failing validation gates). |
| 4. Live adapter code | **done (draft)** | `trading_bot_modules/omega4_6_1_duration_gate_live_draft_20260706.py` -- fail-fast contract checks on both TabM bundles + risk sidecars, `decide_latest()` implementing router priority + sizing + duration gate. Smoke-tested successfully against real bundle/sidecar artifacts and a real feature frame. **Not imported by trading_bot.py, no FINAL_GOVERNOR_* flag** -- explicitly a bounded draft, not production-grade (see its docstring for the scope difference vs. `omega4_6_2_source_parent_live.py`'s 846-line adapter). |
| 5. Redteam-style check | **done, PASS** | `scripts/redteam_omega4_6_1_base_20260706.py`: leverage cap (5.0x), notional cap (1.8x), zero position overlaps, accounting consistency (max error 8e-17), notional contract consistency (max error 2e-16), and **cost-stress PASS at cost1/cost2/cost3** (+145.46% / +140.05% / +134.75% -- the edge does NOT collapse under 3x fee/slip stress, unlike several earlier Sigma-family failures this session). A same-bar-vs-1-bar-lag test on `ou_halflife` produced identical results, which is uninformative rather than reassuring (the feature is smooth enough that a 1-bar shift rarely crosses the threshold) -- not a strong causality confirmation, just not a red flag either. |

## Bottom line

**Still not promotable**, but for a much narrower and more specific reason than before: **Gate 2
genuinely fails**, and it fails on the model's own frozen, untouched original artifacts -- this
is not something introduced by today's extension work, and not something a quick fix resolves
(the 6-row mismatch would need its own root-cause investigation, likely another multi-hour dive
into exactly which upstream file/join changed between when the predictions were frozen and now).
Every other gate checked out reasonably well: the underlying economics are cost-stress-robust,
the duration gate threshold was properly re-derived (not just reused blindly), and a draft live
adapter now exists. If this candidate is revisited, the next concrete step is diagnosing the
16832-vs-16838 row discrepancy specifically (likely a small, fixable data-pipeline drift), not a
larger architectural rework.
