# CORRECTION: Event-Flat Overlay Reverses to a FAILURE Under Genuine Fresh-Forward Replay

Status: `research_correction_prior_result_invalidated`

Last updated: 2026-07-06 KST

This corrects `docs/model_contracts/omega4_6_1_macro_event_veto_20260706.md`'s "haircut=0.00 (flat
during event)" result (previously reported as PnL +156.20%, MDD -10.82%, an improvement over the
+145.46%/-10.82% baseline). That number was flagged in
`docs/model_contracts/omega4_6_1_event_flat_live_promotion_audit_20260706.md` (Gate 1) as
**invalid for promotion** because it was computed by post-hoc numerical editing of an
already-finished saved ledger, not a genuine bar-by-bar fresh-forward causal walk. Per user
request, Gate 1 was actually closed: a real bar-by-bar replay was built
(`scripts/replay_omega4_6_1_event_flat_fresh_forward_20260706.py`, a modified copy of
`train_eval_omega4_2_risk_sidecar_20260622.py::_replay_with_risk` that checks a pre-known macro
event calendar at every bar and force-closes/blocks-new-entries during the flat window, using only
that bar's own state -- no saved ledger timestamps used as input).

## Self-check: reimplementation validated

With the flat-window mask disabled (all-False), this new bar-by-bar engine reproduces the earlier
extended-OOS baseline exactly: PnL 141.14%, MDD -13.78%, trades=33 (matches
`docs/model_contracts/omega4_6_1_extended_oos_20260706_retest.md`'s pre-duration-gate combined
result to 2 decimal places). The reimplementation is faithful.

## Result: the event-flat overlay is HARMFUL when tested honestly

| | PnL | MDD | trades |
|---|---|---|---|
| Baseline (no overlay), combined router | +141.14% | -13.78% | 33 |
| **Event-flat, genuine bar-by-bar** | **+90.33%** | **-29.77%** | 45 |
| Baseline + duration gate | +145.46% | -10.82% | 25 |
| **Event-flat + duration gate, genuine bar-by-bar** | **+39.27%** | **-32.40%** | 37 |

MDD nearly **triples** (-10.8%→-32.4%) and PnL drops by more than half. Trade count also jumps
(25→37, 33→45) because force-closing a position during an event window returns the strategy to
flat, and it can (and does) re-enter shortly after -- often catching a materially worse setup
right around the volatile event window, rather than cleanly avoiding it. The earlier post-hoc
approximation completely missed this because it only rescaled the existing trades' realized PnL
proportionally, without modeling the fact that a real flat-and-reopen cycle changes WHICH trades
happen at all.

## Why this matters

This is a direct, concrete demonstration of why AGENTS.md's Fresh-Forward Rule bans saved-ledger
post-processing as promotion/test evidence "regardless of performance numbers": the invalid
methodology produced a result that looked like a clear improvement (+156.20% vs +145.46% baseline)
but the properly-implemented version shows the same idea is actively destructive to both return
and drawdown. Had this not been re-tested honestly, it would have been a dangerous live-promotion
candidate based on an artifact of bad backtest methodology, not a real edge.

## Conclusion

**Do not pursue the event-flat overlay further.** It fails cleanly and substantially under
genuine causal testing. The remaining promotion-checklist items (artifact-integrity audit report,
live adapter code, redteam) are not worth building for an idea that has now failed honestly --
building promotion infrastructure around a failed idea would be wasted effort. If the user wants
to continue toward a promotable event-aware model, the next step should be testing whether a
MILDER intervention (partial haircut rather than full flat, implemented with the same genuine
bar-by-bar discipline) avoids the "reopen into a bad setup" problem, or abandoning the
macro-event-handling direction for this model (Omega4.6.1's own trading frequency is too low for
this to be a high-value lever regardless, as already noted in
`docs/model_contracts/omega4_6_1_macro_event_veto_20260706.md`).
