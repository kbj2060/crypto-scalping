# Verification Note: omega4_3_valonly_logrisk_tail050 (2026-07-06)

Status: `verified_on_existing_window_extension_blocked`

Scope: quick self-check of `omega4_3_valonly_logrisk_tail050_margin_leverage_sidecar_20260623`
against its own saved artifacts, per the user's request to identify Omega architectures with
genuinely trustworthy PnL under the Fresh-Forward Rule. NOT extended to new OOS data (blocked,
see below) -- this only checks whether the EXISTING published 2025 VAL / 2026 Jan-Feb OOS numbers
hold up under the most rigorous replay already computed for this candidate.

## Finding 1: the contract doc's headline numbers use the FASTER, less rigorous ledger method

The candidate directory has three ledger variants per split: `*_baseline_trade_ledger.csv` (fixed
flat sizing), `*_selected_risk_trade_ledger.csv` (rescaled PnL using the risk sidecar's
margin/leverage but REUSING the baseline's original exit timestamps -- a static rescale, faster
but doesn't re-simulate how a different position size changes the exit-head's timing decision),
and `*_selected_risk_replayed_trade_ledger.csv` (full bar-by-bar re-simulation with the new
size fed into the exit head's position-state inputs each bar -- the rigorous version).
`risk_mapping_ranking.csv` (and the contract doc's headline table) report the FASTER
`selected_risk` variant's numbers, not the replayed ones -- likely because the grid search over
many candidate mappings only replays the top pick (`--full-replay-top-k=1` in the training
script), for speed.

## Finding 2: the rigorous replayed numbers hold up (even improve slightly)

| Split | Reported (contract, fast method) | **Replayed (rigorous)** |
|---|---|---|
| VAL | PnL +30.33%, MDD -7.91%, trades 85, WR 67.1% | **PnL +31.34%, MDD -8.04%, trades 79, WR 68.4%** |
| OOS | PnL +32.44%, MDD -5.72%, trades 65, WR 66.2% | **PnL +33.73%, MDD -4.30%, trades 59, WR 66.1%** |

Trade count drops modestly (rescaling to a different position size changes when the exit-head
decides to close, a real and expected effect) but PnL/MDD are consistent or slightly BETTER under
the rigorous method. This is the OPPOSITE of what happened with the Omega4.6.1 event-flat overlay
(where the rigorous replay revealed a much WORSE result) -- here, the reported numbers appear to
be an honest, if slightly optimistic-on-trade-count, approximation of the true bar-by-bar result.

## Finding 3: extending to new 2026 OOS data is blocked (same wall as Omega6 v1)

This candidate's parent (`omega4_3head_parent72_loose_entry_quality_20260620_smoke_loose_entry_
loose_quality_terminal_giveback_exit_e2_train15k_exit15k_q070`) needs the FULL 172-feature
contract including 41 m7_* columns and 17 NeuralForecast (PatchTST/TiDE/DLinear) columns -- unlike
Omega4.6.1's parents, which happened to need zero of either. Regenerating these for an extended
window hits the exact wall already documented for the original Omega6 v1 frozen winner
(m7/SevenModelEnsemble pipeline restructured) AND the NF forecaster checkpoint incompatibility
found earlier today (`neuralforecast`/`lightning_fabric` version mismatch, `AttributeDict` missing).
**This candidate cannot be extended to fresh 2026 Q2+ data without resolving both blockers.**

## Finding 4: Gate 2 (Artifact Integrity) technically fails as-recorded

The risk sidecar's own `source_report.json` has an empty `precomputed_prediction_dir`, meaning it
was trained by calling the parent's live forward pass directly rather than loading precomputed
prediction CSVs -- this is one of the explicit fail conditions in
`docs/model_contracts/omega_artifact_integrity_policy_20260630.md`. Ironically this is NOT a
Fresh-Forward correctness problem (a live forward pass is if anything MORE causal than reading a
stale CSV) but it is a formal non-compliance with the artifact-integrity paperwork contract as
currently written, and would need to be re-run with `--precomputed-prediction-dir` pointing at
frozen parent prediction CSVs to pass Gate 2 cleanly.

## Verdict

Of the models checked so far, this is the most reassuring: the reported numbers survive rigorous
re-checking on the existing window (PnL/MDD hold, even improve slightly) unlike the event-flat
overlay. It cannot yet be extended to a fresh OOS window (m7/NF blocker) and has an outstanding
Gate 2 paperwork gap, but there is no evidence of a fundamental methodology problem like the one
found in Omega4.6.1's event-flat test.
