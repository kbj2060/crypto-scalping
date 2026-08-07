# Sigma2 Sequence-Zigzag Model — Data Contract

Status: `research_failed_validation_gates_not_promotable`

Last updated: 2026-07-05 KST

Lineage: second from-scratch m7-free architecture (after [[sigma1]] /
`docs/model_contracts/sigma1_seq_barrier_20260704_contract.md`). Designed after the M7
feature lineage was confirmed unrecoverable (see omega6 contract doc, "M7 recovery attempt").

## Design (each choice tied to documented evidence)

| Decision | Rationale |
|---|---|
| Causal GRU 192x2, 64-bar window | Persistence/temporal debouncing was the only lever behind the project's single val+OOS pass |
| zigzag_action swing label | The label behind that only pass; barrier-matched per-bar label confirmed to OOS-flip (Sigma1/priority-1) |
| Train 2024-01..2025-04 (16mo) + holdout 2025-05..06 | 2024 usable now that m7's fit-on-2024 scheme is abandoned; ~2x the frozen winner's training span |
| Validation 2025-07..12 (6mo) | Old Oct-Dec window exhausted by 900+ prior variants; Jul-Sep never used for selection by any round |
| Features: 141 = stationary base + wide24/cmamba/stability-risk overlays | All reproducibility-verified for the 2026 extension; excluded m7_* (unrecoverable), 6 drift-confirmed formulas, level cols, NF ai_* (absent for 2024) |
| 2 seeds from the start | Sign-consistency pre-registered as a gate condition |
| Pre-registered 18-config execution sweep | threshold {0.45,0.55,0.65} x persistence {0,2,3} x (tp,sl) {(15,5),(13,4)} x cd12 |
| Gates (6mo-scaled) | cost1&cost3 PnL>0, MDD>=-20% both, trades>=100, months>=5, AND seedB cost1>0 at same config |

Scripts: `scripts/train_sigma2_seq_zigzag_20260705.py`,
`scripts/precompute_sigma2_tape_20260705.py`, `scripts/replay_sigma2_gates_20260705.py`.
Artifacts: `tmp/causal_regen_20260516/sigma2_seq_zigzag_20260705_seed{A,B}/`,
`sigma2_tape_seed{A,B}_20260705/`, `sigma2_gates_20260705/sigma2_gate_ranking.csv`.

## Result: FAILED (0/18 configs), untouched 2026-03-02..06-30 window NOT scored

- Training: both seeds overfit immediately (holdout loss minimum at epoch 1, rising steeply
  after; early stop preserved epoch-1 weights). The GRU memorizes zigzag segments rather than
  learning transferable structure.
- Best cost1 results looked spectacular on seedA (`qt0.65 p0 tp13/sl4`: cost1 **+71.0%**,
  MDD -18.9%, 247 trades) but **cost3 was negative in 17/18 seedA configs** (best -1.07 at that
  same config), and the single cost3-positive config (`qt0.65 p2 tp13/sl4`: cost3 +11.4%) failed
  the cost1 MDD gate (-22.2%).
- **Severe seed instability**: seedB at the same configs collapsed (e.g. `qt0.65 p0 tp13/sl4`:
  seedA +71.0% vs seedB **+2.1%** cost1; most seedB configs were cost1-negative or barely
  positive). The apparent seedA edge is substantially seed noise, exactly what the
  pre-registered two-seed condition exists to catch.
- Signal rate 75-80% of bars nonzero at default threshold — chattery, like every neural signal
  in this project so far.

Per pre-registered discipline: grid NOT expanded after seeing results; the untouched fresh
window (2026-03-02..06-30) was NOT scored and remains pristine.

## Interpretation (cumulative, now 5 failed approaches on this feature set)

Tabular TabM (needs unrecoverable m7), TabM-no-m7, GRU-barrier-label (Sigma1), GRU-zigzag-label
(Sigma2), plus all filter/sizing/ensemble variations: everything on the current 5-minute
engineered-feature universe fails the cost3 stress gate out-of-sample or is seed-unstable. The
consistent failure signature — great cost1, dead cost3, high signal rate, instant overfit — says
the learnable signal at this frequency/feature-set is mostly noise-level microstructure whose
profits are eaten by realistic costs. The frozen winner's m7-dependent edge cannot be rebuilt
and its uniqueness (1 pass in ~1700 evaluated configs across all rounds) is itself, in
hindsight, plausibly a multiple-comparisons survivor rather than a durable edge.

Recommendation for any future attempt: change the INFORMATION, not the architecture — lower
frequency (15m-1h bars with proportionally larger barriers, so costs are a smaller fraction of
per-trade move), or genuinely new data sources (order-book depth, cross-asset), or accept
longer holding horizons. The 2026-03-02..06-30 window remains unscored and should be preserved
as the one-shot for whatever comes next.
