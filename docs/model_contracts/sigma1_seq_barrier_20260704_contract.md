# Sigma1 Sequence-Barrier Model — Data Contract

Status: `research_failed_validation_gates_not_promotable`

Last updated: 2026-07-04 KST

Lineage: NEW model family (not Omega/Alpha). Designed from the confirmed lessons of the Omega6
v2 session documented in `docs/model_contracts/omega6_synthesis_v1_20260703_contract.md`.

## Design

A causal GRU sequence classifier replacing the Omega tabular-TabM L2, built to address three
documented root causes at the architecture level instead of via bolt-on filters:

| Lesson (from Omega6 v2 sessions) | Sigma1 design response |
|---|---|
| Persistence/hysteresis filter was the only lever that ever produced a val+OOS pass — temporal context matters | Causal GRU over a 48-bar (4h) window learns temporal persistence natively |
| Barrier-matched label was the right target but per-bar path noise caused an OOS sign-flip (priority-1 failure) | Label smoothed: LONG/SHORT only if the per-bar barrier-matched label agrees for 3 consecutive bars, else CASH |
| Omega L2 loader chain cannot reach 2024 data (no 2024 candidates file) | Reads `data/splits/year_oos/*.csv` directly → trains on 2024 + 2025-Jan..Jul (~2x data, includes a different regime year) |
| 2024→2026 price levels differ hugely; level features would be OOD under a train-fit standardizer | All level-like columns (raw OHLC, volumes, OI value, trade counts) excluded; 122 stationary engineered features only |

- Trainer: `scripts/train_sigma1_seq_barrier_20260704.py` (GRU 192x2, dropout 0.1, class-weighted CE,
  AdamW, early stop on internal holdout 2025-08-01..09-30 — inside the train region, fresh-forward
  val untouched during training). Seed 260704. Bundle:
  `tmp/causal_regen_20260516/sigma1_seq_barrier_20260704/sigma1_bundle.pt`.
- Inference tape: `scripts/precompute_sigma1_tape_20260704.py` (v2-replay-compatible schema).
- Gate check: `scripts/replay_sigma1_gates_20260704.py` — a DELIBERATELY SMALL pre-registered
  sweep (threshold {0.45,0.55,0.65} x persistence {0,2,3} x frozen execution mechanics
  tp15x/sl5x/cd12/notional 0.6 = 9 configs) to avoid the multiple-comparisons trap of the
  earlier 900+-variant searches.

## Result: FAILED validation gates (0/9), OOS never touched

| config | cost1 PnL | cost1 MDD | cost3 PnL | cost3 MDD | trades | WR (c1) |
|---|---|---|---|---|---|---|
| qt0.65 p0 (best) | +4.94% | -14.26% | **-1.18%** | -12.72% | 107 | 42.1% |
| qt0.65 p2 | +1.64% | -16.03% | -12.76% | -19.67% | 103 | 42.7% |
| qt0.65 p3 | +3.18% | -20.88% | -16.45% | -22.70% | 103 | 39.8% |
| (remaining 6 configs) | -2.7% .. -14.3% | | -18.0% .. -31.9% | | | |

Every config is cost3-negative; most are cost1-negative too. The model signals on 59% of bars at
the default threshold (chattery, like the original Omega6 L2) and its realized trade win rates
(38-43%) are below the frozen Omega6 v2 winner's (44.6%), despite the architectural changes.

Per pre-registered discipline the sweep was NOT expanded after seeing these results, and the
Jan-Feb 2026 OOS window was not read at any point.

## Interpretation

This is now the third independent modeling approach (tabular TabM on zigzag labels; tabular TabM
on barrier-matched labels; sequence GRU on smoothed barrier-matched labels + 2x data) to fail to
beat — or in this case even match — the frozen Omega6 v2 winner under the cost3 stress gate on
the same validation window. Combined with the earlier findings (ensembling hurt; higher
thresholds hurt; every auxiliary layer hurt), the consistent picture is that the exploitable
edge in this feature set over this window is small, mostly short-side, and already close to
fully captured by the frozen winner. Further architecture iteration on the SAME features/window
has strongly diminishing expected value; the more promising directions are new information
sources (different features/assets/frequencies) or genuinely more training data across regimes —
both data-acquisition problems, not model-architecture problems.

The frozen Omega6 v2 winner (`fin_p3_qt0.58_tp15.0_sl5.0_cd12`: val cost1 +21.96%/cost3 +10.68%,
one-shot OOS +16.81%/+7.16%) remains the only configuration in this project's history to pass
both validation and a one-shot OOS check.

Artifacts: `tmp/causal_regen_20260516/sigma1_seq_barrier_20260704/`,
`tmp/causal_regen_20260516/sigma1_decision_tape_20260704/`,
`tmp/causal_regen_20260516/sigma1_gates_20260704/sigma1_gate_ranking.csv`.
