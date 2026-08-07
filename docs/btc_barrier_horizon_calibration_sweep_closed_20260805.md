# BTC Barrier/Horizon Calibration Sweep — CLOSED 2026-08-05

## What was tried

Per user request, after magnitude-regression and TP-first-classification label
paradigms both closed on the richest available feature set (unified raw panel:
causalfix_final + Regime3 wide24 + DVOL + on-chain — see
`docs/btc_deepfeat_jepa_unified_panel_closed_20260804.md`,
`docs/btc_3way_tpfirst_label_closed_20260804.md`), the remaining untried lever was
the triple-barrier *calibration* itself (horizon length, TP/SL multiplier) rather
than the label paradigm or feature set.

`scripts/eval_btc_barrier_horizon_calibration_sweep_20260805.py`: 6 horizons (24,
48, 96, 192, 384, 576 bars = 2h/4h/8h/16h/32h/48h) x 5 TP/SL multiplier pairs
((1.0,0.6), (1.2,0.8), (1.5,1.0), (2.0,1.2), (2.5,1.5)) = 30 configs, dense-nogate
quality regression (the project's standard label mechanics, same
`_reason_and_return` barrier-touch function), same VAL/OOS split
(2025-09-01 / 2026-01-01-2026-04-01), same cost model, same `n_trades>=15` pass bar
as every prior stage in this line.

## Result: CLOSED, 0/148 threshold configs pass

No (horizon, tp_mult, sl_mult, threshold) combination is VAL-and-OOS-both-positive
with n>=15 trades each side.

The closest-looking results (e.g. `h48_tp2.0_sl1.2` at thresh=0.002: OOS
mean_net=+0.293%, n=16; `h576_tp2.0_sl1.2` at thresh=0.002: OOS mean_net=+0.245%,
n=26) do **not** hold up on inspection: the matching VAL result at the same
threshold is clearly negative (-0.68% and -0.34% respectively). This is the
signature of small-sample OOS noise, not a real edge -- a genuine calibration
improvement would show consistent direction across both splits, not an OOS-only
blip contradicted by VAL.

The TP:SL ratio 2.0:1.2 (~1.67:1) was consistently the least-bad across horizons,
mildly suggesting a real-vs-noise distinction *if* this were being used to pick
among otherwise-passing candidates -- but with zero configs actually passing, this
is not signal to act on, only a note for future calibration priors if a genuine
signal source is ever found.

## Verdict

This closes the last remaining lever identified in this session's arc. In sequence:

1. New data sources (Rho1 panel, DVOL, on-chain) — individually closed 2026-08-04.
2. Union of all raw sources through the existing pipeline — closed (0/7).
3. Self-supervised deep-feature encoder (JEPA) on the union — closed (0/9), learned
   embeddings ranked *weaker* than the raw features they were derived from.
4. TP-first / 3-way meta-label classification on the union — closed (0/24),
   reconfirming BTC v2's 0/12,544 meta-labeling search on a different feature set,
   this time with zero monotonic ranking signal at all.
5. Barrier/horizon calibration sweep on the union, standard regression label —
   closed (0/148).

Feature set, deep representation, label paradigm, and barrier calibration have all
now been varied independently (and the union of new sources jointly) against the
current best BTC feature set, with a consistent negative result throughout. This is
the strongest evidence in this project's history for a genuine ceiling on
extractable BTC 5m edge at this feature richness and evaluation window, rather than
a fixable methodology gap -- consistent with the standing "labeling-paradigm
diagnostic" conclusion in `docs/btc_new_architecture_session_summary_20260804.md`.

## Artifacts

- `scripts/eval_btc_barrier_horizon_calibration_sweep_20260805.py`
- `tmp/btc_barrier_horizon_calibration_sweep_20260805.csv`
