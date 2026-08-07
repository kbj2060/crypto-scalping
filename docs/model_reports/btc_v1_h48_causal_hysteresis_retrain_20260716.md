# BTC v1 H48 causal-hysteresis retrain — 2026-07-16

## Outcome

The BTC v1 downstream stack was retrained with the stabilized H48 quality
target. The candidate is rejected. It lost money in validation and fresh 2026
Q1 OOS after the complete ATR/exit/risk execution contract was applied.

No live artifact or runtime configuration was changed.

## Label contract

- Direction and exit segmentation: existing BTC confirmed-pivot zigzag.
- Quality target source: BTC h48 conservative triple-barrier action.
- Stabilization: causal hysteresis.
  - A candidate state must persist for 6 consecutive 5m bars (30 minutes).
  - The emitted state must dwell for at least 12 bars (60 minutes) before a
    new state can be accepted.
  - The filter uses only the current and previous raw H48 targets and adds no
    future rows beyond the underlying offline H48 label.
- 2025 transitions: 15,019 raw to 2,531 stabilized (-83.15%).
- 2026 transitions: 8,100 raw to 1,458 stabilized (-82.00%).

## Retraining contract

- Frozen input regime artifact: BTC-retrained 2024 HMM/decoder from the
  2026-07-15 full-stack run. It does not depend on the H48 target.
- Retrained models: bull, bear, and chop 3-head TabM experts plus the HGB risk
  sidecar.
- Direction/quality fit rows: complete 78,624-row pre-validation split.
- Exit rows: complete generated zigzag exit path; no 12,000-row cap.
- Epochs: 4, seed 260620.
- Parent threshold search: q0.40 through q0.70.
- Selected threshold: q0.55, highest validation PnL; OOS excluded from
  selection.
- Sidecar selection: validation-only log-risk objective.

The inherited BTC v1 validation boundary is 2025-10-01, not the project
default 2025-09-01. This is explicitly a research candidate.

## Results

| Evaluation | PnL | MDD | Trades | Status |
|---|---:|---:|---:|---|
| Parent q055 validation | +14.84% | -5.31% | 280 | validation-selected |
| Parent q055 extended OOS | -18.06% | -18.53% | 593 | fail |
| Full execution baseline validation | -10.27% | -17.08% | 271 | fail |
| Risk sidecar validation full replay | -13.83% | -22.75% | 289 | fail |
| Full execution baseline Q1 fresh-forward | -11.62% | -16.92% | 302 | fail |
| Risk sidecar Q1 fresh-forward full replay | -9.46% | -19.89% | 295 | fail |
| Risk sidecar extended OOS full replay | -13.35% | -19.89% | 575 | fail |

The sidecar had no eligible mapping under its validation MDD guard and fell
back to `risk_000`; `full_replay_selection_applied=false`. This alone prevents
promotion regardless of OOS PnL.

For context, the current BTC v1 checkpoint reports Q1 +6.21% with MDD -16.46%.
That is not a pure label ablation because the current live parent used the
historical 30k direction/quality and 12k exit caps, whereas this experiment
used the complete train and exit sets. The stabilized candidate nevertheless
fails on its own absolute validation and OOS criteria.

## Fresh-forward evidence

- `fresh_forward_bar_by_bar=true`
- `trade_ledgers_used_as_input=false`
- `saved_parent_exit_timestamps_used=false`
- `future_rows_used_for_entry=false`
- Q1: 2026-01-01 00:00 through 2026-03-31 23:55, 25,920 bars.
- Exact q055 parent predictions matched all runtime-frame timestamps.
- Saved ledgers are evaluation outputs only.

## Artifacts

- Label builder: `scripts/denoise_btc_h48_labels_causal_hysteresis_20260715.py`
- Stabilized labels: `tmp/causal_regen_20260516/btc_h48_conservative_causal_hysteresis_c6_d12_20260715`
- Parent: `tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_20260708_h48stable_c6d12_fullstack_fulltrain_fullexit_20260716`
- Risk sidecar: `tmp/causal_regen_20260516/btc_omega4_2_trade_risk_sidecar_20260708_h48stable_c6d12_fullstack_fulltrain_fullexit_q055_20260716`
- Q1 fresh-forward: `tmp/causal_regen_20260516/btc_omega4_2_trade_risk_sidecar_20260708_h48stable_c6d12_fullstack_fulltrain_fullexit_q055_q1fresh_20260716`

## Decision

`promotion_pass=false`. Keep the current BTC v1 live checkpoint unchanged.
