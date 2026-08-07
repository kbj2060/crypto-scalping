# Omega5 Live Promotion Block

Date: 2026-07-02 KST

## Verdict

`OMEGA5_LIVE_PROMOTION_BLOCKED`

Omega5 must not be promoted or kept as the active live trading model.

## Blocking Finding

A side-thread audit found that the promoted Omega5 path depended on validation/test trade ledgers during model-selection or policy-construction work. That invalidates the live-promotion claim even if the replay PnL is high.

## Runtime Action

- `FINAL_GOVERNOR_OMEGA5_ENABLE` now defaults to `false`.
- `FINAL_GOVERNOR_OMEGA5_SOURCE_PARENT_ENABLE` now defaults to `false`.
- Setting `FINAL_GOVERNOR_OMEGA5_ENABLE=1` fails fast at import time.
- Direct `Omega5LiveAdapter` construction fails fast.

## Re-Promotion Requirements

Omega5 can only be reconsidered after a clean rebuild that proves:

- no validation/test/OOS ledger rows are used as model inputs, selection labels, sizing references, or feature sources;
- selection uses training/calibration data only;
- validation and OOS are readout-only;
- runtime-native replay uses only features available at the decision timestamp;
- a fresh holdout or walk-forward period passes after the final selection rule is frozen.
