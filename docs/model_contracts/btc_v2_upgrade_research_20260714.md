# BTC v2 Upgrade Research - 2026-07-14

Status: `research_negative_result_not_adopted`

Live status: unchanged. BTC v1 remains the live-wired model from
`docs/model_contracts/live_model_v1_checkpoint_20260714.md`.

## Objective

Upgrade the BTC model independently of ETH by fixing the v1 analysis findings:

- align the supervised target with the executed holding/exit contract;
- remove non-stationary raw levels and `ou_halflife` from new candidates;
- separate direction from trade quality;
- reduce overlapping five-minute labels to causal entry events;
- evaluate every entry and exit from market bars, never from a saved trade ledger.

Research references used as design input:

- [TabM](https://arxiv.org/abs/2410.24210): retain an ensemble rather than trust one tabular fit.
- [PatchTST](https://arxiv.org/abs/2211.14730): aggregate the noisy five-minute stream into a
  longer causal patch for direction.
- [FOIL](https://arxiv.org/abs/2406.09130): test explicit temporal-environment/regime agreement
  instead of assuming train and OOS are identically distributed.

## Experiments

### 1. Five-minute overlapping 7-day labels

Artifact: `tmp/causal_regen_20260516/btc_v2_horizon_selective_lgbm_20260714/report.json`

Three temporal LightGBM members predicted side-specific execution wins. The model compared the
two calibrated side probabilities and used the larger barrier EV.

| split | PnL | MDD | trades | result |
|---|---:|---:|---:|---|
| validation 2025 Q4 | -1.99% | -30.99% | 22 | failed |
| OOS 2026-01-01..07-12 | -31.59% | -44.02% | 55 | failed |

Root cause: calibration-period base rates made the independent long probability larger almost
everywhere. OOS produced 54 long trades and one short trade. Comparing probabilities from two
different binary tasks was not a valid direction head.

### 2. Separate one-day direction and four-day quality

Artifact: `tmp/causal_regen_20260516/btc_v2_direction_quality_lgbm_20260714/report.json`

Direction became the sign of the trailing one-day return; the side-specific model was used only
as a quality filter. Label and execution time exit were both four days.

| split | PnL | MDD | trades | result |
|---|---:|---:|---:|---|
| validation 2025 Q4 | -7.35% | -18.62% | 18 | failed |
| OOS 2026-01-01..07-12 | -18.78% | -33.96% | 47 | failed |

This removed the invalid probability comparison but did not remove the overlap problem: every
five-minute row inside one long trend still became another highly correlated training example.

### 3. Independent hourly event parent plus BTC regime gate

Implementation:

- `scripts/train_eval_btc_v2_regime_trendscan_20260714.py`
- `test/test_btc_v2_regime_trendscan_20260714.py`

Artifact: `tmp/causal_regen_20260516/btc_v2_regime_trendscan_hgb_20260714/report.json`

Contract:

- Parent: five-seed HGB trend-scan classifier, 28 BTC-only OHLCV features, trained on
  2024-01-01..2025-06-30.
- No ETH reference, funding, OI, saved ledger, or future entry row.
- The left-labelled hourly parent row for `[t,t+1h)` becomes available at `t+1h`.
- Entry is considered only once, when a new hourly parent signal becomes available.
- The 2024-fit BTC HMM must agree with the parent side at that event.
- Entry and all stop/trailing/time-exit decisions are replayed on five-minute bars.
- `margin_fraction=0.30`, `leverage=2`, `notional=0.60`.
- ATR stop/trailing lines are price moves. Leverage is not multiplied into them again.

Validation selected `quality_threshold=0.55`, `regime_threshold=0.50`. Gate-off controls were
included and did not pass both validation halves.

| split | PnL | MDD | trades | WR |
|---|---:|---:|---:|---:|
| validation 2025-07..09 | +4.80% | -6.04% | 23 | 43.5% |
| validation 2025-10..12 | +6.87% | -7.47% | 23 | 34.8% |
| validation 2025 H2 continuous | +23.01% | -7.47% | 44 | 40.9% |
| OOS 2026 Q1 | -15.78% | -17.56% | 31 | 25.8% |
| OOS 2026-01-01..07-12 | -7.77% | -21.78% | 56 | 28.6% |

The candidate passed validation but failed OOS. OOS long trades compounded to -9.74%; shorts
compounded to +2.18%. January through April were all negative, while June recovered +12.93%.
The HMM agreement gate therefore did not provide invariant direction quality in 2026.

## Decision

Do not promote any BTC v2 candidate from this run. In particular:

- do not wire `btc_v2_regime_trendscan_hgb_20260714` into the live bot;
- do not run the Omega artifact promotion audit as if this were a live candidate;
- do not tune thresholds against the observed 2026 OOS loss;
- keep BTC v1 active until a new, pre-registered validation design is available.

All final evaluations declare:

- `fresh_forward_bar_by_bar=true`
- `trade_ledgers_used_as_input=false`
- `saved_parent_exit_timestamps_used=false`
- `future_rows_used_for_entry=false`

## Next Valid Upgrade Direction

The next experiment should not be another threshold sweep on this OOS window. A valid next step
is a new causal training design with non-overlapping event labels and a separately reserved
future test period. Candidate directions are online/monthly refitting with a full label embargo,
or direct utility ranking across sparse change-point events. Neither is promoted or implied by
the current negative result.
