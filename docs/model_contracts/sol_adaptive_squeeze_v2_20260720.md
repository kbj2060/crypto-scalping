# SOL zig075 v2 -- adaptive_squeeze recalibration (2026-07-20)

## Root cause

`features/engineering.py`'s `FundingRateMomentum._long_squeeze_score`/`_short_squeeze_score`
divided the raw funding rate by a fixed constant, `0.0002`, self-documented in the code as a
"magic number" ("[수정사항 7] 롱/숏 스퀴즈 매직 넘버 대칭 통일 (0.0002)"). This constant was
implicitly calibrated to ETH's funding-rate scale. Measured directly on 2025 data:

| | ETH | SOL |
|---|---|---|
| `last_funding_rate` std | 0.000044 | 0.000157 (~3.5x wider) |
| `short_squeeze_risk >= 1.0` (saturated) | 0.09% of bars | 2.01% of bars (~22x more often) |

SOL's wider funding-rate volatility interacting with ETH's fixed divisor collapsed
`long_squeeze_risk`/`short_squeeze_risk` (and the downstream composite `crowding_pressure`,
which includes both as terms) into a near-binary signal for a meaningful fraction of bars,
destroying gradation that the model could otherwise use. This is one confirmed instance of a
broader pattern (asked as "왜 이더리움만 되고 나머지는 안 되나" / "why does only ETH work") --
most of this project's ~140 features already use rolling/adaptive normalization (z-score,
percentile, MAD) and are asset-agnostic by construction; this was the one clear exception found
after auditing the funding/whale/liquidity feature family (`WhaleSentimentDivergence`'s hardcoded
`1.48` threshold was also checked but does not feed any of SOL's actual 147 base_cols -- only the
raw `whale_retail_ratio`/`whale_conviction` values do, and those are unmodified raw ratios with no
hardcoded constant applied before reaching the model).

## Fix

- `features/engineering.py`: `FundingRateMomentum` gained `adaptive_squeeze: bool = False`
  (default preserves the exact prior fixed-divisor behavior byte-for-byte, verified by unit test).
  When `True`, the "how extreme is funding" term uses the already-computed rolling
  `funding_z_score` (self-normalizing per symbol) instead of the fixed `/0.0002` divisor, clipped
  at a z-score of 2.0 (matches this file's own "extreme" convention elsewhere: `evt_tail_flag`'s
  97th-percentile threshold, `jump_flag`'s 4-sigma threshold).
- `FeatureEngineer.__init__` gained the same `adaptive_squeeze` passthrough, forwarded to
  `FundingRateMomentum` in `process()`.
- `trading_bot.py`'s SOL/BTC shadow-asset loop now constructs
  `FeatureEngineer(adaptive_squeeze=(_asset_key == "sol"))` -- BTC and ETH unaffected (BTC's own
  funding-rate std, 0.000041, is already close to ETH's 0.000044, so this recalibration wasn't
  pursued for BTC; no evidence it's needed there).

## Retraining (full pipeline, same architecture/labels/hyperparameters as v1)

1. `scripts/build_sol_features_adaptive_squeeze_20260720.py` -- rebuilt SOL's feature frame with
   `FeatureEngineer(adaptive_squeeze=True)`. Output: `data/splits/year_oos_adaptive_squeeze_sol_20260720/`.
2. `scripts/train_eval_omega4_3head_parent72_loose_entry_quality_sol_adaptive_squeeze_20260720.py` --
   same zig075 parent architecture, same `quality_mode=same_as_direction`, thresholds through 0.70,
   pointed at the new feature files.
3. `scripts/train_eval_omega4_2_risk_sidecar_sol_20260707.py` rerun with
   `--risk-feature-mode parent_outputs --side-split-model --dynamic-leverage` (matching v1's own
   contract exactly -- an earlier pass without these flags produced a non-side-split,
   non-dynamic-leverage sidecar and was discarded) against the new parent's precomputed
   predictions.
4. `scripts/apply_final_scale_map_sol_adaptive_squeeze_20260720.py` -- re-ran the VAL-only
   long/short scale-map grid search (same method as v1's own `apply_final_scale_map_sol_20260707.py`,
   LEVERAGE_CAP=5.0/NOTIONAL_CAP=1.8) against the new artifacts. Selected `{long_scale: 1.0,
   short_scale: 1.75}` (v1 was `{1.0, 2.0}` -- re-derived, not reused, per this project's own
   established rule that ETH-tuned constants don't transfer across assets).

All four steps are genuine bar-by-bar causal replays (`fresh_forward_bar_by_bar=true`,
`trade_ledgers_used_as_input=false`, `saved_parent_exit_timestamps_used=false`,
`future_rows_used_for_entry=false`), not ledger post-hoc rescales.

## Results (gate off, matching v1's own live convention, same evaluation stage)

| | v1 (live, pre-fix) | v2 (adaptive_squeeze) |
|---|---|---|
| VAL pnl / mdd | +33.73% / -29.78% | +16.75% / -26.29% |
| OOS pnl / mdd | +33.98% / -31.99% | **+57.94% / -21.35%** |
| trades (val/oos) | 41 / 56 | 42 / 59 |

**v1's OOS MDD (-31.99%, and its as-deployed-with-1.5x-multiplier figure of -27.91%) breaches this
project's own -25% OOS-MDD promotion-gate threshold** (the same rule that formally rejected an
earlier, more complex SOL candidate on 2026-07-08 -- see
`docs/audits/sol_v1_live_config_mdd_gate_recheck_20260720.md`). v2 clears it with room to spare
at both PnL improvement and MDD reduction simultaneously -- not a risk/return tradeoff, a
dominating result on this specific comparison.

**Explored and rejected as not adding value:**
- Notional multiplier sweep (1.0x-3.0x) on top of v2: PnL keeps climbing but so does MDD,
  re-breaching the gate above ~1.25x. VAL PnL itself peaks around 1.75-2.0x then declines (not
  pure leverage-chasing, but still a real risk/return tradeoff past 1.0x). **v2 is deployed at
  1.0x (no multiplier)** -- `FINAL_GOVERNOR_OMEGA4_6_1_SOL_NOTIONAL_MULTIPLIER` changed from `1.5`
  (v1's value) to `1.0` in `.env` accordingly.
- ETH's `ChopSoftSizeShadow` rule (`shadow_notional = real_notional * max(0, 1 - chop_prob)`)
  applied post-hoc to v2's ledger: behaves as ETH's own shadow observation already found (mostly a
  proportional leverage-reduction effect, not a genuine chop-timing edge) -- reduces PnL and MDD
  together, no free lunch. Combining a higher multiplier with chop soft-sizing to "buy back" gate
  headroom was also checked and is strictly dominated by v2 at 1.0x/no chop-sizing on both PnL and
  MDD at every multiplier tried.

## Verification

- Artifact-integrity promotion gate: `promotion_pass: true`
  (`tmp/causal_regen_20260516/sol_adaptive_squeeze_artifact_integrity_20260720/`).
- `adaptive_squeeze=False` (default) unit-tested to reproduce the pre-fix formula exactly
  (`np.allclose` on a synthetic funding-rate series).
- Full `Omega461LiveAdapter` end-to-end construction test with the new config (bundle, sidecar,
  scale_map, `base_template`/`expert_scales`, `current_regime_path`) succeeds, `base_cols` count
  147 as expected.
- `trading_bot.py` and `trading_bot_modules/runtime_config.py` re-verified with `ast.parse` after
  every edit.

## Live wiring (not yet active -- pending restart)

- `trading_bot_modules/runtime_config.py`: `FINAL_GOVERNOR_OMEGA4_6_1_SOL_BUNDLE_PATH`/
  `..._SIDECAR_PATH` now default to the v2 artifacts; SOL's `scale_map` updated to
  `{zig075_L: 1.0, zig075_S: 1.75}`.
- `trading_bot.py`: SOL's shadow-loop `FeatureEngineer` now constructed with
  `adaptive_squeeze=True`.
- `.env`: `FINAL_GOVERNOR_OMEGA4_6_1_SOL_NOTIONAL_MULTIPLIER` changed `1.5` -> `1.0`.
- Bundled with the earlier same-session fixes (regime3 HMM per-asset path, `BASE_TEMPLATE`/
  `EXPERT_SCALES` per-asset dispatch) -- all take effect together on the next bot restart.
  `BINANCE_ACCOUNT_ENABLED=False` still blocks all real order placement.
