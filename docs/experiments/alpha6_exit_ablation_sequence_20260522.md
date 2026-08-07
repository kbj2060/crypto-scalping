# Alpha6 Exit Ablation Sequence - 2026-05-22

Baseline active candidate:
- Contract: `docs/model_contracts/alpha6_entry_quality_exit_5bucket_main_20260522_contract.md`
- Artifact: `data/ensemble/supervised/alpha6_entry_quality_exit_5bucket_main_20260522/current_tail111_summary.json`
- Entry threshold: `0.0034163351358086967`
- Exit threshold: scalar `0.35`
- Cost1/2/3: `+15.30% / +14.24% / +12.04%`
- MDD: about `-4.81%`
- Trades: `61 / 63 / 63`

## 1. Bucket-Aware Exit Thresholds

Out dir: `tmp/causal_regen_20260516/alpha6_ablation1_bucket_thresholds_20260522/`

Tested:
- scalar baseline: `0.35`
- bucket config A: `0.28,0.30,0.35,0.42,0.48`
- bucket config B: `0.25,0.30,0.35,0.40,0.45`
- bucket config C: `0.30,0.32,0.35,0.40,0.45`

Result:
- Best remained scalar `0.35`.
- Best bucket config underperformed materially.

Decision:
- Reject bucket-specific thresholds for now.
- Keep scalar `0.35`.

## 2. Regime Drift State

Out dir: `tmp/causal_regen_20260516/alpha6_ablation2_regime_drift_20260522/`

Added state fields:
- `risk_off_delta`
- `whipsaw_delta`
- `regime_confidence_delta`
- `risk_mode_flipped`

Result:
- Cost1/2/3: `+13.17% / +11.14% / +9.21%`
- MDD: about `-4.81%`
- State dim: `27 -> 31`

Decision:
- Reject as standalone addition.
- It preserved MDD but reduced PnL and Cost3 durability.

## 3. Capture Ratio State

Out dir: `tmp/causal_regen_20260516/alpha6_ablation3_capture_ratio_20260522/`

Added state fields:
- `capture_ratio`
- `mfe_expected_ratio`

Expected return table was estimated from train split by target bucket:
- `0`: `0.007130321397523964`
- `1`: `0.010549346576137795`
- `2`: `0.016267385010888136`
- `3`: `0.025955167758988738`
- `4`: `0.04472592851718826`

Result:
- Cost1/2/3: `+10.71% / +9.34% / +7.88%`
- MDD: about `-4.81%`
- State dim: `27 -> 29`

Decision:
- Reject as standalone addition.
- It reduced trade count and lowered PnL across all cost tiers.

## 4. Adaptive Exit Sampling

Out dir: `tmp/causal_regen_20260516/alpha6_ablation4_adaptive_sampling_20260522/`

Sampling changed from strict `step=8` to event-aware dense sampling for:
- `giveback_ratio > 0.5`
- `hold_frac > 0.85`
- `MAE/ATR > 0.8`

Exit training samples:
- Baseline: `5101`, close rate `0.2803`
- Adaptive: `22599`, close rate `0.2113`

Initial threshold `0.35`:
- Cost1/2/3: `+15.69% / +14.63% / +12.83%`
- MDD: about `-7.49%`

Threshold resweep:
- `0.40`: Cost1/2/3 `+16.52% / +16.08% / +15.25%`, MDD about `-7.49%`
- `0.45`: Cost1/2/3 `+18.25% / +17.86% / +4.46%`, MDD about `-7.54%`

Decision:
- Do not replace active main because MDD worsened from about `-4.81%` to about `-7.49%`.
- Keep as high-PnL/high-MDD challenger. If future objective prioritizes PnL over drawdown, resweep around threshold `0.40`.

## Final Decision

Keep current Alpha6 active main unchanged:
- `alpha6_entry_quality_exit_5bucket_main_20260522`
- scalar exit threshold `0.35`
- no regime drift state
- no capture ratio state
- no adaptive exit sampling

Reason:
- Baseline remains the best balanced candidate by Cost3 durability and MDD.
- Adaptive sampling has higher PnL potential, but worsens MDD by roughly `+2.68%p`.

Recommended next test:
- Do not stack the rejected state additions immediately.
- If more testing is desired, investigate adaptive sampling with an MDD guard or max-hold/tail guard, because the alpha gain appears real but currently arrives with worse drawdown.

## 5. Adaptive Sampling + Hard Tail Guards

Script change:
- Added optional backtest-only hard guards to `scripts/alpha6_catboost_entry_quality_exit_policy_20260522.py`.
- Guards tested:
  - `guard_max_target_hold`: close when holding period reaches target horizon.
  - `guard_adverse_atr`: close when adverse return exceeds N ATR.
  - `guard_giveback_ratio`: close when MFE giveback exceeds ratio after minimum MFE.

All tests used:
- Adaptive exit sampling enabled.
- Exit threshold scalar `0.40`.
- CPU only, to avoid disturbing the running DSAC process.

Results:

| Test | Out dir | Cost1 | Cost2 | Cost3 | MDD note | Decision |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Combined target-hold + adverse ATR 1.8 + giveback 0.70 | `tmp/causal_regen_20260516/alpha6_ablation5_adaptive_tail_guard_20260522/` | `-0.45%` | `-1.07%` | `-1.64%` | MDD improved only because almost all edge was cut | Reject |
| Target-hold only | `tmp/causal_regen_20260516/alpha6_ablation5_adaptive_guard_targethold_20260522/` | `-6.22%` | `-21.12%` | `-32.42%` | MDD worsened heavily | Reject |
| Giveback 0.70 only | `tmp/causal_regen_20260516/alpha6_ablation5_adaptive_guard_giveback_20260522/` | `+6.24%` | `+2.86%` | `+4.44%` | MDD about `-7.79%` at Cost3 | Reject |
| Adverse ATR 4.0 only | `tmp/causal_regen_20260516/alpha6_ablation5_adaptive_guard_adverse4_20260522/` | `+0.33%` | `-5.34%` | `-17.73%` | MDD worsened heavily | Reject |

Interpretation:
- Current Alpha6 EQE exit head already learns profitable late exits.
- Rule-based hard guards fire too often on positions that later recover or continue.
- `target_horizon` should remain a model feature, not a hard liquidation deadline.
- `giveback` and adverse ATR are useful candidate state features/training labels, but hard runtime rules are not suitable in this tested form.

Updated final decision:
- Keep active Alpha6 main unchanged.
- Do not promote adaptive sampling until the MDD problem is solved inside the learned exit objective, not by blunt hard exits.

## 6. Entry Adaptive Sampling

Script change:
- Added optional entry candidate sampling to `scripts/alpha6_catboost_entry_quality_exit_policy_20260522.py`.
- Base `stride_bars=3` candidates are preserved.
- Extra entry candidates are selected using only current/past feature conditions:
  - regime transition / transition risk
  - volatility expansion
  - trend or breakout event
  - flow extreme
  - funding-related event/window

Command profile:
- Active Alpha6 5-bucket EQE hyperparameters.
- CPU only.
- Exit structure unchanged from active 5-bucket EQE.
- `--adaptive-entry-sampling`
- `--entry-event-quantile 0.85`
- `--entry-adaptive-max-extra 12000`

Out dir:
- `tmp/causal_regen_20260516/alpha6_ablation6_adaptive_entry_sampling_20260522/`

Entry label candidates:
- Active baseline: `26059`
- Adaptive entry: `34034`
- Added after union: `7975`

Adaptive event counts before cap/union:
- `regime_transition`: `21843`
- `volatility_expansion`: `26561`
- `trend_breakout`: `68909`
- `flow_extreme`: `11727`
- `funding_event`: `31894`

Best result:
- Entry threshold: `0.0037832971033215043`
- Exit threshold: `0.70`
- Cost1/2/3: `+19.94% / +19.11% / +18.23%`
- MDD: about `-4.81%`
- Trades: `27`
- Win rate: Cost1 `85.19%`, Cost3 `81.48%`
- L/S: `5 / 22`

Comparison to active main:
- Active Cost1/2/3: `+15.30% / +14.24% / +12.04%`
- Active MDD: about `-4.81%`
- Active trades: `61-63`
- Active L/S at Cost3: `22 / 41`

Decision:
- Promote to high-priority challenger.
- Do not overwrite active main yet.
- Reason: PnL and Cost3 durability improved materially without MDD degradation, but trade count fell to `27` and short concentration increased. This should pass a walk-forward/stability test before becoming the main Alpha6 runtime model.

Next validation:
- Re-run the same adaptive entry sampling over multiple temporal folds.
- Resweep around exit threshold `0.65-0.80` without retraining if a cached-evaluation path is added.
- Inspect the 27-trade ledger for concentration, single-trade dependency, and bear-regime overfit.

## 7. Entry Adaptive Sampling on Adaptive Exit Sampling

Question:
- Test the same entry adaptive sampling on top of the previous adaptive exit sampling challenger.

Command profile:
- `--adaptive-entry-sampling`
- `--entry-event-quantile 0.85`
- `--entry-adaptive-max-extra 12000`
- `--adaptive-exit-sampling`
- Initial run fixed `--exit-threshold-grid 0.40`, matching the previous adaptive-exit high-PnL threshold.

Out dir:
- `tmp/causal_regen_20260516/alpha6_ablation7_adaptive_entry_on_adaptive_exit_20260522/`

Training sample counts:
- Entry candidates: `34034` (`26059` base + `7975` added after union)
- Exit samples: `23882`
- Exit close rate: `0.2137`

Initial fixed `exit_threshold=0.40` result:
- Cost1/2/3: `+4.74% / +19.44% / +18.53%`
- MDD: Cost1 `-7.41%`, Cost2/3 about `-4.81%`
- Trades: Cost1 `17`, Cost2/3 `30`

Because cost tiers diverged materially, a cached threshold resweep was run without retraining.

Cached resweep artifact:
- `tmp/causal_regen_20260516/alpha6_ablation7_adaptive_entry_on_adaptive_exit_20260522/current_tail111_cached_exit_threshold_resweep.csv`
- `tmp/causal_regen_20260516/alpha6_ablation7_adaptive_entry_on_adaptive_exit_20260522/current_tail111_cached_exit_threshold_resweep_summary.json`

Notable resweep candidates:
- `entry_threshold=0.0037833`, `exit_threshold=0.45`
  - Cost1/2/3: `+20.98% / +20.07% / +15.56%`
  - MDD: about `-4.81%`
  - Trades: Cost1/2 `29`, Cost3 `23`
  - L/S: Cost1 `7 / 22`
- `entry_threshold=0.0029609`, `exit_threshold=0.55`
  - Cost1/2/3: `+19.66% / +18.67% / +18.40%`
  - MDD: about `-4.85%`
  - Trades: `19`
  - L/S: `4 / 15`
- Score-selected cached row was `entry_threshold=0.0037833`, `exit_threshold=0.40`, but Cost1 was weak:
  - Cost1/2/3: `+4.49% / +19.44% / +18.53%`
  - Cost1 MDD `-7.41%`, Cost2/3 MDD about `-4.81%`

Decision:
- Do not promote this combination yet.
- It has attractive Cost2/3 numbers, but cost-tier path divergence is too large.
- The cleaner current challenger remains entry-adaptive-on-active-exit:
  - Cost1/2/3 `+19.94% / +19.11% / +18.23%`
  - MDD about `-4.81%`
  - Trades `27`

Interpretation:
- Entry adaptive sampling is the useful change.
- Stacking adaptive exit sampling on top adds instability rather than a clean improvement.
- If revisited, evaluate `exit_threshold=0.45` and `0.55` with a ledger concentration check, not just aggregate PnL.
