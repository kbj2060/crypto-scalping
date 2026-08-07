# BTC deep-feature encoders (CNN seq / CNN category / Transformer), zigzag soft-label teacher

Fresh standalone line, started 2026-08-06. Explicitly disregards the closed
`docs/btc_deepfeat_jepa_unified_panel_closed_20260804.md` (self-supervised JEPA encoder) and
`docs/btc_new_architecture_session_summary_20260804.md` / 2026-08-06 zigzag-architecture-arc
findings at the user's request -- this is a deliberate re-attempt, not a continuation.

## What this stage builds

1. **Feature categories** -- `scripts/build_btc_feature_categories_20260806.py` splits the
   `causalfix_final` 113-column 5m panel (`data/splits/year_oos/btc_features_causalfix_final_2024_2026.parquet`)
   into 10 named categories (ohlcv_raw, derivatives_funding, orderflow_microstructure, volatility,
   momentum_trend, cross_asset_eth, regime, structural_price_location, time_session,
   mtf_1h_sidecar). Validates the map against the live panel columns on every run (raises if out
   of sync). Manifest: `docs/model_contracts/btc_feature_categories_20260806.json`.

2. **Teacher label** -- reuses the existing zigzag risk-adjusted **soft** label built the same
   day (`scripts/build_btc_5m_zigzag_and_pivot_labels_20260806.py` ->
   `data/splits/year_oos/btc_5m_zigzag_labels_20260806.parquet`, columns `zigzag_soft_cash/long/short`).
   This was not built fresh for this line -- it's the same soft-labeled zigzag target from the
   2026-08-06 label-construction session, already row-aligned 1:1 with the causalfix_final panel
   by timestamp (verified). Label balance: LONG 50.6% / SHORT 44.8% / CASH 4.7% -- not degenerate.

3. **Dataset** -- `ensemble/deep_features/btc_deepfeat_dataset_20260806.py`. Causal window=48 bars
   (4h) per sample, train-only standardization, Fresh-Forward split (VAL 2025-09-01..2025-12-31,
   OOS 2026-01-01..2026-03-31). Windows are sliced lazily from a single standardized `(n_rows, F)`
   array rather than materialized densely (~5.5GB would be needed to hold all windows at once;
   the lazy approach uses ~120MB).

4. **Three encoder architectures**, all trained supervised against the soft label (soft
   cross-entropy against the 3-class probability target), in
   `ensemble/deep_features/btc_deepfeat_encoders_20260806.py`:
   - `cnn_seq` -- Conv1d stack over the time axis, raw 113-dim feature channels (TCN-style, no
     category structure).
   - `cnn_category` -- per-category linear projection per timestep, Conv1d over the *category*
     axis (cross-category mixing), then Conv1d over the time axis.
   - `transformer` -- standard `nn.TransformerEncoder` (2 layers, d_model=64, 4 heads), purely
     supervised end-to-end (no JEPA-style masked pretraining), pooled at the window's last bar.

   All three share the same interface (`forward(x) -> (logits[B,3], embedding[B,32])`), driven by
   one training script: `scripts/train_btc_deepfeat_encoders_20260806.py --arch {cnn_seq,cnn_category,transformer}`.

## Results, v1 (soft-label loss + hard top-1 agreement only -- no backtest run at this stage)

All three converged (best val loss) at epoch 1 of training and began overfitting immediately
after -- early-stopped at epoch 6 (patience=5) in every case.

| arch         | val soft-CE loss | val hard top-1 acc | OOS soft-CE loss | OOS hard top-1 acc |
|--------------|------------------:|--------------------:|-------------------:|---------------------:|
| cnn_seq      | 1.473             | 49.0%               | 1.360               | 50.1%                |
| cnn_category | 1.155             | 54.2%               | 1.294               | 49.8%                |
| transformer  | 1.048             | 60.5%                | 1.020               | 58.0%                |

Majority-class baseline (always predict LONG) = 50.6%. `cnn_seq` and `cnn_category` land at or
near that baseline on OOS; only `transformer` clears it meaningfully on both VAL and OOS.

## Root cause diagnosed + fixed: epoch-1-best was a data-redundancy artifact, not signal ceiling

Consecutive 5m windows at stride 1 overlap in 47/48 bars, so the nominal 175k-row train set was
almost entirely near-duplicate samples -- the model could "memorize" the train distribution
inside a single epoch, at which point every subsequent epoch just pushed it further from the
independent VAL distribution. Fixed via, in order of contribution:

1. **`--train-stride 4`** (`ensemble/deep_features/btc_deepfeat_dataset_20260806.py`) --
   subsamples train window-end rows every 4th row only (VAL/OOS stay dense/unchanged, since those
   must reflect the true bar-by-bar distribution). Cuts train set 175k -> 44k but removes almost
   all of the trivial redundancy.
2. Reduced encoder capacity + raised dropout across all three architectures
   (`ensemble/deep_features/btc_deepfeat_encoders_20260806.py`): cnn_seq channels
   128/128/64->64/64/32, cnn_category cat_hidden 32->24 + time_channels 64/64->32/32, dropout
   0.1->0.25-0.3 everywhere including a new head dropout (0.2) shared by all three.
3. Training-loop changes (`scripts/train_btc_deepfeat_encoders_20260806.py`): lr 1e-3->3e-4,
   weight_decay 1e-4->5e-4, added `ReduceLROnPlateau` (factor 0.5, patience 2) + gradient clipping
   (max_norm=1.0), early-stopping `min_delta=1e-4` (was accepting noise-level improvements),
   patience 5->8.

## Results, v2 (post-fix)

| arch         | best epoch | val soft-CE loss | val hard top-1 acc | OOS soft-CE loss | OOS hard top-1 acc |
|--------------|-----------:|------------------:|--------------------:|-------------------:|---------------------:|
| cnn_seq      | 1 (of 9)   | 1.013             | 54.9%               | 1.012               | 54.0%                |
| cnn_category | 1 (of 9)   | 1.005             | 56.6%               | 1.015               | 52.0%                |
| transformer  | **7 (of 15)** | **0.973**       | **63.8%**            | **0.968**            | **60.2%**             |

Every architecture improved on both VAL and OOS soft-CE loss and hard accuracy versus v1.
`transformer` is the only one whose best epoch moved off epoch 1 (best at epoch 7 of 15, plateauing
0.97-0.98 for ~8 epochs before drifting) -- the CNNs still peak at epoch 1, but now over a
non-degenerate loss curve (gradual multi-epoch drift afterward, not an instant collapse), and both
land meaningfully above the 50.6% baseline on OOS now (v1 cnn_category was at/below baseline on
OOS). Transformer remains the strongest architecture by a clear margin and its edge grew after the
fix (58.0%->60.2% OOS acc).

Outputs per architecture in `tmp/btc_deepfeat_encoders_20260806/{arch}/`:
`deepfeat_bundle.pt` (model + config + standardization stats), `metrics.json` (full per-epoch
history + final VAL/OOS), `deepfeat_embeddings_{train,val,oos}.parquet` (32-dim learned embedding
per bar, timestamp-keyed -- the actual "deep features" for downstream use).

## Raw-feature GBDT baseline check (2026-08-06, same day)

Per the closed JEPA line's precedent (deep embeddings ranked weaker than raw features when both
fed to a tree model), ran the same check here before committing to the transformer line.

`scripts/train_btc_zigzag_gbdt_baseline_20260806.py` -- `sklearn.HistGradientBoostingClassifier`
(LightGBM not installed in this env; HGB is the closest built-in histogram-GBDT) on raw
single-bar causalfix_final features (113 cols, no windowing, no deep encoder), same
rows/split/standardization as the transformer run (window=48, train_stride=4):

| model                                | val hard top-1 acc | OOS hard top-1 acc |
|---------------------------------------|--------------------:|----------------------:|
| GBDT, raw features only (113 cols)    | **65.5%**            | **63.4%**             |
| transformer encoder (standalone)      | 63.8%                | 60.2%                 |

Raw features + a plain tree model beat the standalone transformer on both VAL and OOS.

`scripts/train_btc_zigzag_gbdt_raw_plus_transformer_20260806.py` -- same GBDT config with the
transformer's 32-dim embedding concatenated onto the 113 raw cols (145 total):

| model                                          | val hard top-1 acc | OOS hard top-1 acc |
|-------------------------------------------------|--------------------:|----------------------:|
| GBDT, raw only (113 cols)                        | 65.5%                | 63.4%                 |
| GBDT, raw + transformer embedding (145 cols)     | 65.1%                | 62.4%                 |

Adding the transformer embedding did not help -- it made both VAL and OOS slightly *worse*
(-0.4pt VAL, -1.0pt OOS) than raw features alone. This reproduces the exact pattern from the
closed 2026-08-04 JEPA line: a deep-feature encoder trained on this same causalfix_final panel
adds no information a raw-feature tree model doesn't already have, and mixing it in adds noise
instead. The teacher-label change (self-supervised JEPA -> supervised zigzag soft label) improved
the encoder's own standalone accuracy but did not change this outcome.

**Conclusion: the deep-feature encoder line is not adding value over raw features + GBDT for
this label. Raw-feature GBDT (65.5%/63.4%) is the strongest classifier found so far on this
target and should be the baseline for any further work on the zigzag soft-label direction, not
the transformer embeddings.**

## Transformer hyperparameter sweep (2026-08-06, same day)

Per user's explicit call to keep pushing the transformer line despite the GBDT baseline result
above, ran `scripts/sweep_btc_deepfeat_transformer_20260806.py` -- 16 random configs over
`window in {24,48,96}`, `d_model in {32,48,64,96}`, `n_layers in {1,2,3}`,
`dropout in {0.15,...,0.35}` (n_heads fixed=4, embed_dim fixed=32), each an isolated subprocess
run of the training script, model selection strictly on VAL soft-CE loss (not OOS or accuracy).

**Selected best (by VAL soft-CE loss, the correct selection criterion):**
`window=48, d_model=48, n_layers=1, dropout=0.25` -> val 63.1% / OOS 60.7% hard top-1 acc. This is
close to the original default-hyperparameter run (val 63.8% / OOS 60.2%) -- the sweep did not find
a config that clearly beats the untuned default when selected honestly on VAL loss.

**Caveat worth flagging:** VAL soft-CE loss and hard top-1 accuracy do not rank configs the same
way here. The single best *accuracy* config in the sweep was `window=48, d_model=96, n_layers=3,
dropout=0.25` -> val 65.0% / OOS 62.6% -- close to the GBDT baseline (65.5%/63.4%) and clearly
better than every VAL-loss-ranked pick, but it was NOT the VAL-loss winner (ranked 6th of 16 by
loss). This mismatch is itself a signal that the soft-label loss (which includes calibration
across all 3 classes, including the rare CASH class) is optimizing something slightly different
from directional trade-call accuracy -- worth remembering if this line continues.

All 16 results land in a narrow band (val 63.0-65.0%, OOS 60.0-62.6%) regardless of hyperparameters
-- capacity/window/dropout choices move the needle by ~2pt at most, none closes the ~3pt (OOS) gap
to the raw-feature GBDT baseline. Full per-config results:
`tmp/btc_deepfeat_transformer_sweep_20260806/sweep_summary.json`.

## Soft-label loss/accuracy mismatch diagnosed and fixed (2026-08-06, same day)

Per user's call to fix the loss-vs-accuracy mismatch found by the sweep, first quantified it:
the base zigzag soft label is fully peaked (mean maxprob=1.0000, entropy=0) on CASH bars but only
moderately peaked on active LONG/SHORT bars (mean maxprob~0.79, entropy~0.49) -- and the soft
label's own argmax *disagrees* with the hard direction label on ~9% of active bars (low-quality/
near-boundary wave segments the risk-adjusted score correctly hedges away from). That gap is why
minimizing soft-CE loss doesn't reliably track hard-direction accuracy.

Added two loss-shaping knobs to `_prepare_target()` in
`scripts/train_btc_deepfeat_encoders_20260806.py` (base label file untouched, shared with other
work -- shaping applied only inside this training script's loss computation):
- `--label-sharpen T` (T<1.0): raises the soft label to power `1/T` and renormalizes, pulling
  active-bar targets closer to their own argmax before computing loss.
- `--cash-weight w` (w<1.0): down-weights the already-trivial (fully-peaked, easy-to-fit)
  CASH-labeled samples' contribution to the loss so gradient signal concentrates on the harder
  LONG/SHORT calibration.

Swept both, fixed on the sweep's best-accuracy architecture (`window=48, d_model=96, n_layers=3,
dropout=0.25`):

| label_sharpen | cash_weight | val hard acc | OOS hard acc |
|---:|---:|---:|---:|
| 1.0 (baseline) | 1.0 | 65.0% | 62.6% |
| 0.7 | 1.0 | 65.6% | 63.1% |
| 0.5 | 1.0 | 65.5% | 62.6% |
| 0.3 | 1.0 | 66.0% | 62.8% |
| 0.7 | 0.5 | 66.9% | 64.5% |
| 0.7 | 0.2 | 67.0% | 64.8% |
| 0.7 | 0.1 | 67.0% | 64.9% |
| 0.5 | 0.2 | 67.2% | 64.8% |

Gains plateau around `label_sharpen≈0.5-0.7, cash_weight≈0.1-0.2` -- further lowering cash_weight
past 0.2 stopped helping. **Selected final config: `label_sharpen=0.7, cash_weight=0.2`
(window=48, d_model=96, n_layers=3, dropout=0.25) -> val 67.0% / OOS 64.8% hard top-1 acc.**

## Final comparison

| model                                              | val hard top-1 acc | OOS hard top-1 acc |
|------------------------------------------------------|--------------------:|----------------------:|
| GBDT, raw features only (113 cols)                    | 65.5%                | 63.4%                 |
| transformer, untuned default                          | 63.8%                | 60.2%                 |
| transformer, hyperparameter-swept only                | 65.0%                | 62.6%                 |
| **transformer, hyperparameter-swept + label-shaped**  | **67.0%**            | **64.8%**              |

After sharpening the soft-label target and down-weighting the trivial CASH class, the transformer
line now clears the raw-feature GBDT baseline on both VAL (+1.5pt) and OOS (+1.4pt). Checkpoint +
embeddings for the winning config are in `tmp/btc_deepfeat_sharpen_sweep/cw_0.2/`.

## Trading backtest: two independent fixes tried, both failed to produce a profitable signal (2026-08-06, same day)

Built `scripts/backtest_btc_deepfeat_transformer_20260806.py` on the final tuned checkpoint,
reusing this repo's canonical futures simulator (`core/causal_futures_backtest.py`:
`fit_tail_thresholds` + `simulate_single_position`, VAL-selects entry-threshold quantiles with a
MIN_TRADES floor, OOS confirmed exactly once). Score = P(LONG)-P(SHORT); TP/SL volatility-adaptive
(TP=2.5x/SL=1.2x trailing 24h vol, matching this session's established Layer B convention);
margin_fraction=0.30, leverage=3, roundtrip_cost=10bps.

**Attempt 1 -- continuous per-bar entries:** every bar with score past threshold fired a new
decision. Catastrophic: VAL win rate 29.8-32.5% across all 4 threshold quantile pairs tested, MDD
-88% to -99.9%, best VAL config still -216% cumulative return; OOS confirm of the VAL winner:
-154% return, MDD -78.7%, win rate 28.7%.

**Attempt 2 -- fresh-entry gate** (`_fresh_entry_mask`): only fire on the first bar of a new
directional regime (score crosses INTO the entry zone), not every bar it stays there. Fewer trades
(1324 VAL / 897 OOS vs 2452/1512) but **win rate barely moved** (29.8-32.3% still), because the
TP/SL ratio's breakeven win rate is 32.4% -- the fresh-entry gate wasn't the actual problem.

**Attempt 3 -- quality-score filter:** added a second regression head to the encoder (predicts
`log1p(zigzag_path_calmar)`, see `--quality-head`/`--quality-loss-weight` in
`scripts/train_btc_deepfeat_encoders_20260806.py`; retrained checkpoint at
`tmp/btc_deepfeat_sharpen_sweep/cw_0.2_quality/`, val hard acc 67.1%/OOS 64.8% -- direction
accuracy unaffected by adding the head). `scripts/backtest_btc_deepfeat_transformer_quality_20260806.py`
grids a predicted-quality percentile floor (0/25/50/70/85th) on top of the same fresh-entry
direction gate. **Win rate stayed flat regardless of quality floor (27.4-31.6% VAL, 25.9-30.6%
OOS) and was WORSE at the highest quality percentile (85th: 27.4% VAL / 25.9% OOS) than at no
filter at all (0th: 31.6% VAL / a filter still active since it's the fresh-gate baseline).** The
model's own predicted trade quality has no measurable relationship with actual forward trade
outcome.

**Conclusion:** neither entry-timing (fresh vs continuous) nor a learned quality filter fixes the
win rate, which sits at or below the TP/SL ratio's ~32.4% breakeven line in every configuration
tested. This reproduces the same structural wall documented in
[[project-btc-5m-zigzag-architecture-session-arc-20260806]] (Layer B's direction ceiling itself,
not tuning -- 600-config sweep there also found zero profitable standalone config) and
[[project-btc-zigzag-dual-component-already-failed-20260802]]: bar-level "which zigzag wave am I
in" classification, even at 67% accuracy and even with quality-of-wave regression, does not
translate into "should I enter a trade right now" -- these appear to be genuinely different
prediction targets, and no amount of entry-timing or quality-filtering layered on top of the
former has closed the gap to the latter across two independent architecture generations now (this
transformer line and the earlier LightGBM-based Layer B line).

## Head architecture swap: TabM and tree-based, both frozen-embedding and end-to-end (2026-08-06, same day)

User asked to replace the plain `nn.Linear` direction/quality heads with a TabM-style model or a
tree-based model. Tried both integration modes:

**Frozen-embedding downstream (two-stage)** -- `scripts/train_btc_deepfeat_downstream_heads_20260806.py`:
freeze the tuned encoder (`tmp/btc_deepfeat_sharpen_sweep/cw_0.2_quality/`), extract its 32-dim
embedding, train a TabM-style ensemble head (`ensemble/deep_features/btc_deepfeat_tabm_head_20260806.py`
-- BatchEnsemble-style per-expert input scale/bias + shared MLP trunk, matching this repo's
established `ThreeHeadTabM` convention) and separately a `HistGradientBoostingClassifier`/
`Regressor` pair, both on the same frozen embeddings:

| head (frozen embedding input) | val hard acc | OOS hard acc | val/OOS quality MSE |
|---|---:|---:|---|
| linear (baseline, end-to-end) | 67.1% | 64.8% | 0.665 / 0.616 |
| TabM ensemble (frozen embedding) | 67.2% | 64.6% | 0.656 / 0.613 |
| tree (frozen embedding) | 66.2% | 63.8% | 0.663 / 0.623 |

**End-to-end (TabM head trained jointly with the encoder, not frozen)** -- added `head_type`
param to `DeepFeatModel`/`build_model` (`ensemble/deep_features/btc_deepfeat_encoders_20260806.py`),
`--head-type tabm` on the training script, retrained the full final config from scratch with the
TabM head replacing the linear head end-to-end:

| head_type | val hard acc | OOS hard acc |
|---|---:|---:|
| linear (baseline) | 67.1% | 64.8% |
| tabm (end-to-end) | 66.9% | 64.9% |

**Conclusion:** every head architecture tried (linear, TabM ensemble, tree-based GBDT), in both
integration modes (frozen-embedding downstream and joint end-to-end training), lands within ~1pt
of the same 66-67% VAL / 64-65% OOS accuracy. Swapping the function class on top of the 32-dim
transformer embedding does not move the number -- the embedding itself is the ceiling, not the
head's capacity. This is consistent with (and reinforces) the earlier finding that raw features +
GBDT (65.5%/63.4%) and the tuned transformer (67.0%/64.8%) are close to each other despite very
different architectures: this label/feature combination appears to have a real accuracy ceiling
around 65-67% regardless of model family, and (per the backtest section above) that ceiling does
not convert into a profitable entry signal at the win rates the TP/SL ratio requires.

## Fundamental fix attempt: causal triple-barrier label + corrected TP/SL vol basis (2026-08-06, same day)

Per [[project-btc-deepfeat-acc-pnl-gap-diagnosis-20260806]]'s 3 root causes, rebuilt the training
target from scratch instead of tuning further on the old one:

- **Root cause 1 fix (structural)**: `scripts/build_btc_5m_tripbarrier_tradeoutcome_labels_20260806.py`
  replaces the retrospective zigzag-wave label with a causal triple-barrier label -- for every bar,
  simulate opening a LONG and a SHORT at next bar's open and record which side (if either) hits
  its own TP before its own SL within 24h. This is literally the trading question, not a proxy for
  it. Balanced label (CASH 38.3% / LONG 30.9% / SHORT 30.8%), a much harder and more honest task
  than the old wave label (95%+ non-CASH).
- **Root cause 3 fix (mechanical)**: TP/SL now sized off the rolling dispersion of **12-bar (1h)
  cumulative** log returns (288-bar/24h lookback) instead of single-bar (5m) log-return std. Median
  SL distance moved from 0.15% (old, ≈ or < typical 3-bar noise) to 0.49% (comfortably above
  median/mean 3-bar noise of 0.11%/0.17%). Same TP_MULT=2.5/SL_MULT=1.2/horizon=288 kept for
  continuity with the existing backtest convention.

Retrained the best-known transformer config (window=48/d_model=96/n_layers=3/dropout=0.25, no
label-sharpen/cash-weight shaping -- the new label doesn't have the old label's soft/hard mismatch)
directly against this label: val 40.2% / OOS 36.0% hard accuracy -- barely above the 38.3%
majority-CASH baseline, confirming this is a genuinely much harder prediction target than the old
89-95%-non-CASH wave label (expected: forward-looking prediction is intrinsically harder than
retrospective segmentation).

**Backtest result** (`scripts/backtest_btc_tripbarrier_model_20260806.py`, direct argmax entry --
no threshold-quantile fitting needed since the label already IS the trade decision, using the
exact same TP/SL basis the label was built with):

| split | mode | win rate | mean ret/trade | sum ret | vs old zigzag-label backtest |
|---|---|---:|---:|---:|---|
| VAL | fresh_entry | 33.9% | -0.072% | -34.2% | old: 32.3% win rate, -119% sum ret |
| OOS | fresh_entry | **35.6%** | -0.036% | -14.8% | old: 30.3% win rate, -84.4% sum ret |

Still net negative, but a large, measurable improvement: OOS win rate (35.6%) now clears the
32.4% breakeven line (it didn't before, at 28-32% across every prior attempt), and mean loss per
trade shrank ~3x (-0.036% vs -0.09 to -0.10% before). Back out the fixed 10bps roundtrip cost
(0.09% account-level at margin_fraction=0.30/leverage=3, notional=0.90): gross price-move edge
per trade ≈ -0.036% + 0.09% = **+0.054% positive before transaction costs** -- the model now has a
real, small directional edge; the fixed cost assumption is what pushes net PnL back to negative.
Trade side is heavily SHORT-skewed (272 short vs 144 long, OOS fresh_entry) -- not yet
investigated, could reflect real BTC downtrend in this window or a model bias worth checking.

**Conclusion: fixing root causes 1 and 3 (this session) moved the result from "structurally
impossible" (win rate stuck at/below breakeven regardless of any entry-timing or quality-filter
trick) to "close to breakeven, gross-positive pre-cost, net-negative after a fixed 10bps cost
assumption."** This is a fundamentally different, more tractable place to be than before -- the
next lever is cost/frequency/confidence-threshold tuning on a target that has already been shown
to carry real signal, not more architecture search on a target that couldn't work by construction.

## Oracle (perfect-foresight) ceiling for the triple-barrier label (2026-08-06, same day)

User's question: since this is a teacher label, shouldn't perfect-foresight trading of the label
itself show the maximum achievable return -- and isn't that what the model is being distilled
toward? Ran `scripts/eval_btc_tripbarrier_label_oracle_quality_20260806.py`: same
`simulate_single_position` mechanics as the model backtest, but using the TRUE
`trade_outcome_action` label directly as the entry signal instead of any model prediction.

| split | mode | n_trades | win rate | mean ret/trade | compounded final equity |
|---|---|---:|---:|---:|---:|
| VAL | fresh_entry | 494 | 100.0% (by construction) | +0.770% | 44.05x |
| OOS | fresh_entry | 404 | **100.0%** | **+0.944%** | **44.26x** |

Win rate of exactly 100% confirms the label pipeline is bug-free (every LONG/SHORT-labeled bar
does guarantee its own TP hits before its own SL, as designed) and quantifies the label's true
economic ceiling: compounded ~44x equity over the 3-month OOS window under perfect prediction.

**This reframes the current bottleneck.** The tuned model only reaches 36.0% OOS classification
accuracy (barely above the 38.3% majority-CASH baseline) and 35.6% backtest win rate (vs the
oracle's 100%) -- and critically, **this new label has not yet received ANY of the accuracy-tuning
effort the old zigzag label got** (hyperparameter sweep, label-sharpen/cash-weight shaping, TabM
head, etc. were all done on the old label; the triple-barrier label was trained once with default
hyperparameters). Given the ceiling is proven enormous, the next lever is accuracy tuning on THIS
label -- not cost/frequency/threshold tuning on a still-undertuned model.

## Race-based soft label for triple-barrier: implemented, calibrated correctly, did NOT improve accuracy (2026-08-06, same day)

Replaced the triple-barrier label's flat epsilon-smoothed one-hot soft target with a genuinely
graded one derived from the barrier race itself
(`scripts/build_btc_5m_tripbarrier_tradeoutcome_labels_20260806.py::_triple_barrier_race`): for
each bar, LONG and SHORT each get an independent conviction score
`sign * (1 - bars_to_resolution/horizon)` (fast TP hit -> near +1, fast SL hit -> near -1, slow
resolution -> near 0, timeout -> exactly 0), softmaxed against a fixed CASH=0 baseline
(temperature 0.35). This is provably well-calibrated: because a bar's hard label can only be
LONG/SHORT when that side's score is strictly positive (it actually hit its own TP), a 0 CASH
baseline makes `argmax(soft) == hard_label` hold **99.98%** of the time (the only exceptions are
rare same-bar double-TP whipsaws) -- unlike the old zigzag soft label, which needed
sharpen/cash-weight correction because its own argmax disagreed with the hard label on ~9% of
active bars. Per-class mean max-probability is now genuinely graded (CASH 0.77, LONG 0.88, SHORT
0.89) instead of flat 0.95 everywhere.

**Despite being a better-calibrated, more informative soft target, retraining did not improve
accuracy** -- tried default (no shaping), `label_sharpen` in {1.0, 0.6, 0.3}, and `cash_weight=0.3`,
all on the same window=48/d_model=96/n_layers=3/dropout=0.25 architecture:

| config | val acc | OOS acc |
|---|---:|---:|
| flat-smoothed hard label (original, no shaping needed) | **40.2%** | **36.0%** |
| race-soft label, no shaping | 39.2% | 34.6% |
| race-soft label, sharpen=0.6 | 38.5% | 34.2% |
| race-soft label, sharpen=0.3 | 38.1% | 34.0% |
| race-soft label, cash_weight=0.3 | 29.5% | 34.8% |

Sharpening (which helped enormously on the miscalibrated zigzag label) made things slightly worse
here, consistent with the label already being well-calibrated -- there was no soft/hard mismatch
to correct, so sharpening just discarded real graded information for no benefit. Cash-weighting
(also a big zigzag win, where CASH was 4.7% of data) didn't help either, unsurprising given CASH
is now 38.3% of a genuinely 3-way-balanced label, not a rare minority class needing rebalancing.

**Conclusion: the flat-smoothed hard-label version remains the best triple-barrier model found**
(val 40.2%/OOS 36.0%, the one behind the 35.6% OOS win rate / gross-positive-pre-cost backtest
result above). The race-based soft label is a conceptually correct, well-validated artifact (kept
for future use / documentation) but doesn't currently help this task -- the earlier win from
soft-label shaping was specific to fixing a genuine calibration bug in the old zigzag label,
not a general "soft labels beat hard labels" result.

## Raw-feature GBDT re-check on the triple-barrier label -- fair comparison after tuning (2026-08-06, same day)

First pass (`scripts/train_btc_tripbarrier_gbdt_baseline_20260806.py`, one untuned config) showed
the transformer clearly beating a raw-feature GBDT on this new label -- the opposite of the old
zigzag-label result. User correctly flagged that this wasn't a fair comparison: the transformer
had been through an architecture sweep + label_sharpen sweep + cash_weight sweep, while GBDT got
one default config. Ran `scripts/sweep_btc_tripbarrier_gbdt_20260806.py`: 54 configs
(max_depth∈{4,6,8} × learning_rate∈{0.03,0.05,0.1} × l2_regularization∈{0.5,2.0} ×
cash sample_weight∈{1.0,0.9,0.7}), selected by VAL log_loss, confirmed once on OOS, then backtested
with the same fresh-entry/TP-SL mechanics as the transformer.

| model | val acc | OOS acc | OOS win rate | OOS sum ret |
|---|---:|---:|---:|---:|
| GBDT, untuned (1 config) | 34.5% | 33.4% | 31.1% | -46.4% |
| **GBDT, tuned (54-config sweep, best: max_depth=4/lr=0.03/l2=2.0/cash_weight=1.0)** | 36.9% | 35.4% | 33.6% | **-23.0%** |
| transformer, untuned default | 40.2% | 36.0% | 35.6% | -14.8% |
| transformer, cash_weight=0.9 | 40.0% | 34.4% | 35.5% | **-9.5%** |

Tuning closed most of the gap (win rate 31.1%→33.6%, loss -46.4%→-23.0%) -- the smallest/most
regularized GBDT config won, consistent with GBDT's earlier severe overfitting (87% train acc vs
34.5% val on the untuned run). But the transformer's best configs still win on both win rate and
final PnL after fair tuning on both sides. **Conclusion stands but the margin is much smaller than
the first-pass comparison suggested**: for this genuinely forward-looking triple-barrier target,
the 48-bar sequence window appears to carry real information a single-bar GBDT snapshot can't
fully recover, but it's a modest edge, not a decisive one -- worth remembering when comparing
architectures on any future label without tuning both sides to a comparable degree.

## Not yet done (explicitly out of scope for this stage)

- No trading strategy / TP-SL / backtest built on top of these embeddings or logits.
- No comparison against the raw causalfix_final features as a baseline classifier (the earlier
  closed JEPA line found deep embeddings ranked *weaker* than raw features when both were fed to
  LightGBM -- this new soft-label-supervised line has not been tested that way).
- Early-stopping at epoch 1-2 in all three runs suggests the classification signal is thin and
  overfits fast; hyperparameter tuning (dropout, weight decay, window length, smaller models) has
  not been attempted.
- transformer's advantage may reflect the window's last-token pooling picking up recency bias
  rather than genuine cross-feature learning -- not yet diagnosed.
