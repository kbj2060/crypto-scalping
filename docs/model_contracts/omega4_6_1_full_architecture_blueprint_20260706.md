# Omega4.6.1 — Full Architecture Blueprint (2026-07-06)

Reference document written before live wiring. Consolidates everything established across this
session's verification work (extended-OOS retest, event-flat correction, Gate 2 fix, runtime-native
parity, lookahead audit). For status/verdict see `omega4_6_1_live_path_parity_and_lookahead_audit_20260706.md`;
this document is the "how it actually works" reference.

## One-paragraph summary

Omega4.6.1 trades ETHUSDT perpetual futures at 5-minute bar resolution. Two independently-trained
"3-head" neural direction/quality/exit classifiers (`h48qual`, `zig075`) each watch every bar and
propose LONG/SHORT/CASH with a confidence score; a priority router picks h48qual's opinion when it
has one, otherwise zig075's, and only one position is held at a time. Position size (margin +
leverage) is set by a small gradient-boosted regressor per component that reads the neural model's
own outputs (not raw market data) and calibrates conviction into risk. Stop-loss/take-profit are
volatility-adaptive barriers; while a position is open, a third neural head decides bar-by-bar
whether to exit early. A final gate skips trades entirely when a slow-moving mean-reversion
indicator (`ou_halflife`) says conditions are unfavorable. Typical trading frequency: ~1
trade/week, held for hours to ~2 weeks.

## Pipeline diagram (text form)

```
raw 5m OHLCV+funding/OI/toptrader (ETHUSDT)
        |
        v
[L0] FeatureEngineer batch/live feature computation  (features/engineering.py, features/elite.py)
        |  -> 96 engineered technical/orderflow/volatility/calendar/OU features per bar
        v
[L1] Regime3-Current HMM sidecar (causal filter_proba, per-adapter, NOT in shared pipeline)
        |  -> bull_prob, bear_prob, chop_prob, confidence, margin, entropy   (6 cols)
        v
   === 102 base features now complete, fed to BOTH parents below ===
        |
        +-------------------------------+-------------------------------+
        v                                                               v
[L2a] h48qual parent                                           [L2b] zig075 parent
  3-head TabM, regime-routed to                                  3-head TabM, regime-routed to
  bull/bear/chop expert sub-network                              bull/bear/chop expert sub-network
  direction (CASH/LONG/SHORT) + quality                          direction (CASH/LONG/SHORT) + quality
  quality_threshold = 0.50 (permissive)                          quality_threshold = 0.75 (strict)
        |                                                               |
        v                                                               v
[L3a] ATR-adaptive TP/SL barrier (192-bar ATR)                  [L3b] same formula, own component
        |                                                               |
        v                                                               v
[L4a] risk-sizing sidecar (HGB, side-split)                     [L4b] risk-sizing sidecar (HGB, side-split)
  parent-output features -> margin_fraction, leverage             (same structure, own trained model)
        |                                                               |
        +-------------------------------+-------------------------------+
                                         v
                    [L6] GREEDY PRIORITY ROUTER (single shared position slot)
                         "does h48qual have a signal right now? take it.
                          else does zig075? take it. else stay flat."
                                         |
                                         v
                    [L7] per-component/side notional rescale (SCALE_MAP)
                         + leverage cap 5.0x, notional cap 1.8x
                                         |
                                         v
                    [L8] DURATION GATE: ou_halflife <= 0.005417 -> skip entirely
                                         |
                                         v
                              ENTER at next bar's open
                                         |
                                         v
              [L5] while open: monitor every bar --
                   TP/SL barrier hit? -> exit.
                   else: originating component's exit-head prob >= 0.95? -> exit.
                   else: hold.
```

## Layer-by-layer detail

### L0 — Feature engineering (shared, not Omega-specific)

- Code: `features/engineering.py::FeatureEngineer`, `features/elite.py` (SyntheticAlphaEngine,
  VolatilityModelEngine, NewEliteSignalEngine, RegimeEngine).
- Same class used for both offline dataset building (`training_features_2026_rebuilt.csv`) and
  live (`trading_bot.py`'s `fe_engine = FeatureEngineer()` in the main loop) -- single source of
  truth, no separate live/offline feature code to drift apart.
- Produces ~96 of Omega4.6.1's 102 needed columns: OHLCV, funding rate + derived (roc/z-score/
  pressure/abs), open interest change, top-trader long/short ratio z-score, whale/smart-money flow,
  technical indicators (RSI, MACD histogram, Bollinger width/position), volatility family (Garman-
  Klass, Rogers-Satchell, Parkinson, realized-vol ratio, volatility z-score), CVD/CVP order-flow
  clustering, candle-shape features (body/wick ratios), skew/kurtosis, trend/mean-reversion
  distance measures, Hurst exponents (48/288-bar), calendar (hour/minute/day-of-week sin-cos,
  session flags), OU-based features (`ou_funding_z`, **`ou_halflife`** -- the L8 gate's own input),
  jump/EVT tail-risk flags, `sig_volume_confirm`/`sig_liquidity_trap`/`sig_trend_health` (elite
  signals), BTC cross-correlation/spread features.
- Explicitly does NOT include m7 (SevenModelEnsemble) or NeuralForecast (PatchTST/TiDE/DLinear)
  outputs -- unlike most other Omega4.x candidates, which is why Omega4.6.1 avoided the
  m7-unrecoverability wall that blocks e.g. `omega4_3_valonly_logrisk_tail050`.
- Verified clean of lookahead: no `shift(-)`/`center=True`/`bfill` patterns; rolling windows are
  backward-only by construction.

### L1 — Regime3-Current HMM sidecar (per-adapter, causal)

- Code: `omega4_6_2_source_parent_live.py::Regime3CurrentLiveFeatures` (reused directly by
  Omega4.6.1's live adapter -- NOT part of the shared `FeatureEngineer` pipeline, a real gap found
  and fixed 2026-07-06).
- Model: `regime3_current_sensitive_hmm_wide24_2024.joblib`, a Hidden Markov Model trained once on
  2024 data, applied forward-only via `filter_proba()` (the causal HMM inference mode -- computes
  P(state | observations up to and including now), never uses future observations).
- Produces: `bull_prob`, `bear_prob`, `chop_prob` (softmax over 3 latent regimes), `confidence`
  (=max prob), `margin` (=gap between top-2 probs), `entropy` (uncertainty measure).
- Purpose: (a) feeds the parent models as 6 additional input features (self-aware regime context),
  (b) `argmax(bull,bear,chop)` selects WHICH of the 3 expert sub-networks handles this bar (see L2).

### L2 — Two independent parent models (the actual trading brains)

Both are "3-head TabM" networks (TabM: k=8 parallel sub-networks sharing most weights, a tabular
deep-learning architecture; hidden=192, 3 layers, dropout 0.08). Each parent has 3 EXPERT copies
(bull/bear/chop), and the L1 regime route picks which expert answers for the current bar --
this is a mixture-of-experts design where the "expert" assignment is itself a causal regime call,
not learned end-to-end.

| | h48qual | zig075 |
|---|---|---|
| Direction label | zigzag pivot detection (`zigzag_action`, same for both) | same |
| Quality label | **separate**, conservative "h48" barrier rule (net edge > cost, MAE capped, MFE/MAE ratio, max 288-bar hold) | **same as direction** (quality = "how confident is the direction call") |
| Quality threshold | 0.50 (permissive -- takes more, weaker-conviction trades) | 0.75 (strict -- only its most confident calls) |
| Bundle | `omega4_3head_parent72_..._h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630` | `omega4_3head_parent72_..._current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629` |

Each expert network has 3 output heads:
- **Direction**: 3-class softmax (CASH/LONG/SHORT).
- **Quality**: 3-class softmax, gates whether the direction call is acted on
  (`final_action = direction if quality[direction] >= threshold else CASH`).
- **Exit**: binary softmax (stay/exit), only consulted while a position from THIS component is
  open (see L5) -- takes 13 additional position-state inputs (side, hold_bars, unrealized move,
  MFE, MAE, giveback ratio, distance-to-TP, distance-to-SL, notional, leverage, exposure, TP, SL)
  that are zeroed during entry-decision inference and populated during exit-monitoring inference.

Input contract: 102 base features (L0's 96 + L1's 6) + 13 position-state features = 115 total.
Base-feature list is fail-fast validated against forbidden prefixes (`teacher_`, `regime4_pred_`,
`clean_regime4_`) and tokens (`tp_sl_action_score`) at load time.

### L3 — ATR-adaptive stop-loss/take-profit barriers

Applied identically to both components' decisions, per `eval_omega4_1_atr_safety_sltp_20260622.py`:

```
atr_pct = 192-bar (16h) rolling average true range, as % of price
take_profit = clip(max(0.075, atr_pct * 12.0), 0.0, 0.22)   # 7.5%-22% price move
stop_loss   = clip(max(0.040, atr_pct * 6.0),  0.0, 0.12)   # 4.0%-12% price move
```

These are RAW price-move thresholds -- position size (L4) does not shift where the barrier sits,
only how much money moves when it's hit (`notional_scaled_sltp: false` in both components'
contracts, per the Futures Risk Sizing Contract in this project's operating rules).

### L4 — Risk-sizing sidecar (per component, independently trained)

A small HistGradientBoostingRegressor, side-split (separate model for LONG vs SHORT trades),
`risk_feature_mode: "parent_outputs"` -- it sees ONLY the parent's own decision-time outputs
(router confidence/margin, direction/quality probabilities and derived stats, baseline
notional/leverage/take-profit/stop-loss, current `atr_pct`), never raw market features. It predicts
a scalar "risk score" (roughly: expected trade quality), which is converted to sizing via a
sigmoid mapping:

```
z = clip((score - train_score_q50) / train_score_iqr, -8, 8)
margin_fraction = clip(base_margin * (min_scale + (max_scale-min_scale) * sigmoid(temp * z)), floor, cap)
leverage        = clip(leverage_min + (leverage_max-leverage_min) * sigmoid(leverage_temp * z), leverage_floor, leverage_cap)
```

where `base_margin = base_notional / base_leverage`, and `base_notional = 0.45 * expert_scale`
(expert_scale: bull=0.75, bear=0.90, chop=0.90 -- a fixed discount depending on which regime expert
is currently active). Each component has its own independently-tuned mapping constants
(`min_scale`/`max_scale`/`temp`/`floor`/`cap`/leverage equivalents) -- h48qual's and zig075's
sizing behave differently even holding the score fixed.

### L5 — Exit-head bar-by-bar monitoring (while a position is open)

At every 5-minute bar while a position is open:
1. Check the fixed L3 barriers first (take-profit / stop-loss on raw price move).
2. If neither hit, evaluate the ORIGINATING component's exit head (whichever regime expert is
   active THIS bar, per the current L1 route -- can differ from the expert that made the original
   entry call) using the 13 position-state features described in L2.
3. If `exit_prob >= 0.95` (both components use the same exit threshold), close the position.
4. Otherwise hold to the next bar.

This is what allows holds of hours to ~2 weeks (max observed 282h in the 2026 Jan-Jun OOS window)
-- there is no fixed max-hold; the position is only closed by a barrier or the learned exit call
(or a forced end-of-data close in backtests).

### L6 — Greedy priority router (single shared account)

At any bar with no open position: check h48qual's `final_action` first; if non-CASH and its
quality gate passed, take it. Otherwise check zig075 the same way. Only one position exists at a
time -- if a signal fires while a position from the OTHER component is already open, it is simply
ignored (lost) that bar.

**This was corrected 2026-07-06.** Every backtest before this session's live-parity check computed
h48qual and zig075 as two INDEPENDENT full simulations (each with its own imaginary 100% capital)
and reconciled overlaps after the fact -- impossible for a real system, which must decide with only
one account and no knowledge of the other component's counterfactual future. The genuine greedy
version was re-derived and found to barely change the bottom line (+145.34% vs the
offline-reconciled +145.46% on the 2026 OOS window).

### L7 — Notional/leverage rescale + caps

After L4's raw sizing, a fixed per-component-per-side multiplier is applied (inherited from the
base `omega4_6_plus_t12_nohold_risk1_20260630` model's own tuning):

| | LONG scale | SHORT scale |
|---|---|---|
| h48qual | 0.38x | 2.499x |
| zig075 | 2.446x | 2.478x |

(h48qual longs are heavily discounted relative to its shorts and to zig075's either side -- this
reflects whatever the upstream tuning found about each component/side's realized reliability.)
Result is then clipped to `leverage <= 5.0x`, `notional <= 1.8x` (180% of account equity at risk,
i.e. up to 1.8x leveraged exposure relative to total capital).

### L8 — Duration gate (final filter, applied at entry)

```
if ou_halflife <= 0.005417:  # re-selected via VAL-only grid search 2026-07-06
    skip this trade entirely (margin=leverage=notional=0)
```

`ou_halflife` (from L0) is a normalized Ornstein-Uhlenbeck mean-reversion half-life estimate,
derived from the 5-day rolling AR(1) autocorrelation of `funding_roc_12` (funding-rate rate-of-
change, chosen over raw funding rate specifically because funding is an 8-hour step function that
produces a degenerate/constant half-life estimate at 5-minute resolution -- a documented fix in
`features/elite.py`). Low half-life apparently correlates with conditions unfavorable to this
strategy's trades; skips about 24% of otherwise-valid signals (8 of 33 in the 2026 OOS window) and
empirically improves both PnL and MDD versus not gating at all.

## Execution mechanics

- **Entry**: next bar's open price, with slippage (`SLIP_RATE=0.0002`).
- **Exit**: current bar's close price (or open on forced end-of-data), with slippage.
- **Fees**: `FEE_RATE=0.0005` on both entry and exit, applied to notional.
- **Accounting**: `notional = margin_fraction * leverage`; `PnL = realized_price_move * notional`
  (per this project's Futures Risk Sizing Contract) -- verified accounting-consistent to
  floating-point precision in the redteam-style check.

## What's genuinely novel vs. inherited

- L0-L1, L3, the general 3-head-TabM-plus-sidecar pattern, and the L7 scale/cap contract are all
  inherited unchanged from the base `omega4_6_plus_t12_nohold_risk1_20260630` model (and further
  back, the broader Omega4.x lineage).
- **L8 (duration gate) is Omega4.6.1's own contribution** on top of that base -- everything else is
  the same machinery `omega4_6_2_*` and other Omega4.6-family siblings also use.
- **L6's greedy (vs. reconciled) formulation was newly built this session** to make the router
  honestly live-executable; it did not exist as a real-time algorithm before today.
