# Alpha5.1 Regime4 Interaction No-Teacher No-Deep Contract

Date: 2026-05-17

## Purpose

Alpha5.1 tests whether the Alpha5 Regime4 feature stack improves when the fixed
TP/SL action score is crossed with current and future Regime4 state features.

Architecture:

```text
HGB parent with Regime4 interaction inputs -> direct no-teacher decision -> optional parent scale -> V21.2 runner -> corrected Alpha3 limit-close execution
```

Disabled layers:

```text
teacher sequence gate
V27/V31 deep scout
legacy clean_regime_2024_unsup_v4_* features
```

## Input Contract

Canonical fixed preprocessing inputs:

```text
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/fixed_regime4_tp18_sl10_preprocess_20260517/trade_candidates_2025_regime4_tp18_sl10_fixed.csv
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/fixed_regime4_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_tp18_sl10_fixed.csv
```

Feature contract:

```text
parent feature count                  107
legacy clean_regime_2024_unsup_v4_*   0
clean_regime4_2024_unsup_v1_*         12
regime4_pred_*                        12
tp_sl_action_score                    enabled
tp_sl_x_* interaction features         22
```

Interaction features:

```text
tp_sl_x_current_regime4_bull
tp_sl_x_current_regime4_bear
tp_sl_x_current_regime4_chop
tp_sl_x_current_regime4_whipsaw
tp_sl_x_current_trend_prob
tp_sl_x_current_micro_prob
tp_sl_x_current_directional_bias
tp_sl_x_current_range_prob
tp_sl_x_current_instability_prob
tp_sl_x_current_confidence
tp_sl_x_current_margin
tp_sl_x_future_regime4_bull
tp_sl_x_future_regime4_bear
tp_sl_x_future_regime4_chop
tp_sl_x_future_regime4_whipsaw
tp_sl_x_future_trend_prob
tp_sl_x_future_micro_prob
tp_sl_x_future_directional_bias
tp_sl_x_future_range_prob
tp_sl_x_future_instability_prob
tp_sl_x_future_confidence
tp_sl_x_future_margin
```

Regime4 classes:

```text
bull
bear
chop
whipsaw
```

TP/SL action score:

```text
TP 1.8%
SL 1.0%
horizon 48 bars
entry next-bar open
same-bar tie -> SL wins
```

## Training And Selection

Script:

```text
/home/llewyn/crypto-scalping/scripts/train_eval_alpha5_1_regime4_interactions_no_teacher_no_deep_20260517.py
```

Split:

```text
train      2025-01-01 00:00:00 .. 2025-09-30 23:55:00
selection  2025-10-01 00:00:00 .. 2025-12-31 23:55:00
OOS        2026-01-01 00:00:00 .. 2026-02-28 16:00:00
```

Candidates:

```text
parent_direct_raw_no_teacher
parent_direct_scaled_no_teacher
```

Selection policy:

```text
cost_stress_v2
```

Selection winner:

```text
parent_direct_scaled_no_teacher
```

Selected runtime:

```json
{
  "name": "noflip_c0.56_parent_scale1.10",
  "confidence": 0.56,
  "parent_notional_scale": 1.1,
  "max_notional": 2.75
}
```

Runner config:

```text
v21_2_parent_noop
```

## Artifacts

```text
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha5_1_regime4_interactions_no_teacher_no_deep_20260517/parent.pkl
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha5_1_regime4_interactions_no_teacher_no_deep_20260517/runners/parent_direct_raw_no_teacher_runner.pkl
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha5_1_regime4_interactions_no_teacher_no_deep_20260517/runners/parent_direct_scaled_no_teacher_runner.pkl
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha5_1_regime4_interactions_no_teacher_no_deep_20260517/alpha5_1_regime4_interactions_no_teacher_no_deep_summary.json
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha5_1_regime4_interactions_no_teacher_no_deep_20260517/alpha5_1_regime4_interactions_no_teacher_no_deep_grid.csv
```

## Results

Selected by 2025Q4:

| Variant | Cost1 PnL | MDD | Cost2 PnL | Cost3 PnL | Trades |
|---|---:|---:|---:|---:|---:|
| `parent_direct_scaled_no_teacher` | +65.18% | -23.82% | +68.70% | +65.06% | 75 |

Non-selected OOS reference:

| Variant | Cost1 PnL | MDD | Cost2 PnL | Cost3 PnL | Trades |
|---|---:|---:|---:|---:|---:|
| `parent_direct_raw_no_teacher` | +71.31% | -25.98% | +75.75% | +70.55% | 76 |

Comparison:

```text
Alpha4.3 reference cost1 +183.42%, MDD -21.99%, cost2 +169.76%, cost3 +79.27%
Alpha5 selected   cost1  +86.93%, MDD -24.44%, cost2  +78.99%, cost3 +72.26%
Alpha5.1 selected cost1  +65.18%, MDD -23.82%, cost2  +68.70%, cost3 +65.06%
```

## Verdict

Alpha5.1 passes the feature-contract and no-leak checks, but it does not improve
Alpha5 or Alpha4.3. The TP/SL x Regime4 cross terms are not promoted as the next
Alpha5 direction.

Promotion status:

```text
failed_candidate
not_live_main
```

The useful signal from this run is negative: simple multiplicative TP/SL x
Regime4 interactions increase input dimensionality without improving 2026 OOS.
Next Alpha5 work should focus on regime-conditioned calibration, MoE routing, or
walk-forward specialist heads instead of adding more crossed tabular features to
the same parent.
