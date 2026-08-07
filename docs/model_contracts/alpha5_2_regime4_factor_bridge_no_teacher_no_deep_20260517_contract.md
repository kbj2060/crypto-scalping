# Alpha5.2 Regime4 Factor Bridge No-Teacher No-Deep Contract

Date: 2026-05-17

## Purpose

Alpha5.2 is a feature-only retest of Alpha5 after restoring Alpha4.3-style
regime factor information under the new Regime4 prefix.

No changes were made to the Alpha5 architecture, selection score, runner
selection, execution contract, or routing logic.

Kept unchanged from Alpha5:

```text
HGB parent
no teacher sequence gate
no V27/V31 deep scout
original Alpha5/Alpha4.3 no-teacher runner and scale search
original alpha2._score selection function
corrected Alpha3 limit-close execution
```

Changed surface:

```text
parent feature input only
```

## Input Contract

Canonical fixed preprocessing inputs:

```text
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/fixed_regime4_tp18_sl10_preprocess_20260517/trade_candidates_2025_regime4_tp18_sl10_fixed.csv
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/fixed_regime4_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_tp18_sl10_fixed.csv
```

Feature contract:

```text
parent feature count                  93
legacy clean_regime_2024_unsup_v4_*   0
clean_regime4_2024_unsup_v1_*         20
regime4_pred_*                        12
tp_sl_action_score                    enabled
tp_sl_x_* interactions                disabled
```

New current-Regime4 auxiliary features versus Alpha5:

```text
clean_regime4_2024_unsup_v1_factor_trend
clean_regime4_2024_unsup_v1_factor_flow
clean_regime4_2024_unsup_v1_factor_vol
clean_regime4_2024_unsup_v1_factor_crowding
clean_regime4_2024_unsup_v1_factor_liquidity
clean_regime4_2024_unsup_v1_trend_bias
clean_regime4_2024_unsup_v1_risk_off_prob
clean_regime4_2024_unsup_v1_transition_risk
```

`risk_off_prob` and `transition_risk` are auxiliary scores, not regime classes.
The regime taxonomy remains:

```text
bull
bear
chop
whipsaw
```

## Training And Selection

Script:

```text
/home/llewyn/crypto-scalping/scripts/train_eval_alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517.py
```

Command:

```bash
venv/bin/python scripts/train_eval_alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517.py \
  --model-id alpha5_2_regime4_factor_bridge_no_teacher_no_deep_20260517 \
  --report-stem alpha5_2_regime4_factor_bridge_no_teacher_no_deep \
  --out-dir /home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha5_2_regime4_factor_bridge_no_teacher_no_deep_20260517 \
  --seed 5517
```

Split:

```text
train      2025-01-01 00:00:00 .. 2025-09-30 23:55:00
selection  2025-10-01 00:00:00 .. 2025-12-31 23:55:00
OOS        2026-01-01 00:00:00 .. 2026-02-28 16:00:00
```

Selection winner:

```text
parent_direct_raw_no_teacher
```

## Artifacts

```text
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha5_2_regime4_factor_bridge_no_teacher_no_deep_20260517/parent.pkl
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha5_2_regime4_factor_bridge_no_teacher_no_deep_20260517/runners/parent_direct_raw_no_teacher_runner.pkl
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha5_2_regime4_factor_bridge_no_teacher_no_deep_20260517/runners/parent_direct_scaled_no_teacher_runner.pkl
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha5_2_regime4_factor_bridge_no_teacher_no_deep_20260517/alpha5_2_regime4_factor_bridge_no_teacher_no_deep_summary.json
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha5_2_regime4_factor_bridge_no_teacher_no_deep_20260517/alpha5_2_regime4_factor_bridge_no_teacher_no_deep_grid.csv
```

## Results

Selected by 2025Q4:

| Variant | Cost1 PnL | MDD | Cost2 PnL | Cost3 PnL | Trades |
|---|---:|---:|---:|---:|---:|
| `parent_direct_raw_no_teacher` | +83.24% | -26.91% | +73.79% | +70.68% | 58 |

Comparison:

```text
Alpha4.3 reference cost1 +183.42%, MDD -21.99%, cost2 +169.76%, cost3 +79.27%
Alpha5 selected   cost1  +86.93%, MDD -24.44%, cost2  +78.99%, cost3 +72.26%
Alpha5.1 selected cost1  +65.18%, MDD -23.82%, cost2  +68.70%, cost3 +65.06%
Alpha5.2 selected cost1  +83.24%, MDD -26.91%, cost2  +73.79%, cost3 +70.68%
```

## Verdict

Alpha5.2 confirms that simply restoring Alpha4.3-style factor/risk/transition
features under the Regime4 prefix is not enough to recover Alpha4.3 performance.

Promotion status:

```text
failed_candidate
not_live_main
```

The next useful comparison is not more feature expansion. It should isolate why
Alpha4.3's legacy regime block works better, for example by running a controlled
ablation that keeps Alpha4.3 parent training fixed and swaps only subsets of the
legacy regime block against their Regime4 equivalents.
