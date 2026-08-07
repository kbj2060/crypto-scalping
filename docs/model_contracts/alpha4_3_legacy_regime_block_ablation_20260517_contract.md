# Alpha4.3 Legacy Regime Block Ablation

Date: 2026-05-17

## Purpose

Decompose which subfeatures of the Alpha4.3 legacy clean-regime block contributed
to downstream no-teacher/no-deep performance.

This is a diagnostic ablation, not a promotion candidate.

## Method

The final run used the Alpha4.3 parent feature basis:

```text
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/artifacts/hgb/parent.pkl
```

The Alpha4.3 basis has:

```text
feature count                         84
legacy clean_regime_2024_unsup_v4_*   23
tp_sl_action_score                    enabled
```

Each variant retrained:

```text
HGB parent
parent_direct_raw_no_teacher runner
parent_direct_scaled_no_teacher runner
parent scale runtime selection
runner config selection
```

Kept unchanged:

```text
teacher sequence gate disabled
V27/V31 deep scout disabled
original Alpha5 alpha2._score selection
corrected Alpha3 limit-close execution
2025Q4 selection only, 2026 OOS untouched by selection
```

## Feature Groups

`factor_core`:

```text
factor_trend
factor_flow
factor_vol
factor_crowding
factor_liquidity
trend_bias
```

`risk_transition`:

```text
risk_off_prob
transition_risk
```

`semantic_probs`:

```text
bull_prob
bear_prob
chop_prob
whipsaw_prob
normal_prob
confidence
entropy
```

`cluster_state`:

```text
state_code
cluster
cluster_confidence
cluster_prob_0..4
```

## Artifacts

```text
/home/llewyn/crypto-scalping/scripts/ablate_alpha4_3_legacy_regime_block_20260517.py
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha4_3_legacy_regime_block_ablation_alpha43basis_20260517/alpha4_3_legacy_regime_block_ablation_summary.json
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha4_3_legacy_regime_block_ablation_alpha43basis_20260517/alpha4_3_legacy_regime_block_ablation_results.csv
```

## Results

2026 OOS, selected by 2025Q4:

| Variant | Legacy Features | Cost1 PnL | MDD | Cost2 PnL | Cost3 PnL | Trades |
|---|---:|---:|---:|---:|---:|---:|
| `no_legacy` | 0 | +139.59% | -25.52% | +131.83% | +139.82% | 76 |
| `all_legacy` | 23 | +113.88% | -21.01% | +109.56% | +104.56% | 59 |
| `semantic_probs` | 7 | +89.30% | -22.82% | +74.90% | +69.27% | 66 |
| `risk_transition` | 2 | +73.89% | -22.54% | +83.45% | +68.01% | 79 |
| `cluster_state` | 8 | +66.97% | -28.05% | +75.57% | +44.47% | 75 |
| `factor_core` | 6 | +64.95% | -22.44% | +71.59% | +54.06% | 70 |

Reference:

```text
Alpha4.3 artifact reference cost1 +183.42%, MDD -21.99%, cost2 +169.76%, cost3 +79.27%
```

## Interpretation

No legacy sub-block by itself explains Alpha4.3 performance.

The highest-PnL diagnostic is `no_legacy`, which removes all 23 legacy regime
columns from the Alpha4.3 feature basis. The full legacy block lowers drawdown
but also reduces PnL and trade count. This means the old clean-regime block acted
more like a risk/regularization block than a standalone alpha source.

Sub-block observations:

```text
factor_core       weak standalone alpha
risk_transition   weak standalone alpha on Alpha4.3 basis
semantic_probs    improves drawdown but not enough PnL
cluster_state     high validation score, weak OOS; likely unstable/overfit
all_legacy        better MDD than no_legacy, lower PnL
```

The gap between retrained `all_legacy` and the Alpha4.3 artifact reference also
shows that Alpha4.3's edge is not only the presence of the 23 legacy regime
features. It likely comes from the specific parent artifact, runner/runtime
selection coupling, and the Alpha4.2 training path.

## Verdict

Do not try to recover Alpha4.3 by simply copying legacy regime feature groups
into Alpha5.

Next useful tests:

```text
1. Alpha4.3 artifact-level ablation: keep the original parent fixed and mask legacy groups at inference.
2. Runner/runtime coupling ablation: evaluate Alpha4.3 parent with and without the selected V21.2 jackpot runner.
3. New Regime4 risk regularizer: use current Regime4 risk/transition as a sizing or veto layer, not as extra parent features.
```
