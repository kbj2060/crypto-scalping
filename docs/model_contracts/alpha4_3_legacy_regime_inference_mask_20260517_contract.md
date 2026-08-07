# Alpha4.3 Legacy Regime Inference Mask

Date: 2026-05-17

## Purpose

Isolate which legacy clean-regime feature groups contribute to the already
trained Alpha4.3 artifact.

This differs from the retrain ablation. Here the original Alpha4.3 parent,
runner, runtime, and execution path are fixed. Only selected legacy feature
groups are replaced with their 2025 train medians at inference.

## Fixed Artifacts

```text
parent: /home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/artifacts/hgb/parent.pkl
runner: /home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/teacher_ablation_artifacts/parent_direct_scaled_no_teacher_runner.pkl
runtime: parent_direct_scale0.85, max_notional 2.75
execution: corrected Alpha3 limit-close
teacher: disabled
deep scout: disabled
```

Parent feature contract:

```text
feature count                         84
legacy clean_regime_2024_unsup_v4_*   23
tp_sl_action_score                    enabled
```

Mask policy:

```text
replace selected columns with 2025 train median
```

## Artifacts

```text
/home/llewyn/crypto-scalping/scripts/ablate_alpha4_3_legacy_regime_inference_mask_20260517.py
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha4_3_legacy_regime_inference_mask_20260517/alpha4_3_legacy_regime_inference_mask_summary.json
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha4_3_legacy_regime_inference_mask_20260517/alpha4_3_legacy_regime_inference_mask_results.csv
```

## Results

2026 OOS:

| Mask Group | Masked Cols | Cost1 PnL | Cost1 Delta | MDD | Cost2 Delta | Cost3 Delta |
|---|---:|---:|---:|---:|---:|---:|
| `none` | 0 | +183.42% | +0.00% | -21.99% | +0.00% | +0.00% |
| `factor_core` | 6 | +144.11% | -39.30% | -26.46% | -31.11% | +14.40% |
| `risk_transition` | 2 | +96.02% | -87.40% | -23.13% | -69.01% | +25.54% |
| `semantic_probs` | 7 | +69.41% | -114.01% | -25.74% | -85.89% | +21.15% |
| `cluster_state` | 8 | +193.78% | +10.37% | -22.86% | +6.97% | +54.12% |
| `all_legacy` | 23 | +122.29% | -61.13% | -23.41% | -54.88% | +62.41% |

## Interpretation

For the fixed Alpha4.3 artifact, the strongest positive contributors to
cost1/cost2 are:

```text
semantic_probs
risk_transition
factor_core
```

Masking these groups sharply reduces cost1/cost2 PnL.

`cluster_state` is different: masking it improves PnL, especially cost3. This
suggests the old cluster/state-code features are harmful or overfit in the fixed
artifact, even though they may have helped the artifact's original selection
path.

Masking all 23 legacy columns is not equivalent to removing only harmful
features. It removes both positive and negative groups. The net effect is lower
cost1/cost2 but higher cost3, which means the legacy block contains mixed
signals.

## Practical Conclusion

Do not copy the whole legacy block into Alpha5.

The useful legacy information to translate into Regime4 is:

```text
semantic regime probabilities/confidence/entropy
risk_off_prob
transition_risk
factor_core
```

The old cluster/state-code features should not be ported as parent features
without a separate guard. If reused, they belong in a diagnostic or risk veto
experiment, not in the base parent input.
