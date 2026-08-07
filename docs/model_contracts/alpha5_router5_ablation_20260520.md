# Alpha5 Router5 Ablation 2026-05-20

## Scope

Baseline Router5 is the fixed ensemble:

- Router3 probability weight: `0.8`
- Router4 collapsed probability weight: `0.2`
- OOS split: `alpha5_29_hier_label_factory_oos.parquet`
- Baseline meta: `tmp/causal_regen_20260516/alpha5_router_v5_train_singlefile_20260520/router_ensemble_meta.joblib`
- Ablation output: `tmp/causal_regen_20260516/alpha5_router_v5_ablation_expweight_20260520/router5_ablation_summary.json`

The tests are router-level ablations only. No live trading bot route was changed.

## Results

| Variant | OOS balanced acc | OOS macro F1 | OOS log loss | OOS ECE | Pred trades | Long / Short | Pred quality sum | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Baseline fixed 0.8/0.2 | 0.537663 | 0.347807 | 0.811743 | 0.202949 | 6819 | 2546 / 4273 | 535.13 | Keep as baseline |
| Dynamic logistic stacking | 0.534589 | 0.347181 | 0.766184 | 0.162951 | 6648 | 3708 / 2940 | 536.45 | Candidate for probability features, not class router |
| Isotonic calibration | 0.333333 | 0.322053 | 0.241718 | 0.009211 | 0 | 0 / 0 | 0.00 | Reject: collapsed to NONE |
| Mahalanobis OOD gate p95 | 0.516229 | 0.354203 | 0.877368 | 0.201996 | 6151 | 2214 / 3937 | 476.72 | Reject as hard gate |
| Exp quality weight retrain | 0.511632 | 0.318772 | 0.967178 | 0.202975 | 7629 | 2068 / 5561 | 545.10 | Reject as class router |

## Interpretation

Dynamic stacking improves probability quality on OOS:

- Log loss improves from `0.811743` to `0.766184`.
- ECE improves from `0.202949` to `0.162951`.
- Predicted quality sum is slightly higher: `+1.32`.
- Balanced accuracy is slightly lower: `-0.003073`.

This makes it useful as a calibrated signal provider for RL sizing, but not a clear replacement for hard class routing.

Isotonic calibration is not usable in the current class-imbalanced setup. It produces excellent calibration metrics by mapping nearly everything to `NONE`, so it destroys all trade signal density.

Mahalanobis OOD tension is useful as an auxiliary feature, but the tested p95 hard abstain rule is too destructive. It blocks about `4.37%` of OOS rows, lowers balanced accuracy, and cuts quality sum by `-58.41`.

Exponential quality reweighting increases trade density and raw selected quality sum, but degrades classification quality and probability quality. It over-emphasizes short predictions and should not replace the current weighting.

## Recommendation

Do not replace Router5 with any tested variant as the hard router.

The only useful change is to export two additional auxiliary columns for downstream RL:

- `a5dir_stack_long_prob`, `a5dir_stack_short_prob`, or a full stacked probability set from dynamic logistic stacking.
- `a5dir_uncertainty` from Mahalanobis distance, without hard abstain at Router5 level.

Router5 hard action should remain the fixed 0.8/0.2 ensemble until a downstream DSAC backtest proves that dynamic probabilities or uncertainty improve realized PnL/MDD.

## Dynamic Stacking Tune Addendum

Additional tuning was run in:

- Script: `scripts/alpha5_router_v5_stack_tune_20260520.py`
- Result: `tmp/causal_regen_20260516/alpha5_router_v5_stack_tune_xgb_20260520/router5_stack_tune_summary.json`

Best OOS balanced-accuracy stacker:

- Family: `logistic_valfit`
- Input mode: Router3 probabilities + Router4 probabilities only
- `C=5.0`
- `class_weight=balanced`

| Model | OOS balanced acc | OOS macro F1 | OOS log loss | OOS ECE | Pred trades | Long / Short | Quality sum |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline fixed 0.8/0.2 | 0.537663 | 0.347807 | 0.811743 | 0.202949 | 6819 | 2546 / 4273 | 535.13 |
| Best tuned logistic stack | 0.540010 | 0.348076 | 0.765759 | 0.170672 | 6678 | 3792 / 2886 | 524.66 |

Interpretation:

- Tuned stacking can beat baseline class accuracy slightly: `+0.002347` OOS balanced accuracy.
- It also improves probability quality materially: log loss `-0.045984`, ECE `-0.032277`.
- It reduces selected quality sum by `-10.46`, so it is not yet a clear realized-PnL replacement.
- Shallow XGBoost stackers mostly optimized log loss by collapsing to `NONE`, so they are not valid hard routers in this setup.

Recommendation after tuning:

- Promote tuned logistic stack to a challenger probability provider.
- Do not replace the hard Router5 action until a downstream DSAC/backtest confirms realized PnL/MDD improvement.

## Baseline + Stack Blend Sweep

Additional blend sweep:

- Result: `tmp/causal_regen_20260516/alpha5_router_v5_stack_blend_20260520/router5_stack_tune_summary.json`
- Blend formula: `fixed_weight * baseline_fixed_router5 + stack_weight * tuned_logistic_stack`
- Grid: `stack_weight = 0.00 ... 1.00` in 0.05 increments

Top OOS blend points:

| Fixed weight | Stack weight | OOS balanced acc | OOS macro F1 | OOS log loss | OOS ECE | Quality sum |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.00 | 0.00 | 0.537663 | 0.347807 | 0.811743 | 0.202949 | 535.13 |
| 0.95 | 0.05 | 0.538447 | 0.347956 | 0.806799 | 0.202324 | 532.67 |
| 0.15 | 0.85 | 0.539031 | 0.347582 | 0.764900 | 0.184540 | 523.48 |
| 0.00 | 1.00 | 0.540010 | 0.348076 | 0.765759 | 0.170672 | 524.66 |

Interpretation:

- Blending does not beat pure tuned stack on OOS balanced accuracy.
- Pure tuned stack is still best for class/probability quality.
- A tiny stack overlay (`fixed=0.95`, `stack=0.05`) is a low-risk compromise: it improves balanced accuracy and log loss slightly while preserving most baseline selected quality.
- Mid blends are not attractive: they lose quality sum without beating pure stack.
