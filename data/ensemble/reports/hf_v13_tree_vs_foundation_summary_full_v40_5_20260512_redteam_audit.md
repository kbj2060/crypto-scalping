# Red Team Audit: `hf_v13_tree_vs_foundation_summary_full_v40_5_20260512`

## Verdict

- `lookahead/data leakage`: **not found**
- `experiment integrity`: **pass after fix**
- `deployment readiness`: **conditional pass**

## What Changed

The original blocker was real: the full-parent HGB hyperparameter grid was defined but not applied during candidate training.

That path is now fixed:
- tuned full-parent training helper: [compare_hf_v13_tree_vs_foundation_summary_full_v40_5.py](/home/llewyn/crypto-scalping/scripts/compare_hf_v13_tree_vs_foundation_summary_full_v40_5.py:134)
- candidate loop now calls the tuned path: [compare_hf_v13_tree_vs_foundation_summary_full_v40_5.py](/home/llewyn/crypto-scalping/scripts/compare_hf_v13_tree_vs_foundation_summary_full_v40_5.py:505)
- selected bundle now persists the applied settings under `tuned_tree_hparams`

Verification after rerun:
- grid now contains multiple distinct `tree_name` values in [hf_v13_tree_vs_foundation_summary_full_v40_5_20260512_grid.csv](/home/llewyn/crypto-scalping/data/ensemble/reports/hf_v13_tree_vs_foundation_summary_full_v40_5_20260512_grid.csv)
- selected bundle records non-default tuned parameters in [summary_full_bundle.pkl](/home/llewyn/crypto-scalping/data/ensemble/supervised/hf_v13_tree_vs_foundation_summary_full_v40_5_20260512/summary_full_bundle.pkl)
- selected config changed to `macro4_micro4_dropmicro` + `mi360_lr0p02_leaf63_l20p5_cw0p25`, which confirms the search is no longer a no-op

## Checks Performed

1. Window direction
   - embedding windows are backward-only at [train_eval_hf_v13_multitrack_foundation_parent_v40.py](/home/llewyn/crypto-scalping/scripts/train_eval_hf_v13_multitrack_foundation_parent_v40.py:189)
   - Chronos uses windows ending at the current index at [train_eval_hf_v13_multitrack_foundation_parent_v40.py](/home/llewyn/crypto-scalping/scripts/train_eval_hf_v13_multitrack_foundation_parent_v40.py:415)
   - Kairos uses windows ending at the current index at [train_eval_hf_v13_multitrack_foundation_parent_v40.py](/home/llewyn/crypto-scalping/scripts/train_eval_hf_v13_multitrack_foundation_parent_v40.py:441)

2. Split discipline
   - train/validation split is fixed before selection at [compare_hf_v13_tree_vs_foundation_summary_full_v40_5.py](/home/llewyn/crypto-scalping/scripts/compare_hf_v13_tree_vs_foundation_summary_full_v40_5.py:429)
   - OOS 2026 is held out until after selection and scored only after the winner is chosen at [compare_hf_v13_tree_vs_foundation_summary_full_v40_5.py](/home/llewyn/crypto-scalping/scripts/compare_hf_v13_tree_vs_foundation_summary_full_v40_5.py:548)

3. Train-only factor fitting
   - PCA is fit on train embeddings only, then transformed on validation/OOS at [compare_hf_v13_tree_vs_foundation_summary_full_v40_5.py](/home/llewyn/crypto-scalping/scripts/compare_hf_v13_tree_vs_foundation_summary_full_v40_5.py:485)

4. Tuned full-head application
   - candidate bundles now train all heads through `train_policy_with_hparams(...)` at [compare_hf_v13_tree_vs_foundation_summary_full_v40_5.py](/home/llewyn/crypto-scalping/scripts/compare_hf_v13_tree_vs_foundation_summary_full_v40_5.py:508)
   - action/quality and bucket heads all inherit the selected HGB settings at [compare_hf_v13_tree_vs_foundation_summary_full_v40_5.py](/home/llewyn/crypto-scalping/scripts/compare_hf_v13_tree_vs_foundation_summary_full_v40_5.py:149)

## Remaining Warnings

1. Embedding cache invalidation is still weak.
   - macro/micro caches are reused from fixed filenames at [compare_hf_v13_tree_vs_foundation_summary_full_v40_5.py](/home/llewyn/crypto-scalping/scripts/compare_hf_v13_tree_vs_foundation_summary_full_v40_5.py:459)
   - extraction functions trust an existing cache file immediately at [train_eval_hf_v13_multitrack_foundation_parent_v40.py](/home/llewyn/crypto-scalping/scripts/train_eval_hf_v13_multitrack_foundation_parent_v40.py:409) and [train_eval_hf_v13_multitrack_foundation_parent_v40.py](/home/llewyn/crypto-scalping/scripts/train_eval_hf_v13_multitrack_foundation_parent_v40.py:433)
   - this rerun manually cleared cache before execution, so the current result is clean, but the code path still needs data-dependent cache keys

2. Validation/OOS context is conservative, not leaky.
   - validation and OOS windows are rebuilt from split-local frames, so the first bars of each split do not see prior-split history
   - that reduces realism slightly, but it does not create future leakage

3. Performance warnings are noisy but not integrity issues.
   - repeated `DataFrame is highly fragmented` warnings came from feature assembly in [fully_learned_governor_policy.py](/home/llewyn/crypto-scalping/ensemble/fully_learned_governor_policy.py:187)
   - this is a performance cleanup item, not a red-team blocker

## Rerun Result Snapshot

- selected summary: `macro4_micro4_dropmicro`
- selected tree hp: `max_iter=360`, `learning_rate=0.02`, `max_leaf_nodes=63`, `l2_regularization=0.5`, `cash_weight=0.25`
- validation:
  - baseline cost1 `-8.12%`
  - summary_full cost1 `+52.23%`
- 2026 OOS:
  - baseline cost1 `+13.26%`, MDD `-34.79%`
  - summary_full cost1 `+93.53%`, MDD `-33.37%`
  - summary_full cost2 `+61.79%`
  - summary_full cost3 `+53.29%`

## Red Team Conclusion

After fixing the full-head tuning path and rerunning from cleared embedding caches, I do **not** see direct lookahead or train/OOS contamination in the encoded tree flow. The previous blocking integrity issue is resolved.

The current model now passes red-team integrity review **conditionally**: acceptable for further shadow/live-paper use, but cache-key hardening should be done before treating this path as a stable production baseline.
