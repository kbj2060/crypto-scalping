# ZigZag Second-Stage Retrain Audit - 2026-05-31

## Scope

Retrained and tested only the second-stage AI/M7/Regime families whose previous
outputs were trained from action/direction/tradeability labels. This audit is a
comparison and promotion-candidate pass, not an automatic replacement of the
Omega1 pass/fail feature contract.

- Hard label: `zigzag_action`
- Soft labels available for future soft-target training:
  `zigzag_soft_cash`, `zigzag_soft_long`, `zigzag_soft_short`
- Label artifact:
  `tmp/causal_regen_20260516/zigzag_action_labels_20260531`
- Full retrain summary:
  `tmp/causal_regen_20260516/zigzag_second_stage_retrain_all_20260531/zigzag_second_stage_retrain_all_summary.json`

Regime4 and `regime3_pred_*` were excluded from this comparison retrain.

## Runs

All three run groups completed without interruption:

1. M7 HGB action head:
   `tmp/causal_regen_20260516/zigzag_second_stage_retrain_all_20260531/m7_zigzag_action_hgb`
2. AI/M7/Regime family sweep:
   `tmp/causal_regen_20260516/zigzag_second_stage_retrain_all_20260531/family_sweep`
3. PatchTSMixer representation + CatBoost head:
   `tmp/causal_regen_20260516/zigzag_second_stage_retrain_all_20260531/ai_zigzag_patchmix_catboost`

## 2026 OOS Results

| Family | Feature Count | BAcc | OVR AUC | Verdict |
|---|---:|---:|---:|---|
| `ai_zigzag_patchmix_catboost` | 56 | 0.5498 | 0.7751 | PASS |
| `ai_all_legacy` | 26 | 0.5349 | 0.7688 | PASS |
| `all_second_stage_nonp0` | 75 | 0.5329 | 0.7519 | PASS |
| `ai_role_risk_context` | 17 | 0.5325 | 0.7624 | PASS |
| `m7_zigzag_action_hgb` | 141 | 0.5264 | 0.7403 | PASS_WITH_OVERFIT_RISK |
| `regime3_all_context` | 16 | 0.5093 | 0.7425 | CONTEXT_ONLY |
| `regime3_risk_context` | 9 | 0.4942 | 0.7278 | CONTEXT_ONLY |
| `regime3_current_context` | 7 | 0.4925 | 0.7321 | CONTEXT_ONLY |
| `m7_direction_legacy` | 15 | 0.4469 | 0.6686 | FAIL |
| `m7_all_nonp0` | 33 | 0.4456 | 0.6448 | FAIL |
| `ai_direction_legacy` | 9 | 0.4293 | 0.6672 | FAIL |
| `m7_unsup_risk_context` | 10 | 0.3647 | 0.5710 | FAIL |

## 2025 Selection Check

| Family | Feature Count | BAcc | OVR AUC |
|---|---:|---:|---:|
| `ai_zigzag_patchmix_catboost` | 56 | 0.5585 | 0.7758 |
| `ai_all_legacy` | 27 | 0.5426 | 0.7652 |
| `ai_role_risk_context` | 18 | 0.5387 | 0.7599 |
| `m7_zigzag_action_hgb` | 141 | 0.5268 | 0.7339 |
| `all_second_stage_nonp0` | 84 | 0.5135 | 0.7168 |
| `regime3_all_context` | 18 | 0.5085 | 0.7406 |

## Interpretation

- The prior Omega1 pass/fail feature list remains the source of truth. The
  rows below only show action-label-trained families that were relabeled with
  `zigzag_action` for comparison.
- Non-action-label-derived features such as Regime3 current/risk context,
  TiDE risk, Chronos uncertainty, and pass-only M7 risk/quality context are not
  replaced by these ZigZag outputs.
- AI representation/context features respond best to the new ZigZag action
  label. `ai_zigzag_patchmix_catboost` is the strongest single second-stage
  action-context promotion candidate.
- Legacy M7 direction/action features remain weak after relabeling and should
  not be revived as active direction owners.
- `m7_zigzag_action_hgb` has usable OOS metrics but very high train metrics
  (`train_bacc` about 0.93), so it needs regularization/ablation before active
  promotion.
- Regime3 should remain context/veto/sizing support. The ZigZag-trained
  `regime3_*` action heads are comparison outputs only and do not replace the
  original Regime3 current/risk context features.
- `regime3_pred_*` and Regime4 remain excluded from Omega1 active modeling.

## Correct Usage

- Use this audit to decide whether an old action-label-trained family should be
  regenerated under the ZigZag contract.
- Do not add every PASS row here directly into Omega1 runtime.
- Before promotion, each candidate still needs a downstream PnL/MDD/trade-count
  ablation against the existing Omega1 feature contract.
