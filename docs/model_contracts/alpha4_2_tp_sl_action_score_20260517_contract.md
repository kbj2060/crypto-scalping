# Alpha4.2 TP/SL Action Score Contract - 2026-05-17

## Goal

Inject TP/SL path information into the parent as one scalar feature for
`hold/long/short` selection, not as a post-entry reject gate and not from each
candidate entry's selected TP/SL/max-hold.

## Input State

The base train/eval CSVs are assumed to have passed the red-team feature audit
and contamination blockers before this experiment starts.

- Base train: `tmp/causal_regen_20260516/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv`
- Base eval: `tmp/causal_regen_20260516/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv`
- Augmented train: `tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/trade_candidates_2025_patchtst__tide__dlinear.csv`
- Augmented eval: `tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/trade_candidates_2026_patchtst__tide__dlinear.csv`

## Feature Contract

Feature name: `tp_sl_action_score`

Semantics:

- `> 0`: long path edge
- `< 0`: short path edge
- `= 0`: hold/no usable TP-over-SL edge

Generation:

- Fixed path labels, independent of candidate TP/SL/max-hold.
- Entry reference: next-bar open.
- Horizon: 48 bars.
- Fixed TP: 1.8%.
- Fixed SL: 1.0%.
- Same-bar TP/SL tie: SL wins conservatively.
- 2025 train feature: monthly walk-forward OOF.
- 2026 eval feature: predictor trained on all 2025 labels only.

Feature audit:

- Audit: `tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/tp_sl_path_edge_feature_audit.json`
- Train mean/std: -0.0238 / 0.1005
- Train zero-rate: 71.46%
- Eval mean/std: -0.0036 / 0.0531
- Eval zero-rate: 79.02%

## Retrain Command

```bash
venv/bin/python scripts/eval_alpha4_new_features_full_retrain_20260517.py \
  --train-csv tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/trade_candidates_2025_patchtst__tide__dlinear.csv \
  --eval-csv tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/trade_candidates_2026_patchtst__tide__dlinear.csv \
  --out-dir tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/artifacts \
  --report-out tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/alpha4_2_tp_sl_action_score_summary.json \
  --audit-out tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/alpha4_2_tp_sl_action_score_audit.json \
  --grid-out tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/alpha4_2_tp_sl_action_score_grid.csv \
  --only hgb --teacher-epochs 35 --v27-epochs 80 --stride 6
```

## Artifacts

- Summary: `tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/alpha4_2_tp_sl_action_score_summary.json`
- Audit: `tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/alpha4_2_tp_sl_action_score_audit.json`
- Grid: `tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/alpha4_2_tp_sl_action_score_grid.csv`
- Parent: `tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/artifacts/hgb/parent.pkl`
- Teacher: `tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/artifacts/hgb/teacher_gate.pt`
- Runner: `tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/artifacts/hgb/runner.pkl`
- V27-style utility: `tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517/artifacts/alpha4_v27_style_deep_alpha_utility/alpha4_v27_style_deep_alpha_utility.pt`

## Results

Same CSV frozen Alpha3 reference:

| Variant | Cost1 PnL | MDD | Cost2 PnL | Cost3 PnL | Trades | Deep Entries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen Alpha3 reference | +12.28% | -45.13% | +2.53% | +2.34% | 386 | 317 |
| Alpha4.2 full retrain | +58.89% | -24.26% | +33.49% | +2.46% | 190 | 150 |
| Alpha4.2 no-deep ablation | +78.53% | -18.81% | +40.92% | +40.63% | 44 | 0 |
| Alpha4.2 overlay reselect | +86.09% | -22.74% | +31.01% | +19.45% | 178 | 141 |

Selected full-stack runtime:

- Teacher runtime: `noflip_c0.74_parent_scale0.85`
- Runner config: `v21_2_jackpot_runner_0`
- Parent feature count: 84
- `tp_sl_action_score` included in parent artifact: yes

## Interpretation

The hold-aware scalar is materially better than the naive `long_edge - short_edge`
compression. The prior Alpha4.1 version produced negative OOS PnL, while
Alpha4.2 improves both PnL and MDD versus the same-input frozen Alpha3 reference.

The remaining weakness is still the V27-style deep sleeve:

- Full Alpha4.2 deep entries: 150
- Full Alpha4.2 deep stop-loss exits: 104
- Full Alpha4.2 deep take-profit exits: 14

The no-deep ablation is the more robust candidate under cost stress. It produces
lower trade count, better MDD, and much stronger cost3 survival. Future promotion
should prefer `Alpha4.2 no-deep` unless a separately calibrated deep sleeve
passes high-cost stress.
