# Alpha3 Deep Exit Oracle Contract

Last updated: 2026-05-14 KST

## Scope

- Alias: `alpha3_deep_exit_oracle`
- Full name: `alpha3_deep_exit_oracle_20260514`
- Status: `shadow_candidate`
- Purpose: Add a deep-learning model-wide exit layer to Alpha3 without changing the Alpha3 entry stack or exit reasons.
- Backtest entrypoint: `scripts/eval_alpha3_deep_exit_oracle_20260514.py`

The Alpha3 decision stack is frozen:

- HGB parent: `hf_v13_clean_regime_margin110_20260511`
- Teacher gate: `noflip_c0.56_parent_scale1.10`
- V21.2 jackpot runner: preserved
- V27 deep scout: preserved
- V31 exit reasons: preserved
- Entry execution: Alpha3 baseline `next_open`, `2 bps`, `penetration=0.5`, market fallback

## Model Design

The oracle is a compact gated MLP trained on Alpha3 exit events:

- Input: current 93-feature frame plus position state, unrealized PnL, MFE/MAE, hold time, active owner, parent decision output, V27 long/short utilities, and volatility anchor.
- Output arms:
  - `baseline_exit2_pen05`
  - `exit0_pen0`
  - `exit1_pen0`
  - `exit2_pen0`
  - `exit3_pen0`
  - `exit4_pen0`
- Label: best counterfactual reduce-only exit placement under the same Alpha3 exit event.
- Training window: `2025-07-01..2025-09-30`
- Selection window: `2025-10-01..2025-12-31`
- OOS window: fixed 2026 only after selection

The selected policy is:

```json
{
  "name": "deep_conf0.35_fallback_exit4_pen0",
  "mode": "deep",
  "min_confidence": 0.35,
  "fallback_arm": "exit4_pen0"
}
```

## Event Label Distribution

```json
{
  "train_events": 254,
  "val_events": 359,
  "arms": [
    "baseline_exit2_pen05",
    "exit0_pen0",
    "exit1_pen0",
    "exit2_pen0",
    "exit3_pen0",
    "exit4_pen0"
  ],
  "train_label_counts": [5, 23, 8, 4, 11, 203],
  "val_label_counts": [9, 22, 19, 9, 16, 284]
}
```

Interpretation: the data strongly prefers `exit4_pen0`. The deep model therefore mostly defers to the 4 bps exit-placement fallback. Current evidence supports the exit-placement rule more strongly than a complex early-stopping oracle.

## OOS Metrics

2026 fixed OOS:

| Variant | Cost1 PnL | Cost1 MDD | Cost2 PnL | Cost2 MDD | Cost3 PnL | Cost3 MDD |
|---|---:|---:|---:|---:|---:|---:|
| Taker control | +354.53% | -32.45% | +23.13% | -46.95% | +10.80% | -42.65% |
| Old synthetic L2 control | +718.70% | -26.66% | +443.82% | -28.97% | +360.15% | -29.69% |
| Alpha3 baseline | +747.76% | -27.37% | +510.83% | -29.68% | +436.68% | -30.39% |
| Fixed front-run exit4 | +789.50% | -26.91% | +538.39% | -29.19% | +452.47% | -29.91% |
| Deep exit oracle | +789.50% | -26.91% | +538.39% | -29.19% | +452.47% | -29.91% |

Delta versus Alpha3 baseline:

- Cost1 PnL: `+41.74 pp`
- Cost1 MDD: `+0.46 pp` improvement
- Cost2 PnL: `+27.56 pp`
- Cost3 PnL: `+15.79 pp`

## Artifacts

- Model checkpoint: `data/ensemble/supervised/alpha3_deep_exit_oracle_20260514/deep_exit_oracle.pt`
- Backtest script: `scripts/eval_alpha3_deep_exit_oracle_20260514.py`
- Summary report: `data/ensemble/reports/alpha3_deep_exit_oracle_20260514_summary.json`
- Audit: `data/ensemble/reports/alpha3_deep_exit_oracle_20260514_audit.json`
- Selection grid: `data/ensemble/reports/alpha3_deep_exit_oracle_20260514_grid.csv`
- Event summary: `data/ensemble/reports/alpha3_deep_exit_oracle_20260514_events.json`

## Audit Status

- Verdict: `shadow_candidate`
- Selection uses 2026: `false`
- Blocking issues: none identified in this audit.

Known promotion risks:

- `signal_limit_fill_uses_5m_high_low_touch_proxy_not_queue_fill`
- `real_l2_queue_and_partial_fill_require_forward_shadow_validation`
- `deep_oracle_labels_are_counterfactual_to_ohlc_touch_model_not_real_queue_fills`

## Standing Rule

Do not promote the deep oracle as a live replacement until real L2 queue and partial-fill data confirms that `exit4_pen0` improves actual reduce-only exit fills. The model is currently a shadow execution-layer candidate, not proof that a neural early-stopping layer should own Alpha3 exits.
