# Alpha3 Exit Front-Run Layer Contract

Last updated: 2026-05-14 KST

## Scope

- Alias: `alpha3_exit_front_run_layer`
- Full name: `alpha3_exit_front_run_layer_20260514`
- Status: `shadow_promote_candidate`
- Purpose: Improve Alpha3 PnL and MDD by changing only the model-wide reduce-only exit placement layer.
- Backtest entrypoint: `scripts/eval_alpha3_exit_front_run_layer_20260514.py`

The Alpha3 decision stack is frozen. This candidate does not change the HGB parent, teacher gate, V21.2 jackpot runner, V27 deep scout, V31 exit reasons, position sizing, or entry placement.

## Selected Exit Placement

2025Q4 validation selected:

```json
{
  "name": "entry2_exit4_pen0_fee20",
  "anchor": "next_open",
  "entry_offset_bps": 2.0,
  "exit_offset_bps": 4.0,
  "penetration_bps": 0.0,
  "maker_fee_mult": 0.20,
  "entry_miss": "market_fallback",
  "exit_miss": "market_fallback"
}
```

Interpretation:

- Entry keeps Alpha3's next-open post-only limit, 2 bps passive offset, and market fallback, but this experiment uses the selected config's shared `penetration_bps=0.0` touch proxy rather than Alpha3 baseline `penetration_bps=0.5`.
- Exit becomes independently selected: reduce-only next-open post-only limit, 4 bps passive offset, zero penetration requirement, market fallback.
- The layer applies to all Alpha3 exits: parent-owned V21.2 positions and Deep Scout positions.

## OOS Metrics

2026 fixed OOS after 2025Q4 selection:

| Variant | Cost1 PnL | Cost1 MDD | Cost2 PnL | Cost2 MDD | Cost3 PnL | Cost3 MDD |
|---|---:|---:|---:|---:|---:|---:|
| Taker control | +354.53% | -32.45% | +23.13% | -46.95% | +10.80% | -42.65% |
| Old synthetic L2 control | +718.70% | -26.66% | +443.82% | -28.97% | +360.15% | -29.69% |
| Alpha3 baseline | +747.76% | -27.37% | +510.83% | -29.68% | +436.68% | -30.39% |
| Alpha3 exit front-run layer | +792.42% | -26.91% | +540.25% | -29.19% | +452.86% | -29.91% |

Delta versus Alpha3 baseline:

- Cost1 PnL: `+44.65 pp`
- Cost1 MDD: `+0.46 pp` improvement
- Cost2 PnL: `+29.42 pp`
- Cost3 PnL: `+16.18 pp`

Cost1 route counts:

```json
{
  "signal_immediate_maker_limit": 319,
  "exit_market_fallback_after_limit_miss": 36,
  "entry_market_fallback_after_limit_miss": 44
}
```

## Artifacts

- Backtest script: `scripts/eval_alpha3_exit_front_run_layer_20260514.py`
- Summary report: `data/ensemble/reports/alpha3_exit_front_run_layer_20260514_summary.json`
- Red Team audit: `data/ensemble/reports/alpha3_exit_front_run_layer_20260514_audit.json`
- Selection grid: `data/ensemble/reports/alpha3_exit_front_run_layer_20260514_grid.csv`

## Audit Status

- Verdict: `shadow_promote_candidate`
- Selection uses 2026: `false`
- Blocking issues: none identified in this audit.

Known promotion risks:

- `signal_limit_fill_uses_5m_high_low_touch_proxy_not_queue_fill`
- `real_l2_queue_and_partial_fill_require_forward_shadow_validation`
- `exit_offset_changes_execution_price_model_and_must_be_live_shadow_validated`

This candidate should not replace live Alpha3 until real orderbook snapshots confirm that the larger passive exit offset does not create unacceptable queue-position, partial-fill, or fallback behavior.
