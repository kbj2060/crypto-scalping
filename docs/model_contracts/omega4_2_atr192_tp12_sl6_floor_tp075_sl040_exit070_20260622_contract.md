# Omega4.2 ATR192 Safety SLTP Contract - 2026-06-22

## Status

- Model id: `omega4_2_atr192_tp12_sl6_floor_tp075_sl040_exit070_20260622`
- Alias: `omega4.2_atr_safety_exit070`
- Status: `current_omega_research_baseline_not_live_wired`
- Manifest: `data/ensemble/supervised/omega4_2_atr192_tp12_sl6_floor_tp075_sl040_exit070_20260622/candidate_manifest.json`
- Runtime contract: `tmp/causal_regen_20260516/omega4_2_atr192_tp12_sl6_floor_tp075_sl040_exit070_20260622/runtime_contract.json`
- Source evaluation: `tmp/causal_regen_20260516/omega4_1_atr_safety_sltp_20260622_q070_exit070_wider_floor_grid3/report.json`

This promotion does not introduce new neural weights. Omega4.2 is the Omega4.1
exit-threshold-0.70 baseline bundle plus the selected ATR safety SLTP runtime
contract.

## Lineage

- Weight bundle:
  `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_smoke_loose_entry_loose_quality_terminal_giveback_exit_e2_train15k_exit15k_q070/true_3head_tabm_bundle.pt`
- Weight-training report:
  `tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_smoke_loose_entry_loose_quality_terminal_giveback_exit_e2_train15k_exit15k_q070/report.json`
- Baseline model recorded by that report:
  `omega1_2_true_3head_tabm_20260603_final_tp_sl_on_e28_exit30k_q080`
- Evaluation script:
  `scripts/eval_omega4_1_atr_safety_sltp_20260622.py`

## Inherited Training Data

Omega4.2 inherits the Omega4.1 bundle training data:

- Direction labels:
  `tmp/causal_regen_20260516/zigzag_action_labels_parent72_loose_20260620`
- Quality mode: `same_as_direction`
- Quality target rule:
  active action required, `side_soft_min = 0.7`, `edge_min = 0.0`,
  `mae_max = 0.01`, `mfe_mae_min = 1.5`; otherwise CASH.
- Exit label mode: `entry_label_terminal_giveback`
- Exit label hold offsets:
  `1, 2, 3, 6, 12, 24, 48, 96, 192, 384`
- Exit label positive rate in the training report: `7.76%`
- Input contract: `172` base features plus `13` position features.

No label diagnostic columns are model inputs. Forbidden feature prefixes and
tokens are inherited from the Omega4.1 report, including `clean_regime4_`,
`regime4_pred_`, `regime3_pred_`, `teacher_`, `target`, `future`, `label`,
`pnl`, `zigzag`, `wave3`, and `tp_sl_action_score`.

## Runtime Contract

- Quality threshold: `0.70`
- Exit-head threshold: `0.70`
- Max hold bars: `0`
- Cooldown bars: `0`
- Cost multiplier: `3.0`
- SLTP policy: entry-time ATR percent safety barriers.
- ATR definition:
  rolling mean true range divided by close. True range is the maximum of
  high-low, abs(high-prev_close), and abs(low-prev_close).

Selected ATR safety parameters:

- ATR window: `192` bars
- TP multiple: `12.0`
- SL multiple: `6.0`
- TP floor: `0.075`
- SL floor: `0.040`
- TP cap: `0.22`
- SL cap: `0.12`

Barrier formulas:

```text
tp_price_move = clip(max(0.075, atr_pct_192 * 12.0), 0.0, 0.22)
sl_price_move = clip(max(0.040, atr_pct_192 * 6.0), 0.0, 0.12)
```

SLTP hit checks compare raw directional `price_move` to those price-move
barriers. They do not divide by notional and do not multiply by leverage.
Position PnL remains:

```text
PnL = realized_price_move * notional
notional = margin_fraction * leverage
```

## Selected Metrics

Selected variant: `atr192_tp12_sl6`, chosen by validation first.

| Split | PnL | MDD | WR | Trades | Long | Short |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | `+16.02%` | `-7.11%` | `67.06%` | `85` | `22` | `63` |
| OOS | `+13.32%` | `-4.38%` | `66.15%` | `65` | `20` | `45` |

Exit reasons:

| Split | exit_head | stop_loss | take_profit | forced_end |
| --- | ---: | ---: | ---: | ---: |
| Validation | `62` | `18` | `4` | `1` |
| OOS | `54` | `8` | `2` | `1` |

Reference Omega4.1 exit0.70 baseline:

| Split | PnL | MDD | WR | Trades |
| --- | ---: | ---: | ---: | ---: |
| Validation | `+3.28%` | `-7.82%` | `67.11%` | `149` |
| OOS | `+7.51%` | `-5.61%` | `63.00%` | `100` |

## Selection Notes

- `atr192_tp12_sl6` was the best validation row in the wider-floor grid.
- `atr192_tp16_sl8` had higher OOS PnL, but materially weaker validation PnL
  and validation MDD, so it is not promoted.
- In the observed validation and OOS splits, p50 and p90 TP/SL barriers stayed
  at the floor values. ATR still widens barriers when `atr_pct * multiple`
  exceeds the floor.

## Live Wiring

No live runtime wiring was changed for this promotion. Before using Omega4.2 for
real exchange trading, run runtime-native parity, current live feature-contract
validation, and shadow or paper smoke. Contract mismatches must fail fast; do
not add aliases, fallback prefixes, or compatibility shims on the active path.
