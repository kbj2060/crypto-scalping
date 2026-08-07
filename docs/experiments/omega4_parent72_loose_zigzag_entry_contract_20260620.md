# Omega4 Parent72 Loose Zigzag Entry Contract - 2026-06-20

## Status

- Proposed experiment id: `omega4_parent72_loose_zigzag_entry_contract_20260620`
- Implementation artifact id currently used by wrappers: `omega1_2_true_3head_tabm_parent72_loose_zigzag_entry_20260620`
- Status: `prepared_not_live_promoted`
- Baseline model: `omega1_2_true_3head_tabm_20260603_final_tp_sl_on_e28_exit30k_q080`
- Baseline artifact dir: `tmp/causal_regen_20260516/omega1_2_true_3head_tabm_20260603_final_tp_sl_on_e28_exit30k_q080`
- Baseline manifest: `data/ensemble/supervised/omega1_2_true_3head_tabm_20260603_final_tp_sl_on_e28_exit30k_q080/baseline_manifest.json`

This experiment changes only the entry-head training labels. It must not use
`omega1_2_1_aggressive_compensated_scale200_cap090` as the baseline model,
because that artifact is a parent prediction plus risk-transform replay, not a
new entry model.

## Label Contract

- Label family: parent72 loose zigzag action labels
- Label dir: `tmp/causal_regen_20260516/zigzag_action_labels_parent72_loose_20260620`
- Label files:
  - `zigzag_action_labels_2024.csv`
  - `zigzag_action_labels_2025.csv`
  - `zigzag_action_labels_2026.csv`
- Label audit: `tmp/causal_regen_20260516/zigzag_action_labels_parent72_loose_20260620/zigzag_action_label_audit.json`
- Label column: `zigzag_action`
- Classes: `0=CASH`, `1=LONG`, `2=SHORT`
- Soft label columns: `zigzag_soft_cash`, `zigzag_soft_long`, `zigzag_soft_short`

The label generator keeps the existing parent72 confirmed-pivot-segment and
MAE/MFE risk-adjusted soft-label philosophy from
`zigzag_action_labels_20260531`, with the following loose parameters:

- `min_wave_bars = 6`
- `transition_buffer = 1`
- `atr_window = 14`
- `atr_multiplier = 1.0`
- `zigzag_reversal_pct = 0.009`
- `mae_penalty = 1.10`
- `softmax_temperature = 1.90`
- `min_risk_floor = 0.001`

## Train/Eval Protocol

Reuse the existing 3-head TabM parent training implementation:

- Base trainer: `scripts/train_eval_omega1_2_tabm_3head_20260603.py`
- Direction label loader patched by wrapper:
  `scripts/train_omega1_direction_head_direction_only_20260602.py`
- Smoke wrapper:
  `scripts/train_eval_omega1_2_tabm_3head_parent72_loose_zigzag_entry_20260620.py`
- Full wrapper:
  `scripts/train_eval_omega1_2_tabm_3head_parent72_loose_zigzag_entry_fulltrain_20260620.py`

Smoke command, safe for wiring verification:

```bash
python scripts/train_eval_omega1_2_tabm_3head_parent72_loose_zigzag_entry_20260620.py
```

Full-train command, run only after explicit approval:

```bash
python scripts/train_eval_omega1_2_tabm_3head_parent72_loose_zigzag_entry_fulltrain_20260620.py
```

Expected output roots:

- Smoke:
  `tmp/causal_regen_20260516/omega1_2_true_3head_tabm_parent72_loose_zigzag_entry_20260620_smoke_e4_train30k_exit12k`
- Full:
  `tmp/causal_regen_20260516/omega1_2_true_3head_tabm_parent72_loose_zigzag_entry_20260620_e28_fulltrain_exit30k`

Primary comparison should use `no_exit_head` metrics first, because the stated
experiment is an entry-label swap. Exit-head variants may be inspected as
secondary diagnostics, but should not redefine the entry experiment.

## Risk Accounting

Use the baseline fixed final TP/SL template and Cost3 replay accounting from the
baseline manifest:

- `take_profit = 0.026`
- `stop_loss = 0.014`
- `notional_exposure = 0.405`
- `leverage = 2.0`
- `max_hold_bars = 0`
- `cooldown_bars = 0`
- `cost_multiplier = 3.0`

For any later Omega4 risk-sizing extension, keep margin, leverage, and notional
separate:

- `notional = margin_fraction * leverage`
- `margin_fraction = notional / leverage`
- `PnL = price_move * notional`
- `take_profit = tp_price_move * notional`
- `stop_loss = sl_price_move * notional`

Do not multiply TP/SL price-move lines by leverage again after notional is
derived.

## Known Reference Results

Current reference results, already produced outside this preparation step:

- Loose smoke no-exit-head: validation `+20.41%`, OOS `+20.66%`
- Loose full no-exit-head: validation `-10.18%`, OOS `+9.33%`
- Loose smoke plus aggressive scale200 cap090 replay: validation `+42.27%`, OOS `+43.43%`
- Loose full plus aggressive replay: validation `-20.69%`, OOS `+18.16%`
- Existing aggressive runtime baseline remains stronger: validation `+100.54%`, OOS `+72.76%`

These replay results are diagnostics only. They do not change the Omega4
baseline model definition.

## Promotion Gate

Omega4 cannot be promoted unless all of the following pass:

- Reproduce the raw parent baseline metrics from
  `omega1_2_true_3head_tabm_20260603_final_tp_sl_on_e28_exit30k_q080`.
- Compare validation and OOS against the raw parent baseline using the same
  Cost3 accounting and fixed TP/SL template.
- Beat the raw parent baseline on validation and OOS without relying on an
  aggressive risk-transform replay.
- Show no material MDD degradation that is hidden by OOS-only PnL improvement.
- Preserve fail-fast feature and artifact contracts: no legacy aliases, fallback
  prefixes, implicit renames, or compatibility shims on active/candidate paths.
- Keep `trading_bot.py` and live wiring unchanged until runtime-native parity
  and current live feature-contract validation are explicitly completed.

## Implementation Decision

No new training code is required for this preparation step. The existing smoke
and full wrappers already apply the loose parent72 zigzag label directory to the
3-head TabM trainer and write disjoint experiment artifacts. A future cleanup may
add an Omega4-named wrapper, but it should produce a new disjoint output path and
must not alias old artifacts silently.
