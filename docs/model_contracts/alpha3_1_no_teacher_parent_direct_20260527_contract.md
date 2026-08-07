# Alpha3.1 No-Teacher Parent Direct Contract (2026-05-27)

## Alias

- `alpha3.1`
- `alpha3.1_no_teacher_parent_direct`

## Purpose

`alpha3.1` is the next Alpha3 sub-version candidate created from the Alpha3
state24-v2 stack by removing the Teacher decision layer from the active decision
path.

The selected trading path is `no_teacher_parent_direct`: parent decisions are
passed directly into the existing execution/exit stack without Teacher
confidence gating, Teacher no-flip filtering, or Teacher notional scaling.

## Base Model

- Base family: `alpha3`
- Base candidate: `alpha3_regime4_state24_v2_full_retrain_20260526`
- Base report:
  `data/ensemble/reports/alpha3_regime4_state24_v2_full_retrain_20260526_summary.json`
- Ablation report:
  `data/ensemble/reports/alpha3_teacher_layer_ablation_20260527_summary.json`
- OOS trade ledger:
  `data/ensemble/reports/alpha3_no_teacher_parent_direct_oos_trades_20260527_cost3_ledger.csv`
- OOS chart:
  `data/ensemble/reports/alpha3_no_teacher_parent_direct_oos_trades_20260527_candles_cost3.png`

## Layer Contract

### Active Layers

1. Feature/data contract
   - Uses red-team-passed Alpha3 state24-v2 feature frame.
   - Legacy `clean_regime_2024_unsup_v4_*` is not active.
   - Current Regime4 source is `clean_regime4_state24_sticky090_v2_*`.

2. Parent
   - Same parent artifact as Alpha3 state24-v2 full retrain.
   - Output is used directly:
     `action`, `side`, `notional_exposure`, `leverage`,
     `take_profit`, `stop_loss`, `max_hold_bars`, `cooldown_bars`,
     `quality_score`, `confidence`.

3. Teacher
   - Removed from active path.
   - No Teacher confidence cutoff.
   - No Teacher no-flip filtering.
   - No Teacher learned size or notional scaling.
   - Teacher remains available only for historical comparison in ablation reports.

4. Runner / Scout / Exit
   - V21.2 runner remains active.
   - V27/V31 deep scout/exit stack remains active.
   - Guard overlay used for this candidate:
     `guard_soft3_hard1p45`.

5. Execution
   - Same OHLCV proxy execution contract used in the experiment:
     `next_open_limit_touch0_fee20` with close fallback behavior.
   - Real L2 queue/partial-fill/post-only reject validation is still required
     before live promotion.

## OOS Metrics

Fixed 2026 OOS, Cost3:

- PnL: `+122.02353682607266%`
- MDD: `-30.35985114338784%`
- WR: `36.53846153846153%`
- Trades: `260`
- Trades/day: `4.431818181818182`
- Long / Short: `105 / 155`
- SL ratio: `0.573076923076923`

## Decision

Status: `named_sub_version_candidate_not_live_promoted`

Rationale:

- Compared with the Teacher baseline under the same guard stack,
  `no_teacher_parent_direct` improves OOS Cost3 PnL and MDD.
- `teacher_flip_conf0` is explicitly rejected despite high validation score
  because OOS Cost3 PnL collapses to approximately `+3.20%`.
- `alpha3.1` is therefore the named next Alpha3 sub-version candidate, but not
  yet a live-promoted model.

## Promotion Blockers

- Needs parity test against live bot runtime path.
- Needs L2 queue/partial-fill/post-only reject validation.
- Needs chart-level audit of large-winner concentration and leverage/notional
  realism.
