# Omega5 Live-Only ML Loop Audit - 2026-07-02 15:00 KST

## Scope

- Active evidence source: `data/live/decision_feature_snapshot.jsonl` appended after the live-only loop start.
- Active loop: `data/live/omega5_live_only_upgrade_loop_20260702_v3_ml/`.
- Historical validation/OOS replay, stored ledgers, candidate ledgers, and saved parent exit timestamps are excluded from model selection.

## Runtime Status

- Trading bot process remained running with Omega5 disabled.
- Live-only shadow loop remained running through `scripts/run_omega5_live_only_shadow_loop_20260702.py`.
- Online ML candidates update only after closed live-forward shadow trades.

## 15:00 Readout

- Overall frontier: `omega5_live_short_momentum_v2`.
- Best online ML candidate at this checkpoint: `omega5_live_online_fast_bandit_v4`.
- `omega5_live_online_short_bandit_v5` was added after early live evidence showed short-biased candidates outperforming long exploration.

## Red-Team Notes

- PASS: no historical replay is used by the live-only loop.
- PASS: no stored trade ledger is used as an input signal source.
- PASS: no saved parent exit timestamp is used for entry or exit selection.
- WATCH: sample size is still too small for promotion; results are monitoring evidence only.
- WATCH: active trading bot still logs fully learned TP/SL feature-contract errors, but feature snapshots continue to be written for live-only shadow evaluation.

## 16:01 KST Update

- Overall live-only frontier: `omega5_live_short_momentum_v2`.
- Best ML-only guarded candidate: `omega5_live_online_short_guarded_v7`, with early live-only readout of 2 trades, 2 wins, zero MDD.
- The unguarded online short candidates deteriorated during upward-shock bars, validating the need for the `jump_z > 1.2 and positive return` guard.
- Promotion status remains WATCH only because the sample size is still below a defensible live-forward threshold.

## 16:37 KST Update

- Overall live-only frontier remains `omega5_live_short_momentum_v2`.
- ML-only candidates have not overtaken the rule frontier.
- Best ML family is short-only online bandit/guarded short:
  `omega5_live_online_short_bandit_v5` and `omega5_live_online_short_guarded_v7`.
- The rule-plus-guarded-ML hybrid `omega5_live_rule_plus_guarded_ml_v8` is too conservative so far and has not produced enough trades.
- Current conclusion: do not replace the rule frontier with a pure online ML model yet; keep ML as a shadow filter/extension while live-only evidence accumulates.

## 17:00 KST Update

- Overall live-only frontier remains `omega5_live_short_momentum_v2`.
- Pure/online ML candidates still do not justify replacing the rule frontier.
- Best ML candidates remain positive but materially behind the rule frontier:
  `omega5_live_online_short_bandit_v5`, `omega5_live_online_short_guarded_v7`, and `omega5_live_online_fast_bandit_v4`.
- `omega5_live_rule_plus_guarded_ml_v8` underperformed its intended rule-plus-filter role in early live-only samples.
- Current conclusion remains unchanged: keep Omega5 live promotion blocked; continue live-only shadow collection and use ML candidates as research evidence only.

## 17:34 KST Update

- Overall live-only frontier remains `omega5_live_short_momentum_v2`.
- Online ML candidates weakened as more live-only samples accumulated.
- The short-only guarded ML variants did not sustain their early edge through the next regime segment.
- Current conclusion: no ML replacement candidate should be promoted from this loop; the best live-only evidence still favors the short-momentum rule frontier.

## 18:00 KST Final Readout

- Loop completed normally at the configured 18:00 KST stop.
- Overall live-only frontier: `omega5_live_short_momentum_v2`.
  Final live-only shadow readout: 11 trades, 7 wins, PnL `0.0142137373`, MDD `-0.0051838013`.
- Best positive non-frontier rule candidate: `omega5_live_micro_flow_v2`.
  Final live-only shadow readout: 26 trades, 16 wins, PnL `0.0051912747`, MDD `-0.0105231230`.
- Online ML candidates did not pass the rule frontier:
  `omega5_live_online_fast_bandit_v4` ended barely positive, while the other online ML candidates ended negative or materially weaker than the rule frontier.
- Final decision: do not promote an Omega5 ML replacement from this loop.
  Keep Omega5 live promotion blocked and keep validation/OOS/test evidence live-only going forward.
