# Alpha3 Frozen Backtest Protocol

Last updated: 2026-05-16 KST

## Purpose

This protocol prevents false comparisons when testing Alpha3 changes. A candidate may change only the declared layer or logic. Everything else must remain frozen against the canonical Alpha3 baseline.

Canonical baseline:

- Alias: `alpha3`
- Contract: `alpha3_teacher_l2_limit_fallback_20260514`
- Execution: `next_open_limit_touch0_fee20`
- Backtest entrypoint: `scripts/eval_alpha2_1_signal_immediate_limit_20260514.py`
- 2026 OOS metrics:
  - cost1 PnL `+654.92%`, MDD `-29.62%`, trades `195`
  - cost2 PnL `+602.26%`
  - cost3 PnL `+456.48%`

The deprecated `+747.76%` open-fallback result is not a valid baseline.

## Frozen Stack

Unless explicitly declared as the target layer, freeze:

| Layer | Frozen Artifact / Contract |
|---|---|
| Parent | `data/ensemble/supervised/hf_v13_clean_regime_margin110_20260511/v13_clean_regime_margin110.pkl` |
| Teacher gate | `data/ensemble/supervised/alpha1_l2_teacher_deep_parent_20260514/teacher_deep_parent_l2_replay.pt` |
| V21.2 runner | `data/ensemble/supervised/hf_v13_jackpot_runner_v21_2_20260511/v21_2_jackpot_runner.pkl` |
| V27 scout | `data/ensemble/supervised/hf_v13_deep_alpha_candidate_expansion_v27_20260511/v27_deep_alpha_candidate_expansion.pt` |
| V31 exit | frozen `OverlayConfig` selected through `alpha1_l2_conservative_fee20` |
| Execution | `next_open_limit_touch0_fee20`, entry miss skip, exit miss close fallback |
| Accounting | existing Alpha3 compounding, fee, slippage, notional, leverage, resize fee formulas |
| Data | `trade_candidates_2025_patchtst__tide__dlinear.csv`, `trade_candidates_2026_patchtst__tide__dlinear.csv` |

## Required Experiment Types

### parent_only

Only parent decisions change. Teacher gate, V21.2 runner, V27 scout, V31 exit, execution, and accounting remain frozen.

Use this for the question: “Does a parent replacement improve Alpha3?”

### parent_plus_downstream_retune

Parent, teacher gate, runner, or other downstream artifacts are retrained or reselected together. This is a full-stack retune, not a parent-only test.

Use this for the question: “Can a newly tuned stack beat Alpha3?”

### exit_only

Entries and execution stay frozen. Only exit ownership/timing logic changes. If fill placement changes too, declare whether this is `exit_only_with_frozen_execution` or `execution_only`.

### execution_only

Decisions are frozen. Only route/fill simulation or live execution adapter logic changes.

### full_stack_retune

Multiple layers change. The result must not be compared as if one layer caused the delta.

## Baseline Reproduction Gate

Every candidate script must produce a baseline row before candidate rows.

Required baseline values:

```json
{
  "cost1": {"pnl": 654.9174150098765, "mdd": -29.61731295277763, "trades": 195},
  "cost2": {"pnl": 602.2624624847589, "mdd": -30.093378120960466, "trades": 195},
  "cost3": {"pnl": 456.48201847894717, "mdd": -31.397871677089583, "trades": 198}
}
```

If the baseline row does not match, stop. Do not interpret candidate performance.

## Required Report Fields

```json
{
  "base_model_alias": "alpha3",
  "frozen_protocol": "alpha3_frozen_backtest_protocol_20260515",
  "primary_mutable_surface": "parent_only",
  "changed_layers": ["parent"],
  "frozen_layers": ["teacher_gate", "v21_2_runner", "v27_deep_scout", "v31_exit", "execution", "accounting"],
  "baseline_reproduced": true,
  "baseline_metrics": {},
  "candidate_metrics": {},
  "delta_vs_baseline": {},
  "selection_uses_2026": false,
  "selection_window": "2025Q4",
  "oos_window": "2026 fixed OOS",
  "route_counts": {},
  "warnings": [],
  "red_team_blockers": []
}
```

## Interpretation Rules

- A `+451%` retrained HGB candidate is not a failed Alpha3 reproduction. It is a candidate result if the baseline row in the same report is `+654.92%`.
- A `+456%` number can be canonical Alpha3 `cost3`, not Alpha3 cost1.
- If teacher or runner is retrained during a parent test, the test becomes `parent_plus_downstream_retune` or `full_stack_retune`.
- If execution changes during an exit test, the test becomes `execution_only` or `full_stack_retune`.
- If a candidate beats cost1 but fails cost2/cost3 durability, verdict defaults to `iterate`, not `promote`.

## Minimum Commands

Baseline:

```bash
source /home/llewyn/miniconda3/etc/profile.d/conda.sh
conda activate quant_ai
python scripts/eval_alpha2_1_signal_immediate_limit_20260514.py
```

Candidate scripts should import Alpha3 helpers from the baseline implementation instead of duplicating accounting code.

## Promotion Gate

A candidate can be promoted only if:

- baseline row is reproduced in the same report,
- primary mutable surface is declared,
- cost1/cost2/cost3 all beat or clearly improve the target objective,
- MDD does not worsen beyond the stated budget,
- route counts are explained,
- no Red Team blocker remains,
- live L2 validation status is reported if execution assumptions are touched.

## 2026-05-16 Parity Addendum

For one-month CSV/native parity debugging, use:

- Protocol: `docs/model_contracts/alpha3_csv_native_backtest_parity_20260516.md`
- Red Team audit: `docs/experiments/alpha3_csv_native_parity_redteam_20260516.md`
- Machine-readable audit: `data/ensemble/reports/alpha3_csv_native_parity_redteam_20260516.json`

The aligned one-month parity baseline is not the same as the full-2026 baseline above:

```json
{
  "window": "2026-01-25 07:15:00 through 2026-02-24 07:10:00",
  "start_index_abs": 6999,
  "end_index_abs": 15638,
  "pnl": 338.68067144958997,
  "mdd": -29.617312952777574,
  "trades": 114,
  "action_events": 237
}
```

This parity mode is valid for isolating layer deltas under the canonical CSV loop. It is not a live-promotion proof because maker fills are still OHLCV high/low touch proxies rather than validated L2 queue fills.
