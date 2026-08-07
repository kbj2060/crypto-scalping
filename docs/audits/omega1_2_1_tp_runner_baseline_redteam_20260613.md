# Omega1.2.1 TP Runner Baseline Red-Team Audit - 2026-06-13

## Verdict

`omega1_2_1_tp_runner_only_baseline_20260612` is **deprecated for active, candidate, and clean-OOS baseline use**.

The reported `OOS +205.92%` must be treated as an **OOS-mined research result**, not as clean promotion evidence.

## Audited Artifacts

- Baseline manifest: `data/ensemble/supervised/omega1_2_1_tp_runner_only_baseline_20260612/baseline_manifest.json`
- Runner bundle: `data/ensemble/supervised/omega1_2_1_tp_runner_meta_selector_20260610/tp_runner_meta_selector.joblib`
- Audit script: `scripts/audit_omega1_2_1_tp_runner_baseline_20260613.py`
- Audit report: `tmp/causal_regen_20260516/omega1_2_1_tp_runner_baseline_redteam_audit_20260613/report.json`

## Blocking Findings

1. OOS was used for model/config selection.
   - `scripts/eval_omega1_2_1_tp_runner_20260610.py` ranks by OOS PnL.
   - `scripts/train_eval_omega1_2_1_tp_runner_meta_selector_20260610.py` uses OOS PnL median/seed selection and saves the selected bundle.
   - Later 2026-06-13 experiments load that bundle, so they inherit the contaminated selection.

2. TP/SL execution is close-threshold based, not true intrabar barrier execution.
   - The replay checks unrealized PnL using close prices.
   - Intrabar high/low audit found earlier touches in `11/18` OOS trades and `23/31` validation trades.

3. Execution assumptions are optimistic.
   - Entry/exit maker fills use the next-bar open as limit price if touched.
   - There is no queue, partial fill, or post-only reject model.
   - Ledger `exit_price` records close, not the actual fill price used by accounting.

4. The headline result is sparse and regime dependent.
   - OOS has 18 trades, 15 shorts, median hold 425 bars, max hold 3181 bars.
   - The return is dominated by 2026 Jan-Feb ETH downtrend exposure.

## Re-Audit Metrics

| Mode | Validation PnL | Validation MDD | OOS PnL | OOS MDD | OOS WR | OOS Trades |
|---|---:|---:|---:|---:|---:|---:|
| Original limit/maker replay | 407.56% | -20.34% | 205.92% | -15.60% | 72.22% | 18 |
| Taker next-open sensitivity | 373.70% | -28.01% | 177.14% | -16.13% | 72.22% | 18 |

Taker sensitivity remains high, but it does not repair OOS-mining. The result remains blocked.

## Feature Contract Scan

No direct forbidden columns were found in the audited state/decision frames:

- No `clean_regime4_*`
- No `regime4_pred_*`
- No `teacher_*`
- No `tp_sl_action_score`
- No obvious `future`, `target`, `label`, `pnl`, or `zigzag` state columns

The blocker is provenance and replay accounting, not a visible forbidden-column leak.

## Required Policy

- Do not use `omega1_2_1_tp_runner_only_baseline_20260612` as a current research baseline.
- Do not use its `+205.92%` OOS number for promotion, ranking, or model comparison.
- Do not build new active candidates from `tp_runner_meta_selector_20260610` unless the selector is retrained/selected without 2026 OOS and then evaluated on a fresh untouched holdout.
- Future replay scripts must not rank by OOS. Rank by validation/walk-forward only and report OOS strictly after selection.
- Runtime-equivalent evaluation must log actual fill price, route, fee, slip, and intrabar barrier policy.
