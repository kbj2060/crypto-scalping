# MuZero/AZ Alpha-Preserving Micro-Add v1

Last updated: 2026-05-05 KST

Loop-1 status: `iterate_required`. The smoke report is `smoke_not_promotable` because full OOS was not run, `trades/day` did not increase, and `microadd_entry_count` was `0`. This v1 path also targets the older `467.64%` DT lifecycle comparison baseline rather than the latest `752.65%` rank-1 contract. See `docs/experiments/architecture_loop_2026-05-05_iteration_1.md`.

This experiment freezes the current MuZero/AZ stack and adds two audited layers:

- defensive monotonic active-row sleeve reused from `muzero_az_defensive_sleeve_v1`
- deterministic current-bar vote micro-add sleeve for sequentially flat baseline state only

The full OOS baseline is `eval.current_muzero_az` from `data/ensemble/reports/dt_lifecycle_vs_muzero_az_2026.json`, not the validation result. The full hard-gate target is:

- PnL `467.644256`
- MDD `-25.912969`
- trades `369`
- trades/day `6.289773`
- avg leverage `1.594981`

Smoke runs with row limits mark the baseline reproduction gate as `skipped/development` because limited rows cannot match the full OOS target. Full runs mark a reproduction mismatch as `candidate_failed`.

## Outputs

- Script: `scripts/compare_muzero_az_alpha_preserving_microadd_v1_2026.py`
- Full report: `data/ensemble/reports/muzero_az_alpha_preserving_microadd_v1_2026.json`
- Full artifact dir: `data/ensemble/supervised/muzero_az_alpha_preserving_microadd_v1/`
- Smoke report: `tmp/muzero_az_alpha_preserving_microadd_v1_smoke.json`
- Smoke artifact dir: `tmp/muzero_az_alpha_preserving_microadd_v1_smoke/`

## Smoke Command

```bash
/home/llewyn/miniconda3/bin/conda run -n quant_ai python scripts/compare_muzero_az_alpha_preserving_microadd_v1_2026.py \
  --device cpu \
  --limit-train-rows 1200 \
  --limit-val-rows 900 \
  --limit-eval-rows 900 \
  --max-train-samples 800 \
  --report-out tmp/muzero_az_alpha_preserving_microadd_v1_smoke.json \
  --model-dir tmp/muzero_az_alpha_preserving_microadd_v1_smoke
```

## Full Command

```bash
/home/llewyn/miniconda3/bin/conda run -n quant_ai python scripts/compare_muzero_az_alpha_preserving_microadd_v1_2026.py
```

## Report Contract

The report includes baseline reproduction, state/provenance audit, sequential flat replay audit, vote audit, leakage audit, invariant audit, validation selection trace, cost stress at `1x / 2x / 3x`, monthly OOS, and compact weekly OOS breakdowns.

The micro-add side is never selected from future returns. Future windows are used only for train/validation edge, CVaR, worst-path, and cost-survival labels for deterministic vote candidates.
