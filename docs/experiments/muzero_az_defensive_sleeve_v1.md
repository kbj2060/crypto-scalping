# MuZero/AZ Defensive Sleeve v1

Last updated: 2026-05-05 KST

This experiment freezes the existing Current MuZero/AZ stack and evaluates a defensive-only sleeve on top of its decision frame.

Frozen baseline:

- MuZero Entry Planner
- AZ Risk Overlay
- Stage2 MuZero Sleeve with `g0.55 / p0.00 / d1 / score_floor0.12`
- AZ Exit Governor with threshold `0.45`

Candidate sleeve:

- Lifecycle Hazard / DT Diagnostic Head
- Calibrated Quantile + CVaR Tail Head
- Cost / Turnover Monitor
- Regime Threshold Selector
- Hard Leverage Governor

The sleeve does not create replacement actions. If the baseline is flat, it stays flat. If the baseline is long or short, the sleeve can only keep the same direction, scale down notional, veto to flat, lower confidence/quality for exit context, and cap leverage. The validation grid only uses non-negative edge floors; the failed `min_lower_edge < 0` style gate is not used.

## Outputs

- Script: `scripts/compare_muzero_az_defensive_sleeve_v1_2026.py`
- Model artifact: `data/ensemble/supervised/muzero_az_defensive_sleeve_v1/defensive_sleeve_v1.pkl`
- Report: `data/ensemble/reports/muzero_az_defensive_sleeve_v1_2026.json`

## Full Command

```bash
/home/llewyn/miniconda3/bin/conda run -n quant_ai python scripts/compare_muzero_az_defensive_sleeve_v1_2026.py
```

## Smoke Command

```bash
/home/llewyn/miniconda3/bin/conda run -n quant_ai python scripts/compare_muzero_az_defensive_sleeve_v1_2026.py \
  --device cpu \
  --limit-train-rows 2500 \
  --limit-val-rows 1500 \
  --limit-eval-rows 1500 \
  --max-train-samples 2000 \
  --report-out tmp/muzero_az_defensive_sleeve_v1_smoke.json \
  --model-dir tmp/muzero_az_defensive_sleeve_v1_smoke
```

## Report Contract

The report includes:

- frozen current baseline validation/eval
- defensive sleeve validation grid and selected config
- selected hard leverage ceiling from `2.0 / 2.2 / 2.5`
- OOS eval delta versus current
- monthly breakdowns
- cost stress at `1x / 2x / 3x`
- red-team checklist before any promotion

This is an experiment harness only. It is not a live promotion path until OOF/embargo, fee/slippage, funding/liquidation, resize accounting, and weekly/monthly walk-forward checks pass.
