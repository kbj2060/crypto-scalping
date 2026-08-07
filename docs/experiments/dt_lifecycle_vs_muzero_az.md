# DT Lifecycle vs MuZero/AZ Experiment

Last updated: 2026-05-05 KST

This experiment compares the current MuZero/AZ zero-style stack against a newly isolated `DT Lifecycle + IQL/CQL Gate + CVaR Critic` candidate.

The script does not overwrite existing MuZero/AZ artifacts. It reads current artifacts and writes candidate outputs under:

- `data/ensemble/supervised/dt_lifecycle_iql_cvar/`
- `data/ensemble/reports/dt_lifecycle_vs_muzero_az_2026.json`

## Command

```bash
/home/llewyn/miniconda3/bin/conda run -n quant_ai python scripts/compare_muzero_az_vs_dt_lifecycle_2026.py
```

For fast smoke validation:

```bash
/home/llewyn/miniconda3/bin/conda run -n quant_ai python scripts/compare_muzero_az_vs_dt_lifecycle_2026.py \
  --device cpu \
  --epochs 1 \
  --max-train-samples 4000 \
  --exit-samples 8000 \
  --report-out tmp/dt_lifecycle_vs_muzero_az_smoke.json \
  --model-dir tmp/dt_lifecycle_iql_cvar_smoke
```

## Compared Stacks

Current stack:

- MuZero entry planner
- AZ risk overlay
- Stage2 MuZero sleeve overlay with `g0.55 / p0.00 / d1 / score_floor0.12`
- AZ exit governor with threshold `0.45`

Candidate stack:

- Decision-Transformer-style lifecycle policy
- IQL/CQL-inspired conservative critic gate
- CVaR tail-loss critic
- Separate learned exit governor

Both stacks use the same `fee`, `slip`, `max_notional`, `leverage_cap`, evaluation CSV, and `backtest_no_limit_exit` accounting path.

## Promotion Notes

This is an experiment harness, not a live promotion path. Red Team must still verify OOF/embargo, funding/liquidation approximation, fee/slippage stress, resize accounting, and weekly/monthly walk-forward stability.
