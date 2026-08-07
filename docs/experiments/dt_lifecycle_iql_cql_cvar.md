# DT Lifecycle + IQL/CQL Gate + CVaR Critic Shadow Experiment

This experiment adds a candidate harness without replacing the existing MuZero/AZ scripts.

Entrypoint:

```bash
python3 scripts/compare_dt_iql_cql_cvar_vs_zero_style.py \
  --fee 0.0005 \
  --slip 0.0002 \
  --leverage-cap 3.6 \
  --max-notional 3.6
```

Local smoke without loading existing sklearn pickles:

```bash
./venv/bin/python scripts/compare_dt_iql_cql_cvar_vs_zero_style.py \
  --entry-source heuristic \
  --smoke-rows 1200
```

The candidate is intentionally a fast surrogate:

- `EmpiricalDTLifecyclePolicy`: trajectory-window lifecycle policy interface.
- `EmpiricalLowerBoundGate`: IQL/CQL-style support and conservative lower-bound gate.
- `CVaRCritic`: tail-risk lifecycle critic stub.
- allocator behavior: empirical scale-down/scale-up stub bounded by `--max-notional` and `--leverage-cap`.

The comparison reads the existing Zero-style MuZero/AZ walk-forward report artifact and runs the candidate on the same CSV, fee, slippage, and cap parameters passed through the CLI.
