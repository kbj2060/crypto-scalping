# Alpha3 Rescue Exit Governor v2 Contract

## Change From v1

v1 learned early-exit selector was rejected because it underperformed the corrected Alpha3 baseline and did not reduce SL/max-hold exits. v2 is rescue-only: it preserves the existing Alpha3 TP/SL/max-hold lifecycle and only inserts a reduce-only close when a live-available adverse state is detected.

## Selected Runtime

```json
{
  "name": "giveback_h2_mfe0.015_gb0.50_q-0.002_exit0",
  "min_hold": 2,
  "sl_progress": 99.0,
  "adverse_q_margin": -0.002,
  "min_mfe": 0.015,
  "giveback_frac": 0.5,
  "time_frac": 99.0,
  "exit_arm": "exit0_pen0",
  "maker_fee_mult": 0.2
}
```

## Safety

- Entry stack remains frozen Alpha3 corrected `touch0 skip-entry`.
- No entry, flip, add, or increase action is allowed.
- TP/SL/max-hold remain fallback rails.
- Selection uses 2025Q4 only; 2026 is fixed OOS.
