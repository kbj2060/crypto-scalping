# Alpha3 Guarded Exit Selector v3 Contract

## Purpose

v1 learned early exits and v2 rescue exits both showed that aggressive early-close logic can overfit 2025Q4 and damage 2026. v3 therefore makes the exit layer fail-closed: a rescue policy is allowed only if it beats baseline validation score while keeping trade count and MDD stable.

## Selected Runtime

```json
{
  "name": "disabled_baseline",
  "min_hold": 999,
  "sl_progress": 99.0,
  "adverse_q_margin": 99.0,
  "min_mfe": 99.0,
  "giveback_frac": 99.0,
  "time_frac": 99.0,
  "exit_arm": "exit0_pen0",
  "maker_fee_mult": 0.2
}
```

## Selector Metadata

```json
{
  "selected_mode": "fail_closed_to_baseline",
  "baseline_validation_score": -90.07866444702957,
  "reason": "no_rescue_candidate_passed_trade_count_mdd_and_score_stability_guards",
  "guards": {
    "min_score_improvement": 10.0,
    "max_trade_ratio": 1.0,
    "max_mdd_degradation_pct": 2.0
  }
}
```

## Production Rule

If no candidate passes stability guards, the corrected Alpha3 baseline lifecycle remains active.
