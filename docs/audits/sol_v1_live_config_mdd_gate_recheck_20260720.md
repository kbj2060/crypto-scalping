# SOL v1 live config — MDD promotion-gate recheck (2026-07-20)

## Why this check

`redteam_omega4_6_1_sol_btc_baselines_20260708.py` formally red-teamed an earlier SOL candidate
(duration-gate ON, `sol_final_scale_map_20260707` scale-mapped) and issued a **P1 blocker**
(`sol_oos_mdd_high`) because `oos_extended.mdd = -29.38%` breaches the script's hardcoded
`< -25.0%` threshold (`scripts/redteam_omega4_6_1_sol_btc_baselines_20260708.py:262-263`).

Five days later the project adopted a *different*, simpler SOL config (duration gate OFF, flat
1.5x notional multiplier, no scale-map) as the live v1 baseline, per
`docs/model_contracts/btc_sol_lowcost_tuning_sweep_20260713.md`. That tuning sweep does not contain
a `promotion_pass`/MDD-gate check, and no `docs/audits/*.md` file exists that runs the redteam MDD
rule against this config's actual sidecar path
(`tmp/causal_regen_20260516/sol_omega4_2_trade_risk_sidecar_20260707_zig075_q070_20260707` --
confirmed via `grep -rl` across `docs/audits/`, no hits).

## Result

Per `docs/model_contracts/live_model_v1_checkpoint_20260714.md`'s SOL v1 section (the current live
config, still the live config as of 2026-07-20):

| split | pnl | mdd | trades | wr |
|---|---|---|---|---|
| validation | +46.24% | **-36.17%** | -- | -- |
| oos_extended | +39.98% | **-27.91%** | 59 | 39.0% |
| oos_frozen_q1 | +33.02% | -24.85% | -- | -- |

Applying the exact same rule the 07-08 redteam used
(`if oos_extended.mdd < -25.0: P1 blocker "{asset}_oos_mdd_high"`):

- `oos_extended.mdd = -27.91% < -25.0%` -> **would trigger the identical P1 blocker** that got the
  prior candidate formally rejected.
- Validation MDD (-36.17%) is not gated by this rule at all (the rule only checks `oos_extended`),
  but is numerically worse than the OOS figure and worse than the rejected candidate's own
  validation MDD (-15.87%).

## Conclusion

**The current live SOL v1 config has never been run through the project's own MDD promotion gate,
and would fail it using the exact threshold that rejected its predecessor.** This is not a new
backtest finding (the underlying +39.98%/-27.91% numbers were already reported and reproducibility-
verified on 2026-07-14) -- it is a governance gap: the artifact-integrity gate
(`omega_artifact_integrity_audit_20260630.json`, `promotion_pass: true`, re-verified 2026-07-13)
only checks that prediction files aren't stale/mismatched, not that risk/MDD is within the
project's own historically-applied bound. No new code was run to produce this; it is a direct
application of an existing, already-run gate's own rule to already-reported numbers.

**Safety context**: `BINANCE_ACCOUNT_ENABLED=False` still blocks all real order placement for SOL
(and BTC/ETH) -- this finding does not indicate any real-money risk today, only a gap in the
model's own governance trail that should be resolved before that flag is ever flipped.

**Not addressed here** (out of scope for this recheck, flagged for follow-up): whether the
`< -25.0%` threshold itself is the right bar, whether SOL's structurally higher volatility vs ETH
justifies a looser SOL-specific MDD cap, or what config change (if any) would bring SOL under
-25.0% OOS MDD without giving up most of its PnL.
