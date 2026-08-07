# Omega 4.6 plus_t12 No-Hold Risk1 Red-Team Record - 2026-06-30

## Verdict

`CONDITIONAL_PASS_MAX_HOLD_AND_PNL_TARGET_EXCLUDED_NOT_DAYTRADING_LIVE_PASS`

This model passes the non-excluded research baseline gates and fails day-trading
live-pass interpretation because max hold is too long.

## Evidence

- Candidate report: `tmp/causal_regen_20260516/omega_creative_until_10am_20260630/plus_t12_diagnostic_nohold_risk1_20260630/report.json`
- Artifact audit: `tmp/causal_regen_20260516/omega_creative_until_10am_20260630/plus_t12_diagnostic_nohold_risk1_20260630/omega_artifact_integrity_audit_20260630.json`
- Artifact audit result: `promotion_pass=true`

## Passed Gates

- Validation MDD: `-17.43%`
- OOS MDD: `-13.28%`
- Max leverage: `5.0`
- Max notional: `1.8`
- No overlap in selected trades
- Accounting consistency: pass
- `notional = margin_fraction * leverage`: pass
- Component artifact integrity: pass for `h48qual q050` and `zig075 q075`

## Excluded Gates

- PnL target:
  - Validation is `+117.17%`
  - OOS is `+67.85%`, below a `+100%` OOS target
- Max hold:
  - Validation max hold `222.0h`, with `21/29` trades over 24h
  - OOS max hold `218.5h`, with `12/13` trades over 24h

## Blocker For Full Live/Day Trading

The model is a swing/runner baseline. A 24h forced time-stop changes the exit
contract and collapses validation performance. A full live day-trading successor
must train or select exits under a shorter-hold objective instead of applying a
post-hoc hard 24h stop.

## Red-Team Upgrade Constraints

- Do not promote a successor from trade ledgers alone.
- Do not use OOS to select source-side scales, thresholds, exits, or risk maps.
- Keep max leverage and notional limits explicit.
- Any new max-hold claim must include a replay that recalculates return at the
  actual forced exit price.
