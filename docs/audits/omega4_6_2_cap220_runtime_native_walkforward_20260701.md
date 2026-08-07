# Omega 4.6.2 Runtime Shadow / Holdout Audit - 2026-07-01

## Scope

- Model id: `omega4_6_2_cap220_short_boost125_time_stop120h_20260630`
- Variant: `short_rsi_skip_ge_56p656189__short_boost125_cap220__time_stop_120h`
- Output dir: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_cap220_runtime_native_walkforward_20260701`

## Runtime Replay

| Check | Result |
| --- | --- |
| Ledger-contract parity | `True` |
| GovernorPositionRouter accounting shadow available | `True` |
| trading_bot imported for shadow | `False` |
| Accounting shadow parity | `False` |
| FinalGovernorRuntime.decide replay available | `False` |
| Full runtime-native promotion pass | `False` |

Validation ledger PnL: `211.142583%`, native-shadow PnL: `204.329768%`

OOS ledger PnL: `79.317815%`, native-shadow PnL: `76.809201%`

The accounting shadow uses `GovernorPositionRouter._trade_math()` only. It is not a policy replay through `FinalGovernorRuntime.decide()`, so it cannot satisfy the full runtime-native replay gate.

## Fresh Holdout / Walk-Forward

| Check | Result |
| --- | --- |
| Exact fresh holdout available | `False` |
| Clean-OOS promotion claim allowed | `False` |
| Fixed candidate OOS monthly positive | `True` |
| Fixed candidate OOS monthly count | `2/2` |

Reason fresh holdout is unavailable: Exact candidate artifacts expose validation ledgers for 2025-10..2025-12 and OOS ledgers for 2026-01..2026-02 only. The component eval market ends at or near the OOS window, and no exact post-OOS prediction/ledger artifact is present for this model.

## Monthly Readout

CSV: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_cap220_runtime_native_walkforward_20260701/monthly_walkforward_readout.csv`

## Overall

- Runtime-native replay status: `FAIL_FINAL_GOVERNOR_RUNTIME_DECIDE_NOT_AVAILABLE`
- Fresh holdout status: `FAIL_NO_EXACT_FRESH_HOLDOUT_AVAILABLE`
- Promotion status: `BLOCKED_FOR_FULL_PROMOTION`
