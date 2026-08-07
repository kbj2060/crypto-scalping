# hf_v13 v40.6 No Max-Hold / No Cooldown Main Contract

## Scope

- Model ID: `hf_v13_tree_vs_foundation_target_aware_full_v40_6_20260512`
- Main variant: `v40_6_no_maxhold_no_cooldown`
- Status: `main_execution_contract_decision`
- Owner roles: Model/Data Architect, Implementation Maintainer, Red Team
- Decision date: `2026-05-12 KST`
- Parent artifact: `/home/llewyn/crypto-scalping/data/ensemble/supervised/hf_v13_tree_vs_foundation_target_aware_full_v40_6_20260512/target_aware_full_bundle.pkl`
- Report: `/home/llewyn/crypto-scalping/data/ensemble/reports/hf_v13_tree_vs_foundation_target_aware_full_v40_6_20260512_summary.json`
- Audit: `/home/llewyn/crypto-scalping/data/ensemble/reports/hf_v13_tree_vs_foundation_target_aware_full_v40_6_20260512_audit.json`

This contract records the architectural decision that v40.6 remains the main target-aware fully learned governor, but live/backtest execution should use the `no_max_hold_no_cooldown` variant as the main operating contract.

## Decision

The v40.6 parent policy remains responsible for:

- `action`: `CASH / LONG / SHORT`
- `side`
- `notional_exposure`
- `position_fraction`
- `leverage`
- `take_profit`
- `stop_loss`
- `quality_score`
- `confidence`

The main execution layer must not enforce the v40.6 predicted:

- `max_hold_bars`
- `cooldown_bars`

Effective runtime values:

```text
effective_max_hold_bars = 0
effective_cooldown_bars = 0
```

The raw model predictions may still be retained in audit logs as diagnostic fields, but position management must use the effective values above.

## Rationale

Recent ablation selected the no max-hold/no cooldown execution variant as the main direction.

| Variant | Cost | PnL | MDD | Trades |
|---|---:|---:|---:|---:|
| baseline v40.6 | 1x | `+125.61%` | `-32.84%` | 56 |
| no max-hold + no cooldown | 1x | `+133.365%` | `-34.0015%` | 47 |
| no max-hold + no cooldown | 2x | `+60.978%` | not primary | 49 |
| no max-hold + no cooldown | 3x | `+58.646%` | not primary | 51 |

The main trade-off is accepted: cost1 PnL improves while MDD worsens modestly. The variant also keeps positive cost2/cost3 PnL, so it is preferred over the default v40.6 execution contract for the next main shadow/live test.

## Dataset Split

Use the original v40.6 split and artifact provenance:

```text
Train:      2025-01-01 00:00:00 ~ 2025-09-30
Validation: 2025-10-01 00:00:00 ~ 2025-12-31 23:55:00
OOS:        fixed 2026 OOS window from v40.6 reports
```

The execution-only ablation does not retrain the model and must not use 2026 labels for model selection beyond the already recorded ablation decision.

## Shared Feature Contract

The feature contract is inherited from v40.6:

- Target-aware PLS factors from Chronos-2 and Kairos_23m embeddings.
- Residual raw market, AI, M7, microstructure, and `clean_regime_2024_unsup_v4_*` features.
- Original v40.6 feature count and preprocessing remain unchanged.
- No legacy contaminated regime-v2 features are allowed.

## Layer Contracts

| Layer | Input | Output | Effective contract |
|---|---|---|---|
| v40.6 parent | v40.6 feature frame | action, side, notional, leverage, TP, SL, max_hold, cooldown, quality, confidence | Keep all outputs for diagnostics |
| Execution override | v40.6 decision row | effective max-hold/cooldown | Force `max_hold=0`, `cooldown=0` |
| Position manager | active position, TP/SL, effective max-hold | hold/close | Close only by TP, SL, reverse/flat policy, or external risk/accounting rule, not max-hold |
| Entry gate | v40.6 action, effective cooldown | enter/skip | Do not block entry due to model cooldown |

## Cost/Risk Assumptions

- Cost stress remains `1x / 2x / 3x` fee/slippage parity.
- Notional and leverage remain v40.6 model outputs.
- Removing max-hold can increase tail hold time.
- Removing cooldown can increase immediate re-entry/churn.
- Existing exchange/accounting safety checks remain in force unless explicitly disabled elsewhere.

## Output Contract

Logs and ledgers should distinguish raw and effective fields:

```text
raw_max_hold_bars
raw_cooldown_bars
effective_max_hold_bars = 0
effective_cooldown_bars = 0
execution_contract = "v40_6_no_maxhold_no_cooldown"
```

If only one field can be persisted today, the live execution field must represent the effective value, not the raw model prediction.

## Red Team Gates

Before or immediately after live injection, Red Team must verify:

- `fully_learned_max_hold` cannot trigger for v40.6 main positions.
- `fully_learned_cooldown` cannot block new v40.6 entries.
- Raw model max-hold/cooldown predictions are not accidentally used through restored runtime state.
- Existing open positions created before this decision do not inherit stale max-hold/cooldown state.
- Ledger/dashboard records identify the contract as `v40_6_no_maxhold_no_cooldown`.
- Cost1/cost2/cost3 comparisons remain positive under the same fee/slippage model.
- Average hold time, max hold time, same-direction re-entry interval, flip frequency, and churn are monitored.

## Open Issues

- Implementation Maintainer must wire this contract into `trading_bot.py` or the active live runtime path.
- Red Team must run a live/backtest parity check after implementation.
- If MDD expands beyond the accepted budget, the next architecture loop should add an independent drawdown or reversal governor rather than re-enabling fixed max-hold/cooldown by default.
