# V55 `v55_bal_21` Red-Team Audit

Audit time: 2026-05-13 KST

## Verdict

**BLOCKED FOR PROMOTION / RESEARCH ONLY**

`v55_bal_21` reaches the PnL target on 2026 OOS, but it was found through a 2026 OOS diagnostic sweep rather than selected by the pre-2026 validation rule. It must not be injected into the live bot until a new pre-2026 selection rule can independently choose the same exposure family.

## Model Under Audit

- Parent: `hf_v13_clean_regime_margin110_20260511`
- V27 deep scout: frozen
- V49 raw-all Exit-RL: reused
- Candidate: `v55_bal_21`
- Parent multiplier: `1.35`
- Deep multiplier: `1.15`
- TP/SL scale power: `1.0`
- Add-on fraction: `0.20`
- Add-on total multiplier: `1.35`
- Add-on cap input: `4.14`

## OOS Metrics

| Cost | PnL | MDD | Trades | Trades/day | Avg notional | Cost survival |
|---|---:|---:|---:|---:|---:|---|
| cost1 | `+521.47%` | `-31.50%` | 173 | 2.95 | 2.08 | pass |
| cost2 | `+229.28%` | `-43.52%` | 176 | 3.00 | 2.13 | pass |
| cost3 | `+48.96%` | `-48.90%` | 182 | 3.10 | 2.13 | pass, high MDD |

## Blocking Findings

1. **Selection leakage / OOS-picked configuration**
   - Official V55 validation selected `v55_winner_add_0.65`, not `v55_bal_21`.
   - `v55_bal_21` validation on 2025 Q4:
     - PnL `-37.19%`
     - MDD `-57.40%`
     - cost2 PnL `-66.76%`
     - cost3 PnL `-85.44%`
   - `v55_bal_21` was identified after examining 2026 OOS diagnostics.
   - This is not a clean model-selection path.

2. **Final exposure can exceed `4.14`**
   - Entry notional never exceeded `4.14`.
   - After V21.2 add-on, final notional reached `4.968`.
   - 18 ledger rows had final notional above `4.14`.
   - This may be intentional via `max_total_mult=1.35`, but it means `4.14` is not a true final exposure cap.

3. **Ledger does not explicitly decompose resize/add-on fees**
   - Backtest cash accounting charges add-on fee on delta notional.
   - The ledger has entry and exit fee fields, but no separate resize/add-on fee rows.
   - 24 trades had inferred add-ons, with total added notional delta about `18.02`.
   - This is an auditability issue for live reconciliation.

## Passed Checks

1. **Timestamp separation**
   - Train range: 2025
   - Eval range: 2026-01-01 to 2026-02-28 16:00
   - Train/eval timestamp overlap: `0`

2. **Forbidden feature scan**
   - V49 raw state feature count: `164`
   - Forbidden state features: none detected
   - No `future`, `target`, `label`, `ledger`, `realized_pnl`, `regime_v2`, `hdb`, `hmm`, or `legacy_regime` columns in V49 raw state.

3. **Regime feature quarantine**
   - Uses `clean_regime_2024_unsup_v4_*`
   - No known contaminated legacy regime feature tokens detected.

4. **Next-bar execution parity**
   - Entry fill delay: always 5 minutes after signal.
   - Exit fill delay: always 5 minutes after signal.
   - Entry price equals next-bar open with slippage.

5. **Fee field consistency**
   - Entry fee percent equals `entry_notional * 0.0005 * 100`.
   - Exit fee percent equals `final_notional * 0.0005 * 100`.

## Warnings

- Parent audit reports missing train features zero-filled:
  `garch_vol_z`, `liquidity_vacuum`, `execution_quality`, `jump_z`, `jump_flag`, `evt_tail_flag`, `evt_excess_z`, `funding_abs`, `funding_pressure`, `crowding_pressure`, `whale_conviction`, `patchtst_pred`, `patchtst_confidence`.
- Parent audit reports missing eval features zero-filled:
  `patchtst_pred`, `patchtst_confidence`.
- Cost3 survives, but MDD is `-48.90%`, which is materially worse than V31 cost3 MDD.

## Required Fixes Before Promotion

1. Create a pre-2026 selection rule that can choose the `parent_mult=1.35`, `deep_mult=1.15` exposure family without inspecting 2026.
2. Decide whether final exposure cap is `4.14` or `entry_notional * 1.35`; enforce and document it consistently.
3. Add resize/add-on fee rows or fields to the trade ledger.
4. Re-run cost1/cost2/cost3 audit after the above changes.

