# Alpha Exploration Red-Team Audit 2026-05-06

## Verdict

`state_option_moe_2026_v2` with leak-prone event columns is diagnostic-only and must not be promoted. The valid no-leak rerun rejects. The best currently valid alpha lift remains the clean-base causal conviction sleeve family around `210%` OOS PnL.

## Findings

1. `evt_candidate_side`, `evt_candidate_label`, and `evt_side_margin` are not live-causal features.
   - `scripts/build_trade_candidates_v1.py` computes them from future high/low/close barrier windows.
   - SOMoE v1 and the first SOMoE v2 diagnostic included those columns as model inputs.

2. Leaky diagnostic result:
   - Report: `data/ensemble/reports/state_option_moe_2026_v2.json`
   - OOS PnL: `4007.372308%`
   - MDD: `-10.055092%`
   - Trades/day: `10.745763`
   - Cost3 PnL: `-5.556414%`
   - Status: blocked, diagnostic-only.

3. No-leak SOMoE v2 rerun:
   - Report: `data/ensemble/reports/state_option_moe_2026_v2_noleak.json`
   - Dropped: `evt_candidate_side`, `evt_candidate_label`, `evt_side_margin`
   - OOS PnL: `-32.322168%`
   - MDD: `-39.502679%`
   - Trades/day: `4.542373`
   - Cost3 PnL: `-64.461344%`
   - Invariant audit: pass
   - Status: reject.

4. Causal conviction sleeve v1.2:
   - Report: `data/ensemble/reports/clean_base_plus_causal_conviction_sleeve_v1_2_2026.json`
   - OOS PnL: `210.491277%`
   - MDD: `-18.015155%`
   - Trades/day: `6.152542`
   - Cost2 PnL: `133.150031%`
   - Cost3 PnL: `-9.150155%`
   - Runtime causality audit: pass
   - Core lane preservation: pass
   - Status: reject for strict promotion, but valid alpha-lift shadow candidate versus clean base.

## Current Safe Ranking

| Model | Live-causal runtime | PnL | MDD | Trades/day | Cost3 PnL | Status |
|---|---:|---:|---:|---:|---:|---|
| Clean base | yes | `177.329809%` | `-17.759665%` | `6.187500` | `-7.969395%` | baseline |
| Causal sleeve v1.2 | yes | `210.491277%` | `-18.015155%` | `6.152542` | `-9.150155%` | shadow candidate |
| SOMoE v2 noleak | yes | `-32.322168%` | `-39.502679%` | `4.542373` | `-64.461344%` | reject |
| SOMoE v2 diagnostic | no | `4007.372308%` | `-10.055092%` | `10.745763` | `-5.556414%` | blocked |

## Next Work

- Remove leak-prone event label columns from SOMoE v1 contract/results or mark them diagnostic-only.
- Continue alpha work from `clean_base_plus_causal_conviction_sleeve_v1_2`, not from leaky SOMoE.
- Next promising path: keep clean base entries, learn a causal per-trade exit/scale schedule using only train labels and runtime predicted utility, with MDD constraint no worse than `-17.76%`.
