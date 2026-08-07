# Certified Teacher Cost Aware Allocator V4

- Model ID: `certified_teacher_cost_aware_allocator_v4`
- Purpose: repair V3 overtrading by adding a causal cost-aware opportunity gate.
- Core heads: separate LONG/SHORT execution utility rankers inherited from the V3 candidate stack.
- Allocation: daily entry budget, minimum bar gap, cost2 margin floor, catastrophic q10 veto, and edge/q10 notional scaling.
- Selection uses only 2025 selection data; 2026 is fixed OOS and is not used for config choice.
- Audit: `pass`
- Blocking: `[]`

## Selected Config
- Config: `{'top_k_per_day': 6, 'min_edge_cost2_margin': 0.04, 'trend_lane_only': True, 'micro_cap_per_day': 0, 'max_notional': 1.0, 'min_gap_bars': 12, 'catastrophic_q10_pct': -0.8, 'min_notional': 0.1, 'leverage': 5.0}`

## OOS Cost1
- PnL: `-7.688231631880404`
- MDD: `-8.54704077530708`
- Trades/day: `2.215909090909091`
- Avg notional: `0.2192618170281251`
