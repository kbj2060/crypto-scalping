# Architecture Alpha Loop 3x - 2026-05-06

Status: `completed_no_promotion`

## Research Inputs

- Conservative Q-Learning: offline policies can overestimate value under distribution shift, so new overlays should be conservative unless they prove incremental edge on held-out data. Source: https://papers.nips.cc/paper_files/paper/2020/hash/0d2b2061826a5df3221116a5085a6052-Abstract.html
- Conformal Risk Control: calibration data can be used to control monotone risk-style losses beyond plain classification coverage. Source: https://arxiv.org/abs/2208.02814
- Mamba selective state spaces: selective sequence state propagation is a plausible next architecture for long regime memory, but it was not implemented in these three loops because the current clean-base audit path needs trade-level accounting preservation first. Source: https://arxiv.org/abs/2312.00752

## Baselines

| Model | PnL | MDD | Trades/day | Cost2 PnL | Cost3 PnL | Status |
|---|---:|---:|---:|---:|---:|---|
| clean base | `177.329809%` | `-17.759665%` | `6.187500` | `92.254878%` | `-7.969395%` | safe reference |
| causal sleeve v1.2 / mdd guard ref | `210.491277%` | `-18.015155%` | `6.152542` | `133.150031%` | `-9.150155%` | best safe shadow reference |

## Loop Results

| Loop | Architecture | PnL | MDD | Trades/day | Cost2 PnL | Cost3 PnL | Audit | Decision |
|---|---|---:|---:|---:|---:|---:|---|---|
| 1 | clean base causal trade editor v1.3 | `190.786126%` | `-18.869253%` | `6.152542` | `121.089019%` | `-16.563845%` | preservation/accounting/causality pass | reject vs current best; MDD/cost3 degraded |
| 2 | clean base causal sleeve MDD guard v1.3 | `210.491277%` | `-18.015155%` | `6.152542` | `133.150031%` | `-8.953223%` | preservation/sleeve accounting/causality pass | best safe shadow, but strict promotion gate failed |
| 3 | clean base conformal downside filter v1.4 | `174.345657%` | `-17.438173%` | `6.152542` | `116.558819%` | `-8.249483%` | preservation/accounting/causality pass | shadow only; MDD improves but PnL below clean base |

## Loop 3 Contract

- Implementation: `scripts/train_eval_clean_base_conformal_downside_filter_v1_4.py`
- Runtime features: `side`, `quality`, `confidence`, `core_notional`, `leverage`, `funding_abs`, `funding_pressure`, `liquidity_vacuum`, `amihud_illiquidity_z`, `m7_tail_risk`, `evt_tail_flag`, `ai_adverse_risk`
- Forbidden runtime features: `evt_candidate_side`, `evt_candidate_label`, `evt_side_margin`, future high/low/close
- Train labels: 2025 pre-validation window
- Calibration/selection: 2025 validation window
- OOS: 2026 one-shot after threshold selection
- Output invariant: trade count, entry index, side, and max core notional are preserved; exits can only become earlier.

## Model/Data Architect Conclusion

No candidate should be promoted into live trading yet. The best safe candidate remains causal sleeve MDD guard v1.3, but it is still a shadow candidate because it misses the strict MDD and cost3 gates. The conformal downside filter is useful as a risk-control component because it improved MDD from clean base `-17.759665%` to `-17.438173%`, but it cut too much exposure and reduced PnL to `174.345657%`.

The next high-value architecture should combine loop 2 and loop 3 asymmetrically:

1. Keep the causal sleeve v1.2/v1.3 same-side add logic as the alpha source.
2. Add conformal downside gating only to veto sleeve additions, not to shrink every core trade.
3. Use a separate calibration window or rolling walk-forward calibration before claiming conformal validity.
4. Only after the sleeve-veto version passes audit, test a Mamba-style long-regime state encoder as an offline feature generator, never as an unchecked live action owner.

## Artifacts

- Loop 1 report: `data/ensemble/reports/clean_base_causal_trade_editor_v1_3_2026.json`
- Loop 2 report: `data/ensemble/reports/clean_base_causal_sleeve_mdd_guard_v1_3_2026.json`
- Loop 3 report: `data/ensemble/reports/clean_base_conformal_downside_filter_v1_4_2026.json`
- Loop 3 ledger: `data/ensemble/reports/clean_base_conformal_downside_filter_v1_4_ledger.csv`
- Loop 3 model contract: `docs/model_contracts/clean_base_conformal_downside_filter_v1_4_contract.md`
