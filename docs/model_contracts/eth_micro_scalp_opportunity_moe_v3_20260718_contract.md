# ETH Micro-Scalp Opportunity-MoE v3 Contract

## Status

- Model id: `eth_micro_scalp_opportunity_moe_v3_20260718`
- Parent: `eth_micro_scalp_inventory_moe_ensemble_v2_20260718`
- Purpose: research-only continuation/exit policy
- Live or shadow activation: blocked
- Promotion evidence: none

The v3 policy decides `SHORT`, `CASH`, or `LONG` at every completed one-minute
bar. It has no fixed holding period, maximum holding period, TP/SL, or cooldown.
Elapsed holding duration is not a feature.

## Architecture

Each of three deterministic seeds contains:

1. A causal price TCN with attention pooling.
2. A separate causal microstructure TCN with attention pooling.
3. Three regime-gated latent experts.
4. An inventory-conditioned `3 x 3` Q head.
5. An inventory-conditioned continuation opportunity-cost head.
6. An inventory-conditioned exit-hazard auxiliary head.
7. The existing seven-target market auxiliary head.

The ensemble averages the three mixed predictions and preserves all nine expert
heads for consensus decisions.

The parent encoders, regime gate, latent experts, position embedding, Q head, and
market auxiliary head are frozen exactly at their v2 values. Only the new
continuation and exit-hazard heads are trainable. This prevents the exit task
from silently changing the parent entry/action policy.

Before action selection, an optional conservative value is computed as mixed Q
minus a tune-selected multiple of the cross-expert Q standard deviation. The
small fixed candidate set is `0, 0.10, 0.25, 0.50, 1.00, 2.00`; outer intervals
cannot select this value.

## Opportunity target

For current inventory `p`, the fit-only teacher target is:

`continuation_advantage(p) = Q(p, hold p) - max Q(p, alternative action)`

A negative value means the best alternative has greater modeled value after
transaction cost. The exit-hazard label is one when the teacher's best action
differs from the current inventory. Future returns are used only to construct
fit targets; they are never model inputs.

## Decision rule

The parent Q policy first proposes the next inventory. If it proposes holding a
non-cash inventory, the opportunity overlay may select the best alternative when:

1. Ensemble continuation advantage is below the tune-selected floor; and
2. At least the tune-selected number of expert continuation heads agree.

The rule has no elapsed-time condition. A position may close after one minute or
remain open indefinitely when its learned continuation value stays sufficient.

## Selection and evidence boundary

Only the original tune interval may select the Q margin, switch consensus,
continuation floor, and exit consensus. The historical validation and development
intervals were already observed during v2 development and are now classified as
consumed-development diagnostics. They cannot activate, select, or promote v3.

The serialized execution policy is always disabled and must produce `CASH` until
post-freeze fresh-forward data is available.

Required report flags:

- `fresh_forward_bar_by_bar=true`
- `trade_ledgers_used_as_input=false`
- `saved_parent_exit_timestamps_used=false`
- `future_rows_used_for_entry=false`
- `fixed_holding_period_used=false`
- `outer_results_used_for_policy_selection=false`
- `parent_outer_results_used_for_policy_selection=false`

## Costs and sizing

Historical diagnostics use the same 4.5 bp fee per unit of notional change as the
parent experiment. This is a unit-notional signal study and does not introduce a
new futures margin, leverage, or position-sizing claim.
