# ETH Micro-Scalp Source-Stable Opportunity-MoE v4 Contract

## Purpose

v4 removes v3 inputs that cannot be reconstructed within the frozen scaled
source-parity thresholds from the live public market-data source. Feature
removal uses no returns, trade ledgers, validation outcomes, or development
outcomes.

The model still decides `SHORT`, `CASH`, or `LONG` on every completed one-minute
bar. It has no fixed or maximum holding period, fixed TP/SL, cooldown, or
holding-duration input.

## Source-stable feature set

The following seven base inputs are removed:

- `whale_retail_ratio`
- `whale_conviction`
- `smart_money_flow`
- `squeeze_power`
- `oi_change_rate`
- `long_squeeze_risk`
- `short_squeeze_risk`

The retained contract contains 36 base features and all 24 v3 microstructure
features. Names and order are stored in the checkpoint. Missing names, aliases,
implicit renames, or extra compatibility inputs fail immediately.

The removal boundary is justified only by the frozen v3 feature-stream parity
report. After excluding the seven named channels, the worst exposed retained
feature must satisfy both the maximum and p99 scaled-error thresholds. The
parity report hash is stored in both artifact and report.

## Warm start and training

For each of the three v3 seeds, only the removed columns of
`base_encoder.projection.weight` are deleted. Every other tensor is copied
byte-for-byte and the resulting state must load strictly into the 36-channel
model. All parameters are then jointly adapted on the original fit interval.

The joint objective retains inventory-Q, expert-Q, action classification,
auxiliary future-path distillation, regime-gate balance, continuation advantage,
expert continuation, and exit-hazard losses. Future paths are fit targets only.

Tune selects both a seed subset and the research switching/opportunity policy.
The subset search requires at least two seeds when the full three-seed artifact
is trained; it compares the three two-seed combinations and the full ensemble.
Historical
validation and development are consumed diagnostics and cannot select features,
hyperparameters, policies, or promotion.

## Execution and promotion

The saved execution policy is disabled and `activation_allowed=false`
regardless of historical results. No live or paper order path is present.

The separate non-executing observer starts at `2026-07-18 02:45:00 UTC`, the
first completed one-minute decision bar after the final v4 artifact freeze. It
may persist model decisions and external execution observations only. It cannot
submit or simulate orders, and it must not report PnL without full-fill evidence
for every position-change intent.

Promotion requires a new exact-source post-freeze, bar-by-bar fresh-forward run
with these flags:

- `fresh_forward_bar_by_bar=true`
- `trade_ledgers_used_as_input=false`
- `saved_parent_exit_timestamps_used=false`
- `future_rows_used_for_entry=false`
- `fixed_holding_period_used=false`
