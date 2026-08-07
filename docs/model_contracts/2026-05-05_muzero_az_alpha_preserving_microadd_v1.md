# MuZero/AZ Alpha-Preserving Micro-Add v1 Data Contract

Status: `iterate_required_after_loop_1_smoke`

Last updated: 2026-05-05 KST

Loop-1 result: `smoke_not_promotable`. The candidate passed limited smoke operation and invariant checks, but did not increase trades/day, produced `0` micro-add entries, and was not run on full OOS. This v1 contract also targets the older `467.64%` DT lifecycle comparison baseline; future versions must retarget the latest `752.65%` rank-1 contract unless the user explicitly overrides the baseline. Promotion remains blocked until a later version passes full OOS baseline reproduction and all success gates.

## Scope

- Model id: `muzero_az_alpha_preserving_microadd_v1_2026`
- Loop: `1 / 5`
- Architecture under test: frozen current `MuZero Entry + AZ Risk + Stage2 MuZero Sleeve + AZ Exit` baseline plus selective defense and a separately audited micro-add sleeve.
- Purpose: improve OOS PnL and MDD versus the highest-return baseline while increasing trades/day without turning the sleeve into an unconstrained action owner.
- Owner agents: Data Architect, Model Architect, Red Team, Implementation Maintainer
- Proposed implementation script: `scripts/compare_muzero_az_alpha_preserving_microadd_v1_2026.py`
- Source baseline report: `data/ensemble/reports/dt_lifecycle_vs_muzero_az_2026.json`
- Source defensive-only contract: `docs/model_contracts/2026-05-05_muzero_az_defensive_sleeve_v1.md`
- Reserved report artifact: `data/ensemble/reports/muzero_az_alpha_preserving_microadd_v1_2026.json`
- Reserved smoke report: `tmp/muzero_az_alpha_preserving_microadd_v1_smoke.json`
- Reserved artifact root: `data/ensemble/supervised/muzero_az_alpha_preserving_microadd_v1/`

This candidate must not overwrite any existing MuZero/AZ artifact, rejected DT lifecycle artifact, defensive sleeve v1 artifact, or existing comparison script.

## Baseline Freeze

The baseline to reproduce before applying this candidate is `eval.current_muzero_az` from `data/ensemble/reports/dt_lifecycle_vs_muzero_az_2026.json`:

| Stack | OOS range | PnL | MDD | Trades | Trades/day | Avg leverage |
|---|---|---:|---:|---:|---:|---:|
| Current MuZero/AZ | `2026-01-01` to `2026-02-28 16:00` | `467.64%` | `-25.91%` | `369` | `6.29` | `1.59` |

Loop-1 success gates:

- OOS PnL must exceed `467.64%`.
- OOS MDD must be less severe than `-25.91%`.
- OOS trades/day must exceed `6.29` but stay below the validation-selected churn cap.
- Avg leverage target range: `1.50` to `1.80`.
- Cost `1x / 2x / 3x` must survive and be reported against the same baseline under identical accounting.
- Invariant audit must pass with zero hard violations.

## Dataset Split

| Split | Source | Timestamp range | Rows | Use |
|---|---|---:|---:|---|
| Full train source | `tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv` | `2025-01-01 00:00:00` to `2025-12-31 23:55:00` | `105064` | source before split |
| Train | same as above | `2025-01-01 00:00:00` to `2025-10-31 23:55:00` | `87496` | train defensive diagnostics and micro-add calibration heads |
| Validation | same as above | `2025-11-01 00:00:00` to `2025-12-31 23:55:00` | `17568` | select thresholds, churn quota, micro-add notional, leverage ceiling |
| Test/OOS | `tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv` | `2026-01-01 00:00:00` to `2026-02-28 16:00:00` | `16897` | final comparison only |

Audit inherited from the latest baseline report:

- Train/eval timestamp overlap rows: `0`
- Train duplicate timestamps: `0`
- Eval duplicate timestamps: `0`
- Policy feature count: `79`
- OOS rows must not be used for fitting, threshold selection, calibration, scaler fitting, feature selection, or churn quota selection.

## Shared Feature Contract

- Canonical feature source: `ensemble.fully_learned_governor_policy.FEATURE_COLS`
- Feature count: `79`
- Base cadence: `5m`
- Normalization:
  - Frozen baseline artifacts use their saved normalization payloads.
  - New candidate heads fit scalers/imputers on train split only.
  - Validation and OOS use train-fitted state transforms only.
- Missing fallback: schema-align, replace `inf/-inf` with NaN, then fill with train-defined fallback values.
- Stale handling: report `missing_count`, `inf_count`, `stale_count`, and `unavailable_telemetry`.
- Live availability: every state field must be causal at the current bar.

The 79 base features are inherited from `FEATURE_COLS`; the implementation must persist the exact ordered feature list into the artifact metadata.

## Input State Schema

Baseline decision columns required before candidate application:

```text
action
side
notional_exposure
leverage
position_fraction
quality_score
confidence
cooldown_bars
```

Baseline telemetry required when available:

```text
mz_entry_score_0
mz_entry_score_1
mz_entry_score_2
mz_entry_score_3
mz_entry_prob_0
mz_entry_prob_1
mz_entry_prob_2
mz_entry_prob_3
mz_entry_value
az_risk_scale
az_risk_prob
az_risk_value
stage2_mz_scale
stage2_mz_score
az_exit_prob
```

Trailing-only state required for both sleeves, shifted by one bar:

```text
rolling_trades_1d
rolling_trades_7d
rolling_notional_delta_1d
rolling_notional_delta_7d
rolling_fee_slip_cost_1d
rolling_fee_slip_cost_7d
rolling_realized_pnl_1d
rolling_realized_pnl_7d
rolling_drawdown_1d
rolling_drawdown_7d
time_since_last_entry_bars
time_since_last_exit_bars
time_since_last_microadd_bars
time_since_last_resize_bars
```

Micro-add candidate state, evaluated only where the frozen baseline is flat:

```text
baseline_flat
candidate_side_from_votes
vote_margin
vote_agreement_count
direction_entropy
tail_risk_score
cost_buffer_1x
cost_buffer_2x
cost_buffer_3x
microadd_cooldown_ok
microadd_churn_budget_remaining
```

`candidate_side_from_votes` must be derived only from current-bar point-in-time signals such as `side_hint`, `ai_dir_p_up`, `ai_dir_p_down`, `ai_dir_edge`, `m7_expected_ret`, `m7_confidence`, `patchtst_confidence`, and causal regime fields. It must not be fitted as a free-form future-return side classifier.

## Layer Contracts

| Layer | Input state/features | Train labels | Output | Artifact |
|---|---|---|---|---|
| Frozen baseline builder | `FEATURE_COLS` plus frozen baseline artifacts | existing frozen labels only | `baseline_decision_frame`, telemetry, baseline metrics | existing artifacts only |
| Alpha-preserving defensive gate | baseline-active rows with `FEATURE_COLS`, baseline decision, telemetry, trailing state | baseline-conditioned future net edge, adverse excursion, CVaR, early-exit benefit | `defense_pass`, `defense_scale`, `defense_reason`; can only keep, scale down, or flatten active baseline decisions | `alpha_preserving_defense.pkl` |
| Micro-add candidate generator | baseline-flat rows with current-bar vote features and strict tail/cost gates | none for side; deterministic vote side only | `candidate_side_from_votes`, `candidate_eligible` | script-local config in report |
| Micro-add survival and edge scorer | eligible baseline-flat candidate rows | candidate-side future net edge and tail loss under 1x/2x/3x cost | `microadd_edge`, `microadd_survival_prob`, `microadd_cvar_loss`, `microadd_size_score` | `microadd_scorer.pkl` |
| Threshold and quota selector | validation predictions and realized labels | validation-only threshold ranking | selected defense thresholds, micro-add thresholds, daily churn cap, notional cap, leverage ceiling | `threshold_selector.json` |
| Candidate combiner | baseline decision plus defense and micro-add outputs | none | final decision frame and reason codes | script-local, report diagnostics |
| Accounting and stress | baseline and candidate decision frames with same fills | none | PnL, MDD, trades/day, avg leverage, monthly, weekly, cost stress | `backtest_no_limit_exit` |

## Label Contract

Defensive labels are allowed only on baseline-active rows:

```text
baseline_net_edge_h144
baseline_adverse_excursion_h144
baseline_cvar_loss_alpha_0p10
baseline_early_exit_benefit
```

Micro-add labels are allowed only on rows where the reproduced baseline is flat and a deterministic vote side exists:

```text
microadd_net_edge_h36
microadd_net_edge_h72
microadd_net_edge_h144
microadd_worst_path_loss_h144
microadd_cvar_loss_alpha_0p10
microadd_survives_cost_1x
microadd_survives_cost_2x
microadd_survives_cost_3x
```

Leakage controls:

- Future windows may be used only to create train/validation labels.
- The micro-add side is not a future-return argmax label; future returns only decide whether a current-bar deterministic candidate is tradeable.
- All calibration buckets, thresholds, churn caps, and size caps are selected on validation only.
- OOS is final-read only.

Prohibited labels and behavior:

- No label may teach replacement of an active baseline side.
- No active baseline `LONG` to `SHORT` or `SHORT` to `LONG` transition is allowed.
- No micro-add may open during an active baseline position.
- No negative cost-adjusted edge gate is allowed.

## Action Rules and Invariants

Active baseline rows:

- `final_side == baseline_side` unless flattened to `0`.
- `final_action == baseline_action` unless flattened to cash.
- `final_notional_exposure <= baseline_notional_exposure`.
- `final_leverage <= min(baseline_leverage, selected_active_leverage_ceiling)`.
- `final_position_fraction <= baseline_position_fraction`.

Baseline-flat rows:

- New side can be created only by the micro-add sleeve.
- Micro-add side must equal `candidate_side_from_votes`.
- Micro-add notional must be one of validation-selected caps: `0.10`, `0.15`, `0.20`, `0.25`, `0.35`.
- Micro-add leverage ceiling candidates: `1.0`, `1.2`, `1.4`, `1.6`.
- Micro-add entries/day cap candidates: `0.25`, `0.50`, `0.75`, `1.00`, `1.25`.
- Micro-add is blocked when `tail_risk_score` is high, `cost_buffer_3x <= 0`, `rolling_drawdown_1d` is below the validation stop, or `time_since_last_microadd_bars` violates cooldown.

Hard invariant audit must report zero:

```text
active_side_reversal
active_created_side
active_notional_increase
active_leverage_increase
active_position_fraction_increase
microadd_when_baseline_active
microadd_side_without_vote
microadd_vote_conflict
microadd_notional_cap_violation
microadd_leverage_cap_violation
microadd_churn_cap_violation
nonfinite_decision_value
negative_notional_or_leverage
```

## Recommended Validation Grid

Defensive gate:

```text
defense_hazard_veto: [0.70, 0.80, 0.90]
defense_edge_floor: [0.0000, 0.0006, 0.0012]
defense_max_cvar_loss: [0.020, 0.035, 0.060]
defense_scale_floor: [0.50, 0.70, 0.85]
active_leverage_ceiling: [1.8, 2.0, 2.2, 2.5]
```

Micro-add gate:

```text
vote_margin_min: [0.20, 0.30, 0.40]
vote_agreement_min: [3, 4, 5]
microadd_survival_prob_min: [0.62, 0.70, 0.78]
microadd_edge_floor_3x: [0.0010, 0.0020, 0.0035]
microadd_max_cvar_loss: [0.006, 0.010, 0.015]
microadd_notional_cap: [0.10, 0.15, 0.20, 0.25, 0.35]
microadd_leverage_ceiling: [1.0, 1.2, 1.4, 1.6]
microadd_entries_per_day_cap: [0.25, 0.50, 0.75, 1.00, 1.25]
microadd_cooldown_bars: [12, 24, 36, 72]
```

Selection objective:

```text
score = pnl_delta - 1.6 * max(0, abs(mdd_candidate) - abs(mdd_baseline))
        + 0.25 * trades_per_day_delta
        - 0.35 * cost_3x_pnl_degradation
```

Any config that misses the MDD, avg leverage, cost survival, or invariant gates is ineligible regardless of score.

## Cost/Risk Assumptions

- Fee: `0.0005`
- Slippage: `0.0002`
- Max notional exposure: `3.6`
- Baseline leverage cap retained only for reproduction: `5.0`
- Candidate avg leverage target: `1.50` to `1.80`
- Active-row leverage ceiling candidates: `1.8`, `2.0`, `2.2`, `2.5`
- Micro-add leverage ceiling candidates: `1.0`, `1.2`, `1.4`, `1.6`
- Resize accounting: same `backtest_no_limit_exit` notional-delta accounting path
- Funding and liquidation are limitations; report must explicitly state they are not first-class unless implemented.

Mandatory stress:

```text
cost_1x = fee 0.0005, slip 0.0002
cost_2x = fee 0.0010, slip 0.0004
cost_3x = fee 0.0015, slip 0.0006
```

## Output Contract

Required output decision columns:

```text
action
side
notional_exposure
leverage
position_fraction
quality_score
confidence
reason_code
is_baseline_pass
is_defensive_modified
is_microadd
```

Required report frames:

```text
baseline_muzero_az
alpha_preserving_microadd_v1
delta_vs_baseline
validation_grid_ranked
cost_stress
monthly
weekly
state_audit
invariant_audit
calibration_audit
microadd_audit
defense_audit
```

Required diagnostics:

```text
baseline_pass_count
defense_veto_count
defense_size_down_count
microadd_entry_count
microadd_entries_per_day
microadd_long_entries
microadd_short_entries
microadd_block_reasons
avg_active_leverage_before
avg_active_leverage_after
avg_microadd_notional
avg_microadd_leverage
turnover_before_candidate
turnover_after_candidate
fee_slip_cost_before_candidate
fee_slip_cost_after_candidate
```

## Red Team Gates

- [ ] Baseline reproduction matches `eval.current_muzero_az` within tolerance before candidate application.
- [ ] Train/validation/test timestamp overlap audit is zero.
- [ ] No 2026 row is used for fitting, threshold selection, calibration, feature selection, or quota selection.
- [ ] Active baseline decisions satisfy all monotonic defense invariants.
- [ ] Micro-add occurs only from baseline-flat rows and only from deterministic current-bar vote side.
- [ ] Micro-add side-vote conflicts are blocked and counted.
- [ ] Micro-add daily churn cap is enforced before OOS scoring.
- [ ] Cost `1x / 2x / 3x` is reported for baseline and candidate; cost `3x` must remain positive and MDD-safe.
- [ ] OOS PnL, MDD, trades/day, and avg leverage gates all pass simultaneously.
- [ ] Weekly and monthly degradation counts are reported.
- [ ] Calibration buckets are reported for defense and micro-add predictions.
- [ ] Funding, liquidation, and maintenance-margin limitations are documented before any live shadow.

## Open Issues

- Defensive sleeve v1 has only a smoke report; implementation should run full v1 first or share its reproduced baseline builder before v2 comparison.
- Baseline telemetry availability is still uncertain and must be explicit in `state_audit`.
- Micro-add profitability can be cost fragile; Red Team should treat any PnL gain driven by high churn as blocking.
- OOF/embargo is required before promotion beyond experiment status.
