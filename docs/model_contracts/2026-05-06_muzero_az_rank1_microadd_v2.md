# MuZero/AZ Rank-1 Micro-Add v2 Data Contract

Status: `draft_for_loop_2`

Last updated: 2026-05-06 KST

## Scope

- Model id: `muzero_az_rank1_microadd_v2_2026`
- Loop: `2 / 5`
- Source baseline contract: `docs/model_contracts/2026-05-06_current_top_muzero_az_stage2_azexit.md`
- Source failed loop: `docs/model_contracts/2026-05-05_muzero_az_alpha_preserving_microadd_v1.md`
- Proposed implementation script: `scripts/compare_muzero_az_rank1_microadd_v2_2026.py`
- Reserved report artifact: `data/ensemble/reports/muzero_az_rank1_microadd_v2_2026.json`
- Reserved smoke report: `tmp/muzero_az_rank1_microadd_v2_smoke.json`
- Reserved artifact root: `data/ensemble/supervised/muzero_az_rank1_microadd_v2/`

This candidate retargets the current rank-1 baseline. The historical `467.64%` MuZero/AZ comparison baseline is not a promotion target for this loop.

## Frozen Baseline Semantics

The frozen baseline is `current_top_muzero_az_stage2_azexit_2026`:

```text
MuZero Entry Planner
AZ Risk Overlay
Stage2 MuZero Sleeve Overlay: gamma=0.55, prior=0.00, depth=1, score_floor=0.12
AZ Exit Governor: threshold=0.45
Execution accounting: backtest_no_limit_exit
```

Hard exclusions:

```text
Stage3 exit arbiter
Stage4 regime transition / regime overlay
DSAC as entry owner
```

The implementation must fail the report gate if Stage3/Stage4 artifacts, configs, or names are present in the active candidate path unless a later user request explicitly reintroduces them.

## Promotion Targets

All final gates compare against the current rank-1 baseline from `2026-05-06_current_top_muzero_az_stage2_azexit.md`:

| Metric | Hard gate |
|---|---:|
| OOS PnL | `> 752.65%` |
| OOS MDD | better than `-18.76%` |
| Trades/day | `> 6.02` |
| Avg leverage | `1.50 <= avg_leverage <= 1.80` |
| Cost 2x PnL | `> 279.36%` |
| Cost 3x PnL | `> 75.84%` |
| Invariant audit | zero hard violations |

Full OOS with no row limits and baseline reproduction are mandatory before any promotion decision.

## Dataset Split

| Split | Source | Timestamp range | Rows | Use |
|---|---|---:|---:|---|
| Full train source | `tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv` | `2025-01-01 00:00:00` to `2025-12-31 23:55:00` | `105064` | source before split |
| Train | same as above | `2025-01-01 00:00:00` to `2025-10-31 23:55:00` | `87496` | train diagnostic scorers and calibration transforms |
| Validation | same as above | `2025-11-01 00:00:00` to `2025-12-31 23:55:00` | `17568` | select thresholds, coverage gates, quotas, size caps |
| Test/OOS | `tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv` | `2026-01-01 00:00:00` to `2026-02-28 16:00:00` | `16897` | final read only |

Leakage rules:

- No 2026 row may be used for model fitting, scaler fitting, feature selection, threshold selection, quota selection, or coverage target selection.
- Future windows may be used only for train/validation labels.
- OOS report must include `oos_fit_rows=0` and `oos_threshold_selection_rows=0`.

## Architecture Delta From v1

Loop 1 failed because selected OOS smoke had `microadd_entry_count=0` and `trades/day delta=0`. v2 separates discovery from selection:

1. `Coverage Discovery`: find validation configs that produce nonzero micro-add entries without invariant violations.
2. `Rank-1 Gate Selection`: score only configs that increase validation trades/day while preserving cost survival and rank-1 target feasibility.
3. `Full OOS Confirmation`: run no-row-limit OOS and compare only to current rank-1 targets.

Active baseline rows remain monotonic defense only. Baseline-flat rows are the only rows where micro-add can create a side.

## Feature And State Changes

Base feature source remains:

```text
ensemble.fully_learned_governor_policy.FEATURE_COLS
```

Additional v2 state fields:

```text
rank1_baseline_active
rank1_baseline_flat
rank1_position_open_before
rank1_block_reason
rank1_time_since_flat_bars
rank1_recent_entry_rate_1d
rank1_recent_entry_rate_7d
rank1_recent_microadd_rate_1d
rank1_recent_microadd_rate_7d
microadd_vote_side
microadd_vote_strength
microadd_vote_source_count
microadd_vote_conflict_count
microadd_vote_entropy
microadd_edge_1x
microadd_edge_2x
microadd_edge_3x
microadd_cost_buffer_3x
microadd_tail_loss_q10
microadd_cvar_loss
microadd_coverage_bucket
microadd_quota_remaining_1d
microadd_quota_remaining_7d
stage3_stage4_exclusion_flag
```

State construction requirements:

- `rank1_baseline_flat` must come from sequential replay state, not only from row-level cash action.
- If replay is approximate, the report must set `exact_backtest_state_claimed=false` and Red Team must block promotion until exact baseline position state is available.
- `microadd_vote_side` must be deterministic and current-bar only.
- Allowed vote sources: `ai_dir_p_up/down`, `ai_dir_edge`, `m7_expected_ret`, `m7_confidence`, `conf_patchtst`, `patchtst_confidence`, causal regime ids/confidence, `mtf_trend_1h`, `mtf_trend_4h`, `smart_money_flow`, `net_taker_ratio`.
- Hidden future side labels are prohibited.

## Label Contract

Active-row defensive labels:

```text
rank1_active_net_edge_h144
rank1_active_adverse_excursion_h144
rank1_active_cvar_loss_alpha_0p10
rank1_active_early_exit_benefit
```

Baseline-flat micro-add labels, conditioned on deterministic `microadd_vote_side`:

```text
microadd_net_edge_h24_1x
microadd_net_edge_h36_1x
microadd_net_edge_h72_1x
microadd_net_edge_h144_1x
microadd_net_edge_h24_2x
microadd_net_edge_h36_2x
microadd_net_edge_h72_2x
microadd_net_edge_h144_2x
microadd_net_edge_h24_3x
microadd_net_edge_h36_3x
microadd_net_edge_h72_3x
microadd_net_edge_h144_3x
microadd_worst_path_loss_h144
microadd_cvar_loss_alpha_0p10
microadd_survives_cost_1x
microadd_survives_cost_2x
microadd_survives_cost_3x
```

Prohibited:

- future-return argmax side labels
- OOS-selected side, threshold, quota, or coverage target
- labels that replace active baseline side
- labels that reintroduce Stage3/Stage4 behavior

## Exact Gates

Validation eligibility gates:

```text
validation_microadd_entry_count > 0
validation_trades_per_day > validation_rank1_trades_per_day
validation_microadd_entries_per_day >= 0.10
validation_microadd_entries_per_day <= selected_microadd_entries_per_day_cap
validation_avg_leverage >= 1.50
validation_avg_leverage <= 1.80
validation_cost_1x_pnl > 0
validation_cost_2x_pnl > 0
validation_cost_3x_pnl > 0
validation_invariant_hard_violations == 0
```

Full OOS hard gates:

```text
baseline_reproduction_passed == true
row_limits_present == false
stage3_stage4_exclusion_flag == true
oos_pnl > 752.65
oos_mdd > -18.76
oos_trades_per_day > 6.02
oos_avg_leverage >= 1.50
oos_avg_leverage <= 1.80
oos_cost_1x_pnl > 0
oos_cost_2x_pnl > 279.36
oos_cost_3x_pnl > 75.84
oos_invariant_hard_violations == 0
oos_microadd_entry_count > 0
oos_trades_per_day > reproduced_baseline_trades_per_day
```

Invariant audit hard violations:

```text
active_side_reversal
active_created_side
active_notional_increase
active_leverage_increase
active_position_fraction_increase
microadd_when_rank1_active
microadd_when_position_open
microadd_side_without_vote
microadd_vote_conflict
microadd_notional_cap_violation
microadd_leverage_cap_violation
microadd_quota_violation
microadd_cooldown_violation
stage3_stage4_reintroduced
nonfinite_decision_value
negative_notional_or_leverage
```

## Search Grid

Coverage discovery grid:

```text
vote_strength_min: [0.08, 0.12, 0.16, 0.20, 0.28]
vote_source_count_min: [2, 3, 4]
vote_entropy_max: [0.55, 0.70, 0.85]
survival_prob_3x_min: [0.52, 0.58, 0.64, 0.70]
edge_floor_3x: [0.0002, 0.0006, 0.0010, 0.0018, 0.0030]
max_cvar_loss: [0.004, 0.007, 0.010, 0.014]
notional_cap: [0.03, 0.05, 0.075, 0.10, 0.15, 0.20]
microadd_leverage_ceiling: [1.0, 1.1, 1.2, 1.4]
entries_per_day_cap: [0.25, 0.50, 0.75, 1.00]
cooldown_bars: [6, 12, 24, 36, 72]
```

Active-row defense grid:

```text
defense_mode: [off, light]
defense_hazard_veto: [0.82, 0.90]
defense_edge_floor: [0.0000, 0.0006]
defense_max_cvar_loss: [0.035, 0.060]
active_leverage_ceiling: [2.0, 2.2, 2.5]
```

`defense_mode=off` must be included because rank-1 PnL is strong; micro-add should not hide behind reduced trade frequency.

## Selection Strategy

Stage A: Coverage discovery

- Evaluate micro-add configs on validation.
- Keep only configs with `validation_microadd_entry_count > 0`, `validation_microadd_entries_per_day >= 0.10`, and zero invariant violations.
- Report top rejection reasons if no config survives.

Stage B: Rank-1 feasibility selection

- From Stage A survivors, reject configs that fail validation trades/day increase, avg leverage, and cost survival.
- Rank survivors by:

```text
score =
    pnl_delta_vs_validation_rank1
  + 0.40 * trades_per_day_delta
  - 2.00 * max(0, abs(mdd_candidate) - abs(mdd_validation_rank1))
  - 0.50 * max(0, cost3_pnl_delta_negative)
  - 0.25 * microadd_turnover_penalty
```

- Select the highest score only if it is eligible. No fallback to a zero-microadd config is allowed.

Stage C: Full OOS confirmation

- Run with no row limits.
- Reproduce current rank-1 baseline first.
- Apply selected validation config once.
- Do not alter thresholds after seeing OOS.

## Required Report Sections

```text
baseline_contract
baseline_reproduction
stage3_stage4_exclusion_audit
validation_coverage_discovery
validation_grid_ranked
selected_config
rank1_baseline
rank1_microadd_v2
delta_vs_rank1
cost_stress
monthly
weekly
state_audit
leakage_audit
invariant_audit
microadd_audit
defense_audit
hard_gates
red_team_blockers
```

## Red Team Blocking Risks

- Baseline reproduction uses the older `467.64%` comparison report or includes Stage3/Stage4.
- Micro-add flat state uses row-level cash instead of exact position state.
- Validation selector falls back to zero micro-add.
- PnL gain comes from high churn that fails cost 2x/3x rank-1 thresholds.
- Hidden future side labels enter `microadd_vote_side`.
- OOS result influences threshold, quota, notional cap, or coverage selection.
