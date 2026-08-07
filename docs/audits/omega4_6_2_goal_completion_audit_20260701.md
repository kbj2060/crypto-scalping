# Omega 4.6.2 Goal Completion Audit - 2026-07-01

- Verdict: `GOAL_COMPLETION_EVIDENCED`
- Completion pass: `True`
- Created at KST: `2026-07-01T09:03:01.006021+09:00`
- Registered candidates checked: `36`

## Checks

- `deadline_reached`: `True`
- `core_reports_exist`: `True`
- `all_registered_candidates_have_reports`: `True`
- `all_registered_candidates_have_redteam_json`: `True`
- `all_registered_redteams_research_pass`: `True`
- `best_frontier_improves_pnl_and_reduces_hold_vs_baseline`: `True`
- `sub5h_candidate_keeps_pnl_contract`: `True`
- `ultra_short_candidate_keeps_pnl_contract`: `True`
- `roll12_oos_candidate_recorded`: `True`
- `full_live_blockers_disclosed`: `True`

## Frontier Evidence

| Model | Validation PnL | OOS PnL | Validation Avg Hold | OOS Avg Hold | Max Hold |
| --- | ---: | ---: | ---: | ---: | ---: |
| Baseline: `omega4_6_2_loss_cluster_governor_v5_fine_exposure_20260701` | `274.8817%` | `138.4476%` | `56.6123h` | `60.5577h` | `90.0h` |
| Best PnL/Hold: `omega4_6_2_v5_roll8_side_specific_two_stage_exposure_oos_balanced_20260701` | `462.1947%` | `302.2096%` | `5.8358h` | `6.4733h` | `8.0h` |
| Sub-5h: `omega4_6_2_v5_roll6_side_specific_two_stage_exposure_hold_compressed_20260701` | `347.9707%` | `235.2600%` | `4.9349h` | `4.9863h` | `6.0h` |
| Ultra-short: `omega4_6_2_v5_roll2_side_specific_two_stage_exposure_oos_max_20260701` | `183.4355%` | `161.1111%` | `1.9297h` | `1.9107h` | `2.0h` |
| Roll12 OOS: `omega4_6_2_v5_roll12_side_specific_oos_max_20260701` | `330.0475%` | `178.5726%` | `9.0355h` | `9.8945h` | `12.0h` |

## Full-Live Disclosure

- Research red-team passed for all registered candidates.
- Full-live promotion remains blocked by runtime-native replay adapter and fresh holdout requirements; see the runtime blocker report.

## Artifacts

- JSON: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega4_6_2_goal_completion_20260701/goal_completion_audit_20260701.json`
- Upgrade loop: `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_upgrade_loop_20260701.md`
- Runtime blockers: `/home/llewyn/crypto-scalping/docs/audits/omega4_6_2_runtime_wiring_blockers_20260701.md`
