# Omega4.4 RL Risk Sidecar v1 Full Test - 2026-06-23

## Status

- Model id: `omega4_4_rl_risk_sidecar_v1_full_20260623`
- Status: `research_candidate_not_omega4_4_baseline_upgrade`
- Baseline reference: `omega4_4_topdown_reproducible_architecture_baseline_20260623`
- Red-team verdict: `REDTEAM_PASS_CLEAN_RESEARCH_FULL_TEST_NOT_BASELINE_UPGRADE`

## Scope

Omega4.4 parent, quality gate, exit head, and ATR price-move SLTP are frozen. This experiment replaces only the HGB risk sidecar with offline RL-style sizing policies.

## Algorithms Tested

- `bandit_qnet`: full-information contextual bandit Q-network over 16 risk actions
- `iql_awac`: discrete IQL/AWAC-style behavior-regularized policy
- `td3_bc_continuous`: continuous TD3+BC-style actor/critic over margin and leverage
- `dsac_contextual`: contextual distributional SAC-style discrete policy

## Action Space

```text
margin_fraction = [0.06, 0.12, 0.20, 0.28]
leverage        = [1.0, 1.5, 2.0, 2.5]
notional        = margin_fraction * leverage
```

## Reward

```text
reward = log(1 + net_per_notional * margin_fraction * leverage)
       - 0.5 * tail_excess
       - 0.25 * liquidation_excess
```

OOS is excluded from policy selection and is readout only.

## Ranking

| Policy | Val PnL | Val MDD | Val Utility | OOS PnL | OOS MDD | OOS Utility | OOS Full Replay |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `iql_awac` | `+23.62%` | `-5.59%` | `0.210432` | `+15.40%` | `-5.54%` | `0.143201` | `+15.07%` |
| `td3_bc_continuous` | `+20.14%` | `-6.68%` | `0.179895` | `+15.04%` | `-6.01%` | `0.140030` | `+18.19%` |
| `dsac_contextual` | `+19.72%` | `-6.88%` | `0.178378` | `+15.87%` | `-5.54%` | `0.147318` | `+15.87%` |
| `bandit_qnet` | `+12.55%` | `-10.13%` | `0.061846` | `+22.48%` | `-5.50%` | `0.181924` | `+25.24%` |

## Decision

Selected by validation-only log-risk: `iql_awac`.

- Selected validation: `+23.62%`, MDD `-5.59%`, utility `0.210432`
- Selected OOS readout: `+15.40%`, MDD `-5.54%`, utility `0.143201`
- Omega4.4 HGB baseline OOS: `+22.21%`

Do not promote this v1 RL sidecar as the Omega4.4 baseline. Keep it as a research candidate.
