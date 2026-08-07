# Alpha3 DSAC Exit Owner Contract - 2026-05-15 KST

## Scope

This experiment adapts the existing `ensemble/train_rl_dsac_agent.py` DSAC stack to Alpha3 exit ownership.

Alpha3 entry stack is frozen:

- HGB parent
- Alpha2.1 teacher gate
- V21.2 jackpot add-on
- frozen V27 scout
- existing entry execution

The DSAC layer acts only while a position is active. It can emit `hold` or close 100% of the position through one reduce-only exit placement arm.

## DSAC Components Reused

The implementation reuses `DSACAgent` from `ensemble/train_rl_dsac_agent.py`:

- tanh-Gaussian actor
- distributional twin quantile critic
- CVaR actor objective
- adaptive pessimism
- adaptive entropy
- CQL regularization
- critic target soft update
- critic variance weighting
- primacy soft reset
- REDO-style dormant unit rejuvenation
- gradient clipping

Script: `scripts/train_eval_alpha3_dsac_exit_owner_20260515.py`

## Input / Output

Input state:

- Alpha3 active-position state
- owner (`v21_2` or `deep_alpha`)
- side, hold bars, unrealized PnL, MFE, MAE
- notional, parent notional
- TP, SL, max hold, effective TP/SL
- entry edge, entry volatility anchor
- Alpha3 decision frame
- frozen V27 q outputs
- current causal feature frame

Continuous DSAC actor output is mapped to:

| Actor range | Runtime action |
|---:|---|
| `[-0.15, 0.15]` | hold |
| `< -0.65` | `baseline_exit2_pen05` |
| `[-0.65, -0.35)` | `exit0_pen0` |
| `[-0.35, -0.15)` | `exit1_pen0` |
| `(0.15, 0.45)` | `exit2_pen0` |
| `[0.45, 0.75)` | `exit3_pen0` |
| `>= 0.75` | `exit4_pen0` |

At a forced base Alpha3 exit event, `hold` is converted to the selected fallback exit arm.

## Training

- Train split: `2025-01-01..2025-09-30`
- Selection split: `2025-10-01..2025-12-31`
- 2026/current eval: report only after selection
- Selection uses 2026: `false`
- States: `35,717`
- Episodes: `861`
- Replay samples pushed: `249,158`
- Device: `cuda`

Replay construction:

- A DP-labeled Alpha3 exit-owner replay is built from active position states.
- Each state is expanded across `hold` and all reduce-only exit arms.
- Reward is the DP target scaled/clipped for DSAC.
- This is an offline contextual DSAC replay, not live online interaction.

## Result

Current eval file warning: active `v31.DEFAULT_EVAL` currently covers `2026-01-01..2026-02-28 16:00`, not the older full canonical report horizon.

| Candidate | Cost1 PnL | Cost1 MDD | Cost2 PnL | Cost3 PnL | Score |
|---|---:|---:|---:|---:|---:|
| Alpha3 baseline exit2 pen05 | +358.84% | -27.59% | +283.14% | +215.42% | 541.22 |
| Fixed front-run exit4 pen0 | +369.62% | -27.14% | +285.00% | +218.20% | 553.83 |
| DSAC exit owner selected | +358.84% | -27.59% | +283.14% | +215.42% | 541.22 |

2025Q4 selection:

- Best DSAC runtime: `dsac_exit_owner_minhold12_fb_baseline_exit2_pen05`
- Best DSAC validation score: `-128.93`
- Best overall validation runtime: `fixed_front_run_exit4_pen0`
- Fixed front-run validation score: `-106.33`

## Decision

Do not promote the DSAC exit owner.

The DSAC model trained successfully with the desired architecture and training techniques, but the learned early-exit behavior was weaker than fixed `exit4_pen0` on 2025Q4. On current OOS replay, the selected DSAC runtime effectively collapsed to baseline behavior.

The likely failure mode is not model capacity. It is the action/reward/replay formulation:

- no partial close,
- no true market-close action distinct from limit fallback,
- no TWAP/slicing action,
- no real L2 queue/partial-fill labels,
- DP replay is still derived from OHLCV touch proxy outcomes.

## Artifacts

- Script: `scripts/train_eval_alpha3_dsac_exit_owner_20260515.py`
- Model: `data/ensemble/supervised/alpha3_dsac_exit_owner_20260515/dsac_exit_owner.pt`
- Summary: `data/ensemble/reports/alpha3_dsac_exit_owner_20260515_summary.json`
- Grid: `data/ensemble/reports/alpha3_dsac_exit_owner_20260515_grid.csv`
- Audit: `data/ensemble/reports/alpha3_dsac_exit_owner_20260515_audit.json`
- Dataset: `data/ensemble/reports/alpha3_dsac_exit_owner_20260515_dataset.json`
