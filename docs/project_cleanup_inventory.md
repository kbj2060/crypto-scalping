# Project Cleanup Inventory

Last updated: 2026-04-24 KST

## Current Main Pipeline

The active pipeline is now treated as:

1. Build / split datasets with `pipeline/build_rl_dataset.py`
2. Train and load M7 supervised / unsupervised artifacts
3. Augment RL rows with M7 outputs through `pipeline/augment_m7_dataset.py`
4. Train / evaluate primary DSAC through `ensemble/train_rl_dsac_agent.py`
5. Run live execution through `trading_bot.py`

## Keep

| Area | Paths | Reason |
|---|---|---|
| Live bot | `trading_bot.py`, `run_live_collectors.py` | Current runtime entrypoints |
| Live helpers | `enhanced_trading_engine.py`, `microstructure_scanner.py`, `playbook_router.py`, `polymarket_engine.py`, `tail_risk_interceptor.py` | Imported by live bot / collectors |
| Shared core | `core/`, `features/`, `strategies/`, `common/` | Runtime and feature logic |
| Unified pipeline | `pipeline/` | Current orchestration layer |
| M7 models | `ensemble/supervised/`, `ensemble/unsupervised/`, `ensemble/trend_xgb/`, `ensemble/seven_model_ensemble.py` | Current M7 training / inference layer |
| DSAC | `ensemble/train_rl_dsac_agent.py`, `ensemble/rl_continuous_common.py`, `ensemble/rl_runtime_primitives.py` | Primary RL and shared HMM / feature helpers |
| Active artifacts | `data/ensemble/ckpt/`, `data/ensemble/supervised/`, `data/ensemble/unsupervised/`, `data/trend_xgb/` | Current model weights / fitted artifacts |
| Active year split | `data/splits/year_oos/` | Current 2024/2025/2026 train and OOS CSVs |
| Reports | `data/ensemble/reports/` | Compact final comparison reports |

## Cleaned In This Pass

Moved to `backups/cleanup_unused_20260424_005157`:

| Original path | Reason |
|---|---|
| `data1/` | Old duplicated dataset tree; ignored by git and not referenced by active pipeline |
| `data/ensemble/_feature_prune_backup_20260422/` | Old feature-prune artifact backup |
| `data/ensemble/_null_meta/` | Temporary null meta artifacts |
| `data/ensemble/_tmp_disable_hdbscan/` | Temporary HDBSCAN-disable artifacts |
| `data/ensemble/_tmp_disable_mtl/` | Temporary MTL-disable artifacts |
| `data/ensemble/_tmp_disable_mtl_real/` | Temporary MTL-disable artifacts |
| `data/ensemble/backup/` | Old in-place CSV backup directory |
| `data/ensemble/recheck_unsup/` | Old unsupervised recheck artifacts |
| `best_dsac_agents.pth` | Root-level stale checkpoint; canonical checkpoint is under `data/ensemble/ckpt/` |
| `output.txt` | Temporary status dump |
| `check_duckdb_temp.py` | Temporary one-off script |

Removed directly:

| Path | Reason |
|---|---|
| `__pycache__/` directories outside `backups/`, `venv-win/`, `.git/` | Generated Python cache |
| `lightning_logs/` | Generated training logs; not part of current runtime |

Moved to `backups/cleanup_removed_legacy_dirs_20260424_012618`:

| Original path | Reason |
|---|---|
| `analysis/` | Old exploratory analysis outputs; not part of current M7/DSAC/live pipeline |
| `common/` | No active main-pipeline use after removing legacy tests |
| `macroMFT/` | Legacy PPO/MacroHFT stack; not part of current M7/DSAC pipeline |
| `test/` | Legacy tests target old `core/common/macroMFT` assumptions and are not aligned with the current unified pipeline |

## Next Cleanup Candidates

These should not be deleted blindly. They are not on the current main DSAC
pipeline, but have either historical value, test references, or live-adjacent
uncertainty.

| Priority | Candidate | Current assessment | Suggested action |
|---|---|---|---|
| High impact | `venv-win/` | Very large local virtualenv, while current work uses conda `quant_ai` | Delete if no Windows-side scripts depend on it |
| Medium | `data/ensemble/metrics/` | Large collection of old experiment JSONs | Keep only summaries and move raw trial outputs to archive |
| Medium | archived legacy tests | The old `test/` suite was removed from the active tree | Rebuild a smaller current-pipeline smoke test suite later |
| Medium | old experiment scripts in `scripts/` | Many are one-off backtests / ablations | Split into `scripts/active/` and `scripts/archive/`, or move to backup |
| Low | TD3/PPO docs | Legacy architecture docs, not current DSAC pipeline | Archive old docs after `docs/unified_pipeline_design.md` is treated as source of truth |
| Low | `logs/tensorboard/`, `logs/env_audit/` | Generated logs | Remove if no audit trail is needed |

## Verification

Compiled successfully after cleanup:

- `trading_bot.py`
- `run_live_collectors.py`
- `pipeline/build_rl_dataset.py`
- `pipeline/run_train.py`
- `pipeline/augment_m7_dataset.py`
- `ensemble/train_rl_dsac_agent.py`
- `ensemble/ensemble_router.py`
- `ensemble/seven_model_ensemble.py`
- `ensemble/supervised/live_supervised_hub.py`
- `ensemble/unsupervised/live_unsupervised_hub.py`
- `features/engineering.py`
- `features/registry.py`
- `features/schema.py`
