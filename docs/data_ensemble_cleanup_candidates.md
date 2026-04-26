# Data / Ensemble Cleanup Candidates

Last updated: 2026-04-24 KST

## 기준

현재 메인 파이프라인은 `trading_bot.py` + M7 + Primary DSAC 기준이다.

유지 기준:

- `trading_bot.py` 런타임에서 직접 로드한다.
- `pipeline/`, `features/`, `ensemble/supervised`, `ensemble/unsupervised`, `ensemble/train_rl_dsac_agent.py`의 현재 재학습 경로에 필요하다.
- 현재 선택된 2026 OOS 결과를 재현하는 데 필요하다.

## 유지해야 하는 파일 / 폴더

| Path | 이유 |
|---|---|
| `data/ensemble/ckpt/best_dsac_agents.pth` | 라이브 Primary DSAC 체크포인트 |
| `data/ensemble/ckpt/dsac_checkpoint.pth` | Primary DSAC 재개 학습용 체크포인트 |
| `data/ensemble/ckpt/dsac_train_config_latest.json` | 현재 DSAC 학습 설정 |
| `data/ensemble/ckpt/hmm_init_cache_dsac.npz` | 현재 DSAC HMM 캐시 |
| `data/ensemble/supervised/*.pkl`, `*.json` | 현재 M7 supervised 모델 |
| `data/ensemble/unsupervised/gmm_volatility.*` | 현재 M7 unsupervised 모델 |
| `data/ensemble/unsupervised/isolation_forest.*` | 현재 M7 unsupervised 모델 |
| `data/ensemble/unsupervised/vae_anomaly.*` | 현재 M7 unsupervised 모델 |
| `data/splits/year_oos/training_features_2024.csv` | 2024 M7 재학습 입력 |
| `data/splits/year_oos/rl_base_2024.csv` | 2024 M7 재학습 보조 RL 입력 |
| `data/splits/year_oos/training_features_2025.csv` | 2025 RL/M7 분석 입력 |
| `data/splits/year_oos/rl_base_2025.csv` | 2025 RL base |
| `data/splits/year_oos/rl_training_2025_m7.csv` | 현재 DSAC 학습 CSV |
| `data/splits/year_oos/training_features_2026_rebuilt.csv` | 2026 OOS feature CSV |
| `data/splits/year_oos/rl_training_2026_m7_supervised_redesign_clean.csv` | 현재 DSAC 2026 OOS CSV |
| `data/live/` | 실시간 런타임 상태 / DuckDB |
| `data/nf/` | 현재 PatchTST 로더가 `NeuralForecast.load(path="data/nf")`로 전체 model pack을 로드함 |
| `ensemble/train_rl_dsac_agent.py` | Primary DSAC 학습/라우팅 |
| `ensemble/rl_runtime_primitives.py` | DSAC가 HMM/MTF helper를 재사용함 |
| `ensemble/rl_continuous_common.py` | DSAC 공통 RL env/buffer |
| `ensemble/rl_runtime_primitives.py` | RL 공통 fallback helper |
| `ensemble/seven_model_ensemble.py` | M7 inference/augmentation |
| `ensemble/ensemble_router.py` | PatchTST runtime wrapper |
| `ensemble/supervised/` | 현재 M7 supervised training/live hub |
| `ensemble/unsupervised/` | 현재 M7 unsupervised training/live hub |
| `ensemble/artifact_utils.py`, `ensemble/optuna_helper.py` | M7 train scripts가 사용 |

## 바로 아카이브 가능성이 높은 후보

Status: archived from active tree to `backups/cleanup_data_ensemble_high_conf_20260424_013807`.

| Path | Size approx | 이유 |
|---|---:|---|
| `data/rl_training_data_full copy.csv` | 239M | 중복 copy |
| `data/rl_training_data_full.csv.bak_20260403_175335` | 238M | 과거 백업 |
| `data/rl_training_data_full.csv.bak_20260404` | 110M | 과거 백업 |
| `data/rl_training_data_pruned.csv` | 217M | 과거 prune 실험 산출물 |
| `data/test/` | 88M | 로컬/failfast 테스트 데이터. `trading_bot.py` 실시간 `use_local=False`에는 불필요 |
| `data/trend_xgb/trend_xgb.pkl` | 24M | 오래된 중복 Trend XGB. 현재 모델은 `data/ensemble/supervised/trend_xgb.pkl` |
| `data/trend_xgb/training_results.json` | small | 위 중복 모델의 결과 파일 |
| `data/rl_training_data_latest.csv` | 5.1M | 구형/실험 스크립트 기본값. 현재 선택 DSAC CSV 아님 |
| `data/rl_latest7d.csv` | 0.9M | 단기 실험 데이터 |
| `data/ridge_model.pkl`, `data/ridge_model.joblib` | small | 현재 M7/DSAC/live 경로에서 미사용 |
| `data/ensemble/cache/poly_api_feat_*.parquet` | 5.7M | 과거 Polymarket fine-tune cache |
| `data/ensemble/metrics/proposed_tune_*` | 13M+ | 과거 파라미터 탐험 raw trial outputs |
| `data/ensemble/metrics/blueprint_ab_*` | small | 과거 blueprint AB raw outputs |
| `data/ensemble/metrics/dual_profile_tuning/` | small | 과거 dual profile tuning raw outputs |
| `ensemble/diagnose_side_bias.py` | code | 일회성 진단 스크립트 |
| `ensemble/train_rl_sac_agent.py` | code | SAC 경로. 현재 메인 DSAC 파이프라인에서 미사용 |
| `ensemble/train_rl_meta_agent.py` | code | Meta RL 경로. 현재 메인 DSAC 파이프라인에서 미사용 |
| `ensemble/train_rl_meta_gating.py` | code | Meta gating 경로. 현재 메인 DSAC 파이프라인에서 미사용 |
| `ensemble/unsupervised/train_hdbscan_regime.py` | code | HDBSCAN은 현재 M7에서 제거/비활성 방향 |
| `ensemble/msaf_formula.py` | code | 과거 공식 백테스트 전용. live import 없음 |

## 조건부 제거 후보

| Path | 조건 |
|---|---|
| `data/rl_training_data_full.csv` | `pipeline/build_rl_dataset.py` 재생성 가능하고, 현재 재학습을 year-split CSV만으로 진행하기로 확정하면 아카이브 가능. 단, 많은 레거시 스크립트의 기본값이라 삭제 시 과거 스크립트는 깨질 수 있음 |
| `data/training_features_5m.csv` | 위와 동일. 현재 unified pipeline 기본 입력이므로 당장 삭제는 비추천 |
| `data/btc_5m_1year.csv`, `data/eth_5m_1year.csv` | `core/data_collector.py` fallback/수집 로직에서 참조. 실시간 API만 쓴다면 아카이브 가능 |
| `data/TOTAL_ETHUSDT_metrics.csv`, `data/TOTAL_ETHFIUSDT_fundingRate.csv`, `data/ETHUSDT_FR_History.csv` | 과거 raw/funding helper 데이터. 현재 M7/DSAC live 경로 직접 사용은 낮음 |
| `data/api_execution_30d_5m.csv` | 지정가/실행 overlay 실험용. 현재 market 실행이면 아카이브 가능 |
| `data/ensemble/reports/chronos_bolt_tiny_signal_probe.json` | 실험 근거 보존용. 최종 비교 근거가 필요 없으면 아카이브 |
| `data/ensemble/reports/timemixer_signal_probe.json` | 실험 근거 보존용. 최종 비교 근거가 필요 없으면 아카이브 |
| `data/ensemble/reports/quant_env_*_20260424.json` | 환경 비교 근거. 정리 후 아카이브 가능 |
| `data/ensemble/ckpt/best_dsac_agents_redesign_clean_legacy.pth` | `best_dsac_agents.pth`와 같은 모델임. 백업 폴더에 이미 있으면 active tree에서는 제거 가능 |
| `data/ensemble/ckpt/dsac_checkpoint_redesign_clean_legacy.pth` | 위와 같은 legacy-named duplicate |
| `data/ensemble/ckpt/hmm_cache_redesign_clean_legacy.npz` | 위와 같은 legacy-named duplicate |
| `data/ensemble/ckpt/dsac_train_config_redesign_clean_legacy.json` | 위와 같은 legacy-named duplicate |
| Specialist long/short files | Archived from active tree after removing specialist display/runtime |

## 주의: `data/nf/`

`data/nf`는 현재 `PatchTST`만 쓰고 싶어도 `NeuralForecast.load(path="data/nf")`가 전체 model pack을 읽는다.

현재 로드 결과:

- `PatchTST`
- `NHITS`
- `TiDE`
- `iTransformer`

따라서 `iTransformer_0.ckpt`, `NHITS_0.ckpt`, `TiDE_0.ckpt`를 단독 삭제하면 `NeuralForecast.load`가 깨질 수 있다. 정리하려면 먼저 PatchTST-only pack을 새로 저장하거나 `ensemble/ensemble_router.py`의 로더를 PatchTST-only artifact 구조로 바꿔야 한다.

## 추천 정리 순서

1. `바로 아카이브 가능성이 높은 후보`를 백업 폴더로 이동
2. `data/ensemble/metrics`는 최종 summary/report만 남기고 raw trial output을 archive
3. PatchTST-only 저장 포맷을 만든 뒤 `data/nf`에서 비-PatchTST 모델 제거

## Current Minimal Active Set

After cleanup, the active `data/ensemble` tree keeps only:

- `ckpt/best_dsac_agents.pth`
- `ckpt/dsac_checkpoint.pth`
- `ckpt/dsac_train_config_latest.json`
- `ckpt/hmm_init_cache_dsac.npz`
- `metrics/param_ensemble_result.json`
- `metrics/param_ensemble_lowfreq_grid.json`
- `metrics/param_ensemble_lowfreq_highpnl.json`
- `reports/eval_2026_redesign_clean_legacy.json`
- `supervised/*` current M7 artifacts
- `unsupervised/*` current M7 artifacts
- `dsac_live_state.json`

Archived from active tree to:

- `backups/cleanup_data_ensemble_minimal_20260424_014641`
- `backups/cleanup_specialists_removed_20260424_015438`
