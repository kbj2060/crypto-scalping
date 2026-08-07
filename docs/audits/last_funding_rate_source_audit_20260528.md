# `last_funding_rate` Source Audit - 2026-05-28

## Verdict

`last_funding_rate` was **not clean** in the historical year split files used by many experiments. The active year split files have now been regenerated from ETHUSDT-only backward-asof funding and revalidated.

This is not a normal modeling weakness. It is a feature-generation bug:

1. `data/splits/year_oos/training_features_2024.csv` front-fills the next ETHUSDT funding event.
2. `data/splits/year_oos/training_features_2025.csv` and `data/splits/year_oos/training_features_2026_rebuilt.csv` mostly front-fill the next ETHFIUSDT funding event, not ETHUSDT.
3. `data/training_features_5m.csv` currently matches backward-asof ETHUSDT funding for its 2026-03/2026-04 range.

Directional screens were rerun after regeneration. Funding-family features can be analyzed again from the regenerated split files, but any model artifact trained before this remediation remains suspect if it consumed the contaminated funding columns.

## Code Path

- Source loader: `scripts/update_features.py`
- Historical wrong source path before patch:
  - `data/TOTAL_ETHFIUSDT_fundingRate.csv`
- Correct primary source:
  - `binance_data/funding_rate/ETHUSDT-fundingRate-*.zip`
- Merge method in current code:
  - `pd.merge_asof(..., direction="backward")`

The current merge direction is causal. The contaminated split files were generated earlier or externally with forward alignment and the wrong legacy funding CSV.

## Evidence

Comparison method:

- Build a pure ETHUSDT funding series from `binance_data/funding_rate/ETHUSDT-fundingRate-*.zip`.
- Build an ETHFIUSDT legacy series from `data/TOTAL_ETHFIUSDT_fundingRate.csv`.
- For each feature timestamp, compare current `last_funding_rate` to both previous and next funding events.

| File | Previous ETHUSDT match | Next ETHUSDT match | Previous ETHFI match | Next ETHFI match | Verdict |
|---|---:|---:|---:|---:|---|
| `data/splits/year_oos/training_features_2024.csv` | 31.385% | 99.910% | 0.000% | 0.090% | future ETHUSDT funding leak |
| `data/splits/year_oos/training_features_2025.csv` | 0.000% | 0.000% | 32.447% | 100.000% | future ETHFIUSDT funding contamination |
| `data/splits/year_oos/training_features_2026_rebuilt.csv` | 0.213% | 47.440% | 15.908% | 100.000% | future ETHFIUSDT funding contamination |
| `data/training_features_5m.csv` | 100.000% | 0.321% | 69.854% | 0.321% | causal ETHUSDT funding for this range |

After regeneration/replacement:

| File | Previous ETHUSDT match | Max abs diff | Verdict |
|---|---:|---:|---|
| `data/splits/year_oos/training_features_2024.csv` | 100.000% | 0.0 | clean causal ETHUSDT funding |
| `data/splits/year_oos/training_features_2025.csv` | 100.000% | 0.0 | clean causal ETHUSDT funding |
| `data/splits/year_oos/training_features_2026_rebuilt.csv` | 100.000% | 0.0 | clean causal ETHUSDT funding |
| `data/splits/year_oos/rl_base_2025.csv` | 100.000% | 0.0 | direct funding columns patched |
| `data/splits/year_oos/rl_training_2026_m7_supervised_redesign_clean.csv` | 100.000% | 0.0 | direct funding columns patched |

Example from 2024:

| timestamp | feature value | previous ETHUSDT | next ETHUSDT |
|---|---:|---:|---:|
| `2024-01-01 00:05:00` | `0.000311` | `0.000279` | `0.000311` |

Example from 2025:

| timestamp | feature value | previous ETHFI | next ETHFI | previous ETHUSDT |
|---|---:|---:|---:|---:|
| `2025-01-01 00:05:00` | `0.000050` | `0.000050` | `0.000050` | `0.000100` |

Example from 2026:

| timestamp | feature value | previous ETHFI | next ETHFI | previous ETHUSDT |
|---|---:|---:|---:|---:|
| `2026-01-01 08:05:00` | `0.000050` | `0.000044` | `0.000050` | `0.000088` |

## Impacted Feature Families

These were `BUG_RISK_REGENERATE` before remediation and were regenerated/patched in active split CSVs:

- `last_funding_rate`
- `funding_abs`
- `funding_pressure`
- `funding_roc_12`
- `funding_roc_48`
- `funding_roc_288`
- `funding_z_score`
- `funding_price_divergence`
- `long_squeeze_risk`
- `short_squeeze_risk`
- `squeeze_power`
- any downstream feature using funding sign or funding magnitude

## Fix Applied

`scripts/update_features.py` now uses ETHUSDT monthly funding zips only and fails fast if non-ETHUSDT funding zip files appear in the funding directory.

## Required Next Step

Retrain or rescore derived artifacts that were trained/generated before this remediation if they consumed funding-derived inputs, especially M7/teacher/regime sidecars and any policy model that embeds old funding-family behavior.

## Completed Follow-Up - 2026-05-29

The first derived-artifact remediation pass is complete. M7 active artifacts were retrained and M7 CSVs were rescored; `regime4_pred_tft_h12_nomdjd_all74_20260517` sidecars were regenerated under the same all74 contract; and the Alpha5 `a5dir` / router chain was rebuilt from clean 2024/2025 unified inputs.

Follow-up record:

- `docs/audits/funding_clean_retrain_rescore_20260529.md`

Do not treat older DSAC, Alpha6, Alpha7, or cached unified artifacts as clean unless their manifests point to the clean funding run or they are explicitly retrained/rescored.
