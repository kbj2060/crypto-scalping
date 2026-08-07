# AI PatchTSMixer Direction Core - 2026-05-30

## Status

First-stage AI direction experiment completed.

This experiment only adds the `ai_patch_*` direction-context family. It does
not overwrite existing `ai_dir_*`, `pred_patchtst`, `conf_patchtst`, `data/nf_*`,
M7, teacher, regime, or router artifacts.

## Generator

- `scripts/build_ai_patchmix_direction_core_20260530.py`
- Runner: `scripts/run_ai_patchmix_direction_core_20260530.sh`

Offline model:

- `ibm/patchtsmixer-etth1-pretrain`

Offline settings:

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
local_files_only=True
```

## Train/Score Chain

```text
2024 clean split -> fit heads -> 2025 ai_patch_* score
2025 clean split -> fit heads -> 2026 ai_patch_* score
```

Outputs:

- 2025: `tmp/causal_regen_20260516/ai_patchmix_direction_core_20260530_full/fit2024_score2025/ai_patchmix_direction_core_2025.csv`
- 2026: `tmp/causal_regen_20260516/ai_patchmix_direction_core_20260530_full/fit2025_score2026/ai_patchmix_direction_core_2026.csv`
- Metrics: `tmp/causal_regen_20260516/ai_patchmix_direction_core_20260530_full/oos_metrics.json`

## Inputs

The model consumes only clean upstream market-derived features and causal
derivatives. Forbidden inputs remain:

- `teacher_*`
- `m7_*`
- `a5dir_*`
- existing `ai_*`, `patchtst_*`, `tide_*`, `timesnet_*`, `dlinear_*`
- labels/targets/future/realized/PnL columns

Core per-row feature set:

```text
ret_1
ret_3
ret_6
ret_12
ret_24
atr14_pct
realized_vol_24
compression_ratio
last_funding_rate
funding_pressure
oi_change_rate
smart_money_flow
ofi_acceleration
net_taker_ratio
taker_acceleration
cvp_volume_imbalance
vwap_dev_48
lower_wick_ratio
upper_wick_ratio
```

PatchTSMixer receives 7 compressed sequence channels derived from the above
features:

```text
patch_ch_momentum
patch_ch_vol
patch_ch_funding_oi
patch_ch_flow
patch_ch_cvp
patch_ch_vwap_wick
patch_ch_wick_balance
```

CatBoost direction heads use the 19 core features plus 16 PatchTSMixer sequence
embedding dimensions.

## Label Objective

Horizons:

```text
h12, h24, h48
```

Label classes:

```text
0 = flat
1 = down
2 = up
```

Score formula:

```text
long_score  = long_MFE  - 0.55 * long_MAE  - 0.00055
short_score = short_MFE - 0.55 * short_MAE - 0.00055
edge_floor  = max(0.0012, 0.0011, atr14_pct * 0.22)
margin      = 0.00035
```

## Output Columns

For each horizon:

```text
ai_patch_h{h}_p_flat
ai_patch_h{h}_p_down
ai_patch_h{h}_p_up
ai_patch_h{h}_edge
ai_patch_h{h}_conf
ai_patch_h{h}_entropy
```

Cross-horizon summaries:

```text
ai_patch_consensus
ai_patch_edge_mean
ai_patch_risk_adj_edge
```

Total output features: `21`.

## OOS Direction Diagnostics

2025 score, model fit on 2024:

| Horizon | Balanced Accuracy | OVR AUC | Edge/Fwd Ret Corr |
|---|---:|---:|---:|
| h12 | 0.3973 | 0.6084 | 0.0013 |
| h24 | 0.3830 | 0.5838 | 0.0096 |
| h48 | 0.3395 | 0.5340 | 0.0125 |

2026 score, model fit on 2025:

| Horizon | Balanced Accuracy | OVR AUC | Edge/Fwd Ret Corr |
|---|---:|---:|---:|
| h12 | 0.4854 | 0.6492 | -0.0065 |
| h24 | 0.3655 | 0.6035 | -0.0110 |
| h48 | 0.3381 | 0.6063 | 0.0771 |

## Interpretation

- `h12` is the best first candidate for entry/meta direction context.
- `h24` and `h48` have usable ranking signal by AUC, but their flat/no-trade
  handling is weak. They should not be standalone direction owners.
- The family should be tested as an entry/meta feature block before promotion:
  first `ai_patch_h12_*`, then add cross-horizon summaries, then compare adding
  h24/h48.
- This experiment does not prove live profitability. It only shows that a
  causal offline HF sequence feature can produce non-trivial 2026 direction
  ranking signal.

