# AI Direction Feature Retrain Audit - 2026-05-28

## Current Logic Diagnosis

Current `ai_dir_*` features are generated in `ensemble/ensemble_router.py`.

- `PatchTSTForecaster` loads `data/nf_patchtst`.
- The training contract says the role is direction and the target is `y_dir_edge`.
- Runtime output is a single scalar edge forecast.
- `_apply_refined_batch_logic()` converts that scalar into pseudo probabilities:
  - `ai_dir_p_up`
  - `ai_dir_p_down`
  - `ai_dir_p_flat`
  - `ai_dir_edge`
  - `ai_dir_entropy`

This means `ai_dir_*` is not a calibrated directional classifier. It is a scalar NeuralForecast forecast mapped into probabilities after the fact.

The other AI families are not direct direction models:

- `TiDE`: adverse excursion risk / reward-risk context
- `TimesNet`: VWAP anchor reversion/overheat context
- `DLinear`: flow persistence/exhaustion context

## Main Weakness

The old PatchTST direction model was trained on a 2024-only `y_dir_edge` target, but runtime uses a past-bar proxy as the observed target series. This avoids lookahead during inference, but it creates target/runtime mismatch. The model also uses a narrow exogenous set and does not include the new directional-alpha feature block.

## New Direction Dataset

Script:

- `scripts/retrain_ai_direction_features_20260528.py`

Output roots:

- Invalid first run: `tmp/causal_regen_20260516/ai_direction_retrain_20260528_v1`
- No-leak aggressive labels: `tmp/causal_regen_20260516/ai_direction_retrain_20260528_v2_noleak`
- No-leak strict labels: `tmp/causal_regen_20260516/ai_direction_retrain_20260528_v3_strict_noleak`

The first run is marked invalid because label-derived `dir_long_score`/`dir_short_score` were initially allowed into the feature matrix. The script now rejects every `dir_*` generated label column from model inputs.

## Label Contract

For each row:

- Compute future high/low/close over a fixed horizon.
- Compute long and short MFE/MAE scores.
- Score formula:
  - `score = MFE - mae_penalty * MAE - cost`
- Dynamic edge floor:
  - `max(min_edge, 2 * cost, atr_pct * atr_mult)`
- Assign:
  - `0 = flat`
  - `1 = down`
  - `2 = up`

No active/live `data/nf_*` artifact was overwritten. New outputs use `ai_dir_v2_*`.

## Results

### v2_noleak - Aggressive

Parameters:

- `min_edge=0.0012`
- `atr_mult=0.22`
- `mae_penalty=0.55`
- `direction_margin=0.00035`

Label distribution:

- 2024 train: flat 5,024 / down 50,433 / up 49,899
- 2025 train: flat 3,844 / down 50,677 / up 50,519
- 2026 score: flat 1,087 / down 8,209 / up 7,577

Metrics:

- Fit 2024 -> score 2025: OVR AUC 0.5939, edge IC vs fwd return 0.0434
- Fit 2024 -> extra 2026: OVR AUC 0.6434, edge IC vs fwd return 0.1000
- Fit 2025 -> score 2026: OVR AUC 0.6173, edge IC vs fwd return -0.0270

Interpretation:

- Too few flat labels.
- Useful as broad direction context, not an entry owner.

### v3_strict_noleak - Stricter

Parameters:

- `min_edge=0.0025`
- `atr_mult=0.45`
- `mae_penalty=0.85`
- `direction_margin=0.0010`

Label distribution:

- 2024 train: flat 25,766 / down 40,193 / up 39,397
- 2025 train: flat 22,194 / down 41,618 / up 41,228
- 2026 score: flat 4,228 / down 6,667 / up 5,978

Metrics:

- Fit 2024 -> score 2025: OVR AUC 0.5822, edge IC vs fwd return 0.0275
- Fit 2024 -> extra 2026: OVR AUC 0.6329, edge IC vs fwd return 0.0654
- Fit 2025 -> score 2026: OVR AUC 0.6218, edge IC vs fwd return -0.0298

Interpretation:

- Better label balance.
- Probabilities are less overconfident than the invalid first run.
- Still not strong enough to become a standalone direction owner.

## Recommendation

Use `ai_dir_v2_*` only as a secondary direction-context feature for Alpha7/Alpha6 meta layers.

Do not replace the active `ai_dir_*` contract yet. The current best use is:

- `ai_dir_v2_edge`: weak directional context
- `ai_dir_v2_entropy`: uncertainty / veto context
- `ai_dir_v2_p_flat`: no-trade context

Next useful test:

1. Merge `fit2024_score2025/score_primary.csv` into the 2025 feature frame.
2. Merge `fit2025_score2026/score_primary.csv` into the 2026 feature frame.
3. Run Alpha7.1 feature routing with `ai_dir_v2_*` added to the entry/meta layer only.
4. Keep existing `TiDE/TimesNet/DLinear` risk-context features unchanged.
