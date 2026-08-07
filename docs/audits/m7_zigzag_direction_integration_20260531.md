# M7 ZigZag Direction Integration - 2026-05-31

## Scope

Integrated the top two ZigZag 3-class direction/action candidates into M7-named feature files.

The source models are:

- `alpha_catboost_action_master_like`
- `trend_xgb_like_xgb`

Both were trained in the ZigZag action model zoo and scored in walk-forward style:

- `2024 train -> 2025 score`
- `2025 train -> 2026 score`

## Generated Files

- `data/splits/year_oos/rl_training_2025_m7_zigzag_direction.csv`
- `data/splits/year_oos/rl_training_2025_m7_zigzag_direction.csv.meta.json`
- `data/splits/year_oos/rl_training_2026_m7_zigzag_direction.csv`
- `data/splits/year_oos/rl_training_2026_m7_zigzag_direction.csv.meta.json`
- Summary: `tmp/causal_regen_20260516/zigzag_m7_direction_integration_20260531/summary.json`

The original M7 files were not overwritten.

## Added M7 Columns

CatBoost action-master candidate:

- `m7_zigzag_cat_fl`
- `m7_zigzag_cat_up`
- `m7_zigzag_cat_dn`
- `m7_zigzag_cat_action`
- `m7_zigzag_cat_confidence`
- `m7_zigzag_cat_side_edge`
- `m7_zigzag_cat_trade_prob`

Trend-XGB-style candidate:

- `m7_zigzag_xgb_fl`
- `m7_zigzag_xgb_up`
- `m7_zigzag_xgb_dn`
- `m7_zigzag_xgb_action`
- `m7_zigzag_xgb_confidence`
- `m7_zigzag_xgb_side_edge`
- `m7_zigzag_xgb_trade_prob`

Class mapping:

- `fl`: ZigZag CASH probability
- `up`: ZigZag LONG probability
- `dn`: ZigZag SHORT probability

## Join / Contract Guards

- Exact timestamp join only.
- Row count must not change.
- Missing joined values are runtime failures.
- Probability columns must sum to `1.0` within tolerance.
- Existing M7 columns are never overwritten.

## Integration Counts

2025:

- rows: `105064`
- cols: `133`
- CatBoost action counts: `cash=34716`, `long=30041`, `short=40307`
- XGB action counts: `cash=32852`, `long=26944`, `short=45268`

2026:

- rows: `16897`
- cols: `139`
- CatBoost action counts: `cash=4298`, `long=6012`, `short=6587`
- XGB action counts: `cash=3747`, `long=6688`, `short=6462`

## Use Guidance

Use these as M7 direction-context candidates for downstream parent/meta-policy tests.
They are not teacher-generation inputs unless a separate no-leak stacking contract is
created.
