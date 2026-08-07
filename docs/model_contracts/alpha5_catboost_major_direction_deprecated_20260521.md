# Alpha5 CatBoost Major / Direction Deprecation Contract

Date: 2026-05-21 KST

## Scope

This contract deprecates the CatBoost Major / Direction action-model path:

- `scripts/train_eval_alpha5_24_catboost_gpu_direction_refined_20260519.py`
- `scripts/backtest_alpha5_32_catboost_action_20260519.py`
- Artifacts under `tmp/causal_regen_20260516/alpha5_24_catboost_gpu_direction_refined_20260519`
- Artifacts under `tmp/causal_regen_20260516/alpha5_32_catboost_action_backtest_20260519*`

Status: deprecated, historical-reference only.

## Reason

The current Alpha5/DSAC design makes DSAC the final policy owner. Router-style CatBoost outputs may be used only as causal probability features, not as an independent action owner.

CatBoost Major / Direction was designed as a direct action model:

- entry head decides whether to trade
- direction head decides LONG / SHORT
- composed output becomes `NONE / LONG / SHORT`

That overlaps with the current DSAC responsibility boundary. Keeping this model in the active path creates duplicate action ownership and can force stale CatBoost directional bias into the policy.

## Active Path Rule

CatBoost Major / Direction must not be used in live or frozen backtest active paths.

Allowed:

- historical audits
- reproduction of old reports with an explicit deprecation override
- feature comparison notes

Disallowed:

- live trading decisions
- parent replacement active candidates
- direct `NONE / LONG / SHORT` action ownership
- DSAC final-action overrides

## Replacement Boundary

Use Router5-style outputs as auxiliary features only:

- `a5dir_available`
- `a5dir_none_prob`
- `a5dir_long_prob`
- `a5dir_short_prob`
- `a5dir_prob_max`
- `a5dir_edge`
- `a5dir_margin`
- `a5dir_whipsaw_prob`

The final action must be owned by DSAC or the explicitly selected parent model in a frozen comparison. CatBoost action models cannot be inserted as an intermediate action owner.

## Red Team Gate

Any future CatBoost direction model must pass all of the following before it can be reconsidered:

- causal or OOF scoring for downstream use
- no direct final action ownership unless it is the only tested mutable layer
- frozen baseline comparison with all other layers unchanged
- long/short balance and bear-market OOS diagnostics
- explicit contract update replacing this deprecated status
