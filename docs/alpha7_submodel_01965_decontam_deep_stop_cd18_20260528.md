# Alpha7 Submodel: 01965 Decontaminated Deep Stop CD18

Submodel ID: `alpha7_submodel_01965_decontam_deep_stop_cd18_20260528`

Status:
- Precision retested candidate.
- Production default for `trading_bot.py` as of `2026-05-28`.
- Exchange execution remains controlled by the live execution environment flags.
- Base model artifacts are unchanged from `alpha7_submodel_01965_decontam_v2_tp_20260528`.
- `2026-05-28` shadow observation found a BEAR-regime `deep_alpha` LONG that stopped out quickly. A follow-up shadow candidate added a BEAR LONG veto only for the V31 `deep_alpha` fallback path, but validation MDD was materially worse; the production default was returned to plain `deep_stop_cd18`.

## Change

Only the `deep_alpha` fallback/scout control is changed:

- If a `deep_alpha` position exits by `deep_alpha_hard_stop_loss` or `deep_alpha_soft_stop_loss`, set the deep-only cooldown to at least `18` bars.
- Parent/v21_2 entries, v21.2 add-on runner, parent TP/SL/hold, feature contracts, and execution contract are unchanged.

Runtime config:
- `/home/llewyn/crypto-scalping/data/ensemble/supervised/alpha7_submodel_01965_decontam_v2_tp_20260528/alpha7_decontam_deep_stop_cd18_runtime_config.json`

Follow-up shadow config:
- Model ID: `alpha7_submodel_01965_decontam_deep_stop_cd18_bear_long_veto_20260528`
- Runtime config: `/home/llewyn/crypto-scalping/data/ensemble/supervised/alpha7_submodel_01965_decontam_v2_tp_20260528/alpha7_decontam_deep_stop_cd18_bear_long_veto_runtime_config.json`
- Additional rule: when V31 `deep_alpha` selects `LONG` and the live regime is exactly `BEAR`, fail the deep-alpha entry gate with `v31_deep_alpha_bear_long_veto`.
- Scope: V31 `deep_alpha` fallback only. Parent, fallback parent, V21.2 runner, execution, TP/SL/hold math, and feature contracts are unchanged.
- Sweep check: OOS Cost3 stayed flat (`198.78%` -> `198.76%`) with fewer trades (`109` -> `102`) and fewer deep entries (`88` -> `81`), but validation weakened (`MDD -17.78%` -> `-34.05%`). Treat as a shadow-only risk patch, not a promoted live model.
- LONG/SHORT specialist threshold tests were added to the sweep and rejected. OOS Cost3 fell to `128.89%` for the mild specialist and `96.09%` for the full specialist, mainly from broken sequencing and worse stop-loss ratio.
- Learned LONG/SHORT meta-veto specialists were tested and rejected. The 2025-trained side-specific logistic veto improved in-sample validation but degraded 2026 OOS Cost3 to `91.07%` with `-25.89%` MDD.
- Neural LONG/SHORT specialist heads were tested and rejected. A shared MLP trunk with side-specific heads trained on `12,822` 2025 V31 candidate labels reached 2026 OOS Cost3 `150.73%`; adding the BEAR-long veto reached `166.22%`. Both trail the current `deep_stop_cd18` / `bear_long_veto` candidates.

## Precision Retest

Artifacts:
- Summary: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_decontam_deep_stop_cd18_precision_20260528/summary.json`
- Grid: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_decontam_deep_stop_cd18_precision_20260528/cost_period_grid.csv`
- OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_decontam_deep_stop_cd18_precision_20260528/oos_cost3_ledger.csv`
- Baseline OOS ledger: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_decontam_deep_stop_cd18_precision_20260528/baseline_oos_cost3_ledger.csv`

Cost3 full comparison:

| Variant | Split | PnL | MDD | WR | Trades | Deep Entries | SL Ratio |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline | val | 109.74 | -16.59 | 0.510 | 196 | 173 | 0.122 |
| deep_stop_cd18 | val | 113.53 | -17.78 | 0.513 | 195 | 173 | 0.118 |
| baseline | oos | 162.28 | -17.99 | 0.439 | 107 | 88 | 0.131 |
| deep_stop_cd18 | oos | 198.78 | -18.22 | 0.440 | 109 | 88 | 0.110 |

Cost sensitivity:

| Variant | Split | Cost1 PnL | Cost2 PnL | Cost3 PnL |
|---|---|---:|---:|---:|
| baseline | oos | 212.60 | 196.11 | 162.28 |
| deep_stop_cd18 | oos | 258.65 | 241.99 | 198.78 |

Monthly Cost3:

| Variant | Period | PnL | MDD | WR | Trades | SL Ratio |
|---|---|---:|---:|---:|---:|---:|
| baseline | oos_2026-01 | 17.31 | -17.99 | 0.405 | 42 | 0.190 |
| deep_stop_cd18 | oos_2026-01 | 16.98 | -18.22 | 0.381 | 42 | 0.190 |
| baseline | oos_2026-02 | 52.09 | -27.23 | 0.425 | 73 | 0.137 |
| deep_stop_cd18 | oos_2026-02 | 73.74 | -27.23 | 0.440 | 75 | 0.107 |

## Interpretation

- OOS Cost1/2/3 all improved.
- Cost3 OOS PnL improved from `162.28%` to `198.78%`.
- OOS SL ratio improved from `13.08%` to `11.01%`.
- Most improvement came from February OOS. January slightly weakened, so this should still go through walk-forward or shadow validation before live promotion.
- The change does not remove `deep_alpha`; disabling deep_alpha was worse in the A/B sweep.

## Feature Contract

Forbidden in active/live path:
- `clean_regime_2024_unsup_v4_*`
- `clean_regime4_2024_unsup_v1_*`

Required:
- `clean_regime4_state24_sticky090_v2_*`
- `regime4_pred_*`
- `tp_sl_action_score`

Contract mismatch must fail fast. Do not add alias/fallback prefix/legacy compatibility in active/live or active candidate paths.

## Shadow Live Validation

Original `deep_stop_cd18` shadow started at `2026-05-28 12:16 KST`.

Follow-up `deep_stop_cd18_bear_long_veto` shadow uses isolated ledgers/state under:
- `/home/llewyn/crypto-scalping/data/live/shadow_alpha7_deep_stop_cd18_bear_long_veto/`

Launcher:
- `/home/llewyn/crypto-scalping/scripts/run_alpha7_decontam_deep_stop_cd18_shadow.sh`

Process:
- PID file: `/home/llewyn/crypto-scalping/data/live/shadow_alpha7_deep_stop_cd18/pid`
- Log: `/home/llewyn/crypto-scalping/logs/alpha7_decontam_deep_stop_cd18_shadow.log`

Safety:
- `BINANCE_ACCOUNT_ENABLED=0`
- `BINANCE_POSITION_SYNC_ENABLED=0`
- `BINANCE_EXECUTION_ENABLED=0`
- `BINANCE_EXECUTION_DRY_RUN=1`
- Shadow ledgers/state are isolated under `/home/llewyn/crypto-scalping/data/live/shadow_alpha7_deep_stop_cd18/`.
- Shadow microstructure, tail-risk, and orderbook DuckDB files are isolated from the active live bot.

Capped testnet launcher:
- `/home/llewyn/crypto-scalping/scripts/run_alpha7_decontam_deep_stop_cd18_capped_testnet.sh`

Capped testnet guard:
- Requires Binance testnet mode.
- Uses `BINANCE_EXECUTION_MAX_TARGET_NOTIONAL_USDT`, default `25.0`.
- Real mainnet execution is not enabled by this launcher.

## Rejected Side-Specialist Retrains

The first neural LONG/SHORT specialist experiment used both LONG and SHORT counterfactual rows for each candidate index. That made the training set exactly side-balanced and was rejected as a misleading data source.

A corrected chosen-side retrain was completed with one row per actual V31 selected side only:
- Dataset builder: `/home/llewyn/crypto-scalping/scripts/build_deep_side_specialist_chosen_dataset_20260528.py`
- Train/eval script: `/home/llewyn/crypto-scalping/scripts/train_eval_deep_side_specialist_chosen_nn_veto_20260528.py`
- Summary: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/deep_side_specialist_chosen_nn_veto_20260528/summary.json`

Chosen-side dataset:
- 2025 train all: `6,411` rows (`LONG=2,472`, `SHORT=3,939`)
- 2025 strict train: `3,268` rows (`LONG=1,289`, `SHORT=1,979`)
- 2026 eval all: `4,793` rows (`LONG=1,594`, `SHORT=3,199`)
- 2026 strict eval: `2,339` rows (`LONG=790`, `SHORT=1,549`)

Chosen-side NN OOS Cost3:
- `deep_stop_cd18`: PnL `198.78%`, MDD `-18.22%`, WR `44.04%`, trades `109`
- `deep_stop_cd18_chosen_nn_side_specialist`: PnL `198.59%`, MDD `-17.95%`, WR `43.12%`, trades `109`
- `deep_stop_cd18_chosen_nn_plus_bear_long_veto`: PnL `196.58%`, MDD `-17.95%`, WR `44.12%`, trades `102`

Decision:
- Keep the corrected chosen-side dataset as a reusable research artifact.
- Do not promote either chosen-side NN specialist to active/live. The current active candidate remains `deep_stop_cd18`; the bear-long-veto variant remains shadow-only.

## Day-Opportunity Architecture Direction

Follow-up contract:
- `/home/llewyn/crypto-scalping/docs/model_contracts/alpha7_day_opportunity_deep_stop_cd18_20260529_contract.md`

Purpose:
- Move `deep_stop_cd18` toward a lower-turnover, high-quality opportunity profile without adding a direct runtime trade cap.
- Target behavior is roughly `2-3` strong trades/day, but this must emerge from learned candidate utility, turnover cost, and opportunity-cost labels rather than from a hard daily counter.

Rejected direction:
- Do not use live-path `daily_trade_budget` or daily top-k runtime filtering.
- Do not use session filters for this goal.

Preferred direction:
- Keep parent/fallback/deep candidate generation intact.
- Add a learned Day-Opportunity meta layer that accepts a candidate only when expected cost-adjusted utility beats CASH plus opportunity cost.
- Select thresholds with a soft validation score that penalizes overtrading, MDD, and stop-loss ratio.

First test result:
- `alpha7_day_opportunity_meta_deep_stop_cd18_20260529` was tested and rejected.
- Baseline OOS Cost3 remains better: `198.78%` PnL, `-18.22%` MDD, `109` trades, `1.86` trades/day.
- Best validation-selected day-opportunity variant reached only `53.82%` OOS PnL. Best OOS PnL among swept variants was `91.50%`.
- Diagnosis: the first candidate utility label was non-stationary (`train pass_rate 82.3%` vs `val 31.1%` / `OOS 26.5%`) and over-filtered the deep-alpha convex tail.
