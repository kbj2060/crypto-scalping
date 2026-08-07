# Active Live Docs

Last updated: 2026-07-02 KST

This directory is the operational documentation layer for the active trading bot path. It is intentionally separate from `docs/experiments/` and `docs/model_contracts/`, which contain many historical experiments.

Use these documents when you need to modify live logic, connect a new model, audit a runtime decision, or understand function-level contracts quickly.

## Scope

- Active Alpha model stack and submodels.
- `trading_bot.py` runtime decision flow.
- Module input/output contracts between model, feature, execution, ledger, dashboard, and storage layers.
- Fail-fast rules that must be preserved when code or artifacts change.

## Documents

- [alpha7_live_stack.md](alpha7_live_stack.md): previous Alpha7 model stack, artifacts, feature contracts, runtime overlays, and rejected/forbidden lineage.
- [omega5_live_stack.md](omega5_live_stack.md): active Omega5 live stack, promotion audit, and model contract pointers.
- [trading_bot_runtime.md](trading_bot_runtime.md): `trading_bot.py` live runtime flow, process locking, state files, position transitions, and logging/ledger behavior.
- [module_interfaces.md](module_interfaces.md): core module boundaries, important functions/classes, arguments, return values, and shared files.
- [regime3_policy_20260530.md](regime3_policy_20260530.md): active Regime3 policy. `regime3_pred_*` is removed from action/direction ownership; stability/transition-risk features are risk context only.
- [change_log.md](change_log.md): active-spec change log. Update this when active code, live model IDs, runtime config, or module contracts change.

## Update Rule

When code or model logic changes, update this directory in the same patch if any of the following changed:

- active model ID, artifact path, runtime config, or model owner,
- live decision order or position transition behavior,
- feature contract, required prefix, forbidden prefix, or fail-fast condition,
- ledger schema, dashboard state schema, process lock behavior, or DuckDB storage path,
- module function/class signature used by active live path.

Do not add compatibility aliases or silent fallback descriptions as active behavior. If a contract mismatch is expected to fail, document the failure explicitly.

## Active Live Snapshot

As of this update, Omega5 is blocked from active live promotion.

- blocked model ID: `omega5_event_risk_governor_20260702`
- block report: `docs/audits/omega5_live_promotion_blocked_20260702.md`
- reason: side-thread audit found validation/test ledger dependence in the promoted model-selection path.

Runtime behavior:

- `FINAL_GOVERNOR_OMEGA5_ENABLE` defaults to `false`.
- `FINAL_GOVERNOR_OMEGA5_SOURCE_PARENT_ENABLE` defaults to `false`.
- Explicitly setting `FINAL_GOVERNOR_OMEGA5_ENABLE=1` fails fast.
- Direct `Omega5LiveAdapter` construction fails fast.
- Do not use Omega5 PnL claims for live promotion until a clean rebuild passes the re-promotion requirements in the block report.

Funding red-team status: the previous Alpha7 default is deprecated and blocked for active/candidate reuse because it lacks clean funding provenance. It must not be used as a parent block, fallback block, sidecar source, Alpha8 baseline, or promotion reference. See `docs/audits/funding_feature_redteam_20260529.md`.

Deprecated artifact markers:

- `data/ensemble/supervised/alpha7_1_01965_live_20260527/DEPRECATED_DO_NOT_USE.json`
- `data/ensemble/supervised/alpha7_submodel_01965_decontam_v2_tp_20260528/DEPRECATED_DO_NOT_USE.json`
