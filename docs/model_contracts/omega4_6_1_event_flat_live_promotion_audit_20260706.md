# Live Promotion Audit — Omega4.6.1 + Event-Flat Overlay (2026-07-06)

Status: `promotion_pass=false — NOT PROMOTABLE`

Scope: full audit of whether "Omega4.6.1 duration_ou_halflife_risk_gate, extended to Jan-Jun 2026
OOS, with a flat-during-macro-event overlay" (the best-performing variant tested today, PnL
+156.20%, MDD -10.82%, WR 0.56) can be promoted to live trading, per the gates defined in
`AGENTS.md` (Omega Artifact Integrity Promotion Gate, Fresh-Forward Validation/OOS/Test Rule) and
this project's general live-promotion practice (redteam, registry.json entry, live adapter code).

**Verdict: not promotable. Multiple independent, structural gates fail — this is not a single
fixable blocker but a stack of separate problems.**

## Gate 1 — Fresh-Forward Validation/OOS/Test Rule: FAIL

> "저장된 trade ledger... 를 입력으로 사용한 성과는 승격/모델 선택/test 근거로 쓰면 안 된다... 이
> 규칙을 어기는 평가 결과는 성능 수치와 무관하게 promotion/test 근거로 무효다." (AGENTS.md)

Every number produced today for the event-flat/haircut/profit-lock variants was computed by
**post-hoc numerical transformation of an already-computed saved ledger**
(`combined_router_duration_gated_ledger_extended.csv`) — not a genuine bar-by-bar fresh-forward
causal walk that decides fresh at each bar using only that bar's live-available state. Per the
project's own explicit rule, this makes the result invalid as promotion or model-selection
evidence **regardless of how good the PnL looks**. This is the single most important reason this
candidate cannot be promoted as tested.

## Gate 2 — Omega Artifact Integrity Promotion Gate: FAIL / NOT EVEN CONSTRUCTED

Per `docs/model_contracts/omega_artifact_integrity_policy_20260630.md`, promotion requires running
`scripts/audit_omega_artifact_integrity_20260630.py --report <candidate report.json>` with
`promotion_pass=true`. This candidate has no such `report.json` (no `components` dict with
`out_dir`/`precomputed_prediction_dir`/`precomputed_prediction_tag` fields was ever built for it),
so the audit cannot even run. If one were constructed:

- The audit's `prepared_frame_timestamps()` hardcodes reconstruction of the runtime frame from
  `risk_model.train_csv`/`eval_csv`/`direction_label_dir` recorded in the FROZEN sidecar reports
  -- these all point to the original alpha6/7-lineage files and labels, which stop at
  **2026-02-28**. The audit has no code path to validate a Jan-Jun window; it would need
  modification first (same class of problem the Omega6 v1 project plan already flagged for a
  different model).
- The event-flat overlay itself is a post-hoc ledger edit with no `precomputed_prediction_dir`/
  `qXXX` prediction-file structure at all -- it doesn't fit the artifact-integrity contract's
  shape (parent-prediction-driven signal), because it isn't one.

## Gate 3 — Feature-vintage drift: UNRESOLVED

Documented in `docs/model_contracts/omega4_6_1_extended_oos_20260706_retest.md`: the extended
scoring uses `training_features_2026_rebuilt.csv`, which diverges from the original alpha6/7-era
feature file on 5/96 columns -- most importantly **`ou_halflife` (corr=-0.03, essentially
unrelated)**, which the duration gate rule reads directly. Root cause (a `features/elite.py`
formula change since 2026-05-29) was never fixed, only worked around by using the extended file
uniformly. This means even a corrected, properly-registered artifact-integrity run would likely
fail on legitimate grounds -- the parent/duration-gate inputs are not the same distribution the
model was tuned against.

## Gate 4 — No live adapter code exists

None of today's four things tested (extended-OOS retest, entry veto, exposure haircut, profit-lock
rule) has a `trading_bot_modules/*.py` implementation. They are one-off analysis scripts in
`scripts/` that read precomputed CSVs/pickles and print metrics. There is no real-time decision
path that could be wired into `trading_bot.py` today. Compare to how other promoted-or-considered
candidates in this repo (e.g. `omega5_event_risk_governor_20260702_contract.md`) have an actual
`trading_bot_modules/omega5_live.py::Omega5LiveAdapter` class with fail-fast contract validation --
that layer doesn't exist for any of today's ideas.

## Gate 5 — No redteam audit

Every promoted-or-conditionally-promoted Omega candidate in this repo has a `redteam_verdict` or
`docs/audits/*_redteam_*.md` file (e.g. Omega4.6.2's `CONDITIONAL_DIAGNOSTIC_PASS_FULL_LIVE_FAIL_
FRESH_HOLDOUT_REQUIRED`). None of today's work has been through any adversarial review.

## Gate 6 — Approximation gaps in the overlay logic itself

The event-flat/haircut simulations explicitly ignore the transaction cost of dynamically resizing
an open position (closing/reopening incurs fee+slippage each time) -- a real implementation would
be less profitable than modeled. The profit-lock rule also showed a real philosophical tension
with this project's established "let winners run" finding (Sigma6) by capping several of the
biggest winners early.

## Gate 7 — Small sample size

The specific edge attributed to the event-flat overlay (+156.20% vs +145.46% baseline, ~+10.7pp)
rests on only 11 trades' partial overlaps with event windows, over a single 6-month window. Not
statistically robust enough to trust as a standalone effect.

## Gate 8 — The underlying base model was never fully live-wired either

Even without any of today's changes, `omega4_6_plus_t12_nohold_risk1_20260630` (what Omega4.6.1
sits on top of) carries `status: "current_omega_research_baseline_not_live_wired_conditional_
swing"` and an explicit `redteam_verdict: "CONDITIONAL_PASS_MAX_HOLD_AND_PNL_TARGET_EXCLUDED_
NOT_DAYTRADING_LIVE_PASS"`, with `max_hold_24h` and `pnl_target_validation_oos_100pct` excluded
from its own gate. The foundation this whole chain is built on was already a research/conditional
artifact, not a full live-pass, before any of today's work started.

## What would actually be required to promote something in this family

1. Fix the root cause of `ou_halflife`/`kel`/`evt_excess_z`/`btc_corr_60`/`dual_momentum` drift in
   `features/elite.py`, or retrain/re-tune the duration gate against the current feature vintage.
2. Extend `zigzag_action_labels_2026.csv` and the alpha6/7-lineage eval_csv (or replace them with
   a properly-versioned current pipeline) through the intended OOS end date.
3. Re-run the ORIGINAL frozen bundles' training-time selection process (or an honestly-labeled
   successor) as a genuine bar-by-bar fresh-forward walk over the fixed val/OOS split defined in
   AGENTS.md (val 2025-09-01..12-31, OOS 2026-01-01..03-31, or an explicitly documented
   alternative), with `fresh_forward_bar_by_bar=true`, `trade_ledgers_used_as_input=false`.
4. Build a proper `report.json` with `components`/`precomputed_prediction_dir`/
   `precomputed_prediction_tag` fields and pass `audit_omega_artifact_integrity_20260630.py`
   with `promotion_pass=true`.
5. Write actual `trading_bot_modules/*.py` adapter code for whichever event-handling design (if
   any) survives step 3, with fail-fast contract validation matching the Omega5 pattern.
6. Run a redteam review.
7. None of this is a quick follow-up -- it is comparable in scope to what a new Omega sub-version
   (e.g. Omega4.6.3) would require from scratch.

## Bottom line

Today's event-window experiments (veto/haircut/profit-lock) were useful, honestly-reported
research signals about where this model's edge and weaknesses sit, but they were built with
exploratory diagnostic methodology from the start (as documented in each result), not
promotion-track methodology. **Do not treat any of today's PnL numbers as a live-promotion case.**
