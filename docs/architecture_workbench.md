# New Architecture Workbench

`pipeline/architecture_workbench.py` is the entry point for a new modelling line.
It creates an immutable contract, runs an actual dataset/causality preflight, and
then produces feature statistics before a model runner may consume the data. A
past failed model or logic is never blocked solely because it was tried before.

## Historical-failure guards (mandatory)

Preflight computes these checks from the actual artifact. They are not booleans
that a contract author can claim without evidence.

| Historical failure | Required prevention |
| --- | --- |
| Retired Regime4 surfaces re-enter a feature frame | Block `clean_regime_2024_unsup_v4_`, every `clean_regime4_`, and every `regime4_pred_` input column. Regime3 must be used explicitly if regime context is needed. |
| 1h feature attached to an unfinished 1h bar | Label a source 1h bar by its open but expose it only at `source_open + 1h`; merge lower-timeframe rows by `available_at`, `direction="backward"`. |
| Forward-window trend scan assigned to window start | A trailing window ending at row `r` must write its result to `r`, never to the window's first row. Test stored values against a backward and a forward recomputation. |
| Future values used to fill missing features | Do not use `bfill` on a feature path. Missingness must remain visible or use an explicitly causal prior-value method. |
| Changed raw/features silently reproduce an old report | Verify raw and dataset manifests before training and write their lineage into every report. |
| An adoption rests on a consistency statistic with no effect size | `selection.effect_size_gate` is required on contract schema v3 and enforced by `assert_effect_size_gate(...)`: a minimum \|t\|, a label-permutation percentile, an explicit risk-channel test when the claim is about drawdown, a premise sign-check in the selection window, and (when the winner came from a multi-configuration search) a falsification audit against a noise/microstructure-placebo null. |

`preflight` calls `assert_safe_feature_columns(...)` against the actual frame and
calls `assert_higher_timeframe_availability(...)` against the supplied 1h/4h
availability artifact. The latter rejects even one 5m decision that sees a 1h
feature before the source hour has completed, and the availability artifact
itself must cover every decision timestamp that exists in the pinned dataset --
a standalone sample artifact that isn't built from the dataset's own decision
points is rejected rather than silently accepted as proof.

## Effect-size gate (contract schema v3, added 2026-08-08)

`SCHEMA_VERSION` is now `architecture_experiment_contract_v3`; `v2` stays accepted so existing
contracts keep preflighting, but **only v3 may be used for new lines and v3 must declare
`selection.effect_size_gate`**:

```json
"effect_size_gate": {
  "min_abs_t": 2.0,
  "min_permutation_percentile": 0.90,
  "risk_channel_tested": true,
  "premise_checked_in_selection_window": true,
  "falsification_audit_required": true,
  "min_falsification_percentile": 0.95
}
```

Why it exists: the BTC `czz_trend` regime-sizing overlay was adopted on a paired time-block
bootstrap `P=0.739`, and a later audit found its own per-trade effect was `t=-0.99` (p=0.33), its
**risk channel pointed the wrong way** (the bucket it downsized was *less* volatile, variance
ratio 0.881), and its premise reversed sign between the selection window and the window where it
"worked". **A bootstrap P measures whether a difference's SIGN is consistent across blocks; it
says nothing about the size of that difference** — on ETH the same statistic returned `P=0.979`
for a `t=0.32` difference.

Four helpers implement the standard so each line does not reinvent it:

| Helper | What it answers |
| --- | --- |
| `effect_size_report(a, b)` | Welch t, Cohen's d, Brown-Forsythe variance test, worst-5 tails between two buckets |
| `permutation_label_test(returns, multipliers)` | "How special is THIS labelling?" — reassigns the same multiplier multiset at random, R draws, and locates the real assignment in that null. A bootstrap never asks this. |
| `core.selection_stats.falsification_audit(returns_matrix)` | "Could this exact search have produced its winner out of noise alone?" — compares the real best-of-N Sharpe against a zero-predictability i.i.d. null and a demeaned circular-block-bootstrap microstructure-placebo null, both drawn at the search's own shape |
| `assert_effect_size_gate(contract, report, permutation=..., selection_window_report=..., falsification=...)` | Enforces the contract's declared gate and raises with every failure listed |

`risk_channel_tested: true` makes the drawdown case honest: if the claim is "this reduces MDD",
the bucket being downsized has to actually be riskier, and the gate rejects the claim when the
variance ratio is ≤ 1. `premise_checked_in_selection_window: true` requires passing the
selection-window report too, and rejects a rule whose premise flips sign between windows — the
placement-luck signature. `falsification_audit_required: true` requires passing a
`falsification_audit(...)` report computed over the same (periods x configurations) matrix the
search actually tried, and rejects a winner whose real best-of-N Sharpe is unremarkable
(below `min_falsification_percentile`, default 0.95) against either null — i.e. a search that
clears its own bar as easily on noise of its own shape as it did on the real data. This is meant
to run before the winner is allowed to consume VAL/OOS budget, not after: see
`docs/entry_exit_edge_root_cause_and_literature_review_20260809.md` for why entry-signal searches
in this repo need it most (Nikolopoulos, arXiv:2604.15531, 2026).

`validate_contract` also enforces the project's seed-diversity gate: if
`model.seed_ensemble_claim` is true, `model.seeds` must contain at least 5
seeds and must not be a fixed-increment cluster (e.g. `base, base+5, base+10,
...`), which VAL-vs-OOS history has shown can look diverse on paper while
carrying almost no more information than a single seed.

## Start a new line

```bash
python3 -m pipeline.architecture_workbench init \
  --output docs/experiments/<experiment-id>.json
```

The prompts follow this order:

1. Single-sentence hypothesis, a new research-line ID, and comparison to every related closed line
2. Pinned feature dataset and raw sources, including external-data release lag
3. Feature groups and label/horizon/timeout treatment
4. Chronological train/validation/OOS boundaries
5. Cheap falsification model before the intended architecture
6. Validation-only selection, execution rule, sizing and Fresh-Forward gates

Validate the saved decision record before writing a feature, label, training, or
backtest runner:

```bash
python3 -m pipeline.architecture_workbench preflight \
  docs/experiments/<experiment-id>.json \
  --output docs/experiments/<experiment-id>.preflight.json

python3 -m pipeline.architecture_workbench analyze-features \
  docs/experiments/<experiment-id>.json \
  --preflight docs/experiments/<experiment-id>.preflight.json \
  --output-dir docs/experiments/feature_analysis
```

## Required modules for each accepted contract

Each implementation must remain separated into these small runners. Only
`preflight` and `feature_analysis` are implemented inside
`pipeline/architecture_workbench.py` today; the rest are gates a contract must
still satisfy, implemented by other project-level scripts (or not yet built).
Do not treat an unimplemented row as "no gate exists" -- it means the check has
to be satisfied by hand or by the referenced script until it is automated here.

| Module | Responsibility | Must fail when | Status |
| --- | --- | --- | --- |
| `data` | source snapshot, timestamp/duplicate/gap checks | raw or derived dataset hash drifts | Implemented elsewhere: `scripts/dataset_snapshot.py` |
| `preflight` | actual dataset manifest/hash/schema and higher-timeframe availability | unregistered dataset, hash drift, forbidden inputs, invalid timestamps, future 1h/4h data | Implemented: `architecture_workbench.py preflight` |
| `feature_analysis` | all-feature missingness/distribution/constancy and numeric-pair Spearman correlations | preflight missing or dataset changed after preflight | Implemented: `architecture_workbench.py analyze-features` |
| `features` | causal feature frame and schema | future/label/target/exit input, `bfill`, schema mismatch | Not implemented -- build per experiment; preflight's forbidden-column check is the only automated guard so far |
| `labels` | label creation and purge/embargo | label horizon overlaps split boundary without handling | Not implemented -- build per experiment |
| `cheap_gate` | simple baseline falsification | VAL and OOS pass criteria fail | Not implemented -- build per experiment |
| `train` | selected candidate model | non-train data is fit, schema differs, seed is unlogged | Not implemented -- contract only records intended seeds; nothing yet checks the trained artifact against them |
| `fresh_forward` | runtime-native bar-by-bar evaluation | ledger, future rows, or saved exits are input | Implemented elsewhere: `core/causal_futures_backtest.py` |
| `effect_size` | effect size behind any adoption claim | \|t\| below the declared floor, a risk claim whose downsized bucket is not riskier, a label-permutation percentile below the floor, a premise that flips sign between selection and evaluation windows, or a search winner that is not distinguishable from a noise/microstructure-placebo null | Implemented: `architecture_workbench.effect_size_report` / `permutation_label_test` / `assert_effect_size_gate`; `core.selection_stats.falsification_audit` |
| `audit` | lineage, sizing and live parity | artifact/prediction/dataset contracts differ | Implemented elsewhere: `core/backtest_metrics.py`, `scripts/audit_omega_artifact_integrity_20260630.py` |

The model-specific scripts may use these stages, but none may bypass their gates.

## Prior research-line registry

Every new contract loads [research_line_registry.json](model_contracts/research_line_registry.json).
The registry is a structured memory of prior failures, not a prohibition list. A
contract may retest the exact same model or logic. If it does, it must state (1)
why the previous result might not generalize or be trustworthy and (2) the
independent retest design -- for example frozen-input reproduction, a corrected
availability rule, accounting parity, runtime parity, or a later untouched window.

The registry also turns previous methodological failures into mandatory checks:

- a previously observed/spent OOS period cannot select or promote a candidate;
- a promising component cannot be promoted without the full router, slot, sizing,
  and portfolio simulation;
- label and backtest costs must use the same accounting convention;
- a result that improved only on observed OOS needs a genuinely untouched later
  window before it may be considered;
- the global closed axes from the research redesign are shown as prior evidence,
  and their documented reopening conditions are guidance for a meaningful retest.
