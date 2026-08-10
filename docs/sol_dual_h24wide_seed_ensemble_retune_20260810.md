# SOL dual-component H24-wide router — seed-ensemble + genuine retune (2026-08-10)

> **Decision: CLOSED, no promotion.** OOS collapses to near-flat and underperforms the rule
> baseline on both PnL and MDD. This is the required follow-up to
> [`docs/sol_dual_h24wide_seed_stability_20260729.md`](sol_dual_h24wide_seed_stability_20260729.md)
> and completes (negatively) the last open lever in the 2026-07-29 dual-router line.

## Why this was run

The 2026-07-29 dual-component regime router (bull/bear/chop, ZIG075 x H24-wide) was the only SOL
ML candidate in this project's history to beat the rule baseline on PnL **and** MDD in **both**
VAL and OOS (see
[`sol_dual_h24wide_final_20260729.md`](sol_dual_h24wide_final_20260729.md)). A 5-seed
reproducibility test then showed that result was a seed-lucky outlier: with the router's
margin/leverage mapping frozen and only the parent retrained per seed, seed VAL PnL ranged
-8.85%..+6.66% (mean -1.12%) and OOS -17.24%..+23.52% (mean +3.35%) — nowhere near the original
+25.08%/+21.62%. That doc's own prescribed next step was never executed until now:

> "Do not select the best seed. The next candidate should reduce training variance before
> performance selection, preferably by averaging direction, quality, and exit probabilities
> across a fixed seed ensemble and then running a new, untouched forward test."

## Method

Reused the 5 already-trained seed parent artifacts from 2026-07-29 (seeds 17/29/43/71/101, both
components) — no GPU retraining needed for the averaging step. New orchestrator:
[`scripts/eval_sol_dual_router_seed_ensemble_retune_20260810.py`](../scripts/eval_sol_dual_router_seed_ensemble_retune_20260810.py).

1. **Probability ensemble** — averaged each component's direction/quality probabilities across
   the 5 seeds via the existing, unmodified
   `scripts/build_fixed_seed_prediction_ensemble_20260729.py` (arithmetic mean of softmax
   probabilities before thresholding; the exit head is not ensembled — same documented scope as
   that script's own manifest). Merged the averaged predictions with seed 17's
   `true_3head_tabm_bundle.pt` (exit head + base-feature contract; arbitrary reference seed,
   documented limitation).
2. **Genuine risk-sidecar retune** — retrained each component's risk sidecar from scratch on the
   averaged predictions via a new wrapper,
   [`scripts/run_sol_dual_sidecar_candidate_retune_20260810.py`](../scripts/run_sol_dual_sidecar_candidate_retune_20260810.py),
   **without** `--fixed-mapping-report` — i.e. a real margin/leverage sigmoid-mapping grid search
   on the smoothed ensemble scores, rather than reusing one (possibly overfit) seed's frozen
   mapping. This is the actual difference from the 07-29 seed-stability test, which explicitly
   froze the mapping across all 5 seeds and left this step untested.
3. **Router re-evaluation** — reran the full VAL router-structure grid search (36 variants:
   priority order, conflict rules, all 8 bull/bear/chop hard assignments, `regime_anchor` variants
   with duration-gate and margin-scale sweeps) and one frozen OOS read, via the unmodified
   `scripts/eval_sol_dual_structure_router_20260729.py`.

Every sub-step called the exact tested script/module used in the original 07-29 line; this was an
orchestration exercise, not a reimplementation. Cost model (`cost_multiplier=3.0`,
`leverage_cap=5.0`, `notional_cap=1.8`, `notional = margin_fraction * leverage`), VAL/OOS
boundaries, and the fresh-forward bar-by-bar contract are all inherited unchanged:

```
fresh_forward_bar_by_bar=true
trade_ledgers_used_as_input=false
saved_parent_exit_timestamps_used=false
future_rows_used_for_entry=false
oos_used_for_selection=false
single_shared_position_slot=true
validation_range=[2025-09-01, 2025-12-31]
oos_range=[2026-01-01, 2026-03-31]
```

### Bug found and fixed en route

The first run failed at the sidecar-retrain step with `RuntimeError: train: precomputed
prediction timestamps do not match prepared frame`. Root-caused (not data drift — the underlying
`sol_features_2025/2026.csv` files were last modified 2026-07-21, before the 07-29 line even
started): `train_eval_omega1_2_tabm_3head_20260603.py` and
`train_eval_omega1_2_tabm_diffusion_risk_sol_20260707.py` both default `SPLIT_TS =
pd.Timestamp("2025-10-01")`, but every 07-29 prediction artifact was generated with `SPLIT_TS`
monkeypatched to `2025-09-01` by the original wrapper scripts. Calling
`train_eval_omega4_2_risk_sidecar_sol_20260707.py` directly (as the first version of this
orchestrator did) skipped that monkeypatch, so `_prepare_frames()` built a train/validation
boundary one month later than the boundary baked into every existing prediction CSV. Confirmed via
a direct diagnostic run against seed 17's own unmodified original predictions, which failed
identically. Fixed by writing
`run_sol_dual_sidecar_candidate_retune_20260810.py` — a copy of the original
`run_sol_dual_sidecar_candidate_20260729.py` wrapper with `--fixed-mapping-report` made optional
instead of required, keeping the same `SPLIT_TS` monkeypatch.

## Result

| candidate | VAL pnl | VAL mdd | OOS pnl | OOS mdd |
|---|---:|---:|---:|---:|
| rule_baseline | +23.45% | -7.69% | +7.66% | -12.52% |
| original_single_seed (07-29, lucky) | +25.08% | -7.49% | +21.62% | -9.88% |
| 5seed_mean (07-29, mapping frozen) | -1.12% | -11.68% | +3.35% | -15.87% |
| **seed_ensemble + genuine retune (this run)** | **+68.52%** | **-15.55%** | **+0.93%** | **-22.43%** |

Selected variant: `regime_anchor_bull-h24wide_bear-zig075_chop-zig075` (bull routes to H24-wide,
bear and chop both route to ZIG075), `regime_margin_scale = {bull:1.0, bear:1.0, chop:1.0}` — no
scale-down, unlike the original candidate's `{bull:0.25, bear:0.50, chop:1.0}`. VAL: 46 trades,
52.2% WR. OOS: 30 trades, 36.7% WR.

**Does not beat the rule baseline on PnL and MDD in both windows** (`beats_baseline: false`) —
OOS PnL (+0.93%) is below the baseline's own +7.66%, and OOS MDD (-22.43%) is worse than the
baseline's -12.52%.

Artifacts: `tmp/causal_regen_20260516/sol_dual_structure_router_sidecar_q060_q055_20260729__seedensemble_retune_20260810/report.json`,
`tmp/causal_regen_20260516/sol_dual_router_seedensemble_retune_20260810_summary.json`.

## Diagnosis

Probability averaging did reduce seed-driven noise in the underlying direction/quality signal (as
it did for the 07-31 SOL/BTC single-component test in
[[project-seed-ensemble-averaging-result-20260731]]), but **re-opening the router-structure grid
search on top of that smoothed input just gave the grid search a new, smoother surface to overfit
VAL against.** The winning structure changed materially from the original candidate — `bear`
switched from H24-wide to ZIG075, and every margin-scale reduction disappeared (all regimes at
1.0x instead of 0.25x/0.50x) — which is exactly the shape of a selection procedure finding a new
VAL-favorable combination rather than a genuinely more robust one. VAL PnL (+68.52%) is in fact
the highest ever recorded on this candidate family, precisely because the grid search had 36
variants times a smoother input to search over; OOS punished that the hardest yet.

**This sharpens, rather than contradicts, the 07-31 ensembling finding.** Averaging alone (frozen
structure/mapping) converges toward the seed mean and cannot fix a bad mean. Averaging combined
with a fresh selection step (mapping grid, structure grid) does not even reliably converge toward
the mean — the selection step re-injects a new overfitting surface on top of whatever variance the
averaging removed. Reducing input noise does not neutralize a downstream search process that is
itself free to re-optimize against VAL.

## Conclusion

The 2026-07-29 dual-component H24-wide router line is now closed on every axis that was left
open: frozen-mapping seed reproduction (07-29, failed), and seed-ensemble-averaged genuine retune
(this doc, failed). Combined with the unrelated 08-07/08-08 architecture survey's 12 exhausted
families ([`sol_dl_rl_architecture_survey_20260807.md`](sol_dl_rl_architecture_survey_20260807.md)),
this was the last concretely-motivated, not-yet-executed idea for a SOL entry-side ML upgrade.
SOL live stays on `adaptive_squeeze v2` (rule-based), execution-disable recommendation from
[[project-sol-sidecar-mdd-unfixable-20260730]] unchanged. No further seed-variance or
router-structure tuning should be attempted on this exact candidate family without a genuinely new
input (new data source, not a new selection procedure over the same panel).
