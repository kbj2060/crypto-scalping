# Architecture Improvement Loop 1

Date: 2026-05-05 KST

Status: `iterate_required`

## Baseline Clarification

The `Current MuZero/AZ` result inside `data/ensemble/reports/dt_lifecycle_vs_muzero_az_2026.json` has two different PnL numbers:

| Split | Date range | PnL | MDD | Trades/day | Avg leverage |
|---|---|---:|---:|---:|---:|
| Validation | `2025-11-01` to `2025-12-31 23:55` | `1058.615941%` | `-8.821092%` | `6.262650` | `1.607691` |
| Full OOS/test | `2026-01-01` to `2026-02-28 16:00` | `467.644256%` | `-25.912969%` | `6.289773` | `1.594981` |

For that historical comparison report, `1058.615941%` is validation and `467.644256%` is full OOS/test.

However, the latest model registry and current-top contract supersede this older `467.64%` comparison baseline. Future architecture loops must use `current_top_muzero_az_stage2_azexit_2026` unless the user explicitly overrides the baseline:

| Current rank-1 contract | PnL | MDD | Trades/day | Avg leverage |
|---|---:|---:|---:|---:|
| `docs/model_contracts/2026-05-06_current_top_muzero_az_stage2_azexit.md` | `752.65%` | `-18.76%` | `6.02` | `1.596` |

Loop 1 used the older `467.64%` comparison target because v1 was derived from the DT lifecycle comparison path. That makes loop 1 non-promotable even before the micro-add failure.

## Candidate

- Model id: `muzero_az_alpha_preserving_microadd_v1_2026`
- Contract: `docs/model_contracts/2026-05-05_muzero_az_alpha_preserving_microadd_v1.md`
- Script: `scripts/compare_muzero_az_alpha_preserving_microadd_v1_2026.py`
- Smoke report: `tmp/muzero_az_alpha_preserving_microadd_v1_smoke.json`
- Full OOS report: not run in loop 1
- Promotion baseline issue: v1 targets the older `467.64%` comparison baseline, not the latest `752.65%` rank-1 contract.

## Loop 1 Result

Smoke command used row limits:

```bash
/home/llewyn/miniconda3/bin/conda run -n quant_ai python scripts/compare_muzero_az_alpha_preserving_microadd_v1_2026.py \
  --device cpu \
  --limit-train-rows 1200 \
  --limit-val-rows 900 \
  --limit-eval-rows 900 \
  --max-train-samples 800 \
  --max-grid-configs 24 \
  --report-out tmp/muzero_az_alpha_preserving_microadd_v1_smoke.json \
  --model-dir tmp/muzero_az_alpha_preserving_microadd_v1_smoke
```

| Metric | Limited baseline | Candidate | Delta | Gate |
|---|---:|---:|---:|---|
| PnL | `1.685566` | `4.924237` | `+3.238672` | pass on smoke only |
| MDD | `-1.687125` | `-1.222035` | `+0.465090` | pass on smoke only |
| Trades/day | `2.883204` | `2.883204` | `0.000000` | fail |
| Avg leverage | `1.556613` | `1.557121` | `+0.000508` | pass on smoke only |
| Micro-add entries | n/a | `0` | n/a | fail |
| Invariant audit | n/a | pass | n/a | pass |
| Cost 1x/2x/3x survival | n/a | pass | n/a | pass on smoke only |

Report status is `smoke_not_promotable`. Full OOS gates and baseline reproduction are intentionally not promotable because row limits were used. Additionally, the candidate script's full OOS hard target is the older `467.64%` comparison baseline, so a v2 contract must retarget the current rank-1 baseline before promotion.

## Red Team Verdict

Decision: do not promote.

Blocking reasons:

- `trades/day` did not increase.
- `microadd_entry_count` is `0`, so the core micro-add mechanism did not fire.
- Full OOS was not run.
- The candidate targeted the older comparison baseline rather than the latest rank-1 contract.
- Cost survival evidence is limited to the smoke slice.

## Model/Data Architect Verdict

Decision: iterate.

The v1 implementation is operationally useful because it adds provenance, invariant, cost-stress, and status-gate reporting, but it did not prove the requested architecture improvement.

## Loop 2 Delta

Candidate id: `muzero_az_alpha_preserving_microadd_v2_2026`

Required contract changes:

- Add `microadd_entry_count > 0` as a validation eligibility gate.
- Add `trades_per_day > baseline_trades_per_day` as a validation eligibility gate.
- Require full OOS with no row limits before any promotion decision.
- Require baseline reproduction against current rank-1 `752.65% / -18.76% / 6.02 trades/day / 1.596 avg leverage` unless the user explicitly overrides the baseline.

Search changes:

- Keep frozen MuZero/AZ as the action owner.
- Keep active baseline rows monotonic only: keep, scale down, or flat.
- Keep no side reversal and no micro-add while baseline is active.
- Relax only micro-add gates:
  - include `vote_agreement_min = 2`
  - include `vote_margin_min = 0.10, 0.15, 0.20, 0.30`
  - include smaller notional caps `0.05, 0.075, 0.10, 0.15, 0.20`
  - keep strict `cost_buffer_3x > 0`
  - keep cooldown and daily micro-add cap

Next implementation should select only among configs with nonzero micro-add activity, increased trades/day, invariant pass, avg leverage in `1.50` to `1.80`, and cost `1x/2x/3x` survival.
