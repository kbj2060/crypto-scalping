# Omega Artifact Integrity Policy - 2026-06-30

## Purpose

Prevent validation/OOS PnL drift caused by replaying historical trade ledgers while regenerating parent signals from current code.

## Promotion Rule

An Omega candidate is promotable only if every component has exact-threshold precomputed parent prediction artifacts:

- `train_predictions_qXXX.csv`
- `validation_predictions_qXXX.csv`
- `oos_predictions_qXXX.csv`

`qXXX` is `round(quality_threshold * 100)`, zero padded. Example: `0.55 -> q055`.

The risk sidecar report must record:

- `risk_model.precomputed_prediction_dir`
- `risk_model.precomputed_prediction_tag`

The tag must match the component's actual `contract.quality_threshold`.

## Fail-Fast Conditions

Promotion must fail when:

- The risk sidecar was trained from a current runtime forward pass instead of precomputed parent predictions.
- Any exact-threshold parent prediction CSV is missing.
- Prediction timestamps do not match the runtime frame.
- The risk report and parent report disagree on label/quality contract and no precomputed prediction artifact is used.
- A candidate relies on saved trade ledgers as a substitute for per-bar parent predictions.

## Seed-Diversity Ensemble Promotion Gate (added 2026-07-31)

**Purpose**: prevent promoting a model whose apparent OOS edge is a seed-cluster artifact rather than real
signal.

**Background**: a 2026-08-01 audit of Sigma3-1h found its frozen "5-seed ensemble" checkpoint used
`SEEDS=[270705,270710,270715,270720,270725]` — small fixed increments of one base seed, not a genuinely
diverse sample. Re-ensembling with 8 truly diverse seeds (same data/features/engine/windows) matched VAL
(+22.99% vs +23.85%) but flipped OOS sign (+24.32% -> -13.57%, MDD -32.64%). A follow-up audit confirmed the
currently-live ETH/SOL Omega4.6.1 models are unaffected — they use a single-seed, per-expert-offset
mixture-of-experts architecture (bull/bear/chop), not a multi-seed averaging ensemble.

**Rule**: any model whose promotion claim rests on averaging/bagging across multiple training seeds (this
does NOT include single-seed or per-expert-offset architectures like Omega4.6.1's bull/bear/chop MoE) must
demonstrate OOS sign-agreement across N>=5 genuinely diverse seeds — random draws, not small fixed increments
of one base value. The seed list actually used must be logged in the promotion report.

**Fail-fast condition**: a promotion claim based on a seed count too small, or too tightly clustered, to
distinguish real signal from seed-variance noise is invalid regardless of the headline PnL/MDD metric.

## Required Audit

Run before promoting any Omega candidate:

```bash
PYTHONPYCACHEPREFIX=/tmp/quant_ai_pycache_20260630 /home/llewyn/miniconda3/envs/quant_ai/bin/python \
  /home/llewyn/crypto-scalping/scripts/audit_omega_artifact_integrity_20260630.py \
  --report /path/to/report.json
```

The audit writes:

- `omega_artifact_integrity_audit_20260630.json`
- `omega_artifact_integrity_audit_20260630.md`

Only `promotion_pass=true` is acceptable for live/full-bar promotion.
