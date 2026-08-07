# Alpha7 Submodel: 01965 Decontaminated V2 TP

Active submodel ID: `alpha7_submodel_01965_decontam_v2_tp_20260528`

Status:
- Shadow candidate only.
- Not approved for full live until untouched OOS or walk-forward validation is completed.

Why this exists:
- The original `alpha7_1_01965_live_20260527` lineage used a `tp_sl_action_score` derived from legacy regime features.
- The contaminated original candidate must not be treated as trusted live evidence.
- This patch retrains the Alpha7 parent and fallback on a v2-only `tp_sl_action_score` frame.

Forbidden in active/live path:
- `clean_regime_2024_unsup_v4_*`
- `clean_regime4_2024_unsup_v1_*`

Required behavior:
- Artifact or feature-contract mismatch must fail fast.
- Do not add alias, fallback prefix, silent rename, or legacy compatibility for active/live paths.

Current artifact directory:
- `data/ensemble/supervised/alpha7_submodel_01965_decontam_v2_tp_20260528`

Validation references:
- Decontamination report: `tmp/causal_regen_20260516/alpha7_1_01965_tp_sl_decontam_20260528/report.json`
- Runtime-style retest: `tmp/causal_regen_20260516/alpha7_1_01965_decontam_runtime_retest_20260528/summary.json`

Runtime-style Cost3 retest:
- Validation PnL: `109.74%`
- OOS PnL: `162.28%`
- OOS MDD: `-17.99%`
- OOS WR: `43.93%`
- OOS trades: `107`
