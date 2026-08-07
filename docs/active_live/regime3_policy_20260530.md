# Regime3 Active Policy - 2026-05-30

## Decision

Directional `PRED regime` is removed from active action ownership.

Do not use `regime3_pred_*` future class probabilities as:

- long/short direction inputs,
- primary/fallback action labels,
- hard future regime selectors,
- alpha timing ownership signals.

## Active Regime Contract

CURRENT regime remains the only active regime-state surface:

- `regime3_current_sensitive_wide24_*`
- artifact: `data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/`
- classes: `bull`, `bear`, `chop`

Whipsaw is not a regime class. Treat whipsaw-like behavior as risk, churn, instability, or transition context.

## Replacement For PRED

Use stability and transition-risk features instead of future regime class prediction:

- `regime3_stability_h6_score`
- `regime3_transition_h6_risk_prob`
- `regime3_transition_h6_risk_pred`
- `regime3_churn_h6_risk_score`

Source artifact:

- `data/ensemble/supervised/regime3_stability_risk_h6_20260530/`
- manifest: `data/ensemble/supervised/regime3_stability_risk_h6_20260530/CANDIDATE_STABILITY_RISK_H6_NO_CURRENT_20260530.json`

Use these only for:

- veto,
- size throttle,
- leverage reduction,
- TP/SL/hold tightening,
- uncertainty/risk context,
- transition-churn guardrails.

Do not map these values into direction.

## Validation Basis

The no-current stability/risk head avoids CURRENT probability inputs. The CURRENT sidecar is used only for label generation and evaluation.

2026 OOS:

- transition AUC: `0.6762`
- transition bacc at validation threshold: `0.5872`
- top 20% risk transition rate: `0.2874`
- low 20% risk transition rate: `0.0471`

This separates high-risk transition/churn zones, but it is not reliable enough to own future class direction.

## Fail-Fast Rules

- No Regime4 to Regime3 aliasing.
- No legacy regime compatibility layer in active or candidate paths.
- No silent fallback from missing `regime3_current_sensitive_wide24_*`.
- No silent fallback from missing stability/risk columns.
- If a model still requires `regime3_pred_*` as directional input, the model contract must fail and be corrected directly.
