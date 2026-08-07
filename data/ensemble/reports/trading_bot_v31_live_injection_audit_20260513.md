# Trading Bot V31 Live Injection Audit

Date: 2026-05-13 KST

## Scope

`trading_bot.py` now injects `hf_v13_frozen_v27_rule_exit_overlay_v31_20260511` with the same high-level contract used by the V31 backtest:

- Parent LONG/SHORT entries still use `hf_v13_clean_regime_margin110_20260511`.
- V21.2 jackpot add-on remains available only on parent-owned positions.
- Frozen V27 deep scout is evaluated only when the parent decision is CASH.
- V31 rule exit overlay is applied only to `deep_alpha` positions opened by V27.
- Deep scout cooldown is separate from parent lifecycle cooldown, matching the backtest behavior where parent entries are not blocked by deep sleeve cooldown.

## Artifact Checks

- V31 report: `data/ensemble/reports/hf_v13_frozen_v27_rule_exit_overlay_v31_20260511_summary.json`
- V31 audit: `data/ensemble/reports/hf_v13_frozen_v27_rule_exit_overlay_v31_20260511_audit.json`
- V27 model: `data/ensemble/supervised/hf_v13_deep_alpha_candidate_expansion_v27_20260511/v27_deep_alpha_candidate_expansion.pt`
- V21.2 jackpot: `data/ensemble/supervised/hf_v13_jackpot_runner_v21_2_20260511/v21_2_jackpot_runner.pkl`
- Parent bundle: `data/ensemble/supervised/hf_v13_clean_regime_margin110_20260511/v13_clean_regime_margin110.pkl`

Loaded config: `v31_notional1_time_decay`

## Red Team Result

- V31 audit status: `pass`
- V31 verdict: `promote`
- `selection_uses_2026`: `false`
- `deep_sleeve_only_when_parent_cash`: `true`
- Runtime import: passed in `conda quant_ai`
- Runtime artifact load: passed in `conda quant_ai`
- V31 deep scout inference path: passed on a 2026 feature-frame sample

## Remaining Risks

- Live fills still depend on the existing next-bar-open scheduler in `trading_bot.py`; the V31 model layer now matches the backtest decision contract, but real exchange latency and order rejection are still live-execution risks.
- Existing unrelated whitespace warnings remain in `trading_bot.py`; they were not introduced or cleaned up in this injection pass.
