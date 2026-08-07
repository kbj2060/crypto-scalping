# Alpha8 Research Log 2026-05-29

## Active Contract

- Active regime inputs are fail-fast and v2-only.
- Allowed Layer 1 outputs:
  - `clean_regime4_state24_sticky090_v2_*`
  - `regime4_pred_*`
- Forbidden active prefixes:
  - `clean_regime_2024_unsup_v4_*`
  - `clean_regime4_2024_unsup_v1_*`
- No alias, fallback prefix, or legacy compatibility layer is allowed in active candidate or live paths.

## Clean Funding Reset

The pre-clean-funding Alpha8 high-cap, DSAC, IQN, and Mamba runs are retained only as research logs. They must not be used for model selection or live wiring because they used the pre-clean-funding `01965` candidate frame and related artifacts.

Clean-funding `01965` artifacts were regenerated from the old timestamp skeleton with exact timestamp overlays from the clean funding frame, clean M7 frames, and clean regime predictor sidecars. The regenerated frame passed fail-fast checks:

- `selection_uses_2026`: `false`
- forbidden legacy regime column count: `0`
- current regime surface count: `20`
- future regime predictor count: `12`
- fresh `tp_sl_action_score` sidecar: rebuilt on the clean frame

## Clean Alpha7 01965 Baseline

Clean Alpha7 was retrained on the regenerated `01965` candidate CSVs and selected only by 2025Q4 validation. A red-team follow-up found that `funding`-named columns were clean but several funding-derived columns without `funding` in their names could still be inherited from the old base frame. The builder now overlays the full funding-derived family from the clean 2026 feature frame and uses the M7 CSV only for `m7_*` plus clean `sig_ai_squeeze`.

The selected clean baseline remains `primary_no_tp_fallback_v2` by 2025Q4 validation. `primary_v2_fallback_no_tp` has higher reported OOS but is not the validation-selected baseline.

| Variant | Val PnL | Val MDD | Val Trades | Val WR | OOS PnL | OOS MDD | OOS Trades | OOS WR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `primary_no_tp_fallback_v2` | 42.73% | -29.52% | 90 | 11.11% | 43.95% | -32.23% | 76 | 18.42% |
| `both_v2_tp` | 24.47% | -27.77% | 90 | 11.11% | 46.90% | -32.18% | 78 | 15.38% |
| `primary_v2_fallback_no_tp` | 9.71% | -29.40% | 103 | 9.71% | 55.30% | -31.47% | 73 | 16.44% |
| `both_no_tp` | -7.38% | -34.34% | 103 | 7.77% | 45.51% | -33.49% | 73 | 17.81% |

## Clean Alpha8 Mamba DSAC Retest

Alpha8 was retested against the clean Alpha7 baseline and clean candidate CSVs. Direction ownership stayed with the clean Alpha7 primary/fallback combo. The Mamba DSAC layer selected only TP/SL/hold/mult/cap/veto risk buckets using full counterfactual trade net PnL labels with exact entry/exit notional cost accounting.

| Split | Variant | PnL | MDD | Trades | Trades/Day | WR | Avg Notional |
|---|---|---:|---:|---:|---:|---:|---:|
| Val | `fixed_60_aggressive` | 91.83% | -36.38% | 100 | 1.09 | 46.00% | 4.06 |
| Val | `fixed_54_highcap` | 67.87% | -22.45% | 78 | 0.85 | 46.15% | 2.75 |
| Val | `fixed_52_highwr` | 55.87% | -18.67% | 76 | 0.83 | 46.05% | 2.53 |
| Val | `baseline_combo` | 42.73% | -29.52% | 90 | 0.98 | 11.11% | 2.33 |
| Val | `dist_actor_greedy` | -9.56% | -25.43% | 69 | 0.75 | 37.68% | 2.46 |
| OOS | `fixed_60_aggressive` | 65.81% | -26.94% | 113 | 1.93 | 51.33% | 4.19 |
| OOS | `baseline_combo` | 43.95% | -32.23% | 76 | 1.30 | 18.42% | 2.41 |
| OOS | `fixed_52_highwr` | 10.58% | -37.53% | 73 | 1.24 | 52.05% | 2.73 |
| OOS | `dist_actor_greedy` | 8.85% | -32.81% | 76 | 1.30 | 43.42% | 2.54 |
| OOS | `fixed_54_highcap` | -0.67% | -41.31% | 76 | 1.30 | 51.32% | 2.97 |

## Selection Decision

- `fixed_60_aggressive` is validation-best and reports OOS `65.81%` PnL / `51.33%` WR after the stricter funding-derived overlay, but it is still a validation-selected research result with high MDD and is not live-wired.
- The Mamba DSAC actor is still weak: OOS `8.85%` PnL / `43.42%` WR, so it is research-only.
- The previously promoted-looking fixed 54/55 high-cap contract collapses to OOS `-0.67%` after the stricter funding-derived overlay.
- The previous high-cap `193%` and `225%` results are invalid as model-selection evidence after the clean-funding reset.
- No Alpha8 clean-funding candidate currently passes promotion. Nothing from this retest is live-wired.

## Artifacts

- `/home/llewyn/crypto-scalping/scripts/build_alpha7_01965_cleanfunding_candidates_20260529.py`
- `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/candidate_cleanfunding_audit.json`
- `/home/llewyn/crypto-scalping/scripts/retrain_alpha7_01965_cleanfunding_live_20260529.py`
- `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha7_submodel_01965_cleanfunding_v1_20260529/report.json`
- `/home/llewyn/crypto-scalping/data/ensemble/supervised/alpha7_submodel_01965_cleanfunding_v1_20260529/alpha7_live_manifest.json`
- `/home/llewyn/crypto-scalping/scripts/train_eval_alpha8_highcap_distributional_dsac_risk_20260529.py`
- `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha8_highcap_mamba_seq_dsac_risk_cleanfunding_20260529/summary.json`
- `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha8_highcap_mamba_seq_dsac_risk_cleanfunding_20260529/grid.csv`
- `/home/llewyn/crypto-scalping/logs/alpha8_highcap_mamba_seq_dsac_risk_cleanfunding_20260529.log`
