# Omega4.3.1 Validation-Only Tail-Downside Risk Guard Contract - 2026-06-23

## Status

- Model id: `omega4_3_1_valonly_tail_downside_guard_20260623`
- Status: `research_candidate_redteam_pass_not_live_wired`
- Base model: `omega4_3_valonly_logrisk_tail050_margin_leverage_sidecar_20260623`
- Manifest: `data/ensemble/supervised/omega4_3_1_valonly_tail_downside_guard_20260623/candidate_manifest.json`
- Runtime contract: `tmp/causal_regen_20260516/omega4_3_1_valonly_tail_downside_guard_20260623/runtime_contract.json`
- Risk sidecar: `tmp/causal_regen_20260516/omega4_3_1_valonly_tail_downside_guard_20260623/risk_sidecar.pkl`
- Red-team report: `docs/audits/omega4_3_1_valonly_tail_downside_guard_redteam_20260623.md`

This candidate keeps the Omega4.3 parent direction, quality, exit-head timing,
and raw ATR safety SLTP unchanged. It changes only the risk sidecar sizing
layer by adding a side-split quantile HGB q10 downside guard.

## Paper-Motivated Change

HF Papers reviewed:

- [A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification](https://huggingface.co/papers/2107.07511)
- [Sequential Predictive Conformal Inference for Time Series](https://huggingface.co/papers/2212.03463)
- [Distributional Reinforcement Learning with Quantile Regression](https://huggingface.co/papers/1710.10044)
- [A Model-Based Method for Minimizing CVaR and Beyond](https://huggingface.co/papers/2305.17498)
- [TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling](https://huggingface.co/papers/2410.24210)

Applied idea: keep the mean score HGB, train a side-split quantile HGB q10 on
the same trade-level target, and shrink margin/leverage toward floor only when
predicted downside is in the training tail. This is a small-sample,
distributional tail-risk guard rather than a parent retrain.

## Runtime Contract

```text
notional = margin_fraction * leverage
PnL = realized_price_move * notional
SLTP = raw directional price_move barriers; margin/notional do not move TP/SL lines
```

Selected variant: `risk_0488`

```json
{
  "min_scale": 0.75,
  "max_scale": 1.45,
  "temp": 2.1,
  "floor": 0.06,
  "cap": 0.32,
  "long_scale": 0.65,
  "short_scale": 1.25,
  "leverage_min": 1.25,
  "leverage_max": 3.0,
  "leverage_temp": 1.7,
  "leverage_floor": 1.0,
  "leverage_cap": 3.0,
  "long_leverage_scale": 0.75,
  "short_leverage_scale": 1.25
}
```

Tail-downside guard:

```json
{
  "q10_model": "side_split_quantile_hgb",
  "downside_profile": {
    "downside_q75": 0.06493431837973136
  },
  "downside_alpha": 0.8,
  "downside_min_shrink": 0.65,
  "downside_tail_only": true
}
```

## Metrics

| Split | PnL | MDD | WR | Trades | Avg Notional | Avg Margin | Avg Lev | Log-Risk Utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline validation | `+30.33%` | `-7.91%` | `67.06%` | `85` | `0.5536` | `0.2346` | `2.2574` | `0.205548` |
| Candidate validation | `+30.09%` | `-6.94%` | `67.06%` | `85` | `0.5327` | `0.2230` | `2.2517` | `0.206614` |
| Baseline OOS readout | `+32.44%` | `-5.72%` | `66.15%` | `65` | `0.5613` | `0.2374` | `2.2699` | `0.262865` |
| Candidate OOS readout | `+32.28%` | `-5.54%` | `66.15%` | `65` | `0.5348` | `0.2231` | `2.2639` | `0.263117` |

## Caveats

- Not live-wired.
- Full replay diagnostic still has weaker validation utility than the sizing-only contract.
- Runtime-native parity, exchange leverage/margin sync, and shadow or paper smoke remain required before live use.
