# Omega1.2 SoftFloor00 TabM Diffusion Risk Contract

## Status

- Model id: `omega1_2_softfloor00_tabm_diffusion_risk_20260603`
- Status: research executed, not promoted
- Reason: full active-row OOS underperformed the fixed Omega1.2 risk template.

## Scope

This experiment keeps Omega1.2 TabM ExpertDQ frozen and replaces only the fixed risk template fields:

- `take_profit`
- `stop_loss`
- `leverage`
- `notional_exposure`

The experiment does not change:

- Regime3 routing
- TabM direction heads
- TabM quality heads
- `final_action`
- `quality_for_action`
- `max_hold_bars = 72`
- `cooldown_bars = 6`
- expert notional scale layer: `bull=0.75`, `bear=0.90`, `chop_expert=0.90`

## Source Contract

Frozen TabM source:

- `tmp/causal_regen_20260516/omega1_regime3_routed_expert_direction_quality_tabm_20260602/soft_floor_0p00/training_features_2025_soft_floor_0p00_omega1_regime3_expertdq_oof_20260602.csv`
- `tmp/causal_regen_20260516/omega1_regime3_routed_expert_direction_quality_tabm_20260602/soft_floor_0p00/training_features_2026_rebuilt_soft_floor_0p00_omega1_regime3_expertdq_20260602.csv`

Frame overlays are required to match the Omega1.2 replay baseline:

- `regime3_current_sensitive_wide24_*`
- `regime3_cmamba_h6_sidecar_*`
- `regime3_stability_h6_*`

## Forbidden Features

The active state builder rejects these inputs:

- `clean_regime4_*`
- `regime4_pred_*`
- `regime3_pred_*`
- `tp_sl_action_score`
- `teacher_*`
- `a5dir_*`
- target/future/label/pnl/zigzag/wave3 tokens

No alias or compatibility fallback is allowed. Contract mismatches must fail fast.

## Execution Accounting

The script is self-contained and does not import prior replay/backtest helpers.

It reproduces the Omega1.2 fixed-template Cost3 baseline with:

- `next_open_limit_touch0_fee20`
- entry maker-limit attempt on `i+1` high/low
- entry miss skips the signal
- exit maker-limit attempt on `i+1` high/low
- exit miss falls back to `i+1` close with taker fee/slippage
- Cost3 uses `fee=0.0005`, `slip=0.0002`, `cost_mult=3.0`

Baseline parity check:

- Existing Omega1.2 soft_floor_0p00 OOS Cost3: `+4.3417%`, MDD `-9.0853%`, trades `205`, WR `49.27%`
- Self-contained fixed template OOS: `+4.3417%`, MDD `-9.0853%`, trades `205`, WR `49.27%`

## Diffusion Policy

Architecture:

- conditional denoising MLP
- state encoder: `Linear -> LayerNorm -> SiLU -> Dropout -> Linear -> SiLU`
- sinusoidal diffusion timestep embedding
- denoiser input: state embedding + timestep embedding + noisy risk vector
- output: predicted risk-vector noise for 4 dimensions

Risk bounds:

- `take_profit`: `[0.008, 0.050]`
- `stop_loss`: `[0.006, 0.035]`
- `leverage`: `[1.0, 5.0]`
- `notional`: `[0.10, 0.90]` before expert scale

Training data:

- all 2025 active rows: `30675`
- counterfactual samples per row: `32`
- kept top-k samples per row: `4`
- diffusion samples: `122700`
- scorer samples: `981600`

## Full Run Results

| Variant | Split | PnL | MDD | Trades | WR |
|---|---:|---:|---:|---:|---:|
| fixed_template | validation | -11.17% | -14.70% | 334 | 41.32% |
| fixed_template | OOS | +4.34% | -9.09% | 205 | 49.27% |
| diffusion_direct | validation | +23.03% | -13.70% | 341 | 46.04% |
| diffusion_direct | OOS | -7.08% | -15.91% | 213 | 52.11% |
| diffusion_sample_rerank | validation | -16.66% | -22.72% | 332 | 45.78% |
| diffusion_sample_rerank | OOS | -1.51% | -6.36% | 204 | 50.98% |

## Decision

Do not promote `omega1_2_softfloor00_tabm_diffusion_risk_20260603` to live.

The full active-row diffusion replacement improved validation in the direct path but failed OOS. This indicates the risk generator learned validation-specific risk surfaces rather than a robust replacement for the fixed template.

The rerank path reduced OOS MDD versus fixed template but also reduced PnL below fixed. It may be useful as a risk-throttle research direction, not as the current Omega1.2 risk replacement.

## Artifacts

- Script: `scripts/train_eval_omega1_2_tabm_diffusion_risk_20260603.py`
- Output dir: `tmp/causal_regen_20260516/omega1_2_softfloor00_tabm_diffusion_risk_20260603`
- Report: `tmp/causal_regen_20260516/omega1_2_softfloor00_tabm_diffusion_risk_20260603/report.json`
- Ranking: `tmp/causal_regen_20260516/omega1_2_softfloor00_tabm_diffusion_risk_20260603/ranking.csv`
- Model checkpoint: `tmp/causal_regen_20260516/omega1_2_softfloor00_tabm_diffusion_risk_20260603/diffusion_risk_policy.pt`
