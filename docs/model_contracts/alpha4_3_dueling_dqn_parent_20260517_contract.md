# Alpha4.3 Dueling DQN Parent Replacement Test - 2026-05-17

## Purpose

Replace only the Alpha4.3 HGB parent action model with a Dueling DQN action model.

Held fixed:

- Input CSV: fixed `1.8% / 1.0%` HGB `tp_sl_action_score` Alpha4.3 feature frame.
- Quality model: inherited from Alpha4.3 HGB parent.
- Bucket models: inherited from Alpha4.3 HGB parent.
- Runtime scale: `0.85`.
- V21.2 runner: inherited `parent_direct_scaled_no_teacher_runner.pkl`.
- Teacher: disabled.
- V27/V31 deep scout: disabled.
- Execution: corrected Alpha3 limit-close contract.

This isolates the parent direction decision: `HOLD / LONG / SHORT`.

## Model

Files:

- Module: `/home/llewyn/crypto-scalping/ensemble/dueling_dqn_parent.py`
- Experiment script: `/home/llewyn/crypto-scalping/scripts/train_eval_alpha4_3_dueling_dqn_parent_20260517.py`

Architecture:

- Dueling Q network.
- Actions: `0=HOLD`, `1=LONG`, `2=SHORT`.
- Hidden dim: 256.
- Dropout: 0.05.
- GPU: CUDA, NVIDIA GeForce RTX 3070 Ti.

Training:

- Train window: 2025 rows before `2025-10-01`.
- Validation: 2025Q4.
- OOS: 2026.
- Label source: existing Alpha4 governor utility labels from `build_training_set`.
- Stride: 6.
- Steps: 2500.
- Batch: 512.
- Gamma: 0.82.
- Behavior-cloning stabilizer: 0.20.
- PER-like prioritized sampling: reward magnitude plus non-cash emphasis.

Artifact:

`/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha4_3_dueling_dqn_parent_20260517/dueling_dqn_parent.pkl`

## Raw DQN Result

Report:

`/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha4_3_dueling_dqn_parent_20260517/alpha4_3_dueling_dqn_parent_summary.json`

Training action distribution:

- HOLD: 10546
- LONG: 1203
- SHORT: 1298

Raw DQN decision distribution:

| Split | HOLD | LONG | SHORT |
| --- | ---: | ---: | ---: |
| 2025Q4 validation | 10909 | 1916 | 13671 |
| 2026 OOS | 8015 | 1188 | 7694 |

Raw DQN over-trades badly.

| Split | Cost1 PnL | MDD | Cost2 PnL | Cost3 PnL | Trades |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2025Q4 validation | -18.42% | -49.84% | -60.39% | -48.83% | 604 |
| 2026 OOS | +29.44% | -45.07% | +43.91% | +0.49% | 363 |

## Validation-Selected Gate Test

Because raw DQN ignored the hold prior, a trade confidence gate was tested.

Gate semantics:

```text
if best action is LONG/SHORT and
   (trade_probability < trade_min_prob or trade_probability - hold_probability < trade_margin):
    force HOLD
```

Gate selection used only 2025Q4 validation.

Best validation gate:

- `trade_min_prob = 0.95`
- `trade_margin = 0.90`

Validation result for selected gate:

- cost1 PnL: +12.75%
- MDD: -42.20%
- cost2 PnL: -14.06%
- cost3 PnL: -36.50%
- trades: 530

OOS result for that validation-selected gate:

- cost1 PnL: +123.72%
- MDD: -35.76%
- cost2 PnL: +186.10%
- cost3 PnL: +104.46%
- trades: 269

OOS action distribution:

- HOLD: 10817
- LONG: 485
- SHORT: 5595

Gate sweep artifact:

`/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha4_3_dueling_dqn_parent_20260517/alpha4_3_dueling_dqn_parent_gate_sweep.json`

Gated parent artifact:

`/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha4_3_dueling_dqn_parent_20260517/dueling_dqn_parent_gated.pkl`

## Baseline

Existing Alpha4.3 no-teacher/no-deep:

- cost1 PnL: +183.42%
- MDD: -21.99%
- cost2 PnL: +169.76%
- cost3 PnL: +79.27%
- trades: 66

## Verdict

Do not promote this Dueling DQN parent.

Raw DQN fails because it over-trades and collapses MDD. The validation-selected gate improves 2026 cost2/cost3, but validation quality is poor, trade count remains too high, and MDD is much worse than baseline. The OOS cost2/cost3 lift is therefore not a robust promotion signal.

Next iteration should change the DQN training target before changing architecture:

- Add explicit hold/turnover penalty to the reward.
- Remove non-cash oversampling or make it validation-calibrated.
- Train with conservative CQL-style penalty for unseen trade actions.
- Add target trade-rate regularization around the Alpha4.3 baseline range of 50-90 OOS trades.
- Only after DQN action quality passes validation should DSAC sizing be added.
