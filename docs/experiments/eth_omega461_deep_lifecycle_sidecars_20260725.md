# ETH Omega4.6.1 Deep Lifecycle Sidecars — 2026-07-25

Status: `research_only_not_live_promoted`

No live adapter, `trading_bot.py`, runtime configuration, environment setting, parent bundle, risk
sidecar, or dashboard wiring was changed.

## Priority 1 — Tabular Deep Lifecycle Risk Sidecar

Script:

- `scripts/research_eth_omega461_tabular_deep_lifecycle_sidecar_20260725.py`

Artifacts:

- `tmp/causal_regen_20260516/eth_omega461_tabular_deep_lifecycle_sidecar_20260725/report.json`
- `tmp/causal_regen_20260516/eth_omega461_tabular_deep_lifecycle_sidecar_20260725/model.pt`

Input dataset:

- `tmp/causal_regen_20260516/eth_omega461_censored_stopping_value_20260724/train_live_router_stopping_dataset.csv.gz`

The model is a small PyTorch MLP multi-task sidecar. It predicts:

- `advantage`: re-entry-aware log value of `EXIT now` versus `HOLD under frozen SLTP`
- `risk_label_h96`: three-class landmark outcome over 96 bars (`neither`, `take_profit`, `stop_loss`)

This is a label-learnability diagnostic only. It is not an exit owner and does not produce a
promotion candidate.

## Result

Dataset:

| item | value |
|---|---:|
| state rows | 65,982 |
| independent live-router positions | 85 |
| feature count | 46 |
| stop-loss baseline rows | 36,723 |
| take-profit baseline rows | 29,259 |
| h96 stop-loss rows | 4,501 |

Internal temporal validation split:

| metric | value |
|---|---:|
| validation rows | 13,253 |
| validation episodes | 13 |
| advantage RMSE | 0.0604 |
| advantage correlation | -0.0817 |
| h96 SL Brier | 0.0492 |
| h96 SL AUC | 0.9034 |
| h96 SL base rate | 0.0395 |
| mean predicted h96 SL probability | 0.0934 |

Interpretation:

- The 96-bar stop-loss risk signal is learnable in this feature set.
- The value/advantage target is not stable enough: validation correlation is negative.
- Therefore this model must not be used to decide exits or early liquidation.
- A future sidecar may use only the calibrated risk score as a shadow diagnostic, after separate
  forward collection.

## Deployment Verdict

`DO_NOT_APPLY_TO_LIVE`

Blockers:

- only 85 independent positions
- validation is an internal temporal split of an already researched dataset
- no untouched forward window
- no fresh-forward PnL replay for this model
- no calibration gate for the overpredicted SL probability
- no `trading_bot.py` wiring by design

## Replay Attachment Diagnostic

Script:

- `scripts/eval_eth_omega461_tabular_deep_risk_sidecar_replay_20260725.py`

Artifacts:

- `tmp/causal_regen_20260516/eth_omega461_tabular_deep_risk_sidecar_replay_20260725/report.json`
- `tmp/causal_regen_20260516/eth_omega461_tabular_deep_risk_sidecar_replay_20260725/validation_grid.csv`

The trained tabular MLP was attached only to the offline research replay harness. It uses only
`p_sl_h96`; the unstable `advantage` head is ignored. The candidate exits after TP/SL checks when
`p_sl_h96` clears a validation-selected threshold.

Validation-selected rule:

- `sl_probability_min=0.30`
- `min_hold_bars=96`
- `persistence=1`

Validation result:

| policy | PnL | close-MTM MDD | realized MDD | trades | exit reasons |
|---|---:|---:|---:|---:|---|
| live baseline replay | +18.10% | -19.81% | -14.40% | 31 | SL 20 / TP 10 / censored 1 |
| deep risk sidecar | +21.83% | -15.87% | -13.55% | 79 | deep 60 / SL 8 / TP 10 / censored 1 |

Diagnostic OOS result, Cost1:

| policy | PnL | close-MTM MDD | realized MDD | trades |
|---|---:|---:|---:|---:|
| live baseline replay | +21.87% | -16.77% | -14.51% | 23 |
| deep risk sidecar | +61.98% | -19.08% | -18.28% | 59 |

Diagnostic extension result, Cost1:

| policy | PnL | close-MTM MDD | realized MDD | trades |
|---|---:|---:|---:|---:|
| live baseline replay | -6.65% | -20.13% | -15.97% | 17 |
| deep risk sidecar | -24.50% | -30.12% | -29.31% | 89 |

Verdict: `REJECTED_FOR_LIVE_AND_SHADOW_PROMOTION`

Reason:

- Validation improvement is driven by many early exits and re-entries.
- OOS PnL improves in the already-touched diagnostic window, but MDD worsens.
- Extension fails hard: lower PnL, much worse close-MTM MDD, and much worse realized MDD.
- The candidate changes the trade sequence too aggressively (`31 -> 79`, `17 -> 89` trades).
- This confirms that the deep risk score may rank adverse states, but attaching it as an exit
  action is not robust.

## Priority 2 — Sequence Lifecycle Sidecar

Script:

- `scripts/research_eth_omega461_sequence_lifecycle_sidecar_20260725.py`

Artifacts:

- `tmp/causal_regen_20260516/eth_omega461_sequence_lifecycle_sidecar_20260725/report.json`
- `tmp/causal_regen_20260516/eth_omega461_sequence_lifecycle_sidecar_20260725/model.pt`

The model is a small TCN-style PyTorch sidecar over the last 48 in-position state rows. It uses
the same targets as the tabular MLP: `advantage` and `risk_label_h96`.

Internal temporal validation split:

| metric | tabular MLP | sequence TCN |
|---|---:|---:|
| validation rows | 13,253 | 13,253 |
| validation episodes | 13 | 13 |
| advantage RMSE | 0.0604 | 0.0686 |
| advantage correlation | -0.0817 | -0.1498 |
| h96 SL Brier | 0.0492 | 0.0784 |
| h96 SL AUC | 0.9034 | 0.9049 |
| mean predicted h96 SL probability | 0.0934 | 0.1234 |

Interpretation:

- Sequence context did not improve the current diagnostic materially.
- SL ranking AUC is similar to tabular MLP, but probability calibration is worse.
- Advantage/value target is worse than tabular MLP and remains unusable.
- With only 85 independent positions, TCN capacity is not justified yet.

Verdict: `RESEARCH_ONLY_REJECT_SEQUENCE_FOR_NOW`

## Next Sequence Design Gate

A future sequence experiment should stay research-only and use more independent positions before
trying larger models.

Recommended target:

- input: last 32 to 96 in-position states per trade
- model: small TCN first, then small Transformer/Mamba only if TCN is not enough
- outputs: `p_sl_h96`, `p_tp_h96`, `expected_adverse_move`, optional `advantage`
- allowed use: shadow score only
- forbidden use: direct close signal, TP/SL replacement, live order modification

Acceptance gates before even considering shadow collection:

- at least 300 independent positions, preferably 1,000
- episode-level split, not row-level split
- positive h96 SL AUC on a fresh temporal holdout
- calibrated SL probability, not just ranking AUC
- no negative value-target correlation if `advantage` is exposed
- explicit `research_only_not_live_promoted` artifact status
