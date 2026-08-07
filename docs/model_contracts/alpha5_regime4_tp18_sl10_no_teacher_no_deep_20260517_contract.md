# Alpha5 Regime4 TP18/SL10 No-Teacher No-Deep Contract

Date: 2026-05-17

## Purpose

Alpha5 keeps the Alpha4.3 simplified architecture but replaces the legacy clean-regime inputs.

Architecture:

```text
HGB parent -> direct no-teacher decision -> optional parent scale -> V21.2 runner -> corrected Alpha3 limit-close execution
```

Disabled layers:

```text
teacher sequence gate
V27/V31 deep scout
legacy clean_regime_2024_unsup_v4_* features
```

## Input Contract

Canonical fixed preprocessing inputs:

```text
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/fixed_regime4_tp18_sl10_preprocess_20260517/trade_candidates_2025_regime4_tp18_sl10_fixed.csv
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/fixed_regime4_tp18_sl10_preprocess_20260517/trade_candidates_2026_regime4_tp18_sl10_fixed.csv
```

Feature contract:

```text
parent feature count                  85
legacy clean_regime_2024_unsup_v4_*   0
clean_regime4_2024_unsup_v1_*         12
regime4_pred_*                        12
tp_sl_action_score                    enabled
```

Regime4 classes:

```text
bull
bear
chop
whipsaw
```

TP/SL action score:

```text
TP 1.8%
SL 1.0%
horizon 48 bars
entry next-bar open
same-bar tie -> SL wins
```

## Training And Selection

Script:

```text
/home/llewyn/crypto-scalping/scripts/train_eval_alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517.py
```

Split:

```text
train      2025-01-01 00:00:00 .. 2025-09-30 23:55:00
selection  2025-10-01 00:00:00 .. 2025-12-31 23:55:00
OOS        2026-01-01 00:00:00 .. 2026-02-28 16:00:00
```

Candidates:

```text
parent_direct_raw_no_teacher
parent_direct_scaled_no_teacher
```

Selection winner:

```text
parent_direct_scaled_no_teacher
```

Selected runtime:

```json
{
  "name": "noflip_c0.56_parent_scale1.10",
  "confidence": 0.56,
  "parent_notional_scale": 1.1,
  "max_notional": 2.75
}
```

Runner config:

```text
v21_2_parent_noop
```

## Artifacts

```text
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517/parent.pkl
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517/runners/parent_direct_raw_no_teacher_runner.pkl
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517/runners/parent_direct_scaled_no_teacher_runner.pkl
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517/alpha5_regime4_tp18_sl10_no_teacher_no_deep_summary.json
/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517/alpha5_regime4_tp18_sl10_no_teacher_no_deep_grid.csv
```

## Results

Selected by 2025Q4:

| Variant | Cost1 PnL | MDD | Cost2 PnL | Cost3 PnL | Trades |
|---|---:|---:|---:|---:|---:|
| `parent_direct_scaled_no_teacher` | +86.93% | -24.44% | +78.99% | +72.26% | 66 |

Non-selected OOS reference:

| Variant | Cost1 PnL | MDD | Cost2 PnL | Cost3 PnL | Trades |
|---|---:|---:|---:|---:|---:|
| `parent_direct_raw_no_teacher` | +92.29% | -23.22% | +88.98% | +85.45% | 69 |

## Verdict

Alpha5 is a valid Regime4 replacement candidate and confirms that the new feature contract can produce positive 2026 OOS PnL without teacher/deep layers.

It does **not** replace the stronger Alpha4.3 no-teacher/no-deep reference yet:

```text
Alpha4.3 reference cost1 +183.42%, MDD -21.99%, cost3 +79.27%
Alpha5 selected   cost1  +86.93%, MDD -24.44%, cost3 +72.26%
```

Promotion status:

```text
candidate_only
not_live_main
```

Next useful step is a fair Alpha4.3-vs-Alpha5 comparison where the only mutable surface is the regime feature replacement, while all runtime selection policy is held fixed.
