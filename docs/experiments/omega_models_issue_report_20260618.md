# Omega Model Lineage and Issue Report

Date: 2026-06-18

## Executive Summary

현재까지 테스트한 신규 변경안 중 바로 live로 승격할 만한 모델은 없다.

운영 기준으로 가장 안정적인 기준선은 여전히 `live 8b global cash sleeve + hardcoded risk` 계열이다. 다만 최근 실험에서 `SL`이 너무 타이트할 가능성, fallback risk가 레짐/변동성에 둔감하다는 문제, OOS에서는 ATR 기반 SL 확장이 유효할 수 있다는 신호가 확인됐다.

핵심 결론은 다음과 같다.

- 부모모델은 Regime3-routed True 3-Head TabM 구조이며, primary entry는 quality threshold 때문에 매우 선택적으로 발생한다.
- live 8b sleeve는 부모가 CASH일 때만 작동하는 global EV/utility fallback 모델이다.
- sleeve에 regime threshold 또는 regime expert를 추가하는 실험은 validation 기준으로 실패했다.
- dynamic risk, learned exit, bucket risk 등은 일부 OOS 개선 신호가 있었지만 validation-only 선택 기준을 통과하지 못했다.
- 현재 가장 실용적인 다음 실험은 전체 동적 리스크가 아니라 `SL width`만 제한적으로 넓히는 실험이다.

## Current Live-Oriented Architecture

```text
Market features
-> Regime3 Current Router
-> bull / bear / chop expert
-> True 3-Head TabM parent
   - Direction Head: cash / long / short
   - Quality Head: action quality
   - Exit Head: lifecycle / exit signal
-> quality threshold 0.8
-> parent primary action
-> if parent is CASH:
   -> live 8b global cash sleeve
   -> long EV lower-bound model
   -> short EV lower-bound model
   -> utility agreement filter
   -> support/OOD gate
   -> fallback action
   -> hardcoded risk
```

## Parent Model Specification

Parent artifact:

```text
tmp/causal_regen_20260516/omega1_2_true_3head_tabm_20260603_full_retrain_cash_alpha43_20260608
```

Model ID:

```text
omega1_2_true_3head_tabm_20260603_full_retrain_cash_alpha43_20260608
```

Architecture:

```text
Regime3-routed True 3-Head TabM
experts: bull, bear, chop
base features: 172
position features: 13
total TabM features: 185
quality threshold: 0.8
```

Parent-only operating performance:

| Split | PnL | MDD | WR | Trades | Long | Short |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 100.5427 | -10.6777 | 0.6364 | 33 | 9 | 24 |
| OOS | 72.7600 | -8.1082 | 0.7222 | 18 | 3 | 15 |

Parent primary risk template:

```text
TP = 0.026
SL = 0.014
notional = 0.30
leverage = 2.0
max_hold = 192
```

Important distinction:

The raw `true3head` exit-head threshold sweep is not the live parent baseline. The live comparison baseline is the `cash_alpha43` aggressive primary-only decision builder applied to the full-retrained 3-head parent predictions.

## Live 8b Sleeve Specification

Live 8b bundle:

```text
data/ensemble/supervised/omega1_2_8b_full_retrain_numeric_cash_sleeve_leverage_only_20260618/numeric_cash_sleeve.joblib
```

Model role:

```text
Only activates when parent is CASH.
Uses parent trace + market features.
Predicts conservative long/short fallback EV.
```

Model types:

```text
long EV lower-bound model: HistGradientBoostingRegressor
short EV lower-bound model: HistGradientBoostingRegressor
utility long model: HistGradientBoostingRegressor
utility short model: HistGradientBoostingRegressor
```

Feature count:

```text
42
```

Important features:

```text
router_confidence
router_margin
dir_p_cash / dir_p_long / dir_p_short
quality_p_cash / quality_p_long / quality_p_short
quality_for_action
router_is_bull / router_is_bear / router_is_chop
market volatility and return features
primary cash history features
```

Hardcoded fallback risk:

```text
TP = 0.052
SL = 0.028
notional = 0.81
leverage = 2.0
max_hold = 192
```

Actual exported 8b global replay:

| Split | PnL | MDD | WR | Trades | Fallback Entries |
|---|---:|---:|---:|---:|---:|
| Validation | 151.0178 | -10.6777 | 0.6667 | 42 | 9 |
| OOS | 86.1076 | -8.2860 | 0.5938 | 32 | 14 |

Note:

Earlier `98.17` was from a selected row in the research report. The currently exported live bundle manifest targets `ev_min=0.004`, `utility_min=0.002`, and replays to OOS `86.11`.

## Tested Model Families

### 1. Full-Retrain Numeric Cash Sleeve / Live 8b

Purpose:

Replace simple label fallback with numeric EV lower-bound and utility agreement.

Result:

The family produced the current best live-oriented sleeve structure. It improves parent-only OOS but remains sensitive to fallback risk settings.

Judgment:

Maintain as current baseline. Do not replace with later failed variants.

### 2. Omega3 Dynamic Risk Candidate

Promoted candidate:

```text
omega3_full_retrain_hf7_dynamic_cash_sleeve_20260618
```

Purpose:

Add dynamic risk heads for TP, SL, notional, and leverage.

Observed issue:

Dynamic risk flexibility did not translate into stable validation-selected improvement. Some variants increased notional or tightened effective SL, causing more stop losses.

Judgment:

Do not live-promote over live 8b.

### 3. Margin Fraction / Fixed Leverage Experiments

Tested ideas:

```text
margin_fraction head
leverage fixed at 3
leverage fixed at 2
margin cap 1.0
```

Key finding:

Changing fixed leverage from 3 to 2 did not change performance when the learned margin/notional relationship preserved effective notional. This confirmed that notional exposure, not leverage alone, drives PnL and TP/SL account thresholds.

Judgment:

Useful conceptual correction, but no live model.

### 4. Bucket Margin Risk

Best bucket-style result:

```text
margin cap1 bucket fixedlev2
OOS PnL around 87.47
```

Issue:

Bucket distribution concentrated heavily in one bucket and did not beat the best live 8b baseline robustly.

Judgment:

No live promotion.

### 5. Learned Exit / No Hard SLTP Exit

Purpose:

Stop using SLTP directly for exit and train learned exit using distance-to-TP/SL and market features.

Result:

Validation often improved, but OOS worsened.

Observed issue:

Learned exit overfit validation and failed to beat hard SLTP risk control. Removing hard protective exits made OOS risk worse.

Judgment:

Discard for live path.

### 6. Profit Exit With SL Floor

Purpose:

Keep hard SL floor but replace hard TP with profit-exit classifier near TP.

Result:

Reduced some stop behavior but PnL remained below hardcoded live 8b.

Judgment:

No live promotion.

### 7. Regime Threshold Sleeve

Purpose:

Keep global EV/utility models but select `bull/bear/chop` EV thresholds separately.

Validation-selected result:

```text
thresholds: bull 0.001 / bear 0.001 / chop 0.001
VAL PnL: 170.7858
OOS PnL: 93.9790
```

Issue:

Validation did not select true regime differentiation. It simply lowered all thresholds.

OOS diagnostic best:

```text
bull 0.004 / bear 0.001 / chop 0.006
OOS PnL: 102.2025
```

But this is OOS-selected and cannot be used for live promotion.

Judgment:

Do not promote. The signal suggests a possible future rule family, but not a validated one.

### 8. Regime Expert Sleeve

Purpose:

Train separate bull/bear/chop EV and utility models.

Label rows:

```text
bull: 6464
bear: 5051
chop: 8469
```

Validation-selected result:

```text
thresholds: bull 0.006 / bear 0.003 / chop 0.006
VAL PnL: 125.8884
OOS PnL: 83.3315
```

Global control:

```text
VAL PnL: 151.0178
OOS PnL: 86.1076
```

Issue:

Despite enough rows, splitting into regime experts reduced generalization. The global model handled cross-regime interactions better.

Judgment:

Discard for active path.

### 9. Uncertainty + ATR Dynamic Risk

Purpose:

Implement report recommendation:

```text
uncertainty = 1 - dir_p_cash * router_confidence
notional = base_notional * uncertainty_scale
TP/SL = base TP/SL * ATR scalar
```

Validation-selected result:

```text
static_control won
VAL PnL: 151.0178
OOS PnL: 86.1076
```

OOS diagnostic best:

```text
unc_n060_120_atr080_140
VAL PnL: 144.0286
OOS PnL: 106.1150
OOS MDD: -8.1665
OOS WR: 0.6207
OOS fallback stop_loss: 0
avg fallback notional OOS: 0.7845
avg TP: 0.0691
avg SL: 0.0372
```

Issue:

OOS looks better, but validation-only selection rejects it. It cannot be promoted without violating selection discipline.

Judgment:

Do not promote, but keep the signal. The likely useful component is not full dynamic risk, but wider SL around `0.035` to `0.038`.

## Main Problems Discovered

### Problem 1: Parent Emits Many Direction Signals but Quality Gate Suppresses Most

Direction head has many long/short candidates, but final action is mostly CASH after quality threshold `0.8`.

Validation final actions:

```text
cash: 25587
long: 100
short: 803
```

OOS final actions:

```text
cash: 16330
long: 105
short: 385
```

Implication:

The sleeve operates in a large, filtered CASH region. This region is not uniform. Some rows are true uncertainty; others are quality-gated directional opportunities.

### Problem 2: Fallback Stop Loss Sensitivity

The live 8b fixed SL is:

```text
SL = 0.028 account return
notional = 0.81
implied price move ≈ 0.028 / 0.81 = 3.46%
```

Some dynamic/bucket variants made effective price SL too tight and increased stop losses. Conversely, the uncertainty+ATR diagnostic improved OOS by widening average SL to around `0.037` account return.

Implication:

The next high-value experiment should isolate SL width rather than changing every risk component.

### Problem 3: Regime Splitting Did Not Improve Sleeve

Both regime threshold and regime expert experiments failed validation criteria.

Likely reasons:

- The global HGB model already uses `router_is_bull/bear/chop` and learns useful interactions.
- Splitting experts reduces cross-regime generalization.
- Regime-specific threshold tuning risks overfitting and did not produce validation-stable differentiation.

Implication:

Do not add sleeve regime experts to live path.

### Problem 4: Learned Exit Overfits

Learned exit logic improved some validation cases but degraded OOS. Hard SLTP remains more robust.

Implication:

Protective hard exits should remain. Learned exit may only be tested as an advisory layer, not as a replacement for hard SL.

### Problem 5: Dynamic Multi-Head Risk Is Too Flexible

Learning TP, SL, notional, and leverage simultaneously created unstable behavior. Some improvements were diagnostic-only, not validation-selected.

Implication:

Risk experiments should be constrained and one-dimensional:

```text
1. SL width only
2. TP width only
3. notional cap only
4. max_hold only
```

## Live Promotion Status

| Candidate | Status | Reason |
|---|---|---|
| live 8b global sleeve hardcoded risk | Keep | Current strongest live-oriented baseline |
| regime threshold sleeve | Reject | Validation did not select real regime differentiation |
| regime expert sleeve | Reject | Underperformed global model on validation |
| uncertainty + ATR dynamic risk | Hold | OOS signal strong, validation rejects |
| learned exit / no hard SLTP | Reject | OOS degradation |
| bucket margin risk | Hold/Reject | Did not beat live 8b robustly |
| Omega3 dynamic HF7 risk | Hold | Too much flexibility, unstable |

## Recommended Next Experiments

### Experiment A: SL-Only Width Sweep

Rationale:

The dynamic risk diagnostic suggests the useful part is wider SL, not full dynamic risk.

Test:

```text
Keep live 8b entry model fixed.
Keep TP = 0.052.
Keep notional = 0.81.
Keep leverage = 2.
Sweep SL:
0.028, 0.032, 0.035, 0.037, 0.040, 0.045
```

Selection:

```text
Validation-only.
OOS diagnostic only.
```

Expected success condition:

Validation selected SL is wider than `0.028` and OOS does not degrade MDD materially.

### Experiment B: Regime-Aware SL Width Only

Only run if Experiment A passes.

Test:

```text
chop SL: 0.028 to 0.034
bull/bear SL: 0.035 to 0.045
TP fixed.
notional fixed.
```

Reason:

Previous volatility audit showed bull/bear regimes have larger forward ranges than chop.

### Experiment C: Primary Takeover Protocol Audit

Current fallback exits on primary activation. Need separate accounting:

```text
same-side parent takeover
opposite-side parent override
neutral primary activation
```

Goal:

Determine whether fallback positions are being closed too early on same-side parent activation.

### Experiment D: Cash Reason Taxonomy

Classify parent CASH rows:

```text
direction cash-dominant
quality veto
low router confidence
low direction margin
high entropy
```

Then measure fallback PnL by bucket.

Goal:

Understand where sleeve actually adds value and where it should stay off.

## Current Working Recommendation

Do not promote any of the latest experimental models.

Keep the active live baseline as:

```text
parent: Regime3-routed True 3-Head TabM
sleeve: live 8b global EV lower-bound + utility agreement
risk: TP 0.052 / SL 0.028 / notional 0.81 / leverage 2 / max_hold 192
```

Next action should be a constrained SL-only sweep, because it directly targets the strongest observed failure mode without introducing unstable multi-head risk behavior.
