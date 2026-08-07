# Alpha1 Layer Improvement Review - 2026-05-13

## Baseline

Current reference:

- `alpha1`: parent `hf_v13_clean_regime_margin110_20260511` frozen
- V21.2 jackpot add-on frozen
- V27 deep scout frozen
- V31 rule exit frozen
- deep scout notional = `2.0`

Metrics:

| Model | Cost1 PnL | Cost1 MDD | Cost2 PnL | Cost3 PnL |
|---|---:|---:|---:|---:|
| alpha1 | +361.19% | -31.74% | +88.74% | +0.58% |
| alpha1.4 soft execution proxy | +385.98% | -31.68% | +94.35% | +10.35% |

## Layer Findings

### 1. Parent Layer

Verdict: do not replace now.

Evidence:

- Teacher-constrained deep parent:
  - seed mean cost1: +369.37%
  - cost2 mean: +73.19%
  - cost3 mean: -3.28%
  - beat alpha1 cost1 in 2/3 seeds, but never beat alpha1 cost2/cost3
- Distilled LightGBM/CatBoost parent:
  - best cost1: +218.08%
  - cost2: +42.85%
  - cost3: -19.00%

Interpretation:

The parent is part of the alpha1 timing ecosystem. Replacing it changes active/CASH rhythm and damages V27 scout opportunities or V21.2 lifecycle behavior.

Action:

- Freeze parent.
- Only revisit parent after building a month-by-month selection process that explicitly preserves alpha1 CASH rhythm.

### 2. V27 Deep Scout Entry Layer

Verdict: keep frozen.

Evidence:

- Previous attempts to modify deep entry readout/attention/exit coupling damaged V27 edge.
- Dynamic scout gate on V31 showed no improvement over baseline.
- Deep scout notional increase to 2.0 is already the main alpha1 improvement source.

Action:

- Keep V27 entry model frozen.
- Do not retrain entry head until a clean validation regime is fixed.
- Potential future change: add only an external pre-trade veto, not retrain the TCN itself.

### 3. V31 Exit Layer

Verdict: high risk to modify directly.

Evidence:

- RL exit overlay:
  - cost1 +199.74%, cost2 +22.28%, cost3 -43.44%
- CMA-style exit constants on V31:
  - cost1 +222.76%, cost2 +100.94%, cost3 +18.02
- Most exit changes reduce alpha convexity.

Interpretation:

V31 exit is crude but preserves tail winners. Learned early exits cut too much upside.

Action:

- Do not replace V31 exit.
- Safer future experiment: add a narrow emergency-only adverse-flow kill switch, not a general close/hold policy.

### 4. V21.2 Jackpot Add-on Layer

Verdict: do not replace with broad smart add-on.

Evidence:

- Smart microstructure add-on on V31:
  - cost1 +145.13%, cost2 +86.98%, cost3 -8.10
- V25 lifecycle RL add-on:
  - cost1 +199.53%, cost2 +113.26%, cost3 +24.80
  - more cost durable, but much lower cost1 than alpha1

Interpretation:

V21.2 add-on is already selective. Extra filters can reduce damage but also remove the high-convexity parent wins.

Action:

- Keep V21.2 unchanged for alpha1.4.
- Possible future candidate: add a limited profit-state pyramid from V56, but only after combining with alpha1.4 execution proxy and selecting on cost1/cost2/cost3.

### 5. Sizing / Notional Layer

Verdict: do not use learned sizing yet.

Evidence:

- Alpha1 RL sizing:
  - cost1 +336.87%, cost2 +48.72%, cost3 -51.95
- V54 notional reallocator:
  - cost1 +186.26%, cost2 +46.28%, cost3 -34.55

Interpretation:

Learned sizing over-increases exposure in fragile conditions and breaks cost3 survival.

Action:

- Keep deep scout notional fixed at 2.0 for alpha1.
- Any future sizing change must be cost3-aware and monotonic-constrained.

### 6. Execution Layer

Verdict: best current improvement target.

Evidence:

- V31 execution sniper:
  - cost1 +291.42 vs V31 +277.07
  - cost2 +120.79 vs +112.79
  - cost3 +38.51 vs +20.93
- Alpha1.4 soft execution proxy:
  - cost1 +385.98 vs alpha1 +361.19
  - cost2 +94.35 vs +88.74
  - cost3 +10.35 vs +0.58

Interpretation:

Execution is the only layer that improved alpha1 without damaging the parent/V27/V31 ecosystem.

Action:

- Promote `alpha1.4` as research-best surrounding-layer candidate.
- Before live injection, convert the OHLCV proxy into real post-only/L2 order routing:
  - maker order only when flow is favorable
  - cancel after 1 bar or when price moves away
  - fallback to taker only for stop/loss emergency
  - log route, missed fill, maker fill, fallback fill

## Recommended Next Experiment

`alpha1.5 = alpha1.4 soft execution proxy + narrow profit-state pyramid`

Scope:

- Keep parent frozen.
- Keep V27 frozen.
- Keep V31 exit frozen.
- Keep V21.2 unchanged.
- Add a tiny pyramid only when:
  - position is already profitable
  - MFE is positive and growing
  - alpha1.4 execution route would be maker/proxy favorable
  - drawdown state is below threshold
  - opposite V27 utility is not rising

Reason:

V56 profit-state pyramid improved cost2/cost3 on V31-style stacks but reduced cost1 relative to alpha1. It may become useful only when paired with alpha1.4's better execution costs.

## Current Decision

- Promote for research: `alpha1.4 soft execution proxy`
- Do not promote: parent replacements, learned sizing, broad RL exit, smart add-on replacement
- Next coding target: constrained `alpha1.5` profit-state pyramid on top of alpha1.4, not a core model replacement
