# ETH Omega4.6.1 Exit Learning Research — 2026-07-24

Status: `research_only_not_live_promoted`

No live adapter, runtime configuration, parent bundle, risk sidecar, or environment setting was
changed.

## Question

Does training the exit layer on direction candidates before the quality gate fix the apparent
exit-data shortage, and can a different exit-learning formulation outperform the effectively
pure-SLTP live baseline?

## Existing exit-label diagnosis

The frozen h48qual and zig075 parents do not actually train their selected exit head only on
quality-passed predictions. Both use the same `entry_label_terminal_giveback` ZigZag-segment
dataset:

- 30,000 bar rows
- 732 independent segments
- 2,182 positive rows (7.27%)
- 2,179 `terminal_window_exit` rows
- only 3 `mfe_giveback_exit` rows

The main weakness is therefore label semantics and independent trajectory diversity rather than
the headline bar-row count. At runtime, `EXIT_THRESHOLD=0.95` is empirically inert and TP/SL owns
the observed exits.

## Stage 1 — Competing-risk rescue hazard

Script: `scripts/research_eth_omega461_competing_risk_rescue_20260724.py`

Target classes:

- TP first within 12 / 48 / 384 bars
- SL first within 12 / 48 / 384 bars
- right-censored at 384 bars

The model is a three-member episode-bootstrap `HistGradientBoostingClassifier` ensemble. It exits
only when every member gives sufficient SL probability and the most optimistic estimated
continuation value is below immediate liquidation by a fixed margin. Otherwise it abstains and
keeps the frozen SLTP lifecycle.

### Data expansion

| component | post-quality episodes | pre-quality episodes | change |
|---|---:|---:|---:|
| h48qual | 146 | 237 | +62.3% |
| zig075 | 182 | 242 | +33.0% |

The pre-quality feed did add independent adverse trajectories. It did not produce the best policy:

- unweighted pre-quality over-fired (best VAL run: 67 rescue exits, 86 total trades)
- quality-weighted pre-quality passed some VAL gates but remained below post-quality
- the selected model was post-quality, not pre-quality

This rejects the hypothesis that feeding all pre-quality candidates is by itself the fix.

### Fresh-forward result

Selection used VAL only. OOS was opened once for the selected configuration.

| policy | split | PnL | MDD | trades | log-risk utility |
|---|---|---:|---:|---:|---:|
| pure SLTP baseline | VAL | +42.61% | -27.01% | 29 | -0.9020 |
| competing-risk rescue | VAL | +94.03% | -21.10% | 39 | -0.1846 |
| pure SLTP baseline | OOS | +88.05% | -23.14% | 22 | -0.0620 |
| competing-risk rescue | OOS | +97.94% | -23.14% | 27 | -0.00006 |

Selected contract:

- training universe: post-quality episodes
- minimum ensemble SL probability: 0.60
- continuation-value margin: 0.0025 raw price move
- persistence: 1 bar

OOS produced seven rescue exits. Their frozen-SLTP counterfactual causes were three stop losses and
four 384-bar censors; none was a future TP under the bounded diagnostic horizon.

This is promising research evidence, not promotion evidence. The run used the frozen close-based
replay barrier convention and omitted the empirically inert 0.95 exit-head inference. Current live
intrabar high/low barrier parity, seed stability, additional untouched forward data, and the Omega
artifact-integrity promotion gate remain required.

## Stage 2 — Distributional optimal stopping

Script: `scripts/research_eth_omega461_distributional_stopping_20260724.py`

The model predicts q10/q50/q90 of terminal price move under frozen SLTP continuation. The policy
may exit only when immediate liquidation exceeds q90 continuation value by a margin.

Best VAL result:

| PnL | MDD | trades | log-risk utility |
|---:|---:|---:|---:|
| +70.47% | -21.67% | 29 | -0.6826 |

This beats pure SLTP but does not beat the Stage-1 competing-risk winner, so it was rejected and
OOS was not opened. Internal q90 empirical coverage was 83.8% for h48qual and 89.7% for zig075;
h48qual was not conservative enough to serve as a reliable upper continuation bound.

## Stage 3 — Offline RL gate

Verdict: `DO_NOT_START_YET`

Reasons:

1. The largest available training universe has only 237 h48qual and 242 zig075 independent
   episodes. Tens of thousands of correlated in-position bars are not tens of thousands of RL
   episodes.
2. Pre-quality expansion increased coverage but degraded policy selectivity.
3. The simpler distributional value model failed to beat the hazard policy. A higher-capacity
   CQL/IQL policy would add extrapolation and selection variance before the value target is stable.
4. The same OOS window has now been touched for the Stage-1 winner and cannot be reused for further
   threshold iteration.

Reopen offline RL only after all of the following are true:

- at least 1,000 non-overlapping candidate episodes across the two components
- adequate support in each component × side × broad-regime bucket
- new untouched forward evaluation data
- the Stage-1 hazard result is stable across seeds and rolling training windows
- an explicit HOLD/EXIT behavior-support and off-policy-evaluation audit is defined

Until then, the competing-risk rescue model is the sole research winner and must remain shadow-only.

## Current-live composition audit

The Stage-1 winner was subsequently composed with the actual enabled ETH sizing order rather
than compared with its original research baseline:

1. duration gate off
2. base sidecar/router sizing
3. ETH notional multiplier 1.5
4. portfolio asset cap 1.5 (`total=3.0`, ETH share 0.5)
5. chop soft sizing with threshold 0.3
6. frozen SLTP plus the fixed Stage-1 hazard rescue

Script: `scripts/audit_eth_omega461_live_chop_hazard_composition_20260724.py`

| policy | split | PnL | close-MTM MDD | realized MDD | trades |
|---|---|---:|---:|---:|---:|
| current-live sizing + SLTP | VAL | +46.83% | -18.87% | -14.82% | 29 |
| current-live sizing + SLTP + hazard | VAL | +93.96% | -20.35% | -18.20% | 38 |
| current-live sizing + SLTP | OOS diagnostic | +53.21% | -18.18% | -17.37% | 22 |
| current-live sizing + SLTP + hazard | OOS diagnostic | +27.11% | -20.09% | -19.74% | 27 |

Verdict: `DO_NOT_APPLY_TO_LIVE`. The fixed hazard fails the validation MDD requirement despite
higher validation PnL, and the already-touched OOS diagnostic is worse on both PnL and MDD. Early
rescue exits free the single position slot and permit extra re-entries; this changes the trade
sequence and raises subsequent exposure to losing trades. On OOS the policy removed three future
SL paths but generated five exits whose 384-bar counterfactual remained censored, while total
trades rose from 22 to 27. The hazard must therefore be trained and selected as part of the full
live lifecycle, including post-exit re-entry/re-arm behavior, rather than attached as an isolated
exit classifier.

The OOS figures in this section are diagnostic only because Stage 1 already opened the same
2026-01-01..2026-03-31 window. A newly selected lifecycle policy requires new untouched forward
data before promotion.

## Causal re-arm follow-up

Script: `scripts/research_eth_omega461_live_hazard_rearm_20260724.py`

The hazard model and live contract were frozen. Selection changed only the causal post-rescue
re-arm rule. VAL compared fixed all-entry cooldowns of 12/48/96/192/384 bars and a signal-reset
rule, optionally combined with the same minimum cooldown. OOS was not used for ranking.

VAL selected a 96-bar (8-hour) cooldown:

| policy | PnL | close-MTM MDD | realized MDD | trades |
|---|---:|---:|---:|---:|
| current-live sizing + SLTP | +46.83% | -18.87% | -14.82% | 29 |
| hazard, no re-arm | +93.96% | -20.35% | -18.20% | 38 |
| hazard + 96-bar cooldown | +69.80% | -17.51% | -13.31% | 33 |

The frozen OOS diagnostic did not confirm the full acceptance criterion:

| policy | PnL | close-MTM MDD | realized MDD | trades |
|---|---:|---:|---:|---:|
| current-live sizing + SLTP | +53.21% | -18.18% | -17.37% | 22 |
| hazard + 96-bar cooldown | +37.78% | -16.93% | -16.25% | 27 |

The re-arm rule recovers the MDD objective but retains only 71.0% of baseline OOS PnL, below the
predefined 90% floor. Verdict: `DO_NOT_APPLY_TO_LIVE`. No alternative cooldown may be selected
from this OOS readout. Further lifecycle selection requires a new untouched forward window; the
present evidence supports the mechanism but not deployment.

## Frozen forward-extension audit

Script: `scripts/audit_eth_omega461_hazard_rearm_forward_extension_20260724.py`

The VAL-selected hazard + 96-bar cooldown was frozen without retraining or parameter changes and
run once on 2026-04-01..2026-07-12. This interval was not used to choose the hazard or re-arm
rule. It is not a fully untouched live holdout because the underlying chop-sizing baseline had
previously been researched on overlapping 2026 data.

| policy | Cost | PnL | close-MTM MDD | realized MDD | trades |
|---|---:|---:|---:|---:|---:|
| current-live sizing + SLTP | 1x | +25.58% | -12.69% | -11.84% | 14 |
| frozen hazard + 96-bar cooldown | 1x | -8.89% | -20.42% | -19.41% | 22 |
| current-live sizing + SLTP | 2x | +25.29% | -12.73% | -11.89% | 14 |
| frozen hazard + 96-bar cooldown | 2x | -15.84% | -20.14% | -19.14% | 21 |
| current-live sizing + SLTP | 3x | +25.41% | -12.76% | -11.93% | 14 |
| frozen hazard + 96-bar cooldown | 3x | -15.87% | -20.12% | -19.13% | 21 |

At Cost1, ten rescue exits contributed -14.00 percentage points of additive trade return; nine
were losing exits and one was a small winner. The policy also changed the subsequent entry
sequence. April remained slightly positive, May positive, but June fell to -14.31% versus the
baseline's +7.94%. The extension therefore rejects both the fixed hazard and the proposed re-arm
mechanism as a deployable combination.

Final verdict for this research branch: `RETIRED_DO_NOT_SHADOW_FOR_PROMOTION`. Do not search a new
threshold or cooldown on any interval already reported above. A successor must be trained as a
joint lifecycle policy whose target includes post-exit opportunity cost, then frozen before a new
forward collection starts. The minimum reconsideration gate is 30 independent proposed-exit
events and at least 90 calendar days after the freeze date; until then the current live SLTP +
chop-sizing path remains unchanged.

## Re-entry-aware censored stopping-value successor

Script: `scripts/research_eth_omega461_censored_stopping_value_20260724.py`

This research-only successor uses only entries actually selected by the frozen live router. For
each open-position state it compares `EXIT now + frozen-router re-entry until the original exit`
against `HOLD under frozen SLTP`, with both paths reconverging at the original SLTP exit. Four
landmark competing-risk heads estimate TP-first, SL-first, or neither at 12/48/96/384 bars. A
three-model episode bootstrap and one-sided temporal conformal residual make uncertain decisions
abstain to SLTP.

The exact live barrier order is used here: intrabar stop-loss touch first, then intrabar
take-profit touch, then the learned stopping rule, with execution on the next bar. Earlier sections
used the older close-only research harness, so their baseline numbers are not directly comparable.

Training produced 65,982 state rows but only 85 independent live-router positions (50 SL and 35
TP; 59 `zig075`, 26 `h48qual`). The temporal calibration split had advantage RMSE 0.0598 and a
90th-percentile overprediction error of 0.0782 log-value. On validation, the 80% conformal
advantage lower bound was positive on only 12 of 21,948 held-position states; none of those states
also cleared the conservative 96-bar SL-probability threshold of 0.50. Consequently every one of
the 16 predeclared safe gates made zero model exits and reproduced the SLTP baseline exactly:

| policy | PnL | close-MTM MDD | realized MDD | trades | learned exits |
|---|---:|---:|---:|---:|---:|
| exact-live intrabar SLTP baseline | +18.10% | -19.81% | -14.40% | 31 | 0 |
| safest stopping-value grid (all tied) | +18.10% | -19.81% | -14.40% | 31 | 0 |

Removing the competing-risk requirement as an unsafe diagnostic caused one validation exit and
reduced PnL to +8.38% while worsening realized MDD to -16.33%; it still made no exit on the
already-consumed OOS or extension windows. This confirms that lowering the gate is not a rescue.

Verdict: `DEVELOPMENT_REJECTED_DO_NOT_APPLY_TO_LIVE`. The method abstains correctly but has no
demonstrated incremental value at the current sample size. Do not weaken its uncertainty gate or
retune on the consumed windows. The next useful step is prospective collection of independent
live-router position episodes and proposed-exit states; reconsider only after the previously
declared 90-day / 30-proposed-exit gate, with a stronger target of at least 300 independent
positions before refitting and 1,000 before any offline-RL lifecycle experiment.
