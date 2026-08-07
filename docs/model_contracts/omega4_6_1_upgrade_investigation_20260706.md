# Omega4.6.1 — Upgrade Investigation (2026-07-06, DIAGNOSTIC)

Investigation run while the live bot waited for its first Omega4.6.1 entry. Goal: find any
defensible upgrade to the frozen live-wired Omega4.6.1. Method: form each hypothesis from
structure + the VAL 2025-10..12 selection window, then confirm ONCE on the OOS 2026-01..06 window
(one-shot). All numbers use the frozen artifacts + the genuine greedy replay + caps + duration
gate (`replay_omega4_6_1_greedy_router_20260706.py`). Stored-ledger based -> RESEARCH/DIAGNOSTIC
score, not a live-promotion claim (Fresh-Forward Rule).

## Baseline (frozen live model, greedy + duration gate)

| window | PnL | MDD | trades | WR |
|---|---|---|---|---|
| VAL 2025-10..12 | +54.88% | -31.11% | 22 | 0.455 |
| OOS 2026-01..06 | +145.34% | -10.13% | 24 | 0.542 |

## Component × side breakdown (the thing that started this)

The whole realized edge is `zig075 SHORT`; h48qual is net-NEGATIVE in BOTH windows:

| bucket | VAL sum_ret | OOS sum_ret |
|---|---|---|
| h48qual L | -0.010 | -0.011 |
| h48qual S | -0.028 | -0.055 |
| zig075 L  | +0.053 | -0.033 |
| zig075 S  | +0.414 | +1.092 |

This *looks* like "h48qual is dead weight, drop it" — and on VAL that conclusion is very strong.

## Candidate 1: drop h48qual (zig075-only) — REJECTED

| config | VAL (gate) | OOS (gate) |
|---|---|---|
| FULL (h48qual+zig075) | +54.88% / -31.11% | **+145.34% / -10.13%** |
| zig075-ONLY | **+98.96% / -19.73%** | +88.54% / -22.02% |

The two windows **disagree in opposite directions**: zig075-only nearly doubles VAL PnL and
improves VAL MDD, but on the one-shot OOS it *loses* ~57 pts of PnL and doubles MDD. h48qual's
own trades lose money, yet its greedy *priority preemption* of certain zig075 entries improves the
OOS book (changes which zig075 trades happen / their timing). Because the VAL-selected choice is
reversed out-of-sample, this is window-dependent noise, not a structural edge. **Keep FULL.**

## Candidate 2: activate the (empirically inert) exit head — REJECTED

The exit head never fires at its frozen 0.95 threshold; every trade exits via TP/SL. Sweeping the
threshold lower (letting it cut trades early):

| exit_th | VAL PnL/MDD | OOS PnL/MDD |
|---|---|---|
| 0.95 (frozen) | +54.88 / -31.11 | **+145.34 / -10.13** |
| 0.90 | +54.88 / -31.11 | +89.22 / -14.90 |
| 0.80 | +54.88 / -31.11 | +59.62 / -16.23 |
| 0.70 | +38.99 / -31.11 | +21.66 / -25.02 |
| 0.60 | +48.32 / -15.43 | -18.78 / -39.67 |
| 0.50 | -37.78 / -44.61 | -26.93 / -37.54 |

OOS is unambiguous: **0.95 (effectively "no early exit, pure TP/SL barrier") is the best config by
a wide margin**; every lower threshold destroys PnL. The only VAL hint (0.60 improves VAL MDD)
collapses OOS. **Keep 0.95.**

## Candidate 3 (2026-07-07): learned meta-router -- block h48qual LONG signals only — REJECTED

Follow-up research (HF papers + own reasoning) proposed a *next-version* candidate
(`omega4_6_1_learned_meta_router_20260707`, kept fully separate from the live-wired base --
nothing in `trading_bot.py` touched) refining Candidate 1: instead of dropping h48qual entirely,
only gate out its LONG signals (letting zig075/cash take the slot instead), since a finer-grained
counterfactual (running h48qual alone, un-competed, on VAL) showed its LONG side is the clearly
worse half (1/6 win rate, negative EV) vs its SHORT side (10/22 win rate, roughly breakeven).
Implementation: force h48qual's decision frame to CASH at LONG-signal bars, feed the (otherwise
untouched) genuine greedy_replay engine so gated bars fall through to zig075 exactly as the real
router would.

| config | VAL (gate) | OOS (gate) |
|---|---|---|
| BASELINE (h48qual always priority) | +54.88% / -31.11% / n=22 | **+145.34% / -10.13% / n=24** |
| META-ROUTER (block h48qual LONG) | +43.53% / -31.11% / n=22 | +116.75% / -12.54% / n=23 |

**Rejected at the VAL selection stage itself** (43.53% < 54.88%) -- per this project's own
discipline (VAL selects, OOS confirms once), this candidate would never have been promoted to an
OOS check in a real selection process. Confirmed anyway for transparency: also worse on OOS.
Removing h48qual's LONG trades (which do lose money standalone) still makes the *combined* greedy
book worse in both windows -- the single-position-slot dynamics are not decomposable per-trade;
which bar occupies the slot changes the whole subsequent trade sequence in ways that don't match
naive "remove the losing piece" intuition. This is now the THIRD independent routing-lever attempt
(component pruning, exit-threshold, per-side gating) that fails to beat the frozen baseline.

## Candidate 4 (2026-07-07): conformal-style recalibration of L4 risk-sizing — REJECTED (close call)

Diagnostic first (`scripts/diagnose_risk_sidecar_calibration_20260707.py`): does the L4 sidecar's
raw HGB score actually correlate with realized trade outcome? Using each component run ALONE (own
full position slot, not competing) for more samples than the tiny combined-greedy counts:

| component | VAL spearman(score, return) | OOS spearman(score, return) |
|---|---|---|
| h48qual | -0.043 (p=0.83, noise) | **-0.406 (p=0.036, significant and INVERTED)** |
| zig075 | (raw corr +0.297) | +0.171 (p=0.34) |

Genuine finding: h48qual's sizing is backwards in the recent window (sizes UP exactly when it
should size DOWN) -- invisible from VAL alone. This motivated `omega4_6_1_conformal_sizing_20260707`:
replace the frozen `unit = sigmoid(temp * z)` sizing transform with `unit = sigmoid(a + b * z)`,
where (a, b) are fit via regularized logistic regression of z -> win on each component's VAL-window
alone-ledger (Platt/conformal-style recalibration against realized outcomes instead of a generic
historical-quantile rescaling). Everything else (floor/cap/leverage params/side scales/L7
SCALE_MAP/duration gate) stays frozen.

| config | VAL (gate) | OOS (gate) |
|---|---|---|
| BASELINE (frozen sigmoid sizing) | **+54.88%** / -31.11% / n=22 | +145.34% / -10.13% / n=24 |
| CONFORMAL-RECALIBRATED sizing | +52.04% / -32.84% / n=22 | **+160.50%** / -11.27% / n=24 |

**Rejected on VAL** (52.04% < 54.88%, also slightly worse MDD) -- by a small margin, but per the
pre-registered discipline (VAL decides, OOS confirms once, no cherry-picking after seeing OOS) the
decision is fixed at the VAL step. The OOS number (+160.50% vs +145.34%) looks like a genuine
improvement and is tempting to adopt in hindsight, but doing so would be exactly the kind of
post-hoc selection this project's Fresh-Forward Rule exists to prevent -- the "select on VAL"
commitment has to bind even when OOS later looks better, or the whole discipline is theater. Worth
flagging explicitly as the closest of the four candidates tried and a good illustration of why the
rule matters: without it, this is precisely the kind of result that gets rationalized into a false
promotion.

## Candidate 5 (2026-07-07): let-winners-run trailing-stop exit for zig075 — REJECTED (VAL/OOS disagree)

Follow-up research (HuggingFace/arXiv papers on RL exit-policy/optimal-stopping, plus this
project's own history) identified the exit layer, not the routing/sizing layers already tried, as
the remaining untested structural lever. The one demonstrably-working exit technique in this
project is Sigma6's let-winners-run trailing stop (a different, 1h trend-following signal, OOS
cost1 +45.9%) -- never ported onto Omega4.6.1's own static TP(7.5%)/SL(4%) barrier. Implementation
(`scripts/train_eval_omega4_6_1_trailing_exit_20260707.py`, forked `greedy_replay` with only the
exit-barrier block replaced): once a trade's MFE clears `arm_frac * take_profit`, the stop ratchets
up to `mfe - trail_gap` (monotonic, never loosens); below that MFE the original static stop_loss
stays in force; take_profit remains a hard cap throughout. Applied to zig075 only (h48qual keeps
its static barrier -- already net-negative both windows, not expected to benefit). Grid-searched
`arm_frac in {0.3,0.4,0.5,0.6} x trail_gap in {0.01,0.015,0.02,0.03}` on VAL only, froze the winner,
confirmed once on OOS.

| config | VAL (gate) | OOS (gate) |
|---|---|---|
| BASELINE (static TP/SL) | +54.88% / -31.11% / n=22 | **+145.34% / -10.13% / n=24** |
| BEST VAL trailing (arm=0.6, gap=0.010) | **+58.22% / -17.94% / n=23** | +8.28% / -43.69% / n=37 |

VAL narrowly favors the trailing exit (+3.3pp PnL, MDD nearly halved) and the pre-registered
VAL-only rule would say "adopt". But the frozen config collapses on the one-shot OOS check: PnL
falls from the baseline's +145.34% to +8.28%, MDD roughly quadruples (-10.13% -> -43.69%), and
trade count jumps 24 -> 37 -- the trailing stop is cutting the large trend-following runs that
generate zig075 SHORT's entire edge (trades that would have ridden to the 7.5% TP instead get
stopped out early on pullbacks, and the extra stop-outs also create more re-entries). This is the
same VAL/OOS-disagree-in-opposite-directions pattern as Candidate 1 (drop h48qual): the VAL-window
signal is not structural, it is window-dependent noise. **Keep the static TP/SL barrier.**

## Candidate 6 (2026-07-07): learned bar-level exit model (fitted return-to-go / offline-RL) — REJECTED (VAL/OOS disagree)

Prompted directly by "isn't there an RL or deep-learning candidate?". True episode-level RL
(policy-gradient/Q-learning) is not statistically viable here: VAL/OOS each contain only 22-37
completed trades, far too few episodes for an RL agent to separate signal from noise -- and the
exit_head is already a trained deep-learning exit-timing model that was already tested (Candidate
2) and failed at every non-frozen threshold. What IS viable is a NEW model trained on the much
larger population of in-trade BARS (not trade outcomes): `scripts/train_eval_omega4_6_1_learned_exit_dl_20260707.py`
fits a `HistGradientBoostingRegressor` on TRAIN-window (2025-01-01..09-30, strictly before VAL, no
leakage) bar-level state (hold/move/mfe/mae/giveback/notional/leverage/take_profit/stop_loss/side)
for every bar of every zig075 trade under the static barrier, predicting a Monte-Carlo
return-to-go target `rtg = move_at_actual_exit - move_now` (67,103 bars / 73 trades, train
R²=0.926) -- the simplest form of offline RL (fitted value / return-to-go regression). Exit rule:
exit now if predicted rtg <= a threshold (grid-searched on VAL only, frozen, then scored once on
OOS). h48qual keeps its original static barrier; TP remains a hard cap for zig075 too.

The standalone (zig075-only, no h48qual competing) comparison looked strong in BOTH windows (VAL
+142.96% vs zig075-only-static's +98.96%; OOS +108.94% vs zig075-only-static's +88.54%) -- but the
promotion-relevant comparison is the FULL combined router (h48qual+zig075, single account,
priority routing) against the TRUE live baseline, which
`scripts/train_eval_omega4_6_1_learned_exit_dl_combo_20260707.py` checked directly:

| config | VAL (gate) | OOS (gate) |
|---|---|---|
| TRUE baseline (h48qual+zig075, both static) | +54.88% / -31.11% / n=22 | **+145.34% / -10.13% / n=24** |
| COMBO: zig075 uses learned exit (threshold=-0.005, frozen), h48qual unchanged | **+100.73% / -6.89% / n=26** | +29.91% / -10.11% / n=54 |

VAL again favors the candidate on both PnL and MDD, and the pre-registered VAL-only rule would
adopt it -- but the frozen threshold collapses on the one-shot OOS check: PnL falls from +145.34%
to +29.91%, trade count roughly doubles (24 -> 54) and win rate drops from 0.542 to 0.241. The
TRAIN-fit model generalizes well in-sample (R²=0.926) but the resulting exit POLICY is overfit to
VAL-window price dynamics and reverses badly on OOS -- the third candidate this session (after
Candidate 1 and Candidate 5) to show this exact VAL/OOS-disagree-in-opposite-directions signature.
**Keep the static TP/SL barrier; no learned exit model, RL-flavored or otherwise, has beaten it.**

## Candidate 7 (2026-07-07): trailing-trend veto on zig075 SHORT entries — REJECTED (fails on both out-of-sample windows)

Motivated by the Phase 1 robustness audit finding (`omega4_6_1_phase1_robustness_20260707_results.md`)
that 2025-Q3 inverted the zig075-SHORT edge, followed up by
`scripts/diagnose_omega4_6_1_q3_regime_20260707.py`, which found a genuinely cross-window pattern
(not reverse-engineered from the single Q3 anomaly): across ALL FIVE windows examined, zig075
SHORT's sum_ret tracked the window's broad market trend, not the bar-level regime3 tag (the same
regime3 bull_prob~0.72 tag produced take-profits in Q1 but stop-losses in Q3 -- so the bar-level
tag itself is not a reliable discriminator). Q1 (market -45%): SHORT +0.617. Q2 (+36%): +0.205. Q3
(+67%): -0.517. VAL (-28%): +0.414. OOS (-46%): +1.092.

`scripts/train_eval_omega4_6_1_trend_veto_20260707.py` tested a causal trailing-return veto (skip
a zig075 SHORT entry if `close` has risen more than `threshold` over the trailing `lookback_days`)
with a STRICTER three-way protocol than every prior candidate: select `(lookback_days, threshold)`
on TRAIN 2025-01-01..09-30 ONLY (a window even further removed from OOS than VAL), treat VAL
2025-10-01..12-31 as an interim out-of-sample check (not reselected), and confirm once on OOS.

| config | TRAIN (selection) | VAL (interim check) | OOS (one-shot) |
|---|---|---|---|
| baseline (no veto) | +38.55% / -50.71% / n=53 | +54.88% / -31.11% / n=22 | **+145.34% / -10.13% / n=24** |
| best TRAIN veto (lookback=7d, th=0.05) | **+101.06% / -30.60% / n=51** | +42.86% / -31.11% / n=21 | +80.65% / -24.85% / n=26 |

The TRAIN-selected config more than doubles the TRAIN-window PnL and nearly halves TRAIN MDD, but
it makes BOTH out-of-sample windows worse: VAL PnL -12pp (54.88%->42.86%), OOS PnL -65pp
(145.34%->80.65%) with MDD more than doubling (-10.13%->-24.85%). **REJECTED** under the
pre-registered rule (frozen config must not hurt either out-of-sample window).

This is an important, honest result distinct from the six candidates in
`omega4_6_1_upgrade_investigation_20260706.md`: the underlying cross-window correlation (market
trend vs. zig075-SHORT performance) is REAL and directionally sensible (shorting into a rally is
adverse), and this candidate used a stricter selection protocol than every prior one (TRAIN-only
selection, VAL as a genuine held-out check before OOS) -- yet a simple causal implementation of that
correlation still failed to generalize. This reinforces, rather than contradicts, the project's
core finding: with only ~22-53 trades per window, even a well-motivated, multi-window-validated
hypothesis does not reliably translate into an actionable rule. The correlation is a real
description of the historical record, not evidence of a stable, exploitable causal lever at the
trade counts available. **No veto added; keep Omega4.6.1 exactly as wired.**

## Verdict

No reliable upgrade found across seven attempts spanning every structurally-motivated lever tried
(component/side pruning of the greedy router; exit-head activation; risk-sizing recalibration;
learned meta-routing; trailing-stop exit structure; learned bar-level/offline-RL exit model;
trend-based entry veto). The frozen configuration is the best of everything tested under the
pre-registered VAL-select / OOS-confirm discipline (and, for Candidate 7, an even stricter
TRAIN-select / VAL-check / OOS-confirm-once discipline). Six of seven failed clearly; the
conformal-sizing candidate was a close, honest loss on VAL despite a good-looking OOS number. Three
separate candidates (drop-h48qual, trailing-exit, learned-exit-model) independently show the same
VAL/OOS-disagree-in-opposite-directions signature, and Candidate 7 shows that even a
cross-window-consistent, mechanistically sensible hypothesis fails once implemented and tested with
discipline. Together this is strong evidence that (a) Omega4.6.1's static TP/SL barrier + greedy
priority routing is sitting at a genuine local optimum for this architecture/dataset, and (b) the
project's binding constraint is trade-count scarcity itself, not a missing algorithm -- see
`omega4_6_1_improvement_roadmap_20260707.md` Phase 2/3 for the reactive-monitoring-first response to
this constraint. The honest action is to **leave Omega4.6.1 exactly as wired** (both the live base
and this session's non-live research candidates stay frozen/unpromoted). The live cross-validation
(verifying live `decide_entry()` against an independent standalone re-derivation on the first real
Omega4.6.1 entry, 2026-07-07 00:35, side=1/source=zig075/margin_fraction=0.3226/leverage=5.0x/
notional=1.613, matched to ~7-8 significant figures on quality_score/confidence and exactly on the
rounded log values) is CLOSED -- it was the last open verification gap and confirms the
live-injected model is behaviorally identical to the tested standalone adapter.

Scripts: `scripts/replay_omega4_6_1_greedy_val_20260706.py`,
`scripts/test_omega4_6_1_drop_h48qual_20260706.py`,
`scripts/test_omega4_6_1_exit_threshold_20260706.py`,
`scripts/train_eval_omega4_6_1_learned_router_20260707.py`,
`scripts/diagnose_risk_sidecar_calibration_20260707.py`,
`scripts/train_eval_omega4_6_1_conformal_sizing_20260707.py`,
`scripts/train_eval_omega4_6_1_trailing_exit_20260707.py`,
`scripts/train_eval_omega4_6_1_learned_exit_dl_20260707.py`,
`scripts/train_eval_omega4_6_1_learned_exit_dl_combo_20260707.py`,
`scripts/diagnose_omega4_6_1_q3_regime_20260707.py`,
`scripts/train_eval_omega4_6_1_trend_veto_20260707.py`.
