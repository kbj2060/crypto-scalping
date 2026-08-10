# SOL DL/RL Architecture Survey — 2026-08-07

Contract: [sol_dl_rl_architecture_survey_20260807.json](experiments/sol_dl_rl_architecture_survey_20260807.json)
(preflight PASS, dataset sha256 `85b6edb3…`, 141 numeric features, 0 constant).

First application of the BTC/ETH-hardened protocol to SOL: corrected causal triple-barrier
trade-outcome label (12-bar cumret vol basis, TP2.5/SL1.2, horizon 288 — the 2026-08-06 BTC fix),
fresh-forward bar-by-bar replay via `core/causal_futures_backtest.simulate_single_position`,
10bps roundtrip cost, margin 0.30 × leverage 3, N=5 genuinely random seeds
(903174, 42517, 6688211, 15093, 771442), VAL-only selection, single pre-registered OOS read.
Splits: train ≤2025-08-31 (288-bar purge, +288 embargo for DL), VAL 2025-09-01..12-31,
OOS 2026-01-01..03-31. DL/RL deliberately emphasized per user direction.

Market context: VAL and OOS are both severe SOL bear windows (buy&hold −37.6% / −33.4%).

## Stage 0 — Oracle label ceiling (train+VAL only)

`scripts/eval_sol_tripbarrier_oracle_ceiling_20260807.py`

| split | trades | win rate | sum ret | MDD |
|---|---|---|---|---|
| train | 2,824 | 100% (all TP) | +4,633% | −6.3% |
| VAL | 730 | 100% (all TP) | +1,069% | −4.2% |

Label ceiling is real and much larger than BTC's (median TP move 1.72% vs 10bps cost). Label and
backtest are exactly consistent (oracle never hits SL). Label mix: CASH 36% / LONG 31% / SHORT 33%.

## Stage 1 — LightGBM control (cheap gate)

`scripts/train_eval_sol_tripbarrier_lgbm_cheapgate_20260807.py` — 126 flat features (raw
absolute-level columns excluded), 3-class, purged train.

All six pre-registered entry rules VAL-negative; best −6.90% (side_prob_040, 994 trades, WR 35.9%,
MDD −29.5%). VAL accuracy 34.0% (below the 36% CASH base rate). **Fails the gate — no OOS read.**
Same shape as every BTC line: huge oracle ceiling, zero GBDT capture.

## Stage 2 — Deep learning, N=5 seeds each

`scripts/train_eval_sol_deepfeat_candidates_20260807.py` — soft race-conviction target (KL loss),
train-stat standardization, stride-4 train subsampling, early stop on VAL loss.

### 2a. TabM-style flat MLP ensemble (repo TabMEnsembleHead, 8 experts, 128×3)

| rule | seed-mean VAL PnL | pos seeds | mean trades |
|---|---|---|---|
| argmax | −30.4% | 0/5 | 946 |
| side_prob_055 (selected) | **+0.65%** | 3/5 | 63 |

Earns the OOS read (VAL > 0 and > LGBM control). **Single frozen OOS read: seed-mean +4.08%,
4/5 seeds positive** (per-seed: +3.29/+3.03/−2.62/+16.18/+0.50%, 15–66 trades, MDD −4.8..−7.0%).
Passes its pre-registered adoption bar (OOS seed-mean > 0, ≥3/5 positive). Caveats: VAL margin was
paper-thin, seed dispersion is wide (mean is 1-seed-heavy: without seed 15093 it is +1.05%), and
trade counts are small. Research-positive, not promotion-grade.

### 2b. Causal window-48 transformer (d96/l3/dropout 0.25, BTC G2 config)

Best eligible rule (side_prob_045, 72 trades): seed-mean VAL **−4.42%** (3/5 positive).
Rules with positive means had 6.2/0.2 mean trades — under the 15-trade floor.
**Fails the VAL gate — no OOS read.** Sequence depth added nothing over flat rows (consistent
with the BTC deepfeat finding that the embedding, not the head, is the ceiling).

## Stage 3 — Discrete SAC RL, N=5 seeds

`scripts/train_eval_sol_discrete_sac_rl_20260807.py` — un-shaped mark-to-market reward under the
exact survey cost model (deliberately NOT the ETH-tuned SACTradingEnv shaping); state = 126
standardized features + position/uPnL/hold-time; actions {flat,long,short}; twin-Q, auto-entropy;
200k env steps/seed; greedy deterministic replay for evaluation.

Training was stopped by user decision after 2 of the 5 contract seeds completed — both were
catastrophically VAL-negative, and the remaining 3 seeds would have needed a +57% mean VAL to
reach the gate (`tmp/sol_dl_rl_survey_20260807/rl_discrete_sac/val_results_interim_stopped.json`):

| seed | VAL PnL | MDD | entries | bars long/short/flat |
|---|---|---|---|---|
| 903174 | −73.2% | −73.2% | 1,969 | 8.6% / 7.8% / 83.6% |
| 42517 | −98.7% | −98.7% | 5,091 | 44.0% / 27.6% / 28.4% |

Failure mode identical to BTC discrete SAC (project-btc-dsac-advanced-rl-closed-20260802): the
greedy policy churns positions (2k–5k entries over 4 months) and the 10bps roundtrip cost grinds
equity to zero. Even with an un-shaped mark-to-market reward, the agent finds no cost-beating
signal. **FAIL — no OOS read.**

## Verdict

| candidate | VAL (seed-mean) | gate | OOS (seed-mean) |
|---|---|---|---|
| Oracle ceiling | +1,069% (100% TP) | — | not read |
| LightGBM control | −6.90% (best rule) | FAIL | not read |
| **TabM flat ×5** | **+0.65% (3/5 pos)** | **PASS** | **+4.08% (4/5 pos)** |
| Transformer w48 ×5 | −4.42% (best eligible) | FAIL | not read |
| Discrete SAC ×2 (stopped) | −73% / −99% | FAIL | not read |

1. **The BTC/ETH pattern reproduces on SOL**: a large, real oracle ceiling that GBDT, sequence
   depth, and RL all fail to capture. SOL's bigger per-trade barrier distances (median TP 1.72%)
   did not change the ordering.
2. **TabM flat is the survey's only survivor** and the first candidate in the SOL line to pass a
   pre-registered fresh-forward OOS gate: +4.08% seed-mean, 4/5 seeds positive, in a quarter
   where SOL fell 33%. It is *research-positive only*: the VAL margin was +0.65%, the OOS mean is
   one-seed-heavy (+1.05% without seed 15093), and 15–66 trades/seed is thin. It must not touch
   live sizing/routing without (a) a genuinely untouched later window or shadow ledger, and
   (b) the full router/slot/sizing simulation the registry requires.
3. **RL on SOL is closed at the survey level**: unshaped discrete SAC reproduces the BTC failure
   mode (cost-churn death) on a less efficient market — the problem is not BTC-specific.
4. Reusable for future SOL lines: the corrected TB label parquet (registered in the dataset
   manifest), the oracle-ceiling harness, and the 5-seed DL runner.

Next steps (not started): accumulate TabM-flat shadow evidence on live-adjacent data before any
promotion talk; if pursued, the candidate needs its own contract revision with the full sizing
stack, per the workbench registry rules.

## Rev2 — layered entry/exit extension (same day, user-directed)

Contract: [sol_dl_rl_architecture_survey_rev2_layers_20260807.json](experiments/sol_dl_rl_architecture_survey_rev2_layers_20260807.json)
(preflight PASS). Runner: `scripts/train_eval_sol_layered_entry_exit_20260807.py`.
Frozen parent: the 5-seed TabM-flat entry stack at side_prob_055 (VAL +0.65%).

**Layer A — per-side TP-first quality gates** (two binary LightGBM heads, targets
`soft_long>soft_cash` / `soft_short>soft_cash`, purged train). The q=0 control reproduced the
parent VAL exactly (+0.6507, regression gate passed). Every gate level reduced VAL PnL:

| q | seed-mean VAL PnL | pos seeds | mean trades |
|---|---|---|---|
| 0 (control) | +0.65% | 3/5 | 63.4 |
| 0.40 | −1.40% | 2/5 | 32.0 |
| 0.45 | +0.43% | 3/5 | 23.6 |
| 0.50 | +0.29% | 2/5 | 18.2 |
| 0.55 | −0.56% | 2/5 | 14.6 |

**NOT adopted.** The entry model's own side-probability already carries the quality information;
an independent TP-first head only removes trades the stack was profiting from — the same
"quality/threshold layer is the failure point" shape as BTC zig075.

**Layer B — close-ratchet trailing stop** on the un-gated parent (activation × trail-distance
grid, intrabar priority SL → trail → TP; note: trailing replay MDD is trade-granular, not
bar-level — PnL was the selection metric):

| act | dist | seed-mean VAL PnL |
|---|---|---|
| 0.50 | 0.5 | −2.45% |
| 0.50 | 1.0 | −0.19% |
| 0.75 | 0.5 | −0.24% |
| 0.75 | 1.0 | +0.13% |

**NOT adopted** (best +0.13% < +0.65% baseline). The positive short-hold-TB prior from the
original G1 result did not transfer to SOL: even with late activation and a wide trail, cutting
the barrier race early gives up more TP completions than it saves in SL hits.

**Rev2 verdict: layering axis CLOSED per the pre-registered rule — no OOS read taken, the
parent TabM-flat stack remains the frozen best.** Entry/exit improvement on this candidate now
requires a structurally different lever (e.g. maker-entry fill audit at short-hold cadence, or a
retrained entry model with quality trained jointly rather than as a post-hoc gate), each needing
its own contract.

## Rev3 — joint multi-task quality TabM (same day, user-directed)

Contract: [sol_dl_rl_architecture_survey_rev3_joint_quality_20260807.json](experiments/sol_dl_rl_architecture_survey_rev3_joint_quality_20260807.json)
(preflight PASS). Runner: `scripts/train_eval_sol_joint_quality_tabm_20260807.py`.
Shared 8-expert BatchEnsemble trunk, direction KL head + per-side TP-first BCE head,
loss = KL + λ·BCE; λ ∈ {0.25, 1.0} × entry rules {side_prob_055, +own-quality≥0.45, ≥0.50},
5 contract seeds, closed 6-config grid.

| λ | rule | seed-mean VAL PnL | pos seeds | mean trades |
|---|---|---|---|---|
| 0.25 | side_prob_055 | −1.97% | 1/5 | 58.8 |
| 0.25 | +q≥0.45 | −2.65% | 1/5 | 37.4 |
| 0.25 | +q≥0.50 | −2.12% | 1/5 | 22.6 |
| 1.00 | side_prob_055 | −2.74% | 3/5 | 53.6 |
| 1.00 | +q≥0.45 | −3.62% | 2/5 | 42.6 |
| 1.00 | +q≥0.50 | −2.34% | 2/5 | 30.2 |

**All six configs VAL-negative vs the single-task parent's +0.65% under identical everything
else — the auxiliary quality objective actively degrades the shared representation, at both λ
levels, with and without using the quality head at inference. Joint-quality axis CLOSED, no OOS
read.** Combined with rev2 this closes the quality signal on SOL in both forms (post-hoc gate
AND joint training): the TP-first information the label carries beyond the 3-class soft target
is not additive for this stack.

## Untouched-window confirmation read — candidate CLOSED

`scripts/eval_sol_tabm_flat_untouched_ext_20260807.py` — everything frozen, window
2026-04-01..2026-07-21 (never used for any training/selection/gate in this line), pass rule
pre-registered in the script before the read (same bar as the parent OOS adoption: seed-mean > 0,
≥3/5 seeds positive, ≥15 mean trades).

| seed | PnL | trades | WR | MDD |
|---|---|---|---|---|
| 903174 | −1.80% | 33 | 33.3% | −9.1% |
| 42517 | −18.62% | 83 | 27.7% | −21.3% |
| 6688211 | −12.70% | 76 | 26.3% | −16.9% |
| 15093 | −11.66% | 95 | 30.5% | −20.5% |
| 771442 | +4.45% | 30 | 43.3% | −2.9% |

Seed-mean **−8.07%**, 1/5 seeds positive (buy&hold reference −5.75%). **FAIL — the 2026Q1 +4.08%
was a window artifact.** Notably the 2026Q1 hero seed (15093, +16.2%) is −11.7% here, and the
long/short mix flipped long-heavy into a falling window. This is the project's 6th
selected-positive-flips-on-fresh-data reproduction and the first on SOL.

## Rev4 — cost-aware magnitude-filtered return regression (literature-driven, user-directed)

Contract: [sol_dl_rl_architecture_survey_rev4_costaware_reg_20260807.json](experiments/sol_dl_rl_architecture_survey_rev4_costaware_reg_20260807.json)
(preflight PASS). Runner: `scripts/train_eval_sol_costaware_regression_20260807.py`.
Design ported from arXiv:2606.00060 (2026, hourly BTC, 27-fold walk-forward): next-hour
open-to-open return REGRESSION at 1h decision cadence, position changed only when
|forecast| > k × 10bps (weak signals hold the current position), 5bps/side on changes.
Closed grid k ∈ {1,2,3} × {long_short, long_only}; LightGBM regression (deterministic) and
TabM regression (5 contract seeds).

VAL: the filter mechanism behaved exactly as the paper describes — k=1 configs churn and lose
(TabM −35.6%), k=3 configs are strongly positive (LGBM +15.7%/304 entries; TabM seed-mean
+24.1%, 3/5 seeds, 65 entries). Both families earned their OOS read.

OOS (single frozen read each):

| family | VAL | OOS | verdict |
|---|---|---|---|
| LGBM reg k3 long_short | +15.7% | **−15.2%** (MDD −37%) | FAIL |
| TabM reg k3 long_short ×5 | +24.1% | **−7.0%** seed-mean, 2/5 pos | FAIL |

**Cost-aware-regression axis CLOSED.** The execution filter genuinely cuts turnover, but the
hourly forecast underneath has no stable SOL signal — the VAL profit was window-specific. This is
the project's 7th selected-positive-flips-on-fresh-data reproduction and the 2nd on SOL today.

## Oracle-logic audit (2026-08-08, user challenge: "is the oracle broken? only-ETH-has-edge makes no sense")

`scripts/audit_sol_oracle_logic_controls_20260808.py` — two controls:

1. **ETH oracle control**: the IDENTICAL TB label + oracle replay on ETH raw 5m OHLC, same VAL
   window → 741 trades, 100% WR, +892% sum return (7,005× equity). The enormous ceiling is a
   property of perfect TP/SL-race foresight on ANY liquid asset — ETH's included — and ETH's
   working live edge does NOT come from capturing this 5m oracle (it comes from the multi-day
   swing architecture). The oracle number is a label-consistency check, not a measure of
   available edge; "huge ceiling + zero 5m capture" is the normal state on every asset here.
2. **Leak-injection positive control**: one deliberately leaked feature (label race-score
   difference) added to the unchanged SOL LGBM pipeline → VAL accuracy 98.9%, WR 99.7%,
   VAL +3.94M%. The harness demonstrably converts real signal into PnL when signal exists —
   capture-zero is a property of the data, not a bug in the training/replay machinery.

Contextual correction to "only ETH has edge": SOL does have a live edge (adaptive_squeeze v2,
rule-based). What SOL lacks is an ML-derived 5m entry edge from this panel — and ETH's ML edge
is a one-time frozen survivor (only val+OOS pass in project history, not reproducible by
retraining, not portable to BTC or SOL), which is consistent with edges being rare and fragile,
not with SOL results being artifacts.

## Post-hoc oracle feature analysis (2026-08-08) — the mechanism behind every failure

`scripts/audit_sol_oracle_feature_analysis_20260808.py` — univariate Mann-Whitney AUC of each of
the 126 features against the oracle label (direction: LONG vs SHORT; tradeability: trade vs
CASH), computed separately on train, VAL, and three train sub-windows; then an honest top-20
oracle-filtered LGBM retrain (selection on train only).

Findings:
- Single-feature direction information is tiny everywhere: max |AUC−0.5| = 0.038 train / 0.058 VAL.
- **Feature-AUC Spearman between train and VAL: −0.38 (NEGATIVE).**
- **Top-20 train direction features: 0/20 keep their sign on VAL** — every one of the strongest
  train relationships INVERTS out of window. Only 47/126 features hold a stable sign even across
  the three train sub-windows, and none of the top-20 is both train-stable and VAL-consistent.
- Tradeability side is equally empty (max AUC 0.517 train, collapses on VAL).
- The oracle-filtered top-20 LGBM performs far WORSE than the full-feature control
  (VAL −31.9…−54.2% vs −6.9%): concentrating the model on the most oracle-informative train
  features maximally concentrates the sign-inversion damage.

**Interpretation: the panel's failure mode is not weak signal but ANTI-STABLE signal — the
feature→direction map systematically flips sign between regimes.** This mechanically explains
all 9 closed families and the 7 fresh-data flips (any model fit on one window learns relations
that are inverted in the next), and it explains why the one architecture that ever worked in
this project is ETH's regime-routed MoE (bull/bear/chop experts): regime-conditioning is
precisely a device for letting the feature→direction sign change with the regime. A
regime-conditioned entry stack was NOT among the 9 SOL families tested; it is the one
mechanistically-motivated axis this analysis leaves open (while noting SOL HMM regime models
were fresh-forward-rejected 2026-07-21 — any retry must fix regime detection first).

## Rev6 — regime-conditioned MoE entry stack (2026-08-08, user-directed, mechanism-motivated)

Contract: [sol_dl_rl_architecture_survey_rev6_regime_moe_20260808.json](experiments/sol_dl_rl_architecture_survey_rev6_regime_moe_20260808.json)
(preflight PASS). Runner: `scripts/train_eval_sol_regime_moe_20260808.py`.

**Stage R (regime detection, training-free) PASSED its gate** — and confirmed the sign-inversion
mechanism directly. Detector selected on train-only sub-window stability: D2 = 288-bar close
return with ±4% threshold (train stability 0.717 vs 0.60/0.58 for the alternatives).
Within-regime top-20 feature sign agreement train→VAL: **bull 85%, bear 60%, chop 35%** vs the
unconditional baseline of **0%**. Trend-regime conditioning genuinely stabilizes SOL's
feature→direction map; chop is directionless as expected.

**Stage M (per-regime LightGBM experts) FAILED decisively**: every threshold VAL-negative
(−29.5% best) — worse than the unconditional control (−6.90%). Sign STABILITY is not sign
MAGNITUDE: the within-regime AUCs are only ~0.53–0.55, far below what 10bps costs demand, and
the dominant chop regime (91k of 131k train rows) has no direction signal at all, so the chop
expert bleeds continuously. **Rev6 closed at Stage M per the pre-registered rule — no OOS read.**

The surviving scientific result is Stage R itself: the anti-stability diagnosis is now
mechanistically confirmed AND partially repaired (conditioning restores sign coherence), but the
per-regime signal magnitude on this panel remains an order short of costs. That is the panel's
final, fully-characterized verdict.

## Rev7 — maker-first execution (2026-08-08, user-directed; the cost-bar lever)

Contract: [sol_dl_rl_architecture_survey_rev7_maker_exec_20260808.json](experiments/sol_dl_rl_architecture_survey_rev7_maker_exec_20260808.json)
(preflight PASS). Runner: `scripts/train_eval_sol_maker_execution_20260808.py`; new data:
SOLUSDT 1m klines 2025-08-25..2026-05-01 (`scripts/download_klines_sol_1m_20260808.py`).
Frozen LGBM control, post-only limit at decision-bar close, conservative Kappa1 fill model
(cross-at-open = no fill, strict 1m trade-through, 15-min cancel), 7bps roundtrip vs 10bps taker.

| rule | fills | fill rate | maker VAL PnL | taker VAL PnL | unfilled taker-counterfactual |
|---|---|---|---|---|---|
| argmax | 999 | 74.9% | −19.8% | −16.9% | −15.3% |
| side_prob_040 | 946 | 74.7% | −28.0% | **−6.9%** | **+20.7%** |
| side_prob_045 | 855 | 75.9% | −21.7% | −9.4% | +3.4% |
| side_prob_050 | 738 | 75.3% | −18.3% | −13.1% | −25.9% |
| side_prob_055 | 580 | 75.4% | −20.8% | −25.0% | −35.6% |
| side_prob_060 | 434 | 73.7% | −23.7% | −27.3% | −7.8% |

**All rules negative and mostly WORSE than taker — line closed at VAL, no OOS read.** The 3bps
cost saving and limit-price improvement are overwhelmed by trade-level adverse selection: an
at-touch limit only fills when price first moves AGAINST the signal, so the fill-conditioned
subset over-samples whipsaws, and at the control's own selected rule (side_prob_040) the missed
26% of entries were the profitable ones (+20.7% counterfactual) — the ETH failure shape
reappearing at high frequency. The BTC Kappa1 fill-audit result (~−1bp adverse selection) does
not transfer: that audit measured CONTRARIAN entries in liquidation windows, where being on the
passive side is structurally favorable; this line's entries are momentum-conditioned, where the
passive side is structurally toxic. Cost-bar lever on this signal family: dead.

## Rev8 — per-regime feature distributions + oracle-filtered experts (2026-08-08, user-directed)

Contract: [sol_dl_rl_architecture_survey_rev8_regime_feature_filter_20260808.json](experiments/sol_dl_rl_architecture_survey_rev8_regime_feature_filter_20260808.json)
(preflight PASS). Runner: `scripts/train_eval_sol_regime_feature_filter_20260808.py`.

**Analysis findings (train-only selection, VAL for sign check; charts in
`tmp/sol_dl_rl_survey_20260807/regime_feature_filter_rev8/`):**
- Regime-separated features: trend/positional features shift almost completely across D2 regimes
  (KS ≈ 0.63–0.93 for dual_momentum, vwap distances, volume_profile_signal, turtle_signal,
  cvd_288). Regime-invariant tail: session/time features (KS < 0.05).
- **Bull and bear regimes use nearly DISJOINT information**: Spearman of per-feature direction
  AUC bull-vs-bear = 0.14, top-10 overlap 2/10. Bull's best: whale_retail_ratio (0.565 train →
  0.665 VAL, strengthens). Bear's best: cvd_288 (0.567 → 0.581, holds). Chop: nothing survives
  (3/10 signs hold, all VAL AUCs ≈ 0.5).
- Per-regime sign stability of top-10: bull 7/10, bear 7/10, chop 3/10 — consistent with rev6.

**Experiment (24 pre-registered configs: K∈{10,20} × chop∈{expert, force_cash} × 6 rules):**
chop-expert configs all deeply negative (−10…−51%); **force-cash-in-chop flips VAL positive in
10 of 12 configs** (up to +24.0%). The monthly-stability screen rejected the flashiest configs
(2/4 months) and selected k10/force_cash/side_prob_050: **VAL +9.89%**, 304 trades, 3/4 months,
MDD −18.6% — the strongest VAL of the entire survey and the first regime-family config to earn
an OOS read.

**Single frozen OOS read: −31.2%** (256 trades, WR 32.4%, MDD −37.2%, long-tilted 159/97 in a
falling quarter). **FAIL — the survey's 8th selected-positive-flips-on-fresh-data reproduction.
Rev8 closed; the {regime-conditioning × feature-filtering} design matrix is now fully tested and
every cell is dead.** The within-regime feature structure is real (the analysis stands) but
per-regime models still fit window-specific magnitudes that do not transfer one quarter forward.

## FINAL verdict (supersedes the interim verdict above)

**Zero surviving candidates.** Every architecture family tested against the corrected TB label on
SOL — LightGBM, flat TabM ensemble, windowed transformer, unshaped discrete SAC, post-hoc quality
gates, joint-quality multi-task, trailing exits — is either VAL-negative or fails an untouched
window. The huge oracle ceiling (+1,069% VAL under costs) remains 0% captured, exactly as on BTC.
SOL live remains adaptive_squeeze v2 (execution-disable recommendation from the sizing memo
unchanged). The reusable assets from this line are the label parquet, the oracle harness, the
5-seed DL runner, and one negative structural fact: SOL's inefficiency (bigger barriers vs costs)
does not translate into extractable 5m entry edge for any tested model family.

Rev4 (2026 literature port: cost-aware magnitude-filtered hourly regression) also closed —
VAL +15.7%/+24.1% flipped to OOS −15.2%/−7.0%. Remaining genuinely-unexplored axes for SOL are
new raw information sources (news/sentiment was never acquired for this project) and execution
primitives (maker-fill audit at short-hold cadence); more modelling on the existing panel is not
recommended.

## Rev5 — volatility-breakout bracket primitive (2026-08-08, user-directed "creative" axis)

Contract: [sol_dl_rl_architecture_survey_rev5_bracket_20260808.json](experiments/sol_dl_rl_architecture_survey_rev5_bracket_20260808.json)
(preflight PASS). Runner: `scripts/train_eval_sol_bracket_breakout_20260808.py`.
Idea: stop-entries at ±b·σ around next open let the MARKET choose the side (deferring the
direction commitment that killed all seven prior axes); the model only gates "will this breakout
continue" — the mechanism the live SOL adaptive_squeeze strategy monetizes at coarser cadence.

Stage 0 (mechanical, no model): raw ungated bracket is a heavy loser everywhere (train −99%,
VAL −55…−79%); TP-rate among triggered setups ≈ 32.2–32.8% vs ≈ 32.4% gross breakeven — the
primitive sits exactly at the knife's edge, and the oracle ceiling is enormous. The gate needed
only a few points of TP-rate lift on selected setups.

Stage 1 (per-(b,W) LightGBM continuation gates, 16 pre-registered configs): the gate ranks
setups monotonically (e.g. b0.75/W6: −33.7% → −6.4% as the threshold rises) but **no config
crosses zero on VAL** (best −1.43%, 0 of 16 pass the PnL>0 + ≥3/4-months screen).
**Bracket axis CLOSED at VAL — no OOS read.** Even with the side-commitment problem removed,
the panel cannot lift breakout continuation enough to clear costs.
