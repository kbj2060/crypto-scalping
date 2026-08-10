# Regime research arc — 2026-08-08 (full day)

Eight lines opened, **seven closed**, one survivor downgraded to shadow-only. The day's durable
output is not a model — it is a detector, a diagnosed root cause, and sixteen methodology rules,
several of which invalidated earlier claims made the same day.

**Live stack unchanged throughout.** BTC swingtransition (OOS gated +10.76% / −12.41%) is
untouched; the N=3 multislot shadow keeps running at MARGIN_MULT=1.5; the czz_trend sizing overlay
is **shadow-only and must not be wired live**.

---

## 1. Detector arc (θ=0.5%) — the survivor

Lineage, VAL / OOS agreement vs the θ=0.005 oracle at 100% coverage:

| config | VAL | OOS | note |
|---|---|---|---|
| czz05 (no fitting) | 61.3 | 60.8 | definitional baseline; cannot overfit |
| lgbm_jm λ1 (panel) | 66.0 | 63.5 | jump-penalized decode introduced |
| ens_w65 λ0.5 | 67.2 | 65.3 | OOS gain > VAL gain |
| seedbag8 | 67.6 | 65.6 | **REVOKED** by lag audit |
| **zigzagonly S2fine5 λ0.5** | **70.1** | **68.0** | FROZEN, 5 features |

Two detectors are frozen and downstream code must **name which one it reads**:
- **stability-first** `zigzagonly_S2fine5_lam05` — labelling/charting, 7–8 bar runs, peaks at lag +6
- **timeliness-first** `timeliness_first_boost05` — overlay, no retrain. OOS total 68.4, first-quintile
  45.0 (vs 30.8), detection lag 2 bars, **agreement peaks at lag 0 = true nowcaster**

Key technique: **jump-penalized causal DP decode**
`V_t(s) = −log p_t(s) + min(V_{t−1}(s), min_s' V_{t−1}(s') + λ)`.
λ is one continuous knob trading accuracy against persistence, and it dissolves the 1-bar confetti
that killed the CNN/LGBM classifiers. The same probabilities per-bar-argmaxed score 67.3% VAL with
3-bar runs = unusable — proof the decoder does the work, not the classifier.

**The 130-feature panel was worth −1.2pp OOS.** A zigzag-only 5-feature config beat it on both
windows, which revoked the freeze. The mechanism is **geometric** (fine 0.2/0.35% zigzags turn
before the 0.5% wave), not informational — so do not expect transfer to questions the zigzags do
not describe.

### Oracle-scale defect (found mid-day, invalidated earlier numbers)
The 4% oracle is a multi-day definition (median wave 692 bars / 7.4%) with **ZERO turning points
inside a 7-day window**, so week-level agreement compared one label to one label. Every
4%-scale headline was inflated: JM 69.9% @θ=4% decays to **48.5% @θ=0.5%**. This reversed the
day's own earlier claim — **the retired HMM is ≥ JM at every scale** (VAL θ=4%: 73.4 vs 68.1); the
HMM's real failure was flicker, not accuracy.

---

## 2. Entry axis — closed at three layers

| line | result |
|---|---|
| regime-conditioned 5m TB entry (D2) | Stage R passed, whole config family VAL-positive +18.4% → **OOS −19.5%, all 3 months negative** (9th flip) |
| JM/czz-gated MoE, 108 variants | **0/108 VAL-eligible**; best +3.3% with 0/5 seeds. No OOS spent |
| JM-bear contrarian bounce | fwd-24h +0.52% is real; best variant 1/4 positive months → monthly-consistency fail |
| timeliness-nowcaster retry | failed as a **partition** (Stage R collapsed to 0.45/0.40) AND as a **timing filter** (0/60 cells) |

**Detector accuracy went 61.3 → 68.4% and the lag peak went +12 → 0 bars, and entry PnL moved by
exactly zero.** Detector quality is definitively not the binding constraint.

### Root cause — the regime differential does not survive a quarter
Every prior Stage R ranked features *within* a regime, which never tested whether features behave
*differently across* regimes. Measuring ΔAUC = AUC_bull − AUC_bear directly:

| gate | Δ sign kept VAL | **OOS** | random-subset baseline (OOS) |
|---|---|---|---|
| d2_rule | 96% | **36%** | 44.4% |
| jm_lam32 | 84% | **52%** | 47.4% |
| czz4 | 84% | **48%** | 50.6% |

The differential survives train→VAL and **dies train→OOS, indistinguishable from random feature
subsets**. **Stage R only ever checked train→VAL — exactly the interval where it does survive — so
it was structurally blind.** That is why D2 passed its gate and then lost 19.5%.

Two follow-ups closed the axis completely:
- **Searching for persistent differential features:** train-only selection + permutation null
  (circularly shift the regime vector), R=10 → **10 real qualifiers vs a null mean of 10.5
  (range 4–21)**. Below the null. The 10 read like a genuine finding (whale_retail_ratio, cvp_regime,
  funding_roc_48, short_squeeze_risk…) and would have been reported as one without the permutation test.
- **Constructing regime-biased features:** makes the differential **2x bigger but not stable**
  (qualifier rate 22.2% vs null 10.6%, but 4 vs null max 4, p≈0.1). Contribution test: panel-only
  −22.01% vs panel+constructed −25.77% while the 18 constructed features (12%) absorb **38% of split
  gain** — textbook overfit signature.

Carriers (positioning / funding / CVD) are 20–40% of within-regime tops but only **8% of the
differential top-25** (mean |Δ| 0.0134 vs 0.0203). They carry **unconditional** direction, not a
regime-specific component — which explains why every regime expert built on them failed.

ETH reproduced the same result (ΔAUC OOS 0.48–0.52 vs random 0.55–0.65), closing
"swap the classifier and retrain" on all three assets.

---

## 3. Zigzag-sequence transformer — closed, 0/12, no OOS spent

Asked whether a small causal transformer over a **window** of zigzag states sees what per-bar
LightGBM cannot.

- Stage 0 regression gate **passed** — incumbent reproduced exactly (70.1 / 68.0, seeds matching)
- `state` mode (identical 5 inputs) **69.8–70.2, mean 69.98 vs incumbent 70.1** = inside seed noise
- **Window 32/64/128 and depth 1/2 have no effect** — if multi-bar context carried information,
  more of it would help
- `geo` mode gains ~+0.7pp — but **LightGBM showed the same trade-off in the incumbent's own
  selection** (state 69.2/run 8 → geo 69.9/run 7) and was rejected on the same floor
- **All 12 cells produced a median run of exactly 7.0 bars** vs the 8.0 floor — 12 architectures
  landing on one number shows run length is set by λ and the zigzag geometry, not the classifier

Recorded procedural weakness: the run floor is checked at Stage 1 under the *inherited* λ, while λ
is only swept at Stage 2. Future contracts of this shape should sweep λ inside the stage that
applies the floor. It did not rescue the result (the `state` cells have no VAL gain for λ to work with).

---

## 4. Sizing overlay — adopted, then downgraded, then mechanism-closed

The day's only survivor, and it did not survive intact.

**Adopted (morning):** czz_trend (bear 0.5 / chop 1.0 / bull 1.5 on `margin_fraction` at the entry
bar). MDD −10.34 → −6.63 full period, Calmar 6.98 vs 3.10, better in 3/4 periods including both
post-selection quarters, paired time-block bootstrap P=0.739, all four pre-registered gates passed.
On the N=3 shadow both axes improved (+25.30/−10.77 → +34.71/−9.24).

**Downgraded (afternoon)** by an effect-size audit, triggered by an ETH transfer failure that
exposed a paired bootstrap reporting P=0.979 for a t=0.32 difference:

| test | result |
|---|---|
| mean bear vs bull (n=59) | +0.11% vs +1.14%, t=−0.99, **p=0.33**, d=−0.26 |
| **volatility** | variance ratio **0.881 — bear LESS volatile, WRONG direction** |
| permutation on the labelling | VAL **73.5%**, OOS 94.6% — weak exactly where the map was SELECTED |
| premise check | VAL bear −0.04 vs bull +4.52 → **OOS bear +0.51 vs bull +0.04 — sign reverses** |

**Mechanism-closed (evening)** by the risk-channel test. Hypothesis: 5m czz4's θ=4% detection lag
is 985 min (~16.4 h), so "bear" may be attaching to the calm tail of a down-wave — making 0.881 a
lag artifact. Relabelled the existing ledgers with czz4 at 5m/15m/30m/1h. No learning, no sizing,
no PnL read.

Reconciliation first: recomputed czz4 is identical to the stored states (counts 130776/97197/741;
the apparent mismatch was encoding), and pooling VAL+OOS reproduces **0.881 to three decimals**.

> **The audit's 0.881 is a pooled number hiding a ~67x reversal:
> VAL 0.097 (Brown-Forsythe p=0.009 — significant in the WRONG direction) → OOS 6.49 (p=0.185).**
> The risk channel is not merely inverted, it is unstable — there is nothing to size against.

And the lag hypothesis was refuted twice: earlier detection **monotonically hurts** —
single-slot OOS 6.49 → 6.10 → 6.02 → 3.89 across 5m/15m/30m/1h; N=3 OOS 1.39 → 1.30 → 1.21 → 1.06.
Gate: 0/8. Status: **shadow-only, do not wire live.**

---

## 5. Coarser bars for regime detection — closed, 0/20

Reframed before running: the oracle is defined on the price path by θ, so bar size trades
**resolution** against **feature noise** and only pays when paired with an oracle scale. Headline
diagnostic `bars_per_oracle_wave`.

Agreement vs 5m at equal θ (full period, scored at 5m resolution):

| θ | bars/wave 5m→1h | 15m | 30m | 1h |
|---|---|---|---|---|
| 0.5% | 16.0 → **1.3** | −5.0 | −8.4 | −11.0 |
| 1% | 45 → 3.8 | −2.7 | −4.7 | −7.6 |
| 2% | 155 → 12.9 | −2.3 | −3.5 | −6.0 |
| 3% | 321 → 26.8 | −0.6 | −1.9 | −3.9 |
| 4% | 707.5 → 59.0 | −1.9 | −1.6 | −0.8 |

The mechanism intuition is **confirmed** — the penalty is pure resolution loss and shrinks
monotonically as the wave lengthens relative to the bar. **But it never crosses zero.** At θ=0.5% a
1h detector cannot even represent the wave (1.3 bars).

Secondary, not selected on: **detection lag inverts at θ=4%** — 5m 985 min, 15m 892.5, 30m 812.5,
1h 847.5. Coarser grids detect 4% turns earlier because the causal zigzag's running extreme is set
by 5m noise spikes. That finding motivated §4's risk-channel test, which then closed the axis.

---

## 6. Provenance correction — a recorded live number was wrong, and so was its selection

The N=3 shadow's 1.5x margin multiplier was adopted from a sweep run **on the gated ledgers**
(rescaling trade returns). Invalid: `margin_fraction` is an **input to the exit head**, so changing
it changes the exits and the ledger itself.

Full causal replay over the original grid {1.25, 1.5, 1.75, 2.0} plus 1.0, OOS gated:

| m | VAL PnL / MDD | OOS PnL / MDD |
|---|---|---|
| 1.0 | +22.32 / −3.39 | +16.72 / −7.27 |
| 1.25 | +28.45 / −4.23 | +21.00 / −9.03 |
| **1.5 (live)** | +34.80 / −5.05 | **+25.30 / −10.77** |
| 1.75 | +41.39 / −5.87 | +33.50 / −12.48 |
| 2.0 | +48.21 / −6.69 | +38.56 / −14.17 |

1. **Recorded +19.98 / −10.40 was wrong; actual +25.30 / −10.77.** It went unnoticed because the
   rescale *matches* the replay at 1.25x (exits do not move at that size) and only diverges at 1.5x —
   so the original note's "OOS peak was actually 1.25x, no cherry-pick" rested on a **false shape**.
2. **VAL PnL is monotone in m and VAL MDD never reaches the −8% bar (worst −6.69)**, so the stated
   rule degenerates to "take the largest multiplier in the grid" and correctly applied selects **2.0x**.
3. 2.0x then **fails its own OOS MDD gate** (−14.17 < −12.4), so strict execution of the rule
   **rejects the extension** and falls back to 1.0x.

1.5x does pass all three OOS gates on its own numbers, but choosing it for that reason is OOS
cherry-picking. Honest provenance: **1.5x is OOS-gate-derived, not VAL-selected.**
Decision taken: keep 1.5x, correct the records, let the live shadow record referee.

---

## Net position

**Survivors:** two frozen θ=0.5% detectors; the jump-penalized decode technique; the Stage R design
flaw and its replacement gate; a corrected provenance chain.

**Closed:** regime-conditioned entry (3 layers × 2 gate forms × 3 detectors), regime-expert MoE,
bear-bounce, feature-differential search *and* construction, zigzag-sequence transformer, coarser
bars, the sizing overlay's risk mechanism.

**Reopening requires new raw data** (sentiment is the named untried axis), not another architecture
over the same panel — the differential's collapse is absence of signal, not signal buried in noise.

---

## Methodology rules harvested (the day's most reusable output)

**Measurement**
1. Agreement % is meaningless without the oracle threshold — always score multi-scale.
2. Pair any accuracy metric with a median-run floor; it is what rejects confetti.
3. Separate "high agreement" from "high agreement on the easy half" — force 100% coverage.
4. Score bar scale against `bars_per_oracle_wave`, not bar size; re-derive run floors per bar size
   (state them in time, not raw bars).
5. **Split a pooled effect by window before believing it** — 0.881 was the average of 0.097 and 6.49.

**Statistics**
6. A bootstrap P is not a significance test; it measures sign consistency. Pair with t / Cohen's d /
   permutation percentile.
7. For a RISK claim, test the risk channel — an MDD adoption must show the downsized bucket is
   actually more volatile.
8. Check the premise in the *selection* window; weaker where chosen than where tested = placement luck.
9. Permute the **labelling**, not just the blocks.
10. Permutation-test every "we found N stable features" claim — the null here was 4–21 wide.

**Design**
11. Gate regime-conditioned designs on the **differential's train→OOS** sign persistence vs a
    random-subset baseline — not on within-regime train→VAL agreement.
12. Put a **Stage 0 regression gate inside the script**: rebuild the incumbent on identical
    rows/splits and halt unless it reproduces.
13. Early-stop deep models on an **inner split carved from TRAIN**, never the project VAL window,
    when the baseline trains without early stopping.
14. Never evaluate a sizing change by rescaling a ledger when sizing feeds the exit model.
15. If a selection rule's constraint never binds inside the tested grid, the rule is selecting on
    grid extent, not on data.
16. Reconcile against a prior audit's exact number before reporting one that contradicts it — here
    that turned an apparent bug into the day's main finding.
