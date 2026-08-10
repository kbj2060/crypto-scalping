# BTC θ=0.5% regime nowcaster — zigzag-SEQUENCE transformer line (2026-08-08)

**Result: CLOSED at Stage 1, 0/12 eligible cells. No OOS read spent. The frozen detector stands.**

Contract: `docs/experiments/btc_regime_theta005_zigzag_sequence_transformer_20260808.json` (pre-registered before the first run)
Script: `scripts/train_eval_btc_regime_zigzag_sequence_transformer_20260808.py`
Results: `tmp/regime_zigzag_seqformer_20260808/results.json`

## Question

The frozen detector `btc_regime_theta005_zigzagonly_S2fine5_lam05` feeds 5 causal zigzag direction
states to LightGBM **one bar at a time**. LightGBM has no temporal receptive field, so it cannot
represent "the 0.1% zigzag has flipped against the 0.5% one three times in the last 20 bars."
Does a small causal transformer over a **window** of the same states extract anything the per-bar
view cannot?

## Stage 0 — regression gate PASSED

The incumbent was rebuilt inside the same script on identical rows/splits/decode and reproduced
exactly: **VAL 70.1 / OOS 68.0**, seed bag `[123347, 216905, 543994, 559075, 976118]` matching the
frozen config. Without this the rest of the file would mean nothing.

## Stage 1 — 12-cell screen (VAL 2025Q4, decode λ=0.5, w=1.0)

| cell | channels | VAL | coverage | median run | eligible |
|---|---|---|---|---|---|
| state\|W32\|L1\|d64 | 5 | 69.9 | 100.0 | 7.0 | ✗ |
| state\|W32\|L2\|d32 | 5 | 70.0 | 100.0 | 7.0 | ✗ |
| state\|W64\|L1\|d64 | 5 | 70.2 | 100.0 | 7.0 | ✗ |
| state\|W64\|L2\|d32 | 5 | 69.8 | 100.0 | 7.0 | ✗ |
| state\|W128\|L1\|d64 | 5 | 70.1 | 100.0 | 7.0 | ✗ |
| state\|W128\|L2\|d32 | 5 | 69.9 | 100.0 | 7.0 | ✗ |
| geo\|W32\|L1\|d64 | 15 | **71.0** | 100.0 | 7.0 | ✗ |
| geo\|W32\|L2\|d32 | 15 | 70.9 | 100.0 | 7.0 | ✗ |
| geo\|W64\|L1\|d64 | 15 | 70.6 | 100.0 | 7.0 | ✗ |
| geo\|W64\|L2\|d32 | 15 | 70.5 | 100.0 | 7.0 | ✗ |
| geo\|W128\|L1\|d64 | 15 | 70.7 | 100.0 | 7.0 | ✗ |
| geo\|W128\|L2\|d32 | 15 | 70.8 | 100.0 | 7.0 | ✗ |
| **incumbent (LGBM, per-bar)** | 5 | **70.1** | 100.0 | **8.0** | ✓ |

Eligibility = coverage ≥ 50% AND median run ≥ 8 bars. **Every one of the 12 cells produced a
median run of exactly 7.0 bars**, one bar short of the floor the incumbent clears.

## What the numbers say

**1. On identical inputs the sequence view is worth nothing.** `state` mode is the apples-to-apples
comparison — same 5 channels the incumbent reads, only now as a window instead of a snapshot.
It scores 69.8–70.2 (mean 69.98) against the incumbent's 70.1. That is inside seed noise
(the incumbent's own per-seed OOS std was 0.27 in the seedbag round). The temporal receptive
field LightGBM lacks was not costing anything.

**2. Window length has no effect.** W=32, 64, 128 land within 0.4pp of each other with no monotone
trend, in both modes. If multi-bar context carried information, more of it would help. It doesn't.
Neither does depth (L=1 vs L=2 is a coin flip).

**3. The only gain comes from the `geo` channels, and it is model-agnostic.** Adding
bars-since-flip and overshoot lifts VAL ~+0.7pp and costs one bar of run length — **in both model
families**:

| | state | geo | Δ |
|---|---|---|---|
| LightGBM (incumbent Stage 1, w=0.65) | 69.2, run 8 | 69.9, run 7 | +0.7pp, −1 bar |
| Transformer (this run, w=1.0) | ~70.0, run 7 | ~70.75, run 7 | +0.75pp, — |

LightGBM already had access to these channels, already showed the same VAL-up/run-down trade-off,
and was already rejected on it — that is exactly why the frozen config is `state` mode and not
`geo`. `S5_fine3|geo` scored 70.4 with run 7.0 in that round and was rejected identically. The
transformer reproduces a trade-off the incumbent's own selection round had already mapped; it does
not add a new one.

**4. Uniform 7.0-bar runs are the tell.** Run length here is set by the decode λ and the underlying
zigzag geometry, not by the classifier — 12 architectures with different widths, depths and receptive
fields all landed on the same number. The model family is not the variable that moves persistence.

## Honest note on the procedure

The run floor is checked at Stage 1 under the **inherited** λ=0.5, while λ — the persistence knob —
is only swept at Stage 2. A family that would clear the floor at λ=1 or 2 therefore gets rejected
before λ is ever tuned. That ordering is a real weakness in the contract and is worth fixing in any
future contract of this shape (sweep λ inside the stage where the floor is applied).

It does not rescue this result, for two reasons. The incumbent went through the **identical**
procedure and cleared the floor at the same λ, so the comparison is fair under the same rule. And
more importantly, the `state` cells — the ones that actually answer the research question — show no
VAL gain at all, so no amount of λ tuning turns them into a win. Raising λ buys persistence by
giving up agreement, which would only push them further below 70.1.

The floor was **not** lowered after seeing results. Doing so would be goalpost-shifting, and this
floor is precisely what rejected the 1D-CNN zoo entry (46k flips, VAL 49.7%).

## Conclusion

The sequence axis on zigzag geometry is empty. The frozen detector
`btc_regime_theta005_zigzagonly_S2fine5_lam05` (VAL 70.1 / OOS 68.0, 5 features) stands unchanged,
as does the timeliness-first overlay. Reopening would need a different input family, not a different
architecture over the same one.

Scope reminder recorded in the contract: this was a detector-metric line, not a PnL line. Detector
agreement has already been pushed 61.3 → 68.4% with exactly zero movement in entry PnL (MoE 0/108,
timing filter 0/60), and an oracle-routed detector that knows the true wave position tops out at
75.8% — so +5.7pp is the ceiling for any architecture here.
