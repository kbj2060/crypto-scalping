# BTC v1 architecture ablation — 2026-07-16

## Decision

Do not promote any candidate. The separate Direction + terminal-return Meta
design with recommended `8x/4x` execution SLTP has the best validation/Q1 risk
balance, but validation PnL remains negative and Q1 was replayed during
implementation. The hard triple-barrier Meta label passed validation narrowly
but failed Q1; continuous net-return and percentile-rank regressions also
failed. These results are research guidance, not live evidence.

## Contract

- Stable BTC parent: zigzag direction + causal H48 hysteresis `c6/d12`.
- Parent comparison validation: 2025-10-01 through 2025-12-31. This deliberately
  differs from the default 2025-09-01 boundary to match the parent artifact.
- Q1 diagnostic: 2026-01-01 through 2026-03-31.
- Execution: causal 5-minute replay, next-bar entry, one open position.
- Exit replacement: ATR(192), TP `12x`, SL `6x`, TP floor/cap `7.5%/22%`, SL
  floor/cap `4%/12%`, maximum hold 72 bars.
- Fixed sizing: margin fraction `0.15`, leverage `2`, notional `0.30`.
- Separate Direction: three-seed HGB ensemble trained on hourly zigzag labels.
- Separate Meta: HGB take/skip ensemble trained from purged OOF Direction
  predictions; 72-hour fold purge and validation embargo.
- Meta threshold `0.65` selected on validation only.
- No saved ledger or saved exit timestamp was used as replay input.

## Results

| Stage | Validation PnL | Validation MDD | Q1 PnL | Q1 MDD | Q1 trades |
|---|---:|---:|---:|---:|---:|
| Learned-exit reference | -10.27% | -17.08% | -11.62% | -16.92% | 302 |
| Remove learned exit | -13.81% | -15.94% | -9.19% | -14.54% | 295 |
| + fixed conservative sizing | -10.48% | -12.15% | -6.90% | -11.05% | 295 |
| + parent event-only | -10.60% | -13.22% | -8.44% | -10.12% | 289 |
| Separate Direction, event-only | -9.57% | -10.90% | -2.30% | -5.13% | 236 |
| Separate Direction + Meta | **-1.79%** | **-4.63%** | **+5.33%** | **-4.32%** | **49** |
| + recommended 8x/4x SLTP | -0.39% | -3.19% | +3.18% | **-2.90%** | 56 |
| + triple-barrier Meta target | **+0.35%** | -4.92% | -1.54% | -4.11% | 118 |
| + triple-barrier net-return regression | -3.31% | -5.00% | -3.33% | -5.21% | 87 |
| + triple-barrier percentile-rank regression | -0.11% | **-2.31%** | -0.84% | -3.78% | 71 |

## Interpretation

1. Removing the learned exit helped Q1 but hurt validation. The learned exit is
   not the sole cause of the failure.
2. Conservative fixed sizing reduced both loss and drawdown. This is a risk
   improvement, not an edge improvement.
3. Applying event-only admission to the existing parent did not help. Its
   quality-gated final action still flickers enough that trade count only fell
   from 295 to 289.
4. Separating Direction from the parent stack materially reduced Q1 drawdown,
   but the Direction model alone still had negative PnL.
5. The purged-OOF Meta filter reduced Q1 trades from 236 to 49 and produced the
   only positive Q1 diagnostic. It also improved validation substantially, but
   validation remained negative.
6. In the final candidate, 48 of 49 Q1 trades exited at the 72-bar maximum hold.
   The current ATR floors are too wide for a six-hour holding period, so the
   effective exit is almost entirely fixed-horizon.
7. Applying the recommended TP `8x ATR` / SL `4x ATR`, with TP range `0.8%–3%`
   and SL range `0.5%–1.5%`, made the barriers active and reduced Q1 MDD from
   `-4.32%` to `-2.90%`. Validation also improved from `-1.79%` to `-0.39%`.
8. Replacing the terminal-return Meta target with a hard execution-aligned
   triple-barrier target made validation slightly positive but failed Q1. The
   training target was only 37.3% positive and was dominated by 727 stop-loss
   outcomes versus 291 take-profit outcomes. It admitted 118 Q1 trades, more
   than twice the terminal-return Meta model, and Q1 fell to `-1.54%`.
9. Regressing the exact execution-aligned net return performed worse:
   validation `-3.31%`, Q1 `-3.33%`. The training target mean was `-0.148%`
   and its median was `-0.317%`, so squared-error regression was pulled toward
   a negative central estimate.
10. Regressing the percentile rank of net return was more stable than raw
    regression, but remained negative: validation `-0.11%`, Q1 `-0.84%`.
    Validation selected the top 25% training-score cutoff. This shows that
    preserving order helps, but the current feature set does not reliably rank
    execution-aligned outcomes across regimes.

## Next experiment

Keep the separate Direction + purged-OOF terminal-return Meta architecture and
the recommended `8x/4x` execution barriers. Do not promote the hard
triple-barrier classifier, raw net-return regression, or percentile-rank
regression. A future quality experiment needs different features or a
regime-conditional target; changing only the loss/label representation did not
generalize. Q1 must not be used again for model selection or promotion.

## Artifacts

- Script: `scripts/test_btc_v1_architecture_ablation_20260716.py`
- Report: `tmp/causal_regen_20260516/btc_v1_architecture_ablation_20260716/report.json`
- Summary: `tmp/causal_regen_20260516/btc_v1_architecture_ablation_20260716/architecture_ablation_summary.csv`
- Equity charts and per-stage trade charts are stored in the same artifact directory.
