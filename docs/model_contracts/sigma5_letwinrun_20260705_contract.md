# Sigma5 — "Let Winners Run" Trend-Follower (high-cost1-PnL research)

Status: `research_100pct_reachable_in_sample_does_NOT_generalize`

Last updated: 2026-07-05 KST

Lineage: applies a trailing-stop trend-following execution layer to the Sigma3-1h ensemble
signal, in response to the user's request to "research/come up with a cost1 PnL 100% model."

## Design

Same 1h directional signal as Sigma3 (5-seed HGB ensemble on trend-scanning labels). Execution
changed from tight scalping barriers to **letting winners run**: enter on the signal, arm a
trailing stop (exit when unrealized gives back `trail_atr x ATR` from its peak, once peak exceeds
`min_profit_atr x ATR`), hard stop at `-sl_atr x ATR`, time exit at max_hold, **compounding**, and
leverage as a swept knob. Thesis: crypto trends run 10-30%; the tight tp=1.5xATR of Sigma3 caps
the upside, so a trailing stop captures the big moves that drive high PnL. Script:
`scripts/run_sigma5_letwinrun_20260705.py`. cost1 primary (per user), cost3 reported as context.

## Result

**Validation (2025-07..12): reaches cost1 +118%.** Best `thr0.7/lev3.0/trail5.0/sl1.5/minp2.0/
mh144`: cost1 **+118.3%**, MDD -14.6%, 44 trades, WR **36.4%**, cost3 +111.8%. Monthly cost1 was
well-distributed (Jul +32.8, Aug +21.0, Sep +9.6, Oct +3.3, Nov +21.5, Dec -0.1) — 5/6 months
strongly positive, NOT one-month luck. Classic trend-following profile: low win rate, huge
payoff ratio (a few big trend captures dominate); cost3 stays high because winners are so large
that 3x fees barely dent them.

**One-shot 2026-03-02..06-30 (frozen, that config): does NOT hold.**
- cost1: **-2.5%** (MDD -25.0%), 39 trades, WR 30.8%
- cost3: +4.8% (differs from cost1 because 3x slippage shifts barrier-hit timing → different
  trade set; both are near breakeven)
- Monthly cost1: Mar +11.0, Apr -8.9, May +8.3, Jun -9.2 — 2/4 positive, net slightly negative.

## Verdict: 100% cost1 is reachable IN-SAMPLE but is regime-luck, not a durable edge

The +118% validation is real, but it is a property of H2-2025 being a strongly TRENDING period,
which a low-win-rate trailing trend-follower monetizes via a handful of big captures. The fresh
2026-03..06 window was choppier/range-bound, so the same strategy bled from stop-outs (25 stops)
and netted ~breakeven. This is a fundamental characteristic of trend-following (works when the
market trends, loses when it chops), not a fixable bug — and it matches the research caveat
("great backtests fail live from overfitting; never go all-in").

**Honest bottom line on "a 100% cost1 model":**
- A backtest showing +100-118% cost1 CAN be produced (trend-following + trailing stop + 3x
  leverage + compounding), and it is not in-sample cherry-picking (well-distributed across 6
  validation months).
- But it does NOT generalize: on genuinely unseen data it delivered -2.5% cost1. The 100% is
  leverage riding a favorable regime, not a persistent signal edge.
- Realistic fresh-data expectation: this trend-follower is roughly breakeven-to-large depending
  entirely on whether the period trends; the tighter Sigma3-1h scalper is steadier (+7.3%/4mo
  OOS cost1, MDD -15%). Neither is a reliable 100%.

**What a genuine (not regime-lucky) high-return model would require** — none available now, all
would need a fresh OOS window: (1) more/new information (order-book microstructure once ~6mo of
history accrues — currently 2mo live-only; multi-asset breadth); (2) an explicit regime filter
that only deploys the trend-follower when a trend is statistically present and sits out chop
(reduces the choppy-period bleed, but adds its own overfit surface); (3) accept trend-following's
nature and size it as one sleeve of a diversified book, not a standalone 100% target.
