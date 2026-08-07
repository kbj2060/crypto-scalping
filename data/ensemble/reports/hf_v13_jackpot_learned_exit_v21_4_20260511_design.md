# HF V13 Jackpot Learned Exit V21.4 Design

## Goal

Remove parent TP/SL as direct exit triggers and let a learned exit governor decide when to close an open position.

## Architecture

1. **Parent entry policy**
   - Uses `hf_v13_clean_regime_margin110_20260511`.
   - Keeps the same next-bar execution contract as the current V21.2 live/backtest path.
   - Entry notional is capped at the V21.2 selected cap.

2. **Jackpot add-on layer**
   - Reuses the audited V21.2 jackpot runner.
   - Keeps the same cost-survival gates:
     - jackpot probability
     - q90 upside
     - bad-add-on probability cap
     - cost3 survival probability

3. **Learned exit governor**
   - Parent `take_profit` and `stop_loss` are not direct close conditions.
   - They are allowed only as input context features.
   - The model receives in-position state:
     - side
     - current notional
     - parent notional
     - bars since entry
     - unrealized return
     - MFE/MAE
     - giveback/recovery
     - drawdown
     - parent TP/SL/max-hold context
     - parent feature frame
   - Training labels compare immediate next-bar exit utility against future best-exit utility over the position horizon.
   - A learned classifier predicts whether to exit.
   - A learned advantage regressor predicts immediate-exit advantage versus future best exit.

4. **Safety layer**
   - Direct TP/SL is disabled.
   - A wide `safety_max_hold` remains to prevent orphan positions.
   - The initial implementation evaluates exit every 6 or 12 bars to reduce live latency.

## Split Contract

- Exit model train: `2025-01-01..2025-09-30`
- Exit threshold selection: `2025-10-01..2025-12-31`
- OOS evaluation: fixed `2026` only after selection
- No 2026 selection allowed.

## Current Status

The experiment code exists in:

- `/home/llewyn/crypto-scalping/scripts/train_eval_hf_v13_jackpot_learned_exit_v21_4.py`

The first HGB version and the lightweight decision-tree version were both too slow for an immediate full OOS run in this thread. Because the backtest did not complete, this model is **not audited**, **not promoted**, and **not injected into the live bot**.

## Red Team Decision

Status: `iterate`

Blocking reason:

- Full train/validation/OOS backtest did not complete within practical runtime.
- No OOS metrics exist yet.
- Therefore the model cannot be compared against V21.2 and cannot be injected.

Recommended next design:

- Precompute parent feature rows once per dataset.
- Precompute exit-state features during a single trade replay.
- Cache model predictions per replay/config.
- Then rerun full OOS and audit.
