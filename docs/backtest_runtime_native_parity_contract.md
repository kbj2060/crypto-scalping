# Backtest Runtime-Native Parity Contract

Last updated: 2026-06-11

## Standing Rule

Omega live-model backtests must use the runtime-native path when the goal is live/shadow parity:

1. Feed bars one-by-one.
2. Build the same live feature frame used by `trading_bot.py`.
3. Call `FinalGovernorRuntime.decide()` directly.
4. Execute the signal on the next bar open.
5. Write an isolated replay ledger, never the live ledger.

Vectorized research backtests are allowed only for fast screening. They must not be used as final live-parity evidence.

## Root Causes Found

The previous research backtest and runtime-native replay diverged for three reasons:

1. OOS warmup mismatch:
   The replay discarded the first 1200 OOS bars, while the research backtest started from the first OOS rows. OOS replay now attaches validation-tail prehistory for features but evaluates from the first OOS bar.

2. Split feature-contract mismatch:
   Validation lacked `funding_z_score` and `regime_persistence`, while OOS already had them. Runtime-native replay now builds these with the same causal feature logic:
   `funding_z_score` from rolling `last_funding_rate`, and `regime_persistence` via `features.high_order_state.add_high_order_state_features()`.

3. Accounting/liquidity mismatch:
   Research backtest used next-open maker-limit accounting, but runtime replay close payloads fell back to synthetic default exit liquidity. Replay close accounting now records both entry and exit as `signal_immediate_maker_limit`, producing `maker+maker` ledger rows.

## Canonical Replay Script

Use:

```bash
/home/llewyn/miniconda3/envs/quant_ai/bin/python \
  /home/llewyn/crypto-scalping/scripts/backtest_omega1_2_1_runtime_native_replay_20260610.py \
  --split oos \
  --warmup 1200 \
  --max-bars 0 \
  --progress-every 2000
```

Artifacts:

- Summary: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega1_2_1_runtime_native_replay_20260610/summary.json`
- Decisions: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega1_2_1_runtime_native_replay_20260610/<split>/runtime_native_decisions.csv`
- Trade journal: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega1_2_1_runtime_native_replay_20260610/<split>/runtime_native_trade_journal.csv`
- Closes: `/home/llewyn/crypto-scalping/tmp/causal_regen_20260516/omega1_2_1_runtime_native_replay_20260610/<split>/runtime_native_closes.csv`

## Verification Snapshot

OOS 3000-bar smoke passed after the patch:

- First trade price path matched research: `2977.66 -> 3080.60`
- Close ledger fee model: `maker+maker`
- Entry liquidity: `signal_immediate_maker_limit`
- Exit liquidity: `signal_immediate_maker_limit`
- Smoke metrics: PnL `-6.00%`, MDD `-11.76%`, WR `0.00%`, trades `2`

## Operational Constraint

Do not add legacy feature aliases or compatibility fallbacks to make a backtest pass. If a feature contract is missing, generate the feature from the causal source logic or fail fast.
