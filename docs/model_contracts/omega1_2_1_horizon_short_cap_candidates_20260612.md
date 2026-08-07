# Omega1.2.1 Horizon Router Short-Cap Candidates (2026-06-12)

## Status

Research artifact only. This document does not change the live bot.

The current research direction is the entry-time horizon router layered on the Omega1.2.1 runner. In-position Q-exit experiments are not the active direction because they repeatedly cut winners or reduced OOS robustness.

## Baselines

### `baseline_runner_only`

- Source: `tmp/causal_regen_20260516/omega1_2_1_horizon_router_v2_20260612/report.json`
- Validation: PnL `+277.46%`, MDD `-20.34%`, WR `63.64%`, trades `33`, avg hold `733.67`, max hold `3028`
- OOS: PnL `+186.43%`, MDD `-15.60%`, WR `72.22%`, trades `18`, avg hold `789.89`, max hold `3181`

### Horizon Router Reference

- Variant: `rot_tp065_sl080_floor50_35_hgb_s260611_e0.5_hp0.0005_p065`
- Source: `tmp/causal_regen_20260516/omega1_2_1_horizon_router_v2_20260612/report.json`
- Validation: PnL `+277.22%`, MDD `-19.00%`, WR `63.89%`, trades `36`, avg hold `667.31`, max hold `3665`
- OOS: PnL `+194.86%` in v2 report, MDD `-15.60%`, WR `75.00%`, trades `20`, avg hold `694.65`, max hold `3181`

## Candidate Ranking

### Primary Stable Candidate

Variant: `short_cap2000_min0.035`

Mechanism:

- Keep the horizon-router reference.
- Add a SHORT-only static profit cap.
- If a SHORT runner position has held at least `2000` bars and unrealized return is at least `0.035`, close it.
- No LONG cap is applied.

Metrics:

- Validation: PnL `+358.32%`, MDD `-19.00%`, WR `65.79%`, trades `38`, avg hold `631.32`, max hold `3665`
- OOS: PnL `+201.43%`, MDD `-15.60%`, WR `76.19%`, trades `21`, avg hold `661.52`, max hold `3181`

Audit:

- Seed stability checked across seeds `260601`, `260602`, `260603`, `260611`, `260612`, `260613`, `260621`, `260622`, `260623`.
- All tested seeds produced identical metrics.
- Forbidden feature audit passed through fail-fast `_reject_forbidden`.
- Ledger audit: negative holds `0`, invalid sides `0`, duplicate entries `0`.

Artifacts:

- `tmp/causal_regen_20260516/omega1_2_1_horizon_short_cap_fine_20260612/report.json`
- `tmp/causal_regen_20260516/omega1_2_1_short_cap_seed_stability_20260612/report.json`

Recommendation:

- Use this as the primary research candidate if prioritizing robustness over the highest single OOS result.

### Aggressive Candidate

Variant: `short_cap1760_min0.035`

Mechanism:

- Same as the stable candidate, but the SHORT static profit cap starts at `1760` bars.

Metrics:

- Validation: PnL `+269.96%`, MDD `-19.00%`, WR `63.16%`, trades `38`, avg hold `633.08`, max hold `3665`
- OOS: PnL `+219.02%`, MDD `-15.60%`, WR `80.95%`, trades `21`, avg hold `661.67`, max hold `3181`

Risk note:

- Ledger diff shows the OOS edge is concentrated around the 2026-01-20 SHORT sequence.
- The earlier cap closes one profitable SHORT earlier, then enables two new SHORT entries.
- Validation shows the same mechanism can be harmful: the aggressive candidate underperforms the stable candidate by about `22.1` trade-sum percentage points.

Artifacts:

- `tmp/causal_regen_20260516/omega1_2_1_short_cap_ledger_diff_20260612/report.json`

Recommendation:

- Preserve as an aggressive research candidate.
- Do not treat it as the default live candidate without additional untouched OOS or forward shadow validation.

### Shorter-Hold Candidate

Variant: `long_gb_b2400_mfe0.06_gb0.20`

Mechanism:

- Keep `short_cap2000_min0.035`.
- Add a LONG giveback exit.
- If a LONG position has held at least `2400` bars, has MFE at least `0.06`, and gives back at least `20%` of MFE, close it.

Metrics:

- Validation: PnL `+358.32%`, MDD `-19.00%`, WR `65.79%`, trades `38`, avg hold `631.32`, max hold `3665`
- OOS: PnL `+198.96%`, MDD `-15.60%`, WR `72.73%`, trades `22`, avg hold `631.09`, max hold `2400`

Tradeoff:

- Reduces OOS max hold from `3181` to `2400`.
- Adds one OOS trade.
- Costs about `2.47` PnL points and lowers OOS WR vs the primary stable candidate.

Artifacts:

- `tmp/causal_regen_20260516/omega1_2_1_long_giveback_after_short_cap_20260612/report.json`

Recommendation:

- Use only if operational hold-time reduction is more important than maximizing OOS PnL/WR.

## Consolidated Artifacts

- Summary report: `tmp/causal_regen_20260516/omega1_2_1_hold_research_summary_20260612/report.json`
- Top balanced CSV: `tmp/causal_regen_20260516/omega1_2_1_hold_research_summary_20260612/top_balanced.csv`
- Top OOS PnL CSV: `tmp/causal_regen_20260516/omega1_2_1_hold_research_summary_20260612/top_by_oos_pnl.csv`
- Top shorter-hold CSV: `tmp/causal_regen_20260516/omega1_2_1_hold_research_summary_20260612/top_shorter_hold.csv`

## Live Path Policy

- No live bot changes were made during this research loop.
- Do not wire any of these candidates into `trading_bot.py` unless explicitly requested.
- If promoted later, implement fail-fast feature contracts only. Do not add legacy aliases or compatibility fallbacks.
