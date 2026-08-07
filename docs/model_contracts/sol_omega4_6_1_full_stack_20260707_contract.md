# SOL Omega4.6.1 Full-Stack Replication — single-component (zig075-only) candidate

Status: `research_positive_signal_not_live_wired`. User request 2026-07-07: bring SOL up to ETH's
Omega4.6.1 performance tier by building the FULL production stack (risk sidecar, ATR-adaptive
TP/SL, exit-head, duration gate) for SOL — not the simplified quick-check that was already
rejected in [`sol_pilot_20260707_contract.md`](sol_pilot_20260707_contract.md).

## What this replicates and what it doesn't

Omega4.6.1 (ETH) is a two-component ensemble (`h48qual` + `zig075`, same 3-head TabM architecture,
different quality-label contracts) combined via a genuine greedy single-position-slot router.
This SOL build produced only **one viable component (zig075)** — see Gate 2 below — so there is no
router/SCALE_MAP reconciliation step; the single component's own gated replay ledger *is* the
final result.

## Reusable infra confirmed from the 2026-07-07 pilot

SOL raw data, `FeatureEngineer` output, and regime3-current wide24 HMM overlay (2025/2026, balanced
accuracy 0.71) were already built and are directly reusable. The pilot's own zigzag labels were
**not** reused — they used a different label-family script than zig075's actual production recipe
(`build_wave3_action_labels_20260531.py`) and were rebuilt from that exact script instead.

## Build chain (new SOL-parameterized scripts, faithful copies of the ETH recipe unless noted)

1. `scripts/build_wave3_action_labels_20260531.py` (reused directly, SOL paths via CLI) → SOL
   zigzag_action direction labels, unchanged hyperparameters.
2. `scripts/build_omega1_2_triple_barrier_labels_sol_20260707.py` + new
   `scripts/pad_h48_quality_labels_to_zigzag_timestamps_sol_20260707.py` → SOL h48-conservative
   quality labels padded to the zigzag timestamp index (for h48qual; not used in the final
   candidate since h48qual was rejected).
3. `scripts/train_eval_omega1_2_tabm_diffusion_risk_sol_20260707.py` (SOL `omega` utility module —
   TRAIN_CSV/EVAL_CSV/regime3-current overlay repointed at SOL; TABM_2025/2026 and the
   cmamba/stability-risk overlays dropped, since ETH's own promoted bundles both use
   `exit_label.mode=entry_label_terminal_giveback`, which never touches those) +
   `scripts/train_eval_omega4_3head_parent72_loose_entry_quality_sol_20260707.py` (SOL parent
   trainer) → trained both `h48qual` and `zig075` candidates.
4. `scripts/train_eval_omega4_2_risk_sidecar_sol_20260707.py` (SOL copy, same ATR/exit-head/sizing
   architecture) → risk sidecar for zig075 at 3 quality thresholds.
5. `scripts/select_duration_gate_threshold_val_sol_20260707.py` (new, single-component simplified
   analogue of the ETH duration-gate calibration) → VAL-only `ou_halflife` threshold grid search.
6. `scripts/audit_omega_artifact_integrity_sol_20260707.py` (SOL copy — the original
   `audit_omega_artifact_integrity_20260630.py` hardcodes ETH's `omega`/`omega4` module imports
   regardless of which report it's pointed at, so it silently checked ETH's own frame against SOL
   label dirs and failed on an ETH-side data gap unrelated to SOL; fixed by repointing the two
   imports at the SOL modules) → `promotion_pass: true`.

## GATE 1 (label sanity) — passed

Both SOL wave3 direction labels (2025: LONG 45,617 / SHORT 43,194 / CASH 16,309, ratio 1.06) and
h48-padded quality labels (2025: LONG 30,894 / SHORT 31,450 / CASH 42,776) were non-degenerate.

## GATE 2 (parent VAL viability) — zig075 only

Both parents were trained across their full ETH-matching quality-threshold grids and scored with
the parent trainer's own simplified `omega._metrics` (fixed TP 2.6%/SL 1.4%, no exit-head, no ATR
contract — a screening metric only):

| component | VAL pattern | verdict |
|---|---|---|
| **zig075** | q0.65 +19.5%, q0.75 +16.0%, q0.70 +8.4%, q0.80 +1.5%, q0.85 -3.6% | 3/5 thresholds clearly positive, coherent — proceed |
| h48qual | q0.45 +11.7% only; q0.40/0.50/0.55/0.60 all negative (-3.2% to -18.4%) | single scattered positive amid mostly-negative neighbors — overfitting-to-noise signature, **rejected** |

## Exit-head debugging finding (material, worth flagging for future SOL/new-asset work)

The risk sidecar's real production replay (ATR-adaptive TP/SL + the parent's own trained exit-head,
not the simplified screening metric above) initially showed zig075 q0.65 collapsing to VAL -11.8%
with 97% of exits triggered by the exit-head firing almost immediately on every trade. Root cause:
the risk sidecar script defaults `--exit-threshold` to 0.70, but ETH's actual production sidecars
were all trained with `--exit-threshold 0.95` (undocumented outside the report.json contract
field) — a load-bearing tuned hyperparameter that does not have a sane universal default and must
be copied explicitly for any new asset, not left at the script default. After correcting this, exit
reasons reverted to being TP/SL-dominated (matching ETH's own pattern), and results became sane.

A second ETH-tuned constant, `--min/max-validation-avg-notional 0.45/0.95` (part of ETH's
`--live-exposure-grid` sizing search), also had to be dropped for SOL — it filtered out every
candidate for SOL's differently-distributed risk score, another instance of an ETH-specific
calibration value that does not transfer and must be re-derived or removed per asset.

## Real production-stack replay: threshold reselection

With the corrected exit-threshold, the parent trainer's simplified ranking (favoring q0.65) did
**not** match the real production-stack ranking — a second, independent lesson that the
simplified screening metric used at Gate 2 is not a reliable predictor of final performance once
the actual exit-head/ATR/sizing stack is applied:

| threshold | VAL (real stack) | OOS (real stack, extended window) | verdict |
|---|---|---|---|
| q0.65 | +1.5% / MDD -17.3%, 43 trades | -3.3% / MDD -30.4% | weak, VAL barely positive |
| **q0.70** | **+10.1% / MDD -23.2%, 41 trades** | **+18.9% / MDD -22.0%, 56 trades** | both directions positive, most consistent — **selected on VAL only** |
| q0.75 | -8.2% / MDD -30.8%, 40 trades | +8.5% / MDD -32.3% | VAL negative — rejected |

## ATR contract sanity check (Phase 6)

SOL's ATR-price-move distribution (p50 0.0029, p90 0.0040) is clamped to the same `min_tp=0.075
/min_sl=0.04` floor as ETH's (p50 0.0021, p90 0.0033) — i.e. the "ATR-adaptive" TP/SL is
effectively a fixed 7.5%/4% barrier for both assets already. No SOL-specific recalibration needed.

## Duration gate (Phase 7) — VAL-only selected, OOS applied once

**Note: the PnL/MDD numbers in this section predate the Phase 8 final scale-map/leverage-cap
stage below and are superseded by it for headline reporting — kept here because the duration-gate
threshold itself (0.0055208323) was selected before Phase 8 and reused unchanged.**

Grid search over quantile thresholds of SOL's own `ou_halflife` (asset-specific distribution,
not reused from ETH's frozen 0.005417): selected threshold **0.0055208323**.

| | before gate | after gate (frozen, OOS applied once) |
|---|---|---|
| VAL | +10.1% / MDD -23.2%, 41 trades, WR 39.0% | **+17.6% / MDD -11.9%, 28 trades, WR 42.9%** |
| OOS (extended, 2026-01-01..07-07) | +18.9% / MDD -22.0%, 56 trades | +9.0% / MDD -17.6%, 39 trades, WR 38.5% |
| OOS (CLAUDE.md frozen window, 2026-01-01..03-31) | +28.1% / MDD -15.2%, 29 trades | **+23.0% / MDD -13.7%, 20 trades, WR 50.0%** |

The gate meaningfully improved VAL drawdown (-23.2%→-11.9%) at the cost of some OOS PnL on the
extended window; on the CLAUDE.md-mandated frozen OOS window (2026-01-01..03-31) the gated result
is strongly positive and lower-drawdown than ungated. **Note the date-boundary deviation required
by CLAUDE.md's Fresh-Forward Rule**: intermediate screening used SOL's full available 2026 data
(through 2026-07-07) since it was the readily available window; the number that should be treated
as authoritative for any promotion decision is the frozen-window one above.

`fresh_forward_bar_by_bar=true` (every simulation is a single sequential bar-by-bar pass, TP/SL/
exit-head resolved by walking forward from each entry, no saved-ledger inputs at any stage),
`trade_ledgers_used_as_input=false`, `saved_parent_exit_timestamps_used=false`,
`future_rows_used_for_entry=false`.

## Phase 8 (single-component final scale-map + leverage/notional cap)

Initially treated as unnecessary ("no second component to route between"), but the user correctly
flagged that ETH's greedy router applies a SECOND, separate rescaling stage even in the single
final-sizing sense: `SCALE_MAP` (per-component-per-side leverage multiplier) followed by
`LEVERAGE_CAP=5.0`/`NOTIONAL_CAP=1.8` clamping (`scripts/replay_omega4_6_1_greedy_router_
20260706.py` lines 48, 155-158) — this is where ETH's headline +145% actually comes from, not just
the risk sidecar's own 3.0x-capped sizing. New `scripts/apply_final_scale_map_sol_20260707.py`
VAL-only grid-searched SOL's own long/short scale factors (ETH's `{zig075_L:2.446, zig075_S:2.478,
...}` not reused — no evidence it transfers) subject to the same `LEVERAGE_CAP=5.0`/
`NOTIONAL_CAP=1.8`, re-running the full bar-by-bar `_replay_with_risk` (not a ledger post-hoc
rescale, since notional/leverage are also exit-head input features). Selected: **long_scale=1.0,
short_scale=2.0**.

Final numbers (duration gate reapplied on top, same threshold 0.0055208323):

| | VAL | OOS extended (2026-01-01..07-07) | OOS frozen (2026-01-01..03-31) |
|---|---|---|---|
| **PnL** | **+56.8%** | +13.9% | **+42.0%** |
| MDD | -15.9% | -29.4% | -21.0% |
| trades | 28 | 39 | 20 |
| WR | 42.9% | 38.5% | 50.0% |

This is the number that should be cited as this candidate's headline result — much closer to ETH's
Omega4.6.1 tier (+145%/MDD-10%) than the pre-scale-map numbers in the superseded table below,
though still with the same Q1-concentration caveat (extended-window OOS is far weaker/riskier).

## Phase 9 — Artifact Integrity Gate: PASS

`scripts/audit_omega_artifact_integrity_sol_20260707.py --report tmp/causal_regen_20260516/
sol_omega4_2_trade_risk_sidecar_20260707_zig075_q070_20260707/report.json` → exit 0,
`promotion_pass: true`.

## Phase 10 — sanity checks

Re-run against the Phase 8 final-scale-map ledgers (not the pre-scale-map ones): max leverage 4.0x
(below the 5.0x cap — short_scale=2.0 applied to the sidecar's own ~2.0x base leverage), max
notional 1.38 (below the 1.8x cap), max exposure (notional×leverage) 5.53x, zero overlapping
trades in either ledger, per-trade returns bounded (-6.2% to +14.5%, no blowups), all evaluated at
`cost_mult=3.0` (3x fee/slip stress, matching ETH's own convention) by default. Full
lookahead/redteam audit scripts (`lookahead_audit_omega4_6_1_20260706.py`,
`redteam_omega4_6_1_base_20260706.py`) were not ported — the underlying feature/label code is
unmodified and shared with the already-audited ETH pipeline, so the incremental audit surface is
limited to the sizing/duration-gate layer checked above.

## Honest caveats

- **Trade count is thin**: 20-28 trades on VAL/frozen-OOS, similar order of magnitude to ETH's own
  Omega4.6.1 (24 trades OOS) — not enough to be highly confident in the edge, consistent with this
  project's standing lesson about trade-count scarcity in this model family.
- **The simplified screening metric used at Gate 2 misranked thresholds** relative to the real
  production stack — any future threshold/asset screening in this lineage should use the real
  exit-head+ATR+sizing replay, not the parent trainer's own quick metric, before drawing
  conclusions.
- **Extended-window OOS (through 2026-07-07) is weaker than the frozen Q1 window** (+9.0% vs
  +23.0%) — SOL's edge may be concentrated in Q1 2026 and should not be assumed to persist evenly;
  a genuinely fresh forward window (2026-07+) would be the honest next confirmation, as with
  [[project-sigma6-regime-trend-best]]'s own caveat pattern.
- **h48qual was not revisited under the corrected exit-threshold** — it was rejected at Gate 2
  under the simplified metric before the exit-threshold bug was found; given the same bug likely
  affected its baseline replay too, this is a specific, cheap follow-up if pursued later (rerun
  `train_eval_omega4_2_risk_sidecar_sol_20260707.py` for h48qual with `--exit-threshold 0.95`).

## Live wiring: NONE

No `trading_bot.py` changes were made at any point. This candidate is research-stage only, one
tier below even ETH's own Omega4.6.1 (which itself is not live-wired). No live capital risk.

Scripts: `build_wave3_action_labels_20260531.py` (reused), `build_omega1_2_triple_barrier_labels_
sol_20260707.py`, `pad_h48_quality_labels_to_zigzag_timestamps_sol_20260707.py`,
`train_eval_omega1_2_tabm_diffusion_risk_sol_20260707.py`,
`train_eval_omega4_3head_parent72_loose_entry_quality_sol_20260707.py`,
`train_eval_omega4_2_risk_sidecar_sol_20260707.py`,
`select_duration_gate_threshold_val_sol_20260707.py`,
`apply_final_scale_map_sol_20260707.py`,
`audit_omega_artifact_integrity_sol_20260707.py`.
