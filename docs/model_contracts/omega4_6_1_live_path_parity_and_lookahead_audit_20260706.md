# Omega4.6.1 — Live-Path Parity + Lookahead/Contamination/Lag Audit (2026-07-06, FINAL)

Status: `all_checks_clean_ready_to_wire`

Closes the last two open items from `docs/model_contracts/omega4_6_1_live_clean_final_20260706.md`.

## Live feature-path parity: bug found and fixed

`trading_bot.py`'s shared `processed_df`/`FeatureEngineer` pipeline does **not** compute
`regime3_current_sensitive_wide24_*` columns -- confirmed by grepping the entire 15,672-line file
(zero hits). Every Omega-family live adapter computes this itself via a dedicated class
(`omega4_6_2_source_parent_live.py::Regime3CurrentLiveFeatures`, causal HMM `filter_proba()`
call). The initial draft of `omega4_6_1_duration_gate_live_draft_20260706.py` incorrectly assumed
these columns were already present in the input frame -- this would have raised a `KeyError` (or
worse, silently used stale/zero values) the first time it ran against the real live frame.

Fixed by importing and reusing `Regime3CurrentLiveFeatures` directly (same joblib model, same
causal filtering call). Verified:
- Live-computed wide24 values match the offline-merged CSV (used for all this session's
  backtesting) to ~1e-16 precision over a 5000-row sample.
- End-to-end parity re-run: fed the adapter RAW frames (no pre-merged overlay columns, expanding
  window truncated to each entry timestamp, i.e. a true live-path simulation) across all 32
  extended-OOS entries. Zero genuine mismatches after correcting a test-harness bug (compared
  against the pre-duration-gate ledger initially; the "mismatches" were bars where `ou_halflife`
  correctly triggered the gate and the adapter correctly returned CASH).

## Lookahead / data contamination / lag audit: clean

Per user request, ran both a static code-path review and an empirical lag test.

**Code-path review:**
- `grep`'d `features/engineering.py` and `features/elite.py` for `shift(-`, `center=True`,
  `centre=True`, `bfill` -- zero forward-looking patterns found. Two explicit code comments
  confirm `bfill` was deliberately removed specifically to prevent rolling-window-warmup
  lookahead bias (pre-existing project discipline, not something added today).
- Label columns (`zigzag_action`, `target`, `future`, `pnl`, etc.) are hard-blocked from parent
  model input via `DENY_TOKENS`/`DENY_PREFIXES` fail-fast checks in
  `train_eval_omega1_2_tabm_diffusion_risk_20260603.py`.
- The zigzag labels regenerated today (extended through June 2026, using
  `uses_future_only_for_offline_labeling=true` by design) are used ONLY for the artifact-integrity
  audit's row-count/timestamp alignment check -- never joined into any feature, prediction, or
  PnL computation.
- The VAL-only duration-gate threshold reselection (2025-10-01..12-31) never touched OOS
  (2026-01-01 onward) data during selection; OOS was scored once, after the threshold was frozen.

**Empirical lag test** (`scripts/lookahead_audit_omega4_6_1_20260706.py`): delayed entry execution
by k=0/1/2/3/6 bars (same causally-computed signal, later fill) on the final greedy replay:

| delay (bars) | PnL | MDD | WR |
|---|---|---|---|
| 0 (normal) | +138.19% | -14.15% | 50.0% |
| 1 | +114.45% | -19.58% | 48.5% |
| 2 | +163.02% | -19.69% | 51.5% |
| 3 | +187.43% | -13.91% | 53.1% |
| 6 | **+55.16%** | **-31.31%** | 42.4% |

Noisy but not cliff-shaped across delay 0-3 (expected for an hourly-cadence, multi-day-hold
trend signal -- a few 5-minute bars of delay shouldn't matter much), with real degradation only
appearing at delay=6 (30 minutes). **A lookahead artifact would show the opposite signature**:
dramatically better performance specifically at delay=0 that collapses immediately at delay=1
(because the "edge" would depend on information only available by peeking at the current/next
bar's price action). That signature is absent here.

## Verdict

No data contamination, lookahead, or lagging issues found. Combined with the live-path parity fix,
Omega4.6.1 (base, no event-flat overlay) has now cleared every checkable gate short of the actual
`trading_bot.py` wiring:

- Gate 2 (Artifact Integrity): PASS
- Runtime-native parity: PASS (3 bugs found and fixed along the way: sizing, greedy-routing,
  live regime3 feature computation)
- Lookahead/contamination/lag audit: CLEAN
- Redteam-style checks (leverage/notional caps, overlap, accounting, cost1/2/3 stress): PASS
- Live-achievable PnL: +145.34% / MDD -10.13% / 24 trades / WR 54.2% (2026-01-01..06-30)

**Not done, deliberately**: actually wiring into `trading_bot.py` (connects real capital -- a
high-stakes, hard-to-reverse action requiring explicit separate confirmation, not part of this
research/verification checklist).
