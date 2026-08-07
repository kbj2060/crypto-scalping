# Omega4.6.1 — Improvement Roadmap (2026-07-07)

Addresses the weaknesses identified in this session's review (see
`omega4_6_1_upgrade_investigation_20260706.md` for the six rejected upgrade candidates and
`omega4_7_rl_dsac_20260707_contract.md` for the rejected RL rebuild). This roadmap does NOT propose
another architecture-search candidate -- six failed already, three with the same
VAL/OOS-disagree signature, which is evidence the current config is a local optimum for THIS
dataset size, not that a better model is waiting to be found. The roadmap instead targets the
things that are actually broken or unverified: risk visibility, robustness evidence, and process,
not the model's internals. `trading_bot.py` stays frozen throughout Phase 1-2; only Phase 3 items
are candidate-live-relevant, and only after their own gate passes.

## Phase 1 -- DONE (2026-07-07)

1. **Cost stress test (cost1/2/3).** DONE --
   `scripts/audit_omega4_6_1_phase1_robustness_20260707.py`. Passes: stays clearly positive at 3x
   cost in both VAL (+19.57%) and OOS (+104.56%). See
   `omega4_6_1_phase1_robustness_20260707_results.md`.
2. **Leave-one-out trade sensitivity.** DONE, same script. No single trade IS the whole edge, but
   removing the best/worst trade swings headline PnL by 20-35pp (VAL range ~35-68%, OOS range
   ~114-166%) -- treat reported numbers as a range, not a point estimate.
3. **Rolling walk-forward on 2025-Q1/Q2/Q3.** DONE, same script. Q1 +28.5%/Q2 +40.0% positive, but
   **Q3 -9.7% (MDD -44.4%)** with the zig075-SHORT bucket itself inverting (-0.517 sum_ret, the
   only negative reading across 5 windows) -- the single most concrete quantified risk found this
   session. Follow-up diagnostic (`scripts/diagnose_omega4_6_1_q3_regime_20260707.py`) found this is
   part of a real, cross-window pattern (zig075-SHORT tracks broad market trend, not the bar-level
   regime3 tag). Tested as Candidate 7 (`scripts/train_eval_omega4_6_1_trend_veto_20260707.py`, a
   causal trailing-return veto, TRAIN-select/VAL-check/OOS-confirm) -- **REJECTED**: hurt both VAL
   and OOS despite doubling TRAIN PnL, confirming the binding constraint is trade-count scarcity,
   not a missing predictive rule. Full writeup: Candidate 7 in
   `omega4_6_1_upgrade_investigation_20260706.md`.

## Phase 2 -- DONE (2026-07-07)

4. **Live monitoring / drift alerting.** DONE --
   `scripts/monitor_omega4_6_1_live_drift_20260707.py`. Builds a reference distribution from every
   window scored this session (2025-Q1/Q2/Q3 + VAL + OOS, 137 trades total; worst_wr=0.316,
   worst_mdd=-44.37%, trade_return_range=[-8.77%, +17.59%]), cached at
   `data/ensemble/omega4_6_1_reference_distribution.json`. Parses `data/live/trade_journal.jsonl`
   for Omega4.6.1 ENTER/EXIT events, flags (WARN, not an auto-kill-switch) if live win rate/MDD/any
   single trade return falls outside the historical range once >=8 live trades have closed. Current
   live state: 0 closed trades (1 open position) -- correctly reports "accumulation mode", no false
   alarm. Rerun periodically (cron/supervisor hook) going forward; report at
   `data/ensemble/omega4_6_1_live_drift_report.json`.
5. **Feature-contract drift watch.** DONE --
   `scripts/audit_omega4_6_1_feature_drift_scheduled_20260707.py`. Unlike the one-off
   `verify_live_feature_pipeline_parity_20260706.py` (pinned to a historical CSV), this diffs
   against whatever is the CURRENT live snapshot (`data/live/decision_feature_frame_snapshot.pkl.gz`)
   so it's re-runnable at any point going forward. First run: **PASS, 0/96 columns exceed the 1e-3
   relative-diff tolerance** against the live bot's own June/July 2026 snapshot. Report at
   `data/ensemble/omega4_6_1_feature_drift_report.json`.
6. **Lighter macro-event overlay.** ALREADY DONE in a prior session (2026-07-06, discovered while
   implementing this item -- not re-run to avoid duplicating work): three variants tested
   (`docs/model_contracts/omega4_6_1_macro_event_veto_20260706.md`) -- pure entry veto (null result,
   entries never land near events given ~1 trade/week), exposure haircut while holding through an
   event (small MDD-neutral PnL gain +5.7 to +10.7pp, but based on only 11 trades' partial overlaps,
   too weak to promote), and lock-profit-at-T-30min (PnL +2.77pp but runs counter to this project's
   "let winners run" principle). Conclusion already reached: none strong enough to adopt. No new
   work needed here.

## Phase 3 (medium/long-term, gated on data that doesn't exist yet) -- readiness checkpoint built 2026-07-07

Phase 3 is deliberately gated on data/trade-count that doesn't exist yet; there is nothing
legitimate to build for items 7-9 until their gates are met (doing so now would repeat the exact
mistake of Candidates 1-7 -- searching for a new rule with too few trades to separate structure
from noise). `scripts/audit_omega4_6_1_phase3_readiness_20260707.py` (rerun anytime; report at
`data/ensemble/omega4_6_1_phase3_readiness_report.json`) checks both gates automatically. First run
(2026-07-07): **both gates NOT READY.**

7. **Microstructure/orderbook data.** Gate: needs the LONGEST CONTIGUOUS segment (no gap >6h) to
   span >=4 months, not just total elapsed time (raw span is misleading given the multi-day
   outages found in Phase-1-era analysis). Current: total span 2.1 months, but longest contiguous
   segment is only **0.5 months** -- not ready. Do not redo the correlation screen (last result:
   |corr|<0.015 across every feature at 13,832 samples) until the checkpoint script reports ready.
8. **Diversification as risk management, not alpha-seeking.** Gate: effective trade count (OOS
   baseline 24 + live-closed-since) reaching ~48 (2x). Current: **24/48** -- 0 live trades closed
   yet, not ready. Success criterion when eventually revisited: reduces book MDD (may reduce PnL --
   different objective than Sigma9's failed BTC+ETH PnL-seeking attempt).
9. **Targeted architecture re-opening**, only after Phase 1-2 evidence points to a specific weak
   spot AND the same trade-count gate as item 8 is met. Currently gated by the same 24/48 count.

Rerun `scripts/audit_omega4_6_1_feature_drift_scheduled_20260707.py` and
`scripts/monitor_omega4_6_1_live_drift_20260707.py` periodically (Phase 2) -- the trade-count gate
updates automatically from the live drift monitor's closed-trade count each time it's rerun.

## Explicit non-goals

- No new TP/SL/exit/routing/sizing model search until Phase 1 items 1-2 are done and reviewed --
  repeating architecture search without first quantifying cost-sensitivity and trade-level fragility
  would be searching in the dark again.
- No live wiring changes in Phase 1-2. Phase 3 item 6 (event haircut) is the only item that could
  reach a live-promotion discussion, and only after its own VAL-select/OOS-confirm pass.
