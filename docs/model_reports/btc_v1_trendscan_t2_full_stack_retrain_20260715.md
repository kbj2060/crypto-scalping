# BTC v1 trend-scan t2 full-stack retrain — 2026-07-15

## Outcome

The complete BTC v1 model stack was retrained into isolated research
artifacts. The candidate is rejected because fresh 2026 Q1 PnL is negative.
No live artifact or runtime configuration was changed.

## Retrained stack

1. Regime artifact, fit on BTC 2024 only:
   - 12-state `GaussianStateModel` HMM.
   - learned state-to-bull/bear/chop probability decoder matrix.
   - 24 BTC wide24 features.
   - 2024 Q4 validation balanced accuracy: 79.37%.
2. Regime-weighted 3-head TabM experts:
   - bull expert.
   - bear expert.
   - chop expert.
   - direction target: 1h trend-scan `abs(t) >= 2.0` adapted causally to 5m.
   - quality target: existing h48 conservative target.
   - 78,624 direction/quality rows, 77,364 exit rows, 4 epochs.
3. Risk sidecar:
   - side-split HGB on exact q055 parent prediction artifacts.
   - dynamic margin/leverage mapping selected with validation only.

BTC v1 has one serialized regime joblib, not two independent runtime regime
files. The two learned regime components listed above live inside that joblib.
The transition-hazard and stable-decoder experiments elsewhere in the repo are
not consumed by BTC v1 and were deliberately not added to this architecture.

## Results

| Evaluation | PnL | MDD | Trades |
|---|---:|---:|---:|
| Parent q055 validation | +6.98% | -1.71% | 44 |
| Parent q055 extended OOS | -4.32% | -6.34% | 153 |
| Parent + ATR/exit Q1 fresh-forward | -5.37% | -6.55% | 48 |
| Risk sidecar validation full replay | +26.70% | -5.88% | 32 |
| Risk sidecar Q1 fresh-forward full replay | -15.01% | -17.02% | 47 |
| Risk sidecar extended OOS full replay | -19.05% | -26.48% | 91 |

The new regime artifacts and overlays have different file hashes from the
previous frozen copies, proving that the regime fit and materialization ran.
With the same BTC data, seed, and deterministic implementation, their numeric
route outputs reproduced the prior route at model precision; consequently the
downstream TabM and sidecar metrics also reproduced the preceding partial
retrain.

## Evaluation contract

- `fresh_forward_bar_by_bar=true`
- `trade_ledgers_used_as_input=false`
- `saved_parent_exit_timestamps_used=false`
- `future_rows_used_for_entry=false`
- Q1 OOS: 2026-01-01 00:00 through 2026-03-31 23:55, 25,920 bars.
- Selection scope: validation only; OOS excluded from filtering, ranking, and
  tie-breaking.
- The inherited BTC v1 validation boundary remains 2025-10-01, rather than the
  project default 2025-09-01. This candidate is research-only.

## Artifacts

- Regime report: `data/ensemble/reports/btc_regime3_current_hmm_sensitive_wide24_fullstack_20260715_report.json`
- Regime model/overlays: `data/ensemble/supervised/btc_regime3_current_hmm_sensitive_wide24_fullstack_20260715`
- Parent: `tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_20260708_trendscan_t2_fullstack_fulltrain_fullexit_20260715`
- Risk sidecar: `tmp/causal_regen_20260516/btc_omega4_2_trade_risk_sidecar_20260708_trendscan_t2_fullstack_fulltrain_fullexit_q055_20260715`
- Q1 fresh-forward: `tmp/causal_regen_20260516/btc_omega4_2_trade_risk_sidecar_20260708_trendscan_t2_fullstack_fulltrain_fullexit_q055_q1fresh_20260715`

## Decision

`promotion_pass=false`. Keep the current BTC v1 live checkpoint unchanged.
