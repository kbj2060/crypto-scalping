# BTC v2 Direction + Meta loop results — 2026-07-16

## Decision

Do not promote or retrain the BTC v1/live stack with any candidate from this
loop. The implementation and artifacts pass integrity checks, but no model
passes the historical performance gates. The preregistered future gate is also
not yet observable because local BTC data ends before 2026-07-17.

## Search completed

- 31 completed immutable experiment runs (plus one interrupted empty run).
- 12,544 threshold/model/execution candidates.
- Label families cross-checked: fixed horizon, dollar event, directional
  change, denoised SSL, prior meta-label, reward shaping, zigzag, trend-scan.
- Direction: HGB and CatBoost, balanced/unbalanced waves, trend-t weighting.
- Meta: none, HGB classifier, side-specific HGB, CatBoost, robust continuous
  regression/ranking, terminal target, execution-aligned target, 3x-cost target.
- Features: BTC F0 and F1 microstructure; F0 Direction with F1 Meta.
- Execution: 6/12/24-hour holds, side thresholds, re-entry cadence, momentum
  confirmation, and tighter ATR stop variants.
- Adaptation: causal monthly Meta refit with full outcome delay.
- Regime conditioning: four train-frozen BTC volatility/trend states with
  independent F0 terminal and F1 execution-aligned Meta specialists.

No candidate among all 12,544 simultaneously achieved at least 40 validation
trades and non-negative 3x-cost PnL.

## Best useful diagnostic

`run22_f0_conditional_calibration` is the best risk/generalization diagnostic,
not a promotion candidate:

| Metric | Validation | Q1 diagnostic |
|---|---:|---:|
| PnL | +2.684% | +1.371% |
| MDD | -0.350% | -0.674% |
| Trades | 14 | 13 |
| 3x-cost PnL | +1.485% | +0.271% |

It passes five of eight historical gates. It fails sample size (14 versus 40)
and its candidate-conditional Meta score is not calibrated: Spearman is
-0.0559 and top-quintile lift is 0.0799%. Its positive result therefore comes
from a sparse selected set, not a validated monotonic quality score.

## Closest six-gate candidates

- F1 terminal Meta: validation +1.763%, 14 trades, 3x cost +0.574%; Q1
  -1.275%.
- F1 execution Meta with side thresholds: validation +1.881%, 13 trades, 3x
  cost +0.775%; Q1 -0.641%.
- F0 trend-t weighted with tight stop: validation +1.342%, 12 trades, 3x cost
  +0.327%; Q1 +0.214% but Q1 3x cost -0.290%.

The 40+ trade candidates consistently lose after 3x costs. Monthly Meta refit
raised F0 to 43 trades, but stress PnL was -2.252%. F1 monthly refit produced 53
trades with stress PnL -4.104%.

Regime specialists also solved only the count gate. F0 terminal Meta produced
49 trades with +0.158% base PnL but -3.883% stress PnL and Q1 -2.101%. F1
execution Meta produced 46 trades with +0.158% base PnL but -3.640% stress PnL,
score Spearman 0.052, and Q1 -0.599%.

## Alternative-label replay

Replaying the prior label-family model signals under this exact next-5-minute
TP/SL contract produced negative validation PnL for every family. Dollar-event,
which had been positive under a fixed 24-hour hourly exit, fell to -7.887% here.
This confirms that its old result is execution-contract-specific.

## Verification

- Targeted tests: 12 passed.
- All 31 completed run manifests and causal/feature/execution contracts audited
  cleanly.
- Leading artifact audit: `manifest_pass=true`, `contract_pass=true`,
  `historical_gate_passed=false`, `future_gate_passed=false`,
  `promotion_pass=false`.

The future gate starts 2026-07-17, while the current 5-minute tape ends at
2026-07-12 16:50. It requires 90 days and at least 50 trades, so it cannot be
truthfully passed before the requested morning deadline. The runner preserves
that boundary and will evaluate it when a new immutable run is launched after
enough data exists.

## Artifacts

- Core: `scripts/btc_v2_research_core_20260716.py`
- Runner: `scripts/train_eval_btc_v2_direction_meta_20260716.py`
- Auditor: `scripts/audit_btc_v2_artifact_20260716.py`
- Tests: `test/test_btc_v2_research_core_20260716.py`,
  `test/test_audit_btc_v2_artifact_20260716.py`
- Runs: `tmp/causal_regen_20260516/btc_v2_direction_meta_20260716/`
