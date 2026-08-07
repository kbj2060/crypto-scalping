# Omega5 Event-Risk Governor Live Contract

- Model id: `omega5_event_risk_governor_20260702`
- Model version: `Omega5-event-risk-governor-20260702`
- Live role: Omega5 promotion layer on top of the Omega4.6.2 source parent policy.
- Source Omega4.6.2 model: `omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701`
- Source red-team verdict: `FULL_LIVE_PASS_VALIDATION_ONLY`
- Artifact integrity promotion pass: `True`

## Entry

- Parent cash remains cash.
- Parent long uses long exposure factor and no short veto.
- Parent short is vetoed when `bb_width <= 0.003939593535185601`.
- Parent short is also vetoed when `m7_prob_up >= 0.909727596`.
- Scheduled macro entry veto blocks new entries from 30 minutes before to 120 minutes after rule-based NFP/ISM/S&P Global PMI/FOMC events.
- Shock haircut scales new-entry notional by `0.50` when `jump_flag`, `evt_tail_flag`, `abs(jump_z) >= 3.0`, `abs(1h return) >= 3%`, or `abs(4h return) >= 4%` fires.

## Risk

- `reference_notional = min(parent_notional * reference_side_factor, 4.2, 5.0 * 1.0)`.
- `notional = min(reference_notional * final_side_factor, 4.4, 5.0 * 1.0)`.
- `leverage = 5.0`.
- `margin_fraction = notional / leverage`.
- Long TP/SL price moves: `0.020 / 0.030`.
- Short TP/SL price moves: `0.025 / 0.0385`.
- Runtime TP/SL thresholds are account-PnL thresholds: `price_move * notional`.
- Max hold: `8h = 96 five-minute bars`.

## Live Telemetry

- Decision feature-frame DuckDB table: `decision_feature_frame_omega5_event_risk_governor_20260702`.
- The table is model-contract scoped. Schema mismatches must fail instead of
  coercing old and new feature contracts into one table.
