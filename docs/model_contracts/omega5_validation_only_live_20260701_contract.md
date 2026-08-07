# Omega5 Validation-Only Live Contract

- Model id: `omega5_validation_only_live_20260701`
- Model version: `Omega5-validation-only-live-20260701`
- Live role: Omega5 promotion layer on top of the Omega4.6.2 source parent policy.
- Source Omega4.6.2 model: `omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701`
- Source red-team verdict: `FULL_LIVE_PASS_VALIDATION_ONLY`
- Artifact integrity promotion pass: `True`

## Entry

- Parent cash remains cash.
- Parent long uses long exposure factor and no short veto.
- Parent short is vetoed when `bb_width <= 0.003939593535185601`.
- Parent short is also vetoed when `m7_prob_up >= 0.909727596`.

## Risk

- `reference_notional = min(parent_notional * reference_side_factor, 4.2, 5.0 * 1.0)`.
- `notional = min(reference_notional * final_side_factor, 4.4, 5.0 * 1.0)`.
- `leverage = 5.0`.
- `margin_fraction = notional / leverage`.
- Long TP/SL price moves: `0.020 / 0.030`.
- Short TP/SL price moves: `0.025 / 0.0385`.
- Runtime TP/SL thresholds are account-PnL thresholds: `price_move * notional`.
- Max hold: `8h = 96 five-minute bars`.
