# Clean Base Causal Sleeve Conformal Veto V1.5 Contract

Status: `experimental_challenger`

## Purpose

Preserve clean base/Lifecycle core entries and keep the causal same-side sleeve alpha, while using validation-calibrated downside uncertainty only to veto sleeve additions.

## Runtime Inputs

- Sleeve scorer: current trade state and market context features from `clean_base_plus_causal_conviction_sleeve_v1_1.FEATURES`
- Conformal veto: static trade-entry features from `clean_base_causal_trade_editor_v1_3.EDITOR_FEATURES`
- Closed-equity account drawdown and daily drawdown

## Output Invariants

- Core entry, side, exit, notional, and leverage are unchanged.
- The conformal layer cannot open, flip, resize, or close the core trade.
- The conformal layer can only convert an `ADD_SAME_SIDE_*` sleeve action to `CONFORMAL_VETO`.
- OOS threshold selection is forbidden.
