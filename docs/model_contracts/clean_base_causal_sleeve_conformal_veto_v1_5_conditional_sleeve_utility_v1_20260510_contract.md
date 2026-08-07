# Clean Base Causal Sleeve Conformal Veto V1.5 Conditional Sleeve Utility V1 Contract

Status: `experimental_challenger`

## Purpose

Preserve clean base/Lifecycle core entries and keep the V1.5 causal same-side sleeve/conformal stack, while adding a sleeve-only utility model to veto or half-cap low-quality sleeve additions.

## Runtime Inputs

- Sleeve scorer: current trade state and market context features from `clean_base_plus_causal_conviction_sleeve_v1_1.FEATURES`
- Conformal veto: static trade-entry features from `clean_base_causal_trade_editor_v1_3.EDITOR_FEATURES`
- Utility layer: causal sleeve/conformal/M7/microstructure features listed in the report
- Closed-equity account drawdown and daily drawdown

## Output Invariants

- Core entry, side, exit, notional, and leverage are unchanged.
- The conformal layer cannot open, flip, resize, or close the core trade.
- The conformal layer can only convert an `ADD_SAME_SIDE_*` sleeve action to `CONFORMAL_VETO`.
- The utility layer can only convert `ADD_SAME_SIDE_*` to `UTILITY_VETO` or `UTILITY_HALF`.
- The utility layer cannot create a sleeve when V1.5 selected `NO_SLEEVE` or `CONFORMAL_VETO`.
- OOS threshold selection is forbidden.
