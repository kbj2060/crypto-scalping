# Clean Base Causal Sleeve Conformal Veto V1.5 DSAC Sleeve Layer V1 Contract

Status: `experimental_challenger`

## Purpose

Preserve clean base/Lifecycle core entries and keep the V1.5 causal same-side sleeve/conformal stack, while adding a trained DSAC-style actor head that may only veto or cap a same-side sleeve addition.

## Runtime Inputs

- Sleeve scorer: current trade state and market context features from `clean_base_plus_causal_conviction_sleeve_v1_1.FEATURES`
- Conformal veto: static trade-entry features from `clean_base_causal_trade_editor_v1_3.EDITOR_FEATURES`
- DSAC sleeve layer: selected causal sleeve/conformal/M7/microstructure features listed in the report
- Closed-equity account drawdown and daily drawdown for the original V1.5 sleeve scorer

## Output Invariants

- Core entry, side, exit, notional, and leverage are unchanged.
- The conformal layer cannot open, flip, resize, or close the core trade.
- The conformal layer can only convert an `ADD_SAME_SIDE_*` sleeve action to `CONFORMAL_VETO`.
- The DSAC sleeve layer can only convert `ADD_SAME_SIDE_*` to `DSAC_SLEEVE_VETO` or cap sleeve notional by half.
- The DSAC sleeve layer cannot create a new sleeve when V1.5 selected `NO_SLEEVE` or `CONFORMAL_VETO`.
- OOS threshold selection is forbidden.
