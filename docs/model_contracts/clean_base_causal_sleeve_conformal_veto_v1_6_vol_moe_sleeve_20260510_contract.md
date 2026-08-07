# Clean Base Causal Sleeve Conformal Veto V1.6 Volatility-MoE Sleeve Contract

Status: `experimental_challenger`

## Purpose

Preserve clean base/Lifecycle core entries while replacing the single V1.5 sleeve scorer with causal volatility/tail/liquidity/funding bucket experts. The V1.5 conformal veto remains the final sleeve-only risk layer.

## Runtime Inputs

- Sleeve scorer: current trade state and market context features from `clean_base_plus_causal_conviction_sleeve_v1_1.FEATURES`
- Conformal veto: static trade-entry features from `clean_base_causal_trade_editor_v1_3.EDITOR_FEATURES`
- MoE router: causal bucket features from side, tail risk, liquidity, volatility, and funding pressure
- Closed-equity account drawdown and daily drawdown

## Output Invariants

- Core entry, side, exit, notional, and leverage are unchanged.
- The MoE router only chooses which sleeve scorer expert evaluates a candidate.
- The conformal layer cannot open, flip, resize, or close the core trade.
- The conformal layer can only convert an `ADD_SAME_SIDE_*` sleeve action to `CONFORMAL_VETO`.
- OOS threshold selection is forbidden.
