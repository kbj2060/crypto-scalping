# SOL/BTC Omega4.6.1 Two-Component Router Test — 2026-07-08

Status: `research_diagnostic_not_live_wired`.

Script:

`scripts/replay_omega4_6_1_two_component_router_assets_20260708.py`

This tests the ETH-style greedy router on SOL and BTC:

1. If flat, try `h48qual` first.
2. If `h48qual` is flat, try `zig075`.
3. Hold only one shared position at a time.
4. Exit using the originating component's TP/SL and learned exit-head.
5. Select a VAL-only `ou_halflife` duration gate and apply it once to OOS.

## SOL

Components:

- `h48qual q045`, risk sidecar `sol_omega4_2_trade_risk_sidecar_20260707_h48qual_q045_20260708`, scale `long=1.0`, `short=1.0`
- `zig075 q070`, risk sidecar `sol_omega4_2_trade_risk_sidecar_20260707_zig075_q070_20260707`, scale `long=1.0`, `short=2.0`

Selected duration threshold: `ou_halflife > 0.005520169075`

| split | gate | PnL | MDD | trades | WR |
|---|---|---:|---:|---:|---:|
| VAL | none | -12.56% | -30.48% | 46 | 30.43% |
| VAL | selected | +0.87% | -22.86% | 34 | 35.29% |
| OOS extended | none | +11.00% | -25.37% | 51 | 41.18% |
| OOS extended | selected | +33.04% | -16.18% | 37 | 48.65% |
| OOS Q1 2026 | selected | +37.06% | -9.67% | 17 | 64.71% |

Interpretation: router improves SOL OOS versus no-gate, but VAL is weak and h48qual dominates
the entry slots. Treat as diagnostic; not stronger than a clean single-component candidate on VAL.

## BTC

Components:

- `h48qual q055`, risk sidecar `btc_omega4_2_trade_risk_sidecar_20260708_h48qual_q055_20260708`, scale `long=0.5`, `short=2.5`
- `zig075 q065`, risk sidecar `btc_omega4_2_trade_risk_sidecar_20260708_zig075_q065_20260708`, scale `long=2.5`, `short=2.75`

Selected duration threshold: `ou_halflife > 0.005535701`

| split | gate | PnL | MDD | trades | WR |
|---|---|---:|---:|---:|---:|
| VAL | none | -4.79% | -12.14% | 15 | 33.33% |
| VAL | selected | -0.03% | -7.01% | 7 | 42.86% |
| OOS extended | none | -47.59% | -52.18% | 36 | 25.00% |
| OOS extended | selected | -26.87% | -29.63% | 16 | 25.00% |
| OOS Q1 2026 | selected | -8.28% | -12.63% | 7 | 28.57% |

Interpretation: BTC two-component router fails. The `zig075` fallback is harmful; keep the BTC
single-component `h48qual q055` candidate as the better research path.

Replay flags:

- `fresh_forward_bar_by_bar=true`
- `trade_ledgers_used_as_input=false`
- `saved_parent_exit_timestamps_used=false`
- `future_rows_used_for_entry=false`

Live wiring: none.
