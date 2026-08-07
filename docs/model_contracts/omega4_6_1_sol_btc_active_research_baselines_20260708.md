# Omega4.6.1 SOL/BTC Active Research Baselines — 2026-07-08

Status: `research_baseline_not_live_wired`.

These are the current SOL/BTC Omega4.6.1 replica baselines after comparing
single-component and ETH-style two-component router tests.

## SOL baseline

Use the previous best exact replay candidate:

- Model: single-component `zig075`
- Quality threshold: `q070` / `0.70`
- Final scale-map: `long_scale=1.0`, `short_scale=2.0`
- Duration gate: `ou_halflife > 0.0055208323`
- Report: `tmp/causal_regen_20260516/sol_final_scale_map_20260707/report.json`

| split | PnL | MDD | trades | WR |
|---|---:|---:|---:|---:|
| VAL | +56.75% | -15.87% | 28 | 42.86% |
| OOS extended | +13.92% | -29.38% | 39 | 38.46% |
| OOS frozen Q1 2026 | +41.98% | -21.03% | 20 | 50.00% |

Rationale: The two-component SOL router produced positive OOS but weak VAL. Keep the cleaner
single-component exact replay as the baseline.

## BTC baseline

Use the single-component exact replay candidate:

- Model: single-component `h48qual`
- Quality threshold: `q055` / `0.55`
- Final scale-map: `long_scale=0.5`, `short_scale=2.5`
- Duration gate: `ou_halflife > 0.00541154875`
- Report: `tmp/causal_regen_20260516/btc_final_scale_map_20260708/report.json`

| split | PnL | MDD | trades | WR |
|---|---:|---:|---:|---:|
| VAL | +12.39% | -6.49% | 10 | 40.00% |
| OOS extended | +29.23% | -10.65% | 24 | 41.67% |
| OOS frozen Q1 2026 | +10.17% | -10.65% | 16 | 37.50% |

Rationale: The BTC two-component router failed OOS. The `zig075` fallback is harmful, so the
single-component `h48qual q055` candidate remains the baseline.

Live wiring: none.
