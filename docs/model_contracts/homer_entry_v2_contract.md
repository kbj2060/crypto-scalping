# Homer Entry v2 Data Contract

Status: `research_closed — no arm beats F0 (2026-09-04); entry model = F0 economic-label (shadow running)`

Last updated: 2026-09-04 KST

## Scope

- Model id: `homer_entry_v2_econ_label_regime_material_20260904`
- Architecture: TabPFN v2 classifier, 5-seed 18k-row context ensemble (repo-established stack; HGB proxy for smoke/gates). No new DL architecture.
- Purpose: ETH 5m entry model rebuilt on the Homer lineage only (Omega excluded by user decision 2026-09-04). Baseline F0 = V-rebound economic-label model (VAL +3.63 / OOS +7.98 / HOLDOUT +6.09bp, audited A~D clean). Arms add regime classifier (ETH S12_K3, BTC S24_K3, OOF) and evidence-signal material (8 signals, OOF).
- Owner agents: single session (no architect team).
- Implementation script: `scripts/research_homer_entry_v2_20260904.py`
- Report artifact: `tmp/homer_entry_v2_20260904/report_{hgb,tabpfn}.json`, `layer_gates.json`
- Model artifacts: none promoted yet (research); F0 live artifact = `data/labels/eth_5m_v_rebound_econ_label_20260902/tabpfn_train_context_frozen_econ_5seed_20260902.csv`

## Dataset Split

| Split | Source | Timestamp range | Rows | Use |
|---|---|---:|---:|---|
| Train | ETH 5m klines every-bar × 2 sides | 2024-05-01 ~ 2025-08-31 | filled by build stage | fit (18k context sample per seed) |
| Validation | same | 2025-09-01 ~ 2025-12-31 | " | threshold (top 5%) + selection |
| Test/OOS | same | 2026-01-01 ~ 2026-03-31 | " | one look per arm |
| HOLDOUT | ≥ 2026-04-01 | NOT LOADED | — | consumed in this lineage; forward shadow is the test |

Audit:

- Timestamp overlap: none (contiguous splits; label window 200 bars may cross split boundary — reported, not purged, same as F0 protocol)
- Duplicate timestamps: klines de-duplicated on load
- Warmup handling: 2024-01~04 dropped for all arms (regime/material OOF warm-up)
- OOF/embargo: regime and material features are expanding-window OOF (folds 2024-05/09, 2025-01/05; final < 2025-09-01). L3 fails if any TRAIN row carries a `final` source.

## Shared Feature Contract

- Canonical feature source: `research_eth_v_rebound_every_bar_scoring_feasibility_20260901.py::build_all_bar_frame` + `research_eth_v_rebound_label_grid_screen_stage1_20260901.py::long_frame_for` (identical to the live scorer's `_build_features`/`_every_bar_rows` formulas)
- Feature count: F0 23 / F1 29 / F2 48 / F3 54
- Normalization: none (tree/TabPFN)
- Missing fallback: rows with any NaN in F0 dropped (feasibility convention); regime code -1 and inactive material → zeros
- Stale handling: n/a (batch)
- Live availability: F0 live today (econ scorer). Regime live scorer exists (GBM3 endpoint); material live scorer exists (evidence-signal metalabel). A v2 live scorer would need the OOF-equivalent = final models, same as today's deployment.

Feature list: see `tmp/homer_entry_v2_20260904/model_card.json` (`arms`).

## Layer Contracts

| Layer | Input state/features | Train labels | Output | Artifact |
|---|---|---|---|---|
| L4 decision timing | bar τ close | — | entry at open[τ+1] | fills.csv (`fi`,`btf`) |
| L2 label | open[τ+1], ATR(τ), 200 forward bars | `sim_exit` (5.0/1.5/0.1) − 10bp > 0 | y, net_bp | frame.parquet |
| L3 features | Tier0 + regime OOF + material OOF | — | X per arm | bar_features.parquet |
| model | X | y | p (5-seed mean) | report json |
| portfolio | p ≥ VAL top-5% cut, cap 5 | — | trades, bp stats | report json |

## Label Contract

- Horizon: 200 bars (16.7h) max hold; trailing exit
- Cost included: yes, 10bp round trip (taker+taker); maker-entry 7.8bp reported as reference only
- Future path usage: label only (forward window from entry); features use bar τ and earlier
- Leakage controls: layer gates L4/L1/L2/L2P/L3 + T1/T2; OOF for stacked features
- Known limitations: negative skew (SL 5 ATR), bear-market concentration of VAL/OOS, execution sensitivity (F0 robustness: 1~2 bar delay ok, cost 15bp → +3.05bp)

## Cost/Risk Assumptions

- Fee: 5bp per side (taker), no discounts assumed
- Slippage: 0 in label (F0 robustness check: 6bp round-trip slippage still positive)
- Max notional exposure: n/a (research; sizing not designed — negative skew requires it before any real execution)
- Leverage cap: n/a
- Funding: ignored (holds ≤ 16.7h)
- Liquidation/maintenance margin: n/a
- Resize accounting: n/a

## Output Contract

Required decision columns (research report): `timestamp, side, p, entry_bar, exit_bar, pnl_bp`.

## Results

TabPFN 5-seed ensemble, VAL top-5% threshold per arm, sequential portfolio (cap 5), OOS one look per arm
(`tmp/homer_entry_v2_20260904/report_tabpfn.json`; full tables in `docs/experiments/homer_entry_v2_prereg_20260904.md` §8-10).

| Arm | VAL bp | OOS bp | Δ vs F0 OOS (day-cluster CI) | Verdict |
|---|---:|---:|---|---|
| F0 Tier0 23 | +2.63 | +8.33 | — | baseline reproduced (econ model +3.63/+7.98) |
| F1 +regime | +0.15 | +4.06 | −3.44 [−9.5, +2.0] | not better |
| F2 +material | +0.88 | −2.78 | −11.03 [−21.7, −2.1] | significantly worse |
| F3 both | −0.67 | −1.54 | −12.31 [−22.4, −2.3] | significantly worse |

- Multiple testing (4 trials): DSR 0.74 (< 0.95), PBO 0.075.
- Layer gates: L4/L1/L2/L2P/L3/T1 PASS (L2 required declaring the label's trailing convention: `trail_anchor=entry`, `atr_is_absolute=true`);
  T2 FAIL on the 1-minute-reconstructed label for all arms (HGB-regression tail estimate is dtype-fragile) → bp figures are not precise expectations.
- Decision: regime and evidence-signal materials add no out-of-sample increment; entry model stays F0 (already in shadow). No promotion, no grid extension.
