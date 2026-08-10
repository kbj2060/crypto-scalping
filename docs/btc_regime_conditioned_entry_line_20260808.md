# BTC regime-conditioned entry line — 2026-08-08 — CLOSED at OOS

Contract: [btc_regime_conditioned_entry_20260808.json](experiments/btc_regime_conditioned_entry_20260808.json)
(preflight PASS; revised pre-VAL to add the ETH-architecture-analog TabM family).
Runners: `scripts/train_eval_btc_regime_conditioned_entry_20260808.py`,
`scripts/train_eval_btc_regime_tabm_experts_20260808.py`.
Panel: fresh `btc_features_2024_2026_regimeline.csv` (identical FeatureEngineer pipeline,
secondary=ETH), corrected TB labels; both manifest-registered.

Motivation: the 60-symbol screen ranked BTC #1 on within-regime signal magnitude (0.107 mean,
100% bear sign agreement) and every prior BTC failure was an unconditional model.

| stage | result |
|---|---|
| Stage 0 oracle | PASS — VAL 739 trades, 100% TP, +551% |
| Stage R sign gate | PASS 2/3 — bear 75% (+0.073 signed VAL edge), bull 80% (+0.074), chop 25%; trend occupancy only 9.4% |
| Unconditional control | all rules −31…−50% VAL (known BTC failure reproduced) |
| LGBM expert grid | hp_conservative × per-regime-top20 × chop-cash: ALL 6 rules VAL-positive (+2.9…+18.4%), selected side_prob_045 **VAL +18.4%** (79 trades, MDD −4.6%, 3/4 months) |
| TabM expert family (5 seeds) | best +4.69% seed-mean but 1.8/4 positive months — no eligible config |
| **Single frozen OOS read** | **−19.5%** (129 trades, WR 29.5%, MDD −22.8%, monthly −4.8/−10.0/−6.0, 59L/70S) — **FAIL, not adopted** |

## Verdict

The project's 9th selected-positive-flips-on-fresh-data reproduction, and the most damning one:
this candidate had everything the post-mortems said a survivor needs — top-of-universe screened
magnitude, a passed mechanism gate, a beaten asset-matched control, and a whole config family
(not a lone spike) positive on VAL with monthly stability — and it still lost every OOS month.

Interpretation: within-regime sign/magnitude structure measured on train→VAL does not persist
into the NEXT quarter. The regime composition and per-regime feature→direction map keep
reorganizing at the quarterly scale on BTC exactly as on SOL; the 60-way screen's #1 rank was
partly multiple-comparisons flattery, as pre-registered in the contract's own caveat. The
regime-conditioned axis is now closed on BOTH assets. The live BTC swingtransition stack (a
different, multi-day family) is untouched by this result.

Keeps: the BTC regimeline panel + labels (registered, reusable), the Stage R gate result
(the mechanism is real — sign stability holds; it is the quarter-to-quarter persistence of the
magnitude that fails), and the definitive negative on 5m TB regime-conditioning for BTC.
