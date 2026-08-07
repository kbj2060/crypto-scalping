# Omega 4.6 Handoff

Date: 2026-06-30 KST

Omega 4.6 is now the current Omega research/upgrade baseline:

- Model id: `omega4_6_plus_t12_nohold_risk1_20260630`
- Contract: `docs/model_contracts/omega4_6_plus_t12_nohold_risk1_20260630_contract.md`
- Runtime contract: `tmp/causal_regen_20260516/omega4_6_plus_t12_nohold_risk1_20260630/runtime_contract.json`
- Promotion manifest: `tmp/causal_regen_20260516/omega4_6_plus_t12_nohold_risk1_20260630/promotion_manifest.json`
- Candidate manifest: `data/ensemble/supervised/omega4_6_plus_t12_nohold_risk1_20260630/candidate_manifest.json`
- Red-team record: `docs/audits/omega4_6_plus_t12_nohold_risk1_redteam_20260630.md`

## Classification

Omega 4.6 is a conditional swing/runner baseline, not a day-trading live-pass
model. Live wiring remains unchanged.

Conditional pass excludes:

- Max hold `24h`
- Validation/OOS PnL target `>= 100%`

Non-excluded gates pass:

- Artifact integrity
- MDD within 20%
- Leverage within 5x
- Notional within 1.8
- No overlap
- Accounting consistency

## Metrics

| Split | PnL | MDD | WR | Trades | Max Hold |
| --- | ---: | ---: | ---: | ---: | ---: |
| Validation | `+117.17%` | `-17.43%` | `51.72%` | `29` | `222.0h` |
| OOS readout | `+67.85%` | `-13.28%` | `53.85%` | `13` | `218.5h` |

## Upgrade Direction

Prioritize upgrades in this order:

1. Preserve the no-hold swing alpha while reducing tail hold time.
2. Test partial TP, trailing giveback, and profit-lock policies instead of a
   blunt 24h forced stop.
3. Improve OOS toward `100%` using validation-only selection and blind OOS
   readout.
4. Keep the artifact integrity gate mandatory for every successor.

## Hard Rules

- Do not treat this as full live/day-trading PASS.
- Do not use OOS for selection.
- Do not use historical trade ledgers as promotion substitutes.
- Do not change the futures sizing contract:
  `notional = margin_fraction * leverage`, `PnL = price_move * notional`.

## Current Conditional Upgrade Candidate

Omega4.6.1 duration OU-halflife risk gate is the current best conditional
upgrade candidate:

- Model id: `omega4_6_1_duration_ou_halflife_risk_gate_20260630`
- Contract: `docs/model_contracts/omega4_6_1_duration_ou_halflife_risk_gate_20260630_contract.md`
- Runtime contract: `tmp/causal_regen_20260516/omega4_6_1_duration_ou_halflife_risk_gate_20260630/runtime_contract.json`
- Candidate manifest: `data/ensemble/supervised/omega4_6_1_duration_ou_halflife_risk_gate_20260630/candidate_manifest.json`
- Artifact audit: `tmp/causal_regen_20260516/omega4_6_1_duration_ou_halflife_risk_gate_20260630/omega_artifact_integrity_audit_20260630.json`

Rule: at entry time, skip exposure when `ou_halflife <= 0.005415348`; otherwise
keep baseline exposure. The rule scales notional/leverage only and does not
change the exit head, TP/SL, or trade path.

Metrics versus Omega4.6 baseline:

| Split | Baseline | Omega4.6.1 |
| --- | ---: | ---: |
| Validation PnL / MDD / trades / max hold | `+117.17% / -17.43% / 29 / 222h` | `+175.86% / -10.60% / 21 / 115.33h` |
| OOS readout PnL / MDD / trades / max hold | `+67.85% / -13.28% / 13 / 218.5h` | `+72.59% / -7.47% / 9 / 133.5h` |

Classification remains conditional upgrade candidate, not live-wired and not
day-trading PASS. Max-hold `24h` and PnL target remain excluded gates. Before
runtime wiring, verify live feature parity for `ou_halflife`; also treat the
trade-count reduction and small validation sample as active risks.
