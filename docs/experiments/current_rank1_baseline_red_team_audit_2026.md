# Current Rank-1 Baseline Red-Team Audit

Date: 2026-05-06 KST

Verdict: `unsafe_for_live_shadow_only`

## Audited Baseline

Model id: `current_top_muzero_az_stage2_azexit_2026`

Active path:

```text
MuZero Entry
AZ Risk Overlay
Stage2 MuZero Sleeve: gamma=0.55, prior=0.00, depth=1, score_floor=0.12
AZ Exit Governor: threshold=0.45
backtest_no_limit_exit accounting
```

Stage3 and Stage4 are excluded from the active path.

## Local Audit Artifacts

- Script: `scripts/audit_current_rank1_baseline_safety_2026.py`
- JSON report: `data/ensemble/reports/current_rank1_baseline_red_team_audit_2026.json`

Pinned artifact hashes:

| Artifact | SHA256 |
|---|---|
| `mz_latent_governor.pt` | `b1efe0c940615835d7106438b00d8619d2d960d37c1ae8691dcc027bb2386209` |
| `az_risk_overlay.pt` | `320859d68698290513d6201c9e0675a10247469b94c047c678725db563c34be0` |
| `wf_stage2_sleeve_mz.pt` | `442a9dae1b46e94c4bd93d7123c437483d62e17590804337e010008988ae64b4` |
| `az_policy_value_governor.pt` | `24eb6f51eeb8c6d12109e5a09167e02bfd8cfae629e35ddc3cf72504f3061f0b` |

## What Passed

| Check | Result |
|---|---|
| Baseline reproduction | pass |
| Stage3/Stage4 exclusion in audited path | pass |
| Decision invariants | pass |
| Timestamp overlap train/eval | pass |
| Cost 1x | `752.648580%` |
| Cost 2x | `279.363854%` |
| Cost 3x | `75.840037%` |
| Weekly negative periods | `0` |

Reproduced full OOS:

| Metric | Value |
|---|---:|
| PnL | `752.648580%` |
| MDD | `-18.755787%` |
| Trades | `353` |
| Trades/day | `6.017045` |
| Avg notional | `1.425435` |
| Avg leverage | `1.596029` |

## Blocking Findings

1. Live verdict is blocked by missing realism in accounting:
   - funding cost is not applied
   - liquidation and maintenance margin are not modeled
   - order book depth, partial fills, and market impact are not modeled beyond constant slippage
   - trade-level ledger and full equity curve are not returned by the backtest function

2. Stress fragility appears above 3x cost:
   - cost 4x PnL: `-20.760508%`
   - cost 5x PnL: `-31.401338%`

3. Stage4 report ambiguity remains a production risk:
   - the walkforward report contains the Stage2 metrics used here
   - the same report's final decision points at a Stage4 candidate with worse OOS
   - production loaders must fail closed if Stage3/Stage4 artifacts are selected for this baseline

4. There is OOS selection risk:
   - Stage3/Stage4 were evaluated and then excluded after seeing 2026 OOS degradation
   - live promotion needs untouched holdout or forward shadow evidence

5. The contract mentions resize/reverse accounting, but the audited path enters only while flat and exits through the exit governor. In-position resize/reverse accounting invariants are therefore not proven for this baseline.

## Red-Team Category

```text
Safe for live: no
Shadow only: yes, after identity and loader guards are enforced
Unsafe for live: yes in the current state
```

## Required Before Live

- Pin artifact hash manifest and make the loader verify it.
- Add a loader guard that hard-fails on Stage3/Stage4/regime/exit-arbiter artifacts for this baseline.
- Add funding, liquidation, maintenance margin, partial fill, and order book impact simulation.
- Add trade-level ledger and equity curve audit.
- Run forward paper/shadow with exchange-style fills, funding, latency, rejected orders, partial fills, and kill switch telemetry.
