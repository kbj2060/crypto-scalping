# MODEL_NAME Data Contract

Status: `draft`

Last updated: YYYY-MM-DD KST

## Scope

- Model id:
- Architecture:
- Purpose:
- Owner agents:
- Implementation script:
- Report artifact:
- Model artifacts:

## Dataset Split

| Split | Source | Timestamp range | Rows | Use |
|---|---|---:|---:|---|
| Train |  |  |  |  |
| Validation |  |  |  |  |
| Test/OOS |  |  |  |  |

Audit:

- Timestamp overlap:
- Duplicate timestamps:
- Warmup handling:
- OOF/embargo:

## Shared Feature Contract

- Canonical feature source:
- Feature count:
- Normalization:
- Missing fallback:
- Stale handling:
- Live availability:

Feature list:

```text
PASTE_FEATURES_HERE
```

## Layer Contracts

| Layer | Input state/features | Train labels | Output | Artifact |
|---|---|---|---|---|
|  |  |  |  |  |

## Label Contract

- Horizon:
- Cost included:
- Future path usage:
- Leakage controls:
- Known limitations:

## Cost/Risk Assumptions

- Fee:
- Slippage:
- Max notional exposure:
- Leverage cap:
- Funding:
- Liquidation/maintenance margin:
- Resize accounting:

## Output Contract

Required decision columns:

```text
action
side
notional_exposure
leverage
position_fraction
quality_score
confidence
```

Required report metrics:

```text
pnl
mdd
trades
trades_per_day
wr
avg_notional
avg_leverage
monthly
cost_stress
```

## Red Team Gates

- [ ] Train/validation/test timestamp overlap audit is zero.
- [ ] No bfill/full-sample scaler/future feature enters live state.
- [ ] Fee/slippage 1x/2x/3x ranking is reported.
- [ ] Score/probability buckets are calibrated against realized net PnL.
- [ ] Monthly/weekly walk-forward is reported.
- [ ] Live train state parity is checked.
- [ ] Funding/liquidation limitations are documented.

## Open Issues

- 

