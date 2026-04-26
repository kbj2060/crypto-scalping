# Integrated Overlay Design

## Goal

Turn the three live DuckDB feeds into one execution overlay layer that sits on
top of DSAC.

The overlay should not replace DSAC direction. It should answer four narrower
questions:

1. Is entry allowed right now?
2. Should size be scaled up or down?
3. Should the current position be exited early?
4. Should the system enter a temporary cooldown state?

## Data Sources

### 1. Microstructure

Source: `data/live/microstructure.duckdb`

Main table:
- `microstructure_1m`

Primary fields:
- `obi`
- `taker_buy_ratio`
- `nif_whale`
- `eai`
- `oi_delta_pct`
- `funding_rate`
- `kelly_mult`
- `signal_bias`
- `shadow_toxicity_score`
- `shadow_queue_collapse`
- `shadow_absorption_score`
- `shadow_queue_bias`
- `shadow_regime_tag`
- `shadow_regime_conf`

Interpretation:
- Best for entry timing and execution quality.
- Strongest signals are short-horizon flow, toxicity, and queue state.

### 2. Tail Risk

Source: `data/live/tail_risk.duckdb`

Main table:
- `tail_risk_1m`

Primary fields:
- `shadow_aftershock_prob`
- `shadow_decay_half_life`
- `shadow_risk_bucket`
- `long_usd_1m`
- `short_usd_1m`

Interpretation:
- Best for post-shock defense, cooldowns, and forced de-risking.
- This is a state-of-market-risk layer, not a directional alpha layer.

### 3. Polymarket

Source: `data/live/polymarket.duckdb`

Main table:
- `polymarket_markets_10s_json`

Stored form:
- top 5 bucket labels with probabilities every ~10 seconds

Derived fields we should compute from raw rows:
- `weighted_target`
- `weighted_std`
- `mode_label`
- `mode_prob`
- `top2_gap`
- `entropy_norm`
- `support_low`
- `support_high`
- `target_delta_1m`
- `target_delta_3m`

Interpretation:
- Best for event repricing, fair-value anchoring, and confidence filters.
- Not ideal as a standalone exit trigger.

## Design Principle

DSAC remains the primary direction model.

The overlay becomes a thin control system:

- DSAC decides `LONG`, `SHORT`, `HOLD`
- Microstructure decides whether entry timing is tradable
- Polymarket decides whether external event repricing agrees or disagrees
- Tail risk decides whether the environment is temporarily unsafe

This keeps responsibilities clean.

## Overlay Outputs

The integrated overlay should output one compact dict every cycle:

```python
{
    "allow_entry": bool,
    "size_mult": float,
    "exit_now": bool,
    "cooldown_bars": int,
    "entry_score": float,
    "risk_score": float,
    "confidence_score": float,
    "reasons": list[str],
}
```

## Stage 1: Derived Features

### Microstructure Features

Use these normalized fields:

- `ms_flow_align`
  - combines `signal_bias`, `nif_whale`, `taker_buy_ratio`
- `ms_toxicity`
  - from `shadow_toxicity_score`
- `ms_queue_risk`
  - combines `shadow_queue_collapse` and weak absorption
- `ms_regime_conf`
  - from `shadow_regime_conf`

Example:

```text
ms_flow_align = 0.45 * signal_bias + 0.35 * nif_whale + 0.20 * (2*taker_buy_ratio - 1)
ms_queue_risk = 0.6 * shadow_queue_collapse + 0.4 * (1 - shadow_absorption_score)
```

### Tail Risk Features

- `tr_aftershock`
  - from `shadow_aftershock_prob`
- `tr_decay_pressure`
  - normalized from `shadow_decay_half_life`
- `tr_bucket_risk`
  - `normal=0`, `watch=0.5`, `danger=1`

Example:

```text
tr_risk = 0.65 * tr_aftershock + 0.20 * tr_decay_pressure + 0.15 * tr_bucket_risk
```

### Polymarket Features

Compute from raw bucket distribution:

- `poly_gap`
  - `(weighted_target - current_price) / current_price`
- `poly_momentum_1m`
  - change in `weighted_target` over 1 minute
- `poly_momentum_3m`
  - change in `weighted_target` over 3 minutes
- `poly_confidence`
  - `top2_gap * (1 - entropy_norm)`
- `poly_uncertainty`
  - normalized `weighted_std`
- `poly_tail_bias`
  - optional up-mass minus down-mass

## Stage 2: Four Decisions

### A. Entry Gate

Entry should be blocked if any of these is true:

- `tr_aftershock` is high
- `ms_toxicity` is high
- queue collapse is high and absorption is weak
- Polymarket confidence is low and DSAC edge is weak
- Polymarket strongly disagrees with DSAC direction

Example policy:

```text
block entry if
tr_aftershock >= 0.60
or ms_toxicity >= 0.85
or (ms_queue_risk >= 0.75 and abs(ms_flow_align) < 0.10)
or (poly_confidence < 0.08 and abs(dsac_raw) < 0.20)
```

### B. Size Multiplier

Use multiplicative scaling from 0.35x to 1.25x.

Increase size when:

- microstructure agrees with DSAC
- Polymarket gap and momentum agree with DSAC
- tail risk is low

Decrease size when:

- toxicity rises
- aftershock rises
- Polymarket uncertainty widens
- microstructure and Polymarket disagree

Example decomposition:

```text
size_mult =
    base
  * micro_mult
  * poly_mult
  * tail_mult
```

Suggested first version:

- `micro_mult`: `0.75 ~ 1.10`
- `poly_mult`: `0.80 ~ 1.10`
- `tail_mult`: `0.50 ~ 1.00`
- final clip: `0.35 ~ 1.25`

### C. Early Exit

Exit should be rarer than size-down.

Use exit only when:

- tail risk spikes hard
- Polymarket reprices sharply against the current position
- microstructure also flips adverse at the same time

Example:

```text
exit_now if
tr_aftershock >= 0.80
or (
    adverse_poly_repricing
    and ms_flow_flip
    and ms_toxicity >= 0.70
)
```

### D. Cooldown

Cooldown is the correct response after shock clusters.

Instead of chaining repeated exits and re-entries, set a short hold-off:

- `3-6 bars` after medium shock
- `6-12 bars` after strong shock

Cooldown should be driven mostly by tail risk, not by Polymarket alone.

## Stage 3: Scoring Model

Define three intermediate scores:

### Entry Score

```text
entry_score =
    0.50 * align(dsac, ms_flow_align)
  + 0.35 * align(dsac, poly_gap + poly_momentum_1m)
  - 0.15 * ms_toxicity
```

### Confidence Score

```text
confidence_score =
    0.55 * poly_confidence
  + 0.25 * ms_regime_conf
  + 0.20 * abs(ms_flow_align)
```

### Risk Score

```text
risk_score =
    0.50 * tr_aftershock
  + 0.25 * ms_toxicity
  + 0.15 * ms_queue_risk
  + 0.10 * poly_uncertainty
```

Then convert:

- `allow_entry = entry_score > threshold and risk_score < threshold`
- `size_mult = f(confidence_score, risk_score)`
- `exit_now = risk_score very high and adverse alignment`

## Runtime Placement

The overlay should run after DSAC action selection and before final position update.

Recommended runtime order inside `trading_bot.py`:

1. DSAC produces `final_action` and base `kelly`
2. Load latest microstructure snapshot
3. Load latest tail-risk snapshot
4. Load latest Polymarket-derived snapshot
5. Build integrated overlay decision
6. Apply:
   - entry block
   - size multiplier
   - exit override
   - cooldown
7. Write all overlay diagnostics into dashboard state

## Dashboard Contract

Add one compact overlay card instead of exposing raw internals everywhere.

Suggested dashboard payload:

```json
{
  "overlay": {
    "entry_score": 0.42,
    "confidence_score": 0.61,
    "risk_score": 0.28,
    "allow_entry": true,
    "size_mult": 0.92,
    "exit_now": false,
    "cooldown_bars": 0,
    "reasons": ["MS_ALIGN", "POLY_SUPPORT", "TAIL_NORMAL"]
  }
}
```

Also expose compact raw sub-features:

- `micro.flow_align`
- `micro.toxicity`
- `tail.aftershock`
- `poly.gap`
- `poly.momentum_1m`
- `poly.confidence`

## MVP Implementation Plan

### Phase 1

Build a shared helper:
- `features/integrated_overlay.py`

Functions:
- `build_polymarket_overlay_features(...)`
- `build_micro_overlay_features(...)`
- `build_tail_overlay_features(...)`
- `compute_integrated_overlay(...)`

No strategy behavior change yet.
Only compute and log outputs.

### Phase 2

Enable size scaling only:

- no entry veto
- no forced exit
- only `size_mult`

This is the safest first live deployment.

### Phase 3

Add entry veto.

Only block when:

- tail risk high
- microstructure toxic
- Polymarket strongly adverse

### Phase 4

Add early exit and cooldown.

This should be the last stage because it has the most behavioral risk.

## Backtest Plan

Backtest should be split by intervention type:

1. Size-only overlay
2. Entry-veto overlay
3. Exit/cooldown overlay
4. Full integrated overlay

Metrics:

- total pnl
- mdd
- win rate
- trade count
- avg pnl per trade
- pnl on vetoed trades
- pnl on downsized trades
- false exits

We should prefer overlays that improve:

- drawdown
- trade quality

without collapsing:

- trade count
- total pnl

## Recommended First Live Policy

The first live version should be conservative:

- `allow_entry`: only block obvious bad states
- `size_mult`: main control lever
- `exit_now`: only on strong multi-source agreement
- `cooldown`: only tail-risk driven

That means:

- Polymarket should mostly adjust conviction
- Microstructure should mostly adjust timing
- Tail risk should mostly control defense

## Summary

The integrated overlay should follow one simple contract:

- DSAC chooses direction
- Microstructure decides tradability
- Polymarket decides external agreement and fair value
- Tail risk decides whether to de-risk or pause

This keeps the system modular, testable, and much easier to tune than mixing
all signals directly into one ad hoc rule block.
