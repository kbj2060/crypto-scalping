# Alpha5 Momentum Feature Drop Contract - 2026-05-21

## Decision

`mom_1d`, `mom_3d`, `mom_21d` and their DSAC normalized state fields are removed from the active Alpha5 Router5 and DSAC feature contracts.

## Removed Fields

Router feature contract:

- `mom_1d`
- `mom_3d`
- `mom_21d`

DSAC state contract:

- `mom_1d_norm`
- `mom_3d_norm`
- `mom_21d_norm`

## Reason

The current routed RL CSVs contain these momentum columns as constant zero fields. Because the values have no variance, feature-importance saliency and permutation tests correctly report zero impact, but that does not represent a useful learned market signal.

Keeping the fields would add unstable behavior later if they are silently regenerated with non-zero values. The safer contract is to remove them now and keep the next `a5dir_*` repair focused on OOF/causal router probabilities.

## Code Scope

Updated files:

- `ensemble/train_rl_dsac_agent.py`
- `scripts/alpha5_router_v2_train_20260519.py`
- `scripts/alpha5_router_v4_4class_train_20260520.py`
- `scripts/alpha5_router_v5_train_20260520.py`
- `scripts/alpha5_direction_router_score_rl_csv_20260519.py`
- `scripts/analyze_dsac_5d_feature_importance_20260520.py`

## Consequences

- Router3/Router4/Router5 feature count drops from 41 to 38.
- Alpha5 DSAC extra state drops by 3 dimensions.
- Existing DSAC checkpoints trained with the old state schema are not compatible with the new state schema.
- Next valid run must regenerate `a5dir_*` with an OOF or causal router scoring path before feature importance or ablation is resumed.
