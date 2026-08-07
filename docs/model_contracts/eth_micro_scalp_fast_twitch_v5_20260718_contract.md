# ETH Micro-Scalp Fast-Twitch Opportunity-MoE v5 Contract

## Problem being corrected

The fit teacher's stateful inventory path has a five-minute median holding time
and a 20–22 minute p95, while v4 holds for multiple hours. The mismatch is a
model response problem, not a fixed-horizon teacher problem. v4 also removes a
seed using tune, which is not retained in v5.

## Architecture change

v5 keeps the exact 36 base and 24 microstructure source-stable feature contract
and freezes both 60-minute causal encoders. The downstream regime gate, three
experts per seed, inventory-Q, continuation, exit-hazard, auxiliary heads, and
the new fast-twitch adapter are trainable on fit only. This preserves the
source representation while allowing the three seeds' action coordinates to
align.

A causal fast-twitch head receives only:

- the current scaled 60-feature vector;
- its one-minute change;
- its five-minute change.

The 180 inputs pass through a 64-unit normalized MLP and produce a shared 3x3
inventory/action Q residual. The final layer is initialized to zero, so the
untrained v5 model must be exactly equivalent to v4. The residual is added to
both mixed Q and each expert Q before stateful policy evaluation.

Switch targets receive twice the action-loss weight of hold targets. This is a
training loss only; no fixed holding time, maximum holding time, cooldown, or
time-based liquidation is introduced.

An intermediate joint-adaptation experiment was rejected after consumed outer
diagnostics showed that changing the full v4 representation was unstable. A
second fully frozen-parent experiment was rejected because tune selected the
disabled policy. The final frozen-encoder architecture is therefore
research-only; historical outer intervals are not fresh evidence for it. They
are reported as diagnostics and a post-freeze fresh-forward interval remains
mandatory.

## Selection and safety

All three trained seeds are used. Fit supplies teacher targets and training.
Tune alone selects switching margin, consensus, uncertainty, and any opportunity
overlay. Historical validation/development are evaluated once as consumed
diagnostics and cannot change v5.

The artifact execution policy is always disabled and
`activation_allowed=false`. No live, paper, or simulated order capability is
present. A new post-v5-freeze exact-source fresh-forward interval is required
before any promotion consideration.
