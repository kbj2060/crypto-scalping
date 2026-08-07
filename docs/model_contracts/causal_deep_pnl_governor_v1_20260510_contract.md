# Causal Deep PnL Governor V1 Contract

- Legacy regime_v2/segment/event-detector/outcome columns are forbidden.
- Clean 2024-only unsupervised regime prediction features are allowed when present.
- Selection uses late-2025 validation only; 2026 is evaluated once after final 2025 retraining.
- The model directly predicts action, notional, leverage, TP, SL, max-hold, and cooldown buckets.
