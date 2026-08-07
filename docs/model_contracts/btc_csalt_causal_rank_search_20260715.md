# BTC CSALT Causal-Rank Gate Search — 2026-07-15

Status: `predeclared_fifth_development_protocol_not_promotion_artifact`

Absolute classifier probability thresholds produced fold activity counts 37/18/1/3. The fifth loop
keeps the same frozen 7-class OOF teacher and changes only activation calibration: at event `t`, compare
the predicted non-CASH side probability with the quantile of predictions observed strictly before `t`.

- teacher: BTC-native `dp_policy` only
- first 100 events: CASH warmup
- causal expanding lookback cap: last `288` or `576` events
- active quantile: `0.80`, `0.85`, `0.90`
- minimum selected-side probability: `0.40`, `0.50`
- stress side agreement gate: off/on
- horizon: selected side에서 action probability가 가장 높은 horizon

총 후보 24개다. 각 event threshold는 현재/미래 row를 포함하지 않는다. 기존과 동일하게
T1–T4 각각 PnL > 0, aggregate 1.5x-cost PnL > 0, 합계 trades >= 40이어야 freeze한다.

