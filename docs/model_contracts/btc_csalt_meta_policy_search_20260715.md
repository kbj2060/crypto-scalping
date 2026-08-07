# BTC CSALT Two-Stage Meta-Policy Search — 2026-07-15

Status: `predeclared_fourth_development_protocol_not_promotion_artifact`

7-class policy distillation은 거래를 복원했지만 확률 임계값 0.60에서 fold별 활동량이
37/18/1/3으로 붕괴했다. DP training target의 활성 action 대부분은 H24였고 H168은 사실상
없었다. 네 번째 루프는 실행 여부와 방향을 분리하는 표준 meta-label 구조를 쓴다.

1. natural-prevalence binary model: DP 최적 action이 CASH가 아닌지 예측
2. active row 전용 balanced binary model: LONG/SHORT 방향 예측
3. horizon은 고정 H24

고정 grid:

- feature: `derived11`, `btc_native_stationary`
- target: normal-cost DP policy, 1.5x-cost DP policy
- minimum active probability: `0.30, 0.40, 0.50`
- minimum side probability: `0.55, 0.65`
- stress gate: off/on
- 5개 day-block seed, HGB depth 3/100 trees/leaf 40/L2 1/lr 0.05/early stopping off

총 후보 48개다. T1–T4 각각 PnL > 0, aggregate 1.5x-cost PnL > 0, 합계 trades >= 40을
모두 만족해야 freeze한다. 아니면 T5/T6을 열지 않는다.

