# BTC CSALT Three-Class Side-Policy Search — 2026-07-15

Status: `predeclared_sixth_development_protocol_not_promotion_artifact`

7-class policy는 가장 좋은 경제성을 보였으나 horizon 희소 클래스가 확률을 분산시켰다.
이 실험은 DP action을 `CASH/LONG/SHORT`로 합쳐 class-balanced 3-class HGB로 증류하고,
실행 action은 LONG_H24/SHORT_H24로 고정한다.

- feature: derived11, BTC-native stationary
- target: normal-cost DP side, 1.5x-cost DP side
- minimum selected-side probability: 0.50, 0.60, 0.70
- selected-side probability minus CASH: 0.00, 0.10
- stress side agreement: off/on
- 기존 5 seed/HGB 설정 유지

총 후보 48개다. T1–T4 각각 양수, aggregate 1.5x-cost 양수, 합계 40거래 이상이어야
freeze하며, 아니면 T5/T6을 열지 않는다.

