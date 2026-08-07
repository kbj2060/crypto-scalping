# BTC CSALT Policy-Distillation Search Protocol — 2026-07-15

Status: `predeclared_third_development_protocol_not_promotion_artifact`

DP-advantage quantile regression은 exact-action 및 side-consensus 버전 모두 OOF action value를
0 아래로 수축시켰다. 세 번째 개발 루프는 값의 크기를 회귀하지 않고 training history의
finite-SMDP 최적 action을 class-balanced HGB로 직접 증류한다. T5/T6은 계속 봉인한다.

## 고정 후보군

- feature: `derived11`, `btc_native_stationary`
- teacher target: normal-cost DP argmax, 1.5x-cost DP argmax
- 5개 day-block bootstrap seed
- 7-class HGB: depth 3, iterations 100, leaf 40, L2 1.0, lr 0.05, early stopping off
- training sample weight: inverse class frequency, `[0.25, 10]` clip
- minimum predicted side probability: `0.40, 0.50, 0.60`
- side probability minus CASH probability: `0.00, 0.05, 0.10`
- minimum seed side vote: `0.40, 0.60`
- stress gate: off/on. on이면 1.5x-cost teacher도 같은 side를 CASH보다 높게 평가해야 함

Side는 action probability를 LONG/SHORT horizon별로 합산해 선택하고, 선택된 side 안에서
확률이 가장 큰 horizon을 action label로 쓴다. 총 후보 144개, 후보별 T1–T4 chart를 저장한다.

T1–T4 각각 PnL > 0, aggregate 1.5x-cost PnL > 0, 합계 trades >= 40을 모두 만족해야
후보 하나를 freeze한다. 통과 후보가 없으면 `development_fail`이며 T5/T6을 열지 않는다.
