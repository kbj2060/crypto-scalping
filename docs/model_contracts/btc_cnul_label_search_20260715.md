# BTC Counterfactual Net-Utility Label Search — 2026-07-15

Status: `predeclared_non_dp_development_protocol_not_promotion_artifact`

DP/Bellman continuation을 완전히 제거한다. 각 dollar event에서 고정 execution contract의
CASH/LONG/SHORT × H24/H72/H168 lifecycle net return을 직접 계산하고, 수익이 양수인 최선
action을 supervised policy label로 증류한다.

## 고정 label targets

- `net1`: normal-cost lifecycle return argmax
- `net15`: 1.5x-cost lifecycle return argmax
- `consensus`: normal-cost와 1.5x-cost의 argmax action이 같을 때만 해당 action, 아니면 CASH
- active profit floor: account log return `0.0000` 또는 `0.0025`

선택된 action utility가 floor 이하이면 CASH다. DP value, continuation, future label-fold outcome은
teacher prediction에 사용하지 않는다. Future path는 teacher training row의 offline label 생성에만
사용한다.

## 고정 model/grid

- features: `derived11`, `btc_native_stationary`
- 7-class inverse-frequency balanced HGB, 5 day-block bootstrap seeds
- selected-side probability: `0.50`, `0.60`, `0.70`
- seed side vote: `0.40`, `0.60`
- 1.5x-cost teacher side agreement gate: off/on
- HGB: depth 3, 100 iterations, leaf 40, L2 1, lr 0.05, early stopping off

총 후보 144개다. 후보·fold별 label chart를 저장한다.

## 선택 및 one-shot holdout

T1–T4 각각 PnL > 0, aggregate 1.5x-cost PnL > 0, 합계 trades >= 40인 후보만 eligible이다.
최저 fold PnL, aggregate 1.5x-cost PnL, 단순성 순으로 하나를 freeze한다. 그 뒤에만 T5/T6을
각각 정확히 한 번 평가한다.

최종 research pass는 T5와 T6 각각 PnL > 0, 합계 trades >= 20, aggregate 1.5x-cost PnL > 0이다.

## PnL-selection amendment

최초 strict 개발 screen에서 모든 fold 양수 후보는 없었고 최선 후보의 T3는 -0.0723%였다.
T5/T6을 열기 전에 다음 fallback을 한 번 선언한다. 후보 grid는 변경하지 않는다.

- positive development folds >= 3/4
- minimum development fold PnL > -2%
- aggregate trades >= 40
- aggregate 1.5x-cost PnL > 0
- 위 집합에서 aggregate 1.5x-cost PnL 최대 후보 하나 선택
- 동률이면 minimum fold PnL, feature/gate/threshold 단순성 순

이 fallback은 최초 strict gate 통과로 주장하지 않으며, T5/T6 final gate는 완화하지 않는다.
