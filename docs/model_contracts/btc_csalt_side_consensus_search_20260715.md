# BTC CSALT Side-Consensus Search Protocol — 2026-07-15

Status: `predeclared_second_development_protocol_not_promotion_artifact`

첫 DP-advantage grid는 864개 후보 모두 T1–T4 합계 20거래 미만이었다. 5개 seed가
side와 horizon을 동시에 맞혀야 하는 exact-action vote가 신호를 소거했다. 두 번째 개발
루프는 이 구조만 수정하며 T5/T6은 계속 봉인한다.

## 고정 구조

각 seed에서 LONG action 3개의 최대 advantage와 SHORT action 3개의 최대 advantage를 먼저
구한다. Seed vote는 exact horizon이 아니라 `CASH/LONG/SHORT` 방향 합의로 계산한다. 방향이
활성화된 뒤 ensemble median score가 가장 큰 24h/72h/168h action을 horizon label로 선택한다.

- feature set: `derived11`, `btc_native_stationary`
- target: `dp_advantage`, `stress15_dp_advantage`
- quantile: `0.25`, `0.50`
- uncertainty penalty: `0.0`, `0.5`
- minimum advantage: `0.0`, `0.00025`, `0.00050`
- minimum side vote: `0.40`, `0.60`
- 1.5x-cost gate: off/on
- seeds와 HGB hyperparameter는 첫 protocol과 동일

총 후보는 feature set당 96개, 전체 192개다. 후보별 T1–T4 label chart를 저장한다.

## 선택 및 봉인 규칙

선택 규칙과 성공 조건은 첫 protocol과 동일하다: T1–T4 각각 PnL > 0, aggregate 1.5x-cost
PnL > 0, 합계 trades >= 40을 모두 만족해야 후보 하나를 freeze한다. 통과 후보가 없으면
T5/T6을 열지 않고 `development_fail`로 종료한다. 결과를 본 뒤 이 grid를 확장하지 않는다.

