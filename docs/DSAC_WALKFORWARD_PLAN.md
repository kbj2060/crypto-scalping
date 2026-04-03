# DSAC Walk-Forward Plan

## 목적
- `M7`은 계속 `2024` 지도/비지도 학습 전용으로 유지
- `DSAC`만 `2025` 데이터에서 레짐 편향을 점검하고 재학습
- `2024 + 2025` 혼합 K-fold는 당장 보류

## 권장 순서
1. `2025` 전체에서 현재 `best_dsac_agents.pth`의 방향 편향을 측정
2. `2025`를 시간순 walk-forward로 나눠 월별/구간별 편향 확인
3. 편향이 강하면 `DSAC` 보상 또는 샘플링만 수정
4. 그래도 안 되면 `2024 pretrain -> 2025 finetune`

## 2025 Walk-Forward Split
- Fold 1:
  - train: 2025-01 ~ 2025-04
  - val: 2025-05
- Fold 2:
  - train: 2025-01 ~ 2025-05
  - val: 2025-06
- Fold 3:
  - train: 2025-01 ~ 2025-06
  - val: 2025-07
- Fold 4:
  - train: 2025-01 ~ 2025-07
  - val: 2025-08
- Fold 5:
  - train: 2025-01 ~ 2025-08
  - val: 2025-09 이후

## 진단 지표
- flat 상태에서 `LONG / SHORT / HOLD` 결정 비율
- 포지션 체류 시간 비율 `LONG / SHORT / FLAT`
- side별 `mean pnl`, `win rate`, `avg hold`
- 월별 `long_short_entry_ratio`
- 레짐별 `flat_to_long_ratio`, `flat_to_short_ratio`

## 숏 편향 보정 후보
- `reward`에 방향 불균형 패널티 추가
  - 예: 최근 1,000 step에서 `LONG` 비중이 너무 높으면 소폭 감점
- replay sampling에서 `SHORT` 진입/성공 transition 비중 가중
- warmup random action을 상승 편향 없는 대칭 샘플로 유지
- validation 점수에 side balance 항목 추가
  - 한쪽 거래만 잘하는 정책은 best 모델로 채택하지 않음

## 1차 구현 우선순위
- 먼저 진단만 추가
- 다음으로 validation metric에 side balance 반영
- 마지막에만 reward shaping 또는 pretrain 도입
