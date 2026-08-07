# Model Data Contracts

Last updated: 2026-05-05 KST

이 디렉터리는 새 모델 아키텍처가 제안될 때마다 Data Architect가 작성하는 데이터/상태 계약서의 기준 위치다. 목적은 Model Architect, Red Team, Implementation Maintainer가 같은 feature, split, artifact, output 정의를 보고 판단하게 만드는 것이다.

## Required Workflow

1. Model Architect가 새 모델 구조를 제안한다.
2. Data Architect가 구현 전에 이 디렉터리에 모델별 계약서를 만든다.
3. Implementation Maintainer는 계약서에 적힌 feature/state/split/output만 코드에 연결한다.
4. Red Team은 계약서의 leakage, cost, live parity, output calibration gate를 기준으로 검증한다.
5. 계약서가 없는 새 모델은 live 후보나 성능 비교 후보로 승격하지 않는다.

## Required Fields

모든 모델 계약서는 최소한 아래 항목을 포함해야 한다.

| Section | Required content |
|---|---|
| Scope | 모델 이름, 목적, status, owner, 관련 실험/스크립트 |
| Dataset Split | train/validation/test 파일, timestamp range, row count, overlap/duplicate audit |
| Shared Feature Contract | canonical feature list, state dim, normalization, missing/stale fallback |
| Layer Contracts | 각 레이어별 input features/state, causal rule, output columns, artifact path |
| Label Contract | target horizon, label 생성 방식, OOF/embargo 여부, future label 격리 |
| Cost/Risk Assumptions | fee, slippage, max notional, leverage cap, funding/liquidation 처리 |
| Output Contract | decision columns, score/prob/value columns, logs/reports, owner attribution |
| Red Team Gates | leakage, fee/slip stress, calibration, walk-forward, live/train parity |
| Open Issues | 구현 전/후 해결해야 할 데이터 계약 리스크 |

## Registry

[registry.json](registry.json)이 최신 계약서를 가리킨다. 다른 서브에이전트는 새 모델을 검토할 때 먼저 registry의 `active_contracts`를 확인한다.

