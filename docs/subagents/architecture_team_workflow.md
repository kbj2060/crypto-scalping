# Architecture Team (아키텍쳐 팀)

Last updated: 2026-08-11 KST

이 문서는 `.claude/agents/`에 정의된 실행형(Claude Code) 아키텍쳐 팀의 구성과, 목적성 서브프로젝트마다 따르는 폴더/계약 문서/승인 워크플로를 정의한다. 각 역할의 상세 스탠딩 컨텍스트(현재 baseline, 계약, gate)는 기존 [Subagents README](README.md)의 서면 역할 문서를 그대로 따른다 — 이 문서는 그 위에 실행 권한/승인 체계를 얹은 것이다.

## 팀 구성

| 역할 | 모델 | Claude Code 정의 | 대응하는 서면 역할 |
|---|---|---|---|
| 팀장 (Lead) | opus5 | [.claude/agents/architecture-lead.md](../../.claude/agents/architecture-lead.md) | [Model Architect](model_architect.md) |
| 실행 팀원 (Executor) | sonnet5 | [.claude/agents/architecture-executor.md](../../.claude/agents/architecture-executor.md) | [Backtest Implementation Maintainer](implementation_maintainer.md) |
| 데이터 관리 팀원 (Data Manager) | sonnet5 | [.claude/agents/architecture-data-manager.md](../../.claude/agents/architecture-data-manager.md) | [Docs Manager](docs_manager.md), [Data Architect](data_architect.md) |

- **팀장**은 아이디어 뱅크다. 최신 논문/아이디어를 조사하고 모델링 계획을 세운다. 모든 팀원 작업은 팀장 승인 후에만 실행된다.
- **실행 팀원**은 팀장이 승인한 작업만 받아 코드를 작성한다. 스스로 범위를 넓히지 않는다.
- **데이터 관리 팀원**은 기존 리소스/코드 파일이 어디서 어떻게 쓰이는지 정리하고 유지보수하기 쉽게 지속적으로 최적화·관리한다. 팀장이 승인한 작업 결과를 받아 계약 문서에 정리한다.

새 팀원이 추가되면 이 표와 `.claude/agents/`의 해당 정의를 함께 갱신한다.

## 서브프로젝트 폴더 + 계약 문서 규칙

- 목적이 있는 새 서브프로젝트를 시작할 때마다 `experiments/<subproject_name>/` 폴더를 만든다 (레포의 기존 `experiments/*` 관례를 따른다).
- 폴더가 만들어지면 팀이 구성되어 계약 문서 `experiments/<subproject_name>/CONTRACT.md`를 만든다. 템플릿은 [CONTRACT_TEMPLATE.md](CONTRACT_TEMPLATE.md).
- 계약 문서에는 목적, 팀장의 리서치/계획, 승인된 작업 로그, (데이터 관리 팀원이 정리한) 실행 결과, 리소스/코드 파일 매핑, 변경 이력을 담는다.
- 기존 `docs/model_contracts/`의 모델 아티팩트 계약과는 별개다 — 그쪽은 승격된 모델의 데이터/아티팩트 계약이고, 이쪽은 서브프로젝트 단위의 팀 작업 계약이다.

## 승인 워크플로

1. 팀장이 목적을 리서치하고 계획을 세운 뒤, 계약 문서에 계획과 작업 항목을 기록한다.
2. 팀장은 승인한 작업만 실행 팀원 또는 데이터 관리 팀원에게 위임한다 — 위임 자체가 승인 행위다. 팀원은 위임받지 않은 범위로 스스로 확장하지 않는다.
3. 실행 팀원은 작업을 완료하면 변경된 파일과 검증 내용을 팀장에게 보고한다.
4. 팀장이 결과를 검토(실제 diff 확인 포함)해 승인하면, 승인된 결과를 데이터 관리 팀원에게 전달한다.
5. 데이터 관리 팀원은 승인된 결과만 계약 문서에 정리하고, 관련 리소스/코드 파일 매핑을 최신화한다. 팀장 승인 없이 스스로 판단한 결과를 기록하지 않는다.
6. 실행/데이터 관리 팀원이 계획에 없던 추가 작업이 필요하다고 판단하면, 먼저 팀장에게 승인을 요청하고 승인 전에는 실행하지 않는다.

## 기존 프로젝트 gate와의 관계

아키텍쳐 팀의 모든 산출물은 이 레포의 기존 gate를 그대로 따른다:

- `.claude/CLAUDE.md`의 Omega Artifact Integrity Promotion Gate, Seed-Diversity Ensemble Promotion Gate, Fresh-Forward Validation/OOS/Test Rule, Futures Risk Sizing Contract.
- [Subagents README](README.md)의 Shared Rules (fail-fast, alias/fallback 금지, funding-clean 등).

팀장은 이 gate를 위반하는 산출물을 승인하지 않는다.
