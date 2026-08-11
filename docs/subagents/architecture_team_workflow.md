# Architecture Team (아키텍쳐 팀)

Last updated: 2026-08-11 KST

이 문서는 `.claude/agents/`에 정의된 실행형(Claude Code) 아키텍쳐 팀의 구성과, 목적성 서브프로젝트마다 따르는 폴더/계약 문서/승인 워크플로를 정의한다. 각 역할의 상세 스탠딩 컨텍스트(현재 baseline, 계약, gate)는 기존 [Subagents README](README.md)의 서면 역할 문서를 그대로 따른다 — 이 문서는 그 위에 실행 권한/승인 체계를 얹은 것이다.

## 팀 구성

| 역할 | 모델 | Claude Code 정의 | 대응하는 서면 역할 |
|---|---|---|---|
| 팀장 (Lead) | opus5 | [.claude/agents/architecture-lead.md](../../.claude/agents/architecture-lead.md) | [Model Architect](model_architect.md) |
| 실행 팀원 (Executor) | sonnet5 | [.claude/agents/architecture-executor.md](../../.claude/agents/architecture-executor.md) | [Backtest Implementation Maintainer](implementation_maintainer.md) |
| 데이터 관리 팀원 (Data Manager) | sonnet5 | [.claude/agents/architecture-data-manager.md](../../.claude/agents/architecture-data-manager.md) | [Docs Manager](docs_manager.md), [Data Architect](data_architect.md) |

- **팀장**은 아이디어 뱅크다. 최신 논문/아이디어를 조사하고 모델링 계획을 세운다. 작업을 계약 문서에 제안하고, 실행 팀원의 결과를 실제 파일 기준으로 기술 검수한다.
- **실행 팀원**은 사용자가 승인한 작업만 받아 코드를 작성한다. 스스로 범위를 넓히지 않는다.
- **데이터 관리 팀원**은 기존 리소스/코드 파일이 어디서 어떻게 쓰이는지 정리하고 유지보수하기 쉽게 지속적으로 최적화·관리한다. 승인되고 팀장 검수를 통과한 작업 결과를 받아 계약 문서에 정리한다.

새 팀원이 추가되면 이 표와 `.claude/agents/`의 해당 정의를 함께 갱신한다.

## 알려진 실행 환경 제약 (중요)

`.claude/agents/architecture-lead.md`로 정의된 팀장은 **Claude Code 서브에이전트로 실행되면 Agent(서브에이전트 생성) 툴 자체를 받지 못한다.** 이 하네스는 서브에이전트가 다시 서브에이전트를 만드는 재귀적 위임을 막기 때문에, 팀장이 `architecture-executor`/`architecture-data-manager`를 직접 호출해 위임하는 것은 구조적으로 불가능하다. `tools: "*"`를 줘도 이 제약은 우회되지 않는다.

`experiments/team_workflow_smoke_test/`에서 이 문제를 실제로 검증했다: 팀장이 `SendMessage(to="architecture-executor")`를 시도했으나 "No agent named 'architecture-executor' is reachable" 오류가 났고, `ListAgents`에도 팀장 자신만 보였다.

이 제약 때문에 승인의 정본(source of truth)은 **"팀장의 Agent 툴 호출"이 아니라 "계약 문서의 승인된 작업 로그 기재"**로 정의한다. 실제 위임(Agent 툴 호출)은 팀을 운영하는 세션 — 사용자를 대리해 Agent 툴을 가진 쪽(사용자 본인 세션 또는 그 세션의 Claude) — 이 수행한다. 팀장은 리서치·계획·제안·기술 검수를 맡고, **실행 승인(이 작업을 실제로 위임해도 되는지)은 사용자**가 한다.

## 서브프로젝트 폴더 + 계약 문서 규칙

- 목적이 있는 새 서브프로젝트를 시작할 때마다 `experiments/<subproject_name>/` 폴더를 만든다 (레포의 기존 `experiments/*` 관례를 따른다).
- 폴더가 만들어지면 팀이 구성되어 계약 문서 `experiments/<subproject_name>/CONTRACT.md`를 만든다. 템플릿은 [CONTRACT_TEMPLATE.md](CONTRACT_TEMPLATE.md) — 복사한 뒤 워크플로 링크 상대경로를 `../../docs/subagents/architecture_team_workflow.md`로 유지한다.
- 계약 문서에는 목적, 팀장의 리서치/계획, 승인된 작업 로그, (데이터 관리 팀원이 정리한) 실행 결과, 리소스/코드 파일 매핑, 변경 이력, 적용 gate를 담는다.
- 기존 `docs/model_contracts/`의 모델 아티팩트 계약과는 별개다 — 그쪽은 승격된 모델의 데이터/아티팩트 계약이고, 이쪽은 서브프로젝트 단위의 팀 작업 계약이다.

## 승인 워크플로

1. 팀장이 목적을 리서치하고 계획을 세운 뒤, 계약 문서 승인된 작업 로그에 작업을 **"제안됨"** 상태로 기록한다.
2. **사용자가 제안된 작업을 검토하고 실행을 승인한다.** 팀장은 스스로 위임할 수 없으므로, 사용자의 승인이 곧 실행 authorize다.
3. 사용자 승인 후, 팀을 운영하는 세션이 승인된 작업만 정확히 그 정의대로 실행 팀원(또는 데이터 관리 팀원)에게 위임한다. 팀원은 위임받지 않은 범위로 스스로 확장하지 않는다.
4. 실행 팀원은 작업을 완료하면 변경된 파일과 검증 내용을 보고한다.
5. 팀장이 **실제 파일/diff를 직접 읽어** 기술 검수한다 — 팀원의 요약 보고를 그대로 승인 근거로 쓰지 않는다. 문제나 누락을 발견하면 반려하거나, 별도의 신규 작업으로 분리해 계약 문서에 "제안됨" 상태로 추가한다.
6. 검수를 통과하면, 팀을 운영하는 세션이 데이터 관리 팀원에게 위임해 계약 문서의 "실행 결과" / "리소스·코드 파일 매핑" / "변경 이력"을 정리하게 한다. 데이터 관리 팀원은 팀장이 검수 승인한 내용만 기록하고, 스스로 판단한 내용을 추가하지 않는다.
7. 팀원이 계획에 없던 추가 작업이 필요하다고 판단하면, 팀장이 검토해 신규 작업으로 계약 문서에 제안하고, 다시 2번(사용자 승인)부터 반복한다.

## 여러 대화 세션에 걸쳐 진행하기

한 서브프로젝트를 여러 대화 세션(다른 날, 다른 컨텍스트, 심지어 다른 컨테이너)에 나눠 진행해도 된다 — 오히려 권장한다. 팀장/실행/데이터 관리 팀원은 스폰될 때마다 이전 대화를 전혀 기억하지 못하고 항상 계약 문서와 관련 `docs/`를 다시 읽고 시작하도록 설계되어 있으므로, 상태의 정본은 처음부터 대화 메모리가 아니라 `experiments/<name>/CONTRACT.md` 파일이다. 한 세션에 리서치+실행+검수+기록을 전부 몰아넣으면 컨텍스트만 불어나 느려질 뿐이니, 승인된 작업 로그의 작업 ID 단위(예: SMOKE-1, SMOKE-2)로 세션을 쪼개는 편이 낫다.

세션을 나눌 때 지킬 것:

1. **세션을 끝내기 전에 커밋 + 푸시한다.** 원격 세션은 매번 브랜치를 새로 클론해서 시작하므로, 푸시하지 않은 변경은 다음 세션에서 아예 보이지 않는다.
2. **새 세션은 이어서 하는 게 아니라 계약 문서를 다시 읽고 판단한다.** 승인된 작업 로그의 상태 컬럼(제안됨/승인됨/진행중/완료/반려됨/취소됨)이 정확하면 어느 세션에서 봐도 같은 결론이 나온다. 상태 갱신을 누락한 채 세션을 끝내지 않는다.
3. **같은 서브프로젝트를 동시에 여러 세션에서 돌리지 않는다.** 병렬로 진행하려면 서로 다른 작업 ID를 맡겨 파일 충돌과 승인 로그 경합을 피한다.

## 기존 프로젝트 gate와의 관계

아키텍쳐 팀의 모든 산출물은 이 레포의 기존 gate를 그대로 따른다:

- `.claude/CLAUDE.md`의 Omega Artifact Integrity Promotion Gate, Seed-Diversity Ensemble Promotion Gate, Fresh-Forward Validation/OOS/Test Rule, Futures Risk Sizing Contract.
- [Subagents README](README.md)의 Shared Rules (fail-fast, alias/fallback 금지, funding-clean 등).

이런 gate가 서브프로젝트에 적용되지 않는 경우(예: 모델을 학습/평가/승격하지 않는 순수 워크플로 점검), 계약 문서의 "적용 gate" 섹션에 적용 대상이 아닌 이유를 명시한다. 팀장은 이 gate를 위반하는 산출물을 승인하지 않는다.
