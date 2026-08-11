# team_workflow_smoke_test 계약 문서

- 폴더: `experiments/team_workflow_smoke_test/`
- 생성일: 2026-08-11 KST
- 팀장 승인: architecture-lead, 2026-08-11 KST

워크플로 정의: [architecture_team_workflow.md](../../docs/subagents/architecture_team_workflow.md)

## 목적

아키텍쳐 팀 승인 워크플로 자체를 검증하는 스모크 테스트다.

`docs/subagents/architecture_team_workflow.md`에 정의된 팀장(architecture-lead) → 실행 팀원(architecture-executor) → 데이터 관리 팀원(architecture-data-manager) 승인 체계가 end-to-end로 실제 동작하는지 확인하는 것이 유일한 목표다. 구체적으로 검증하려는 것:

1. 서브프로젝트 폴더 + 계약 문서 생성 규칙이 실제로 따라지는가.
2. "위임 = 승인" 규칙이 지켜지는가 (승인되지 않은 작업이 실행되지 않는가).
3. 실행 팀원이 위임받은 범위를 스스로 넓히지 않는가.
4. 팀장이 실제 diff를 확인해 검수하는 단계가 작동하는가.
5. 데이터 관리 팀원이 승인된 결과만 계약 문서에 기록하는가.

이 서브프로젝트는 트레이딩 성과나 모델 성능을 목표로 하지 않는다.

## 팀장 리서치 & 계획

- 참고 논문/아이디어: **없음 (해당 없음).** 이번 태스크는 실제 모델 리서치가 아니라 워크플로 점검용이다. 논문 조사, 선행 연구 검토, baseline 비교를 수행하지 않았고 수행할 필요도 없다. 이 섹션을 "리서치를 했다"는 근거로 인용하면 안 된다.
- 모델링 계획: **없음 (해당 없음).** 모델을 학습하거나 평가하지 않는다. 산출물은 트레이딩/모델 파이프라인과 완전히 분리된, 순수 함수 하나 + 그 유닛 테스트뿐이다.
- 워크플로 검증 계획:
  1. 팀장이 `experiments/team_workflow_smoke_test/` 폴더와 이 계약 문서를 만들고, 승인할 작업을 **정확히 하나만** 정의해 아래 승인된 작업 로그에 기록한다.
  2. 그 작업 하나만 `architecture-executor`에게 Agent 툴로 위임한다 (위임 = 승인).
  3. 실행 팀원 보고 후 팀장이 `git status` / `git diff` 로 실제 변경 파일을 직접 확인한다. 범위 이탈(기존 트레이딩/모델 코드 수정, 승인 외 파일 생성)이 있으면 반려한다.
  4. 승인 시 `architecture-data-manager`에게 위임해 결과를 이 문서의 "실행 결과", "리소스/코드 파일 매핑"에 정리하고 "변경 이력"을 갱신하게 한다.
- 범위 제한 (하드 제약):
  - 변경 허용 경로는 `experiments/team_workflow_smoke_test/` 하위뿐이다.
  - 기존 트레이딩/모델/백테스트 코드(`trading_bot.py`, `trading_bot_modules/`, `ensemble/`, `features/`, `scripts/`, `strategies/` 등)는 **절대 건드리지 않는다.**
  - 신규 서드파티 의존성을 추가하지 않는다. 이 레포에는 `pytest`가 설치되어 있지 않으므로 테스트는 stdlib `unittest`로 작성한다.
  - git add / commit / push는 하지 않는다. 사용자가 직접 처리한다.
- 성공 기준:
  - `python -m unittest` 로 유닛 테스트가 전부 통과한다.
  - `git status`에 `experiments/team_workflow_smoke_test/` 밖의 변경이 하나도 나타나지 않는다.
  - 승인된 작업 로그에 없는 작업이 실행된 흔적이 없다.

## 기존 프로젝트 gate와의 관계

이 서브프로젝트는 모델을 학습/평가/승격하지 않고, funding-family 컬럼이나 예측 아티팩트를 일절 소비하지 않는다. 따라서 Omega Artifact Integrity Promotion Gate, Seed-Diversity Ensemble Promotion Gate, Fresh-Forward Validation/OOS/Test Rule, Futures Risk Sizing Contract는 **적용 대상이 아니다 (해당 산출물 없음)**. 이 문서의 어떤 산출물도 promotion evidence로 인용될 수 없다.

`docs/subagents/README.md`의 Shared Rules 중 이번 산출물에 적용되는 항목은 "alias/fallback/legacy compatibility layer 금지"와 "fail-fast" 원칙이며, 순수 함수는 잘못된 입력에 대해 조용히 보정하지 말고 예외를 내야 한다.

## 승인된 작업 로그

| ID | 작업 설명 | 담당 | 승인일 | 상태 |
|---|---|---|---|---|
| SMOKE-1 | `experiments/team_workflow_smoke_test/` 안에 완전히 독립적인 순수 함수 유틸리티 1개(이동평균 교차 판정)와 stdlib `unittest` 기반 최소 유닛 테스트를 추가한다. 허용 신규 파일: `__init__.py`, `ma_cross.py`, `test_ma_cross.py` (모두 이 폴더 하위). 함수는 fast/slow 이동평균 시퀀스를 받아 마지막 bar에서의 교차 상태를 판정하는 순수 함수여야 하며, 외부 I/O·전역 상태·레포 내 다른 모듈 import가 없어야 한다. 길이 불일치/길이 부족 같은 잘못된 입력은 조용히 보정하지 말고 예외로 fail-fast 한다. 기존 트레이딩/모델 코드는 일절 수정 금지, 신규 의존성 추가 금지, git commit 금지. | 실행 팀원 (architecture-executor) | 2026-08-11 | **완료 — 팀장 검수 승인 (2026-08-11)** |
| SMOKE-2 | `experiments/team_workflow_smoke_test/test_ma_cross.py`의 dead `if __name__ == "__main__": unittest.main()` 블록을 해소한다. 현재 line 5의 상대 import(`from .ma_cross import ...`) 때문에 이 블록 경로는 `ImportError`로 항상 실패하므로, 파일이 동작하지 않는 실행 방법을 광고하고 있다. 둘 중 하나만 택한다: (a) `__main__` 블록을 삭제하고 docstring에 정규 실행 명령(`python -m unittest experiments.team_workflow_smoke_test.test_ma_cross`)을 명시, 또는 (b) 상대 import를 제거해 블록이 실제로 동작하게 만든다. 허용 변경 파일은 `test_ma_cross.py` 하나뿐이다. `ma_cross.py` 로직은 건드리지 않는다. 신규 의존성 금지, git commit 금지. | 실행 팀원 (architecture-executor) | 2026-08-11 | 대기 (위임 대기) |

### SMOKE-1 팀장 검수 기록 (2026-08-11)

보고를 그대로 신뢰하지 않고 실제 파일을 직접 읽고 명령을 재실행해 확인했다.

- `ma_cross.py` 교차 판정 로직 정확성 확인. golden은 `is_above`, death는 `is_below`를 각각 요구해 상호배타적이므로 분기 순서에 따른 오판 없음. `prev_fast == prev_slow` 동일값 진입 케이스도 양방향 모두 정상 판정.
- fail-fast 준수 확인. 길이 불일치/2개 미만에서 조용한 보정 없이 `ValueError`. Shared Rules의 alias/fallback 금지, fail-fast 원칙에 부합.
- 순수성 확인. 외부 I/O·전역 상태 없음, 레포 내 타 모듈 import 없음. `typing.Sequence`만 사용.
- 테스트 재실행 확인. `python -m unittest experiments/team_workflow_smoke_test/test_ma_cross.py -v` → 10 tests OK (팀장이 직접 재실행). 점 표기법 `python -m unittest experiments.team_workflow_smoke_test.test_ma_cross`도 10 tests OK.
- 범위 확인. `git status --porcelain` 결과 `experiments/team_workflow_smoke_test/` 외 변경 없음. 기존 트레이딩/모델 코드 무변경. 신규 의존성 없음. 커밋 없음.
- `__pycache__/`는 테스트 실행 부산물이며 `.gitignore:38`의 `__pycache__/`에 이미 포함되므로 범위 위반 아님.

**범위 확장 판정 — 상대 import: 승인 (사후 승인).** SMOKE-1 허용 파일에 `__init__.py`가 명시적으로 포함되어 패키지 의미론이 이미 승인된 것으로 본다. 해당 import는 같은 승인 폴더 안의 형제 파일을 가리키며, 계약이 금지한 "레포 내 다른 모듈 import"에 해당하지 않는다. 신규 의존성도 없다. 실행 팀원이 이를 숨기지 않고 먼저 보고한 점은 워크플로 6번(승인 전 확장 금지)이 의도대로 작동한 사례로 기록한다.

**검수에서 새로 발견된 결함 (실행 팀원 미보고): `test_ma_cross.py:57-58`의 `__main__` 블록이 dead code다.** line 5의 상대 import 때문에 `python experiments/team_workflow_smoke_test/test_ma_cross.py` 직접 실행은 `ImportError: attempted relative import with no known parent package`로 실패한다. SMOKE-1의 명시된 성공 기준(`python -m unittest` 통과, 폴더 밖 변경 없음)은 모두 충족하므로 SMOKE-1 자체는 승인하되, 이 결함은 별도 후속 작업 SMOKE-2로 분리해 승인한다.

## 워크플로 검증 결과

스모크 테스트는 최초 시도에서 **2단계(위임)에서 차단**됐고, 상위 세션이 위임을 대행해 3~4단계를 완료했다. 워크플로 문서와 실제 실행 환경의 불일치는 그대로 남아 있다.

- `docs/subagents/architecture_team_workflow.md` 승인 워크플로 2번과 `.claude/agents/architecture-lead.md` 책임 3번은 팀장이 **Agent 툴**로 `architecture-executor` / `architecture-data-manager`에게 위임하도록 규정한다.
- 그러나 `architecture-lead`가 **서브에이전트로 실행될 때 Agent(서브에이전트 생성) 툴을 부여받지 못한다.** 이 하네스에서 서브에이전트는 다시 서브에이전트를 생성할 수 없다. `.claude/agents/architecture-lead.md`의 `tools: "*"`는 이 제약을 우회하지 못한다.
- 확인한 근거:
  - 팀장 세션의 툴 목록에 Agent/Task 계열 생성 툴이 없다 (`TaskCreate`/`TaskUpdate`/`TaskList`는 할 일 추적용이지 에이전트 위임용이 아니다).
  - `ListAgents` 결과에 `architecture-lead` 자신만 존재한다.
  - `SendMessage(to="architecture-executor")` → `No agent named 'architecture-executor' is reachable.`
- **우회 경로:** 팀장을 호출한 상위 세션이 팀장이 승인 로그에 기록해둔 SMOKE-1을 그 정의 그대로 `architecture-executor`에게 대신 위임했다. 즉 승인 주체(팀장)와 위임 실행 주체(상위 세션)가 분리됐다. 이 우회 덕분에 3~4단계는 검증할 수 있었으나, "위임 = 승인"이라는 워크플로의 핵심 등식은 이 경로에서 성립하지 않는다.

검증된 부분 / 미검증 부분:

| 검증 항목 | 결과 |
|---|---|
| 1. 서브프로젝트 폴더 + 계약 문서 생성 규칙 | 검증됨 (동작) |
| 2. "위임 = 승인" 규칙 | **부분 검증 — 팀장이 직접 위임하는 경로는 부재. 승인(팀장)과 위임(상위 세션)이 분리된 대행 경로로만 성립** |
| 3. 실행 팀원의 범위 준수 | **검증됨 — 준수. 허용 3개 파일 밖으로 나가지 않았고, 계약에 없던 상대 import를 스스로 중단·보고했다** |
| 4. 팀장의 실제 diff 검수 | **검증됨 — 그리고 유효하게 작동. 실행 팀원이 보고하지 않은 dead `__main__` 블록 결함을 검수에서 적발해 SMOKE-2로 분리** |
| 5. 데이터 관리 팀원의 승인분만 기록 | 미검증 (데이터 관리 팀원 미기동 — 상위 세션 대행 위임 대기) |

4번 항목은 이 스모크 테스트에서 가장 의미 있는 결과다. 실행 팀원의 보고("10 tests 전부 OK, 범위 이탈 없음")는 사실이었지만 완전하지 않았고, 실제 파일을 읽어야만 드러나는 결함이 있었다. 워크플로 4번의 "보고를 그대로 믿지 말고 실제 diff를 확인한다"는 요구가 실질적 가치를 가진다는 것이 실증됐다.

## 실행 결과 (데이터 관리 팀원 정리)

| 작업 ID | 변경된 파일 | 요약 | 검증 방법 | 기록일 |
|---|---|---|---|---|
| SMOKE-1 | `experiments/team_workflow_smoke_test/__init__.py` (신규, 빈 파일), `experiments/team_workflow_smoke_test/ma_cross.py` (신규), `experiments/team_workflow_smoke_test/test_ma_cross.py` (신규) | 독립 순수 함수 `detect_ma_cross(fast, slow)` 추가. 마지막 두 bar 비교로 `golden_cross`/`death_cross`/`none` 판정. 길이 불일치·2개 미만은 `ValueError`로 fail-fast. stdlib `unittest` 기반 `TestCase` 10개 동반. 기존 트레이딩/모델 코드 무변경, 신규 의존성 없음 | 팀장이 실제 파일을 직접 읽고 `python -m unittest experiments/team_workflow_smoke_test/test_ma_cross.py -v` 재실행 → 10 tests OK. 점 표기법 실행도 10 tests OK. `git status --porcelain`에 해당 폴더 외 변경 없음 확인. 검수 중 `test_ma_cross.py:57-58` dead `__main__` 블록 결함 적발 → SMOKE-2로 분리 (SMOKE-1은 승인) | 2026-08-11 |

## 리소스/코드 파일 매핑

| 파일 | 용도 | 사용처 | 비고 |
|---|---|---|---|
| `experiments/team_workflow_smoke_test/CONTRACT.md` | 이 서브프로젝트의 팀 작업 계약 문서 | 아키텍쳐 팀 워크플로 | 유일한 산출물. 트레이딩/모델 런타임에서 참조되지 않음 |
| `experiments/team_workflow_smoke_test/__init__.py` | 패키지 마커. `test_ma_cross.py`의 상대 import 성립 조건 | 이 폴더 내부 전용 | 빈 파일 |
| `experiments/team_workflow_smoke_test/ma_cross.py` | 순수 함수 `detect_ma_cross(fast, slow)` — MA 교차 판정 | `test_ma_cross.py`만 사용 | 워크플로 스모크 테스트 전용. 트레이딩/백테스트/모델 경로에서 참조 금지. 레포 내 타 모듈 import 없음 |
| `experiments/team_workflow_smoke_test/test_ma_cross.py` | `detect_ma_cross` 유닛 테스트 10개 | 수동 실행 (`python -m unittest experiments.team_workflow_smoke_test.test_ma_cross`) | CI 미연결. line 57-58 `__main__` 블록은 상대 import로 인해 현재 dead code (SMOKE-2에서 처리 예정) |

## 변경 이력

- 2026-08-11: 최초 생성. 팀장이 목적/계획/범위 제한을 기록하고 SMOKE-1 작업을 승인.
- 2026-08-11: 팀장이 SMOKE-1 위임을 시도했으나 실패. 서브에이전트 실행 컨텍스트에 Agent(서브에이전트 생성) 툴이 없어 `architecture-executor`에 도달 불가. SMOKE-1 상태를 "차단됨 — 위임 불가"로 기록하고 워크플로 검증 결과 섹션 추가. 팀장이 팀원 작업을 대신 수행하지 않음 (테스트 무결성 유지).
- 2026-08-11: 상위 세션이 팀장을 대신해 SMOKE-1을 architecture-executor에게 위임, 실행 완료. 팀장이 실제 파일 직접 확인 후 승인. 상대 import 범위 확장은 사후 승인. 검수 중 발견된 dead __main__ 블록 결함은 SMOKE-2로 분리 승인.
- 2026-08-11: 데이터 관리 팀원이 SMOKE-1 실행 결과와 리소스/코드 파일 매핑을 기록.
