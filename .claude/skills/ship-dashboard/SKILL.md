---
name: ship-dashboard
description: dashboard/ 변경을 main 에 올려 배포하고 공개 URL 의 서빙 바이트로 확인한다 — 렌더 확인 → 캐시버스터 → CI 문법 → 커밋 → drift 게이트 → 푸시 → 워처 → curl 검증. 사용자가 «배포», «바로 배포», /ship-dashboard 로 부를 때만.
disable-model-invocation: true
---

# 대시보드 배포 (커밋 → 게이트 → 푸시 → 워처)

근거: CLAUDE.md «Deploy/Git Two-Channel Conflict Contract», 메모리 `feedback_deploy_order_is_commit_then_push_20260916` · `feedback_dashboard_deploy_cache_buster_and_verify_served_content` · `feedback_run_ci_js_syntax_check_before_push`.
🔴**`handoff.sh push` 로 먼저 올리지 않는다.** 커밋보다 rsync 가 먼저면 드리프트를 내가 만든다. 배포는 워처가 한다.

## 0. 전제
- 화면 모양이 바뀌었으면 `/impeccable` 절차를 이미 거쳤는가(CLAUDE.md 대시보드 규칙). 안 거쳤으면 멈추고 묻는다.
- 바뀐 파일 목록: `git diff --name-only origin/main...HEAD -- dashboard/ scripts/live_*.py`

## 1. 실데이터 렌더 (화면 파일을 바꿨을 때)
- `python3 -u test/render_dashboard_layout_check_20260922.py --anchor '<바꾼 카드 선택자>' --widths 1920,390 --shot tmp/shots` (로컬 바이트를 서버 페이지 위에 덮어 배포 전에 실물로 본다)
- JS 오류 0 · 빈 상태(데이터 없음·지연·준비 안 됨) 확인. 스크린샷은 크게 보여준다.

## 2. 캐시버스터
- `index.html`/`app.js`/`styles.css` 중 하나라도 바꿨으면 `index.html` 의 `?v=` 두 곳을 **내가 새로 쓰는 문자열**로 올린다(`YYYYMMDD-<기능슬러그>-vN`). 남이 올려둔 값 재사용 금지.

## 3. CI 와 같은 문법검사
- `python scripts/ci_js_syntax_check.py <바꾼 .js>` (node --check 아님 — `?.[`·`?.(` 는 CI 가 못 읽는다)
- `python -m py_compile <바꾼 .py>` · `bash -n <바꾼 .sh>`
- 프로젝트 훅이 수정 때마다 이미 돌리지만 푸시 직전에 한 번 더 — 남의 커밋이 섞였을 수 있다.

## 4. 커밋
- 저장소 커밋 스타일: `feat(dashboard): <무엇> -- <근거·검증 수치> -- 사용자 «<지시 원문>»`.
- 편집은 머지 전에 **반드시 먼저 커밋**한다(워킹트리에 남의 미커밋 파일이 상시 있다 — 커밋 안 한 채 머지하면 내 편집이 사라진다).

## 5. drift 게이트 → main
- `bash scripts/ops/check_deploy_drift.sh` 를 **파이프 없이** 돌려 종료코드 0 확인. 워크트리면 `HANDOFF_HOSTS_FILE=/home/kbj20/crypto-scalping/scripts/ops/handoff.hosts.conf` 를 준다(없으면 «판정 불가 2»).
  - 프로젝트 PreToolUse 훅도 `git push … main` · `gh pr merge` 를 만나면 같은 검사를 돌려 0 이 아니면 막는다.
- 0 이 아니면 머지하지 않는다. 출력의 «조치» 순서대로 정리하고 사용자에게 보고.
- 0 이면 `git fetch origin && git rebase origin/main` → `git push origin HEAD:main`.

## 6. 워처 배포 대기
- 워처는 5분 cron, CI `syntax-check` 통과 후 ff 머지하고 `dashboard/` 가 바뀌었으면 대시보드를 스스로 재시작한다. **손으로 kill·재시작하지 않는다.**
- 확인(읽기 전용): `bash scripts/ops/handoff.sh launch server ship_chk -- /bin/bash -c 'cd /home/llewyn/crypto-scalping && tail -5 logs/deploy_watcher_cron.log && git rev-parse --short HEAD'` → `deploy OK` 와 내 sha. 폴링은 sleep 을 넣어 띄엄띄엄.
- `deploy OK` 만 믿지 않는다 — 7번이 판정이다.

## 7. 서빙 바이트 검증
- `curl -s https://thesan.xyz/ | grep -o 'app.js?v=[^"]*'` 로 새 `?v=` 가 나오는지.
- 그 **정확한 URL** 로 `curl -s "https://thesan.xyz/app.js?v=<그 값>" | grep -c '<새로 넣은 함수명·ID·문구>'` ≥ 1. bare 경로 curl 은 엣지 캐시라 믿지 않는다.
- `dashboard/*.py` 를 바꿨으면 해당 `/api/...` 를 curl 해 새 필드가 나오는지 + 기존 엔드포인트(`/api/state` 등) 무회귀.
- 하나라도 실패하면 «배포됨»이라고 말하지 않는다. 실패 출력을 그대로 보고.

## 8. 보고
- 커밋 sha · 공개 URL 검증 결과(grep 건수) · 렌더 스크린샷. `docs/deploy_queue.md` 에 있던 항목이면 그 줄 상태를 갱신하고 같이 커밋.
