#!/usr/bin/env python3
"""Claude Code 훅 -- CLAUDE.md·메모리에 «글로만» 있던 규칙 넷을 기계로 막는다. (2026-10-29)

  pre  (PreToolUse)  : ① .env(실키) 읽기·수정 차단 ② 실주문·로컬 바이낸스 REST 차단
                       ③ main 전진(push main · gh pr merge · main 위 git merge) 전에 check_deploy_drift.sh 종료코드 0 요구
  post (PostToolUse) : ④ 수정 파일을 CI(syntax-check)와 같은 방식으로 검사 ⑤ 새 라벨·데이터셋 빌더면 누수 감사 안내

종료코드 2 = 막고 stderr 를 Claude 에게 돌려준다. 자체점검: python3 .claude/hooks/guard.py selftest
ponytail: 명령 «글자»만 본다 -- 스크립트 파일 안에서 주문·REST 를 부르면 못 잡는다(그건 코드 리뷰·leak-auditor 몫).
"""
from __future__ import annotations
import json, os, re, subprocess, sys
from pathlib import Path

ROOT = Path(os.environ.get("CLAUDE_PROJECT_DIR") or Path(__file__).resolve().parents[2])

ENV_TOKEN = re.compile(r"(^|[\s'\"=/<])\.env($|[\s'\")])")
READERS = re.compile(r"\b(cat|head|tail|less|more|grep|rg|sed|awk|cut|sort|nl|od|xxd|strings|base64|python3?|node)\b")
ORDER = re.compile(r"\b(futures_)?(create_(test_)?order|cancel_order|order_(market|limit)\w*|new_order)\b|/(fapi|dapi)/v1/order|/api/v3/order")
BINANCE_REST = re.compile(r"\b(api\d?|fapi|dapi|testnet)\.binance(future)?\.(com|vision)\b")
SEARCH_ONLY = {"grep", "rg", "git", "graft", "find", "ls", "sed", "cat", "head", "tail", "wc", "less", "awk"}
PUSH_MAIN = re.compile(r"\bgit\b[^|;&]*\bpush\b[^|;&]*\bmain\b")
PR_MERGE = re.compile(r"\bgh\s+pr\s+merge\b")
GIT_MERGE = re.compile(r"\bgit\b[^|;&]*\bmerge\b")
LEAK_NAME = re.compile(r"(label|dataset|exit_head|relabel|barrier)", re.I)


def first_word(cmd: str) -> str:
    # 앞쪽 `cd X &&` 는 건너뛴다
    seg = re.sub(r"^\s*(cd\s+\S+\s*(&&|;)\s*)+", "", cmd).strip()
    return seg.split()[0].rsplit("/", 1)[-1] if seg else ""


def block_reason(tool: str, inp: dict) -> str | None:
    path = inp.get("file_path") or inp.get("notebook_path") or inp.get("path") or ""
    if tool != "Bash":
        name = Path(path).name
        if name == ".env" or (name.startswith(".env.") and name != ".env.example"):
            return f".env 는 실키가 들어 있어 읽지도 고치지도 않습니다 ({path})."
        return None
    cmd = inp.get("command", "")
    if ENV_TOKEN.search(cmd) and READERS.search(cmd):
        return ".env 내용을 출력·복사하는 명령입니다 -- 실키가 대화 기록에 남습니다."
    if first_word(cmd) in SEARCH_ONLY:
        return None
    if ORDER.search(cmd):
        return "실주문(생성·취소) 호출이 들어 있습니다 -- 실주문 테스트는 절대 금지입니다."
    if BINANCE_REST.search(cmd) and "handoff.sh" not in cmd and "data.binance.vision" not in cmd:
        return "로컬에서 바이낸스 REST 를 부르면 서버와 IP 한도를 같이 써 밴됩니다 -- handoff.sh launch server 로 서버에서 돌리세요."
    return None


def advances_main(cmd: str, branch: str) -> bool:
    return bool(PUSH_MAIN.search(cmd) or PR_MERGE.search(cmd) or (branch == "main" and GIT_MERGE.search(cmd)))


def git(cwd: str, *args: str) -> str:
    r = subprocess.run(["git", "-C", cwd, *args], capture_output=True, text=True)
    return r.stdout.strip()


def drift_gate(cwd: str) -> str | None:
    env = dict(os.environ)
    # 워크트리엔 handoff.hosts.conf 가 없다(gitignored) -> 메인 체크아웃 것을 쓴다. 없으면 스크립트가 «판정 불가(2)»
    main_co = Path(git(cwd, "rev-parse", "--path-format=absolute", "--git-common-dir") or ROOT / ".git").parent
    hosts = main_co / "scripts/ops/handoff.hosts.conf"
    if "HANDOFF_HOSTS_FILE" not in env and hosts.exists():
        env["HANDOFF_HOSTS_FILE"] = str(hosts)
    r = subprocess.run(["bash", str(ROOT / "scripts/ops/check_deploy_drift.sh")], capture_output=True, text=True, env=env)
    if r.returncode == 0:
        return None
    tail = "\n".join((r.stdout + r.stderr).strip().splitlines()[-15:])
    return f"check_deploy_drift.sh 종료코드 {r.returncode} -- main 을 전진시키지 않습니다.\n{tail}"


def syntax_error(path: Path) -> str | None:
    suf = path.suffix
    try:
        if suf == ".py":
            compile(path.read_text(encoding="utf-8"), str(path), "exec")
        elif suf == ".json":
            json.loads(path.read_text(encoding="utf-8"))
        elif suf in (".yml", ".yaml"):
            import yaml
            yaml.safe_load(path.read_text(encoding="utf-8"))
        elif suf in (".sh", ".js"):
            cmd = ["bash", "-n", str(path)] if suf == ".sh" else [sys.executable, str(ROOT / "scripts/ci_js_syntax_check.py"), str(path)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode:
                return (r.stdout + r.stderr).strip()
    except ImportError:
        return None
    except Exception as exc:  # noqa: BLE001 -- 파서가 무엇을 던지든 CI 에선 실패다
        return f"{type(exc).__name__}: {exc}"
    return None


def leak_hint(tool: str, path: Path) -> str | None:
    try:
        rel = path.resolve().relative_to(ROOT)
    except ValueError:
        return None
    if tool == "Write" and rel.parts[0] == "scripts" and path.suffix == ".py" and LEAK_NAME.search(path.name):
        return (f"{rel} 는 라벨·데이터셋 빌더로 보입니다. CLAUDE.md 계약대로 동결 전에 "
                "`python scripts/audit_event_label_feature_boundary_20260908.py --scan` 을 돌리고 "
                "(포지션 피쳐 파리티 감사 스크립트는 f2eb2377 에서 지워졌다 -- pos_* 피쳐는 leak-auditor 로 대조), "
                "층 출력을 동결하기 전엔 `scripts/gate_eth_entry_layers_20260903.py --pipeline <dir>` 를 통과시킵니다. "
                "독립 검토는 leak-auditor 서브에이전트에 코드만 넘깁니다.")
    return None


def main(mode: str) -> int:
    ev = json.load(sys.stdin)
    tool, inp, cwd = ev.get("tool_name", ""), ev.get("tool_input") or {}, ev.get("cwd") or str(ROOT)
    if mode == "pre":
        why = block_reason(tool, inp)
        if not why and tool == "Bash" and advances_main(inp.get("command", ""), git(cwd, "rev-parse", "--abbrev-ref", "HEAD")):
            why = drift_gate(cwd)
        if why:
            print(why, file=sys.stderr)
            return 2
        return 0
    path = Path(inp.get("file_path") or "")
    if not path.is_file():
        return 0
    err = syntax_error(path)
    if err:
        print(f"CI(syntax-check)가 막을 문법 오류입니다 -- 워처가 배포를 멈춥니다:\n{err}", file=sys.stderr)
        return 2
    hint = leak_hint(tool, path)
    if hint:
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "PostToolUse", "additionalContext": hint}}, ensure_ascii=False))
    return 0


def selftest() -> None:
    import tempfile
    B = lambda c: block_reason("Bash", {"command": c})
    assert block_reason("Read", {"file_path": "/x/crypto-scalping/.env"})
    assert block_reason("Edit", {"file_path": "/x/.env.local"})
    assert not block_reason("Read", {"file_path": "/x/.env.example"})
    assert not block_reason("Read", {"file_path": "/x/environment.yml"})
    assert B("cat .env") and B("grep KEY /home/kbj20/crypto-scalping/.env") and B("python -c \"print(open('.env').read())\"")
    assert not B("ls -la .env") and not B("rsync -a --exclude .env src/ dst/") and not B("cat environment.yml")
    assert B("python -c 'c.futures_create_order(symbol=\"ETHUSDC\")'") and B("curl -X POST https://fapi.binance.com/fapi/v1/order")
    assert B("cd x && python -c 'c.order_market_buy()'")
    assert not B("grep -rn create_order trading_bot.py") and not B("cd x && grep -n cancel_order live/")
    assert B("curl -s https://fapi.binance.com/fapi/v1/klines?symbol=ETHUSDT")
    assert not B("bash scripts/ops/handoff.sh launch server j -- curl -s https://fapi.binance.com/fapi/v1/time")
    assert not B("curl -O https://data.binance.vision/data/futures/um/daily/klines/x.zip")
    assert advances_main("git push origin main", "x") and advances_main("git push origin HEAD:main", "x")
    assert advances_main("gh pr merge 12 --squash", "x") and advances_main("git merge --ff-only claude/a", "main")
    assert not advances_main("git push -u origin claude/rl-v27", "x") and not advances_main("git merge main", "claude/a")
    with tempfile.TemporaryDirectory() as d:
        bad, good, js = Path(d, "a.py"), Path(d, "b.json"), Path(d, "c.sh")
        bad.write_text("def f(:\n"), good.write_text('{"a": 1}'), js.write_text("if then fi\n")
        assert syntax_error(bad) and not syntax_error(good) and syntax_error(js)
        conflict = Path(d, "d.py"); conflict.write_text("<<<<<<< Updated upstream\nx = 1\n")
        assert syntax_error(conflict)
    assert leak_hint("Write", ROOT / "scripts/build_eth_new_labels_20261029.py")
    assert not leak_hint("Edit", ROOT / "scripts/build_eth_new_labels_20261029.py")
    assert not leak_hint("Write", ROOT / "dashboard/server.py")
    print("guard selftest OK")


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
        sys.exit(0)
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "pre"))
