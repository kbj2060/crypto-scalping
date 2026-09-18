#!/usr/bin/env python3
"""«정의 전에 쓰는 이름»을 정적으로 잡는다 (2026-09-19).

🔴왜 생겼나: `dashboard/server.py` 의 `footprint_add` 에서 리팩터가 `now = time.time()` 한 줄을
지웠는데, 아래 조건이 그 이름을 계속 쓰고 있었다:

    if footprint_state["ready"] and now - footprint_state["saved_at"] >= ...

**단락평가 때문에 `ready` 가 False 인 동안은 조용했다.** 백필이 끝나 ready 가 되는 순간부터
모든 체결이 NameError 로 터졌고, WS 루프가 그걸 잡아 재연결하므로 화면은 백필로 채워져
«멀쩡해 보였다» -- 프로덕션에서 40분 동안 519회 터지는 동안 아무도 몰랐다.

파이썬은 이걸 import 시점에 못 잡는다(UnboundLocalError 는 런타임이다). 그래서 AST 로 본다.
같은 함수 안에서 어떤 이름이 **처음 읽히는 줄**이 **처음 대입되는 줄**보다 앞서면 의심한다.

한계(일부러 남긴다):
  - 분기 안 대입(`if c: x = 1` 뒤에 `x` 사용)은 «대입이 먼저»라 안 걸린다. 그건 다른 문제다.
  - 루프 뒤에 쓰는 값처럼 «읽기가 텍스트상 먼저»인 정상 패턴은 아래 예외 규칙으로 거른다.
목적은 완전성이 아니라 **이 사고의 재발 차단**이다.

python test/test_no_use_before_assign_20260919.py
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    ROOT / "dashboard" / "server.py",
    ROOT / "scripts" / "live_trade_tape_collector_20260916.py",
]


def _binding_names(node: ast.AST) -> set[str]:
    """컴프리헨션·for·with·except 가 «묶는» 이름. 이들은 읽기보다 뒤에 적히는 게 정상이다."""
    out: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, (ast.comprehension,)):
            out |= {t.id for t in ast.walk(n.target) if isinstance(t, ast.Name)}
        elif isinstance(n, (ast.For, ast.AsyncFor)):
            out |= {t.id for t in ast.walk(n.target) if isinstance(t, ast.Name)}
        elif isinstance(n, ast.withitem) and n.optional_vars is not None:
            out |= {t.id for t in ast.walk(n.optional_vars) if isinstance(t, ast.Name)}
        elif isinstance(n, ast.ExceptHandler) and n.name:
            out.add(n.name)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            out |= set(n.names)
    return out


def _own_body(fn: ast.AST) -> list[ast.AST]:
    """중첩 함수 본문은 제외한다 -- 그 안의 이름은 그 함수의 문제다."""
    out: list[ast.AST] = []
    for stmt in fn.body:
        for n in ast.walk(stmt):
            if n is not stmt and isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                continue
            out.append(n)
    nested: set[int] = set()
    for stmt in fn.body:
        for n in ast.walk(stmt):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                for inner in ast.walk(n):
                    if inner is not n:
                        nested.add(id(inner))
    return [n for n in out if id(n) not in nested]


def suspects(path: Path) -> list[tuple[str, str, int, int]]:
    tree = ast.parse(path.read_text())
    found: list[tuple[str, str, int, int]] = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        params = {a.arg for a in fn.args.args + fn.args.kwonlyargs + fn.args.posonlyargs}
        if fn.args.vararg:
            params.add(fn.args.vararg.arg)
        if fn.args.kwarg:
            params.add(fn.args.kwarg.arg)
        skip = params | _binding_names(fn)
        first_load: dict[str, int] = {}
        first_store: dict[str, int] = {}
        for n in _own_body(fn):
            if isinstance(n, ast.Name):
                d = first_store if isinstance(n.ctx, ast.Store) else first_load
                d.setdefault(n.id, n.lineno)
        for name, store_line in first_store.items():
            if name in skip:
                continue
            load_line = first_load.get(name)
            if load_line is not None and load_line < store_line:
                found.append((fn.name, name, load_line, store_line))
    return found


def test_no_use_before_assign() -> None:
    bad: list[str] = []
    for path in TARGETS:
        for fn, name, load, store in suspects(path):
            bad.append(f"{path.relative_to(ROOT)}:{load} {fn}() -- "
                       f"'{name}' 을 {load}행에서 읽는데 대입은 {store}행이다")
    assert not bad, "정의 전에 쓰는 이름:\n  " + "\n  ".join(bad)


def test_detector_catches_the_real_bug() -> None:
    """검출기가 실제로 그 모양을 잡는지 -- 픽스처로 못박는다(검출기가 죽으면 조용해진다)."""
    import tempfile
    buggy = '''
import time
def footprint_add(qty):
    state = {"ready": True, "saved_at": 0.0}
    state["updated"] = time.time()
    if state["ready"] and now - state["saved_at"] >= 30.0:
        pass
    now = time.time()
'''
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(buggy)
        tmp = Path(f.name)
    hits = suspects(tmp)
    tmp.unlink()
    assert any(name == "now" for _fn, name, _l, _s in hits), hits


if __name__ == "__main__":
    test_detector_catches_the_real_bug()
    test_no_use_before_assign()
    print("ok — 검출기 픽스처 통과 · 대상 2개 파일에 정의 전 사용 없음")
