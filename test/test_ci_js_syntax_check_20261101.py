"""CI JS 검사기: 옵셔널 체이닝 `?.[` `?.(` 를 구문 오류로 오판하지 않고, 진짜 깨진 구문은 여전히 잡는다."""
import importlib.util

import pytest

pytest.importorskip("esprima")
sp = importlib.util.spec_from_file_location("c", "scripts/ci_js_syntax_check.py")
c = importlib.util.module_from_spec(sp)
sp.loader.exec_module(c)


def _ok(tmp_path, src):
    f = tmp_path / "t.js"
    f.write_text(src)
    return c.check(str(f))


def test_optional_chaining_forms(tmp_path):
    assert _ok(tmp_path, "const k = a?.b?.[`${x}:${y}`] ?? f?.(1);\n")


def test_real_syntax_error_still_fails(tmp_path):
    assert not _ok(tmp_path, "function f( { return 1; }\n")
