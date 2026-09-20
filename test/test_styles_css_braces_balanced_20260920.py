"""styles.css 의 중괄호가 맞는가, 2026-09-20.

🔴CSS 는 문법 오류로 **죽지 않는다** -- 규칙 하나를 조용히 버리고 계속 간다. 그래서
짝 없는 `}` 하나가 3주 동안 아무 경고 없이 규칙을 먹고 있었다.

정확히 무슨 일이 있었나: 최상위에 뜬 `}` 는 «무시»되지 않는다. CSS Syntax L3 의
consume-a-list-of-rules 는 at-keyword 가 아닌 토큰을 만나면 **qualified rule 로 재소비**하고,
consume-a-qualified-rule 은 **다음 `{` 까지**를 prelude 로 먹는다. 그래서 선택자가
`} .asset-tabs` 가 되어 그 규칙이 통째로 무효가 됐다 -- 2026-09-19 «코인 탭의 껍데기를
걷어내라» 요청이 안쪽 버튼에만 적용되고 컨테이너는 옛 모양으로 남은 반쪽 상태의 정체다.
(브라우저 실측으로 확인: 괄호 한 줄을 지우자 페이지가 8px 짧아졌다.)

⚠️그래서 이 시험은 «스타일 검사»가 아니라 **한 규칙이 조용히 사라지는 것**을 막는 장치다.
  눈으로는 못 잡는다 -- 먹힌 규칙이 하필 미세한 여백이면 아무도 모른다.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path

CSS = Path(__file__).resolve().parents[1] / "dashboard/live/styles.css"


def _strip(css: str) -> str:
    """주석과 문자열을 지운다. **줄 수는 보존**한다 -- 안 그러면 오류 위치를 못 짚는다."""
    css = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), css, flags=re.S)
    css = re.sub(r'"(?:[^"\\\n]|\\.)*"', '""', css)
    return re.sub(r"'(?:[^'\\\n]|\\.)*'", "''", css)


class StylesCssBracesTest(unittest.TestCase):
    def test_no_stray_or_missing_brace(self) -> None:
        raw = CSS.read_text(encoding="utf-8")
        lines = raw.split("\n")
        depth = 0
        line = 1
        stray = []
        for ch in _strip(raw):
            if ch == "\n":
                line += 1
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth < 0:
                    stray.append((line, lines[line - 1].strip()[:80]))
                    depth = 0          # 계속 훑어 **전부** 보고한다(첫 개만 고치면 또 온다)
        self.assertEqual(
            stray, [],
            "짝 없는 `}` -- 이 자리 **다음 규칙이 통째로 먹힌다**(docstring 참고): "
            + "; ".join(f"{n}행 {t!r}" for n, t in stray))
        self.assertEqual(depth, 0, f"닫히지 않은 블록이 {depth}개 남았다")


if __name__ == "__main__":
    unittest.main()
