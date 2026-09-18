#!/usr/bin/env python3
"""변경된 .js 파일의 **구문**만 검사한다. (2026-09-18)

왜: `dashboard/live/app.js` 는 브라우저가 파싱하는 **라이브 서빙 파일**인데 CI 는
`.py/.sh/.json/.yaml` 만 검사했다. 구문 오류 한 줄이면 대시보드가 통째로 안 뜬다
(이 저장소는 2026-08-24·09-01 두 번 대시보드가 실제로 깨졌다).

🔴한계: 쓸 수 있는 파서(esprima)가 옵셔널 체이닝(`?.`)·널 병합(`??`)을 모른다. 그래서
**검사 전에 그 둘을 등가 형태로 낮춘다.** 「이 파일이 ES2020 으로 유효한가」가 아니라
「괄호·블록·문장 구조가 깨지지 않았는가」를 본다 -- 배포를 막고 싶은 사고는 그쪽이다.
"""
from __future__ import annotations
import re, sys
import esprima


def check(path: str) -> bool:
    src = open(path, encoding="utf-8").read()
    lowered = re.sub(r"\?\s*\.", ".", src).replace("??", "||")
    try:
        esprima.parseScript(lowered, {"tolerant": False})
    except Exception as exc:                      # noqa: BLE001 -- 파서가 무엇을 던지든 실패다
        print(f"🔴 {path}: {exc}")
        return False
    print(f"✅ {path} ({len(src):,} bytes)")
    return True


if __name__ == "__main__":
    files = [f for f in sys.argv[1:] if f.endswith(".js")]
    if not files:
        print("검사할 .js 없음"); raise SystemExit(0)
    raise SystemExit(0 if all(check(f) for f in files) else 1)
