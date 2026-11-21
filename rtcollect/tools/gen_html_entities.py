#!/usr/bin/env python3
"""파이썬 html.unescape 가 쓰는 표 셋(html.entities.html5 · html._invalid_charrefs · html._invalid_codepoints)을 Rust 소스로 떠 둔다.
/api/news 의 제목 풀기(server.py news_payload·news_key 의 html_unescape)가 쓴다. 파이썬 판이 바뀌면 다시 돌린다(서버 3.11).

    python3 rtcollect/tools/gen_html_entities.py   # rtcollect/src/html.rs 의 표 부분(첫 `fn html5(` 앞)만 바꿔 쓴다 -- 2026-11-21 rtserve 에서 옮김
"""
import html
import html.entities
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "src" / "html.rs"


def lit(s: str) -> str:
    return '"' + "".join(c if c.isascii() and c.isalnum() or c in ";" else f"\\u{{{ord(c):x}}}" for c in s) + '"'


rows = sorted(html.entities.html5.items())
old = OUT.read_text()
head = old[:old.index("/// html.entities.html5")]          # 모듈 머리 설명은 그대로
rest = old[old.index("fn html5("):]                        # unescape 함수들은 그대로
out = [head.rstrip("\n") + f"\n//! (표 = 파이썬 {sys.version.split()[0]})", "", "/// html.entities.html5(이름 → 글자) · 이름 순(이진 탐색)", "pub static HTML5: &[(&str, &str)] = &["]
out += [f"    ({lit(k)}, {lit(v)})," for k, v in rows]
out += ["];", "", "/// html._invalid_charrefs(숫자 → 대신 쓸 글자) · 숫자 순", "pub static INVALID_CHARREFS: &[(u32, &str)] = &["]
out += [f"    ({k}, {lit(v)})," for k, v in sorted(html._invalid_charrefs.items())]
out += ["];", "", "/// html._invalid_codepoints(지우는 숫자) · 숫자 순", "pub static INVALID_CODEPOINTS: &[u32] = &["]
out += [f"    {k}," for k in sorted(html._invalid_codepoints)]
out += ["];", ""]
OUT.write_text("\n".join(out) + rest)
print(f"{OUT}: html5 {len(rows)} · invalid_charrefs {len(html._invalid_charrefs)} · invalid_codepoints {len(html._invalid_codepoints)}")
