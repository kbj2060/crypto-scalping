#!/usr/bin/env python3
"""**조용히 넘어가기(silent skip)** 패턴 감사 (2026-09-03).

2026-09-03 하루에 같은 부류의 결함을 두 번 만났다:
  ① 바이낸스 파생 엔드포인트가 최근 500행만 주는데 `_fetch_data_api`가 그대로 반환 →
     `dropna(DERIV)`가 4,319봉을 499봉으로 잘랐고, FeatureEngineer가 min_periods로 계산을
     이어가 **136피쳐 중 13개가 어긋났다**(`funding_pressure`는 상관 −0.526으로 부호까지 뒤집힘).
     경고 한 줄 없었다.
  ② 재료 텐서 빌더가 `if regime_parquet.exists():`로 감싸 파일이 없자 **레짐 2열을 통째로
     건너뛰고** 41열(정상 43열) 텐서를 만들었다. 역시 조용했다.

둘 다 **실패해야 할 곳에서 통과**시켰다. 이 스크립트는 같은 부류를 기계적으로 찾는다:

  P1  `if <path>.exists():` 뒤에 `else` 없음 -- 파일이 없으면 그 블록이 통째 생략된다
  P2  `except ...: pass|continue` -- 예외를 삼킨다
  P3  `.get(key, <상수>)` -- 없는 키를 상수로 대체(특히 피쳐/설정)
  P4  `if not X: return`/`continue` 인데 로그 없음
  P5  `merge(..., how="left")` 후 결측 확인 없음

⚠️전부 결함은 아니다 -- **의도적 degrade**(라이브가 죽지 않게)인 경우도 많다.
그래서 이건 verdict가 아니라 **사람이 읽어야 할 후보 목록**이고, 라이브/재료 경로를
우선순위로 정렬한다.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PRIORITY = ("scripts/live_", "scripts/build_", "dashboard/")
PATS = [
    ("P1 exists() 뒤 else 없음", re.compile(r"if\s+[\w\.\[\]\"']+\.exists\(\)\s*:")),
    ("P2 예외 삼킴", re.compile(r"except[^\n]*:\s*(?:#[^\n]*)?\n\s+(?:pass|continue)\b")),
    ("P3 get(키, 상수)", re.compile(r"\.get\(\s*[\"'][\w_]+[\"']\s*,\s*(?:0\.0|0|-1|None|\[\]|\{\})\s*\)")),
    ("P5 left merge 후 미확인", re.compile(r"how\s*=\s*[\"']left[\"']")),
]


def main() -> int:
    only_live = "--live" in sys.argv
    files = sorted(set(list((ROOT / "scripts").glob("live_*.py"))
                       + list((ROOT / "scripts").glob("build_*.py"))
                       + list((ROOT / "dashboard").glob("*.py"))))
    if not only_live:
        files += sorted((ROOT / "scripts").glob("*20260903*.py"))
    files = sorted(set(files))
    print(f"검사 대상 {len(files)}개 파일 (라이브 + 재료빌더 + 오늘 작성분)\n")

    hits = {}
    for f in files:
        try:
            src = f.read_text()
        except Exception:
            continue
        lines = src.splitlines()
        for tag, pat in PATS:
            for m in pat.finditer(src):
                ln = src[:m.start()].count("\n") + 1
                # P1: 같은 들여쓰기의 else가 뒤따르는지 확인
                if tag.startswith("P1"):
                    ind = len(lines[ln - 1]) - len(lines[ln - 1].lstrip())
                    has_else = any(
                        (len(l) - len(l.lstrip())) == ind and l.lstrip().startswith(("else:", "elif "))
                        for l in lines[ln:ln + 60]
                        if l.strip() and (len(l) - len(l.lstrip())) <= ind)
                    if has_else:
                        continue
                hits.setdefault(str(f.relative_to(ROOT)), []).append((tag, ln,
                                                                     lines[ln - 1].strip()[:100]))
    def prio(p):
        return (0 if any(p.startswith(x) for x in PRIORITY) else 1, p)

    for f in sorted(hits, key=prio):
        live = "⭐라이브/재료" if any(f.startswith(x) for x in PRIORITY) else "  "
        print(f"{live} {f}  ({len(hits[f])}건)")
        for tag, ln, txt in hits[f][:6]:
            print(f"      {tag:22s} :{ln:<5d} {txt}")
        if len(hits[f]) > 6:
            print(f"      ... 외 {len(hits[f])-6}건")
    tot = sum(len(v) for v in hits.values())
    nlive = sum(len(v) for k, v in hits.items() if any(k.startswith(x) for x in PRIORITY))
    print(f"\n총 {tot}건 / 파일 {len(hits)}개 · ⭐라이브·재료 경로 {nlive}건")
    print("⚠️전부 결함은 아니다 -- 의도적 degrade와 구분해 사람이 읽어야 한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
