#!/usr/bin/env python3
"""**체결 봉 크레딧 결함**이 다른 전략에도 있는가 (2026-09-03).

오늘 진입 모델에서 확인된 결함: 지정가가 직전 종가보다 3 ATR 떨어져 있으면 **그 봉의 반대쪽
극단은 거의 확실히 체결 이전**인데, 청산 시뮬이 체결 봉 `f`부터 평가해 그 값을 진입 후 MFE로
크레딧했다. 전체 후보 PF 2.86 → 0.95, 승률 84.3% → 69.9%.

⭐**이 결함은 intrabar 진입(지정가 터치)에만 생긴다.** 봉 시가/종가 진입은 그 봉 전체가
진입 이후이므로 안전하다. 그래서 판정 규칙은 둘의 조합이다:

  조건1  진입 인덱스가 **가격 터치 스캔**으로 정해지는가 (`low<=lim` / `high>=lim`)
  조건2  청산/라벨 시뮬이 **그 같은 인덱스부터** 시작하는가 (`h[f:...]`, `.iloc[f:]` 등)

둘 다 참이면 **의심**, 조건1만이면 진입만 intrabar(청산은 다음 봉부터 = 안전),
조건2만이면 진입이 봉 경계(= 안전).

⚠️이건 verdict가 아니라 **사람이 읽어야 할 후보 목록**이다. 파일마다 관례가 다르다.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TODAY = ("20260903",)                     # 오늘 작업분 -- 이미 감사됨

TOUCH = re.compile(r"(?:low|lo|l)\s*\[[^\]]+\]\s*<=\s*\w*lim|"
                   r"(?:high|hi|h)\s*\[[^\]]+\]\s*>=\s*\w*lim|"
                   r"\.low\b[^\n]{0,40}<=\s*\w*lim|\.high\b[^\n]{0,40}>=\s*\w*lim")
# 체결 인덱스부터 시작하는 슬라이스: h[f:f+H], hi[fi:...], .iloc[fill:...]
FROMFILL = re.compile(r"\[\s*(?:f|fi|fill|fill_i|entry_i|ei0|f_)\s*:\s*")
# 다음 봉부터: h[f+1:...]
FROMNEXT = re.compile(r"\[\s*(?:f|fi|fill|fill_i|entry_i|f_)\s*\+\s*1\s*:")
OPENENTRY = re.compile(r"open\s*\[[^\]]*\+\s*1\s*\]|다음 봉 시가|next bar'?s? open|"
                       r"entry.{0,12}=\s*open")


def main() -> int:
    files = sorted(ROOT.glob("scripts/*.py")) + sorted(ROOT.glob("scripts/**/*.py"))
    rows = []
    for f in sorted(set(files)):
        try:
            s = f.read_text()
        except Exception:
            continue
        t = bool(TOUCH.search(s))
        if not t:
            continue
        ff = bool(FROMFILL.search(s))
        fn = bool(FROMNEXT.search(s))
        oe = bool(OPENENTRY.search(s))
        today = any(d in f.name for d in TODAY)
        if ff and not fn:
            verdict, mark = "⚠️의심 -- 체결 봉부터 평가", 0
        elif ff and fn:
            verdict, mark = "△혼재 -- 둘 다 등장, 확인 필요", 1
        elif fn:
            verdict, mark = "✅다음 봉부터", 2
        elif oe:
            verdict, mark = "✅봉 시가 진입", 2
        else:
            verdict, mark = "△청산 시작점 불명", 1
        rows.append((mark, today, f.relative_to(ROOT), verdict))

    rows.sort()
    print(f"지정가 터치 진입을 쓰는 파일 {len(rows)}개\n")
    print(f"{'':3s}{'파일':>72s}  판정")
    for mark, today, rel, verdict in rows:
        tag = "[오늘]" if today else "     "
        print(f"{tag} {str(rel):>72s}  {verdict}")
    n_bad = sum(1 for m, t, *_ in rows if m == 0 and not t)
    n_chk = sum(1 for m, t, *_ in rows if m == 1 and not t)
    print(f"\n⚠️오늘 작업분 제외: 의심 {n_bad}개 · 확인필요 {n_chk}개")
    print("⚠️판정이 아니라 후보 목록이다 -- 파일마다 관례가 다르므로 직접 읽어야 한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
