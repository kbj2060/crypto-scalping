#!/usr/bin/env python3
"""장 달력 표(rtduck/src/sessions.json) -- 원본 `duckdb_persist_worker.session_flags` 가 쓰는 pandas_market_calendars(JPX·LSE·NYSE)의
schedule 을 그대로 떠 둔다(새 달력 크레이트 금지 · 사용자 지시). 칸 = 그 UTC 날짜 0시부터 분: [개장, 마감] 또는 [개장, 마감, 점심 시작, 점심 끝].
없는 날 = 휴장. 범위 밖이면 Rust 는 원본의 예외 길(UTC 시각 규칙)로 간다. 🔴서버의 pandas_market_calendars 판과 같은 판으로 떠야 한다
(개발기 5.4.0 · 서버 판 미확인 -- JPX 마감이 2024-11 에 15:00→15:30 으로 바뀐 것처럼 판마다 규칙이 다를 수 있다).

    bash rtduck/tools/server_env.sh /tmp/venv_mcal && /tmp/venv_mcal/bin/python rtduck/tools/gen_sessions.py     # 서버 판 표
    python3 rtduck/tools/gen_sessions.py /tmp/s54.json                                                         # 다른 판과 비교용
"""
import json
from pathlib import Path

import pandas as pd
import pandas_market_calendars as mcal

LO, HI = "2024-01-01", "2035-12-31"
out = {"_source": f"pandas_market_calendars {mcal.__version__} · {LO}~{HI}"}
for name in ("JPX", "LSE", "NYSE"):
    s = mcal.get_calendar(name).schedule(start_date=LO, end_date=HI)
    rows = {}
    for d, r in s.iterrows():
        day = pd.Timestamp(d.date(), tz="UTC")
        m = lambda x: int((pd.Timestamp(x) - day).total_seconds() // 60)   # noqa: E731
        v = [m(r["market_open"]), m(r["market_close"])]
        bs, be = r.get("break_start", pd.NaT), r.get("break_end", pd.NaT)
        if pd.notna(bs) and pd.notna(be):
            v += [m(bs), m(be)]
        for x in (r["market_open"], r["market_close"]):
            assert pd.Timestamp(x).second == 0, (name, d, x)     # 분 단위라야 분 표로 같다
        rows[d.strftime("%Y-%m-%d")] = v
    out[name] = rows
import sys
# 기본 = 서버 판(5.3.1.1)으로 뜬 표를 rtduck/src/sessions.json 에 -- 다른 판과 비교할 때는 둘째 인자로 다른 곳에
p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "src" / "sessions.json"
if len(sys.argv) == 1:
    assert mcal.__version__ == "5.3.1.1", f"서버 판(5.3.1.1)으로 떠야 한다 -- 지금 {mcal.__version__} (bash rtduck/tools/server_env.sh <venv>)"
p.write_text(json.dumps(out, separators=(",", ":")))
print(p, p.stat().st_size // 1024, "KB", {k: len(v) for k, v in out.items() if k != "_source"})
