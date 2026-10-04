#!/usr/bin/env python3
"""머리 칩 «실전 성적» 집계 + 알림 센터 payload 계약 (2026-10-05). 프레임워크 없이 실행한다.

  · 경보기 성적 = 예고 봉 중 다음 1~6봉 안 탐지(감사 replay.py 의 fut_fire 정의) · 뒤 30분이 비어 있는 봉은 안 센다
  · 닿음 칩은 2026-10-05 제거 -- 옛 reach 줄이 파일에 남아 있어도 집계에 안 나온다
  · 14일 밖 줄·다른 코인 줄·깨진 줄은 안 읽는다
  · 알림 센터 = 판정 달력(날짜순) + 데몬의 PUSH_KINDS + 최근 보낸 알림(최신 먼저)

실행: python test/test_chip_score_notify_center_20261005.py
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("dash_srv_chip", ROOT / "dashboard" / "server.py")
srv = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = srv      # dataclass 가 자기 모듈을 찾는다
spec.loader.exec_module(srv)

T0 = 1_790_000_100 // 300 * 300
bo = [{"k": "bo", "a": "eth", "t": T0 + 300 * i, "pw": i in (0, 3), "det": i == 2} for i in range(12)]
reach = [{"k": "reach", "a": "eth", "t": T0, "p": 0.7, "hit": 1}, {"k": "reach", "a": "eth", "t": T0 + 300, "p": 0.3, "hit": 0}]
burst = [{"k": "burst", "a": "eth", "t": T0, "side": "long", "rev_bp": 8.0}, {"k": "burst", "a": "eth", "t": T0, "side": "short", "rev_bp": -2.0}]
s = srv.chip_score_summary(bo + reach + burst)
# 뒤 30분이 다 있는 봉 = 0..5(6개). 예고 봉 0 → 2번 봉에서 탐지(맞음), 예고 봉 3 → 4~9 에 탐지 없음(틀림)
assert s["prewarn"] == {"bars": 6, "warn": 2, "hits": 1, "base": 2 / 6, "since": T0}, s["prewarn"]
assert "reach" not in s, s
assert s["burst"] == {"n": 2, "rev_mean_bp": 3.0, "rev_share": 0.5}, s["burst"]
assert srv.chip_score_summary([]) == {"days": 14}

with tempfile.TemporaryDirectory() as td:
    srv.LIVE_DIR = pathlib.Path(td)
    now = T0 + 20 * 86400
    with open(srv.chip_score_path(now), "w") as fh:
        fh.write(json.dumps({"k": "reach", "a": "eth", "t": now - 100, "p": .5, "hit": 1}) + "\n")
        fh.write(json.dumps({"k": "reach", "a": "sol", "t": now - 100, "p": .5, "hit": 1}) + "\n")
        fh.write(json.dumps({"k": "reach", "a": "eth", "t": now - 15 * 86400, "p": .5, "hit": 1}) + "\n")
        fh.write("{깨진 줄\n")
    assert [r["a"] for r in srv.chip_score_rows(now, "eth")] == ["eth"], srv.chip_score_rows(now, "eth")

    srv.PUSH_SENT_LOG_PATH = pathlib.Path(td) / "sent.jsonl"
    srv.PUSH_SENT_LOG_PATH.write_text("".join(json.dumps({"title": f"n{i}"}) + "\n" for i in range(3)))
    nc = srv.notify_center_payload()
    dates = [v["date"] for v in nc["verdicts"]]
    assert dates and dates == sorted(dates), dates
    assert {k["key"] for k in nc["kinds"]} == {"verdict", "risk", "burst_hold", "prewarn_hold"}, nc["kinds"]
    assert all(k["name"] and k["when"] for k in nc["kinds"])
    assert [r["title"] for r in nc["recent"]] == ["n2", "n1", "n0"], nc["recent"]
print("chip score + notify center ok")
