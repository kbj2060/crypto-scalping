"""추세 veto 선 갱신 재시도(app.js historyRetryDue) -- 본문을 떼어 node 로 **실제로 돌린다** (2026-09-26).

사용자 «풋프린트의 추세 veto 라인이 매 5분마다 업데이트가 안 돼». 서버 실측: 마감봉 값은 마감 뒤 ~60~65초에
붙는데 재시도가 +70초 한 번뿐이라, 그 한 발이 옛 프레임을 받으면 다음 5분 폴링까지 선이 멈췄다.
    python3 -m pytest -q test/test_history_retry_due_20260926.py
"""
import json
import pathlib
import re
import subprocess

APP = pathlib.Path(__file__).resolve().parents[1] / "dashboard" / "live" / "app.js"


def due(closed_time, sma, now_s, last_retry_s):
    src = APP.read_text("utf-8")
    fn = re.search(r"^function historyRetryDue\(.*?^}\n", src, re.S | re.M).group(0)
    js = ("const HISTORY_RETRY_AFTER_CLOSE_S = 60, HISTORY_RETRY_EVERY_MS = 15000, CHART_CANDLE_MIN = 5;\n" + fn
          + f"console.log(JSON.stringify(historyRetryDue({closed_time}, {json.dumps(sma)}, {now_s * 1000}, {last_retry_s * 1000})));")
    return json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)


T = 1_790_000_000          # 마감봉 시작(초). 마감 = T+300
CLOSE = T + 300


def test_waits_for_server_ttl_then_retries_every_15s_until_value_arrives():
    assert due(T, None, CLOSE + 30, 0) is False             # 서버 TTL 60초 전엔 물어도 같은 답
    assert due(T, None, CLOSE + 61, 0) is True
    # 🔴옛 코드는 여기서 끝이었다(봉당 한 번). 그 한 발이 옛 프레임이면 다음 5분 폴링까지 멈췄다.
    assert due(T, None, CLOSE + 70, CLOSE + 61) is False     # 15초 간격
    assert due(T, None, CLOSE + 77, CLOSE + 61) is True      # 값이 아직 없으면 또 묻는다


def test_stops_when_value_arrived_or_two_bars_passed():
    assert due(T, 2680.5, CLOSE + 90, 0) is False            # 붙었으면 그만
    assert due(T, None, CLOSE + 600, 0) is False             # 두 봉이 지나면 정기 폴링에 맡긴다
    assert due(0, None, CLOSE + 90, 0) is False              # 캔들이 없으면 없음
