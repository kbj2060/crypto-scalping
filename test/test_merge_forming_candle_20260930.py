"""형성 중 봉 병합(app.js mergeFormingCandle) -- 본문을 떼어 node 로 실제로 돌린다 (2026-09-30).

SOL·XRP 서버 market-history 는 형성 봉을 60초 캐시로 준다. 통째로 교체하면 라이브가 넓힌 고가·저가가 줄었다.
    python3 -m pytest -q test/test_merge_forming_candle_20260930.py
"""
import json
import pathlib
import re
import subprocess

APP = pathlib.Path(__file__).resolve().parents[1] / "dashboard" / "live" / "app.js"


def merge(prev, nxt, now_s):
    src = APP.read_text("utf-8")
    fn = re.search(r"^function mergeFormingCandle\(.*?^}\n", src, re.S | re.M).group(0)
    js = ("const CHART_CANDLE_MIN = 5;\n" + fn
          + f"console.log(JSON.stringify(mergeFormingCandle({json.dumps(prev)}, {json.dumps(nxt)}, {now_s})));")
    return json.loads(subprocess.run(["node", "-e", js], capture_output=True, text=True, check=True).stdout)


T = 1_790_000_100 - 1_790_000_100 % 300


def c(t, h, l, cl=1.0):
    return {"time": t, "open": 1.0, "high": h, "low": l, "close": cl}


def test_forming_bar_keeps_live_extremes():
    out = merge([c(T - 300, 2, 0.5), c(T, 1.3, 0.7)], [c(T - 300, 2, 0.5), c(T, 1.1, 0.9, 1.05)], T + 100)
    assert out[-1]["high"] == 1.3 and out[-1]["low"] == 0.7 and out[-1]["close"] == 1.05


def test_closed_bar_and_new_bar_untouched():
    # 마감된 봉(now 가 봉 끝을 지남)은 서버 값 그대로
    out = merge([c(T, 1.3, 0.7)], [c(T, 1.1, 0.9)], T + 300)
    assert out[-1]["high"] == 1.1 and out[-1]["low"] == 0.9
    # 🔴2026-10-11 ETH(서버는 마감봉만): 서버보다 새 라이브 봉은 버리지 않고 이어 붙인다 -- 버리면 지금 가격으로 다시 열려 시가·고저가 리셋됐다
    out = merge([c(T, 1.3, 0.7)], [c(T - 300, 1.1, 0.9)], T + 100)
    assert [o["time"] for o in out] == [T - 300, T] and out[0]["high"] == 1.1 and out[-1]["high"] == 1.3
    assert merge(None, [], T) == []
