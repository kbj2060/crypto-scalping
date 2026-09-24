"""캔들 행 메모: 같은 프레임이면 다시 안 만들고, 프레임이 바뀌면 다시 만든다. 2026-09-24.

`python3 test/test_market_history_memo_20260924.py` -- 클로저 안이라 소스를 떼어 내 돌린다.
"""
import asyncio, re
from pathlib import Path
import numpy as np, pandas as pd

src = (Path(__file__).resolve().parents[1] / "dashboard" / "server.py").read_text(encoding="utf-8")
start = src.index("    market_rows_memo: dict")
end = src.index("    async def load_market_history(asset: str)")
chunk = re.sub(r"^    ", "", src[start:end], flags=re.M)
calls = []
ns = {"Any": object, "web": None, "evidence_signal_cache": {}, "trend_veto_rows": lambda df: calls.append(df) or {}}
async def warm(): pass
ns["load_chart_klines_frames"] = warm
exec(compile(chunk, "memo", "exec"), ns)
load = ns["load_market_history_from_evidence_cache"]

def frame(p0):
    ts = pd.date_range("2026-09-24", periods=300, freq="5min", tz="UTC")
    c = p0 + np.arange(300.0)
    return pd.DataFrame({"timestamp": ts, "open": c, "high": c + 1, "low": c - 1, "close": c})

async def main():
    eth, btc = frame(2600), frame(60000)
    ns["evidence_signal_cache"]["frames"] = (eth, btc, None)
    a = await load("eth")
    assert len(a) == 200 and a[-1]["close"] == 2899.0
    assert await load("eth") is a and len(calls) == 1          # 같은 프레임 -> 다시 안 만든다
    b = await load("btc")
    assert b[-1]["close"] == 60299.0 and len(calls) == 2       # 자산별로 따로 든다
    ns["evidence_signal_cache"]["frames"] = (frame(2700), btc, None)
    c = await load("eth")
    assert c is not a and c[-1]["close"] == 2999.0 and len(calls) == 3   # 새 프레임 -> 새 행
    assert await load("btc") is b and len(calls) == 3
    print("ok")

asyncio.run(main())
