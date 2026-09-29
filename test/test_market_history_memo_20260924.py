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
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard import market_ctx as mctx  # 2026-09-29 캔들 행에 세션 VWAP(mctx.session_vwap)이 붙는다
ns = {"Any": object, "web": None, "evidence_signal_cache": {}, "trend_veto_rows": lambda df: calls.append(df) or {}, "mctx": mctx}
async def warm(): pass
ns["load_chart_klines_frames"] = warm
exec(compile(chunk, "memo", "exec"), ns)
load = ns["load_market_history_from_evidence_cache"]

def frame(p0):
    ts = pd.date_range("2026-09-24", periods=300, freq="5min", tz="UTC")
    c = p0 + np.arange(300.0)
    return pd.DataFrame({"timestamp": ts, "open": c, "high": c + 1, "low": c - 1, "close": c, "volume": 1.0})

async def main():
    eth, btc = frame(2600), frame(60000).drop(columns=["volume"])   # 실제 BTC 프레임은 OHLC 뿐 -- 거래량 없어도 행이 나와야 한다(09-29 사고)
    ns["evidence_signal_cache"]["frames"] = (eth, btc, None)
    a = await load("eth")
    assert len(a) == 200 and a[-1]["close"] == 2899.0
    assert await load("eth") is a and len(calls) == 1          # 같은 프레임 -> 다시 안 만든다
    b = await load("btc")
    assert b[-1]["close"] == 60299.0 and len(calls) == 2       # 자산별로 따로 든다
    assert "vwap" in a[-1] and "vwap" not in b[-1]             # 거래량 있으면 VWAP, 없으면 VWAP 없이(예외 아님)
    ns["evidence_signal_cache"]["frames"] = (frame(2700), btc, None)
    c = await load("eth")
    assert c is not a and c[-1]["close"] == 2999.0 and len(calls) == 3   # 새 프레임 -> 새 행
    assert await load("btc") is b and len(calls) == 3
    print("ok")

asyncio.run(main())
