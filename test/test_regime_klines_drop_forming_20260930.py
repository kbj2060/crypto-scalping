"""레짐 워커 공용 `_fetch_klines` 가 형성 중 봉을 버리는지 (2026-09-30). 네트워크 없이 돈다.

ETH(wide24)·BTC·XRP 레짐 워커가 모두 이 함수를 import 한다. 가짜 응답 = 마감봉 3개 + 형성 봉 1개.
    python3 -m pytest -q test/test_regime_klines_drop_forming_20260930.py
"""
import importlib.util
import sys
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STEP = 5 * 60 * 1000


def test_forming_bar_dropped():
    sp = importlib.util.spec_from_file_location("rw_forming_under_test",
                                                ROOT / "scripts" / "live_regime_wide24_signal_20260826.py")
    mod = importlib.util.module_from_spec(sp)
    sys.modules[sp.name] = mod
    sp.loader.exec_module(mod)
    now = int(time.time() * 1000)
    cur = now - now % STEP                          # 형성 중 봉 시작
    opens = [cur - 3 * STEP, cur - 2 * STEP, cur - STEP, cur]
    rows = [[o, "1", "1", "1", "1", "1", o + STEP - 1, "1", 1, "1", "1", "0"] for o in opens]
    mod.requests = types.SimpleNamespace(get=lambda *a, **k: types.SimpleNamespace(
        raise_for_status=lambda: None, json=lambda: rows))
    df = mod._fetch_klines("XRPUSDT", opens[0], now)
    assert len(df) == 3
    assert int(df["timestamp"].iloc[-1].timestamp() * 1000) == cur - STEP   # 마지막 = 방금 마감된 봉
