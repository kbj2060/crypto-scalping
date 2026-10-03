"""2026-10-03 Bybit 청산 합산: side 뒤집기(Buy = 롱 청산) · 수집 시작이 걸친 봉은 안 붙인다 · 봉별 금액."""
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import dashboard.server as srv  # noqa: E402


def test_bybit_liq_events_and_merge(tmp_path):
    d = tmp_path / "b.sqlite"
    c = sqlite3.connect(d)
    c.execute("CREATE TABLE bybit_liquidations(ts_ms INTEGER, symbol TEXT, side TEXT, qty REAL, price REAL, recv_ms INTEGER)")
    now = int(time.time()) // 300 * 300
    c.executemany("INSERT INTO bybit_liquidations VALUES (?,?,?,?,?,?)",
                  [((now - 900) * 1000, "ETHUSDT", "Buy", 1, 2000, 0), ((now - 290) * 1000, "ETHUSDT", "Sell", 2, 2000, 0)])
    c.commit()
    ev, start = srv.bybit_liq_events("ETHUSDT", d, ttl=0)
    assert [e["side"] for e in ev] == ["long", "short"] and start == (now - 900) * 1000
    bars = [{"ts": datetime.fromtimestamp(now - k * 300, timezone.utc).isoformat(), "long_usd": 10.0, "short_usd": 0.0, "events": 1}
            for k in (3, 2, 1)]
    m = srv.merge_bybit_liq(bars, ev, start, 300)
    assert "bybit" not in m[0], "시작이 걸친 봉은 반쪽 -- 안 붙인다"
    assert m[2]["short_usd"] == 4000.0 and m[2]["bybit"]["n"] == 1 and m[1]["bybit"]["n"] == 0
    assert srv.merge_bybit_liq(bars, [], None, 300) is bars, "수집기 없으면 그대로"
