"""5분봉 청산 원에 HL 고래 청산 합산(2026-09-24).   python -m pytest -q test/test_hl_liq_5m_merge_20260924.py"""
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard.server import hl_whale_liq_events, merge_hl_liq  # noqa: E402


def iso(t):
    return datetime.fromtimestamp(t, timezone.utc).isoformat()


def test_merge_adds_and_keeps_breakdown():
    bars = [{"ts": iso(1_800_000_000), "long_usd": 100.0, "short_usd": 0.0, "events": 1},
            {"ts": iso(1_800_000_300), "long_usd": 0.0, "short_usd": 0.0, "events": 0}]
    ev = [(1_800_000_010_000, 5000.0, True, "0xa"), (1_800_000_200_000, 700.0, False, "0xb"),
          (1_800_000_310_000, 50.0, True, "0xa"), (1_700_000_000_000, 9.0, True, "0xz")]   # 창 밖은 버린다
    out = merge_hl_liq(bars, ev, 300)
    assert (out[0]["long_usd"], out[0]["short_usd"], out[0]["events"]) == (5100.0, 700.0, 3)
    assert out[0]["hl"] == {"long_usd": 5000, "short_usd": 700, "n": 2, "users": ["0xa", "0xb"]}
    assert out[1]["long_usd"] == 50.0 and out[1]["hl"]["n"] == 1
    assert "hl" not in bars[0] and bars[0]["long_usd"] == 100.0, "입력(캐시 객체)을 고치면 안 된다"


def test_events_reader(tmp_path):
    assert hl_whale_liq_events(0, tmp_path / "none.duckdb") == [], "수집기 DB 가 없으면 빈 목록"
    db = tmp_path / "p.duckdb"
    c = duckdb.connect(str(db))
    c.execute("""CREATE TABLE hl_liquidations(detected_ms BIGINT, tid BIGINT, user VARCHAR, coin VARCHAR,
                 fill_ms BIGINT, px DOUBLE, sz DOUBLE, side VARCHAR, dir VARCHAR, start_position DOUBLE,
                 closed_pnl DOUBLE, mark_px DOUBLE, method VARCHAR)""")
    c.executemany("INSERT INTO hl_liquidations VALUES (0,?,?,?,?,?,?,?,?,0,0,0,'market')",
                  [(1, "0xa", "ETH", 2000, 2500.0, 2.0, "A", "Close Long"),
                   (2, "0xb", "ETH", 3000, 2600.0, 1.0, "B", "Close Short"),
                   (3, "0xc", "BTC", 3000, 1.0, 1.0, "A", "Close Long"),
                   (4, "0xd", "ETH", 500, 2500.0, 1.0, "A", "Close Long")])      # since 앞
    c.close()
    got = sorted(hl_whale_liq_events(1000, db))
    assert got == [(2000, 5000.0, True, "0xa"), (3000, 2600.0, False, "0xb")], got
