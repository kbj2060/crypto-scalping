"""2026-11-07 OKX 청산 재시작 구멍: 메모리 시작 전 봉은 hot 에서 채운다.   python -m pytest -q test/test_okx_liq_hot_merge_20261107.py

실측: 서버 hot 에 11-06 12:00~23:20 UTC 약 $3.5M 이 있는데 API 응답 그 봉들엔 okx_detail 이 없어 24시간 합계가 그만큼 모자랐다."""
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import dashboard.server as srv  # noqa: E402

BAR = 300
T0 = 1_790_000_100          # 봉 경계


def bar(t):
    return {"ts": datetime.fromtimestamp(t, timezone.utc).isoformat(), "long_usd": 10.0, "short_usd": 0.0, "events": 1}


def test_bar_before_restart_gets_okx_from_hot():
    """재시작 전 봉(메모리에 없음)에도 hot 금액이 합쳐져야 24시간 합계가 맞는다."""
    live_from = T0 + BAR              # 재시작이 이 봉에 걸쳤다 -- 메모리는 다음 봉부터 온전
    bars = [bar(T0), bar(T0 + BAR), bar(T0 + 2 * BAR)]
    live = [{"ts_ms": (T0 + 2 * BAR + 5) * 1000, "side": "short", "usd": 7.0},
            {"ts_ms": (T0 + BAR + 200) * 1000, "side": "long", "usd": 999.0}]   # 걸친 봉의 뒷부분뿐 -- 쓰면 안 된다
    hot = {T0: [100.0, 0.0, 2], T0 + BAR: [50.0, 5.0, 3], T0 + 2 * BAR: [999.0, 999.0, 9]}
    orig = [dict(b) for b in bars]
    m = srv.merge_okx_liq(bars, live, live_from, hot, (T0 - BAR) * 1000, BAR)
    assert m[0]["okx"] and m[0]["okx_detail"] == {"long_usd": 100, "short_usd": 0, "n": 2} and m[0]["long_usd"] == 110.0
    assert m[1]["okx_detail"] == {"long_usd": 50, "short_usd": 5, "n": 3}, "걸친 봉은 hot 이 온전하다"
    assert m[2]["okx_detail"] == {"long_usd": 0, "short_usd": 7, "n": 1}, "메모리 구간은 메모리만 -- 이중 합산 금지"
    assert bars == orig, "입력(swr 캐시일 수 있음)을 고치면 안 된다"


def test_no_live_flow_yet_uses_hot_only():
    """OKX 체결이 아직 안 왔어도(first_bar 0) hot 은 있다."""
    m = srv.merge_okx_liq([bar(T0)], [{"ts_ms": (T0 + 5) * 1000, "side": "long", "usd": 1.0}], 0, {T0: [3.0, 0.0, 1]},
                          (T0 - BAR) * 1000, BAR)
    assert m[0]["okx_detail"] == {"long_usd": 3, "short_usd": 0, "n": 1}


def test_bars_before_hot_start_stay_unknown():
    """기록 시작 전·시작이 걸친 봉에 0 을 넣으면 «청산 없음»이라는 거짓 -- 그대로 둔다. 시작 뒤 빈 봉은 0."""
    bars = [bar(T0 - BAR), bar(T0), bar(T0 + BAR)]
    m = srv.merge_okx_liq(bars, [], T0 + BAR, {}, (T0 + 10) * 1000, BAR)
    assert "okx" not in m[0] and "okx" not in m[1]
    assert m[2]["okx_detail"]["n"] == 0, "기록 시작 뒤 빈 봉 = 청산 없음"
    m = srv.merge_okx_liq(bars, [], T0 + 5 * BAR, {}, (T0 - 2 * BAR) * 1000, BAR)
    assert m[0]["okx_detail"]["n"] == 0
    assert srv.merge_okx_liq(bars, [], T0 + 5 * BAR, {}, None, BAR)[0] is bars[0], "hot 없으면 그대로"


def test_hot_reader_groups_bars_by_side_and_inst(tmp_path):
    """수집기 표 그대로 -- 봉으로 모으고, 다른 종목·sz 0 은 빼고, 보관 창 밖은 안 읽는다. 없으면 ({}, None)."""
    d = tmp_path / "okx_ctx.sqlite"
    c = sqlite3.connect(d)
    c.execute("""CREATE TABLE okx_liquidations(inst_id VARCHAR, inst_family VARCHAR, ts_ms BIGINT, side VARCHAR, pos_side VARCHAR,
                 bk_px DOUBLE, sz_contracts DOUBLE, sz_base DOUBLE, bk_loss DOUBLE)""")
    now = int(time.time()) // BAR * BAR
    old = now - (srv.FOOTPRINT_KEEP_BARS + 5) * BAR
    row = lambda inst, t, pos, px, base: (inst, "", t * 1000, "", pos, px, 0, base, 0)  # noqa: E731
    c.executemany("INSERT INTO okx_liquidations VALUES (?,?,?,?,?,?,?,?,?)", [
        row("ETH-USDT-SWAP", old, "long", 2000.0, 1.0),
        row("ETH-USDT-SWAP", now - 600, "long", 2000.0, 1.0),
        row("ETH-USDT-SWAP", now - 500, "long", 2000.0, 0.5),
        row("ETH-USDT-SWAP", now - 400, "short", 2000.0, 2.0),
        row("ETH-USDT-SWAP", now - 200, "short", 2000.0, 0.0),
        row("BTC-USDT-SWAP", now - 600, "long", 90000.0, 1.0)])
    c.commit()
    bars, start = srv.okx_liq_hot_bars("ETH-USDT-SWAP", d, BAR)
    assert start == old * 1000 and bars == {now - 600: [3000.0, 4000.0, 3]}
    assert srv.okx_liq_hot_bars("ETH-USDT-SWAP", tmp_path / "none.sqlite", BAR) == ({}, None)
