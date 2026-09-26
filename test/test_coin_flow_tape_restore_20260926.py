"""코인별 흐름 엔진 (2026-09-26 SOL·XRP 확장).   python -m pytest -q test/test_coin_flow_tape_restore_20260926.py

① ETH 외 코인은 REST 백필(가중치 20, IP 밴 사고) 대신 **체결 테이프**에서 되살린다 -- 칸 폭 변환(테이프 0.0001 ->
   풋프린트 0.0003)·6칸 순서([매수, 매도, 고래매수, 고래매도, 리테일매수, 리테일매도])·WS 가 셀 경계 초 제외.
② 코인 설정은 전부 **USDT 시장**이다(사용자 규칙: 데이터 USDT · 주문만 USDC).
"""
import asyncio
import dataclasses
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dashboard.server as srv  # noqa: E402

BAR = 300
T0 = 1_790_000_100            # 봉 경계


def _tape(path: Path, rows):
    con = duckdb.connect(str(path))
    con.execute("""CREATE TABLE trade_tape_1s (symbol VARCHAR, ts_sec BIGINT, price_bin INTEGER,
                   buy_qty DOUBLE, sell_qty DOUBLE, whale_buy_qty DOUBLE, whale_sell_qty DOUBLE,
                   retail_buy_qty DOUBLE, retail_sell_qty DOUBLE)""")
    con.execute("CREATE TABLE verify_1m (symbol VARCHAR, ts_min BIGINT, rel_err DOUBLE)")
    con.execute("CREATE TABLE gaps (symbol VARCHAR, from_ms BIGINT, to_ms BIGINT)")
    con.executemany("INSERT INTO trade_tape_1s VALUES (?,?,?,?,?,?,?,?,?)", rows)
    con.close()


def test_non_eth_restores_from_tape_not_rest(tmp_path, monkeypatch):
    db = tmp_path / "trade_tape_xrp.duckdb"
    # XRP 테이프 칸 0.0001 -> 15363·15364·15365 는 풋프린트 0.0003 칸 5121 로 모인다(15363/3=5121, 15365/3≈5121.7→5122)
    _tape(db, [("xrpusdt", T0 - BAR + 10, 15363, 1.0, 2.0, 0.0, 0.5, 1.0, 1.5),     # 앞 봉
               ("xrpusdt", T0 + 5, 15364, 3.0, 0.0, 3.0, 0.0, 0.0, 0.0),              # 진행 봉
               ("xrpusdt", T0 + 60, 15365, 4.0, 1.0, 0.0, 0.0, 4.0, 1.0),             # = WS 첫 초 -> 빠져야 한다
               ("xrpusdt", T0 + 70, 15365, 9.0, 9.0, 0.0, 0.0, 0.0, 0.0)])            # WS 이후 -> 빠져야 한다
    monkeypatch.setattr(srv, "tape_default_db", lambda sym: db)
    monkeypatch.setattr(srv.time, "time", lambda: T0 + 70.0)                          # 창 계산의 «지금»

    async def rest_must_not_be_called(*a, **k):
        raise AssertionError("ETH 외 코인이 REST 를 불렀다")
    spec = dataclasses.replace(srv.FLOW_SPECS["xrp"], snapshot_path=tmp_path / "fp.json")
    f = srv.make_coin_flow(spec, rest_must_not_be_called, {"session": None})
    asyncio.run(f.footprint_tape_restore(0, (T0 + 60) * 1000 + 400))                 # WS 첫 체결 T0+60.4초
    bars = f.footprint_state["bars"]
    assert f.footprint_state["ready"] is True
    assert bars[T0 - BAR] == {5121: [1.0, 2.0, 0.0, 0.5, 1.0, 1.5]}
    assert bars[T0] == {5121: [3.0, 0.0, 3.0, 0.0, 0.0, 0.0]}, "WS 가 세는 경계 초(T0+60)·이후를 테이프에서 또 더했다"


def test_coin_specs_are_usdt_data_markets():
    for a, sp in srv.FLOW_SPECS.items():
        assert sp.symbol == f"{a.upper()}USDT" and sp.okx_inst.endswith("-USDT-SWAP"), sp
        assert sp.spot_url.endswith(f"/{a}usdt@aggTrade"), sp
        assert "USDC" not in f"{sp.symbol}{sp.okx_inst}{sp.spot_url}".upper(), "데이터에 USDC 시장이 섞였다"
        tape = srv.TAPE_BUCKETS[sp.symbol.lower()]
        assert abs(sp.bucket / tape - round(sp.bucket / tape)) < 1e-9, "풋프린트 칸이 테이프 칸의 정수배가 아니다"
    assert srv.FLOW_SPECS["eth"].bucket == srv.FOOTPRINT_BUCKET and srv.FLOW_SPECS["eth"].oi_poll_s == srv.OI_1S_POLL_SECONDS


def test_hl_whale_liq_filters_coin_and_keeps_xrp_decimals(tmp_path):
    """HL 멀티코인 DB 에서 **그 코인만** 묶는다(ETH 가 XRP 레벨에 섞이면 안 된다) · XRP 는 4자리."""
    db = tmp_path / "hl.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE hl_cycles (ts_ms BIGINT, n_ok INTEGER)")
    con.execute("CREATE TABLE hl_positions (ts_ms BIGINT, coin VARCHAR, szi DOUBLE, liq_px DOUBLE)")
    con.execute("INSERT INTO hl_cycles VALUES (1000, 5)")
    con.executemany("INSERT INTO hl_positions VALUES (?,?,?,?)", [
        (1000, "XRP", 50000.0, 1.4301), (1000, "XRP", 20000.0, 1.4302), (1000, "XRP", -30000.0, 1.6519),
        (1000, "ETH", 10.0, 2500.0)])
    con.close()
    r = srv.hl_whale_liq(db, 0.003, "XRP", 4)
    assert r["ok"] and r["n_positions"] == 3, r
    assert r["clusters"] == [[1.431, 70000.0, 0.0, 2], [1.653, 0.0, 30000.0, 1]], r["clusters"]
    assert srv.HL_LIQ_BY_ASSET["xrp"][0] == srv.HL_LIQ_BY_ASSET["sol"][0] != srv.HL_LIQ_BY_ASSET["eth"][0]


def test_tape_seconds_matches_live_1s_cell_order(tmp_path):
    """재시작 1초 수급 복원: 칸 순서가 라이브 supply_1s_cell 과 같아야 한다 -- [리테일매수, 리테일매도, 고래매수,
    고래매도, 총매수, 총매도, 가격]. 테이블 열 순서(총·고래·리테일)와 다르므로 뒤섞이면 고래가 리테일로 그려진다."""
    db = tmp_path / "t.duckdb"
    _tape(db, [("xrpusdt", T0, 15360, 1.0, 3.0, 0.0, 2.0, 1.0, 0.5),
               ("xrpusdt", T0, 15370, 3.0, 0.0, 3.0, 0.0, 0.0, 0.0),
               ("xrpusdt", T0 + 1, 15360, 9.0, 9.0, 0.0, 0.0, 0.0, 0.0)])     # hi 배타 -> 빠진다
    got, top = srv.tape_seconds(db, "xrpusdt", T0 - 10, T0 + 1, 0.0001)
    assert top == T0 + 1 and set(got) == {T0}
    rb, rs, wb, ws, tb, ts, px = got[T0]
    assert (rb, rs, wb, ws, tb, ts) == (1.0, 0.5, 3.0, 2.0, 4.0, 3.0), got
    assert abs(px - (1.5360 * 4 + 1.5370 * 3) / 7) < 1e-9          # 칸 가중평균
