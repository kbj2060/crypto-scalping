"""시장 맥락 -- 수집기 락 때 마지막 성공값 · 느린/실패 계산에도 JSON 본문 (2026-10-01).

    python3 -m pytest -q test/test_market_ctx_lock_and_empty_20261001.py

클로저 안 함수라 소스를 떼어 내 실제로 돌린다(test_swr_cached_key_isolation_20260912.py 와 같은 방식).
락은 **다른 프로세스**가 쓰기 연결을 쥐고 있게 해서 만든다(서버에서 난 것과 같은 모양).
"""
import asyncio, json, re, subprocess, sys, time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import duckdb
from aiohttp import web

SRC = (Path(__file__).resolve().parents[1] / "dashboard" / "server.py").read_text(encoding="utf-8")


def _chunk(start: str, end: str, dedent: bool) -> str:
    s = SRC[SRC.index(start):SRC.index(end)]
    return re.sub(r"^    ", "", s, flags=re.M) if dedent else s


def test_ls_keeps_last_good_rows_while_writer_holds_lock(tmp_path):
    db = tmp_path / "oi_lsratio.duckdb"
    con = duckdb.connect(str(db))
    con.execute("CREATE TABLE oi_lsratio_5m (ts TIMESTAMPTZ, global_ls_ratio DOUBLE, top_pos_ls_ratio DOUBLE, taker_ls_ratio DOUBLE)")
    con.execute("INSERT INTO oi_lsratio_5m VALUES (now() - INTERVAL 5 MINUTE, 2.7, 1.6, 0.7)")
    # 2026-10-04 코인별 표(oi_lsratio_collector: ETH 아닌 코인 = oi_lsratio_5m_<coin>)
    con.execute("CREATE TABLE oi_lsratio_5m_sol AS SELECT ts, 1.1::DOUBLE AS global_ls_ratio, 1.2::DOUBLE AS top_pos_ls_ratio, 0.9::DOUBLE AS taker_ls_ratio FROM oi_lsratio_5m")
    con.close()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from scripts.data_store import read_rows   # _read_only_rows 가 넘겨준다(2026-10-01 저장 재설계 3단계)
    miss = tmp_path / "missing.duckdb"
    ns: dict[str, Any] = {"duckdb": duckdb, "time": time, "Path": Path, "Any": Any,
                          "read_rows": read_rows, "OI_LSRATIO_DB_PATH": db, "OKX_CTX_DB_PATH": miss, "HL_CTX_DB_PATH": miss,
                          "HL_LIQ_BY_ASSET": {"eth": miss, "sol": miss},
                          "cctx": {a: SimpleNamespace(mc_last_good={}) for a in ("eth", "sol")},
                          "flows": {a: SimpleNamespace(spec=SimpleNamespace(okx_inst=f"{a.upper()}-USDT-SWAP")) for a in ("eth", "sol")}}
    exec(_chunk("def _read_only_rows(", "def hl_whale_liq_events(", False), ns)
    exec(_chunk("    def _mc_collectors(", "    def _mc_history(", True), ns)
    assert ns["_mc_collectors"]("sol")["ls"][0][1:] == (1.1, 1.2, 0.9)   # SOL 은 자기 표 -- ETH 값이 섞이지 않는다
    first = ns["_mc_collectors"]()
    assert first["errors"] == {} and first["ls"][0][1:] == (2.7, 1.6, 0.7), first
    # 다른 프로세스가 쓰기 연결을 쥔다 = 수집기 upsert 중
    holder = subprocess.Popen([sys.executable, "-c", f"import duckdb,time; c=duckdb.connect({str(db)!r}); print('held', flush=True); time.sleep(30)"],
                              stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        locked = ns["_mc_collectors"]()
    finally:
        holder.kill(); holder.wait()
    assert "lock" in locked["errors"]["ls"].lower(), locked["errors"]      # 락은 실제로 났다
    assert locked["ls"] == first["ls"], locked                              # 그래도 ls 는 마지막 성공 행


def test_market_context_always_returns_json_body():
    ns: dict[str, Any] = {"asyncio": asyncio, "time": time, "web": web, "Any": Any, "NOCACHE": {},
                          "cctx": {"eth": SimpleNamespace(mc_last_payload={})}}
    exec(_chunk("    MC_RESPONSE_S, MC_STALE_MAX_S", "    def situation_payload(", True), ns)
    ns["MC_RESPONSE_S"] = 0.2
    mode = {"m": "ok"}

    async def fake_payload(asset):
        if mode["m"] == "slow":
            await asyncio.sleep(1.0)
        if mode["m"] == "boom":
            raise RuntimeError("x")
        return {"available": True, "v": 1}
    ns["market_context_payload"] = fake_payload
    call = lambda: json.loads(asyncio.run(ns["api_market_context"](SimpleNamespace(query={}))).body)   # noqa: E731

    assert call() == {"available": True, "v": 1}
    for m in ("slow", "boom"):          # 식은 캐시(느림)·예외 -> 마지막 성공값 + stale
        mode["m"] = m
        t0 = time.time(); got = call()
        assert time.time() - t0 < 0.8 and got["stale"] is True and got["v"] == 1 and got["error"], (m, got)
    ns["cctx"]["eth"].mc_last_payload["at"] -= 1000  # 너무 낡은 값은 현재처럼 보이지 않게 -> «지연»
    got = call()
    assert got["available"] is False and "RuntimeError" in got["error"], got
    try:   # 꺼진 코인은 404 -- ETH 값을 다른 코인 이름으로 주지 않는다
        asyncio.run(ns["api_market_context"](SimpleNamespace(query={"asset": "hype"})))
        raise AssertionError("hype 가 404 가 아니다")
    except web.HTTPNotFound:
        pass
