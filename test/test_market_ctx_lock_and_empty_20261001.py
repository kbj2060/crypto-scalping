"""시장 맥락 -- 수집기 락 때 마지막 성공값 (2026-10-01).

    python3 -m pytest -q test/test_market_ctx_lock_and_empty_20261001.py

클로저 안 함수라 소스를 떼어 내 실제로 돌린다(test_swr_cached_key_isolation_20260912.py 와 같은 방식).
락은 **다른 프로세스**가 쓰기 연결을 쥐고 있게 해서 만든다(서버에서 난 것과 같은 모양).
"""
import asyncio, json, re, subprocess, sys, time
from pathlib import Path
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
    con.close()
    ns: dict[str, Any] = {"duckdb": duckdb, "time": time, "Path": Path, "Any": Any, "OKX_INST": "ETH-USDT-SWAP",
                          "OI_LSRATIO_DB_PATH": db, **{k: tmp_path / "missing.duckdb" for k in
                                                        ("OKX_CTX_DB_PATH", "HL_CTX_DB_PATH", "HL_POS_DB_PATH")}}
    exec(_chunk("def _read_only_rows(", "def hl_whale_liq_events(", False), ns)
    exec(_chunk("    mc_last_good: dict", "    def _mc_history(", True), ns)
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

