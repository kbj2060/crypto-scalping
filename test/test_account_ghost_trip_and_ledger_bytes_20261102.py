#!/usr/bin/env python3
"""계좌 카드 «롱 순환 2개» · 원장 깨진 바이트 (2026-11-02, 서브 에이전트 검증 결함 1·2). 프레임워크 없이: python test/test_account_ghost_trip_and_ledger_bytes_20261102.py

  · 재동기화(84e43b5b)가 창 앞 포지션의 «열린 파편»을 남기면 진짜 열린 왕복과 둘이 된다 -> 열린 왕복은 하나여야 한다
  · 원장에 비 UTF-8 바이트가 한 줄 있어도 키·행 읽기가 그 줄만 건너뛴다(전엔 seen 이 비어 중복 기록 · 계좌 응답 실패)
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def test_resync_leaves_single_open_trip() -> None:
    sys.path.insert(0, str(ROOT))
    acct = _load("lba_t", "scripts/live_binance_account_20260910.py")
    def fill(i, t, side, qty, pnl):
        return {"symbol": "ETHUSDC", "id": i, "time": t, "side": side, "price": "100", "qty": str(qty), "realizedPnl": str(pnl),
                "commission": "0", "positionSide": "LONG"}
    # 창 앞에서 열린 롱(수량 2)을 창 안에서 SELL 2 로 정리 -> 새 롱 BUY 1 이 열려 있다(지금 열린 수량 1)
    trips = acct.round_trips([fill(1, 1000, "SELL", 2, 5), fill(2, 2000, "BUY", 1, 0)], {("ETHUSDC", "LONG"): 1.0})
    opened = [t for t in trips if not t["closed"]]
    assert len(opened) == 1 and opened[0]["entry_time"] == 2000 and not opened[0].get("truncated_open"), trips


def test_ledger_survives_bad_bytes() -> None:
    sys.path.insert(0, str(ROOT))
    sys.modules.setdefault("duckdb", types.ModuleType("duckdb"))
    m = _load("dash_orders_t", "dashboard/orders.py")
    good = lambda e: json.dumps({"symbol": "ETHUSDT", "side": "LONG", "entry_time": e, "exit_time": e + 5, "max_qty": 1.0, "net_pnl": 1.0}) + "\n"
    with tempfile.TemporaryDirectory() as td:
        m.ACCOUNT_TRIP_LEDGER_PATH = pathlib.Path(td) / "trips.jsonl"
        m.ACCOUNT_TRIP_LEDGER_PATH.write_bytes(good(1000).encode() + b"\xff\xfe broken\n" + good(3000).encode())
        assert len(m.load_account_trip_keys()) == 2, "깨진 줄만 건너뛰고 나머지 둘은 읽는다"
        assert isinstance(m.load_account_trip_rows(), list)   # 예외 없이


if __name__ == "__main__":
    test_resync_leaves_single_open_trip(); test_ledger_survives_bad_bytes(); print("ok")
