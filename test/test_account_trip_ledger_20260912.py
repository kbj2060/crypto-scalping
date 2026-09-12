#!/usr/bin/env python3
"""실계좌 왕복 원장 자체점검 (2026-09-12). 프레임워크 없이 그냥 실행한다.

왜 있나: 바이낸스 userTrades 는 7일 롤링이라 그 앞의 왕복이 API 에서 사라진다 -- 2026-09-11 에
보이던 09-03~04 건이 09-12 조회에는 없었다. 크기 배분이 손익을 얼마나 갈랐는지 재는 표본이
정확히 이 왕복이라, 새 거래가 들어올수록 오래된 게 밀려나 표본이 15건 근처에 정체한다.

여기서 지키는 계약:
  · 미청산 왕복은 안 적는다 (나중에 exit/net_pnl 이 채워지므로 반쪽 판이 영구히 남는다)
  · 같은 왕복을 두 번 적지 않는다
  · **헤지 모드**: 같은 심볼·같은 진입시각이라도 LONG/SHORT 는 서로 다른 왕복이다
  · 깨진 줄 하나가 원장 전체를 버리지 않는다 (그 왕복은 다음 주기에 다시 적힌다)
  · ok=false payload(키 만료·시계드리프트)에는 아무것도 안 적는다

실행: python test/test_account_trip_ledger_20260912.py
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
import tempfile
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    sys.path.insert(0, str(ROOT))
    sys.modules.setdefault("duckdb", types.ModuleType("duckdb"))  # 조회 경로를 안 타므로 스텁으로 족하다
    spec = importlib.util.spec_from_file_location("dash_srv", ROOT / "dashboard" / "server.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    def trip(entry: int, side: str = "LONG", closed: bool = True, **extra) -> dict:
        return {"symbol": "ETHUSDT", "side": side, "entry_time": entry, "max_qty": 1.0,
                "net_pnl": -1.0, "closed": closed, **extra}

    def payload(*trips, ok: bool = True) -> dict:
        return {"ok": ok, "trades": list(trips)}

    def lines() -> list[dict]:
        return [json.loads(x) for x in m.ACCOUNT_TRIP_LEDGER_PATH.read_text().splitlines() if x.strip()]

    with tempfile.TemporaryDirectory() as td:
        m.ACCOUNT_TRIP_LEDGER_PATH = pathlib.Path(td) / "trips.jsonl"
        record, load_keys = m.record_account_trips, m.load_account_trip_keys

        assert load_keys() == set(), "파일이 없으면 빈 집합"

        seen: set[str] = set()
        assert record(payload(trip(1000), trip(2000, closed=False)), seen) == 1, "미청산은 적지 않는다"
        assert record(payload(trip(1000), trip(2000, closed=False)), seen) == 0, "같은 왕복 재기록 금지"
        assert [t["entry_time"] for t in lines()] == [1000]

        # 그 미청산 건이 청산되면 그때 적힌다
        assert record(payload(trip(1000), trip(2000, net_pnl=5.0)), seen) == 1, "청산 후엔 적힌다"
        assert [t["entry_time"] for t in lines()] == [1000, 2000]
        assert lines()[1]["net_pnl"] == 5.0
        assert all("recorded_at" in t for t in lines()), "언제 봤는지 남긴다"

        # 헤지 모드: 같은 심볼·같은 진입시각이라도 방향이 다르면 별개 왕복
        assert record(payload(trip(1000, side="SHORT")), seen) == 1, "LONG/SHORT 는 별개"
        assert len(lines()) == 3

        # ok=false 면 아무것도 안 적는다 (키 만료·시계드리프트 때 원장을 오염시키지 않는다)
        assert record(payload(trip(9000), ok=False), seen) == 0, "ok=false 무시"
        assert record({"ok": True}, seen) == 0, "trades 없음 무시"
        assert len(lines()) == 3

        # 재기동 복원: 디스크에서 읽은 신원만으로 중복이 막힌다
        restored = load_keys()
        assert len(restored) == 3, f"복원 실패: {restored}"
        assert record(payload(trip(1000), trip(2000), trip(1000, side="SHORT")), restored) == 0

        # 깨진 줄이 섞여도 나머지는 읽고, 그 왕복은 다시 적을 수 있다
        with m.ACCOUNT_TRIP_LEDGER_PATH.open("a") as fh:
            fh.write("{절반만 쓰다 죽은 줄\n")
        after = load_keys()
        assert len(after) == 3, f"깨진 줄 때문에 원장을 버렸다: {after}"
        assert record(payload(trip(4000)), after) == 1, "깨진 줄 뒤에도 계속 적을 수 있다"

    # ── 단건 상한 (2026-09-12) ────────────────────────────────────────────────
    # 왕복이 SIZING_CAP_MIN_TRIPS 미만이면 상한을 **만들지 않는다** -- 3건짜리 중앙값은
    # 표본 하나에 휘둘리고, 그걸로 크기를 자르면 결과선택 편향이다.
    with tempfile.TemporaryDirectory() as td:
        m.ACCOUNT_TRIP_LEDGER_PATH = pathlib.Path(td) / "trips.jsonl"

        assert m.sizing_cap() == {"available": False, "reason": "ledger_missing"}

        def write(qtys, price=1000.0):
            m.ACCOUNT_TRIP_LEDGER_PATH.write_text("".join(
                json.dumps({"symbol": "ETHUSDT", "side": "LONG", "entry_time": 1000 + i,
                            "max_qty": q, "entry_price": price}) + "\n"
                for i, q in enumerate(qtys)))

        write([1.0] * (m.SIZING_CAP_MIN_TRIPS - 1))
        cap = m.sizing_cap()
        assert cap["available"] is False and cap["reason"] == "not_enough_trips", cap
        assert cap["trips"] == m.SIZING_CAP_MIN_TRIPS - 1, cap

        # 명목 1,000 짜리 10건 + 명목 50,000 짜리 1건 → 중앙은 1,000 이고 이상치에 안 끌린다
        write([1.0] * 10 + [50.0])
        cap = m.sizing_cap()
        assert cap["available"] and cap["trips"] == 11, cap
        assert cap["median_notional_usdt"] == 1000.0, cap
        assert cap["cap_notional_usdt"] == m.SIZING_CAP_MULT * 1000.0, cap

        # 깨진 줄이 섞여도 나머지로 상한을 낸다
        with m.ACCOUNT_TRIP_LEDGER_PATH.open("a") as fh:
            fh.write("{절반만 쓰다 죽은 줄\n")
        assert m.sizing_cap()["trips"] == 11, m.sizing_cap()

    print("통과 13/13 — 왕복 원장 + 단건 상한 계약 유지")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
