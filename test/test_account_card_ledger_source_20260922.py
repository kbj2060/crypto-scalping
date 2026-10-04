#!/usr/bin/env python3
"""계좌 카드 원장의 **원천** 계약 (2026-09-22). 프레임워크 없이 그냥 실행한다.

왜 있나 — 사용자 신고 "계좌 카드의 원장이 안 맞는다". 원인이 넷이었다:

  1. 카드 성과 섹션이 ASSET_CONFIG 의 심볼(ETHUSDT)을 하드코딩했다. 바로 위 타일은
     snapshotAccountPosition() 으로 exec_symbol(ETHUSDC)을 보는데 원장만 갈라져 있었고,
     수동 주문이 USDC 로 넘어간 뒤 최근 19왕복 +$577 이 화면에서 통째로 사라졌다.
  2. 거래소 userTrades 7일 롤링이 창 앞 포지션의 **진입 체결**을 잘라먹어, 폴딩이 포지션
     한가운데서 시작했다. 실측 ETHUSDT 8건 중 3건이 유령·6건이 항등식 불일치였고 카드는
     8건/승률 50%/+$117 을 그렸다(디스크 원장의 같은 구간은 5건/80%/+$115.92).
  3. 절단 감지가 len(fills) >= trade_limit 뿐이라 안 걸렸다 -- limit 1000 에 체결 44건.
  4. _fold 가 측면을 positionSide 가 아니라 첫 체결 방향으로 정해, 잘린 LONG 조각이
     "SHORT" 로 찍혔다. 그 가짜 겹침 때문에 원장 겹침 가드가 멀쩡한 왕복을 영구히 거부했다.

지키는 계약:
  · 카드는 payload.trades 가 아니라 payload.ledger 를 그린다
  · 카드는 심볼이 아니라 **코인**으로 거른다(USDT/USDC 둘 다)
  · load_account_trip_rows 는 표시용 필드만 주고, 깨진 줄·없는 파일에 안 죽는다
  · 헤지 스트림의 측면은 positionSide 가 정한다
  · 잘린 스트림은 truncated_open 이 서고 그 심볼이 trades_truncated 에 들어간다

실행: python test/test_account_card_ledger_source_20260922.py
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re
import sys
import tempfile
import types

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(name: str, path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    sys.path.insert(0, str(ROOT))
    sys.modules.setdefault("duckdb", types.ModuleType("duckdb"))
    checks = 0

    # ── 1. 서버: 원장 로더 ────────────────────────────────────────────────────
    srv = load("dash_srv", ROOT / "dashboard" / "server.py")
    with tempfile.TemporaryDirectory() as td:
        srv.ACCOUNT_TRIP_LEDGER_PATH = pathlib.Path(td) / "trips.jsonl"
        assert srv.load_account_trip_rows() == [], "파일이 없으면 빈 목록"
        checks += 1

        rows = [{"symbol": "ETHUSDC", "side": "LONG", "entry_time": 1, "exit_time": 2,
                 "entry_price": 10.0, "exit_price": 11.0, "max_qty": 1.0, "net_pnl": 1.0,
                 "recorded_at": "x", "fills": 3, "commission": 0.1}]
        with srv.ACCOUNT_TRIP_LEDGER_PATH.open("w") as fh:
            fh.write(json.dumps(rows[0]) + "\n")
            fh.write("{쓰다 만 마지막 줄\n")
        got = srv.load_account_trip_rows()
        assert len(got) == 1, f"깨진 줄 하나가 원장 전체를 버리면 안 된다: {got}"
        assert set(got[0]) == {"symbol", "side", "entry_time", "exit_time", "entry_price",
                               "exit_price", "max_qty", "net_pnl"}, got[0]
        checks += 2

        # 꼬리만 싣는다 -- 원장은 무한히 자라는데 응답은 안 그래야 한다
        with srv.ACCOUNT_TRIP_LEDGER_PATH.open("w") as fh:
            for i in range(srv.ACCOUNT_TRIP_CARD_ROWS + 25):
                fh.write(json.dumps({**rows[0], "entry_time": i}) + "\n")
        tail = srv.load_account_trip_rows()
        assert len(tail) == srv.ACCOUNT_TRIP_CARD_ROWS, len(tail)
        assert tail[-1]["entry_time"] == srv.ACCOUNT_TRIP_CARD_ROWS + 24, "가장 최근이 끝에"
        checks += 2

    # ── 2. 수집기: 측면과 절단 ────────────────────────────────────────────────
    acc = load("live_acct", ROOT / "scripts" / "live_binance_account_20260910.py")
    acc._self_check()          # 폴딩 전반의 회귀 시험을 여기서도 돌린다
    checks += 1

    def fill(i, t, side, price, qty, pnl, pos):
        return {"symbol": "ETHUSDT", "id": i, "time": t, "side": side, "price": str(price),
                "qty": str(qty), "realizedPnl": str(pnl), "commission": "0", "positionSide": pos}

    # 창이 진입을 잘라먹은 헤지 LONG: 보이는 건 청산 다리(SELL)뿐
    cut = [fill(1, 1000, "SELL", 2100, 2, 200, "LONG"),
           fill(2, 2000, "BUY", 2000, 2, 0, "LONG")]
    trips = acc.round_trips(cut)
    assert all(t["side"] == "LONG" for t in trips), f"측면은 positionSide 가 정한다: {trips}"
    assert all(t.get("truncated_open") for t in trips), f"잘린 스트림 전체에 표식: {trips}"
    checks += 2

    # 멀쩡한 스트림에는 안 선다(대조군)
    ok = [fill(1, 1000, "BUY", 2000, 2, 0, "LONG"), fill(2, 2000, "SELL", 2100, 2, 200, "LONG")]
    clean = acc.round_trips(ok)
    assert not any(t.get("truncated_open") for t in clean), clean
    assert abs(clean[0]["pnl_check_bp"]) < 1e-6, clean
    checks += 2

    # 잘린 LONG 과 같은 시각의 정상 SHORT 가 «겹침» 으로 오인되면 안 된다 -- 원장이 그 SHORT 를
    # 영구히 거부하던 실제 버그다. 측면이 제대로 붙으면 겹침 판정이 서로를 안 본다.
    both = cut + [fill(3, 1200, "SELL", 2100, 1, 0, "SHORT"),
                  fill(4, 1500, "BUY", 2050, 1, 50, "SHORT")]
    sides = {t["side"] for t in acc.round_trips(both)}
    assert sides == {"LONG", "SHORT"}, sides
    short = [t for t in acc.round_trips(both) if t["side"] == "SHORT"]
    assert len(short) == 1 and not short[0].get("truncated_open"), short
    assert not srv.overlaps_recorded(
        short[0], {srv.trip_key(t): srv.trip_span(t) for t in acc.round_trips(both)
                   if t["side"] == "LONG"}), "측면이 다르면 겹침이 아니다"
    checks += 3

    # ── 3. 카드: 원천과 필터 ──────────────────────────────────────────────────
    js = (ROOT / "dashboard" / "live" / "app.js").read_text(encoding="utf-8")
    perf = js[js.index("// 오른쪽 성과"):js.index("const wins = net.filter")]
    assert "latestBinanceAccount.ledger" in perf, "카드는 서버 원장을 그린다"
    assert "latestBinanceAccount.trades" not in perf, \
        "거래소 payload 의 trades 는 7일 창이라 유령 왕복이 섞인다"
    checks += 2
    assert re.search(r"replace\(/USD\[TC\]\$/", perf), \
        "심볼이 아니라 코인으로 거른다 -- USDT/USDC 를 한 코인으로 본다"
    assert "ASSET_CONFIG[activeSnapshotAsset]?.symbol ===" not in perf
    assert "t.symbol === symbol" not in perf, "심볼 동등 비교로 돌아가면 한쪽 심볼이 사라진다"
    checks += 3

    print(f"통과 {checks}/{checks} — 계좌 카드 원장 원천 계약 유지")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
